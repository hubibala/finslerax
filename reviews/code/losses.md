# Code Review: `training/losses.py`
**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

The losses module provides 16 modular `LossComponent` subclasses covering reconstruction, KL divergence, alignment, curvature-aware, and physics-informed objectives. The overall design is clean and correctly leverages Equinox's `eqx.Module` pattern and `eqx.filter_checkpoint` for memory-efficient backprop through solvers. However, there are several issues ranging from JAX tracing violations (`hasattr` inside traced code, bare `return 0.0`), numerical stability gaps (unsafe `sqrt`, unguarded `inner_product` sign), inconsistent epsilon conventions (four different epsilon values across the file), and significant test coverage gaps (only 3 of 16 loss classes are exercised in tests). The `EulerLagrangeResidualLoss` and `FinslerianFlowMatchingLoss` are particularly well-constructed from a JAX differentiation standpoint, using `jax.jvp` for efficient Hessian-vector products.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/training/losses.py:51,73,100,143,268,302,340,464` | `hasattr()` calls inside `__call__` will be evaluated at Python trace time, not at JAX runtime. When the model is traced once and the attribute check result is baked into the XLA program, switching between metric types with/without `_get_zermelo_data` on the same jitted function will silently use the wrong branch. This is fragile and will break if the pipeline caches a jitted `loss_fn` across phases with different metric types. | Replace with a unified metric interface method (e.g. `metric.get_zermelo_data_or_default(z)`) that always returns `(H, W, inv_H)` — returning identity/zeros for non-Randers metrics. Alternatively, accept the trace-time dispatch as intentional and document it clearly, but never mix metric types in the same jitted scope. |
| 2 | **BUG** | `src/ham/training/losses.py:143,338` | `MetricAnchorLoss` and `WindThermodynamicLoss` return bare Python `0.0` in the else-branch. Under `jax.grad`, this returns a non-JAX scalar, which can cause silent type-mismatch errors or prevent gradient propagation. The pipeline sums losses: `total_loss += val` where `val` may be Python float `0.0` rather than `jnp.float32(0.0)`. | Return `jnp.float32(0.0)` instead of `0.0` in all else-branches. |
| 3 | **BUG** | `src/ham/training/losses.py:226–228` | `LongTrajectoryAlignmentLoss.__call__` has a duplicated comment block (`# 3. Penalize the difference` appears twice) and inconsistent indentation — the `return` statement is indented at 8 spaces while the preceding block is at 4 spaces. This is not a runtime error but suggests a copy-paste mistake; verify the intended indentation is correct. | Remove the duplicate comment and fix indentation to be consistent. |
| 4 | **RISK** | `src/ham/training/losses.py:59–60` | `ZermeloAlignmentLoss` normalises vectors using `jnp.maximum(norm, 1e-6)` but `norm_w` and `norm_v` are computed via `model.manifold._minkowski_norm()` which may return values of any sign for spacelike/timelike vectors on pseudo-Riemannian manifolds. If the norm is negative, `jnp.maximum(negative, 1e-6)` yields `1e-6`, silently clobbering the direction. | Guard against negative norms explicitly (e.g. `jnp.maximum(jnp.abs(norm), 1e-6)`) or document that this loss is only valid for spacelike vectors. |
| 5 | **RISK** | `src/ham/training/losses.py:82` | `GeodesicSprayLoss` computes `spray_norm = model.metric.inner_product(z_mean, dot_z, spray_vec, spray_vec)`. The inner product $g_{ij} s^i s^j$ may be negative on pseudo-Riemannian manifolds. If this is used as a penalty term, a negative loss would reward large sprays instead of penalizing them. | Use `jnp.abs(spray_norm)` or `jnp.maximum(spray_norm, 0.0)` if the intent is to penalize spray magnitude. |
| 6 | **RISK** | `src/ham/training/losses.py:109` | `VelocityDirectionAlignmentLoss` returns `(1.0 - cos_sim) * self.weight` without `jnp.mean()`. If `cos_sim` has a batch shape from `jnp.sum(..., axis=-1)`, this returns a non-scalar when the pipeline expects a scalar. Compare with `FinslerActionMatchingLoss` which uses `jnp.mean(energy)`. The pipeline vmaps over batch (line 57 of `pipeline.py`) so each call receives a single sample — but if this loss is ever called outside the pipeline, it will break. | Add `jnp.mean(...)` for safety, consistent with other losses. |
| 7 | **RISK** | `src/ham/training/losses.py:247,451` | `EulerLagrangeResidualLoss` and `FinslerianFlowMatchingLoss` return bare Python `0.0` when `len(batch) < 3 or batch[2] is None`. Same issue as #2 — not a valid JAX value under tracing. Additionally, `batch[2] is None` will fail inside a `vmap` since `None` is not a valid pytree leaf for comparison. | Return `jnp.float32(0.0)`. Move the `None`-check outside the vmapped scope (i.e. validate batch shape before the loss is called, in the pipeline). |
| 8 | **RISK** | `src/ham/training/losses.py:171–180` | `_solve_and_integrate_impl` computes `dt = 1.0 / (N - 1)` where `N = trajectory.shape[0]`. If the solver returns a single-point trajectory (`N=1`), this produces `inf`. Additionally, `jnp.diff(trajectory, axis=0)` returns an empty array when `N=1`, making `jax.vmap(model.metric.energy)(positions, velocities)` fail on shape mismatch. | Add a guard: `N = jnp.maximum(trajectory.shape[0], 2)` or assert `N >= 2`. |
| 9 | **RISK** | `src/ham/training/losses.py:274` | Inside `L_smooth`, `v_norm_sq = jnp.dot(v_pt, jnp.dot(H_pt, v_pt))` can be negative if `H_pt` is not positive-definite (e.g. on a Hyperboloid with Minkowski signature). The subsequent `jnp.sqrt(v_norm_sq + self.epsilon**2)` would compute `sqrt(negative + 1e-8)` which may still be negative under the square root, producing `NaN`. | Use `jnp.sqrt(jnp.maximum(v_norm_sq + self.epsilon**2, self.epsilon**2))` to clamp. |
| 10 | **RISK** | `src/ham/training/losses.py:155–160` | `MetricSmoothnessLoss` uses `jax.jacfwd(get_w_single)(parent_z)` which computes a full Jacobian matrix. For high-dimensional latent spaces (e.g. `D=64`), this allocates a `(D, D)` matrix per sample. Memory cost scales as $O(BD^2)$ per batch. Not a bug, but a performance footgun. | Document the scaling or provide an option for stochastic Hutchinson trace estimation of $\|\nabla W\|_F^2$. |
| 11 | **RISK** | `src/ham/training/losses.py:119–124` | `ContrastiveAlignmentLoss` does not check for `_get_zermelo_data` — it calls `model.metric._get_zermelo_data(parent_z)` unconditionally. If this loss is used with a non-Randers metric, it will raise `AttributeError` at runtime. Every other loss that calls `_get_zermelo_data` has an `if hasattr(...)` guard. | Add the standard `hasattr` guard with an identity/zeros fallback, or make the loss's `__init__` validate the metric type. |
| 12 | **STYLE** | `src/ham/training/losses.py:57,61,105,106` | Inconsistent epsilon constants: `1e-6` (lines 57, 61, 481), `1e-8` (lines 105, 106), `1e-4` (line 241, EL epsilon). The codebase defines canonical constants in `ham/utils/math.py` (`GRAD_EPS=1e-12`, `NORM_EPS=1e-8`, etc.) but none are used here. | Import and use the canonical constants from `ham.utils.math` for all safe-division guards. |
| 13 | **STYLE** | `src/ham/training/losses.py:8–9` | `LossComponent.weight` and `LossComponent.name` are not declared with `eqx.field(static=True)`. Since they are non-JAX values (Python float and str), they should be marked static to avoid being traced as dynamic pytree leaves by Equinox. Currently works because Equinox auto-handles non-array leaves, but being explicit is best practice. | Add `weight: float = eqx.field(static=True)` and `name: str = eqx.field(static=True)`. |
| 14 | **STYLE** | `src/ham/training/losses.py:422–430` | `WindAssistedTrajectoryAlignmentLoss.__call__` creates a new `ExponentialMap` instance on every call: `ivp_shooter = ExponentialMap(max_steps=self.rollout_steps)`. This is wasteful; the solver should be stored as an attribute. | Store the `ExponentialMap` instance as an `eqx.Module` field in `__init__`. |
| 15 | **STYLE** | `src/ham/training/losses.py:1–6` | `ExponentialMap` is imported at the top level but only used by `WindAssistedTrajectoryAlignmentLoss`. Minor import cleanliness. | Move import inside the class or keep it — low impact. |

## Test Coverage Assessment

| Public Class/Function | Tested? | Test Location | Notes |
|---|---|---|---|
| `LossComponent` | **Yes** | `tests/test_pipeline.py:171–203` | Contract tests (scalar output, weight scaling) via `MSELoss` subclass |
| `ReconstructionLoss` | **Yes** | `tests/test_joint_training.py:86` | Used in pipeline integration test |
| `KLDivergenceLoss` | **Yes** | `tests/test_joint_training.py:86` | Used in pipeline integration test |
| `ZermeloAlignmentLoss` | **Partial** | `tests/test_joint_training.py:18` | Imported but not directly tested in isolation |
| `GeodesicSprayLoss` | **No** | — | No test coverage |
| `VelocityDirectionAlignmentLoss` | **No** | — | No test coverage |
| `ContrastiveAlignmentLoss` | **Yes** | `tests/test_joint_training.py:135` | Used in metric phase integration test |
| `MetricAnchorLoss` | **Yes** | `tests/test_joint_training.py:98` | Used in metric phase integration test |
| `MetricSmoothnessLoss` | **Partial** | `tests/test_joint_training.py:19` | Imported but not directly tested |
| `LongTrajectoryAlignmentLoss` | **No** | — | No test coverage |
| `EulerLagrangeResidualLoss` | **No** | — | No test coverage; complex JAX differentiation logic untested |
| `AVBDPathEnergyLoss` | **No** | — | No test coverage |
| `WindThermodynamicLoss` | **No** | — | No test coverage |
| `KinematicPriorLoss` | **No** | — | No test coverage |
| `FinslerActionMatchingLoss` | **No** | — | No test coverage |
| `WindAssistedTrajectoryAlignmentLoss` | **No** | — | No test coverage |
| `FinslerianFlowMatchingLoss` | **No** | — | No test coverage |
| `_solve_and_integrate_impl` | **No** | — | No test coverage |
| `_solve_avbd_trajectory_impl` | **No** | — | No test coverage |

**Gap analysis:** 11 of 16 public loss classes have no dedicated unit test. The most critical gaps are `EulerLagrangeResidualLoss` (complex nested `jax.grad`/`jax.jvp` that should be validated with `jax.test_util.check_grads`), `GeodesicSprayLoss` (uses `inner_product` which may be negative), and `_solve_and_integrate_impl` (division by `N-1` edge case). Recommend adding at minimum:
- Smoke tests for each loss returning a finite scalar on a mock model.
- Gradient tests (`check_grads`) for `EulerLagrangeResidualLoss` and `FinslerianFlowMatchingLoss`.
- Edge-case tests: zero velocity, identity metric, single-point trajectory.

## Positive Patterns

1. **`eqx.filter_checkpoint` on solver calls** (`_solve_and_integrate`, `_solve_avbd_trajectory`): Correctly reduces memory usage during backpropagation through the AVBD solver by recomputing activations instead of storing them.
2. **`EulerLagrangeResidualLoss` JVP pattern** (lines 287–290): Excellent use of `jax.jvp` to compute Hessian-vector products efficiently without materializing the full Hessian. This is the canonical JAX pattern for physics-informed losses.
3. **`stop_gradient` on frozen H** (line 306): Correctly prevents gradient flow through the metric tensor used for residual norm evaluation, avoiding circular gradients.
4. **Modular LossComponent design**: Clean separation via `eqx.Module` inheritance, enabling declarative phase composition in `HAMPipeline`. Aligns well with `spec/ARCH_SPEC.md § 5` module structure.
5. **`safe_norm` usage** in `VelocityDirectionAlignmentLoss`: Correctly imports and uses the canonical gradient-safe norm from `ham.utils.math` (though not used consistently across all losses).
