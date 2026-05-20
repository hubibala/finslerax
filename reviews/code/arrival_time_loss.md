# Code Review: `ArrivalTimeLoss` & `test_arrival_time_loss.py`
**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-17  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`ArrivalTimeLoss` (lines 538–597 of `src/ham/training/losses.py`) is a well-structured loss for metric recovery from arrival-time observations. It correctly uses `jax.vmap` over observation points and computes discrete arc length from AVBD-solved geodesic paths. However, it breaks the established `LossComponent` API (extends `eqx.Module` directly instead of `LossComponent`), its `__call__` signature is incompatible with the `HAMPipeline` contract, it lacks numerical guards on `metric_fn` at zero-length segments, and the test suite, while covering the essential cases, misses `jit` compilation, `vmap`-batching, and edge-case stability tests.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/training/losses.py:538` | `ArrivalTimeLoss` extends `eqx.Module` instead of `LossComponent`. This breaks the API contract established by all 16 other losses in the file. Specifically: (a) It lacks the `name` field used by `HAMPipeline` for logging/diagnostics. (b) Its `__call__(self, metric, source, x_obs, t_obs)` signature is incompatible with the standard `__call__(self, model, batch, key)` protocol, so it **cannot** be used with `HAMPipeline` or any code that iterates `LossComponent` instances. (c) `isinstance` checks or duck-typing on `LossComponent` will fail. | Either subclass `LossComponent` and override `__call__` with the standard `(model, batch, key)` signature (unpacking `source`, `x_obs`, `t_obs` from `batch`), or document this as an intentionally standalone loss not usable with `HAMPipeline`. Both approaches are valid; the current state is ambiguous. |
| 2 | **RISK** | `src/ham/training/losses.py:585–587` | `segments = jnp.diff(path, axis=0)` can produce zero-length segments when two adjacent AVBD vertices coincide (e.g., solver converges early or boundary point duplicated). `metric_fn(positions, segments)` then evaluates `F(x, 0)` which, for the Randers implementation (`zoo/randers.py:67–81`), guards against this via `is_zero = v_sq_raw < GRAD_EPS`, returning `0.0`. So no NaN occurs, but the arc length silently drops the contribution of that segment, which can bias the loss. More critically, if a *different* metric subclass is plugged in that does **not** guard `metric_fn(x, 0)`, this will produce NaN gradients. | Add a `safe_norm`-based guard on segments: `segments = jnp.where(jnp.sum(segments**2, axis=-1, keepdims=True) < GRAD_EPS, jnp.ones_like(segments) * jnp.sqrt(GRAD_EPS), segments)` or equivalently add a comment documenting the reliance on `metric_fn`'s zero-guard. |
| 3 | **RISK** | `src/ham/training/losses.py:579–588` | The `single_arrival_time` closure captures `self`, `metric`, and `source` from the enclosing scope and is vmapped over `x_obs`. Inside, it calls `self.solver.solve(metric, source, x_target, n_steps=self.solver_steps)`. The AVBD solver uses `jax.lax.scan` internally (in `train_mode=True`). Nesting `vmap` over a `lax.scan` that itself contains `vmap` (AVBD's vertex sweep) creates a multi-level vmap stack. This is technically correct in JAX but (a) can cause excessive memory usage ($O(K \times T \times D)$ for $K$ observations), and (b) forces JIT compilation of the full scan body at every unique `(solver_steps, solver_iters)` combo, with no reuse across different `n_steps` values since `n_steps` determines the scan length (a static argument). | Document the memory cost. Consider batching manually with `jax.lax.map` (sequential vmap) if memory is a concern, or provide a `batch_size` kwarg to chunk the `vmap`. |
| 4 | **RISK** | `src/ham/training/losses.py:588` | `step_costs = jax.vmap(metric.metric_fn)(positions, segments)` computes $F(x_k, \Delta x_k)$, not $F(x_k, \Delta x_k / \Delta t)$. For arc length, we want $\sum F(x_k, \Delta x_k)$ which is the discrete approximation of $\int F(\gamma, \dot\gamma) dt$ with $\Delta t = 1$. This is consistent with the docstring's formula. However, for the Randers metric, `F` is 1-homogeneous in $v$, so $F(x, c \cdot v) = c \cdot F(x, v)$ for $c > 0$. The choice of parameterization ($\Delta t = 1$ vs. $\Delta t = 1/N$) does not affect the sum because the segment length already embeds $\Delta t$. This is **correct** but non-obvious — it silently depends on 1-homogeneity. If a non-homogeneous metric is ever used, the result would be wrong. | Add a brief comment: `# Correct for 1-homogeneous F: F(x, Δx) = F(x, v)*Δt`. |
| 5 | **RISK** | `tests/test_arrival_time_loss.py:38–43` | `test_identity_metric_distance` uses a tolerance of `loss < 0.05` for a squared-error loss. With 3 observations, a per-observation error of ~0.13 in arc length would pass. The AVBD solver with only 80 iterations and 15 steps on a Euclidean metric should converge to machine precision. The loose tolerance may mask solver regressions. | Tighten to `loss < 0.01` or even `loss < 1e-3` after verifying solver convergence. |
| 6 | **RISK** | `tests/test_arrival_time_loss.py:45–64` | `test_gradient_flows` only checks `jnp.isfinite` on gradient leaves. It does not check that gradients are non-zero (a common failure mode where `stop_gradient` or a disconnected computation graph silently kills gradients). The comment says "Check that at least some gradients are non-zero" but the actual assertion only checks finiteness. | Add: `has_nonzero = any(jnp.any(jnp.abs(g) > 0) for g in grad_leaves if g.size > 0)` and `assert has_nonzero`. |
| 7 | **STYLE** | `src/ham/training/losses.py:538` | Every other loss in the file has a `name` field for pipeline logging. `ArrivalTimeLoss` lacks one. Even if it's not used with `HAMPipeline`, consistency matters for diagnostics. | Add `name: str = eqx.field(static=True, default="ArrivalTime")` or subclass `LossComponent`. |
| 8 | **STYLE** | `src/ham/training/losses.py:548` | The `solver` field is typed as `eqx.Module`. It should be typed as `AVBDSolver` (or `GeodesicSolver` ABC) for clarity and static analysis. The import for `AVBDSolver` is not present at the top of `losses.py`. | Use `from ham.solvers.avbd import AVBDSolver` and type accordingly, or use a `Protocol` type. |

## Test Coverage Assessment

| Public Method / Behavior | Tested? | Test Location | Notes |
|---|---|---|---|
| `ArrivalTimeLoss.__init__` | **Yes** | `tests/test_arrival_time_loss.py:34` | Implicitly via construction |
| `ArrivalTimeLoss.__call__` (identity metric) | **Yes** | `tests/test_arrival_time_loss.py:30–43` | Verifies loss ≈ 0 for correct distances on flat space |
| `ArrivalTimeLoss.__call__` (gradient flow) | **Yes** | `tests/test_arrival_time_loss.py:45–64` | Checks finite gradients w.r.t. NeuralRanders params |
| `ArrivalTimeLoss.__call__` (loss ordering) | **Yes** | `tests/test_arrival_time_loss.py:66–82` | Correct T gives lower loss than wrong T |
| JIT compilation | **Partial** | `tests/test_arrival_time_loss.py:54` | Used inside `eqx.filter_jit` but no explicit recompilation test |
| vmap over batch | **No** | — | The `vmap` inside `__call__` is exercised, but no test vmaps over multiple source points |
| Edge case: source == target | **No** | — | Zero-distance edge case untested; may trigger zero-segment issues |
| Edge case: collinear points | **No** | — | All test targets are axis-aligned or diagonal; no general-position test |
| Edge case: high solver_steps | **No** | — | Performance/convergence not tested at different discretizations |
| Integration with `HAMPipeline` | **No** | — | Cannot work due to signature mismatch (Issue #1) |

### Gap Analysis

The test suite covers the three essential scenarios (correctness, differentiability, loss ordering) but lacks:
1. **Non-zero gradient assertion** (Issue #6) — currently only checks finiteness.
2. **Zero-distance edge case** — `source == x_obs[i]` would produce a degenerate geodesic.
3. **`vmap`-over-sources** — no test verifies that the loss can be batched across multiple source points.
4. **Tighter numerical tolerance** — the `0.05` threshold is too loose for a flat-space identity metric.

## Positive Patterns

1. **Clean use of `jax.vmap` over observation points** — the `single_arrival_time` closure is well-structured and avoids Python loops.
2. **Correct discrete arc length formula** — using `metric_fn` (not `energy`) gives the proper 1-homogeneous cost, consistent with `spec/MATH_SPEC.md § 1.2`.
3. **`eqx.Module` fields correctly marked** — `solver_steps` and `weight` are `static=True`, preventing retracing.
4. **Docstring quality** — the class and method docstrings are thorough, including shapes, mathematical formulation, and spec references.
5. **Test structure** — `_identity_metric()` helper and class-based test organization follow pytest best practices.
