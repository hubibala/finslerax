# Code Review: `bio/vae.py`
**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

The VAE module provides a `WrappedNormal` distribution on manifolds and a `GeometricVAE` that combines encoder/decoder networks with Finsler geometry and Zermelo navigation. The module is compact (148 lines) and structurally sound as an `eqx.Module` hierarchy. However, it contains several issues: a **BUG** in `WrappedNormal.kl_divergence_std_normal` that computes a Euclidean KL rather than a geometry-aware one (inconsistent with the manifold-based sampling); a **RISK** from `isinstance` checks that break under `jax.jit` tracing; **RISK** from `hasattr` branching in `loss_fn` that is not trace-safe; and missing test coverage for the monolithic `loss_fn`, `project_control`, and `decode` methods. The `project_control` JVP-based velocity projection is an elegant pattern worth preserving.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/bio/vae.py:44` | `WrappedNormal.kl_divergence_std_normal` computes a standard Euclidean-space KL divergence formula: $-\log\sigma + \frac{\sigma^2}{2} - \frac{1}{2}$. This is only correct for a flat (Euclidean) ambient space. For the Hyperboloid or Sphere, the KL between a wrapped normal and the uniform/standard prior requires a log-determinant Jacobian correction from the exponential map. The current formula silently under- or over-estimates the true KL on curved manifolds. | Implement the manifold-specific KL correction. For the Hyperboloid, this involves the $\log\bigl(\sinh(r)/r\bigr)$ volume element term. At minimum, document the approximation explicitly so users are aware. |
| 2 | **RISK** | `src/ham/bio/vae.py:28-37` | `WrappedNormal.sample` uses `isinstance(self.manifold, Hyperboloid)` for control flow. This is not safe inside `jax.jit`-traced functions because `isinstance` evaluates at trace time and the branch choice becomes baked into the compiled program. If the same jitted function is called with different manifold types, only the first branch will ever execute. | Replace `isinstance` dispatch with a method on the `Manifold` base class (e.g., `manifold.lift_tangent(origin, v_flat)`) so each manifold handles its own embedding logic. This also eliminates the fragile `else` fallback. |
| 3 | **RISK** | `src/ham/bio/vae.py:131-133` | `loss_fn` uses `hasattr(self.metric, '_get_zermelo_data')` for branching. Inside a jitted context, `hasattr` on a traced `eqx.Module` is evaluated at trace time, which is safe for the initial trace but produces wrong results if the same jitted function is later called with a metric that does/doesn't have `_get_zermelo_data`. More critically, the `else` branch returns `jnp.zeros_like(u_lat)`, which makes `align_loss` and `spray_loss` meaningless for non-Zermelo metrics — the total loss silently degrades. | Define `_get_zermelo_data` on the `FinslerMetric` base class with a default implementation (identity H, zero W) so that the `hasattr` check is eliminated. |
| 4 | **RISK** | `src/ham/bio/vae.py:136-137` | `loss_fn` calls `self.manifold._minkowski_norm(W)` and `self.manifold._minkowski_norm(u_lat)`. These methods only exist on `Hyperboloid`. If `GeometricVAE` is instantiated with a `Sphere` or `EuclideanSpace` manifold, this will crash with `AttributeError`. The monolithic `loss_fn` is therefore Hyperboloid-only despite no type guard. | Use a generic norm method (e.g., `safe_norm` from `ham.utils.math`) or add a `norm` method to the `Manifold` ABC. The modular losses in `losses.py` have the same issue (duplicated from this code). |
| 5 | **RISK** | `src/ham/bio/vae.py:44` | `kl_divergence_std_normal` uses `jnp.log(self.scale + 1e-6)` with a hardcoded epsilon. If `scale` is very small (e.g., from a poorly initialized network), the `1e-6` additive constant biases the log. Conversely, if operating in `float16`, `1e-6` is below the representable range. | Use `jnp.log(jnp.maximum(self.scale, 1e-6))` (clamp, not add) for correctness, or better, use the project's canonical `NORM_EPS` from `ham.utils.math`. |
| 6 | **RISK** | `src/ham/bio/vae.py:86` | `_get_dist` applies `jax.nn.softplus(log_scale) + 1e-4` to produce the scale. This is reasonable, but the variable is named `log_scale` even though `softplus` is applied to it (softplus is not `exp`). If a future maintainer replaces `softplus` with `exp` (matching the name), the numerical stability guarantee is lost. | Rename `log_scale` → `scale_raw` or `scale_param` to avoid semantic confusion. |
| 7 | **STYLE** | `src/ham/bio/vae.py:55-57` | `GeometricVAE` declares `classifier_head` with a default `None` and a comment `# add this field`, suggesting incomplete refactoring. The field is never used anywhere in the class body. | Remove `classifier_head` if unused, or implement the classification forward path. Remove the `# add this field` comment regardless. |
| 8 | **STYLE** | `src/ham/bio/vae.py:148` | `loss_fn` hardcodes loss weights (`1e-4`, `0.1`, `1.0`) with no parameterization. The modular `LossComponent` classes in `training/losses.py` have already superseded this function. The docstring says "backwards compatibility" but the function creates a maintenance burden: any bug fix must be applied in two places. | Deprecate or remove `loss_fn` in favor of the modular losses. If kept for backwards compatibility, implement it as a thin wrapper that delegates to the `LossComponent` instances. |
| 9 | **STYLE** | `src/ham/bio/vae.py:54` | `metric: FinslerMetric` type annotation is correct, but `manifold: Any` on line 56 (and `solver: Any`) loses type information. The manifold is always `metric.manifold`, so it's redundant storage. | Type `manifold` as `Manifold` (imported from `ham.geometry.manifold`) and `solver` as `Optional[GeodesicSolver]` or similar. Consider removing the redundant `self.manifold` field and using `self.metric.manifold` throughout. |
| 10 | **STYLE** | `src/ham/bio/vae.py:1-5` | `Tuple` and `Optional` are imported from `typing`, but since Python 3.10+ these are available as built-in generics (`tuple`, `X | None`). | Use modern type syntax: `tuple[int, ...]`, `Any | None`. Minor, no functional impact. |

## Test Coverage Assessment

| Public API | Test File | Covered? | Notes |
|---|---|---|---|
| `WrappedNormal.__init__` | `test_hyperbolic_vae.py` | Yes | Tested via `test_wrapped_normal_sampling` |
| `WrappedNormal.sample` | `test_hyperbolic_vae.py` | Yes | Checks samples lie on manifold |
| `WrappedNormal.kl_divergence_std_normal` | `test_hyperbolic_vae.py` | **Partial** | Called indirectly via `KLDivergenceLoss`; no test verifies the KL value is correct (e.g., against a known analytical result) |
| `GeometricVAE.__init__` | `test_hyperbolic_vae.py`, `test_joint_training.py` | Yes | |
| `GeometricVAE._get_dist` | `test_hyperbolic_vae.py` | Yes | Called indirectly via forward pass |
| `GeometricVAE.encode` | `test_joint_training.py` | **Indirect** | Used inside loss components but never tested in isolation |
| `GeometricVAE.decode` | `test_hyperbolic_vae.py` | **Indirect** | Called inside `ReconstructionLoss` |
| `GeometricVAE.project_control` | — | **No** | No dedicated test. Critical JVP-based method with no coverage for correctness of the tangent projection. |
| `GeometricVAE.loss_fn` | — | **No** | The monolithic loss function has zero test coverage. `test_hyperbolic_vae.py` uses the modular losses instead. |
| `GeometricVAE` with custom `encoder_net`/`decoder_net` | — | **No** | The `__init__` path for user-provided networks is untested. |
| `GeometricVAE` with `classifier_head` | — | **No** | Field exists but is never exercised. |
| `WrappedNormal.sample` on Sphere/Euclidean | — | **No** | Only tested with `Hyperboloid`; the `isinstance`-based branching for Sphere is untested. |

**Gap Analysis:** 4 of 7 public methods lack dedicated tests. The `project_control` method is particularly critical as it uses `jax.jvp` to map data-space velocities into latent-space tangent vectors — an incorrect projection would silently corrupt all downstream alignment and spray losses. The `loss_fn` method is untested but also deprecated in favor of modular losses. The `WrappedNormal` distribution is only tested on `Hyperboloid`; the `Sphere` and fallback branches in `sample` are completely uncovered.

## Positive Patterns

1. **JVP-based velocity projection** (`project_control`, line 108–115): Using `jax.jvp` to push forward RNA velocity through the encoder is the correct and efficient approach. It avoids materializing the full Jacobian and composes cleanly with downstream `jit`/`grad`.

2. **Equinox module discipline**: Both `WrappedNormal` and `GeometricVAE` correctly inherit from `eqx.Module`, use `eqx.field(static=True)` for non-differentiable configuration, and avoid mutable state. This makes them valid JAX PyTrees out of the box.

3. **Softplus for scale**: Using `jax.nn.softplus` + floor (line 86) to produce the distribution scale is numerically safer than `exp` (which can overflow) and ensures strictly positive output.

4. **Clean separation of concerns**: The encoder outputs raw parameters, `_get_dist` handles the manifold projection and scale activation, and `sample`/`kl_divergence_std_normal` are encapsulated on `WrappedNormal`. This makes it straightforward to swap distribution families.

5. **Modular loss migration**: The existence of both `loss_fn` and the `LossComponent` classes in `training/losses.py` shows a deliberate migration toward composable losses, which is the right architectural direction per `spec/ARCH_SPEC.md`.
