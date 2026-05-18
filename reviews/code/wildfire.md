# Code Review: wildfire.py
**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-18
**Arch Spec Version:** 1.1.0 (ARCH_SPEC.md)

## Summary

The implementation is **sound and production-quality**. It follows HAMTools conventions (Equinox `Module`, `FinslerMetric` ABC, `GRAD_EPS`/`PSD_EPS` from `ham.utils.math`), is compatible with `jit`/`vmap`/`grad`, and has good test coverage. Two RISKs and one STYLE finding are noted; none are showstoppers.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | RISK | wildfire.py:103 | `jnp.sqrt(jnp.maximum(norm_sq, GRAD_EPS))` — previously used `0.0`; now fixed with `GRAD_EPS`. Verify that the gradient of `scale` w.r.t. `b` is finite even at `b=0` (from MLP output). | Already fixed in current version. Add `test_gradients_near_zero_v` for `project_b_norm` itself. |
| 2 | RISK | wildfire.py:_bilinear_interp | `x0 = jnp.floor(px).astype(jnp.int32)` — if `px` is traced as float64, the cast is fine. However `fx = px - x0` will implicitly convert `x0` (int32) to float64 before subtraction, producing a float64 fractional weight. Verify this is consistent with `v00 * (1-fx) * (1-fy)` dtype expectations. | No action required — JAX promotes correctly; add a dtype assertion in tests if strict dtype contracts are needed. |
| 3 | RISK | wildfire.py:pixel_spacing_m | Declared as `jax.Array` (non-static leaf) to allow `eqx.tree_at` updates, deviating from the task spec which states `eqx.field(static=True)`. This means `pixel_spacing_m` is treated as a differentiable parameter — gradients will flow through it. | Wrap usages in `jax.lax.stop_gradient(self.pixel_spacing_m)` inside `_bilinear_interp` to prevent unintended gradient flow through the raster resolution. |
| 4 | STYLE | wildfire.py:137 | Module docstring lists `pixel_spacing_m: jax.Array` but the class-level comment says "Stored as a regular JAX leaf so eqx.tree_at can update it". This discrepancy from the spec's `eqx.field(static=True)` annotation should be documented more explicitly. | Add a `# Note:` comment at the field declaration explaining the deviation. |

---

## Test Coverage Assessment

| Public Function / Method | Tested | Notes |
|---|---|---|
| `project_spd` | ✓ | Eigenvalue bounds, already-SPD, vmap, grad |
| `project_b_norm` | ✓ | Norm bound, already-in-bound, grad |
| `CovariateConditionedRanders.__init__` | ✓ (implicit via fixtures) | |
| `bind_scene` | ✓ | Implicit — `bound_model` fixture calls it |
| `_bilinear_interp` | Partial | Tested indirectly via metric_fn; no direct unit test for boundary clipping |
| `_get_covariates` | Partial | Tested indirectly; no test for fuel embedding gradient |
| `_get_params` | ✓ (indirect) | Covered by `test_metric_fn_riemannian_matches_sqrt_Gv` |
| `metric_fn` | ✓ | Positivity, homogeneity, Riemannian limit, JIT, vmap, grad w.r.t. x and v |

**Gap:** No test verifies gradient flow through `fuel_embedding` (differentiability w.r.t. the embedding table). This is listed in task requirements. Recommend adding `test_fuel_embedding_gradient`.

---

## Positive Patterns

1. **Python `if self.use_wind`** inside `_get_params` correctly exploits the static field for trace-time specialisation — this is the recommended Equinox pattern.
2. **`is_leaf=lambda x: x is None`** in `bind_scene` correctly handles optional fields — this is the documented Equinox workaround for `eqx.tree_at` with `None` leaves.
3. **`jnp.stack` instead of `jnp.array`** for constructing traced 2×2 matrices avoids potential concretisation issues inside `jit`.
4. **`GRAD_EPS` from `ham.utils.math`** used consistently for sqrt guards, matching the library-wide epsilon convention.
5. **`is_zero` mask pattern** in `metric_fn` mirrors `zoo/randers.py` exactly — consistent with the codebase convention for zero-vector handling.
