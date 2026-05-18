# Code Review: terrain.py
**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-18
**Arch Spec Version:** ARCH_SPEC.md (as of 2026-05-18)

## Summary
The implementation is clean, follows HAMTools conventions, and all 13 tests pass. Two RISK items and one STYLE item are raised. No BUG-severity issues are present.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | RISK | `terrain.py:CovariateMeshRanders.metric_fn` | `face_idx = jnp.argmax(weights)` is non-differentiable. If the metric is ever used inside a `jax.grad` call that traces through `face_idx` (e.g. as a penalty on face selection), the gradient w.r.t. `face_covariates` flows through the `face_covariates[face_idx]` gather, which works due to JAX dynamic indexing, but the gradient w.r.t. `x` does not flow through `argmax`. For the intended usage (evaluation on a fixed mesh during training), this is acceptable since gradients are not needed w.r.t. `x` per face-selection. | Add a docstring note: "Gradient w.r.t. x does not propagate through face selection (argmax is non-differentiable). For differentiable face blending, replace argmax with softmax-weighted interpolation of face parameters." |
| 2 | RISK | `terrain.py:interpolate_covariates_to_vertices` | The `mesh` parameter is accepted but never used. This is not a bug but is misleading — a caller may expect vertex positions to influence the interpolation (e.g. for non-aligned meshes). | Change signature to not accept `mesh`, or add an assertion that `mesh.vertices.shape[0] == H * W`. |
| 3 | STYLE | `terrain.py:dem_to_mesh` | Vertices are cast to `float32` at the end. The rest of the codebase (wildfire.py) uses `float64` for rasters and world coordinates. | Use `jnp.asarray(elevation_raster)` without hardcoding dtype, letting the caller control precision. Or match `wildfire.py`'s `float64` convention for consistency. |

---

## Test Coverage Assessment

| Public Function / Class | Tested? | Notes |
|------------------------|---------|-------|
| `dem_to_mesh` | ✓ | Shape, vertex coords, dtype all tested |
| `interpolate_covariates_to_vertices` | ✓ | Single and multi-channel tested |
| `pixel_to_world_3d` | ✗ | No test present — function is simple but API surface warrants a smoke test |
| `compute_face_normals` | ✓ | Flat DEM and unit-length both tested |
| `compute_face_slopes_aspects` | ✓ | Flat DEM slope=0 tested |
| `CovariateMeshRanders.__init__` | ✓ (indirect) | Via bind_scene test |
| `CovariateMeshRanders.bind_scene` | ✓ | Immutability tested |
| `CovariateMeshRanders._get_face_params` | ✗ | Private, but worth a unit test on shape correctness |
| `CovariateMeshRanders.metric_fn` | ✓ | Positive, zero velocity, homogeneity, no-wind all tested |

**Gap:** `pixel_to_world_3d` has no test. Low priority but should be added.

---

## Positive Patterns

1. **Static fields used correctly:** `eps_G`, `max_G`, `max_b_norm`, `use_wind`, `fuel_emb_dim` are all `eqx.field(static=True)`, matching `CovariateConditionedRanders` conventions. This ensures JIT compilation does not retrace on config changes.
2. **`bind_scene` uses `eqx.tree_at`:** Follows the established pattern from `CovariateConditionedRanders.bind_scene` — returns a new frozen pytree with updated leaves rather than mutating the object.
3. **Phase W2 approximation clearly documented:** Both in the class docstring and in `metric_fn`, the isotropic approximation is explicitly noted as a Phase W2 simplification with a future work pointer.
4. **`GRAD_EPS` imported from `ham.utils.math`:** Consistent with the codebase convention. No magic numbers.
5. **Test for homogeneity:** `test_covariate_mesh_randers_homogeneous` explicitly verifies $F(x, \lambda v) = \lambda F(x, v)$ at `rtol=1e-4`, which is the core Finsler property.
