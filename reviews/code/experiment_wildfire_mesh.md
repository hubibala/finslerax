# Code Review: experiment_wildfire_mesh.py
**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-18
**Arch Spec Version:** 1.1.0 (ARCH_SPEC.md)

## Summary

The script follows Phase W1 conventions closely. The key HAMTools patterns (Equinox Module, `eqx.filter_value_and_grad`, `eqx.is_inexact_array` optimizer filter) are applied correctly. Three RISKs and two STYLEs are noted.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | RISK | `build_scene_mesh` L~155 | `jnp.mean(mesh.vertices[mesh.faces, 2], axis=1)` — fancy indexing with a non-static `faces` array inside a JIT context is fine for forward pass, but if this is ever traced under `jax.grad`, the index array creates a non-differentiable branch. Currently called outside gradient, so safe. | Add a comment `# called outside gradient — mesh is a compile-time constant`. |
| 2 | RISK | `_predict_arrivals_mesh_chunked` | `jax.vmap(_single_arrival)` is called with `_single_arrival` closing over `bound_metric` which contains `TriangularMesh`. `TriangularMesh.get_face_weights` itself calls `jax.vmap(dist_fn)(self.triangles)`. Nested vmaps are valid in JAX but produce O(F × chunk_size) distance computations per call. For large meshes (F≫1000) this may OOM. | Document chunk_size sensitivity; add a config parameter for per-chunk vmap depth. Currently `chunk_size=50` with F=722 is fine. |
| 3 | RISK | `run_synthetic_mesh` training loop | `def _loss(m, _fc5=..., _ffc=..., ...)` is redefined on every epoch iteration. JAX traces a new computational graph each epoch, which is correct but defeats XLA caching. Phase W1 avoids this by defining `_loss` once outside the loop. | Hoist `_loss` outside the epoch loop (define it once with the closed-over constants). |
| 4 | STYLE | `_val_pearson_r_mesh` | Repeats the fuel-embedding lookup + concatenation logic that `bind_mesh_scenario` already encapsulates. | Replace the inline lookup with a call to `bind_mesh_scenario(metric, face_cov_5, face_fuel_codes, scenario.weather_vec)`. |
| 5 | STYLE | `train_scene_mesh` test evaluation block | The fuel-embedding lookup is also repeated inline (not calling `bind_mesh_scenario`). | Same fix as #4. |

---

## Test Coverage Assessment

This is an experiment script, not a library module; unit tests are not strictly required. A `tests/test_experiment_wildfire_mesh.py` smoke test that imports and calls `run_synthetic_mesh(get_config(quick=True), ...)` would be sufficient for CI.

---

## Positive Patterns

- `eqx.is_inexact_array` (not `eqx.is_array`) for optimizer initialization correctly excludes integer mesh indices (`faces`, `face_fuel_codes`) from the Adam state. **STRONG**
- Default-argument capture `_fc5=face_cov_5` in loss closures prevents late-binding closure bugs. **STRONG**
- Gradient accumulation over batch_size_fires follows the same pattern as Phase W1. **STRONG**
