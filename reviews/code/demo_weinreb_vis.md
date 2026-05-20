# Code Review: `examples/demo_weinreb_vis.py`

**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary

This demo generates a synthetic differentiation tree on the Hyperboloid and visualises it on the Poincaré disk. It has one clear bug—the PRNG key is not re-split inside the inner branch loop, causing all sibling branches to collapse onto identical points. There are also two moderate risks around vector overwriting and incorrect norm usage, plus minor style issues around hardcoded dimensions and redundant object construction.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `examples/demo_weinreb_vis.py:30-34` | `subkey` is produced once per parent (`key, subkey = jax.random.split(key)` at line 30) but reused unchanged for every `b in range(n_branches)` at line 34. All branches from the same parent receive the same `raw_dir`, so sibling children are placed at the exact same point and the tree degenerates. | Split a new sub-key per branch: move the split inside the inner loop, e.g. `key, subkey = jax.random.split(key)` immediately before line 34, or use `jax.random.split(subkey, n_branches)` before the loop and index per `b`. |
| 2 | **RISK** | `examples/demo_weinreb_vis.py:64-69` | When a parent has multiple children, `vectors[p_idx] = v` (line 69) is overwritten by each successive child. Only the last child's log-map vector survives, so the wind field silently discards earlier branch directions. | Accumulate with `vectors[p_idx] += v` (and optionally normalise afterwards), or store one vector per edge rather than per node. |
| 3 | **RISK** | `examples/demo_weinreb_vis.py:37-38` | `jnp.linalg.norm` computes the Euclidean (Frobenius) norm of the tangent vector, not the Minkowski norm used by the Hyperboloid geometry. The intended `step_size = 0.6` therefore does not correspond to a geodesic distance of 0.6. | Use `manifold._minkowski_norm(tangent_dir)` (or expose a public `norm` method) for normalisation consistent with the geometry. |
| 4 | **STYLE** | `examples/demo_weinreb_vis.py:33` | Ambient dimension is hardcoded as `3` in `jax.random.normal(subkey, (1, 3))`. If `intrinsic_dim` is changed the shape will silently mismatch. | Use `(1, manifold.ambient_dim)` instead. |
| 5 | **STYLE** | `examples/demo_weinreb_vis.py:62` | A second `Hyperboloid(intrinsic_dim=2)` is instantiated in `main()` even though `generate_synthetic_tree` already creates one internally. | Return the manifold from `generate_synthetic_tree` and reuse it, or accept it as a parameter. |
| 6 | **STYLE** | `examples/demo_weinreb_vis.py:84` | Output filename `"weinreb_hyperbolic_vis.png"` is a bare relative path with no existence check on the target directory. Running the script from an unexpected CWD may cause confusion. | Use `pathlib.Path` or `os.path.join` relative to the script's own directory, or at minimum document the expected CWD. |

## Test Coverage Assessment

This is a standalone demo script with no corresponding test file in `tests/`. Demo scripts are not required to have dedicated unit tests, but the underlying APIs (`Hyperboloid.exp_map`, `Hyperboloid.log_map`, `plot_poincare_disk`) are tested via:

- `tests/test_hyperboloid.py` — covers `exp_map`, `log_map`, `project`, `to_tangent`.
- `tests/test_surfaces.py` — covers surface classes including `Hyperboloid`.

No gaps in library-level coverage are introduced by this demo.

## Positive Patterns

- Clean separation between data generation (`generate_synthetic_tree`) and visualisation (`main`).
- Correct use of `manifold.to_tangent` to project random ambient vectors onto the tangent plane before stepping.
- Proper use of `jax.random.split` for top-level key management (the issue is only the missing inner split).
- Good use of the library's `exp_map`/`log_map` API rather than manual coordinate formulas.
