# Code Review: `vis.py`
**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0  

## Summary

`src/ham/vis/vis.py` is a pure-visualization utility module (matplotlib helpers for 3D plots, indicatrices, and icosphere generation). It is not on any JAX-transform hot path, so the risks are limited to runtime errors and performance rather than gradient correctness. The main concerns are: (1) `plot_indicatrices` evaluates `metric_fn` in a Python loop instead of using `vmap`, which is extremely slow and will silently prevent JIT tracing; (2) several functions assume 3D inputs without validation, which will produce confusing IndexErrors on 2D data; (3) there are **zero tests** for this module. No critical correctness bugs were found.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | `src/ham/vis/vis.py:72` | `plot_indicatrices` calls `metric.metric_fn` in a Python `for` loop over `us_jax`. For `n_theta=40` per point this issues 40 individual JAX dispatches per point. With many points this is orders of magnitude slower than a single `vmap` call, and it prevents the metric from being JIT-compiled across the batch. | Replace the list comprehension with `costs = jnp.array(jax.vmap(metric.metric_fn, in_axes=(None, 0))(p_jax, us_jax))`. |
| 2 | **RISK** | `src/ham/vis/vis.py:60-62` | `plot_indicatrices` assumes points lie on a sphere-like surface (uses the point position as the surface normal `n = p / norm(p)`). This silently produces nonsense for any non-sphere manifold (torus, hyperboloid, paraboloid) and crashes for `p = [0,0,0]` despite the `1e-8` guard (the resulting normal is near-zero so the cross products degenerate). | Document the sphere-only assumption or accept an optional `normals` argument. For the origin guard, add an explicit check or fall back to a default normal. |
| 3 | **RISK** | `src/ham/vis/vis.py:29-33` | `plot_vector_field` hard-codes 3-column indexing (`p[:,0], p[:,1], p[:,2]`). Passing 2D data raises an `IndexError` with no helpful message. Same issue in `plot_trajectory` (line 52-53) and `plot_indicatrices`. | Add a shape assertion at the top: `assert p.shape[-1] == 3, f"Expected 3D points, got {p.shape[-1]}D"` or handle the 2D case separately. |
| 4 | **RISK** | `src/ham/vis/vis.py:97` | `generate_icosphere` returns `jnp.array(faces)` with integer dtype. JAX integer arrays cannot be used as indices in JIT-compiled code without static shapes. Returning faces as `jnp.array` provides no advantage over `np.array` and may confuse downstream JIT consumers. | Return `(jnp.array(verts), np.array(faces))` — vertices benefit from JAX (they go into `metric_fn`), but faces are purely topological and should stay as NumPy. |
| 5 | **STYLE** | `src/ham/vis/vis.py:2` | `from mpl_toolkits.mplot3d import Axes3D` is imported but never used directly. Matplotlib registers the `'3d'` projection on import, so this side-effect import works but triggers linter warnings. | Add a `# noqa: F401` comment or remove the import (modern matplotlib ≥ 3.2 auto-registers the projection). |
| 6 | **STYLE** | `src/ham/vis/vis.py:35-55` | `plot_trajectory` is wrapped in `# --- FIXED FUNCTION ---` / `# ----------------------` banner comments. These are development artifacts that should not be in released code. | Remove the banner comments. |
| 7 | **STYLE** | `src/ham/vis/vis.py:60` | `plot_indicatrices` is missing a docstring. All other public functions in the module either have or should have one. | Add a docstring describing parameters, the sphere-only assumption, and the `scale` semantics. |
| 8 | **STYLE** | `src/ham/vis/vis.py` | The module-level file has no module docstring. `ARCH_SPEC.md § 5` lists `vis/` as a first-class subpackage. | Add a one-line module docstring. |

## Test Coverage Assessment

| Public Function | Tested? | Gap |
|-----------------|---------|-----|
| `setup_3d_plot` | No | No test file exists for `vis`. Smoke test should verify it returns `(fig, ax)` and accepts `elev`/`azim`. |
| `plot_sphere` | No | Smoke test: call with default args, assert no exception. |
| `plot_vector_field` | No | Test with empty array (should be a no-op) and a 3D array. |
| `plot_trajectory` | No | Test all three input branches (`SolverResult`, tuple, raw array) and the empty-array guard. |
| `plot_indicatrices` | No | Test with a Euclidean metric on 1-2 points; assert the loop is closed (first == last row). |
| `generate_icosphere` | No | Test vertex count formula (`10 * 4^s + 2`), unit-norm of all vertices, and correct face shape. |

**Overall:** 0 / 6 public functions are tested. This is the largest gap in the `vis/` package.

## Positive Patterns

1. **Graceful empty-input handling.** Both `plot_vector_field` and `plot_trajectory` short-circuit on empty arrays — prevents matplotlib errors.
2. **`**kwargs` passthrough.** `plot_trajectory`, `plot_vector_field`, and `plot_indicatrices` all forward extra keyword arguments to matplotlib, keeping the API flexible without bloating the signature.
3. **Clean SolverResult dispatch.** `plot_trajectory` uses `hasattr(traj, 'xs')` duck-typing to accept multiple trajectory representations, which is idiomatic Python.
4. **Icosphere subdivision is correct.** Midpoint caching, final normalization, and face topology are properly implemented — a non-trivial algorithm done right.
