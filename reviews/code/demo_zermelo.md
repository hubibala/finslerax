# Code Review: `examples/demo_zermelo.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0

## Summary

This demo script showcases Zermelo navigation on $S^2$ by comparing Riemannian vs Randers geodesics and a discrete mesh path. It is concise and the visualization code is clear. However, it contains one high-severity bug where `Sphere` is constructed with the wrong positional argument (radius passed as `intrinsic_dim`), a comment–code mismatch on the wind strength, and minor style issues in import ordering.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `examples/demo_zermelo.py:13` | `Sphere(radius)` passes `1.0` as the first positional arg `intrinsic_dim` (constructor signature is `__init__(self, intrinsic_dim: int = 2, radius: float = 1.0)`). This creates an $S^1$ (circle in $\mathbb{R}^2$) instead of $S^2$ (sphere in $\mathbb{R}^3$). The `ambient_dim` property returns 2 instead of 3, which is inconsistent with the 3D points and wind vectors used throughout the script. The bug is latent because the solver path does not currently check `ambient_dim`, but any downstream code relying on manifold metadata will be wrong. Other demos (e.g. `demo_trajectories.py:78`) correctly use `Sphere(radius=1.0)`. | Change to `Sphere(radius=radius)` or `Sphere(2, radius)`. |
| 2 | **RISK** | `examples/demo_zermelo.py:17` | Comment says "Strength 0.8 at equator" but the code uses `0.9 * base` on line 20. The wind magnitude at equatorial points is $0.9 \cdot \lVert\text{base}\rVert = 0.9$, which is close to the Zermelo causality bound $\lVert W \rVert_H < 1$ (with $H = I$). While technically valid, this leaves only 10% margin and the solver may become ill-conditioned. More importantly, the stale comment misleads readers about the actual configuration. | Update the comment to match the code (`0.9`), or reduce the wind strength to `0.8` to match the comment and improve numerical margin. |
| 3 | **STYLE** | `examples/demo_zermelo.py:1-4` | Imports are not organized per PEP 8 (stdlib → third-party → local). `numpy` is imported after `matplotlib`, and `jax` modules are mixed with plotting. | Group imports: (1) stdlib (none here), (2) third-party (`jax`, `jax.numpy`, `matplotlib.pyplot`, `numpy`), (3) local (`ham.*`). |
| 4 | **STYLE** | `examples/demo_zermelo.py:18-22` | `w_net` and `h_net` are bare functions while the `Randers` API expects callables typed as `Callable[[jnp.ndarray], jnp.ndarray]`. Using `lambda x: jnp.zeros(3)` inline (line 25) is inconsistent with the named-function pattern used for `w_net`. Additionally, `jnp.zeros(3)` hard-codes the dimension; consider `jnp.zeros_like(x)` for robustness. | Either define both as named functions or both as lambdas for consistency. Replace `jnp.zeros(3)` with `jnp.zeros_like(x)`. |
| 5 | **STYLE** | `examples/demo_zermelo.py:27-28` | The section comment says "Mission: South → North" but start/end are `[1,0,0]` (equator, x-axis) and `[0,0,1]` (north pole), so the path goes Equator → North, not South → North. | Update comment to "Equator → North Pole" to match the actual endpoints. |

## Test Coverage Assessment

This is a demo/example script and does not have a dedicated test file. No unit tests exercise the specific configuration used here (equatorial wind, Randers vs Riemannian comparison on $S^2$).

| Aspect | Covered? | Notes |
|--------|----------|-------|
| `AVBDSolver.solve` | Yes | Tested in `tests/test_solver.py` |
| `Randers` metric | Yes | Tested in `tests/test_zoo.py` |
| `Sphere` manifold | Yes | Tested in `tests/test_surfaces.py` |
| `TriangularMesh` | Yes | Tested in `tests/test_mesh.py` |
| Demo-specific integration (Randers on Sphere with strong wind) | No | No integration test covers this exact scenario |

## Positive Patterns

- **Clear sectioned structure**: The script is divided into numbered sections (Physics, Mission, Solve, Visualization) that make the workflow easy to follow.
- **Good use of `jax.vmap`**: Energy computation on line 41 correctly vmaps over trajectory segments rather than using a Python loop.
- **Informative visualization**: Wind vectors, indicatrices, and multiple paths are overlaid on a single plot with proper labels and legend.
- **Consistent use of HAM APIs**: Solver, metric, and visualization functions are used as designed in `spec/ARCH_SPEC.md`.
