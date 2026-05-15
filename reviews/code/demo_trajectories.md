# Code Review: `examples/demo_trajectories.py`

**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary

This demo script compares three trajectory strategies on a sphere with a Rossby–Haurwitz wind field: an optimal Randers geodesic (BVP), pure passive advection, and a verification geodesic shot. The script is generally well-structured and readable as a demo, but contains one likely runtime bug (incorrect `isinstance` check against `typing.NamedTuple`), a shadowed `Trajectory` name that could confuse readers, an unexplained magic constant for velocity scaling, and duplicated wind-function definitions.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `examples/demo_trajectories.py:46` | `isinstance(trajectory, (tuple, NamedTuple))` — `typing.NamedTuple` is **not** a valid runtime type for `isinstance`. On Python ≥ 3.11 this raises `TypeError`; on earlier versions it silently returns `False`. The check currently works only because `tuple` in the union catches all named tuples (which are subclasses of `tuple`), but this is fragile and will break on newer Python. | Replace with `isinstance(trajectory, tuple)`. All `typing.NamedTuple` subclasses are subclasses of `tuple`, so the single check is sufficient. |
| 2 | **RISK** | `examples/demo_trajectories.py:16-18` | A local `Trajectory(NamedTuple)` with field `xs` is defined here, while `ham.solvers.avbd` exports its own `Trajectory` with fields `xs, vs, energy, constraint_violation`. The name collision means the local class shadows any future import of the AVBD one, and readers may confuse which `Trajectory` a given variable holds. | Rename the local container (e.g., `AdvectionResult`) or simply return the raw `jnp.ndarray` from `pure_advection`, since only positions are stored. |
| 3 | **RISK** | `examples/demo_trajectories.py:99` | `v_optimal_approx = traj_bvp.vs[0] * 40.0` — the magic constant `40.0` is unexplained. It appears to rescale the BVP initial velocity (which is normalised over `n_steps=40`) back to a physical magnitude, but if `n_steps` changes this scaling silently becomes wrong. | Derive the factor from `n_steps` explicitly, e.g. `v_optimal_approx = traj_bvp.vs[0] * n_steps_bvp`, and define `n_steps_bvp = 40` once. |
| 4 | **RISK** | `examples/demo_trajectories.py:108` | `xs_verif = traj_verif.x if hasattr(traj_verif, 'x') else traj_verif[0]` — `ExponentialMap.trace()` returns a plain tuple `(xs, vs)`, so `hasattr(traj_verif, 'x')` is always `False`. The attribute-access branch is dead code, masking the fact that the return type is being handled by tuple indexing only. If `trace()` were ever changed to return a NamedTuple with `.x`, the dead-code branch would silently take over. | Remove the dead branch: `xs_verif = traj_verif[0]`, or unpack explicitly: `xs_verif, _ = traj_verif`. |
| 5 | **RISK** | `examples/demo_trajectories.py:105-107` | `path_length` computes chord-length (sum of inter-point Euclidean distances). The three trajectories have very different discretisations (40, 1201, and 601 points), so the chord-length approximation quality differs across them, potentially producing misleading quantitative comparisons. | Add a brief comment noting the approximation, or resample trajectories to a common step count before computing length. |
| 6 | **STYLE** | `examples/demo_trajectories.py:77,91` | The wind function is defined twice with the same body: `w_net = lambda x: 0.8 * wind_flow(x)` and `wind_fn = lambda x: 0.8 * wind_flow(x)`. | Delete `wind_fn` on line 91 and pass `w_net` to `pure_advection` on line 92. |
| 7 | **STYLE** | `examples/demo_trajectories.py:79` | `end = end / jnp.linalg.norm(end)` is redundant — the vector `[0, 0, 1]` is already unit length. | Remove the normalisation or add a comment explaining it is a safety guard for future edits. |
| 8 | **STYLE** | `examples/demo_trajectories.py:5` | `Union` is imported from `typing` but is only used in the `path_length` type hint. `NamedTuple` is also imported and used in `isinstance` (incorrectly — see #1). With the fix for #1, `NamedTuple` is still needed for the `Trajectory` class definition at line 16, but `Union` could be replaced by PEP 604 syntax (`X | Y`) on Python ≥ 3.10 for consistency with modern JAX requirements. | Minor — keep or modernise at author's discretion. |

## Test Coverage Assessment

This is a standalone demo script, not a library module, so there are no dedicated unit tests. The relevant library components exercised by this script (`AVBDSolver.solve`, `ExponentialMap.trace`, `rossby_haurwitz`, `Randers`, `Sphere`) are tested in their respective test files (`test_solver.py`, `test_geodesic.py`, `test_fields.py`, `test_zoo.py`).

| Function | Tested? | Notes |
|----------|---------|-------|
| `pure_advection` | No | Demo-local helper; would benefit from a smoke test verifying output shape and sphere projection. |
| `path_length` | No | Demo-local helper; the `isinstance` bug (#1) would be caught by a test on Python ≥ 3.11. |
| `main` | No | Interactive visualisation; not unit-testable without mocking `plt.show()`. |

## Positive Patterns

- **Clear narrative structure:** The numbered sections (Physics Setup → BVP → Advection → Verification → Comparison → Visualisation) make the demo easy to follow.
- **Good use of `jax.lax.scan`** in `pure_advection` for trace-friendly integration.
- **Manifold projection** (`p_next / norm`) is applied at every advection step, preventing ODE drift off the sphere.
- **Quantitative comparison** printed alongside the plot gives users immediate numerical feedback.
