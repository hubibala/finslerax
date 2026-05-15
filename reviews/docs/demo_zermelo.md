# Documentation Review: `examples/demo_zermelo.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2026-05-15

## Summary

Overall documentation quality: **needs work**.

The script demonstrates the Zermelo navigation problem on $S^2$ using a Randers metric with a rotational wind field, comparing it against a Riemannian baseline and a discrete mesh path. The demo is structurally organized into numbered sections (Physics, Mission, Solve, Visualization), which is good. However, it lacks a module-level docstring, contains a factual error in a comment (wind strength, start-point description), and provides no mathematical context that would orient either audience. Non-trivial parameter choices (`beta=10.0`, `iterations=200`, wind magnitude `0.9`) are undocumented.

## Issue Tracker

| # | Severity | Location | Issue | Suggested Text |
|---|----------|----------|-------|----------------|
| 1 | **MISSING** | `examples/demo_zermelo.py:1` | No module-level docstring. Neither audience can understand the script's purpose, the problem it demonstrates, or what output to expect without reading the entire file. | Add: `"""Zermelo Navigation on S²: compares a wind-optimal Randers geodesic against a Riemannian great circle and a discrete mesh path on a unit sphere. A rotational wind field (0.9× equatorial counter-clockwise) biases the Randers geodesic away from the great circle. Output: a 3-D Matplotlib figure with all three paths and wind vectors."""` |
| 2 | **INACCURATE** | `examples/demo_zermelo.py:17` | Comment says "Strength 0.8 at equator" but the code on line 19 multiplies by `0.9`. | Change to `# Strength 0.9 at equator. Counter-Clockwise.` |
| 3 | **INACCURATE** | `examples/demo_zermelo.py:25` | Comment says "South -> North" but `start = [1, 0, 0]` is on the equator (the south pole of a unit sphere at $z=-1$ is `[0, 0, -1]`). The mission is actually Equator → North Pole. | Change to `# --- 2. Mission: Equator (x-axis) -> North Pole ---` |
| 4 | **MISSING** | `examples/demo_zermelo.py:12-13` | No inline comment explaining that `radius = 1.0` is the sphere radius, or why the unit sphere is chosen (simplest case for verifying against known great-circle geodesics). | Add: `# Unit sphere — analytic great circles available for Riemannian baseline.` |
| 5 | **UNCLEAR** | `examples/demo_zermelo.py:18-19` | The wind function `w_net` returns an ambient 3-D vector but there is no comment explaining it will be projected to the tangent plane by the `Randers` metric internals. An ML engineer unfamiliar with embedded geometry would wonder why a 3-D vector is used for a 2-D manifold. | Add comment: `# Wind is defined in ambient R³; the Randers metric projects it to T_x S².` |
| 6 | **UNCLEAR** | `examples/demo_zermelo.py:21` | `h_net` returns `jnp.eye(3)` (the ambient identity) but there is no comment explaining this represents the canonical "flat sea" Riemannian metric (i.e., the round metric on $S^2$ induced from the ambient Euclidean metric). A mathematician would want to know this is $h_{ij} = \delta_{ij}$. | Add comment: `# Flat sea: ambient identity induces the round metric on S².` |
| 7 | **MISSING** | `examples/demo_zermelo.py:30` | Solver hyper-parameters `step_size=0.05`, `beta=10.0`, `iterations=200`, `tol=1e-6` are not explained. `beta` is the AVBD penalty stiffness (see `spec/ARCH_SPEC.md` § 4.2) and `200` iterations is 10× the default; the reader has no idea why these values were chosen. | Add comment: `# beta: constraint penalty stiffness; 200 iters for tight convergence on the curved manifold.` |
| 8 | **MISSING** | `examples/demo_zermelo.py:23` | The zero-wind Riemannian baseline `metric_riem` is created by passing a lambda that returns zeros, but there is no comment explaining the intent — that this recovers the purely Riemannian (great-circle) solution for comparison. | Add comment: `# Zero wind → pure Riemannian metric (great-circle baseline).` |
| 9 | **UNCLEAR** | `examples/demo_zermelo.py:36-37` | `batch_energy = jax.vmap(metric_randers.energy)` always evaluates both paths with the Randers energy. For the Riemannian baseline this is misleading — the Randers energy of a path solved with zero wind is not the same as the Riemannian energy of that path. The comment "(Fixes dimensions error)" explains an implementation detail, not the mathematical intent. | Clarify what is being measured: `# Evaluate both paths under the RANDERS cost to compare how wind-optimal the trajectories are.` |
| 10 | **MISSING** | `examples/demo_zermelo.py:46-48` | The discrete mesh section creates an `Euclidean` metric on the mesh but does not explain that this is an isotropic (no-wind) mesh baseline, nor why an icosphere with `subdivisions=3` (~1280 faces) was chosen. | Add: `# Isotropic mesh baseline (no wind) on a high-res icosphere (~1280 faces) to test mesh solver.` |
| 11 | **MISSING** | `examples/demo_zermelo.py` (global) | No reference to `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization) or any literature reference for Zermelo's navigation problem. Both audiences would benefit from a pointer. | Add to module docstring: `See spec/MATH_SPEC.md § 5 for the Zermelo navigation formula.` |
| 12 | **UNCLEAR** | `examples/demo_zermelo.py:27` | Comment says "The solver will now perturb this slightly to break the symmetry" but the code shows no perturbation; the statement describes internal solver behavior that is not visible here. This is confusing for anyone reading the demo. | Either remove the comment or clarify: `# Note: the AVBD solver may add a small random perturbation internally to break symmetry.` |
| 13 | **MISSING** | `examples/demo_zermelo.py:62-63` | `plot_indicatrices` is called with `scale=0.15` and sampled every 6th point (`traj_rand.xs[::6]`). No comment explains what indicatrices are (Finsler unit balls in the tangent plane) or why they are plotted. | Add: `# Indicatrices: unit "balls" of the Randers norm in each tangent plane — elongation shows wind effect.` |

## Coverage Matrix

| Public Symbol / Section | Has Comment | Purpose Documented | Params Documented | Math Notation | Spec Reference |
|------------------------|-------------|-------------------|-------------------|---------------|----------------|
| Module-level docstring | No | No | — | — | — |
| `w_net` | Partial | Partial ("rotation around Z-axis") | Wind magnitude wrong (0.8 vs 0.9) | No | No |
| `h_net` | No | No | No | No | No |
| `metric_randers` | No | No | — | No | No |
| `metric_riem` | No | No | — | No | No |
| `start` / `end` | Partial | Inaccurate ("South") | No | No | No |
| `AVBDSolver(...)` | No | No | No | No | No |
| Energy comparison | Partial | Misleading | — | No | No |
| Mesh section | Partial | Partial | Subdivisions noted | No | No |
| Indicatrices | No | No | No | No | No |

## Spec Alignment Notes

1. **`spec/MATH_SPEC.md` § 5 — Zermelo Parameterization:** The demo demonstrates the Zermelo navigation problem but never names or cites the spec section. The formula $F(x,v) = \frac{\sqrt{\lambda \|v\|_h^2 + \langle W,v\rangle_h^2} - \langle W,v\rangle_h}{\lambda}$ is implemented inside `Randers` but the demo does not orient the reader to this.
2. **`spec/ARCH_SPEC.md` § 3.1 — Randers Specialization:** The spec documents `h_net` as outputting positive-definite matrices and `w_net` as outputting the wind vector, with the constraint $\|W\|_h < 1$. The demo's wind strength (`0.9`) is close to the causality boundary but valid. This proximity should be noted as intentional (demonstrating strong wind effects) or warned about (numerical sensitivity).
3. **`spec/ARCH_SPEC.md` § 4.2 — AVBDSolver:** The spec documents the solver as "optimizes discrete path points directly" with a loss combining energy and constraint penalties. The demo uses non-default `iterations=200` (default is `20`) and `tol=1e-6` (default is `1e-4`) without explaining why.
