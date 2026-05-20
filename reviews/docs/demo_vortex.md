# Documentation Review: `examples/demo_vortex.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2026-05-15

## Summary

Overall documentation quality: **needs work**.

The script demonstrates an important concept — the effect of the AVBD solver's `beta` (stiffness) parameter on geodesic paths through a strong Randers wind field on the sphere — but lacks a module-level docstring entirely and provides only sparse inline comments. A mathematician would not find the connection to the Zermelo navigation problem or the Randers metric definition ($F = \sqrt{v^T M v} + \beta \cdot v$, `spec/MATH_SPEC.md § 5`). An ML engineer would not understand what `beta` controls in the solver without consulting `AVBDSolver` internals. Several parameter choices (`strength=10.0`, `decay=5.0`, `n_steps=40`) are unexplained.

## Issue Tracker

| # | Severity | Location | Issue | Suggested Text |
|---|----------|----------|-------|----------------|
| 1 | **MISSING** | `examples/demo_vortex.py:1` | No module-level docstring. The script's purpose, audience, and what it demonstrates are unstated. | Add: `"""Solver stiffness comparison on a spherical Randers manifold.\n\nDemonstrates how the AVBDSolver penalty parameter β (beta) controls\npath curvature through a strong Gaussian vortex wind field on S².\nA high beta freezes the path quickly (near-geodesic for Euclidean),\nwhile a low beta allows the path to curve with the Randers wind.\n\nMathematical context:\n  - Manifold: S² (unit sphere, ambient R³)\n  - Metric: Randers via Zermelo navigation (spec/MATH_SPEC.md § 5)\n  - Solver: AVBD boundary-value relaxation (spec/ARCH_SPEC.md § 4.2)\n\nUsage:\n    python examples/demo_vortex.py\n"""` |
| 2 | **MISSING** | `examples/demo_vortex.py:13-20` | The `vortex_field` function has no docstring. Its arguments, return value, and the mathematical formula it implements are undocumented. | Add a docstring: `"""Gaussian vortex wind field on S².\n\nCreates a tangential rotational flow centred at 'center' on the sphere.\nThe magnitude decays as strength * exp(-decay * d²), where d is the\ngeodesic distance from 'center'.\n\nArgs:\n    center: (3,) array, point on S² around which the vortex is centred.\n    strength: peak angular speed of the vortex (default 1.0).\n    decay: Gaussian width parameter; larger = more localised (default 2.0).\n\nReturns:\n    Callable mapping (3,) points on S² to (3,) ambient tangent vectors.\n"""` |
| 3 | **UNCLEAR** | `examples/demo_vortex.py:15` | `cos_dist = jnp.dot(x, center)` is actually the cosine of the geodesic distance, not the distance itself. No comment clarifies that `dist = arccos(cos_dist)` is the geodesic distance on $S^2$. | Add inline comment: `# Geodesic distance d(x, c) = arccos(<x, c>) on the unit sphere` |
| 4 | **UNCLEAR** | `examples/demo_vortex.py:17` | `v_rot = jnp.cross(center, x)` produces a vector tangent to S² (perpendicular to both `center` and `x`), giving the rotational direction. This is not obvious without a comment. | Add: `# Tangential rotation axis: cross(c, x) lies in T_x S² and is perpendicular to the radial direction` |
| 5 | **MISSING** | `examples/demo_vortex.py:26-27` | The choice of `strength=10.0` and `decay=5.0` is unexplained. These are deliberately extreme values ("Very strong wind") but the reasoning (force path curvature to make the stiffness difference visible) deserves a comment. | Expand inline comment: `# Very strong, localised vortex to exaggerate stiffness effects.\n# strength=10 produces wind magnitude ≈ 10 at vortex centre;\n# decay=5 concentrates the field within ~0.5 rad of the centre.` |
| 6 | **MISSING** | `examples/demo_vortex.py:28` | `h_net = lambda x: jnp.eye(3)` is the Riemannian background metric but has no comment linking it to the "sea" in Zermelo navigation (`spec/MATH_SPEC.md § 5`). | Add: `# Flat Riemannian background ("sea" in Zermelo navigation): h_ij = δ_ij` |
| 7 | **UNCLEAR** | `examples/demo_vortex.py:35-36` | `beta=20.0` in `AVBDSolver` is the constraint penalty stiffness, **not** the Randers 1-form $\beta$ from `spec/MATH_SPEC.md § 5`. The two uses of "beta" in the same script are confusing for mathematicians. | Add a clarifying comment: `# NOTE: 'beta' here is the AVBD penalty stiffness (spec/ARCH_SPEC.md § 4.2),\n# NOT the Randers 1-form β from the Zermelo parameterisation.` |
| 8 | **MISSING** | `examples/demo_vortex.py:35` | `iterations=100` and `step_size=0.05` are undocumented. It would help the reader to know these are standard defaults from `AVBDSolver`. | Add: `# iterations & step_size are AVBD defaults (see AVBDSolver docstring)` |
| 9 | **MISSING** | `examples/demo_vortex.py:31-32` | The endpoint `[-0.99, 0.1, 0.0]` is nearly antipodal to the start but slightly off-axis. No comment explains why (to avoid the exact antipodal singularity). | Add: `# Near-antipodal target (offset from [-1,0,0] to avoid the geodesic-ambiguity singularity)` |
| 10 | **MISSING** | `examples/demo_vortex.py:37` | `n_steps=40` (path discretisation) is undocumented. | Add: `# 40 interior path vertices for the discrete geodesic` |
| 11 | **UNCLEAR** | `examples/demo_vortex.py:23` | The `main()` print banner says "HAM Stiffness Fix Demo" which suggests this was written to verify a bug fix, not as a standalone demo. The title is misleading for a user browsing examples. | Change to: `print("--- HAM Solver Stiffness Comparison Demo ---")` |
| 12 | **UNCLEAR** | `examples/demo_vortex.py:55` | The plot title mentions a "'D' shape" with no prior explanation of what that means geometrically (the relaxed solver's path bulges into a D because wind pushes it laterally). | Add a comment above `plt.show()`: `# The relaxed path curves into a "D" shape because the low penalty\n# allows the Randers wind to deflect interior vertices laterally.` |
| 13 | **MISSING** | `examples/demo_vortex.py` (top-level) | No `requirements` or dependency comment (JAX with x64, matplotlib). Other demo scripts in the repo also lack this, but it would improve self-containedness. | Add near imports: `# Requires: jax (with x64), matplotlib, numpy` |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:---:|:---:|:---:|:---:|:---:|
| `vortex_field()` | No | No | No | No | Used inline |
| `main()` | No | N/A | N/A | N/A | N/A |

## Spec Alignment Notes

1. **`beta` overloading.** The AVBD solver parameter `beta` (`spec/ARCH_SPEC.md § 4.2`) and the Randers 1-form $\beta$ (`spec/MATH_SPEC.md § 5`) share the same symbol. The script uses the solver `beta` but never disambiguates, which will confuse readers cross-referencing the math spec. See Issue #7.

2. **Randers construction.** The script constructs a `Randers` metric with `h_net = lambda x: jnp.eye(3)` (flat Euclidean background) and a custom `w_net` (the vortex field). This matches the Zermelo parameterisation in `spec/MATH_SPEC.md § 5`, but the script never references the Zermelo framework or the convexity constraint $\|W\|_h < 1$. Given `strength=10.0`, the vortex magnitude at the centre is ~10, which **violates** $\|W\|_h < 1$. The `Randers` class internally enforces this via `tanh` gating (`src/ham/geometry/zoo.py`), but the script does not mention it, which could mislead a mathematician into thinking the metric is invalid.

3. **Solver documentation.** `AVBDSolver` is described in `spec/ARCH_SPEC.md § 4.2` as using penalty stiffness for equality constraints. The script's comparison of `beta=20.0` vs `beta=0.5` is a useful illustration but never links back to the spec.
