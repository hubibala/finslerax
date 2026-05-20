# Documentation Review: `examples/demo_discrete_zermelo.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2026-05-15

## Summary

Overall documentation quality: **needs work**.

The script demonstrates an important concept — comparing continuous and discrete Randers (Zermelo) geodesics on $S^2$ — but lacks a module-level docstring, contextual explanation of the mathematics, parameter justification, and audience-appropriate commentary. Both mathematicians and ML engineers would struggle to understand the script's purpose and interpret the results without consulting external sources.

## Issue Tracker

| # | Severity | Location | Issue | Suggested Text |
|---|----------|----------|-------|----------------|
| 1 | **MISSING** | `examples/demo_discrete_zermelo.py:1` | No module-level docstring. The script has no explanation of its purpose, what it demonstrates, what the Zermelo navigation problem is, or what the expected output should be. | Add a docstring such as: `"""Discrete vs Continuous Zermelo Navigation on S².\n\nDemonstrates that the DiscreteRanders metric on a triangulated icosphere\nrecovers the same energy-minimising path as the analytical Randers metric\nunder a rotational wind field.\n\nZermelo's Navigation Problem: Given a Riemannian metric h (the 'sea') and a\nwind field W with ||W||_h < 1, the time-optimal path is a geodesic of the\ninduced Randers metric F(x,v) = (sqrt(λ||v||²_h + <W,v>²_h) - <W,v>_h) / λ\nwhere λ = 1 - ||W||²_h. See spec/MATH_SPEC.md § 5.\n\nUsage:\n    python examples/demo_discrete_zermelo.py\n\nOutput:\n    zermelo_demo.png — three paths (great circle, continuous Randers,\n    discrete Randers) with equatorial wind vectors and indicatrices.\n"""` |
| 2 | **MISSING** | `examples/demo_discrete_zermelo.py:12` | The `radius = 1.0` parameter is not documented. Section headers (`# --- 1. Continuous Physics ---`) are present but contain no explanation of what "continuous physics" means in this context. | Add a comment explaining the section, e.g.: `# --- 1. Continuous Physics ---\n# Define the unit sphere S² and a Randers metric induced by Zermelo navigation.\n# h_net: identity matrix → round sphere metric; w_net: rotational wind around Z-axis.` |
| 3 | **UNCLEAR** | `examples/demo_discrete_zermelo.py:16–17` | The wind function `w_net` returns `0.8 * [-x[1], x[0], 0.0]`, which is a tangent vector in *ambient* $\mathbb{R}^3$. A mathematician would wonder: (a) why this is tangent to $S^2$ at $x$, (b) what 0.8 means physically and whether it satisfies $\|W\|_h < 1$ (the Randers convexity condition from `spec/MATH_SPEC.md § 5`). An ML engineer would wonder what `w_net` and `h_net` represent. | Add inline comments: `# Wind: counter-clockwise rotation about the Z-axis in ambient R³.\n# At any point x on S², the vector [-x[1], x[0], 0] is tangent to the sphere\n# (it is orthogonal to x). Strength 0.8 < 1.0 ensures Randers convexity\n# (||W||_h < 1, see spec/MATH_SPEC.md § 5).` |
| 4 | **UNCLEAR** | `examples/demo_discrete_zermelo.py:19` | `h_net(x) = jnp.eye(3)` is the ambient identity matrix, but it is not explained that this represents the round metric on $S^2$ (the Riemannian "sea" in Zermelo's formulation). | Add: `# h_net: The Riemannian background metric (the "sea" in Zermelo's analogy).\n# Identity in R³ → standard round metric when restricted to S².` |
| 5 | **UNCLEAR** | `examples/demo_discrete_zermelo.py:22` | `metric_riem` is created with zero wind. The name `metric_riem` and the inline intent (Riemannian geodesic = great circle) are not explained. A reader unfamiliar with the code would not understand that setting wind to zero reduces Randers to Riemannian. | Add: `# Zero wind reduces the Randers metric to a Riemannian metric;\n# its geodesics are great circles on S².` |
| 6 | **MISSING** | `examples/demo_discrete_zermelo.py:25–26` | Start and end points are given without explanation of why these were chosen (equator-to-pole traverse, perpendicular to equatorial wind). | Add: `# Mission: Start on the equator (0, 1, 0), end at the north pole (0, 0, 1).\n# This path is transverse to the equatorial wind, making wind-drift visible.` |
| 7 | **UNCLEAR** | `examples/demo_discrete_zermelo.py:28` | `AVBDSolver(step_size=0.05, beta=10.0, iterations=500, tol=1e-6)` — none of the solver hyperparameters are documented. `beta` is the penalty/damping parameter but this is not explained. Readers from both audiences will wonder what controls each value. | Add: `# AVBD solver parameters (see spec/ARCH_SPEC.md § 4.2):\n#   step_size: gradient descent learning rate\n#   beta: boundary-condition penalty weight\n#   iterations: max optimisation steps\n#   tol: convergence tolerance on energy change` |
| 8 | **MISSING** | `examples/demo_discrete_zermelo.py:35–38` | The energy comparison is the main quantitative result of the demo, but there is no comment explaining *what* the comparison proves (discrete mesh approximation recovers similar energy to the analytical solution). | Add: `# Compare Finsler energy of the Randers path and the naive Riemannian path.\n# The Randers geodesic should have *lower* energy because it exploits the wind.` |
| 9 | **UNCLEAR** | `examples/demo_discrete_zermelo.py:42` | `generate_icosphere(radius=1.0, subdivisions=1)` — the subdivision level controls mesh resolution but this is not explained. A reader cannot evaluate whether 1 subdivision is appropriate without knowing the mesh density. | Add: `# subdivisions=1 → 42 vertices, 80 faces (a coarse mesh; increase for finer approximation).` |
| 10 | **MISSING** | `examples/demo_discrete_zermelo.py:45–46` | The wind-sampling step (`face_centers`, `face_winds`) is critical — it explains *how* the continuous wind field is discretised onto mesh faces — but has no mathematical explanation. | Add: `# Sample the continuous wind field at face barycentres to build\n# per-face wind vectors for the discrete Randers metric.` |
| 11 | **MISSING** | `examples/demo_discrete_zermelo.py:48` | `DiscreteRanders` is used without explaining what it is or how it differs from `Randers`. Per `spec/ARCH_SPEC.md § 3`, it computes the metric via differentiable face weights rather than the analytical Zermelo formula. | Add: `# DiscreteRanders: mesh-based Finsler metric where each face carries\n# its own wind vector; energy is accumulated per-face along the path.` |
| 12 | **UNCLEAR** | `examples/demo_discrete_zermelo.py:56–57` | The title string contains the main claim ("Discrete Matches Continuous") but the script never explicitly checks or asserts convergence. A reader cannot tell what energy difference is "good enough." | Add a comment such as: `# We expect the discrete energy to converge toward the continuous energy\n# as mesh resolution increases (subdivisions → ∞).` |
| 13 | **MISSING** | `examples/demo_discrete_zermelo.py:1–82` | No references to literature. Zermelo's navigation problem (Bao, Robles, Shen, 2004; arXiv: math/0311233) is a well-known result that should be cited for both audiences. | Add to the module docstring: `References:\n    Bao, Robles, Shen — "Zermelo navigation on Riemannian manifolds"\n    (J. Differential Geometry 66, 2004; arXiv:math/0311233).` |
| 14 | **TYPO** | `examples/demo_discrete_zermelo.py:57` | The title string uses `\\n` inside an f-string. While Python will interpret this correctly, the doubled backslash is unconventional and may confuse readers scanning for string literals. Standard practice is a raw newline or explicit `\n`. | Verify intent; consider using `\n` (single backslash) or splitting the title across two `ax.set_title` lines. |

## Coverage Matrix

| Public Symbol / Section | Has Docstring / Comment | Args Documented | Math Concept Explained | Audience Clarity | Example Output Described |
|-------------------------|------------------------|-----------------|----------------------|-----------------|-------------------------|
| Module-level | No | N/A | No | Neither audience | No |
| `w_net(x)` | Minimal (`# Wind: Rotational around Z`) | No | Partially (no convexity note) | Unclear to both | N/A |
| `h_net(x)` | None | No | No | Unclear to both | N/A |
| `metric_randers` | None | N/A | No | Unclear | N/A |
| `metric_riem` | None | N/A | No | Unclear | N/A |
| `AVBDSolver(...)` | None | No | No | Unclear to both | N/A |
| Section 2 (Mission) | Minimal (`# --- 2. Mission ---`) | Partially | No | Minimal | N/A |
| Section 3 (Discrete) | Minimal section header | No | No | Unclear to both | N/A |
| Section 4 (Visualization) | None | N/A | N/A | Minimal | No |

## Spec Alignment Notes

| Spec Reference | Demo Status | Note |
|----------------|-------------|------|
| `spec/MATH_SPEC.md § 5` (Zermelo Parameterization) | Not cited | The demo implements Zermelo navigation but never references or explains the formula $F(x,v) = (\sqrt{\lambda\|v\|_h^2 + \langle W,v\rangle_h^2} - \langle W,v\rangle_h)/\lambda$. |
| `spec/MATH_SPEC.md § 5` (Convexity constraint $\|W\|_h < 1$) | Implicit only | Wind strength 0.8 satisfies the constraint but is not annotated as doing so. |
| `spec/ARCH_SPEC.md § 3.1` (Randers Specialization) | Not cited | `Randers` and `DiscreteRanders` are used but their spec definitions are not referenced. |
| `spec/ARCH_SPEC.md § 4.2` (AVBDSolver) | Not cited | Solver parameters are chosen without referencing the architecture spec. |
| `spec/ARCH_SPEC.md § 3` (DiscreteRanders) | Not cited | The distinction between `Randers` (analytical) and `DiscreteRanders` (face-weight-based) is never explained. |
