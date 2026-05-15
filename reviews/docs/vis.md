# Documentation Review: `ham.vis.vis`

**Reviewer:** Doc Reviewer Agent
**Date:** 2026-05-15
**File:** `src/ham/vis/vis.py`

## Summary

Overall documentation quality: **needs work**.

Five of the six public functions exported from this module have **no docstring at all**. The one function that does have a docstring (`plot_trajectory`) documents neither its arguments nor its return value. The module also lacks a module-level docstring, provides no mathematical context for domain-specific concepts (e.g. Finsler indicatrices), and contains a development-artifact comment (`--- FIXED FUNCTION ---`) that should be removed or replaced with a proper changelog note.

---

## Issue Tracker

| # | Severity | Symbol / Section | Issue | Suggested Text |
|---|----------|-----------------|-------|----------------|
| 1 | **MISSING** | `vis.py` (module) | No module-level docstring. Users browsing the API have no orientation for what this module provides. | `"""3-D visualization utilities for Finsler manifolds on the sphere. Provides wireframe surfaces, vector-field quivers, geodesic trajectory plots, and Finsler indicatrix rendering."""` |
| 2 | **MISSING** | `setup_3d_plot` (`src/ham/vis/vis.py:6`) | Entirely undocumented public function. No description of purpose, args, or returns. | `"""Create a matplotlib 3-D figure with transparent panes.\n\nArgs:\n    elev: Elevation angle for the 3-D view (degrees). If None, matplotlib default is used.\n    azim: Azimuth angle for the 3-D view (degrees). If None, matplotlib default is used.\n\nReturns:\n    Tuple of (fig, ax) — the matplotlib Figure and 3-D Axes.\n"""` |
| 3 | **MISSING** | `plot_sphere` (`src/ham/vis/vis.py:20`) | Entirely undocumented public function. | `"""Draw a wireframe sphere on a 3-D axis.\n\nArgs:\n    ax: A matplotlib 3-D Axes instance.\n    radius: Sphere radius. Default 1.0.\n    alpha: Wireframe transparency. Default 0.1.\n    color: Wireframe color. Default 'gray'.\n"""` |
| 4 | **MISSING** | `plot_vector_field` (`src/ham/vis/vis.py:27`) | Entirely undocumented public function. No description of what `points` and `vectors` shapes are expected or what `scale` means. | `"""Plot a 3-D quiver field.\n\nArgs:\n    ax: A matplotlib 3-D Axes instance.\n    points: (N, 3) array of base-point positions.\n    vectors: (N, 3) array of vector components at each base point.\n    scale: Arrow length scaling factor. Default 0.2.\n    color: Arrow color. Default 'blue'.\n    alpha: Arrow transparency. Default 0.5.\n    label: Legend label.\n    **kwargs: Forwarded to ax.quiver.\n"""` |
| 5 | **UNCLEAR** | `plot_trajectory` (`src/ham/vis/vis.py:34`) | Docstring exists but is minimal: says "Plots a 3D trajectory" and "Now accepts **kwargs" (a changelog-style note, not a description). Args (`ax`, `traj`, `color`, `label`, `**kwargs`) are not documented. The three accepted input types for `traj` (object with `.xs`, tuple, raw array) are not described. | `"""Plot a 3-D geodesic trajectory.\n\nAccepts several trajectory representations: a SolverResult (with .xs attribute), a tuple whose first element is the position array, or a raw (T, 3) NumPy/JAX array.\n\nArgs:\n    ax: A matplotlib 3-D Axes instance.\n    traj: Trajectory data — SolverResult, tuple, or (T, 3) array.\n    color: Line and start-marker color. Default 'red'.\n    label: Legend label.\n    **kwargs: Forwarded to ax.plot (e.g. linewidth, linestyle, alpha).\n"""` |
| 6 | **MISSING** | `plot_indicatrices` (`src/ham/vis/vis.py:56`) | Entirely undocumented public function. This is the most domain-specific function in the module — it computes the Finsler indicatrix $\{v \in T_x M : F(x,v)=1\}$ and plots it. Both audiences need guidance: mathematicians need to know which definition of indicatrix is used; ML engineers need to know the expected `metric` interface. | `"""Plot Finsler indicatrices (unit circles of the metric) at given surface points.\n\nFor each point p on the sphere, computes the set of tangent directions v such that F(p, v) = 1 (the Finsler indicatrix, see spec/MATH_SPEC.md § 1.1), then scales and renders them on the surface.\n\nArgs:\n    ax: A matplotlib 3-D Axes instance.\n    metric: A FinslerMetric (or subclass) providing metric_fn(x, v).\n    points: (N, 3) array of points on the manifold.\n    scale: Visual scaling factor for the indicatrix loops. Default 0.15.\n    n_theta: Number of angular samples around each indicatrix. Default 40.\n    color: Loop color. Default 'purple'.\n    alpha: Loop transparency. Default 0.8.\n    **kwargs: Forwarded to ax.plot.\n"""` |
| 7 | **MISSING** | `generate_icosphere` (`src/ham/vis/vis.py:77`) | Entirely undocumented public function. Returns JAX arrays but callers may expect NumPy. Shape semantics of the returned faces array are not documented. | `"""Generate an icosphere by recursive subdivision.\n\nStarts from a regular icosahedron and subdivides each face the requested number of times, then projects all vertices onto the sphere.\n\nArgs:\n    radius: Sphere radius. Default 1.0.\n    subdivisions: Number of recursive subdivision passes. Default 3.\n\nReturns:\n    Tuple of (vertices, faces) as JAX arrays:\n        vertices: (V, 3) coordinates on the sphere.\n        faces: (F, 3) integer indices into the vertex array.\n"""` |
| 8 | **INACCURATE** | `plot_trajectory` (`src/ham/vis/vis.py:37`) | Docstring contains the line `"Now accepts **kwargs for alpha, linewidth, linestyle, etc."` — this reads as a changelog entry, not API documentation. It implies a prior version existed that did not accept kwargs, which is irrelevant to users. | Remove the changelog line. Replace with proper parameter documentation (see #5). |
| 9 | **UNCLEAR** | `plot_trajectory` (`src/ham/vis/vis.py:34-52`) | The `# --- FIXED FUNCTION ---` / `# ----------------------` banner comments are development artifacts that may confuse readers into thinking the surrounding code is provisional or unstable. | **Recommended Action:** Remove the `--- FIXED FUNCTION ---` and `----------------------` comment banners. |
| 10 | **UNCLEAR** | `plot_indicatrices` (`src/ham/vis/vis.py:56`) | The function assumes all points lie on a sphere (it computes the surface normal as `p / ‖p‖`). This constraint is nowhere stated. An ML engineer passing points on a torus or embedded surface would get silently wrong indicatrices. | Add a note to the docstring: *"Points are assumed to lie on a sphere centered at the origin; the surface normal is computed as p / ‖p‖."* |
| 11 | **MISSING** | `vis.py` (module) | No type annotations on any function signature (except `hyperbolic.py` which uses them consistently). This hampers IDE support and auto-generated documentation. | **Recommended Action:** Add type hints for at least `ax: Axes3D`, array args as `np.ndarray`, and return types. |

---

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:------------:|:---------------:|:-----------------:|:-------------:|:-------:|
| `setup_3d_plot` | No | No | No | N/A | No (but used in `examples/demo_zermelo.py`) |
| `plot_sphere` | No | No | No | N/A | No (but used in `examples/demo_zermelo.py`) |
| `plot_vector_field` | No | No | No | N/A | No (but used in `examples/demo_learned_wind.py`) |
| `plot_trajectory` | Partial | No | No | No | No (but used in `examples/demo_zermelo.py`) |
| `plot_indicatrices` | No | No | No | No | No (but used in `examples/demo_zermelo.py`) |
| `generate_icosphere` | No | No | No | N/A | No (but used in `examples/demo_zermelo.py`) |

---

## Spec Alignment Notes

1. **`vis.py` not listed in `spec/ARCH_SPEC.md` § 5 (directory tree).** The spec lists `vis/hyperbolic.py` (`spec/ARCH_SPEC.md:186`) but does not list `vis/vis.py`. Since `vis.py` is the primary visualization module and is re-exported from `ham.vis.__init__`, it should be included in the spec's directory tree.

2. **Indicatrix definition.** `plot_indicatrices` computes $\{v/F(x,v) : v \text{ sampled in } T_xM\}$, which is the *unit indicatrix* per `spec/MATH_SPEC.md § 1.1`. The docstring (once written) should reference this definition and link to the spec.

3. **`metric_fn` interface.** `plot_indicatrices` calls `metric.metric_fn(p_jax, u)`, matching the `FinslerMetric.metric_fn(x, v)` signature from `spec/ARCH_SPEC.md § 2.2`. The docstring should state this dependency explicitly so users know what object to pass.
