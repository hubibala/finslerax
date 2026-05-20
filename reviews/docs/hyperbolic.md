# Documentation Review: `vis/hyperbolic.py`

**Reviewer:** Doc Reviewer Agent
**Date:** 2026-05-15
**Source:** `src/ham/vis/hyperbolic.py`

## Summary

Overall documentation quality: **needs work**.

All three public functions have docstrings, but they lack formal `Args`/`Returns`/`Raises` sections, omit mathematical context that would serve the differential-geometry audience, and leave two of six parameters in `plot_poincare_disk` entirely undocumented. The projection functions embed their formulas only as inline code comments rather than in the docstrings, making them invisible to `help()` consumers and documentation generators.

---

## Issue Tracker

| # | Severity | Symbol / Section | Issue | Suggested Text |
|---|----------|-----------------|-------|----------------|
| 1 | **MISSING** | `project_to_poincare` (`src/ham/vis/hyperbolic.py:6`) | No `Args` section. Parameter `x` shape and semantics are described only in a terse inline annotation, not as a structured docstring field. | Add: `Args:\n    x: jnp.ndarray, shape (..., D+1). Points on the upper-sheet hyperboloid satisfying the Minkowski constraint $-x_0^2 + \sum_{i=1}^D x_i^2 = -1$.` |
| 2 | **MISSING** | `project_to_poincare` (`src/ham/vis/hyperbolic.py:6`) | No `Returns` section. Return shape `(..., D)` and the constraint that the result lies inside the open unit ball are only in the one-line body comment. | Add: `Returns:\n    jnp.ndarray, shape (..., D). Poincaré ball coordinates $y = x_{\mathrm{spatial}} / (1 + x_0)$, with $\|y\| < 1$.` |
| 3 | **UNCLEAR** | `project_to_poincare` (`src/ham/vis/hyperbolic.py:6`) | The Minkowski constraint is written as `-x0^2 + x1^2 + ... = -1` (code-style). Mathematicians expect LaTeX notation; ML engineers need an explicit shape description. Neither audience is well served. | Use LaTeX: `$-x_0^2 + \sum_{i=1}^{D} x_i^2 = -1$` and state this is the Hyperboloid model of hyperbolic space (cf. `spec/MATH_SPEC.md § 4`). |
| 4 | **UNCLEAR** | `project_to_poincare` (`src/ham/vis/hyperbolic.py:6`) | No mathematical description of the projection map itself in the docstring. The formula $y = x_{\mathrm{spatial}} / (1 + x_0)$ is only a code comment on line 14. | Add a one-line formula in the docstring: `Maps via stereographic projection: $y_i = x_i / (1 + x_0)$ for $i = 1, \ldots, D$.` |
| 5 | **MISSING** | `project_vector_to_poincare` (`src/ham/vis/hyperbolic.py:17`) | No `Args` section. Parameters `x` (base point) and `v` (tangent vector) are completely undocumented—shapes, constraints, and relationship are absent. | Add: `Args:\n    x: jnp.ndarray, shape (..., D+1). Base point on the hyperboloid.\n    v: jnp.ndarray, shape (..., D+1). Tangent vector at x in ambient Minkowski space.` |
| 6 | **MISSING** | `project_vector_to_poincare` (`src/ham/vis/hyperbolic.py:17`) | No `Returns` section. | Add: `Returns:\n    jnp.ndarray, shape (..., D). Pushed-forward tangent vector in the Poincaré ball model.` |
| 7 | **UNCLEAR** | `project_vector_to_poincare` (`src/ham/vis/hyperbolic.py:17`) | The differential (pushforward) formula is buried in the inline comment on lines 27-29 and not in the docstring. A geometer looking at `help()` gets only "Pushforward of the projection map." | Add to docstring: `Differential of the stereographic projection:\n$$w_i = \frac{v_i (1 + x_0) - x_i v_0}{(1 + x_0)^2}, \quad i = 1, \ldots, D.$$` |
| 8 | **MISSING** | `plot_poincare_disk` (`src/ham/vis/hyperbolic.py:34`) | Parameters `title` and `ax` are present in the signature but absent from the `Args` docstring block. | Add: `title: str. Plot title. Defaults to "Hyperbolic Embedding (Poincaré Disk)".\nax: matplotlib.axes.Axes or None. If None, a new (8, 8) figure is created.` |
| 9 | **MISSING** | `plot_poincare_disk` (`src/ham/vis/hyperbolic.py:34`) | No `Returns` section. The function returns `ax` but this is undocumented. | Add: `Returns:\n    matplotlib.axes.Axes. The axes object with the plotted disk.` |
| 10 | **UNCLEAR** | `plot_poincare_disk` (`src/ham/vis/hyperbolic.py:34`) | The `colors` parameter type is described only as "Array for coloring points." No type annotation or shape specification. The type should be `np.ndarray` or `Sequence` of shape `(N,)`. | Rewrite as: `colors: np.ndarray of shape (N,), optional. Scalar values for coloring each point (e.g., pseudotime or cell-type index). Mapped via 'viridis' colormap.` |
| 11 | **UNCLEAR** | `plot_poincare_disk` (`src/ham/vis/hyperbolic.py:34`) | The `vectors` docstring says "(e.g. the Wind)" but does not explain that these are tangent vectors in the ambient Hyperboloid model that will be pushed forward to the disk. An ML engineer may pass Poincaré-model vectors and get wrong arrows. | Clarify: `vectors: np.ndarray of shape (N, 3), optional. Tangent vectors in the ambient Hyperboloid (Minkowski) coordinates. Internally pushed forward to the Poincaré ball via project_vector_to_poincare.` |
| 12 | **MISSING** | `plot_poincare_disk` (`src/ham/vis/hyperbolic.py:34`) | No docstring example or pointer to the tutorial example `examples/demo_weinreb_vis.py`. | Add: `Example:\n    See examples/demo_weinreb_vis.py for a full usage with a synthetic differentiation tree.` |
| 13 | **UNCLEAR** | `plot_poincare_disk` (`src/ham/vis/hyperbolic.py:34`) | No mention that lineage edges are drawn as straight lines (chords) rather than true hyperbolic geodesic arcs. The inline comment on line 62 notes this, but the user-facing docstring should too. | Add note: `Note: Lineage edges are drawn as Euclidean straight lines, not geodesic arcs on the Poincaré disk.` |
| 14 | **TYPO** | `plot_poincare_disk` (`src/ham/vis/hyperbolic.py:34`) | The docstring says `points: (N, 3)` hard-coding ambient dimension 3. This is inconsistent with `project_to_poincare` which handles generic `(N, D+1)`. | Change to: `points: (N, D+1) Hyperboloid coordinates. For 2-D visualization, D+1 = 3.` |
| 15 | **MISSING** | Module level (`src/ham/vis/hyperbolic.py:1`) | No module-level docstring. The module's purpose, its relationship to the Hyperboloid surface (`spec/MATH_SPEC.md § 4`), and the two-model convention (Hyperboloid ↔ Poincaré) are not documented at the top of the file. | Add: `"""Poincaré disk visualization utilities.\n\nProjects points and tangent vectors from the Hyperboloid (Minkowski) model of\nhyperbolic space to the Poincaré ball model for 2-D plotting. See\nspec/MATH_SPEC.md § 4 for the Hyperboloid formulation.\n"""` |

---

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|--------------|-----------------|-------------------|---------------|---------|
| `project_to_poincare` | Yes (terse) | No | No | Informal code-style only | No |
| `project_vector_to_poincare` | Yes (one line) | No | No | None (formula in code comment only) | No |
| `plot_poincare_disk` | Yes (partial) | Partial (4/6 params) | No | None | No |

---

## Spec Alignment Notes

1. **`spec/ARCH_SPEC.md § 5`** lists `vis/hyperbolic.py` as "Poincaré disk visualization." The module fulfils this role, but only `plot_poincare_disk` is re-exported via `ham.vis.__init__`. The two projection helpers (`project_to_poincare`, `project_vector_to_poincare`) are not in `__all__` but have no underscore prefix, creating ambiguity about their public status. If they are intended as public API, they should be added to `__all__`; if internal, they should be prefixed with `_`.

2. **`spec/MATH_SPEC.md § 4`** defines the Hyperboloid as the "upper sheet in Minkowski space" with the constraint $-x_0^2 + \sum x_i^2 = -1$. The docstring for `project_to_poincare` uses the same constraint but in informal code-style notation (`-x0^2 + x1^2 + ... = -1`). The notation should be unified with the spec's LaTeX form.

3. **`spec/MATH_SPEC.md § 4`** notes that exact geometric maps for Hyperboloid "cause severe numerical instability" in VAE training loops. The visualization module does not warn users about this, though it is arguably out of scope for a plotting utility.

4. The `plot_poincare_disk` signature hard-codes 3-column input (`(N, 3)`), which implicitly restricts to 2-D hyperbolic space. The spec and `project_to_poincare` both support arbitrary `D`; the docstring should acknowledge the 2-D restriction of the plotting function.
