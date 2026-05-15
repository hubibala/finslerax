# Documentation Review: `src/ham/geometry/mesh.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2026-05-15

## Summary

Overall documentation quality: **needs work**.

`mesh.py` defines the `TriangularMesh` class — a discrete manifold built from triangulated surfaces in $\mathbb{R}^N$. It is a public API exported via `ham.geometry.__init__` and validated in `tests/test_mesh.py`. Despite being structurally complete and tested, its documentation is sparse: the class docstring is a single line, `__init__` lacks a docstring entirely, most overridden abstract methods carry no docstrings, and the key geometric helper `_point_triangle_distance` has only a parenthetical stub. No mathematical context connects the barycentric projection or area-weighted sampling to standard discrete-geometry references. Both target audiences (differential geometers and ML engineers) are underserved.

---

## Issue Tracker

| # | Severity | Symbol / Section | Location | Issue | Suggested Text |
|---|----------|-----------------|----------|-------|----------------|
| 1 | **MISSING** | `TriangularMesh` (class) | `src/ham/geometry/mesh.py:8` | Class docstring is a single line with no description of usage, constructor args, mathematical context, or relation to the `Manifold` ABC. `spec/ARCH_SPEC.md § 5` lists it as "Triangular mesh manifold" and § 6 marks it validated — the docstring should reflect this. | See §Suggested Text below. |
| 2 | **MISSING** | `__init__` | `src/ham/geometry/mesh.py:15` | No docstring. Constructor parameters `vertices` and `faces` are undocumented. The derived attribute `triangles` is unexplained. | `"""Construct a TriangularMesh.\n\nArgs:\n    vertices: Array of shape (V, N) — the V vertex positions in R^N.\n    faces: Integer array of shape (F, 3) — each row indexes three vertices\n        forming a triangle.\n\nDerived Attributes:\n    triangles: Array of shape (F, 3, N) — vertex coordinates of each face,\n        computed as ``vertices[faces]``.\n"""` |
| 3 | **MISSING** | `ambient_dim` | `src/ham/geometry/mesh.py:20` | No docstring. The parent `Manifold.ambient_dim` has one; the override should either inherit it visibly or repeat it for clarity. | `"""The dimension N of the ambient embedding space R^N."""` |
| 4 | **MISSING** | `intrinsic_dim` | `src/ham/geometry/mesh.py:24` | No docstring. Always returns 2 but does not state that the mesh is assumed to be a 2-manifold. | `"""The intrinsic dimension of the mesh (always 2 for a triangulated surface)."""` |
| 5 | **UNCLEAR** | `_point_triangle_distance` | `src/ham/geometry/mesh.py:28` | Docstring is `"Computes distance and closest point. (Logic identical to previous version)"`. This is a leftover refactoring note, not documentation. No Args, Returns, or algorithm description. The inline comment "Metric Tensor entries" (line 33) is misleading — `d3, d4, d5` are the Gram-matrix entries of edge vectors, not the Finsler fundamental tensor $g_{ij}$ from `spec/MATH_SPEC.md § 1.1`. This will confuse a mathematician reading the code. | Replace docstring with: `"""Compute squared distance and closest point from p to triangle tri.\n\nUses barycentric coordinates to test interior containment, then\nfalls back to edge-segment projection for exterior points.\n\nArgs:\n    p: Point in R^N.\n    tri: Array of shape (3, N) — the three triangle vertices.\n\nReturns:\n    dist_sq: Squared Euclidean distance from p to the closest point.\n    closest: The closest point on the triangle (shape (N,)).\n"""`. Rename inline comment from "Metric Tensor entries" to "Gram-matrix entries of edge vectors". |
| 6 | **MISSING** | `project` | `src/ham/geometry/mesh.py:59` | No docstring on the override. The parent class documents the interface, but `project` here has mesh-specific behavior (iterates over all triangles) that should be stated, including the $O(F)$ linear scan and the JIT compilation. | `"""Project a point from ambient space onto the nearest triangle of the mesh.\n\nPerforms a linear scan over all F faces and returns the closest point.\n\nArgs:\n    x: Point in R^N.\n\nReturns:\n    Closest point on the mesh surface, shape (N,).\n"""` |
| 7 | **UNCLEAR** | `get_face_index` | `src/ham/geometry/mesh.py:65` | Docstring says "Returns the index of the triangle closest to x" — adequate for ML engineers but missing Args/Returns typing. | Expand to: `"""Return the index of the face closest to the query point.\n\nArgs:\n    x: Point in R^N.\n\nReturns:\n    Scalar integer index into self.faces / self.triangles.\n"""` |
| 8 | **UNCLEAR** | `get_face_weights` | `src/ham/geometry/mesh.py:71` | Docstring says "Returns differentiable weights for each face based on proximity." Missing: the softmax formulation, meaning of `temperature`, and shape of the returned array. An ML engineer needs to know the output sums to 1; a mathematician needs to know the weighting kernel. | Expand to: `"""Compute differentiable face-proximity weights via softmax.\n\nWeights are computed as softmax(−d² · temperature) where d² is the\nsquared distance from x to each face. Higher temperature yields a\nsharper (more one-hot) distribution.\n\nArgs:\n    x: Point in R^N.\n    temperature: Inverse-bandwidth parameter (default 100.0).\n\nReturns:\n    Array of shape (F,) summing to 1, giving each face's weight.\n"""` |
| 9 | **MISSING** | `to_tangent` | `src/ham/geometry/mesh.py:77` | No docstring on the override. The Gram–Schmidt orthonormalization of the face basis is undocumented. A geometer needs to know the basis is constructed from edges AB and AC; an ML engineer needs to know the output lies in the span of those edges. | `"""Project an ambient vector onto the tangent plane of the nearest face.\n\nConstructs an orthonormal basis {e1, e2} for the face containing x via\nGram–Schmidt on edge vectors (B−A) and (C−A), then returns the\ncomponent of v in that subspace.\n\nArgs:\n    x: Base point in R^N (need not be exactly on the mesh).\n    v: Ambient vector in R^N.\n\nReturns:\n    Tangent-projected vector in R^N (lies in span{e1, e2}).\n"""` |
| 10 | **MISSING** | `retract` | `src/ham/geometry/mesh.py:90` | No docstring. Implements `project(x + delta)` — a "projected retraction" — which the parent class `Manifold.retract` explicitly mentions as a common choice, but this override is silent. | `"""Retraction via projection: returns project(x + delta).\n\nArgs:\n    x: Base point on the mesh.\n    delta: Tangent vector (or approximate tangent vector).\n\nReturns:\n    Point on the mesh closest to x + delta.\n"""` |
| 11 | **MISSING** | `random_sample` | `src/ham/geometry/mesh.py:94` | No docstring. The method uses area-weighted triangle selection and the fold-over trick for uniform barycentric sampling — both non-trivial and worth describing. The inline comment "(Same area-weighted logic as verified in tests)" is a refactoring artifact, not documentation. | `"""Sample points uniformly on the mesh surface.\n\nTriangles are selected with probability proportional to area. Points\nwithin each triangle are sampled uniformly via the standard\nfold-over method (reflect barycentric coordinates when r1+r2 > 1).\n\nArgs:\n    key: JAX PRNG key.\n    shape: Output batch shape; returned array has shape (*shape, N).\n\nReturns:\n    Points on the mesh surface, shape (*shape, N).\n"""` |
| 12 | **INACCURATE** | `_point_triangle_distance` (inline) | `src/ham/geometry/mesh.py:33` | Comment `# Metric Tensor entries` refers to `d3, d4, d5` which are dot products of edge vectors (`ab·ab`, `ab·ac`, `ac·ac`). In the HAMTools spec (§ 1.1), "metric tensor" refers to the Finsler fundamental tensor $g_{ij}(x,v)$. Using the same term here for a Euclidean Gram matrix creates a false association. | Change comment to `# Edge-vector Gram matrix entries` or `# Dot products of edge vectors`. |
| 13 | **TYPO** | `_point_triangle_distance` (inline) | `src/ham/geometry/mesh.py:28` | Parenthetical note "(Logic identical to previous version)" is a leftover VCS note with no value to readers. | Remove the parenthetical. |
| 14 | **UNCLEAR** | `TriangularMesh` (general) | whole file | No module-level docstring. The file has no top-of-file description stating what it contains or how it fits into the `ham.geometry` package. | Add a module docstring: `"""Triangular mesh manifold for discrete surfaces in R^N.\n\nProvides the TriangularMesh class, which implements the Manifold ABC\nfor piecewise-linear surfaces defined by vertex/face arrays. Used by\nDiscreteRanders (zoo.py) for anisotropic mesh-based metrics.\n\nSee also: spec/ARCH_SPEC.md § 5 (Module Structure).\n"""` |

---

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---|---|---|---|---|---|
| `TriangularMesh` (class) | Minimal (1 line) | No | N/A | No | No |
| `__init__` | No | No | N/A | N/A | No |
| `ambient_dim` | No | N/A | No | No | No |
| `intrinsic_dim` | No | N/A | No | No | No |
| `_point_triangle_distance` | Stub only | No | No | No | No |
| `project` | No | No | No | No | No |
| `get_face_index` | Yes (brief) | No | Partial | No | No |
| `get_face_weights` | Yes (brief) | No | No | No | No |
| `to_tangent` | No | No | No | No | No |
| `retract` | No | No | No | No | No |
| `random_sample` | No | No | No | No | No |

**Coverage: 2/11 symbols have any docstring; 0/11 have complete documentation.**

---

## Spec Alignment Notes

1. **`spec/ARCH_SPEC.md § 2.1`** defines the `Manifold` ABC with documented signatures for `project`, `to_tangent`, and `random_sample`. The `TriangularMesh` overrides provide no docstrings of their own, so the only documentation a reader can find is in the ABC. Since the mesh implementations have non-trivial mesh-specific behavior (linear face scan, Gram–Schmidt basis, area-weighted sampling), the ABC docstrings are insufficient.

2. **`spec/ARCH_SPEC.md § 5`** lists `mesh.py` as `# Triangular mesh manifold`. The class docstring repeats this almost verbatim but adds nothing about constructor arguments or usage.

3. **`spec/ARCH_SPEC.md § 3`** mentions `DiscreteRanders` as an "Anisotropic Mesh Metric" that uses `get_face_weights`. There is no cross-reference from `get_face_weights` to its downstream consumer in `zoo.py`, making it hard for either audience to understand the function's purpose in the larger system.

4. **`spec/MATH_SPEC.md § 1.1`** defines the fundamental tensor $g_{ij}$ as $\frac{1}{2}\frac{\partial^2 F^2}{\partial v^i \partial v^j}$. The inline comment "Metric Tensor entries" at `mesh.py:33` reuses this terminology for simple Euclidean dot products, creating a notation collision (Issue #12).

5. **`spec/ARCH_SPEC.md § 2.1`** defines `retract` as a separate abstract method on `Manifold`. The `TriangularMesh.retract` implements the "projected retraction" pattern mentioned in `manifold.py:101–113` but does not document this choice.

---

## Recommended Priority

| Priority | Issues |
|----------|--------|
| High | #1, #2, #5, #11, #12 — class identity, constructor, core algorithm, and misleading terminology |
| Medium | #6, #8, #9, #10, #14 — override docs and module docstring |
| Low | #3, #4, #7, #13 — property docs and typo cleanup |
