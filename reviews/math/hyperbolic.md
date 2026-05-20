# Math Review: hyperbolic (vis)

**Reviewer:** Math Reviewer Agent
**Date:** 2025-05-15
**Spec Version:** 1.1.0 (Berwald Revision)
**Source:** `src/ham/vis/hyperbolic.py`

## Summary

**Correct.** Both analytical formulas (stereographic projection and its pushforward) are mathematically correct and consistent with the hyperboloid convention established in `src/ham/geometry/surfaces.py`. The only findings are minor: a missing input-validation guard and the use of straight-line edges as an approximation to Poincaré geodesic arcs (acknowledged in code comments).

---

## Formula-by-Formula Audit

### 1. `project_to_poincare` — Stereographic Projection

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, § 4.1 (Hyperboloid row in hierarchy table; surface formulations).
- **Literature Reference:** Cannon et al., "Hyperbolic Geometry," §7; Ratcliffe, *Foundations of Hyperbolic Manifolds*, §4.5.
- **Implementation** (`src/ham/vis/hyperbolic.py:12–14`):
  ```python
  x0 = x[..., 0:1]
  x_spatial = x[..., 1:]
  return x_spatial / (1.0 + x0)
  ```
- **Expected formula:** The standard stereographic projection from the hyperboloid $\mathbb{H}^n = \{x \in \mathbb{R}^{n+1} \mid -x_0^2 + \sum_{i=1}^n x_i^2 = -1,\; x_0 > 0\}$ to the Poincaré ball $\mathbb{B}^n$ projects from the point $(-1, 0, \ldots, 0)$:
  $$y^i = \frac{x^i}{1 + x_0}, \quad i = 1, \ldots, n.$$
  On the upper sheet $x_0 \geq 1$, so $1 + x_0 \geq 2 > 0$; the map is well-defined.
- **Verdict:** OK
- **Notes:** The formula matches the standard projection exactly. Convention ($x_0$ as the timelike component, Lorentzian signature $(-,+,\ldots,+)$) is consistent with `Hyperboloid._minkowski_dot` in `src/ham/geometry/surfaces.py:313`.

---

### 2. `project_vector_to_poincare` — Pushforward (Differential of Projection)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1 (surface tangent maps, implicit).
- **Literature Reference:** Standard differential-geometry computation; see Lee, *Introduction to Smooth Manifolds*, Proposition 3.24 (pushforward via Jacobian).
- **Implementation** (`src/ham/vis/hyperbolic.py:23–31`):
  ```python
  v0 = v[..., 0:1]
  v_spatial = v[..., 1:]
  denom = (1.0 + x0)**2
  num = v_spatial * (1.0 + x0) - x_spatial * v0
  return num / denom
  ```
- **Expected formula:** The projection $\pi^i(x) = x^i / (1 + x^0)$ has Jacobian entries:
  $$\frac{\partial \pi^i}{\partial x^0} = \frac{-x^i}{(1+x^0)^2}, \qquad \frac{\partial \pi^i}{\partial x^j} = \frac{\delta^i_j}{1+x^0} \quad (j \geq 1).$$
  The pushforward is therefore:
  $$w^i = d\pi_x(v)^i = \frac{v^i(1+x^0) - x^i v^0}{(1+x^0)^2}.$$
- **Verdict:** OK
- **Notes:** Direct quotient-rule application, correctly implemented. The tangent vector $v$ is not required to satisfy the hyperboloid tangent constraint ($\langle x, v \rangle_L = 0$); the formula is valid for any ambient vector, which is appropriate for visualization of arbitrary vector fields (e.g., wind).

---

### 3. `plot_poincare_disk` — Geodesic Edge Approximation

- **Spec Reference:** N/A (visualization utility).
- **Literature Reference:** Geodesics in the Poincaré ball are circular arcs orthogonal to the boundary circle (see Ratcliffe §4.5, Cannon et al. §7).
- **Implementation** (`src/ham/vis/hyperbolic.py:63–68`):
  ```python
  # Geodesics would be arcs, straight lines are approx
  start_pts = points_p[lineage_pairs[:, 0]]
  end_pts = points_p[lineage_pairs[:, 1]]
  lines = np.stack([start_pts, end_pts], axis=1)
  ```
- **Verdict:** WARNING
- **Notes:** Lineage edges are rendered as Euclidean straight-line segments instead of true hyperbolic geodesic arcs (which are circular arcs orthogonal to $\partial \mathbb{B}^n$). The comment in the code acknowledges this. Straight lines are a reasonable first-order approximation near the disk center but become visually misleading near the boundary where the conformal factor $(1 - \|y\|^2)^{-2}$ grows.  
  **Recommended Action:** If geodesic accuracy matters for publications, implement the Möbius geodesic arc rendering. For exploratory visualization this is acceptable.

---

### 4. Input Validation — Hyperboloid Membership

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1 (upper-sheet constraint $x_0 > 0$, $-x_0^2 + \|x_{\text{sp}}\|^2 = -1$).
- **Implementation:** No explicit check that inputs lie on the upper sheet.
- **Verdict:** WARNING
- **Notes:** If a user passes points not on the upper hyperboloid sheet (e.g., $x_0 < 0$, or points from the Euclidean ambient space), the projection formula still executes without error but produces geometrically meaningless results (potentially outside the unit ball). Since this is a pure visualization utility and the `Hyperboloid` class already provides a `project` method for constraint enforcement, the risk is low.  
  **Recommended Action:** Consider adding an optional assertion `assert (x[..., 0] > 0).all()` or a bounds check on the output $\|y\| < 1$ to aid debugging.

---

## Open Questions

1. **Geodesic arc rendering:** Should the visualization draw true Poincaré geodesic arcs (circular arcs orthogonal to $\partial \mathbb{B}$) for lineage edges? This would require computing the Möbius midpoint and arc radius, adding non-trivial complexity to a visualization utility. The answer depends on whether these plots are intended for publication or for quick inspection.

2. **Conformal scaling of quiver arrows:** The pushforward $d\pi$ is mathematically correct, but the Euclidean lengths of the projected vectors do not reflect their hyperbolic norms (since $ds_{\text{Poincaré}}^2 = \frac{4}{(1-\|y\|^2)^2} \|dy\|^2$). Vectors near the boundary will appear shorter than their true hyperbolic magnitude. Is this intentional, or should the arrows be rescaled by the conformal factor for visual fidelity?
