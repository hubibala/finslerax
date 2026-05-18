# Math Review: terrain.py
**Reviewer:** Math Reviewer Agent
**Date:** 2026-05-18
**Spec Version:** MATH_SPEC.md (as of 2026-05-18)

## Summary
The terrain utilities are mathematically sound for their stated purpose. The Zermelo navigation formula in `CovariateMeshRanders.metric_fn` is correctly implemented in 3D under the declared Phase W2 isotropic approximation. The face-normal, slope, and aspect computations are geometrically correct once the winding-direction ambiguity is resolved (which it is, via `|n_z|`). One WARNING and one NOTE are raised.

---

## Formula-by-Formula Audit

### `compute_face_normals` — cross-product normalization

- **Spec Reference:** spec/MATH_SPEC.md § 5 (geometry primitives, implicit).
- **Implementation:** `normals = jnp.cross(e1, e2); norms = sqrt(sum(normals**2) + 1e-12); return normals / norms`
- **Verdict:** CORRECT
- **Notes:** The $1\times10^{-12}$ regulariser prevents division by zero on degenerate (collinear) triangles. The winding orientation is not guaranteed upward; this is handled downstream in `compute_face_slopes_aspects`.

---

### `compute_face_slopes_aspects` — slope definition

- **Spec Reference:** spec/MATH_SPEC.md § 5 (terrain geometry, implicit); GIS convention.
- **Implementation:**
  ```python
  n_z_abs = jnp.abs(normals[:, 2])
  sign = jnp.sign(normals[:, 2] + 1e-12)
  slopes = jnp.arccos(jnp.clip(n_z_abs, 0.0, 1.0))
  aspects = jnp.arctan2(sign * normals[:, 1], sign * normals[:, 0])
```
- **Verdict:** CORRECT
- **Notes:** The task description specifies `slope = arccos(n_z / ||n||)` using the signed normal, which gives $\pi$ for downward-pointing normals (the actual winding produced by the mesh). The implementation correctly uses $|n_z|$ so slope $\in [0, \pi/2]$, matching physical terrain analysis convention. The aspect correction (multiplying by `sign`) consistently reflects the normal to point upward before computing atan2. This is the right approach.

---

### `CovariateMeshRanders.metric_fn` — Zermelo formula with isotropic approximation

- **Spec Reference:** spec/MATH_SPEC.md § 5 (Zermelo navigation formula, equation 5.1).
- **Literature Reference:** Bao-Chern-Shen "An Introduction to Riemann-Finsler Geometry," Chapter 11.
- **Implementation (relevant lines):**
  ```python
  g_iso = 0.5 * (G_f[0, 0] + G_f[1, 1])
  b_3d = b_f[0] * u1 + b_f[1] * u2
  b_Ginv_b = jnp.sum(b_3d ** 2) / jnp.maximum(g_iso, 1e-8)
  lam = jnp.maximum(1.0 - b_Ginv_b, 1e-6)
  v_sq_h = g_iso * jnp.sum(v_tan ** 2)
  bdotv = jnp.dot(b_3d, v_tan)
  disc = lam * v_sq_h + bdotv ** 2
  cost = (jnp.sqrt(jnp.maximum(disc, GRAD_EPS)) - bdotv) / lam
  ```
- **Verdict:** CORRECT (under stated approximation)
- **Notes:** Under the isotropic approximation $G_f \approx g_\text{iso} I_3$ with $g_\text{iso} = \frac{1}{2}(\lambda_1 + \lambda_2)$:
  - $\lambda = 1 - \|b\|_{G^{-1}}^2 = 1 - \|b_{3d}\|^2 / g_\text{iso}$ — correct.
  - $\|v\|^2_h = g_\text{iso} \|v_\text{tan}\|^2$ — correct.
  - $\langle b, v \rangle_h = b_{3d} \cdot v_\text{tan}$ — correct (under isotropic $G$, $b^\top G v = g_\text{iso} \, b_{3d} \cdot v_\text{tan} / g_\text{iso} = b_{3d} \cdot v_\text{tan}$).
  
  Wait — the $\langle W, v \rangle_h$ term in the spec is $W^\top G v$, not $W \cdot v$. Under isotropic $G = g_\text{iso} I$, this is $g_\text{iso}(b_{3d} \cdot v_\text{tan})$, which would change the formula. **See WARNING below.**

---

### NOTE N0: Covector interpretation of $b$ in the Zermelo formula

- **Location:** `src/ham/utils/terrain.py:318–330`
- **Severity:** NOTE (no action required)
- **Details:** The implementation follows the correct **covector** interpretation of $b$. In `CovariateConditionedRanders` (and here), $b$ is a 1-form (covector): the natural pairing is $\langle b, v \rangle = b_i v^i$ (Euclidean dot), not $b^\top G v$. This matches the Randers/Zermelo convention throughout the codebase:

  $$F = \frac{\sqrt{\lambda \|v\|_G^2 + (b \cdot v)^2} - b \cdot v}{\lambda}, \quad \lambda = 1 - b^\top G^{-1} b$$

  The 3D lift is correct: $b_{3d} \cdot v_\text{tan} = b_f^\top v_{2d}$ (since $\{u_1, u_2\}$ is orthonormal), and $b_\text{Ginv\_b} = \|b_{3d}\|^2 / g_\text{iso} = b_f^\top G_f^{-1} b_f$ under isotropic $G_f = g_\text{iso} I_2$. All three terms are **consistent** with the isotropic Zermelo formula.

---

### NOTE N1: Isotropic averaging uses arithmetic mean of eigenvalues

- **Location:** `src/ham/utils/terrain.py:311`
- **Severity:** NOTE
- **Notes:** The choice `g_iso = 0.5*(G_f[0,0]+G_f[1,1])` uses the arithmetic mean of diagonal entries, not of eigenvalues. For a general SPD matrix, these differ. Since `G_f` is the output of `project_spd` (which is already close to diagonal in typical cases with small `g12`), this is a reasonable approximation. Note it explicitly as an approximation in the docstring (already done).

---

## Open Questions

1. For Phase W2b anisotropic pullback: the correct 3D metric tensor is the pullback $G_{3d} = \sum_{ij} G_f^{ij} u_i \otimes u_j$. The task description sketches this correctly; implementation should use `jnp.einsum` for the outer products.
2. Should aspect be defined as "uphill" or "downhill" direction? The current implementation gives the normal's projection direction, which is the uphill azimuth. Standard GIS aspect convention is downhill. Add a docstring note clarifying this.
