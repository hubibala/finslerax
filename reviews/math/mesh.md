# Math Review: mesh.py

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** `src/ham/geometry/mesh.py`

## Summary

**Verdict: Minor Issues.** The module implements a discrete piecewise-linear manifold via triangular meshes. All core mathematical operations—barycentric projection, Gram-Schmidt tangent basis, area-weighted sampling, and edge-segment projection—are analytically correct. No CRITICAL formula errors were found. Two WARNING-level issues relate to gradient discontinuities inherent to piecewise-linear geometry and a `jnp.where` NaN-leakage risk in automatic differentiation. One WARNING concerns the barycentric coordinate solver behaviour on degenerate (zero-area) triangles.

---

## Formula-by-Formula Audit

### 1. Barycentric Coordinate Solver (`_point_triangle_distance`, lines 22–35)

- **Spec Reference:** Not in `spec/MATH_SPEC.md` (discrete geometry, no continuous Finsler analog).
- **Literature Reference:** Eberly, *Distance Between Point and Triangle in 3D*, Geometric Tools (2021); also standard in any computational geometry text (e.g., Ericson, *Real-Time Collision Detection*, §5.1.5).
- **Implementation:**
  Given triangle vertices $a, b, c$ and query point $p$, define edge vectors $\vec{ab} = b - a$, $\vec{ac} = c - a$, $\vec{ap} = p - a$. The code solves for barycentric parameters $(s, t)$ via the normal equations of the least-squares projection onto the triangle plane:

  $$\begin{pmatrix} \vec{ab}\cdot\vec{ab} & \vec{ab}\cdot\vec{ac} \\ \vec{ab}\cdot\vec{ac} & \vec{ac}\cdot\vec{ac} \end{pmatrix} \begin{pmatrix} s \\ t \end{pmatrix} = \begin{pmatrix} \vec{ab}\cdot\vec{ap} \\ \vec{ac}\cdot\vec{ap} \end{pmatrix}$$

  The code assigns:
  ```
  d1 = ab·ap, d2 = ac·ap, d3 = ab·ab, d4 = ab·ac, d5 = ac·ac
  ```
  and solves by Cramer's rule:
  ```python
  det = max(d3*d5 - d4*d4, 1e-10)
  s = (d5*d1 - d4*d2) / det    # Correct: (ac·ac)(ab·ap) - (ab·ac)(ac·ap)
  t = (d3*d2 - d4*d1) / det    # Correct: (ab·ab)(ac·ap) - (ab·ac)(ab·ap)
  ```
- **Verdict:** OK
- **Notes:** Cramer's rule for the $2\times 2$ Gram matrix is analytically correct. The determinant $\det = \|\vec{ab}\|^2\|\vec{ac}\|^2 - (\vec{ab}\cdot\vec{ac})^2 = \|\vec{ab} \times \vec{ac}\|^2 \geq 0$ by the Cauchy-Schwarz inequality (and Lagrange's identity in higher dimensions), so clamping at $10^{-10}$ is numerically safe.

---

### 2. Degenerate Triangle Handling (line 30)

- **Literature Reference:** Cauchy-Schwarz: $\det = 0$ iff $\vec{ab} \parallel \vec{ac}$ (collinear or coincident vertices).
- **Implementation:**
  ```python
  det = jnp.maximum(d3 * d5 - d4 * d4, 1e-10)
  ```
- **Verdict:** WARNING
- **Notes:** When the triangle is degenerate ($\det \to 0$), the computed $(s, t)$ blow up to $O(1/\epsilon)$ and the resulting `p_in` is far from the actual triangle. The inside test `is_inside = (s >= 0) & (t >= 0) & (s + t <= 1)` will correctly evaluate to `False`, so the edge-projection fallback activates and the final closest point is correct. However, during **reverse-mode AD**, JAX evaluates gradients through both branches of `jnp.where(is_inside, p_in, p_edge)` (line 48). If `p_in` contains extreme values from a near-degenerate triangle, the gradient can overflow or produce large spurious values even when `p_edge` is selected.
- **Recommended Action:** Clamp `s` and `t` to a reasonable range (e.g., `[-10, 10]`) before computing `p_in`, or use `jax.lax.cond` to avoid evaluating the unused branch during differentiation.

---

### 3. Inside-Triangle Test (line 44)

- **Literature Reference:** Standard barycentric membership: a point $a + s\vec{ab} + t\vec{ac}$ lies inside the triangle iff $s \geq 0$, $t \geq 0$, $s + t \leq 1$.
- **Implementation:**
  ```python
  is_inside = (s >= 0) & (t >= 0) & (s + t <= 1)
  ```
- **Verdict:** OK
- **Notes:** Correct and standard.

---

### 4. Segment Projection (`project_segment`, lines 38–42)

- **Literature Reference:** Orthogonal projection of point $p$ onto segment $[u, v]$: $\pi(p) = u + \text{clamp}\!\left(\frac{(p - u)\cdot(v - u)}{\|v - u\|^2}, 0, 1\right)(v - u)$.
- **Implementation:**
  ```python
  frac = jnp.clip(jnp.dot(up, uv) / jnp.maximum(len_sq, 1e-10), 0.0, 1.0)
  return u + frac * uv
  ```
- **Verdict:** OK
- **Notes:** Correct. Clipping to $[0, 1]$ restricts the result to the segment. The `jnp.maximum(len_sq, 1e-10)` prevents division by zero for zero-length edges. Vertex coverage is complete: vertex $a$ is reachable as `project_segment(a, b)` at $\text{frac}=0$ and `project_segment(c, a)` at $\text{frac}=1$, and similarly for $b$ and $c$.

---

### 5. Closest-Edge Selection (lines 45–47)

- **Implementation:**
  ```python
  d_edge_vals = jnp.array([dist_sq(p_ab), dist_sq(p_bc), dist_sq(p_ca)])
  best_edge_idx = jnp.argmin(d_edge_vals)
  p_edge = jnp.stack([p_ab, p_bc, p_ca])[best_edge_idx]
  ```
- **Verdict:** WARNING
- **Notes:** `jnp.argmin` is non-differentiable (piecewise constant). When the query point is equidistant from two edges, the gradient of the selected closest point is undefined. This is inherent to the piecewise-linear geometry and affects any downstream gradient-based optimization (e.g., geodesic learning on meshes). The mathematical result (closest point) is correct; only the gradient is discontinuous.
- **Recommended Action:** If smooth gradients are required, consider a soft-minimum selection (analogous to the `get_face_weights` softmax approach already implemented at line 67).

---

### 6. Tangent Projection via Gram-Schmidt (`to_tangent`, lines 73–81)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1 (tangent space concept); discrete analog not in spec.
- **Literature Reference:** Standard orthogonal projection onto a 2D subspace: construct ONB $\{e_1, e_2\}$ via Gram-Schmidt, then $\text{proj}_{T_x M}(v) = \langle v, e_1\rangle e_1 + \langle v, e_2\rangle e_2$.
- **Implementation:**
  ```python
  u = b - a
  w = c - a
  e1 = u / (norm(u) + 1e-10)
  w_perp = w - dot(w, e1) * e1
  e2 = w_perp / (norm(w_perp) + 1e-10)
  return dot(v, e1) * e1 + dot(v, e2) * e2
  ```
- **Verdict:** OK
- **Notes:** Mathematically correct. Gram-Schmidt produces an orthonormal basis $\{e_1, e_2\}$ for the plane spanned by edges $(b-a)$ and $(c-a)$. The projection formula $v_\parallel = (v \cdot e_1) e_1 + (v \cdot e_2) e_2$ is the standard orthogonal projection onto a subspace. Works correctly for arbitrary ambient dimension $N \geq 2$.

  **Verified:** $\langle v_\parallel, n \rangle = 0$ for any normal $n \perp \text{span}(e_1, e_2)$, and $\|v_\parallel\| \leq \|v\|$.

---

### 7. Face Proximity Weights (`get_face_weights`, lines 65–68)

- **Literature Reference:** Soft nearest-neighbour via Boltzmann distribution: $w_i = \frac{\exp(-T \cdot d_i^2)}{\sum_j \exp(-T \cdot d_j^2)}$.
- **Implementation:**
  ```python
  return jax.nn.softmax(-dists_sq * temperature)
  ```
- **Verdict:** OK
- **Notes:** Correct differentiable approximation to hard face assignment. As $T \to \infty$, the weights concentrate on the nearest face. The use of **squared** distances rather than distances in the exponent is a valid design choice (Gaussian kernel). The gradient with respect to $x$ is well-defined everywhere, unlike the hard `argmin` in `get_face_index`.

---

### 8. Retraction (`retract`, lines 83–84)

- **Literature Reference:** Projected retraction: $R_x(\delta) = \pi(x + \delta)$ where $\pi$ is the closest-point projection onto $M$.
- **Implementation:**
  ```python
  candidate = x + delta
  return self.project(candidate)
  ```
- **Verdict:** OK
- **Notes:** Standard projected retraction for embedded submanifolds. First-order accurate: $R_x(t\delta) = x + t\delta + O(t^2)$. For piecewise-linear manifolds this is the natural choice, as no closed-form exponential map exists.

---

### 9. Triangle Area Computation (`random_sample`, lines 90–93)

- **Spec Reference:** Not in `spec/MATH_SPEC.md`.
- **Literature Reference:** Lagrange's identity / generalized cross product: for vectors $u, v \in \mathbb{R}^N$,
  $$\|u \times v\|^2 = \|u\|^2\|v\|^2 - (u \cdot v)^2 = \det\begin{pmatrix} u \cdot u & u \cdot v \\ u \cdot v & v \cdot v \end{pmatrix}$$
  Area $= \frac{1}{2}\sqrt{\|u\|^2\|v\|^2 - (u\cdot v)^2}$.
- **Implementation:**
  ```python
  u_sq, v_sq = jnp.sum(u**2, axis=1), jnp.sum(v**2, axis=1)
  uv_dot = jnp.sum(u * v, axis=1)
  areas = 0.5 * jnp.sqrt(jnp.maximum(u_sq * v_sq - uv_dot**2, 1e-10))
  ```
- **Verdict:** OK
- **Notes:** Correct and dimension-agnostic. Uses the Gram determinant, which is the proper generalization of $\|u \times v\|$ to $\mathbb{R}^N$ for $N > 3$. The `jnp.maximum(..., 1e-10)` guard is appropriate since $\|u\|^2\|v\|^2 - (u\cdot v)^2 \geq 0$ by Cauchy-Schwarz, but floating-point arithmetic can produce small negative values.

---

### 10. Uniform Triangle Sampling (`random_sample`, lines 97–102)

- **Literature Reference:** Osada et al., *Shape Distributions*, ACM TOG 2002, §4.2. Standard folding method: draw $(r_1, r_2) \sim \text{Uniform}([0,1]^2)$, if $r_1 + r_2 > 1$ then $(r_1, r_2) \leftarrow (1 - r_1, 1 - r_2)$. The point $p = (1 - r_1 - r_2)\,A + r_1\,B + r_2\,C$ is uniformly distributed over the triangle $ABC$.
- **Implementation:**
  ```python
  mask = r1 + r2 > 1
  r1 = jnp.where(mask, 1 - r1, r1)
  r2 = jnp.where(mask, 1 - r2, r2)
  pts = (1 - r1[:,None] - r2[:,None]) * tris[:,0] + r1[:,None] * tris[:,1] + r2[:,None] * tris[:,2]
  ```
- **Verdict:** OK
- **Notes:** Correct. After folding, the barycentric weights $(\lambda_0, \lambda_1, \lambda_2) = (1 - r_1 - r_2,\; r_1,\; r_2)$ satisfy $\lambda_i \geq 0$ and $\sum \lambda_i = 1$, ensuring the point lies inside the triangle. The folding bijectively maps $[0,1]^2$ onto the standard simplex, preserving uniformity.

  **Verified:** The area-weighted face selection `jax.random.choice(..., p=areas/sum(areas))` ensures overall uniform sampling across the mesh surface (probability proportional to face area).

---

### 11. `exp_map` / `log_map` (inherited from `Manifold` base class)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2 (Geodesic Spray), § 4.1 (Surface Formulations).
- **Literature Reference:** For piecewise-linear surfaces, geodesics are sequences of straight segments that obey Snell's-law-like angle-matching at edge crossings (Polthier & Schmies, *Straightest Geodesics on Polyhedral Surfaces*, 1998).
- **Implementation:** `exp_map` falls back to `retract` (projected retraction). `log_map` falls back to tangent projection of the ambient secant with distance rescaling (`manifold.py:136–150`).
- **Verdict:** WARNING
- **Notes:** The base-class `log_map` projects the ambient secant $y - x$ onto $T_x M$ and rescales to preserve ambient distance. For a triangular mesh, this is only valid when $x$ and $y$ lie on the **same** face or on adjacent faces with small dihedral angle. When $x$ and $y$ are separated by multiple face crossings, the ambient secant may point in a direction unrelated to the intrinsic geodesic, and the intrinsic distance (sum of segment lengths along the unfolded path) differs from the ambient distance.

  This is a fundamental limitation of the piecewise-linear setting rather than a formula error. True discrete geodesics require edge-unfolding algorithms (Mitchell-Mount-Papadimitriou, or the heat method of Crane et al., 2013).
- **Recommended Action:** Document this limitation in the class docstring. If intrinsic geodesics on meshes are needed, consider implementing the heat method or an unfolding-based exact geodesic solver.

---

## Open Questions

1. **Gradient continuity through `project`:** The `project` function chains `argmin` (face selection) → `argmin` (edge selection) → `jnp.where` (inside/outside). All three operations introduce gradient discontinuities. Is any downstream use case (e.g., `FinslerMetric` energy evaluation on `TriangularMesh`) expected to differentiate through `project`? If so, the `get_face_weights` softmax approach should be extended to the full projection pipeline.

2. **Intrinsic metric on the mesh:** `TriangularMesh` implements the manifold topology (projection, tangent space) but not the `FinslerMetric` interface. How is the energy $E(x, v)$ defined on this domain — is it the ambient Euclidean metric restricted to each face, or is a separate `FinslerMetric` object composed with this manifold? The mathematical consistency of the spray/geodesic machinery depends on this pairing being correctly specified.

3. **Higher-dimensional Gram-Schmidt stability:** For meshes in $\mathbb{R}^N$ with $N \gg 3$, the Gram-Schmidt procedure is well-conditioned as long as the triangle is non-degenerate. However, no explicit check is performed for near-degenerate triangles (where $\|w_\perp\| \approx 0$ implies $\vec{ac} \approx k \cdot \vec{ab}$). The $10^{-10}$ guard prevents division by zero but produces a near-arbitrary $e_2$ direction. Is this acceptable for the intended use cases?
