# Math Review: `examples/demo_discrete_zermelo.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Minor Issues.** The Zermelo navigation problem is correctly formulated: the Randers metric formula, the wind causality constraint, and the discrete approximation all match `spec/MATH_SPEC.md` § 5. However, the `Sphere` constructor is called with an incorrect positional argument, creating an $S^1$ metadata object while performing $S^2$ computations. The wind field is analytically tangent to the sphere, which is exemplary. The discrete–continuous energy comparison has a well-definedness caveat. No formula is mathematically wrong.

---

## Formula-by-Formula Audit

### 1. Sphere Construction

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 (Geometric Hierarchy, $S^2$ row)
- **Implementation:** `examples/demo_discrete_zermelo.py:14`
  ```python
  sphere_cont = Sphere(radius)  # radius = 1.0
  ```
  The `Sphere.__init__` signature is `(intrinsic_dim: int = 2, radius: float = 1.0)` ([surfaces.py](src/ham/geometry/surfaces.py#L28)). Passing `1.0` as the first positional argument sets `intrinsic_dim = 1` (i.e. $S^1 \subset \mathbb{R}^2$), not $S^2 \subset \mathbb{R}^3$ as intended.
- **Verdict:** WARNING
- **Notes:** All `Sphere` methods (`project`, `to_tangent`, `exp_map`, `log_map`) are dimension-agnostic — they operate on whatever array shape is provided — so the $S^2$ computations with 3D points are numerically correct. The error is in the metadata (`ambient_dim` returns 2 instead of 3, `intrinsic_dim` returns 1 instead of 2). This would become **CRITICAL** if any downstream code dispatches on these properties.
- **Recommended Action:** Change to `Sphere(intrinsic_dim=2, radius=1.0)` or `Sphere(2, 1.0)`.

---

### 2. Wind Field Tangency

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo: $W^i(x)$ must be a tangent vector field)
- **Implementation:** `examples/demo_discrete_zermelo.py:17–19`
  ```python
  def w_net(x):
      base = jnp.array([-x[1], x[0], 0.0])
      return 0.8 * base
  ```
- **Verdict:** STRONG
- **Notes:** The wind field $W(x) = 0.8\,(-x_1, x_0, 0)$ satisfies $x \cdot W(x) = -x_0 x_1 + x_1 x_0 = 0$ for all $x$, so it is exactly tangent to every sphere centred at the origin. On the unit sphere with the induced Euclidean metric ($h = I_3$), the wind norm is $\|W\|_h = 0.8\sqrt{x_0^2 + x_1^2} = 0.8\sin\theta \le 0.8 < 1$, strictly satisfying the Zermelo causality constraint $\|W\|_h < 1$ everywhere.

---

### 3. Continuous Randers Metric Formula

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, Zermelo Randers formula
- **Literature Reference:** Bao, Robles, Shen, "Zermelo navigation on Riemannian manifolds," *J. Differential Geom.* 66 (2004), 377–435.
- **Implementation:** [zoo.py](src/ham/geometry/zoo.py#L126–L133)
  ```python
  v_sq_h = jnp.sum(v_safe * Hv, axis=-1)       # v^T H v = ||v||_h^2
  W_dot_v = jnp.sum(v_safe * HW, axis=-1)       # v^T H W = <W,v>_h
  discriminant = lam * v_sq_h + W_dot_v**2
  cost = (jnp.sqrt(...discriminant...) - W_dot_v) / lam
  ```
  This implements:
  $$F(x,v) = \frac{\sqrt{\lambda\,\|v\|_h^2 + \langle W,v\rangle_h^2} - \langle W,v\rangle_h}{\lambda}, \quad \lambda = 1 - \|W\|_h^2$$
- **Verdict:** CORRECT
- **Notes:** Exact match with the spec formula. The sign convention ($-\langle W,v\rangle_h$ in the numerator) correctly encodes "tailwind reduces cost, headwind increases cost."

---

### 4. Wind Squashing (Causality Enforcement)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, constraint $\|W\|_h < 1$
- **Implementation:** [zoo.py](src/ham/geometry/zoo.py#L102–L107) (continuous) and [zoo.py](src/ham/geometry/zoo.py#L148–L150) (discrete)
  ```python
  # Continuous:
  squash_factor = (max_speed * jnp.tanh(w_norm)) / (w_norm + 1e-8)
  # Discrete:
  scale = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
  ```
- **Verdict:** WARNING
- **Notes:** The $\tanh$ squashing is a valid differentiable enforcement of $\|W\|_h < 1$, but it distorts even sub-critical wind magnitudes. For $\|W_{\mathrm{raw}}\| = 0.8$, the effective wind norm becomes $(1-\epsilon)\tanh(0.8) \approx 0.664$, not $0.8$. Both continuous and discrete versions apply the same distortion, so relative comparisons between them are fair. However, a user specifying "0.8 wind" receives $\approx 0.664$ effective wind. An alternative preserving magnitude for sub-critical inputs (e.g. $\min(w, 1-\epsilon)$-style clamping) would better respect user intent.

---

### 5. Discrete Randers Formula

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo formula with $h_{ij} = \delta_{ij}$)
- **Implementation:** [zoo.py](src/ham/geometry/zoo.py#L145–L157)
  ```python
  v_sq = jnp.dot(v, v)
  W_dot_v = jnp.dot(W, v)
  discriminant = lam * v_sq + W_dot_v**2
  cost = (jnp.sqrt(jnp.maximum(discriminant, 1e-8)) - W_dot_v) / lam
  ```
- **Verdict:** CORRECT
- **Notes:** This is the Zermelo–Randers formula with $H = I$ (Euclidean background). For the demo, the continuous `h_net` returns $I_3$, so the discrete and continuous formulas are analytically identical when evaluated at the same $(x,v,W)$. The $\lambda$ computation `lam = 1.0 - (w_norm * scale)**2 = 1 - \|W\|^2$ is correct since $\|W\| = \|W_{\mathrm{raw}}\| \cdot |\mathrm{scale}| = w_{\mathrm{norm}} \cdot \mathrm{scale}$.

---

### 6. Discrete Wind Field Assignment

- **Spec Reference:** N/A (discrete approximation, not in MATH_SPEC)
- **Literature Reference:** Standard piecewise-constant FEM approximation on simplicial meshes.
- **Implementation:** `examples/demo_discrete_zermelo.py:43–44`
  ```python
  face_centers = jnp.mean(verts[faces], axis=1)
  face_winds = jax.vmap(w_net)(face_centers)
  ```
- **Verdict:** CORRECT
- **Notes:** Evaluating the wind at triangle barycenters and assigning a constant wind per face is a standard $O(h)$ piecewise-constant approximation. The face centroids are not exactly on $S^2$ (they lie slightly inside the sphere), but $W(x) = 0.8(-x_1, x_0, 0)$ still satisfies $x \cdot W = 0$ at any $x$, so the vectors are tangent to the sphere even at interior points. With `subdivisions=1` (80 faces), the approximation is coarse; this is acceptable for a demo.

---

### 7. Discrete Wind Interpolation (Missing Tangent Projection)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 ($W^i$ must be a tangent vector field)
- **Implementation:** [zoo.py](src/ham/geometry/zoo.py#L145–L150)
  ```python
  weights = self.manifold.get_face_weights(x)
  W_raw = jnp.dot(weights, self.face_winds)
  ```
  Compare with the continuous version at [zoo.py](src/ham/geometry/zoo.py#L96):
  ```python
  W_raw = self.manifold.to_tangent(z, W_raw)
  ```
- **Verdict:** WARNING
- **Notes:** The continuous `Randers` projects the raw wind to the tangent space $T_x S^2$ before squashing. `DiscreteRanders` skips this step: the softmax-weighted average of per-face winds from varying tangent planes can produce a vector with a small normal component. For this demo's rotationally symmetric wind field, the error is negligible (the wind is exactly tangent everywhere), but for general wind fields on curved meshes this omission would introduce a systematic bias.
- **Recommended Action:** Add `W_raw = self.manifold.to_tangent(x, W_raw)` after the softmax interpolation in `DiscreteRanders.metric_fn`.

---

### 8. Energy Comparison Across Manifolds

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 ($E(x,v) = \frac{1}{2}F^2(x,v)$)
- **Implementation:** `examples/demo_discrete_zermelo.py:47–48`
  ```python
  e_mesh = batch_energy(traj_mesh.xs[:-1], traj_mesh.vs).sum()
  ```
  where `batch_energy = jax.vmap(metric_randers.energy)` (the **continuous** Randers metric).
- **Verdict:** WARNING
- **Notes:** The mesh trajectory `traj_mesh` is solved on the `TriangularMesh` manifold; its points lie on mesh faces, not exactly on $S^2$. Evaluating the continuous `metric_randers.energy` at these off-sphere points is not strictly well-defined because `_get_zermelo_data` calls `self.manifold.to_tangent(z, W_raw)` with $z$ as a `Sphere` point, but the actual $z$ is slightly off-sphere. Additionally, velocities `traj_mesh.vs` are computed via `TriangularMesh.log_map` (tangent-projected secant), not `Sphere.log_map` (arccos-based). For a dense mesh these errors are small, but the comparison is not exact.
- **Recommended Action:** Either (a) evaluate all energies using a common ambient formula, or (b) add a note in the demo that the comparison is approximate.

---

### 9. Mission Comment Inaccuracy

- **Implementation:** `examples/demo_discrete_zermelo.py:23–25`
  ```python
  # --- 2. Mission: South -> North ---
  start = jnp.array([0.0, 1.0, 0.0])
  end   = jnp.array([0.0,  0.0, 1.0])
  ```
- **Verdict:** NOTE
- **Notes:** The start point $(0,1,0)$ lies on the equator (positive $y$-axis), not the south pole $(0,0,-1)$. The mission is "Equator $\to$ North Pole," not "South $\to$ North."

---

### 10. Energy Functional ($E = \frac{1}{2}F^2$)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2
- **Implementation:** [metric.py](src/ham/geometry/metric.py#L36)
  ```python
  def energy(self, x, v):
      return 0.5 * self.metric_fn(x, v)**2
  ```
- **Verdict:** CORRECT
- **Notes:** Matches $E(x,v) = \frac{1}{2}F^2(x,v)$ exactly.

---

### 11. AVBD Solver Discrete Action

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (Euler–Lagrange $\Rightarrow$ geodesic equation via energy minimisation)
- **Implementation:** [avbd.py](src/ham/solvers/avbd.py#L99–L103)
  ```python
  def local_action(x_prev, x, x_next):
      v_in = metric.manifold.log_map(x_prev, x)
      v_out = metric.manifold.log_map(x, x_next)
      return metric.energy(x_prev, v_in) + metric.energy(x, v_out)
  ```
- **Verdict:** CORRECT
- **Notes:** The discrete action $\sum_i E(x_i, \log_{x_i}(x_{i+1}))$ is the standard Regge-type discretisation of the continuous energy functional $\int E(x, \dot{x})\,dt$. Minimising over interior vertices recovers discrete geodesics. The use of `log_map` for velocities (rather than raw secants) correctly accounts for manifold curvature.

---

## Open Questions

1. **Sphere constructor intent:** Should `Sphere` accept `radius` as a keyword-only argument to prevent the positional-arg confusion seen here? This is an API design question beyond math review scope.
2. **Squashing calibration:** Should the Zermelo wind squashing preserve sub-critical magnitudes exactly (e.g. via $\min$-clamp instead of $\tanh$)? The current $\tanh$ approach is smooth and safe for autodiff but systematically attenuates user-specified wind. This is a modelling decision.
3. **DiscreteRanders generality:** Should `DiscreteRanders` accept an optional per-face metric $h_{ij}$ to match the full Zermelo parameterisation? Currently it implicitly assumes $h = I$.
