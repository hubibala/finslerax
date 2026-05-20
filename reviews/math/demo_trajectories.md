# Math Review: `examples/demo_trajectories.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

The demo is **mathematically sound** with **minor issues**. The Randers metric, Rossby–Haurwitz wind field, geodesic ODE, and sphere constructions are all correctly set up. The main concerns are: (1) a misleading comment about the start/end geometry, (2) a mismatch between the effective wind seen by the Randers geodesic (post-tanh squashing) and the wind used for passive advection, and (3) a heuristic velocity scaling for the verification shot that is principled but approximate.

---

## Formula-by-Formula Audit

### 1. Randers metric construction

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization)
- **Literature Reference:** Bao–Robles–Shen, "Zermelo navigation on Riemannian manifolds," *J. Differential Geom.* 66 (2004), 377–435
- **Implementation:** `examples/demo_trajectories.py:78`
  ```python
  h_net = lambda x: jnp.eye(3)
  metric = Randers(sphere, h_net, w_net)
  ```
- **Verdict:** CORRECT
- **Notes:** Using the ambient Euclidean metric $H = I_3$ as the Riemannian "sea" is valid. For $S^2 \hookrightarrow \mathbb{R}^3$, the restriction $v^T I_3 \, v = \|v\|^2$ for tangent vectors $v \perp x$ reproduces the standard round metric. The Randers `metric_fn` in `src/ham/geometry/zoo.py:118–131` correctly implements
  $$F(x, v) = \frac{\sqrt{\lambda\,\|v\|_H^2 + \langle W, v \rangle_H^2} - \langle W, v \rangle_H}{\lambda}$$
  with $\lambda = 1 - \|W\|_H^2$, matching the spec.

---

### 2. Rossby–Haurwitz stream function

- **Spec Reference:** N/A (not in MATH_SPEC; standard atmospheric dynamics)
- **Literature Reference:** Williamson et al., "A standard test set for numerical approximations to the shallow water equations in spherical geometry," *J. Comput. Phys.* 102 (1992), 211–224
- **Implementation:** `src/ham/sim/fields.py:49–70`
  ```python
  term1 = -omega * z                         # −ω sin(φ)
  term2 = K * (rho_xy ** R) * z * cos_R_lon  # K cos^R(φ) sin(φ) cos(Rλ)
  ```
- **Verdict:** CORRECT
- **Notes:** With $z = x_3 = \sin\phi$ and $\rho_{xy} = \sqrt{x_1^2 + x_2^2} = \cos\phi$, the stream function is
  $$\psi = -\omega\sin\phi + K\cos^R\!\phi\;\sin\phi\;\cos(R\lambda)$$
  which is the standard Rossby–Haurwitz form. The longitude phase $\cos(R\lambda)$ is computed via de Moivre's theorem $\operatorname{Re}((x_1+i\,x_2)^R / |x_1+i\,x_2|^R)$, which is exact. ✓

---

### 3. Divergence-free flow from stream function

- **Spec Reference:** N/A
- **Literature Reference:** Standard result in spherical fluid dynamics
- **Implementation:** `src/ham/sim/fields.py:9–14`
  ```python
  v = jnp.cross(grad_psi, x)  # v = ∇ψ × x
  ```
- **Verdict:** CORRECT
- **Notes:** For a scalar $\psi : \mathbb{R}^3 \to \mathbb{R}$ and a point $x \in S^2$, $v = \nabla\psi \times x$ is:
  - Tangent to $S^2$: $\langle v, x \rangle = \langle \nabla\psi \times x, x \rangle = 0$ (scalar triple product with two equal vectors). ✓
  - Surface-divergence-free: the field is the Hodge dual of the exact 1-form $d\psi$. ✓

---

### 4. Passive advection with sphere projection

- **Spec Reference:** N/A (not a Finsler computation)
- **Implementation:** `examples/demo_trajectories.py:33–41`
  ```python
  p_next = p + dt * delta
  p_next = p_next / jnp.linalg.norm(p_next)
  ```
- **Verdict:** CORRECT
- **Notes:** Forward Euler + normalization retraction is a standard first-order integrator on $S^2$. Since the wind field is tangent to the sphere by construction (Finding 3), the retraction error is $O(dt^2)$. ✓

---

### 5. BVP solver geodesic (AVBD)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1, § 2.2
- **Implementation:** `examples/demo_trajectories.py:87–88`; solver at `src/ham/solvers/avbd.py:102–109`
  ```python
  v_in = metric.manifold.log_map(x_prev, x)
  E = metric.energy(x_prev, v_in) + metric.energy(x, v_out)
  ```
- **Verdict:** CORRECT
- **Notes:** The AVBD solver minimizes discrete Finsler energy $\sum_i E(x_i, \log_{x_i}(x_{i+1}))$ over interior path vertices, which converges to the continuous action functional as $n_\text{steps} \to \infty$. The use of $\log$-map velocities and manifold-projected gradient descent is standard. ✓

---

### 6. Verification shot — velocity scaling

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (geodesic equation)
- **Implementation:** `examples/demo_trajectories.py:98`
  ```python
  v_optimal_approx = traj_bvp.vs[0] * 40.0
  ```
- **Verdict:** WARNING
- **Notes:** `traj_bvp.vs[0] = \log_{x_0}(x_1)$ is the discrete velocity of the first segment of the 40-step BVP path. Multiplying by $n_\text{steps} = 40$ recovers the total initial velocity under the assumption of **uniform parameterisation** ($F(\gamma_i, \dot\gamma_i) \approx \text{const}$). For a Randers metric (asymmetric $F$), the BVP solver does not enforce uniform speed, so the scaling is approximate.

  Additionally, the `ExponentialMap` integrates with `t_max = 1.0` by default, so the geodesic traverses Finsler length $F(x_0, v_0) \cdot 1.0 = 40\,F(x_0, \text{vs}[0])$ (by 1-homogeneity). This matches the BVP total length only if parameterisation is uniform. The comment "rough scaling" correctly flags this, but the comparison metric (mean pointwise deviation) is sensitive to parameterisation mismatch even when the geometric paths coincide.

  **Recommended Action:** Consider using the full arc-length reparameterisation of the BVP path to extract a more accurate initial velocity, or document the expected magnitude of the scaling error.

---

### 7. Effective wind mismatch: Randers vs passive advection

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo causality constraint $\|W\|_h < 1$)
- **Implementation:** `examples/demo_trajectories.py:77` vs `examples/demo_trajectories.py:90`; squashing at `src/ham/geometry/zoo.py:100–104`
  ```python
  # Demo
  w_net = lambda x: 0.8 * wind_flow(x)       # for Randers
  wind_fn = lambda x: 0.8 * wind_flow(x)      # for passive advection

  # Inside Randers._get_zermelo_data:
  squash_factor = (max_speed * jnp.tanh(w_norm)) / (w_norm + 1e-8)
  W_safe = W_raw * squash_factor
  ```
- **Verdict:** WARNING
- **Notes:** The Randers metric internally squashes the wind via $\tanh$ to enforce $\|W\|_H < 1$. For $\|W_\text{raw}\|_H \approx 0.8$, the effective norm after squashing is
  $$\|W_\text{eff}\| = 0.8 \cdot \frac{\tanh(0.8)}{0.8} \approx 0.8 \times 0.831 \approx 0.664$$
  The passive advection trajectory uses the unsquashed wind $\|W\| = 0.8$. The Randers geodesic therefore "sees" a $\sim 17\%$ weaker wind than the passive tracer. The physical comparison ("optimal steering vs passive drift") is meaningful only if both experience the same external flow.

  **Recommended Action:** Either (a) pass the pre-squashed wind to the passive advection integrator, or (b) document that the comparison involves different effective wind magnitudes and explain why.

---

### 8. Start / end geometry — misleading comment

- **Spec Reference:** N/A
- **Implementation:** `examples/demo_trajectories.py:81–83`
  ```python
  start = jnp.array([1.0, 0.0, 0.0])
  end   = jnp.array([0.0, 0.0, 1.0])
  # comment on line 83: "almost antipodal"
  ```
- **Verdict:** NOTE
- **Notes:** The geodesic distance on the unit sphere between $(1,0,0)$ and $(0,0,1)$ is
  $$d = \arccos(\langle \text{start}, \text{end}\rangle) = \arccos(0) = \frac{\pi}{2} \approx 1.57$$
  which is a quarter great circle, not "almost antipodal" (that would be $d \approx \pi$). The computation is unaffected, but the comment is incorrect.

  **Recommended Action:** Change the comment to "quarter great-circle apart" or similar.

---

### 9. `path_length` — Euclidean chord lengths vs arc lengths

- **Spec Reference:** N/A
- **Implementation:** `examples/demo_trajectories.py:58–59`
  ```python
  diffs = jnp.diff(xs, axis=0)
  return float(jnp.sum(jnp.linalg.norm(diffs, axis=-1)))
  ```
- **Verdict:** NOTE
- **Notes:** This computes the sum of Euclidean chord lengths $\sum_i \|x_{i+1} - x_i\|$ rather than the intrinsic (great-circle) arc lengths $\sum_i \arccos(x_i \cdot x_{i+1})$. The two agree to $O(\Delta s^2)$ per segment. The output is explicitly labelled "Euclidean length," so no confusion arises. The Randers energy from the BVP solver (printed alongside) provides the intrinsic cost.

---

### 10. Geodesic spray and ODE integration

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1, § 2.2
- **Implementation:** `src/ham/geometry/metric.py:43–62` (spray), `src/ham/solvers/geodesic.py:33–50` (RK4 step)
- **Verdict:** CORRECT
- **Notes:** The spray is computed via the linear system
  $$g_{ij}\,\ddot{x}^j = \nabla_{x^i} E - \frac{\partial^2 E}{\partial v^i \partial x^k}\,v^k$$
  implemented as `acc = solve(Hess_v(E), ∇_x E − Jac_x(∇_v E)·v)`, yielding $\text{acc} = \ddot{x} = -2G$. Then `spray = −\tfrac{1}{2}\,\text{acc} = G^i$ and `geod_acceleration = −2G$. The IVP integrator solves $\dot{x}=v,\;\dot{v}=-2G$ via RK4 with manifold projection, matching the geodesic equation $\ddot{x}^i + 2G^i = 0$. ✓

---

### 11. Sphere exponential and logarithmic maps

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 (Geometric Hierarchy — Sphere)
- **Implementation:** `src/ham/geometry/surfaces.py:60–78` (exp), `src/ham/geometry/surfaces.py:80–98` (log)
- **Verdict:** STRONG
- **Notes:** The exponential map $\gamma(1) = \cos\theta\,x + \frac{\sin\theta}{\theta}\,v$ with $\theta = \|v\|/r$ is exact. The log map uses $\arccos$-based inversion with Taylor expansion for small $\theta$. Both correctly handle the sphere radius $r$. This is clean, analytically correct code.

---

## Open Questions

1. **Uniform parameterisation assumption (Finding 6):** Is the AVBD solver known to produce approximately uniformly parameterised geodesics? If not, the verification-shot comparison is unreliable for quantitative claims.
2. **Wind squashing impact on physical interpretation (Finding 7):** Was the $\tanh$ squashing intended to also be applied to the passive tracer, or is the mismatch a deliberate modelling choice (e.g., the passive tracer "feels" the full wind but the vessel has limited interaction with it)?
