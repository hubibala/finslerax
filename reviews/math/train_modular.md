# Math Review: train_modular
**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary
The modular training script `src/ham/bio/train_modular.py` is primarily a *configuration/orchestration* file that composes geometric loss functions and a two-phase training pipeline. It contains no novel mathematical formulae itself, but its correctness depends on the mathematical validity of the objects it assembles. The pipeline structure (block coordinate descent over VAE parameters and metric parameters) is mathematically sound. Two **WARNING**-level issues are identified in the loss functions it invokes: an inner-product mismatch in directional alignment losses, and a subtle convention point regarding the KL divergence approximation. No **CRITICAL** issues found.

## Formula-by-Formula Audit

### 1. GeodesicSprayLoss — velocity composition `dot_z = u_lat + W`
- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Navigation)
- **Literature Reference:** Bao, Chern, Shen — *An Introduction to Riemann-Finsler Geometry*, Chapter 11
- **Implementation:** `src/ham/training/losses.py:79–84`
  ```python
  dot_z = u_lat + W
  spray_vec = model.metric.spray(z_mean, dot_z)
  spray_norm = model.metric.inner_product(z_mean, dot_z, spray_vec, spray_vec)
  ```
- **Verdict:** OK
- **Notes:** In Zermelo navigation, the total velocity of the navigating particle is $\dot{z} = u + W$ where $u$ is the control (latent RNA velocity) and $W$ is the drift (wind). The spray $G^i(z, \dot{z})$ is then evaluated at this total velocity. Penalising $\|G\|_{g(z,\dot{z})}^2 = g_{ij}(z,\dot{z}) G^i G^j$ correctly encourages geodesic motion since geodesics satisfy $G^i = 0$. The norm is taken in the fundamental tensor at the reference direction $\dot{z}$, which is the canonical choice in Finsler geometry.

### 2. GeodesicSprayLoss — underlying spray implementation
- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1–2.2
- **Literature Reference:** Bao, Chern, Shen — Chapter 5, Proposition 5.4.1
- **Implementation:** `src/ham/geometry/metric.py:42–65`
  ```python
  rhs = grad_x - mixed_term  # ∇_x E − Jac_x(∇_v E) · v
  acc = jnp.linalg.solve(hess_v + 1e-4 * jnp.eye(x.shape[0]), rhs)
  return -0.5 * acc
  ```
- **Verdict:** OK
- **Notes:** The implementation solves $\text{Hess}_v(E) \cdot (-2G) = \nabla_x E - \text{Jac}_x(\nabla_v E) \cdot v$, giving $G = -\frac{1}{2}\,g^{-1}(\nabla_x E - \text{Jac}_x(\nabla_v E) \cdot v)$. This is equivalent to the standard formula $G^i = \frac{1}{2}g^{il}\bigl(\frac{\partial^2 E}{\partial v^l \partial x^k}v^k - \frac{\partial E}{\partial x^l}\bigr)$, which follows from $2E = F^2$ applied to the Bao-Chern-Shen spray formula. Matches § 2.2 of the spec. Note: § 2.1 of the spec writes the coefficient as $\frac{1}{4}$ with $(2\cdot\text{mixed} - \text{grad}_x)$, which should read $\frac{1}{4}(2\cdot\text{mixed} - 2\cdot\text{grad}_x)$ — a typo in the spec, not the code.

### 3. ZermeloAlignmentLoss — inner product for directional alignment
- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Randers cost formula)
- **Implementation:** `src/ham/training/losses.py:59–68`, invoked in `train_modular.py:79`
  ```python
  norm_w = model.manifold._minkowski_norm(W)
  norm_v = model.manifold._minkowski_norm(u_lat)
  w_dir = W / jnp.maximum(norm_w, 1e-6)[..., None]
  v_dir = u_lat / jnp.maximum(norm_v, 1e-6)[..., None]
  return -model.manifold._minkowski_dot(w_dir, v_dir) * self.weight
  ```
- **Verdict:** WARNING
- **Notes:** The loss measures cosine similarity between $W$ and $u_{\text{lat}}$ using the ambient **Minkowski** inner product $\langle \cdot, \cdot \rangle_\eta$ (restricted to the tangent space). However, in the Zermelo-Randers framework, the wind's effect on cost is governed by $\langle W, v \rangle_H$ (the learned Riemannian sea inner product). The cost reduction from a favourable wind is:
  $$F(z,v) = \frac{\sqrt{\lambda \|v\|_H^2 + \langle W,v\rangle_H^2} - \langle W,v\rangle_H}{\lambda}$$
  so maximising $\langle W, v \rangle_H$ — not $\langle W, v \rangle_\eta$ — minimises $F$. When $H$ departs significantly from the Minkowski restriction, these two alignment criteria diverge. During Phase 1 (metric frozen at random init), this mismatch is harmless since both are arbitrary. During joint or metric-only training, it could guide the encoder toward a suboptimal alignment.
  
  **Recommended Action:** Consider computing alignment as $W^T H v / (\|W\|_H \|v\|_H)$ to be consistent with the Randers cost. Alternatively, document the choice explicitly as a stable-target design decision.

### 4. ContrastiveAlignmentLoss — inner product and log map
- **Spec Reference:** `spec/MATH_SPEC.md` § 5
- **Implementation:** `src/ham/training/losses.py:121–129`, invoked in `train_modular.py:92`
  ```python
  v_tan = model.manifold.log_map(parent_z, child_z)
  align_score = -model.manifold._minkowski_dot(W_out, v_tan)
  ```
- **Verdict:** WARNING
- **Notes:** Same inner-product mismatch as Finding 3. Additionally, `log_map` computes the initial tangent vector of the geodesic under the **Hyperboloid's natural Riemannian metric** (Levi-Civita geodesic), not the Randers geodesic. The Randers geodesic direction from parent to child would require solving a BVP, which is expensive. Using the Riemannian log map is a reasonable first-order approximation (both directions coincide when $W \to 0$), but the alignment target is approximate.
  
  **Recommended Action:** Document that this uses the base-manifold geodesic direction as a proxy. Consider switching `_minkowski_dot` to the $H$-inner product for consistency with the Randers cost.

### 5. KLDivergenceLoss — Wrapped Normal KL approximation
- **Spec Reference:** Not in `spec/MATH_SPEC.md` (VAE-specific)
- **Literature Reference:** Mathieu et al. (2019), "Continuous Hierarchical Representations with Poincaré VAEs" (arXiv:1901.06033)
- **Implementation:** `src/ham/bio/vae.py:37–39`, invoked via `train_modular.py:78`
  ```python
  kl = -jnp.log(self.scale + 1e-6) + (self.scale**2) / 2.0 - 0.5
  return jnp.sum(kl, axis=-1)
  ```
- **Verdict:** WARNING
- **Notes:** The standard Gaussian KL divergence for $\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0,1)$ is:
  $$KL = -\log\sigma + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}$$
  The implementation omits the $\mu^2/2$ term. For Wrapped Normal distributions on manifolds, this is a known approximation: the "location" contribution is implicitly handled by the wrapping (parallel transport of samples from origin to $\mu$), and the KL is approximated as a scale-only divergence. This follows conventions in the hyperbolic VAE literature (arXiv:1901.06033). The approximation is valid when the curvature correction to the Jacobian of the exponential map is small (i.e., $\mu$ is near the origin or the manifold is nearly flat locally). For points far from the origin on a highly curved manifold, this underestimates the true KL.
  
  **Recommended Action:** No code change needed, but consider documenting the approximation and its regime of validity.

### 6. MetricAnchorLoss — Frobenius regularization of $H$
- **Spec Reference:** Not in `spec/MATH_SPEC.md`
- **Implementation:** `src/ham/training/losses.py:139–142`, invoked in `train_modular.py:93`
  ```python
  I = jnp.eye(dim)
  return jnp.mean((H_out - I)**2) * self.weight
  ```
- **Verdict:** OK
- **Notes:** Computes $\frac{1}{d^2}\|H - I\|_F^2$, a standard Frobenius-norm regulariser anchoring the Riemannian sea metric to the identity. Prevents degenerate solutions (rank-deficient $H$ or exploding eigenvalues). Mathematically, this encourages the Randers metric to stay "close to Euclidean" in the sea component, which is a valid inductive bias.

### 7. MetricSmoothnessLoss — Jacobian penalty on $W$
- **Spec Reference:** Not in `spec/MATH_SPEC.md`
- **Implementation:** `src/ham/training/losses.py:148–155`, invoked in `train_modular.py:94`
  ```python
  jac = jax.jacfwd(get_w_single)(parent_z)
  return jnp.mean(jac**2) * self.weight
  ```
- **Verdict:** OK
- **Notes:** Computes $\frac{1}{d^2}\|\nabla_z W\|_F^2$ via forward-mode AD. Standard Tikhonov/Sobolev regularisation for spatial smoothness of the wind field. The Jacobian is computed through the causality squasher ($\tanh$-based norm clamping in `_get_zermelo_data`), so it regularises the *constrained* wind field. Mathematically sound.

### 8. Randers metric_fn (used transitively by all losses)
- **Spec Reference:** `spec/MATH_SPEC.md` § 5
- **Literature Reference:** Bao, Chern, Shen — Chapter 11, §11.1
- **Implementation:** `src/ham/geometry/zoo.py:131–141`
  ```python
  discriminant = lam * v_sq_h + W_dot_v**2
  cost = (jnp.sqrt(jnp.maximum(discriminant, 0.0) + 1e-9) - W_dot_v) / lam
  ```
- **Verdict:** OK
- **Notes:** Implements the Zermelo-Randers cost:
  $$F(x,v) = \frac{\sqrt{\lambda\|v\|_H^2 + \langle W,v\rangle_H^2} - \langle W,v\rangle_H}{\lambda}$$
  with $\lambda = 1 - \|W\|_H^2 > 0$ (guaranteed by the causality squasher). Matches the spec exactly. The $+1e\text{-}9$ inside the square root provides $\epsilon$-regularisation per § 6.1.

### 9. Phase structure — block coordinate descent
- **Spec Reference:** Not in `spec/MATH_SPEC.md` (optimisation strategy)
- **Implementation:** `src/ham/bio/train_modular.py:66–99`
- **Verdict:** OK
- **Notes:** Phase 1 trains encoder/decoder with frozen metric; Phase 2 trains metric with frozen encoder/decoder. This is block coordinate descent on the joint objective $\mathcal{L}(\theta_{VAE}, \theta_{metric})$. Convergence of BCD requires each block sub-problem to have a unique minimum (or at least to decrease the objective), which is generally satisfied here since each phase uses convex-in-parameters neural network losses with Adam optimisation. No mathematical issues.

### 10. Velocity projection via JVP (project_control)
- **Spec Reference:** Not in `spec/MATH_SPEC.md`
- **Literature Reference:** Standard differential geometry (pushforward of smooth maps)
- **Implementation:** `src/ham/bio/vae.py:93–103`
  ```python
  z_mean, u_lat = jax.jvp(mean_fn, (x,), (v_rna,))
  u_lat = self.manifold.to_tangent(z_mean, u_lat)
  ```
- **Verdict:** OK
- **Notes:** Computes the pushforward $u_{\text{lat}} = Df(x) \cdot v_{\text{RNA}}$ where $f$ is the encoder mean map. This is the correct differential-geometric construction for projecting a data-space velocity to latent space. The subsequent `to_tangent` ensures the result lies in $T_{z_{\text{mean}}} M$.

## Open Questions

1. **Inner product choice in alignment losses:** The use of the ambient Minkowski inner product rather than the learned $H$-inner product for directional alignment (Findings 3, 4) is defensible as a design choice for training stability (fixed alignment target vs. co-evolving one). However, it introduces a mathematical inconsistency with the Randers cost function. Should this be treated as an approximation or corrected? Requires human expert judgment on the stability-vs-accuracy trade-off.

2. **Wrapped Normal KL approximation validity range:** The $\mu$-free KL approximation (Finding 5) is standard in the literature but may underestimate the true KL for encodings far from the manifold origin. Is this acceptable for the biological datasets targeted by HAM, or should a curvature-corrected KL (e.g., via the log-determinant of the exponential map Jacobian) be investigated?

3. **Spec formula typo in § 2.1:** The spray coefficient formula in the spec reads $G^i = \frac{1}{4}g^{il}(2\frac{\partial^2 E}{\partial v^l \partial x^k}v^k - \frac{\partial E}{\partial x^l})$. The correct formula (consistent with §2.2 and the code) should have a factor of 2 on the $\frac{\partial E}{\partial x^l}$ term: $G^i = \frac{1}{4}g^{il}(2\frac{\partial^2 E}{\partial v^l \partial x^k}v^k - 2\frac{\partial E}{\partial x^l})$. Should the spec be corrected?
