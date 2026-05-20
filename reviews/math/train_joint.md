# Math Review: train_joint (bio/train_joint.py)

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Minor Issues.** The file implements a dual-phase training procedure: Phase 1 learns manifold structure (encoder/decoder + geometric losses) by delegating to `GeometricVAE.loss_fn`; Phase 2 learns the Randers wind field $W(x)$ via a contrastive alignment objective on lineage pairs. The mathematical structure of Phase 2 is largely sound — the log-map tangent direction, H-regularization, and smoothness penalty are correctly formulated. However, two **WARNING**-level issues exist: (1) the alignment loss uses the Minkowski inner product $\langle W, v \rangle_L$ instead of the Zermelo-consistent Riemannian inner product $\langle W, v \rangle_H$, which is only valid when $H \approx I$; (2) a single PRNG key is reused across all samples in a batch during Phase 2 encoding, producing correlated latent representations. Phase 1 inherits the **CRITICAL** KL divergence issue documented in the `vae.md` review.

---

## Formula-by-Formula Audit

### 1. `step_manifold` — Phase 1 Training Step (lines 13–31)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1, § 5.
- **Implementation:**
  ```python
  losses, stats = jax.vmap(m.loss_fn)(x_batch, v_batch, keys)
  return jnp.mean(losses), stats
  ```
  Delegates entirely to `GeometricVAE.loss_fn` (reviewed in `reviews/math/vae.md`).
- **Verdict:** OK (modulo inherited issues)
- **Notes:** The vmapped loss correctly averages over the batch. The gradient is taken via `eqx.filter_value_and_grad` with `has_aux=True`, which differentiates through the mean total loss while passing auxiliary per-loss statistics. No additional mathematical operations are introduced. **Inherited CRITICAL**: the KL divergence term within `loss_fn` omits the geodesic distance and log-Jacobian terms (see `reviews/math/vae.md` § 2).

---

### 2. `step_metric` / `contrastive_loss` — Log Map for Tangent Direction (line 44)

- **Spec Reference:** Hyperboloid log map; `spec/MATH_SPEC.md` § 4 (Geometric Hierarchy).
- **Literature Reference:** Nickel & Kiela 2018 (arXiv:1806.03417).
- **Implementation:**
  ```python
  v_tan = jax.vmap(m.manifold.log_map)(z_parent, z_child)
  ```
  Computes $v = \log_{z_p}(z_c) \in T_{z_p}M$, the tangent vector pointing from parent to child.
- **Verdict:** OK
- **Notes:** The log map returns a tangent vector $v$ such that $\exp_{z_p}(v) = z_c$. On the hyperboloid this uses the $\operatorname{arcsinh}$-based formula (see [surfaces.py](src/ham/geometry/surfaces.py#L353-L365)), which is correct. The tangent vector $v$ encodes both the direction and the geodesic distance $\|v\|_L = d(z_p, z_c)$ of the lineage transition. Using this as the biological "velocity" direction is the correct geometric construction.

---

### 3. `step_metric` / `contrastive_loss` — Alignment Loss (lines 56–61)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization).
- **Implementation:**
  ```python
  align_scores = -jax.vmap(m.manifold._minkowski_dot)(W_batch, v_tan)
  loss_align = jnp.mean(align_scores)
  ```
  Computes:
  $$
  \mathcal{L}_{\text{align}} = -\frac{1}{B}\sum_{b=1}^B \langle W(z_p^{(b)}),\, \log_{z_p^{(b)}}(z_c^{(b)}) \rangle_L
  $$
- **Verdict:** WARNING
- **Notes:**

  **Issue — Wrong inner product for Zermelo alignment.**
  In the Randers/Zermelo framework (`spec/MATH_SPEC.md` § 5), the wind $W$ lives in the tangent space equipped with the Riemannian "sea" metric $H$. The geometrically correct alignment quantity is $\langle W, v \rangle_H = W^T H v$, not $\langle W, v \rangle_L = -W_0 v_0 + \sum_i W_i v_i$. The Randers cost function:
  $$
  F(x, v) = \frac{\sqrt{\lambda \|v\|_H^2 + \langle W, v \rangle_H^2} - \langle W, v \rangle_H}{\lambda}
  $$
  is reduced by a positive $\langle W, v \rangle_H$, so maximizing the $H$-inner product (not the Minkowski inner product) is the correct objective for aligning the wind with the data velocity.

  When $H \approx I$ (which the H-regularization in § 5 below enforces), the two inner products coincide on the spatial components. However, the Minkowski inner product also introduces a $-W_0 v_0$ term from the ambient time coordinate that has no counterpart in the intrinsic Riemannian metric. For tangent vectors on the hyperboloid, this term is nonzero and geometrically spurious.

  **Recommended Action:** Replace the Minkowski alignment with the Riemannian alignment: `align = -v_tan @ H @ W`, using the learned $H$ from `_get_zermelo_data`. This couples the alignment objective to the metric structure consistently.

---

### 4. `step_metric` / `contrastive_loss` — Zermelo Data Retrieval (lines 49–54)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization).
- **Implementation:**
  ```python
  H, W = get_fields_vmap(z_parent)
  ```
  where `get_fields` calls `m.metric._get_zermelo_data(pt)`, returning the learned Riemannian tensor $H(z)$ and the causality-squashed wind $W(z)$.
- **Verdict:** OK
- **Notes:** The retrieved fields $(H, W)$ are the same objects used in the Randers metric $F(x,v)$ (`spec/MATH_SPEC.md` § 5). The `_get_zermelo_data` implementation (see [zoo.py](src/ham/geometry/zoo.py#L88-L129)) correctly:
  1. Symmetrizes $H$,
  2. Enforces positive definiteness via diagonal clamping,
  3. Squashes $\|W\|_H < 1 - \epsilon$ via tanh to maintain causality ($\lambda > 0$).

  The causality constraint is critical: if $\|W\|_H \geq 1$, the Randers metric loses strong convexity and $\lambda \leq 0$ invalidates the formula. The squashing guarantees this cannot happen.

---

### 5. `step_metric` / `contrastive_loss` — H-to-Identity Regularization (lines 63–67)

- **Spec Reference:** N/A (regularization, not in spec).
- **Implementation:**
  ```python
  dim = H_batch.shape[-1]
  I = jnp.eye(dim)
  loss_h_reg = jnp.mean((H_batch - I)**2)
  ```
  Computes $\mathcal{L}_{H} = \frac{1}{B}\sum_b \|H(z^{(b)}) - I\|_F^2$.
- **Verdict:** OK
- **Notes:** This Frobenius-norm penalty anchors the Riemannian component to the identity, preventing the degenerate mode $H \to 0$ (which would allow $\|W\|_H \to 0$ trivially even for large ambient $W$, collapsing the metric to a flat space). The penalty also implicitly prevents $H \to \infty$ (which would over-penalize all velocities). This is a standard practice for learned Riemannian metrics (see Arvanitidis et al. 2018 "Latent Space Oddity", arXiv:1710.11379).

  However, when $H \equiv I$, the Randers metric reduces to:
  $$
  F(x,v) = \frac{\sqrt{(1-\|W\|^2)\|v\|^2 + (W \cdot v)^2} - W \cdot v}{1 - \|W\|^2}
  $$
  where the norms/dots are Euclidean in the spatial coordinates. A strong anchor weight ($\alpha_H = 1.0$) thus limits the model's capacity to learn non-trivial Riemannian curvature, biasing toward a Randers metric with flat base metric and learned wind only.

---

### 6. `step_metric` / `contrastive_loss` — Jacobian Smoothness Penalty (lines 69–79)

- **Spec Reference:** N/A (regularization).
- **Implementation:**
  ```python
  jac_fn = jax.vmap(jax.jacfwd(get_w_single))
  jacobians = jac_fn(z_parent)
  loss_smooth = jnp.mean(jacobians**2)
  ```
  Computes $\mathcal{L}_{\text{smooth}} = \frac{1}{B}\sum_b \left\|\frac{\partial W}{\partial z}\bigg|_{z^{(b)}}\right\|_F^2$.
- **Verdict:** OK
- **Notes:** Penalizing the Frobenius norm of the Jacobian $\partial W^i / \partial z^j$ encourages spatially smooth vector fields. This is a Tikhonov-type regularizer on the gradient of $W$. Mathematically, it penalizes the first-order variation of the drift field, which prevents chaotic oscillations in $W$ that would make geodesic integration numerically unstable.

  The `jax.jacfwd` is the correct choice here: forward-mode AD is efficient when the input dimension $\leq$ output dimension, which holds for $W: \mathbb{R}^{d_{\text{amb}}} \to \mathbb{R}^{d_{\text{amb}}}$.

---

### 7. `step_metric` / `contrastive_loss` — Total Loss Weighting (lines 83–84)

- **Spec Reference:** N/A.
- **Implementation:**
  ```python
  total_loss = loss_align + 1.0 * loss_h_reg + 0.1 * loss_smooth
  ```
- **Verdict:** NOTE
- **Notes:** The alignment loss $\mathcal{L}_{\text{align}} = -\langle W, v \rangle_L$ is unbounded below (grows with $\|W\|$ and $\|v\|$), while $\mathcal{L}_H$ and $\mathcal{L}_{\text{smooth}}$ are both non-negative. The balance depends on the causality squashing limiting $\|W\|_H$, but the log-map magnitude $\|v_{\text{tan}}\|$ is unconstrained (it equals the geodesic distance between parent and child). For distant lineage pairs, the alignment term can dominate and overwhelm the regularizers. A normalized alignment (e.g., cosine similarity via $-\langle \hat{W}, \hat{v} \rangle$) would be more scale-invariant.

---

### 8. `GeometricTrainer.train` — Phase 2 Encoding (lines 167–172)

- **Spec Reference:** N/A (optimization procedure).
- **Implementation:**
  ```python
  key_p, key_c = jax.random.split(subkey, 2)
  z_parents = jax.lax.stop_gradient(
      jax.vmap(lambda x: self.model.encode(x, key_p))(x_parents))
  z_children = jax.lax.stop_gradient(
      jax.vmap(lambda x: self.model.encode(x, key_c))(x_children))
  ```
- **Verdict:** WARNING
- **Notes:**

  **Issue — Shared PRNG key across batch introduces correlated sampling.**
  `self.model.encode(x, key)` calls `WrappedNormal.sample(key)`, which draws a noise vector $\epsilon \sim \mathcal{N}(0, I)$ from the provided key. Since `key_p` is identical for all samples in the vmap, every sample in the batch receives the **same** noise realization $\epsilon$. Each sample still gets a different $z$ (because $z = \exp_{\mu_i}(\text{PT}(\sigma_i \odot \epsilon))$ differs via the per-sample $\mu_i, \sigma_i$), but the noise direction is perfectly correlated across the batch.

  Since `stop_gradient` prevents gradients from flowing through the encoder, this affects only the quality of the latent representations used to train the metric. The correlated noise biases the empirical distribution of $z$ values in each batch: all samples are perturbed in the same tangent-space direction relative to their respective means. Over multiple epochs and batches, different noise directions are sampled, so the bias averages out, but within each batch the metric sees a systematically shifted snapshot of the latent space.

  **Recommended Action:** Generate per-sample keys: `keys = jax.random.split(subkey, 2 * batch_size)`, then `key_parents = keys[:batch_size]`, `key_children = keys[batch_size:]`, and vmap over both data and keys.

---

### 9. `GeometricTrainer.train` — Stop-Gradient on Phase 2 Encodings (lines 170–172)

- **Spec Reference:** N/A (bi-level optimization strategy).
- **Implementation:**
  ```python
  z_parents = jax.lax.stop_gradient(...)
  z_children = jax.lax.stop_gradient(...)
  ```
- **Verdict:** OK
- **Notes:** Stopping gradients on the encoder output during Phase 2 decouples the metric-learning objective from the encoder parameters. This is the correct strategy for alternating optimization: Phase 1 learns the embedding $\mu_\theta(x)$; Phase 2 learns the Randers wind field $W_\phi(z)$ on the fixed latent space. Without `stop_gradient`, the alignment loss would distort the encoder to trivially align $W$ and $v$ by collapsing the latent space.

  One subtlety: the optimizer state (Adam moments) from Phase 1 persists into Phase 2. Since Phase 2 produces zero gradients for encoder/decoder parameters, their Adam first moments exponentially decay to zero while the second moments persist. This is harmless in a strict two-phase setup, but if the phases were interleaved, the stale optimizer state could cause instability.

---

### 10. Spray and Inner Product (used via `loss_fn`, evaluated in `metric.py`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (Spray), § 2.2 (JAX Implementation), § 1.1 (Fundamental Tensor).
- **Implementation (metric.py):**
  ```python
  rhs = grad_x - mixed_term
  acc = jnp.linalg.solve(hess_v + 1e-4 * jnp.eye(x.shape[0]), rhs)
  return -0.5 * acc
  ```
  Solves $g_{ij} \cdot (-2G^j) = \nabla_{x^i} E - \frac{\partial^2 E}{\partial v^i \partial x^k}v^k$ for $G$.
- **Verdict:** OK
- **Notes:** The implementation correctly solves the Euler-Lagrange linear system. From the E-L equations:
  $$
  g_{ik}\ddot{x}^k = \frac{\partial E}{\partial x^i} - \frac{\partial^2 E}{\partial v^i \partial x^k}v^k
  $$
  With $\ddot{x}^i = -2G^i$:
  $$
  G^i = -\frac{1}{2}g^{ij}\left(\frac{\partial E}{\partial x^j} - \frac{\partial^2 E}{\partial v^j \partial x^k}v^k\right) = \frac{1}{2}g^{ij}\left(\frac{\partial^2 E}{\partial v^j \partial x^k}v^k - \frac{\partial E}{\partial x^j}\right)
  $$
  This matches the standard Bao-Chern-Shen formula (eq. 2.3.6):
  $$
  G^i = \frac{1}{4}g^{il}\left(\frac{\partial^2 [F^2]}{\partial x^k \partial y^l}y^k - \frac{\partial [F^2]}{\partial x^l}\right) = \frac{1}{2}g^{il}\left(\frac{\partial^2 E}{\partial x^k \partial v^l}v^k - \frac{\partial E}{\partial x^l}\right)
  $$

  **Spec discrepancy (NOTE):** `spec/MATH_SPEC.md` § 2.1 writes:
  $$
  G^i = \frac{1}{4} g^{il}\left( 2 \frac{\partial^2 E}{\partial v^l \partial x^k} v^k - \frac{\partial E}{\partial x^l} \right)
  $$
  This has the factor of 2 on the mixed term but **not** on $\frac{\partial E}{\partial x^l}$, yielding $\frac{1}{4}(2M - N)$ instead of the correct $\frac{1}{4}(2M - 2N) = \frac{1}{2}(M - N)$. The spec § 2.1 formula has a **missing factor of 2** on the second term. The code and the spec § 2.2 implicit-solve formulation are both correct; only the explicit § 2.1 formula has this typo.

  Additionally, `spec/MATH_SPEC.md` § 2.2 writes $\text{Hess}_v(E) \cdot (2G) = \nabla_x E - \text{Jac}_x(\nabla_v E) \cdot v$, but the correct sign is $\text{Hess}_v(E) \cdot (-2G) = \nabla_x E - \text{Jac}_x(\nabla_v E) \cdot v$ (matching the code's own comment). The code is correct; the spec has a sign error.

---

## Open Questions

1. **Alignment inner product choice:** Should the Phase 2 alignment loss use the H-Riemannian inner product $\langle W, v \rangle_H$ instead of the Minkowski inner product $\langle W, v \rangle_L$? The current formulation is consistent only when the H-regularization keeps $H \approx I$. If the anchor weight is reduced to allow more expressive metrics, the alignment and the Randers cost function will disagree on what "aligned" means.

2. **Spray loss interpretation:** As noted in `reviews/math/vae.md` § 8, penalizing $\|G(z, \dot{z})\|_g^2 \to 0$ pushes toward flat geometry ($G \equiv 0$), not toward geodesic trajectories. Is this the intended regularization effect, or should it be replaced by a trajectory-based geodesic deviation loss?

3. **Normalization of alignment:** The unnormalized alignment $-\langle W, v \rangle_L$ scales with geodesic distance $d(z_p, z_c)$. Distant lineage pairs contribute disproportionately. Should a cosine-similarity formulation be used instead (as is done in `loss_fn` in `vae.py`)?

4. **Spec errata:** The spray formula in `spec/MATH_SPEC.md` § 2.1 has a missing factor of 2 on $\frac{\partial E}{\partial x^l}$, and § 2.2 has a sign error ($2G$ should be $-2G$). These should be corrected to prevent confusion for future contributors.
