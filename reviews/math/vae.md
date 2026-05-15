# Math Review: vae (bio/vae.py)

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Major Issues.** The file implements a Geometric VAE with a Wrapped Normal posterior on a Finsler manifold. The reparameterization trick (sample in tangent space, parallel transport, exp map) is structurally correct. However, one **CRITICAL** error exists: the KL divergence formula computes only the tangent-space divergence $D_{KL}(\mathcal{N}(0,\Sigma)\|\mathcal{N}(0,I))$, omitting both (a) the mean-dependent geodesic distance term $\frac{1}{2}d(\mu,o)^2$ and (b) the log-Jacobian correction from the exponential map. Even in the flat Euclidean limit, this reduces to the wrong formula. The monolithic `loss_fn` inherits two additional **WARNING**-level geometric issues from the loss design (spray-as-flatness regularizer, Minkowski-specific alignment).

---

## Formula-by-Formula Audit

### 1. `WrappedNormal.sample` — Reparameterization Trick (lines 24–39)

- **Spec Reference:** N/A (not in `MATH_SPEC.md`; this is a variational inference construction).
- **Literature Reference:** Nagano et al. 2019 "A Wrapped Normal Distribution on Hyperbolic Space for Hierarchical Representation Learning" (arXiv:1902.02992); Mathieu et al. 2019 "Continuous Hierarchical Representations with Poincaré VAEs" (arXiv:1901.06033).
- **Implementation:**
  1. Sample $v_{\text{flat}} \sim \mathcal{N}(0, \sigma^2 I_{d})$ in $\mathbb{R}^d$ (intrinsic dim), line 26.
  2. Embed as tangent vector at origin $o$: for Hyperboloid $o=(1,0,...,0)$, $v_o=(0, v_{\text{flat}})$; for Sphere $o=(0,...,0,r)$, $v_o=(v_{\text{flat}}, 0)$, lines 28–36.
  3. Parallel transport $v_o$ from $o$ to $\mu$: $w = \text{PT}_{o \to \mu}(v_o)$, line 38.
  4. Exponential map: $z = \exp_\mu(w)$, line 39.

- **Verdict:** OK
- **Notes:**
  Tangent-space membership is verified:
  - Hyperboloid: $\langle o, v_o \rangle_L = -1 \cdot 0 + 0 = 0$ ✓
  - Sphere: $o \cdot v_o = r \cdot 0 = 0$ ✓

  The construction correctly implements the wrapped normal reparameterization: sample in the tangent space at a canonical origin, parallel-transport the sample to the posterior mean, then apply the exponential map. This is the standard scheme from the literature.

---

### 2. `WrappedNormal.kl_divergence_std_normal` — KL Divergence (lines 41–43)

- **Spec Reference:** N/A (variational inference, not in `MATH_SPEC.md`).
- **Literature Reference:** Nagano et al. 2019 (arXiv:1902.02992, §3.3); Kingma & Welling 2014 (arXiv:1312.6114).
- **Implementation:**
  ```python
  kl = -jnp.log(self.scale + 1e-6) + (self.scale**2) / 2.0 - 0.5
  return jnp.sum(kl, axis=-1)
  ```
  This computes, per dimension $i$:
  $$
  \hat{D}_i = -\log\sigma_i + \frac{\sigma_i^2}{2} - \frac{1}{2}
  $$

- **Verdict:** CRITICAL
- **Notes:**

  **Issue 1 — Missing mean term (incorrect even in the Euclidean limit).**
  The standard KL divergence between $q = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p = \mathcal{N}(0, I)$ is:
  $$
  D_{KL}(q \| p) = \sum_i \left[-\log\sigma_i + \frac{\sigma_i^2 + \mu_i^2}{2} - \frac{1}{2}\right]
  $$
  The code omits the $\frac{1}{2}\|\mu\|^2$ contribution entirely. For the wrapped normal, the analogous term is the squared geodesic distance $\frac{1}{2}d^2(\mu, o)$, where $o$ is the base point of the prior. Its absence means the model has **no regularization pressure on the mean** — the posterior mean $\mu$ can drift arbitrarily far from the prior's center without any KL penalty.

  For the EuclideanSpace manifold (where exp, log, and parallel transport are all identity), the wrapped normal reduces to an ordinary Gaussian, and this formula gives $D_{KL}(\mathcal{N}(0,\sigma^2) \| \mathcal{N}(0, I))$ instead of the correct $D_{KL}(\mathcal{N}(\mu,\sigma^2) \| \mathcal{N}(0,I))$.

  **Issue 2 — Missing log-Jacobian of exponential map (incorrect on curved manifolds).**
  On a Riemannian manifold, the density of the wrapped normal on $M$ relates to the tangent-space density by the change-of-variables formula:
  $$
  \log p_{\mathcal{W}}(z) = \log p_v(v) - \log \left|\det J_{\exp_\mu}(v)\right|
  $$
  For the hyperboloid (Nagano et al. 2019, Eq. 9):
  $$
  \log \left|\det J_{\exp_o}(v)\right| = (n-1)\log\frac{\sinh\|v\|_L}{\|v\|_L}
  $$
  These Jacobian determinant terms from both $q$ and $p$ must appear in the KL. The code omits them entirely.

  **Consequence:** The ELBO $\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - D_{KL}(q\|p)$ as implemented is **not** a valid lower bound on $\log p(x)$ for any manifold.

  **Recommended Action:** Implement the full wrapped normal KL divergence. At minimum, add the geodesic distance term $+\frac{1}{2}d^2(\mu, o)$. For geometrically rigorous training on curved manifolds, also include the log-Jacobian correction terms as in Nagano et al. 2019 or use Monte Carlo KL estimation.

---

### 3. `GeometricVAE._get_dist` — Encoder Parameterization (lines 75–85)

- **Spec Reference:** N/A.
- **Implementation:**
  Encoder outputs $d_{\text{amb}} + d_{\text{int}}$ scalars. The first $d_{\text{amb}}$ are projected onto $M$ to give $\mu$. The remaining $d_{\text{int}}$ parameterize the tangent-space scale via $\sigma = \text{softplus}(\cdot) + 10^{-4}$.
- **Verdict:** OK
- **Notes:** The split is dimensionally correct: $\mu$ needs $d_{\text{amb}}$ coordinates (ambient representation), while $\sigma$ needs $d_{\text{int}}$ coordinates (one per tangent-space degree of freedom). The softplus activation ensures $\sigma > 0$.

  Minor naming inconsistency: the variable `log_scale` (line 80) is passed through `softplus`, not `exp`. Softplus is $\text{softplus}(x) = \log(1 + e^x) \neq e^x$. The name suggests the output of the network is $\log\sigma$, but the transformation applied is not $\exp$. This is a **NOTE**-level notation mismatch only; the math is sound.

---

### 4. `GeometricVAE.project_control` — Pushforward of RNA Velocity (lines 93–104)

- **Spec Reference:** N/A (application-specific; uses differential geometry of the encoder map).
- **Implementation:**
  ```python
  z_mean, u_lat = jax.jvp(mean_fn, (x,), (v_rna,))
  u_lat = self.manifold.to_tangent(z_mean, u_lat)
  ```
  Computes the differential (pushforward) of the encoder mean map $\mu_\theta : \mathbb{R}^{d_{\text{data}}} \to M$:
  $$
  u_{\text{lat}} = \Pi_{T_{\mu}M}\left(d\mu_\theta(x) \cdot v_{\text{rna}}\right)
  $$
- **Verdict:** OK
- **Notes:** The JVP correctly computes the Jacobian-vector product through the differentiable encoder-then-project pipeline. The subsequent `to_tangent` projection is necessary because the JVP of the ambient projection may produce a vector with a small normal component due to the chain rule through `project`. This is mathematically sound.

---

### 5. `GeometricVAE.loss_fn` — Reconstruction Loss (line 112)

- **Spec Reference:** N/A (standard VAE).
- **Implementation:** `recon_loss = jnp.mean((x - x_rec)**2)`
- **Verdict:** OK
- **Notes:** Standard MSE reconstruction. No geometric content.

---

### 6. `GeometricVAE.loss_fn` — KL Loss (line 113)

- **Spec Reference:** See §2 above.
- **Implementation:** `kl_loss = jnp.mean(dist.kl_divergence_std_normal())`
- **Verdict:** CRITICAL
- **Notes:** Inherits the critical issues from `WrappedNormal.kl_divergence_std_normal()` (§2). The KL weight `1e-4` (line 131) further diminishes the already incomplete KL term, making the posterior essentially unregularized.

---

### 7. `GeometricVAE.loss_fn` — Alignment Loss (lines 117–125)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization).
- **Implementation:**
  ```python
  w_dir = W / jnp.maximum(norm_w, 1e-6)[..., None]
  v_dir = u_lat / jnp.maximum(norm_v, 1e-6)[..., None]
  align_loss = -self.manifold._minkowski_dot(w_dir, v_dir)
  ```
  Computes $-\langle \hat{W}, \hat{u} \rangle_L$ using Minkowski inner product and Minkowski norm for normalization.
- **Verdict:** WARNING
- **Notes:** The alignment uses `_minkowski_norm` and `_minkowski_dot`, which are methods of the `Hyperboloid` class. This is only geometrically meaningful when the base manifold is a hyperboloid. For the Sphere (Euclidean inner product) or a general learned manifold, the Minkowski inner product is the wrong bilinear form. The wind vector $W$ in the Zermelo construction lives in the tangent space with inner product induced by the Riemannian "sea" metric $h_{ij}$, not the ambient Minkowski metric. This conflates two different inner product structures.

  Furthermore, if the manifold is `EuclideanSpace`, calling `_minkowski_dot` would fail at runtime since that method does not exist on `EuclideanSpace`.

---

### 8. `GeometricVAE.loss_fn` — Spray Loss (lines 127–129)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (Spray Coefficients).
- **Implementation:**
  ```python
  dot_z = u_lat + W
  spray_vec = self.metric.spray(z_mean, dot_z)
  spray_loss = self.metric.inner_product(z_mean, dot_z, spray_vec, spray_vec)
  ```
  Computes $\|G(z, \dot{z})\|^2_{g(z,\dot{z})}$ where $\dot{z} = u_{\text{lat}} + W$.
- **Verdict:** WARNING
- **Notes:** The spray coefficients $G^i(x, v)$ are a property of the **geometry** at a point $(x,v) \in TM$, not a residual of a trajectory. Along a geodesic, $\ddot{x}^i + 2G^i(x, \dot{x}) = 0$, so $G^i$ is generically nonzero. Penalizing $\|G\|^2 \to 0$ pushes the learned metric toward Euclidean flatness ($G^i = 0$ iff the space is flat), not the trajectory toward geodesicity. A correct geodesic-deviation loss would penalize $\|\ddot{z} + 2G(z,\dot{z})\|^2$, but this requires a trajectory (not a single point).

  The composition $\dot{z} = u_{\text{lat}} + W$ is consistent with Zermelo navigation: $u_{\text{lat}}$ is the "boat velocity" and $W$ the wind, so $\dot{z}$ is the total velocity in the ambient frame. The spray is correctly evaluated at total velocity per the geodesic ODE $\ddot{x}^i + 2G^i(x,\dot{x}) = 0$.

  The inner product $g_{ij}(z,\dot{z}) G^i G^j$ correctly uses the fundamental tensor evaluated at the reference direction $\dot{z}$, consistent with `spec/MATH_SPEC.md` § 1.1.

---

### 9. `GeometricVAE.loss_fn` — Total Loss Weighting (line 131)

- **Spec Reference:** N/A.
- **Implementation:**
  ```python
  total_loss = recon_loss + 1e-4 * kl_loss + 0.1 * align_loss + 1.0 * spray_loss
  ```
- **Verdict:** WARNING
- **Notes:** The KL weight of $10^{-4}$ is four orders of magnitude below the reconstruction weight. Combined with the already incomplete KL formula (§2), this means the variational posterior receives negligible regularization. The ELBO interpretation is doubly violated: the KL term is both mathematically incorrect and numerically suppressed. While $\beta$-VAE weighting is a valid design choice, $\beta = 10^{-4}$ effectively eliminates the KL and makes the model a deterministic autoencoder with geometric regularizers.

---

### 10. Hyperboloid Parallel Transport used by `WrappedNormal.sample` (surfaces.py:381–386)

- **Spec Reference:** `spec/MATH_SPEC.md` § 3.2 (Parallel Transport — Berwald connection).
- **Literature Reference:** Nickel & Kiela 2018 "Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry" (arXiv:1806.03417).
- **Implementation:**
  ```python
  def parallel_transport(self, x, y, v):
      xy = self._minkowski_dot(x, y)
      yv = self._minkowski_dot(y, v)
      denom = jnp.maximum(1.0 - xy, 2.0)
      scale = yv / denom
      return v + scale[..., None] * (x + y)
  ```
  Implements:
  $$
  \Gamma_{x \to y}(v) = v + \frac{\langle y, v \rangle_L}{1 - \langle x, y \rangle_L}(x + y)
  $$
- **Verdict:** OK
- **Notes:** Verified by explicit computation. For $x = (1,0,0)$, $y = (\cosh\theta, \sinh\theta, 0)$, and $v = (0,1,0) \in T_x\mathbb{H}^2$:

  $P(v) = (\sinh\theta, \cosh\theta, 0)$.

  Checks: $\langle y, P(v) \rangle_L = 0$ (tangent at $y$) ✓; $\langle P(v), P(v) \rangle_L = 1 = \langle v, v \rangle_L$ (norm-preserving) ✓.

  The denominator $1 - \langle x, y \rangle_L \geq 2$ since $\langle x, y \rangle_L \leq -1$ on $\mathbb{H}^n$, so the `jnp.maximum(..., 2.0)` is a safety clamp for the degenerate case $x = y$.

  **Caveat:** This is the Levi-Civita parallel transport on the hyperboloid, not the Berwald parallel transport of the Randers metric. For the purpose of sampling the wrapped normal (which uses the base Riemannian structure), Levi-Civita transport is the correct choice. However, this is a different notion of transport than `spec/MATH_SPEC.md` § 3.2.

---

## Open Questions

1. **Is the missing KL mean term intentional?** The code may be deliberately using only the scale component of the KL as a form of $\beta$-VAE with soft prior on variance only, relying on the geometric losses to regularize the mean. If so, this should be documented, as it breaks the ELBO interpretation.

2. **Manifold generality of `loss_fn`:** The `loss_fn` calls `_minkowski_dot` and `_minkowski_norm` which are Hyperboloid-specific. Is this function intended only for the Hyperboloid manifold, or should it work with Sphere/EuclideanSpace as well? If the latter, it will crash.

3. **Interaction between KL and spray loss:** With KL weight $10^{-4}$ and spray weight $1.0$, the geometric flatness regularizer dominates the variational regularizer by four orders of magnitude. Is this the intended balance? The spray loss pushes the geometry toward Euclidean flatness, which may conflict with the goal of learning non-trivial Finsler structure.
