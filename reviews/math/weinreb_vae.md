# Math Review: weinreb_vae

**Reviewer:** Math Reviewer Agent
**Date:** 2025-05-15
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Major Issues.** The training script is a well-structured multi-objective VAE pipeline, but two mathematically significant problems were identified. First, the KL divergence formula in the underlying `WrappedNormal` class is incomplete — it drops the mean-dependent term, which makes the ELBO an invalid variational bound. Second, the trajectory coherence loss receives mismatched data (random cells instead of lineage-matched day-2 cells), rendering its midpoint and direction constraints mathematically meaningless. Several minor issues around loss weighting and dead code are also noted.

---

## Formula-by-Formula Audit

### 1. KL Divergence — `WrappedNormal.kl_divergence_std_normal()`

- **Spec Reference:** Standard ELBO derivation; Kingma & Welling (2014) Appendix B.
- **Literature Reference:** Nagano et al. (2019) "A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based Learning" (arXiv:1902.02992), Appendix A.
- **Implementation:** `src/ham/bio/vae.py:43–44`
  ```python
  kl = -jnp.log(self.scale + 1e-6) + (self.scale**2) / 2.0 - 0.5
  return jnp.sum(kl, axis=-1)
  ```
- **Called from:** `examples/weinreb_vae.py:236` via `AnnealedKLLoss.__call__`.
- **Verdict:** **CRITICAL**
- **Notes:**
  The standard KL divergence for a diagonal Gaussian $q = \mathcal{N}(\mu, \sigma^2 I)$ against the standard normal prior $p = \mathcal{N}(0, I)$ is:

  $$KL(q \| p) = \sum_j \left[ -\log \sigma_j + \frac{\sigma_j^2 + \mu_j^2}{2} - \frac{1}{2} \right]$$

  The implementation is missing the $\mu_j^2 / 2$ term. Since this script uses `EuclideanSpace` (where `project` is identity and wrapped normal reduces to an ordinary Gaussian), the omission means the prior imposes **no penalty on the mean**. Encoder means can drift arbitrarily far from the origin without any KL cost.

  For the general wrapped normal on the hyperboloid, the correct KL contains a curvature-dependent correction (Nagano et al. 2019, Eq. 10):

  $$KL(q_{\mu,\Sigma} \| p_{o,I}) = KL_{\text{Euclidean}}(\mathcal{N}(0,\Sigma) \| \mathcal{N}(0,I)) + (n-1)\log\frac{\sinh d(o,\mu)}{d(o,\mu)}$$

  Neither the mean term nor the curvature correction is present.

  **Recommended Action:** For the Euclidean case, add `+ (self.mean**2) / 2.0` inside the sum. For the hyperbolic case, add the $\log(\sinh/d)$ correction from Nagano et al.

---

### 2. Reconstruction Loss — `ReconstructionLossDeterministic`

- **Spec Reference:** Standard Gaussian observation model $p(x|z) = \mathcal{N}(\text{dec}(z),\, I)$.
- **Implementation:** `examples/weinreb_vae.py:331–348`
  ```python
  loss_stoch = jnp.mean((x - v_decode(z_sample)) ** 2)
  loss_det   = jnp.mean((x - v_decode(z_mean))  ** 2)
  return (stochastic_weight * loss_stoch + deterministic_weight * loss_det) * self.weight
  ```
- **Verdict:** **WARNING**
- **Notes:**
  The standard ELBO uses *only* the stochastic path: $\mathbb{E}_{q(z|x)}[\|x - \text{dec}(z)\|^2]$. Adding the deterministic mean-path reconstruction is a valid regularization technique, but with the default 50/50 split (`stochastic_weight = deterministic_weight = 0.5`), the stochastic reconstruction is down-weighted by a factor of 2. This implicitly halves the effective reconstruction signal in the ELBO, which is equivalent to doubling the assumed observation noise variance. Combined with the missing $\mu^2$ term in the KL, the resulting objective is no longer a valid variational lower bound.

  **Recommended Action:** Document that this is a modified ELBO. If a proper variational bound is needed, use `stochastic_weight = 1.0, deterministic_weight = 0.0` (or add the deterministic term as a separate auxiliary loss with its own weight).

---

### 3. Trajectory Coherence Loss — `TrajectoryCoherenceLoss`

- **Implementation:** `examples/weinreb_vae.py:179–209`
- **Batch wiring:** `examples/weinreb_vae.py:551–561`
  ```python
  batch_main = (
      X[idx_main],          # x — RANDOM cells
      dataset.V[idx_main],  # v
      labels[idx_main],     # labels
      X[i4],                # x_day4 — from lineage triples
      X[i6],                # x_day6 — from lineage triples
  )
  ```
- **Verdict:** **CRITICAL**
- **Notes:**
  The loss docstring states "`batch[0] = x_day2`," and computes:

  $$\mathcal{L}_{\text{mid}} = \|z_4 - \tfrac{1}{2}(z_2 + z_6)\|^2, \qquad \mathcal{L}_{\text{dir}} = 1 - \cos\angle(z_4 - z_2,\; z_6 - z_2)$$

  However, `batch[0] = X[idx_main]` is a **random permutation** of all cells, while `batch[3]` and `batch[4]` are the day-4 and day-6 entries from lineage triples. The day-2 index `i2 = trip3[:, 0]` is computed at [line 548](examples/weinreb_vae.py#L548) but never used. The resulting loss computes midpoint and direction constraints between *unrelated* cells, which is mathematically undefined for its stated purpose.

  **Recommended Action:** Pass `X[i2]` into the batch (e.g., as `batch[5]`) and use it in the coherence loss instead of `batch[0]`. Alternatively, restructure the batch so that position 0 contains the day-2 lineage data for the coherence loss and separate random-cell data is passed for the ELBO losses.

---

### 4. KNN Triplet Loss — `KNNTripletLoss`

- **Spec Reference:** Standard margin-based triplet loss (Schroff et al. 2015).
- **Implementation:** `examples/weinreb_vae.py:149–168`
  ```python
  da = jnp.sum((za - zp) ** 2, axis=-1)
  dn = jnp.sum((za - zn) ** 2, axis=-1)
  loss = jax.nn.relu(da - dn + self.margin)
  ```
- **Verdict:** **CORRECT**
- **Notes:**
  Standard triplet loss with squared Euclidean distances. The margin $m$ is applied to squared distances (not raw distances), which is a common and valid variant. The loss is computed on deterministic encoder means (no sampling noise), which is appropriate for a geometric regularizer.

---

### 5. Cyclic KL Annealing — `cyclic_beta`

- **Spec Reference:** Fu et al. (2019) "Cyclical Annealing Schedule" (arXiv:1903.10145).
- **Implementation:** `examples/weinreb_vae.py:220–226`
  ```python
  phase = (epoch % cycle_len) / cycle_len
  ramp = min(phase * 2.0, 1.0)
  return beta_min + (beta_max - beta_min) * ramp
  ```
- **Verdict:** **CORRECT**
- **Notes:**
  Linear ramp for the first 50% of each cycle, constant at $\beta_{\max}$ for the second 50%. This is a standard saw-tooth cyclical schedule. The formula is correct and consistent with Fu et al.

---

### 6. AnnealedKLLoss — dead `self.weight`

- **Implementation:** `examples/weinreb_vae.py:229–237`
  ```python
  def __init__(self, beta_max: float = 5e-4):
      super().__init__(beta_max, "AnnealedKL")  # stores self.weight = beta_max

  def __call__(self, model, batch, key, beta: float = 0.0):
      ...
      return jnp.mean(dist.kl_divergence_std_normal()) * beta  # uses beta, not self.weight
  ```
- **Verdict:** **NOTE**
- **Notes:**
  `self.weight` (set to `beta_max = 5e-4`) is never used in `__call__`; the dynamic `beta` from the cyclic schedule is used instead. This is not mathematically wrong (the correct cyclic $\beta$ is applied), but it is misleading: the inherited `weight` field has no effect. No mathematical impact.

---

### 7. Cell Type Classification Loss — `CellTypeClassificationLoss`

- **Implementation:** `examples/weinreb_vae.py:248–263`
  ```python
  log_p = jax.nn.log_softmax(logits, axis=-1)
  oh    = jax.nn.one_hot(labels, self.n_classes)
  return -jnp.mean(jnp.sum(oh * log_p, axis=-1)) * self.weight
  ```
- **Verdict:** **CORRECT**
- **Notes:**
  Standard categorical cross-entropy: $\mathcal{L} = -\frac{1}{B}\sum_i \sum_c y_{ic}\log p_{ic}$. Correctly uses `log_softmax` for numerical stability and acts on deterministic encoder means.

---

### 8. Velocity Consistency Loss — `VelocityConsistencyLoss`

- **Implementation:** `examples/weinreb_vae.py:273–310`
- **Verdict:** **CORRECT**
- **Notes:**
  The pushforward of RNA velocity through the encoder is computed via `jax.jvp` in `GeometricVAE.project_control` ([src/ham/bio/vae.py:101–110](src/ham/bio/vae.py#L101)). This correctly implements the differential map $f_*: T_x \mathcal{X} \to T_{f(x)} \mathcal{Z}$ as a Jacobian-vector product. The cosine-similarity weighting scheme (penalise latent dissimilarity only for data-space-similar velocity pairs, gated by `relu(cos_sim_data)`) is geometrically sound. The zero-velocity mask correctly prevents noise from cells with negligible velocity.

---

### 9. Pullback Metric Evaluation — `compute_pullback_det`

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 (Fundamental Tensor), § 4 table (Riemannian row).
- **Implementation:** `examples/weinreb_vae.py:635–647`
  ```python
  J = jax.jacfwd(v_mod.decode)(z)
  G = jnp.dot(J.T, J) + 1e-6 * jnp.eye(latent_dim)
  sign, ld = jnp.linalg.slogdet(G)
  ```
- **Verdict:** **CORRECT**
- **Notes:**
  The pullback metric $G_{ij}(z) = \sum_a \frac{\partial f^a}{\partial z^i}\frac{\partial f^a}{\partial z^j} = (J^T J)_{ij}$ is the standard Riemannian metric induced by the decoder immersion $f: \mathbb{R}^d \to \mathbb{R}^D$. The $\epsilon I$ regularization prevents $\log\det$ singularities at rank-deficient points. Consistent with the spec's energy-based formalism: for this Riemannian case, $E(z,v) = \frac{1}{2}v^T G(z) v$ and $g_{ij} = \frac{1}{2}\partial^2 F^2 / \partial v^i \partial v^j = G_{ij}$.

---

### 10. WrappedNormal Sampling (Euclidean path)

- **Implementation:** `src/ham/bio/vae.py:25–40`
- **Verdict:** **CORRECT**
- **Notes:**
  For `EuclideanSpace`, `ambient_dim == intrinsic_dim`, so the sampling path is:
  1. `v_flat = N(0, \text{scale}^2)` — tangent vector at origin.
  2. `v_origin = v_flat` (Euclidean: no ambient embedding overhead).
  3. `v_at_mean = parallel_transport(0, \mu, v_flat) = v_flat` (Euclidean: identity transport).
  4. `z = exp_map(\mu, v_flat) = \mu + v_flat` (Euclidean: addition).

  Result: $z \sim \mathcal{N}(\mu, \text{diag}(\sigma^2))$. Correct for Euclidean space.

---

### 11. Hyperboloid Parallel Transport (used by WrappedNormal in non-Euclidean mode)

- **Implementation:** `src/ham/geometry/surfaces.py:377–382`
  ```python
  xy = self._minkowski_dot(x, y)
  yv = self._minkowski_dot(y, v)
  denom = jnp.maximum(1.0 - xy, 2.0)
  scale = yv / denom
  return v + scale[..., None] * (x + y)
  ```
- **Literature Reference:** Cho et al. (2019) "Large-Margin Classification in Hyperbolic Space" (arXiv:1902.10207); Nickel & Kiela (2018) "Learning Continuous Hierarchies in the Lorentz Model" (arXiv:1806.03417).
- **Verdict:** **CORRECT**
- **Notes:**
  The formula implements $P_{x \to y}(v) = v + \frac{\langle y, v \rangle_L}{1 - \langle x, y \rangle_L}(x + y)$. On the hyperboloid $\langle x, y \rangle_L \leq -1$, so $1 - \langle x, y \rangle_L \geq 2$; the `jnp.maximum(..., 2.0)` clamp is a no-op for valid inputs and a safety guard for degenerate cases ($x = y$). Tangency verified analytically: $\langle P(v), y \rangle_L = 0$. This is the standard closed-form parallel transport on the Lorentz model of $\mathbb{H}^n$.

---

### 12. Total Loss Combination

- **Implementation:** `examples/weinreb_vae.py:500–504`
  ```python
  total = l_recon + l_kl + l_cls + l_coh + l_trip + l_vel
  ```
- **Verdict:** **WARNING**
- **Notes:**
  The total objective is:

  $$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta(t)\cdot \mathcal{L}_{\text{KL}} + w_{\text{cls}}\mathcal{L}_{\text{cls}} + w_{\text{coh}}\mathcal{L}_{\text{coh}} + w_{\text{trip}}\mathcal{L}_{\text{trip}} + w_{\text{vel}}\mathcal{L}_{\text{vel}}$$

  This is a multi-objective loss, not a pure ELBO. The auxiliary terms ($\mathcal{L}_{\text{cls}}, \mathcal{L}_{\text{coh}}, \mathcal{L}_{\text{trip}}, \mathcal{L}_{\text{vel}}$) are task-specific regularizers that do not appear in the standard variational inference framework. This is a valid and common approach in structured VAE training, but the resulting objective is **not** a variational lower bound on $\log p(x)$, even before accounting for the KL formula error (Finding 1).

---

## Open Questions

1. **Is the missing $\mu^2$ term intentional?** Some wrapped-normal VAE implementations deliberately drop the mean term to avoid mode collapse on curved manifolds, relying on auxiliary losses (triplet, classification) to constrain the latent means instead. If this is the intent, it should be documented as a deliberate deviation from the standard ELBO, with justification.

2. **Trajectory coherence data mismatch (Finding 3):** Is the use of `idx_main` instead of `i2` for the day-2 cells a known issue or a deliberate design choice (e.g., using random anchors as a baseline)? The current formulation has no geometric or biological justification.

3. **Interaction between halved reconstruction weight and missing KL mean term:** With the stochastic reconstruction down-weighted to 0.5 *and* the KL lacking the mean penalty, the effective pressure toward a structured latent space comes almost entirely from the auxiliary losses (triplet, classification, velocity). Is this the intended training dynamic? If so, the ELBO components are largely decorative.
