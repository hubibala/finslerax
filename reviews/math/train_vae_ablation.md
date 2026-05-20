# Math Review: train_vae_ablation

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Minor Issues.** The file `examples/train_vae_ablation.py` is a thin wrapper that imports all mathematical components from `examples/weinreb_vae.py` and invokes `train_vae()` with `vel_weight=0.0`. The ablation script itself introduces no new mathematical formulae — it only selects a parameter configuration. The mathematical validity of the ablation condition (zeroing the velocity consistency loss) is sound. However, the script inherits every mathematical issue present in the upstream `weinreb_vae.py` losses, and two of these are **CRITICAL**. These inherited issues are itemised below for completeness, with cross-references to the existing `reviews/math/vae.md` review where applicable.

---

## Formula-by-Formula Audit

### 1. Ablation Condition: `vel_weight=0.0`

- **Spec Reference:** N/A (experimental design, not a geometric formula).
- **Literature Reference:** Standard ablation methodology.
- **Implementation:** `examples/train_vae_ablation.py:81`
  ```python
  vel_weight = 0.0,   # ← KEY ABLATION
  ```
  This zeroes the `VelocityConsistencyLoss` weight in the total loss (defined at `examples/weinreb_vae.py:496`):
  ```python
  total = l_recon + l_kl + l_cls + l_coh + l_trip + l_vel
  ```
  With `vel_weight=0.0`, the term $\ell_{\text{vel}} = 0$ identically because `VelocityConsistencyLoss.__call__` (line 296) returns its value scaled by `self.weight`.
- **Verdict:** CORRECT
- **Notes:** Setting a loss weight to zero is a mathematically valid ablation — it cleanly removes one additive term from the objective without affecting the remaining terms. The remaining loss components (reconstruction, KL, triplet, coherence, classification) are mathematically independent of the velocity consistency term. No gradient leakage or coupling exists.

---

### 2. Velocity Scaling: `V_pca_n = V_pca / (scaler.scale_ + 1e-8)`

- **Spec Reference:** N/A (data preprocessing, not geometry).
- **Implementation:** `examples/train_vae_ablation.py:52`
  ```python
  V_pca_n = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)
  ```
- **Verdict:** NOTE
- **Notes:** RNA velocity vectors are divided by the per-feature standard deviation (from `StandardScaler.fit_transform` applied to positions), without subtracting the mean. This is the correct procedure: velocities are tangent vectors (differences), so centering is inappropriate, but scale-normalisation is needed to match the normalised position space. The $10^{-8}$ additive constant prevents division by zero. Since `vel_weight=0.0` in this script, this normalisation has no effect on the trained model, but the code path is mathematically sound for the non-ablated case.

---

### 3. Inherited: `AnnealedKLLoss` — KL Divergence (`weinreb_vae.py:237–240`)

- **Spec Reference:** N/A (variational inference, not in `MATH_SPEC.md`).
- **Literature Reference:** Kingma & Welling 2014 (arXiv:1312.6114); Nagano et al. 2019 (arXiv:1902.02992).
- **Implementation:** `examples/weinreb_vae.py:237–240`
  ```python
  dist = v_get_dist(x)
  return jnp.mean(dist.kl_divergence_std_normal()) * beta
  ```
  which delegates to `src/ham/bio/vae.py:44–45`:
  ```python
  kl = -jnp.log(self.scale + 1e-6) + (self.scale**2) / 2.0 - 0.5
  return jnp.sum(kl, axis=-1)
  ```
  Per dimension this computes:
  $$\hat{D}_i = -\log\sigma_i + \frac{\sigma_i^2}{2} - \frac{1}{2}$$
- **Verdict:** CRITICAL
- **Notes:** This is the KL divergence $D_{KL}\!\bigl(\mathcal{N}(0, \text{diag}(\sigma^2)) \,\|\, \mathcal{N}(0, I)\bigr)$. The correct formula for the isotropic Gaussian prior is:
  $$D_{KL}\!\bigl(\mathcal{N}(\mu, \text{diag}(\sigma^2)) \,\|\, \mathcal{N}(0, I)\bigr) = \sum_i \left[-\log\sigma_i + \frac{\sigma_i^2 + \mu_i^2}{2} - \frac{1}{2}\right]$$
  The $\frac{1}{2}\|\mu\|^2$ term is missing entirely. The posterior mean $\mu$ is therefore **unregularised** — it can drift arbitrarily far from the prior centre with zero KL penalty.

  **Recommended Action:** Add the squared-mean term. In `src/ham/bio/vae.py:44`, change to:
  ```python
  kl = -jnp.log(self.scale + 1e-6) + (self.scale**2 + self.mean**2) / 2.0 - 0.5
  ```
  (Cross-reference: this is the same CRITICAL finding as in `reviews/math/vae.md` §2.)

---

### 4. Inherited: `ReconstructionLossDeterministic` — ELBO Decomposition (`weinreb_vae.py:310–335`)

- **Spec Reference:** N/A (standard VAE).
- **Literature Reference:** Kingma & Welling 2014 (arXiv:1312.6114).
- **Implementation:** `examples/weinreb_vae.py:326–335`
  ```python
  loss_stoch = jnp.mean((x - v_decode(z_sample)) ** 2)
  z_mean = dist.mean
  loss_det   = jnp.mean((x - v_decode(z_mean))  ** 2)
  return (stochastic_weight * loss_stoch + deterministic_weight * loss_det) * self.weight
  ```
  The standard ELBO reconstruction term is $\mathbb{E}_{q(z|x)}[\log p(x|z)]$. For Gaussian $p(x|z) = \mathcal{N}(f_\theta(z), \sigma_x^2 I)$, this reduces to $-\frac{1}{2\sigma_x^2}\mathbb{E}_q\|x - f_\theta(z)\|^2$ plus constants. The implementation uses a single-sample Monte Carlo estimate (line 329: `z_sample`) which is standard.
- **Verdict:** WARNING
- **Notes:** The deterministic-path term $\|x - f_\theta(\mu)\|^2$ is **not** part of the ELBO. The ELBO reconstruction requires the expectation under $q(z|x)$, not evaluation at the mode. The combined objective:
  $$\ell_{\text{recon}} = \alpha_s \|x - f_\theta(z_{\text{sample}})\|^2 + \alpha_d \|x - f_\theta(\mu)\|^2$$
  with $\alpha_s = \alpha_d = 0.5$ is a valid training loss, but it is a **modified ELBO**, not the standard ELBO. The deterministic term acts as an additional regulariser that stabilises the decoder Jacobian (as stated in the docstring). This is mathematically legitimate but should be noted: the resulting objective is not a proper variational lower bound on $\log p(x)$. Its gradient contains a bias toward the posterior mean that is not justified by the variational framework.

  **Recommended Action:** Document that this is a deliberately modified objective (not a strict ELBO), or provide a citation if this construction appears in the literature (e.g., related to "deterministic warm-up" or "two-path VAE" approaches).

---

### 5. Inherited: `KNNTripletLoss` — Triplet Margin Loss (`weinreb_vae.py:148–163`)

- **Spec Reference:** N/A (metric learning, not differential geometry).
- **Literature Reference:** Schroff et al. 2015 "FaceNet" (arXiv:1503.03832).
- **Implementation:** `examples/weinreb_vae.py:159–163`
  ```python
  da = jnp.sum((za - zp) ** 2, axis=-1)
  dn = jnp.sum((za - zn) ** 2, axis=-1)
  loss = jax.nn.relu(da - dn + self.margin)
  return jnp.mean(loss) * self.weight
  ```
  Standard triplet loss: $\ell = \max(0,\; \|z_a - z_p\|^2 - \|z_a - z_n\|^2 + m)$.
- **Verdict:** CORRECT
- **Notes:** This is the standard squared-Euclidean triplet loss, consistent with the Euclidean training manifold used here (no pullback metric during training). The formula is correct.

---

### 6. Inherited: `TrajectoryCoherenceLoss` — Midpoint + Direction (`weinreb_vae.py:177–206`)

- **Spec Reference:** N/A (application-specific regulariser).
- **Implementation:** `examples/weinreb_vae.py:193–206`
  ```python
  z_mid    = 0.5 * (z2 + z6)
  mid_loss = jnp.mean(jnp.sum((z4 - z_mid) ** 2, axis=-1))
  ...
  dir_loss = jnp.mean(1.0 - jnp.sum(v_en * v_fn, axis=-1))
  return (mid_loss + dir_loss) * self.weight
  ```
  The midpoint loss penalises $\|z_4 - \frac{1}{2}(z_2 + z_6)\|^2$, encouraging the day-4 embedding to be the Euclidean midpoint of day-2 and day-6 embeddings.
  The direction loss penalises $1 - \cos\angle(z_4 - z_2,\; z_6 - z_2)$.
- **Verdict:** CORRECT
- **Notes:** Both components are mathematically well-defined. The per-sample normalisation (lines 200–203) correctly computes cosine similarity per sample vector, fixing the original bug noted in the docstring. The $10^{-8}$ floor in the norm prevents division by zero.

---

### 7. Inherited: `CellTypeClassificationLoss` — Cross-Entropy (`weinreb_vae.py:260–269`)

- **Spec Reference:** N/A.
- **Implementation:** `examples/weinreb_vae.py:266–269`
  ```python
  log_p = jax.nn.log_softmax(logits, axis=-1)
  oh    = jax.nn.one_hot(labels, self.n_classes)
  return -jnp.mean(jnp.sum(oh * log_p, axis=-1)) * self.weight
  ```
  Standard categorical cross-entropy: $-\frac{1}{B}\sum_b \sum_c y_{bc}\log p_{bc}$.
- **Verdict:** CORRECT
- **Notes:** The formula is the standard softmax cross-entropy. The use of `log_softmax` (rather than `softmax` followed by `log`) is numerically stable. The small-init classifier head (line 382–388) ensures initial logits are $O(0.01)$, preventing NaN from $\log(\text{softmax})$ saturation.

---

### 8. Inherited: `VelocityConsistencyLoss` — Cosine Alignment (`weinreb_vae.py:278–310`)

- **Spec Reference:** N/A (application-specific).
- **Implementation:** `examples/weinreb_vae.py:293–308`
  ```python
  cos_sim_lat = jnp.sum(v_lat_i * v_lat_j, axis=-1) / (safe_norm(v_lat_i) * safe_norm(v_lat_j))
  weight = jax.nn.relu(cos_sim_data)
  loss_vals = weight * (1.0 - cos_sim_lat)
  return jnp.mean(jnp.where(valid_mask, loss_vals, 0.0)) * self.weight
  ```
  Penalises $w \cdot (1 - \cos\angle(v^{\text{lat}}_i, v^{\text{lat}}_j))$ where $w = \max(0, \cos\angle(u_i, u_j))$ and the mask zeros out cells with $\|u_i\|^2 < 10^{-6}$.
- **Verdict:** CORRECT
- **Notes:** This term is zeroed out in the ablation (`vel_weight=0.0`), so it has no effect on this training run. The formula itself is mathematically sound: it is a cosine-distance loss weighted by data-space alignment, which is a standard construction in velocity-regularised representation learning.

---

### 9. Cyclic KL Annealing Schedule (`weinreb_vae.py:218–225`)

- **Spec Reference:** N/A (training heuristic).
- **Literature Reference:** Fu et al. 2019 "Cyclical Annealing Schedule" (arXiv:1903.10145).
- **Implementation:** `examples/weinreb_vae.py:223–225`
  ```python
  phase = (epoch % cycle_len) / cycle_len
  ramp = min(phase * 2.0, 1.0)
  return beta_min + (beta_max - beta_min) * ramp
  ```
  Produces a sawtooth: $\beta$ ramps linearly from $\beta_{\min}$ to $\beta_{\max}$ over the first half of each cycle, then holds at $\beta_{\max}$ for the second half.
- **Verdict:** CORRECT
- **Notes:** This is a standard cyclical annealing schedule. The linear ramp with plateau is consistent with the "proportional" variant from Fu et al. 2019.

---

### 10. Total Loss Composition (`weinreb_vae.py:490–496`)

- **Spec Reference:** N/A.
- **Implementation:** `examples/weinreb_vae.py:490–496`
  ```python
  total = l_recon + l_kl + l_cls + l_coh + l_trip + l_vel
  ```
  Each component has its own weight baked in (via `self.weight`). With the ablation parameters:
  - $\ell_{\text{recon}}$: weight $1.0$, split 50/50 stochastic/deterministic
  - $\ell_{\text{KL}}$: weight $\beta(t)$, cycled $0 \to 5 \times 10^{-4}$
  - $\ell_{\text{cls}}$: weight $0.15$
  - $\ell_{\text{coh}}$: weight $0.3$
  - $\ell_{\text{trip}}$: weight $1.0$, margin $1.0$
  - $\ell_{\text{vel}}$: weight $0.0$ (ablated)
- **Verdict:** WARNING
- **Notes:** All terms are additive and non-negative (reconstruction and KL by construction; triplet by ReLU; coherence by squared-distance plus $1-\cos\theta \geq 0$; cross-entropy is negated so the sign is correct). However, the total loss is **not** a valid ELBO due to: (a) the missing $\|\mu\|^2$ term in KL (§3), and (b) the deterministic reconstruction path (§4). The auxiliary losses (triplet, coherence, classification) are valid regularisers that do not claim to be part of a variational bound. The overall objective is best described as: "modified reconstruction + incomplete KL + auxiliary regularisers."

---

## Open Questions

1. **Is the missing $\|\mu\|^2$ in the KL divergence intentional?** The code comment in `vae.py` does not explain this omission. If the intent is that the mean is regularised by other terms (e.g., triplet loss acts as a soft centering), this should be documented. Otherwise, the ELBO is incorrect and the posterior mean is unregularised.

2. **Is the deterministic reconstruction path a deliberate departure from the ELBO?** The docstring in `ReconstructionLossDeterministic` states it "anchors the decoder geometry," but does not acknowledge that this invalidates the variational bound interpretation. A citation or mathematical justification for this two-path construction would strengthen the design.

3. **Ablation completeness:** The ablation zeroes `vel_weight` only. A fully controlled ablation study might also compare removing `coherence_weight` and `cls_weight` independently to disentangle the contribution of each auxiliary loss. This is an experimental design question, not a mathematical error.
