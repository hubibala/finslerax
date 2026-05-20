# Math Review: preprocess_weinreb

**Reviewer:** Math Reviewer Agent
**Date:** 2025-05-15
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

The preprocessing script is **mathematically sound with minor issues**. The normalisation pipeline (library-size normalisation → $\log(1+x)$ → HVG selection → z-score → PCA → whitening) follows the standard Seurat-style scRNA-seq workflow and is correctly implemented. The clonal pseudo-velocity construction is a well-motivated finite-displacement estimator, but conflates different time horizons ($\Delta t$) across cells without compensation, which introduces a systematic directional and magnitude bias. This is partially mitigated by the per-component variance normalisation of velocities and by the directional (cosine-like) nature of the downstream loss. No critical formula errors were found.

---

## Formula-by-Formula Audit

### 1. Library-Size Normalisation + Log Transform (lines 94–95)

- **Spec Reference:** Not in `spec/MATH_SPEC.md` (data preprocessing, not geometry).
- **Literature Reference:** Satija et al. (2015), *Nature Biotechnology* 33, 495–502 (Seurat normalisation).
- **Implementation:**
  ```python
  sc.pp.normalize_total(adata, target_sum=1e4)   # x_ij → x_ij * 1e4 / Σ_j x_ij
  sc.pp.log1p(adata)                              # x_ij → log(1 + x_ij)
  ```
  The composition is $x_{ij} \mapsto \log\!\bigl(1 + 10^4 \cdot x_{ij} / s_i\bigr)$ where $s_i = \sum_j x_{ij}$.
- **Verdict:** CORRECT
- **Notes:** Standard CPM+log1p pipeline. The choice of `target_sum=1e4` is conventional and does not affect downstream geometry (PCA is translation/scale invariant up to the z-score step).

---

### 2. Z-Score Scaling (line 97)

- **Spec Reference:** N/A (data preprocessing).
- **Implementation:**
  ```python
  sc.pp.scale(adata, max_value=10)
  ```
  Per-gene: $x_{ij} \mapsto \min\!\bigl(\max\!\bigl((x_{ij} - \bar{x}_j) / \sigma_j,\; -10\bigr),\; 10\bigr)$.
- **Verdict:** CORRECT
- **Notes:** Clipping at $\pm 10$ prevents outlier genes from dominating PCA. Standard practice.

---

### 3. PCA Whitening (line 101)

- **Spec Reference:** N/A (data preprocessing).
- **Implementation:**
  ```python
  adata.obsm['X_pca'] = adata.obsm['X_pca'] / (np.std(adata.obsm['X_pca'], axis=0) + 1e-8)
  ```
  Each principal component $k$ is rescaled: $z_{ik} \mapsto z_{ik} / \hat{\sigma}_k$ where $\hat{\sigma}_k = \operatorname{std}(\{z_{ik}\}_i)$.
- **Verdict:** WARNING
- **Notes:**
  - This discards the eigenvalue spectrum of the covariance matrix. PC1 (capturing, say, 15 % of variance) and PC50 (capturing < 0.1 %) are placed on equal footing. Low-variance components may carry predominantly noise; amplifying them could inject spurious structure into the Finsler metric downstream.
  - The comment on line 101 states this is intentional ("so all components contribute equally"). Downstream experiments apply `StandardScaler` again (`examples/experiment_h2_directional.py:102`), which is approximately an identity on already-whitened data, so no further harm.
  - **Recommended Action:** Consider whether retaining eigenvalue weighting (or truncating PCA to fewer components) would reduce noise in the tail components. If the current choice is validated empirically, document the rationale.

---

### 4. Pseudo-Velocity: Temporal Mixing of Displacements (lines 168–184)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 — the velocity field $W^i(x)$ enters the Zermelo parameterisation as the wind; it must represent a *local* tangent vector.
- **Implementation:**
  ```python
  for t_idx, t_curr in enumerate(unique_times[:-1]):
      t_later_cells = idxs[times > t_curr]   # ALL later time points
      ...
      mean_later = X_pca[t_later_cells].mean(axis=0)
      for idx in t_curr_cells:
          V_pseudo[idx] += mean_later - X_pca[idx]
  ```
  For a clone spanning days $\{2, 4, 6\}$:
  - A day-2 cell gets $v_i = \overline{X}_{\{4,6\}} - X_i$ (effective $\Delta t \sim 3$ days, pooling heterogeneous descendants).
  - A day-4 cell gets $v_i = \overline{X}_{\{6\}} - X_i$ (effective $\Delta t = 2$ days).
- **Verdict:** WARNING
- **Notes:**
  - The displacement vectors $v_i$ have **inconsistent time horizons** across cells. Day-2 cells point toward a mixture of intermediate and terminal states (biased by relative cell counts at each time point), while day-4 cells point toward terminal states only. These are then pooled and normalised by a single per-component standard deviation (line 189), collapsing the distinction.
  - For the downstream directional loss (`RNAVelocityWindLoss`, `examples/weinreb_experiment.py:83–107`), direction is the primary signal and the magnitude is down-weighted (`magnitude_weight=0.1`). So the impact is reduced but not eliminated: mixing $\Delta t = 2$ and $\Delta t \sim 3{-}4$ directions changes the direction itself when the differentiation trajectory is curved in PCA space.
  - **Recommended Action:** Consider restricting `t_later_cells` to the *next* time point only (`times == unique_times[t_idx + 1]`) to ensure a uniform $\Delta t$ across all cells. If the current multi-horizon averaging is preferred for robustness, document this choice and note that it estimates a *chord* direction rather than a tangent direction.

---

### 5. Pseudo-Velocity: Displacement vs. Velocity (lines 168–184)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 — the geodesic equation $\ddot{x}^i + 2G^i(x, \dot{x}) = 0$ requires $\dot{x}$ in units of $\mathrm{length}/\mathrm{time}$.
- **Implementation:**
  ```python
  V_pseudo[idx] += mean_later - X_pca[idx]     # displacement, not velocity
  ```
- **Verdict:** NOTE
- **Notes:**
  - The computed quantity is a finite displacement $\Delta X$, not a velocity $\Delta X / \Delta t$. Since the subsequent per-component normalisation (line 189) and the directional nature of the downstream loss make the absolute magnitude irrelevant, this omission has no practical effect on the learned wind direction.
  - The `magnitude_weight` term in `RNAVelocityWindLoss` attempts to match $\|W\|_H$ to $\|v_{\text{lat}}\|_H$; both are in arbitrary units after normalisation, so the match is well-posed but uninterpretable in physical (time) units.

---

### 6. Per-Component Velocity Normalisation (lines 189–190)

- **Spec Reference:** N/A (data preprocessing).
- **Implementation:**
  ```python
  v_std = np.std(V_pseudo[np.any(V_pseudo != 0, axis=1)], axis=0) + 1e-8
  V_pseudo_norm = V_pseudo / v_std[None, :]
  ```
- **Verdict:** CORRECT
- **Notes:**
  - Computing $\sigma_k$ over nonzero-velocity cells only is correct; including the zero-velocity cells would deflate the variance estimate.
  - The $\epsilon = 10^{-8}$ guard prevents division by zero if any PCA component has constant displacement. Appropriate.
  - Zero-velocity cells remain zero after division, consistent with the `valid` guard in `RNAVelocityWindLoss` (`examples/weinreb_experiment.py:87`).

---

### 7. Train/Test Clone Split (lines 152–157)

- **Spec Reference:** N/A (experimental design, not geometry).
- **Implementation:**
  ```python
  split_idx = int(0.8 * len(unique_clones))
  train_clones = set(unique_clones[:split_idx])
  test_clones  = set(unique_clones[split_idx:])
  ```
  Velocity is computed only for train clones (line 161: `if clone_id not in train_clones: continue`), so test-clone cells get $V = 0$.
- **Verdict:** STRONG
- **Notes:**
  - Splitting at the clone level prevents information leakage: no cell from a test clone contributes to the pseudo-velocity used for training the wind field. This is the correct granularity for the clonal validation design.

---

### 8. Minimum Descendant Threshold (line 173)

- **Spec Reference:** N/A.
- **Implementation:**
  ```python
  if len(t_later_cells) < 3 or len(t_curr_cells) == 0:
      continue
  ```
- **Verdict:** CORRECT
- **Notes:** Requiring $\geq 3$ descendant cells before computing a mean displacement is a sensible variance-reduction filter. With fewer cells, the sample mean is dominated by individual-cell noise in 50-dimensional PCA space.

---

### 9. Independence of Position and Velocity Normalisations

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 — the fundamental tensor $g_{ij}(x,v)$ couples position and velocity.
- **Implementation:** Position (line 101) and velocity (line 189) are normalised independently with different scaling factors.
- **Verdict:** NOTE
- **Notes:**
  - After whitening, $X_{\text{pca}}$ has per-component $\sigma = 1$. The velocity $V$ is then computed in this whitened space and re-normalised to its own unit variance. If $\sigma_V \neq 1$ in the position-whitened space, the two normalisations impose different scales on base and fibre coordinates of the tangent bundle $(x, v)$.
  - This is acceptable because: (a) the downstream encoder maps $(x, v)$ into a latent space where the metric is learned, so any affine mismatch can be absorbed; (b) the velocity is used only for wind alignment, not for ODE integration in the preprocessing space.

---

## Open Questions

1. **Eigenvalue truncation:** Has the effect of amplifying low-variance PCA components (Finding 3) been tested empirically? If PC 40–50 are noise-dominated after whitening, they may inject spurious gradients into the Finsler energy.

2. **Next-step vs. multi-step displacement:** Has restricting the pseudo-velocity to the *next* time point (Finding 4) been compared against the current all-later-timepoints strategy? For curved trajectories the two estimates diverge.

3. **Velocity magnitude interpretation:** The downstream `RNAVelocityWindLoss` includes a magnitude term (`magnitude_weight=0.1`). After the double normalisation (position whitening + velocity whitening), what does $\|W\|_H \approx \|v_{\text{lat}}\|_H$ actually enforce in biological units? If this term is active, its target magnitude is arbitrary.
