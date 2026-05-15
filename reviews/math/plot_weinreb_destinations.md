# Math Review: plot_weinreb_destinations

**Reviewer:** Math Reviewer Agent
**Date:** 2025-05-15
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

Minor Issues. This file is a visualization script with no Finsler-geometric computations. Its sole non-trivial mathematical operation is a kernel-weighted velocity projection from PCA space to the SPRING 2D embedding. The projection formula is sound and follows the standard scRNA-seq velocity embedding approach (cf. scVelo / velocyto). Two minor warnings are raised regarding bandwidth estimation and near-zero velocity edge cases. Two exemplary practices are noted.

## Formula-by-Formula Audit

### 1. Global Gaussian Bandwidth

- **Spec Reference:** Not in `spec/MATH_SPEC.md` (visualization utility, not a geometric object).
- **Literature Reference:** Standard kernel bandwidth selection via median heuristic; see Garreau et al. "Large sample analysis of the median heuristic" (arXiv:1707.07269).
- **Implementation:** `examples/plot_weinreb_destinations.py:55–57`
  ```python
  global_dists, _ = nn.kneighbors(x_pca[sample_idx])
  global_sigma2 = np.median(global_dists**2)
  ```
  Computes $\sigma^2 = \operatorname{median}\!\bigl(\{d_{ij}^2 : j \in \mathcal{N}_k(i),\; i \in S\}\bigr)$ where $S$ is a random subsample and $k = 50$.
- **Verdict:** WARNING
- **Notes:** Because `nn` was fit on the full dataset and `sample_idx` is a subset of that same dataset, `kneighbors` returns each query point as its own nearest neighbour with $d = 0$. The first column of `global_dists` is therefore all zeros. For $k = 50$ this is only 2 % of entries, so the median is barely affected — but the estimator is formally contaminated by self-distances. **Recommended Action:** Pass `x_pca[sample_idx]` through a fresh `NearestNeighbors` that excludes self-matches, or slice off the first column: `global_dists[:, 1:]`.

### 2. Velocity Projection (PCA → SPRING)

- **Spec Reference:** Not in `spec/MATH_SPEC.md`.
- **Literature Reference:** La Manno et al. "RNA velocity of single cells" (Nature 2018, doi:10.1038/s41586-018-0414-6); Bergen et al. "Generalizing RNA velocity…" (Nature Biotechnol. 2020, doi:10.1038/s41587-020-0591-3).
- **Implementation:** `examples/plot_weinreb_destinations.py:113–128`
  ```python
  disp_pca_norm = disp_pca / (np.linalg.norm(disp_pca, axis=1, keepdims=True) + 1e-8)
  proj = np.dot(disp_pca_norm, v_pca[i])
  weights = np.exp(-dists[k]**2 / (global_sigma2 + 1e-8))
  combined_weights = weights * proj
  v_spr_dir = np.sum(disp_spr * combined_weights[:, None], axis=0)
  v_spr[k] = v_spr_dir / (np.linalg.norm(v_spr_dir) + 1e-8) * 2.0
  ```
  The projected velocity is:
  $$
  \tilde{v}_i^{\,\text{SPR}} = \sum_{j \in \mathcal{N}(i)} \underbrace{\exp\!\Bigl(-\frac{d_{ij}^2}{\sigma^2}\Bigr)}_{\text{proximity}} \;\cdot\; \underbrace{\bigl(\hat{\delta}_{ij}^{\,\text{PCA}} \cdot v_i^{\,\text{PCA}}\bigr)}_{\text{directional projection}} \;\cdot\; \delta_{ij}^{\,\text{SPR}}
  $$
  followed by unit-normalisation and scaling to length 2.
- **Verdict:** CORRECT
- **Notes:** The signed scalar projection $\hat{\delta}_{ij} \cdot v_i$ is mathematically preferable to the thresholded $\max(0, \cos\theta)$ used in some scVelo implementations, because it allows anti-aligned neighbours to contribute negative weight, yielding a proper weighted sum rather than a biased one. Normalisation to fixed arrow length (2.0) discards magnitude information; this is standard for quiver-plot visualisation and does not introduce error.

### 3. Near-Zero Projected Velocity Normalisation

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1 (epsilon regularisation philosophy).
- **Implementation:** `examples/plot_weinreb_destinations.py:127`
  ```python
  v_spr[k] = v_spr_dir / (np.linalg.norm(v_spr_dir) + 1e-8) * 2.0
  ```
- **Verdict:** WARNING
- **Notes:** When the signed projections nearly cancel (ambiguous velocity direction), $\|\tilde{v}\| \approx 10^{-6}\text{–}10^{-8}$. For $\|\tilde{v}\| \sim \varepsilon = 10^{-8}$ the division produces a unit-length vector in a numerically noisy direction, displayed as a full-length arrow. This can be visually misleading. **Recommended Action:** Introduce a magnitude threshold (e.g., `if norm < threshold: v_spr[k] = 0`) or scale arrow length proportionally to $\|\tilde{v}\|$ so that ambiguous cells produce no arrow.

### 4. PCA Displacement Normalisation (Epsilon Guard)

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1 (epsilon regularisation).
- **Implementation:** `examples/plot_weinreb_destinations.py:120`
  ```python
  disp_pca_norm = disp_pca / (np.linalg.norm(disp_pca, axis=1, keepdims=True) + 1e-8)
  ```
- **Verdict:** CORRECT
- **Notes:** The $\varepsilon = 10^{-8}$ additive guard prevents division by zero when a neighbour coincides with the query point (which happens for the self-match in k-NN; see Finding 1). The guard is correctly placed in the denominator and does not distort non-degenerate displacements.

### 5. Zero-Velocity Pre-filter

- **Spec Reference:** N/A.
- **Implementation:** `examples/plot_weinreb_destinations.py:108`
  ```python
  valid_vel = np.linalg.norm(v_pca[vel_idx_plot], axis=1) > 0
  ```
- **Verdict:** NOTE
- **Notes:** The strict `> 0` test relies on exact floating-point zero. In practice RNA velocity vectors are unlikely to be exactly zero, so this is benign. A threshold like `> 1e-10` would be more robust but is not necessary for correctness.

### 6. Use of Global Bandwidth (Design Choice)

- **Spec Reference:** N/A.
- **Implementation:** `examples/plot_weinreb_destinations.py:52–57`
- **Verdict:** STRONG
- **Notes:** Using a single global $\sigma^2$ rather than a per-cell local bandwidth avoids density-dependent distortions in the velocity field projection. The comment at line 52 explicitly documents this rationale.

### 7. Signed Projection Weights

- **Spec Reference:** N/A.
- **Implementation:** `examples/plot_weinreb_destinations.py:123–124`
- **Verdict:** STRONG
- **Notes:** Retaining signed projections (including $\hat{\delta} \cdot v < 0$) is the mathematically clean choice. It computes a proper weighted average of embedding displacements. The comment at line 122 documents the fix and its motivation.

## Open Questions

1. The velocity field `velocity_pca` is taken directly from the AnnData object. Its provenance (scVelo, velocyto, or other) is not documented in this script. Different velocity estimation methods may produce qualitatively different projection results. This is outside the scope of mathematical review but relevant for reproducibility.
2. The relationship between this velocity projection and the Finsler velocity field learned by HAMTools is not established in this script. It would be valuable to compare the projected RNA velocity arrows with the geodesic spray directions $-2G^i(x, v)$ from the learned metric, as a qualitative validation.
