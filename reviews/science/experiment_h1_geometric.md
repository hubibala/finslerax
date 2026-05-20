# Science Audit: `experiment_h1_geometric.py`

**Auditor:** Science Auditor Agent  
**Date:** 2026-05-15  
**Source:** [examples/experiment_h1_geometric.py](examples/experiment_h1_geometric.py)  
**Spec refs:** `spec/RESEARCH_LOG.md` § 3.1, `spec/MATH_SPEC.md` § 1–4

---

## Summary

**Overall scientific rigor: needs revision**

H1 claims to test whether the pullback metric $G = J^\top J$ "encodes fate topology." The experiment computes three sensible diagnostics (silhouette score, KNN preservation, pullback determinant heatmap), but critically: (1) there are no baselines, (2) there is no statistical testing, (3) the hardcoded conclusion is printed regardless of metric values, and (4) the single-run design with only one random subsample makes the result non-reproducible and non-generalizable.

---

## Claims Audit

### Claim 1: "The pullback metric $J^\top J$ encodes fate topology"

- **Evidence provided:** Silhouette score, 15-NN preservation score, and a 2D heatmap of $\log \det G(z)$ over the first two latent PCs. No threshold, no baseline, no variance estimate.
- **Literature context:** Pullback metrics from decoder Jacobians as tools for latent geometry analysis are well-established (Arvanitidis et al., "Latent Space Oddity," ICML 2018, `arXiv:1710.11379`; Hauberg, "Only Bayes should learn a manifold," `arXiv:1906.06841`). These works always compare against a Euclidean/isotropic baseline and report confidence intervals.
- **Verdict:** **FLAW** — CRITICAL
- **Detail:** The experiment prints `"CONCLUSION: H1 Geometry phase successfully extracts meaningful topology independent of flow."` unconditionally at line 87, regardless of metric values. A silhouette score of 0.01 or a KNN preservation of 0.05 would still produce this conclusion. This is not hypothesis testing; it is confirmation bias encoded in the script.
- **Recommended Action:** (a) Define acceptance thresholds a priori (e.g., silhouette > 0.15, KNN preservation > random baseline). (b) Replace the hardcoded conclusion with a conditional pass/fail.

---

### Claim 2: Silhouette score measures cluster separation in latent space

- **Evidence provided:** Single scalar, no confidence interval, no comparison.
- **Literature context:** Silhouette score is a standard metric (`sklearn`). However, its value depends heavily on the number of clusters and is sensitive to the distance metric (here Euclidean in 8-D latent space, but the claim is about pullback metric topology).
- **Verdict:** **WARNING**
- **Detail:** The silhouette score is computed with `metric="euclidean"` on the raw latent coordinates, not with the pullback metric distance. This measures encoder quality, not pullback metric quality. If the claim is specifically about $J^\top J$ encoding topology, the silhouette should be computed using geodesic distances under $G$, or at minimum compared between Euclidean and pullback-weighted distances.
- **Recommended Action:** Add a Euclidean-baseline silhouette (e.g., on PCA space directly) and, if computationally feasible, a pullback-distance silhouette to separate "encoder learns clusters" from "pullback metric encodes topology."

---

### Claim 3: KNN preservation measures local topology retention

- **Evidence provided:** Single scalar at $k=15$, no confidence interval, no sensitivity analysis.
- **Literature context:** KNN preservation (a.k.a. trustworthiness / continuity, Venna & Kaski 2006) is standard for dimensionality reduction evaluation. Typically reported with multiple $k$ values and compared against a random embedding baseline.
- **Verdict:** **WARNING**
- **Detail:** A single $k$ value is reported. KNN preservation can be high for trivial reasons (e.g., if the VAE is near-linear the PCA neighbors are trivially preserved). No random-embedding baseline is provided to calibrate the score.
- **Recommended Action:** (a) Report at $k \in \{5, 15, 50, 100\}$. (b) Include a random-projection baseline. (c) Report mean ± std over multiple subsamples.

---

### Claim 4: $\log \det G(z)$ heatmap shows the pullback metric varies smoothly

- **Evidence provided:** 2D contour plot of $\log \det G$ evaluated on a $20 \times 20$ grid over PCA projections of the 8-D latent space.
- **Literature context:** This is the standard visualization from Arvanitidis et al. (2018). However, projecting an 8-D latent space to 2D via PCA of $Z$, then evaluating $G$ at those 2D points (padded back to 8-D via `pca2.inverse_transform`) introduces significant distortion — the heatmap only reflects a 2-D slice of the full geometry.
- **Verdict:** **WARNING**
- **Detail:** The `compute_pullback_det` function in [examples/weinreb_vae.py](examples/weinreb_vae.py#L636) evaluates $G$ at points reconstructed via 2-component PCA inverse transform. These points lie on a 2-D affine subspace of $\mathbb{R}^8$, which may not be representative of the actual data manifold. A regularization term `1e-6 * I` is added to $G$ before `slogdet` (line 648), which is reasonable for numerical stability but should be documented.
- **Recommended Action:** (a) Report the fraction of variance explained by the 2-component PCA of $Z$ — if it's low, the heatmap is misleading. (b) Consider t-SNE/UMAP for visualization coordinates but evaluate $G$ at the actual data points rather than on a grid.

---

## Methodological Issues

### Issue 5: No baselines of any kind
- **Verdict:** **FLAW** — CRITICAL
- **Detail:** H1 reports three metrics but compares against nothing. The RESEARCH_LOG (§ 3.1) states "H1: Geometric topology (pullback metric structure) validated" but the experiment provides no null hypothesis test. In contrast, H2 and H3 include Riemannian null baselines and statistical tests. H1 is the weakest link in the experiment chain.
- **Recommended Action:** At minimum, compare against: (a) a random decoder (untrained VAE, same architecture), (b) PCA alone (no VAE), (c) a linear autoencoder. These establish whether the nonlinear decoder contributes meaningful geometry.

### Issue 6: Single-run, single-subsample design
- **Verdict:** **FLAW** — CRITICAL
- **Detail:** The experiment draws one subsample of 15,000 cells with `rng(42)`, computes one silhouette score (itself internally subsampled to 5,000 by `sklearn`), one KNN score, and one heatmap. No variance estimate is possible. This is insufficient for any scientific claim.
- **Recommended Action:** Run $N \geq 5$ independent subsamples, report mean ± std for all scalar metrics, and apply a one-sample $t$-test or bootstrap CI against the null baseline.

### Issue 7: Double-scaling of PCA features
- **Verdict:** **WARNING**
- **Detail:** The preprocessing script (`preprocess_weinreb.py`, line ~100) already applies unit-variance scaling to `X_pca`. The experiment then applies `StandardScaler` again at line 36: `X_norm = scaler.fit_transform(X_pca)`. If the preprocessing truly produced unit-variance features, the second scaling is near-identity (just recentering). But if there is any distributional shift in the subsample, the second scaling changes the input distribution relative to training. The comment at line 34 acknowledges this ambiguity ("Note: X_pca is already unit-variance scaled... but the VAE diagnostic used standard scaler originally. Let's replicate exact inputs") — this suggests the double-scaling is a workaround rather than a principled choice.
- **Recommended Action:** Verify empirically that the double-scaling produces the same distribution as during training. Document the decision. Ideally, save and reload the scaler fitted during training.

### Issue 8: Silhouette `sample_size` truncation
- **Verdict:** **WARNING**
- **Detail:** `silhouette_score(..., sample_size=min(5000, len(Z)))` subsamples 5,000 from 15,000 points. This internal subsampling introduces additional variance that is not accounted for. The sklearn docs warn that `sample_size` is for speed, not for variance reduction.
- **Recommended Action:** Either compute on the full subsample (15,000 is feasible for silhouette) or report the variance over multiple `random_state` values.

---

## Reproducibility Checklist

- [x] Random seeds fixed — `PRNGKey(42)` for JAX, `default_rng(42)` for numpy sampling
- [ ] Hyperparameters logged — `LATENT_DIM=8`, `sample_size=15000` hardcoded but not written to a results file or logged with a timestamp
- [ ] Data preprocessing deterministic and versioned — double-scaling ambiguity (Issue 7); no hash/version check on `weinreb_preprocessed.h5ad`
- [ ] Results include variance estimates — **No**. Single-run only.
- [ ] Baselines are appropriate and fairly implemented — **No baselines at all**

---

## Comparison with Other Experiments in the Suite

| Property | H1 (this) | H2 | H3 | H4 |
|---|---|---|---|---|
| Baselines | None | Randers vs Riemannian null | Randers vs PullbackRiemannian | Randers vs Riemannian null |
| Statistical test | None | Wilcoxon signed-rank | Effect size + paired test | Distance to fate centroid |
| Multiple seeds | No | 5 seeds × 3 σ values | Multiple seeds | Single run |
| Variance reported | No | mean ± std per σ | Yes | No |
| Hardcoded conclusion | Yes (always "success") | No (data-driven) | No (data-driven) | Partial |

H1 is structurally the weakest experiment. It serves as a sanity check but is presented as hypothesis validation.

---

## Suggested Experiments

1. **Untrained baseline:** Re-run H1 with a randomly-initialized (untrained) VAE using the same architecture. If silhouette and KNN scores are comparable, the trained decoder contributes no geometric information.

2. **PCA-only baseline:** Compute silhouette and KNN preservation directly on PCA coordinates (no VAE encoding). This establishes the floor.

3. **Pullback-aware silhouette:** Compute pairwise geodesic distances under $G = J^\top J$ for a subsample, then compute silhouette using those distances. This directly tests the claim about $G$ encoding topology, not just the encoder.

4. **Multi-seed robustness:** Run with $\geq 5$ subsamples and report mean ± 95% CI for all scalar metrics.

5. **Sensitivity to latent dimension:** The experiment fixes `LATENT_DIM=8`. Ablate over $\{4, 8, 16, 32\}$ to establish whether the geometric structure is robust to dimensionality.

6. **Conditional pass/fail:** Replace the hardcoded conclusion with explicit acceptance criteria (e.g., silhouette $> 0.15$, KNN $> 2 \times$ random baseline). Document these thresholds before running.

---

## Severity Summary

| # | Finding | Severity |
|---|---------|----------|
| 1 | Hardcoded unconditional conclusion | **CRITICAL** |
| 5 | No baselines | **CRITICAL** |
| 6 | Single-run, no variance estimates | **CRITICAL** |
| 2 | Silhouette uses Euclidean, not pullback distance | **WARNING** |
| 3 | Single $k$ value, no random baseline for KNN | **WARNING** |
| 4 | 2D PCA slice of 8D heatmap may be unrepresentative | **WARNING** |
| 7 | Double-scaling ambiguity | **WARNING** |
| 8 | Silhouette internal subsampling variance | **WARNING** |
