# Science Audit: experiment_h3_discriminative.py

**Auditor:** Science Auditor Agent  
**Date:** May 15, 2026  
**File:** `examples/experiment_h3_discriminative.py`

---

## Summary

**Overall scientific rigor assessment: needs revision.**

The experiment tests a well-formulated hypothesis (H3: Randers metric discriminates observed vs. counterfactual fate trajectories) with a structurally sound design: paired comparison, proper null baseline, and non-parametric statistical test. However, there are several methodological weaknesses — notably the absence of effect size reporting, missing multiple-comparison correction, potential information leakage through the velocity-derived wind field, and insufficient variance estimation — that would need to be addressed before these results could support a publication claim.

---

## Claims Audit

### Claim 1: "Randers metric assigns lower cost to correct-fate paths than wrong-fate paths (E_cf / E_obs > 1)"
- **Evidence provided:** Paired Wilcoxon signed-rank test of arc-length-normalised Randers ratios vs. Riemannian ratios; fraction of triples with ratio > 1.0.
- **Literature context:** The counterfactual energy test follows the paradigm established in optimal transport trajectory inference (Schiebinger et al., *Cell* 2019, DOI:10.1016/j.cell.2019.01.006; Tong et al., *ICML* 2020, arXiv:2006.06994). However, those methods report effect sizes (e.g., earth-mover distance reduction) and bootstrap confidence intervals, which are absent here.
- **Verdict:** WARNING
- **Recommendation:** Report Cohen's $d$ (or rank-biserial $r$) and 95% bootstrap CI on the mean ratio difference. The parent script `weinreb_experiment.py:427` already computes Cohen's $d$ — the omission here is inconsistent. Without effect size, statistical significance alone is uninformative (large $n$ can make trivially small effects significant).

---

### Claim 2: "Randers geodesics pass closer to observed intermediate states (day-4 cells) than Riemannian geodesics"
- **Evidence provided:** Paired Wilcoxon signed-rank test on minimum Euclidean distance between geodesic trajectory and the day-4 latent embedding.
- **Literature context:** Geodesic interpolation accuracy is evaluated similarly by PRESCIENT (Yeo et al., *Nature Biotechnology* 2021, DOI:10.1038/s41587-021-01094-4), though they use held-out cell populations rather than individual clonal intermediates.
- **Verdict:** WARNING
- **Recommendation:** (a) The proximity metric uses Euclidean distance in latent space. If the claim is that Finsler geometry is superior, the proximity should also be measured in the Finsler metric itself (using $d_F$) to avoid trivialising the geometric claim. (b) Report the median absolute improvement, not just the fraction of improvement.

---

### Claim 3: "Riemannian metric serves as an appropriate null baseline (ratio ≈ 1)"
- **Evidence provided:** `PullbackRiemannian` is structurally independent — it uses $G(z) = J^T J$ with NO wind field, constructed via `build_riemannian_fallback()` at `experiment_h3_discriminative.py:25`.
- **Literature context:** This is the standard approach: the Riemannian metric $G(z) = J^T J$ produces a symmetric cost function, so the expected ratio is $\approx 1$ for randomly paired observed/counterfactual paths of equal Euclidean distance. This is methodologically correct.
- **Verdict:** STRONG
- **Recommendation:** None — the structural independence of the Riemannian null from the Randers model is correct and clearly documented.

---

### Claim 4: "Arc-length normalisation controls for Euclidean distance"
- **Evidence provided:** `arc_length_normalized()` at line 36 divides Finsler arc length by the Euclidean path length of the **actual geodesic trajectory**, not the chord length. This is a proper distance-controlled ratio.
- **Literature context:** Normalising by path length rather than chord length is the correct approach; chord normalisation would be confounded by curvature. This follows standard differential geometry practice.
- **Verdict:** STRONG
- **Recommendation:** None.

---

### Claim 5 (Implicit): "AVBD solver produces reliable geodesics at 20 iterations"
- **Evidence provided:** `AVBDSolver(iterations=20)` at line 193. No convergence diagnostics are reported.
- **Literature context:** Vertex-relaxation BVP solvers typically require convergence verification (energy plateau or gradient norm below threshold). 20 iterations is very low — `AVBD` default `tol` is $10^{-4}$ and `energy_tol` is $10^{-4}$ (`src/ham/solvers/avbd.py:43-47`), but with only 20 iterations these tolerances may not be reached, meaning the solver returns unconverged paths.
- **Verdict:** WARNING
- **Recommendation:** (a) Log and report the mean/max constraint violation and energy convergence of the solver across all evaluated pairs. (b) Run a sensitivity analysis at iterations ∈ {20, 50, 100} to verify that the H3 result is stable w.r.t. solver precision. If geodesics are unconverged, the Finsler/Euclidean ratio is meaningless.

---

## Methodological Issues

### M1: Information Leakage via Velocity-Derived Wind — CRITICAL

The wind field $W(z)$ in `DataDrivenPullbackRanders` is a kernel smoother over pseudo-velocities projected into latent space (`src/ham/models/learned.py:98-116`). These pseudo-velocities are computed from clonal structure in `preprocess_weinreb.py:119-167` — specifically, velocity$_i$ = mean(X$_{\text{descendants}}$ − X$_i$) for cells sharing a clone ID.

The test triples (day2 → day4 → day6) are from **held-out clones** (`preprocess_weinreb.py:155`, test clones = 20%). However, the wind field's kernel smoother uses **all dataset velocities** (`attach_datadriven_randers_metric` uses the full `dataset`, line 130 of `experiment_h3_discriminative.py`). This means:

- The wind field was constructed using velocity information from **training clones**, which share cell types and PCA neighbourhoods with test clones.
- In high-density regions, the Nadaraya-Watson smoother will interpolate training-clone velocities onto test-clone query points, creating a **transductive leakage**: the wind field "knows" the developmental direction at test-cell locations because nearby training cells were used to build it.

This is not full data snooping (the test triples themselves are held out), but it weakens the claim of independent validation.

**Recommended Action:** (a) Rebuild the wind field using only training-clone velocities (filter `dataset.V` to training-clone cells before calling `attach_datadriven_randers_metric`). (b) Report results under both conditions. If the effect survives, the claim is strengthened.

---

### M2: No Multiple Comparison Correction — WARNING

The experiment performs two hypothesis tests (energy ratio and proximity), each at $\alpha = 0.05$, on the same dataset. No Bonferroni or Holm–Bonferroni correction is applied (`experiment_h3_discriminative.py:217-239`).

With two tests, the family-wise error rate is $1 - (1 - 0.05)^2 \approx 0.0975$, nearly double the nominal rate.

**Recommended Action:** Apply Holm–Bonferroni correction. With only 2 tests this is trivial: the smaller $p$-value must survive $\alpha/2 = 0.025$ and the larger must survive $\alpha/1 = 0.05$.

---

### M3: Missing Variance / Confidence Intervals — WARNING

Results report only mean ratio and fraction > 1.0 (`experiment_h3_discriminative.py:213-216`). No standard deviation, standard error, confidence interval, or effect size is reported.

The parent experiment `weinreb_experiment.py:415-430` computes Cohen's $d$, median, and std — this experiment should report the same statistics for consistency and interpretability.

**Recommended Action:** Add `np.std(r_rand)`, `np.median(r_rand)`, Cohen's $d$ = (mean − 1) / std, and bootstrap 95% CI for the mean ratio.

---

### M4: Counterfactual Selection Bias — WARNING

The counterfactual endpoint $z_6^{cf}$ is chosen as the **closest wrong-fate attractor** to $z_2$ (`experiment_h3_discriminative.py:165`). This is described as "hardest comparison" (analogous to `weinreb_experiment.py:360`).

However, since `TARGET_FATES = ["Monocyte", "Neutrophil"]` (only 2 fates), every Monocyte cell's counterfactual is Neutrophil, and vice versa. The "closest wrong-fate" logic degenerates to the single alternative, making the distance-control unnecessary. More critically:

- With only 2 target fates, the counterfactual is always the same attractor centroid for all cells of a given fate. This means $z_6^{cf}$ has **zero variance** within each fate class, while $z_6^{obs}$ has the full biological variance. The asymmetry in endpoint variance inflates the ratio.

**Recommended Action:** (a) Clarify that with 2 fates, no "closest" selection is needed — it's simply the other fate. (b) Consider using **individual** wrong-fate day-6 cells as counterfactuals (matched by Euclidean distance to $z_2$) rather than fate centroids, to better control for variance.

---

### M5: Experiment Uses Only Test Triples but Wind Field Uses Full Data — WARNING

`TEST_TRIPLES` (line 97) are loaded for the evaluation, but `dataset` (line 118-119) is constructed from the full `adata`. The `attach_datadriven_randers_metric(vae_p1, dataset)` call (line 130) thus constructs the wind field from all cells, including those in test triples.

This is distinct from M1 (which concerns velocity/clone leakage); here the concern is that the **encoder** saw all cells during Phase 1 training. However, since Phase 1 is purely reconstruction + KNN topology (no clonal/trajectory supervision), this is acceptable for the encoder. The issue is only with the wind field, as flagged in M1.

**Verdict:** INFO (acceptable for encoder; problematic only for wind field, covered in M1).

---

### M6: Test Skips the Two-Segment Path (z2→z4→z6) — WARNING

The header and print statement claim the test evaluates trajectory **energy** along the two-segment path z2→z4→z6 (`experiment_h3_discriminative.py:101`). But `batch_eval` (line 186) actually computes `arc_length_normalized(m_rand, z2, z6_obs, solver)` — a **single geodesic** from z2 to z6, skipping the observed intermediate z4 entirely for the energy test.

The z4 intermediate is used only for the **proximity** sub-test (line 192). This means:
- The energy discriminative test evaluates $\text{ArcLen}(z_2 \to z_6^{obs})$ vs. $\text{ArcLen}(z_2 \to z_6^{cf})$, not the two-segment cost.
- The `two_segment_normalized_cost` function (line 53) is defined but **never called**.

The parent experiment (`weinreb_experiment.py:375`) correctly evaluates the two-segment energy $E(z_2 \to z_4 \to z_6)$.

**Recommended Action:** Either (a) use `two_segment_normalized_cost` for the energy test to match the stated hypothesis and the parent experiment, or (b) update the hypothesis statement to clarify that a single-segment geodesic is tested.

---

## Reproducibility Checklist

- [x] Random seeds fixed — `np.random.default_rng(42)` for subsampling, `jax.random.PRNGKey(42)` for model loading (`experiment_h3_discriminative.py:123,127`)
- [ ] Hyperparameters logged — AVBD iterations (20), n_steps (20), kernel sigma (default 0.5), n_anchors (default 5000), latent dim (8) are hardcoded but **not saved to a results file**
- [x] Data preprocessing deterministic and versioned — `preprocess_weinreb.py` uses fixed seed, deterministic PCA, and saves to versioned `.h5ad`
- [ ] Results include variance estimates — **Missing**: no std, CI, or effect size reported
- [x] Baselines are appropriate and fairly implemented — `PullbackRiemannian` is structurally independent (no wind), using the same decoder Jacobian; this is a proper null

---

## Suggested Experiments

1. **Solver convergence sensitivity:** Re-run H3 with AVBD iterations ∈ {20, 50, 100, 200} and verify the ratio and $p$-value are stable. Report mean energy convergence residual.

2. **Wind field holdout:** Reconstruct the `DataDrivenPullbackRanders` wind field using only training-clone velocities. If the H3 effect persists, the transductive leakage concern (M1) is resolved.

3. **Per-fate stratification:** Report energy ratios separately for Monocyte and Neutrophil fates. Asymmetric effects (one fate drives all the signal) would indicate the wind field has a directional bias rather than a general geometric advantage.

4. **Additional baselines:** Compare against (a) Euclidean distance (no metric learning at all), (b) a random wind field (same magnitude, random direction), and (c) PRESCIENT/WOT optimal-transport baselines on the same held-out triples.

5. **Two-segment test:** Fix M6 by running the energy test with the two-segment path z2→z4→z6, which is the biologically meaningful trajectory.

6. **Bootstrap power analysis:** Estimate the minimum detectable effect size given $n = 1000$ triples and the observed variance, to ensure the test is not underpowered for scientifically meaningful effect sizes.

7. **Generalisation to other lineages:** Extend `TARGET_FATES` beyond Monocyte/Neutrophil. The Weinreb dataset contains Basophil, Eosinophil, Erythrocyte, Megakaryocyte, and Lymphoid fates. Demonstrating the effect across ≥4 fates would substantially strengthen the claim and require proper multiple-comparison correction.
