# Science Audit: `weinreb_experiment.py`

**Auditor:** Science Auditor Agent
**Date:** 2026-05-15
**Scope:** `examples/weinreb_experiment.py` (main experiment), plus `examples/preprocess_weinreb.py` (data pipeline) and `examples/weinreb_vae.py` (Phase 1 VAE).

---

## Summary

**Overall scientific rigor assessment: NEEDS REVISION — two critical issues, several methodological weaknesses.**

The experiment tests a well-formulated hypothesis — that a Randers (Finsler) metric learned from clonal pseudo-velocity assigns higher energy to counterfactual wrong-fate trajectories than to observed correct-fate trajectories. The experimental design is conceptually sound and the statistical testing framework (Wilcoxon + t-test + Cohen's d) is appropriate. However, two issues threaten the validity of the reported results: (1) a train/test data leakage path in the validation triples file, and (2) the pseudo-velocity used to construct the wind field is derived from the same clonal structure used for validation, creating partial circularity that is acknowledged but insufficiently controlled.

---

## Claims Audit

### Claim 1: "Randers metric assigns higher energy to wrong-fate paths than to observed correct-fate trajectories, when W is learned from clonal pseudo-velocity alone."

*Source:* `weinreb_experiment.py:24–28` (docstring)

- **Evidence provided:** Energy ratio $E_\text{cf}/E_\text{obs}$ distribution, correct-fate fraction, paired Wilcoxon signed-rank test (Randers vs Riemannian), one-sample t-test (ratio > 1), Cohen's d effect size.
- **Literature context:** Randers metrics ($F = \sqrt{g_{ij} v^i v^j} + \beta_i v^i$) are the standard first-order Finsler perturbation of Riemannian geometry used to model asymmetric costs (Randers, 1941; Bao, Chern & Shen, *An Introduction to Riemann-Finsler Geometry*, Springer, 2000). The Zermelo navigation interpretation ($W$ as wind field) is well-established (Bao, Robles & Shen, J. Diff. Geom., 2004). Applying Finsler geometry to trajectory inference in single-cell biology is novel and not yet benchmarked in the literature.
- **Verdict:** **WARNING — partial circularity and data leakage path**
- **Recommendation:** See Claims 3 and 4 below for specific fixes.

---

### Claim 2: "Riemannian metric is symmetric so ratio ≈ 1 always (null baseline)."

*Source:* `weinreb_experiment.py:21–22`

- **Evidence provided:** The Riemannian baseline is constructed by `build_riemannian_baseline()` (`weinreb_experiment.py:433`) which creates a `PullbackRiemannian` with $G(z) = J^\top J$ from the *same* frozen decoder. The Riemannian energy $E(z, v) = v^\top G(z) v$ is symmetric in $v$, so $E(z_a \to z_b) = E(z_b \to z_a)$ to first order.
- **Literature context:** This is mathematically correct — a Riemannian metric tensor is symmetric and positive-definite, so the energy functional is direction-independent for a linear (chord) approximation.
- **Verdict:** **WARNING**
- **Detail:** The claim that the ratio is "≈ 1 always" holds only for the two-segment chord approximation used (`two_segment_energy` at `weinreb_experiment.py:244`). For true geodesic arc lengths, even Riemannian metrics produce asymmetric costs between different endpoints because the metric field $G(z)$ varies spatially. The null hypothesis should more precisely be: "the Riemannian ratio is not systematically > 1 (centered around 1 with symmetric noise)," which is a weaker and more accurate statement.
- **Recommendation:** Verify empirically that the Riemannian ratio distribution is centered at 1.0 by reporting its mean ± std and a two-sided t-test against 1.0. The experiment already computes Riemannian ratios — just report the test.

---

### Claim 3: "No direct circularity in the geodesic MSE test" (pseudo-velocity derivation)

*Source:* `preprocess_weinreb.py:17–18`

- **Evidence provided:** The docstring asserts: "we train on ALL clone pairs and validate on held-out day4 intermediates — so there is no direct circularity."
- **Literature context:** Similar circularity concerns arise in RNA velocity (La Manno et al., Nature, 2018; Bergen et al., Nature Biotech., 2020) and PRESCIENT (Yeo et al., Nature Biotech., 2021) when velocity fields are derived from the same temporal data used for evaluation. Standard practice is to hold out entire clones, not just intermediate timepoints.
- **Verdict:** **CRITICAL — partial circularity inadequately controlled**
- **Detail:**
  1. The pseudo-velocity for cell $i$ at time $t$ is: $v_i = \text{mean}(X_\text{descendants} - X_i)$ over all clonal descendants at later timepoints (`preprocess_weinreb.py:133–160`). This includes day-6 endpoint cells.
  2. The validation uses day2→day4→day6 triples from the *same* clones. The wind field $W(z)$ is built from kernel-smoothed push-forwards of these velocities (`attach_datadriven_randers_metric`, `weinreb_experiment.py:190–222`).
  3. Thus, the wind field was *trained* on information about where cells go (their day-6 descendants), and the validation asks whether the wind field correctly predicts where cells go. The day-4 intermediates are not truly "held out" — they merely weren't used as *anchors* for the velocity estimation.
  4. The preprocessing script *does* split clones 80/20 into train/test (`preprocess_weinreb.py:151–153`), but `weinreb_experiment.py` loads `data/weinreb_lineage_triples.npy` (line 636), **not** `data/weinreb_test_triples.npy`. This file does not exist in the preprocessing output — `preprocess_weinreb.py` generates only `weinreb_train_triples.npy` and `weinreb_test_triples.npy`. It is unclear what `weinreb_lineage_triples.npy` contains or how it was generated. If it contains all clones (train + test), the validation evaluates on training data.
- **Recommendation:**
  1. **Immediately** change `TRIPLES` in `main()` from `"data/weinreb_lineage_triples.npy"` to `"data/weinreb_test_triples.npy"`.
  2. Ensure the anchor velocity subsampling in `attach_datadriven_randers_metric` draws **only from train clones**.
  3. Report results separately on train and test triples to quantify overfitting.

---

### Claim 4: "The data-driven wind field uses kernel-smoothed RNA velocities — no neural network training required."

*Source:* `weinreb_experiment.py:190–222`, `src/ham/models/learned.py:98–118`

- **Evidence provided:** Nadaraya-Watson kernel smoother with Gaussian kernel ($\sigma = 0.4$) over 2000 latent-space anchors.
- **Verdict:** **WARNING — sensitivity to kernel bandwidth uncharacterized**
- **Detail:** The kernel bandwidth $\sigma = 0.4$ is a critical hyperparameter. Too small → overfitting to anchor locations (wind field memorises individual velocity vectors). Too large → over-smoothing (directional signal washed out, ratio → 1). The experiment uses a single value. The H2 experiment (`experiment_h2_directional.py:86`) does sweep $\sigma \in \{0.2, 0.4, 0.6\}$, but `weinreb_experiment.py` does not.
- **Recommendation:** Include the $\sigma$ sweep in the main experiment or cite the H2 results. Report how the energy ratio varies with $\sigma$ to demonstrate robustness.

---

### Claim 5: "Counterfactual = closest wrong-fate attractor to $z_2$ (most challenging comparison)"

*Source:* `weinreb_experiment.py:225–232`, `run_validation` lines 351–356

- **Evidence provided:** For each observed day-6 fate, the counterfactual endpoint is the mean latent position of the closest *wrong-fate* attractor to $z_2$ (the starting cell).
- **Literature context:** This is a reasonable adversarial design — it avoids trivially inflated ratios from comparing to distant, irrelevant fates.
- **Verdict:** **WARNING — attractor estimation is fragile**
- **Detail:** The "fate attractors" are computed as the mean latent position of *all* day-6 cells of that type in the dataset (`build_fate_attractors`, line 256–279). This collapses each fate to a single point, ignoring within-fate heterogeneity. In hematopoiesis, the Monocyte and Neutrophil fates are biologically broad — collapsing them to centroids may not represent realistic counterfactual endpoints.
- **Recommendation:** Consider using a sampled counterfactual: for each triple, randomly select a specific day-6 cell from the wrong fate (rather than the centroid). Report results for both centroid and sampled counterfactuals.

---

### Claim 6: Two-segment discrete energy approximation is adequate

*Source:* `two_segment_energy()` at `weinreb_experiment.py:240–246`

- **Evidence provided:** Energy computed as $E = F(z_2, z_4 - z_2)^2/2 + F(z_4, z_6 - z_4)^2/2$.
- **Literature context:** The true Finsler energy is $E[\gamma] = \frac{1}{2}\int_0^1 F(\gamma, \dot\gamma)^2 dt$. A two-segment chord approximation is standard for short paths but introduces discretization error proportional to the curvature of the actual geodesic.
- **Verdict:** **NOTE**
- **Detail:** H3 (`experiment_h3_discriminative.py`) uses the AVBD solver to compute actual geodesics and normalized arc lengths. This main experiment uses the simpler chord approximation. This is acceptable as a first test but less rigorous than H3.
- **Recommendation:** Consider citing the H3 geodesic-based results alongside these chord-based results to show consistency.

---

## Reproducibility Checklist

- [x] **Random seeds fixed** — JAX PRNGKey seeds are set: 2026 (model init, `weinreb_experiment.py:683`), 42 (anchor subsampling, `weinreb_experiment.py:206`), 77 (Riemannian baseline, `weinreb_experiment.py:304`), 7 (validation, `weinreb_experiment.py:708`).
- [ ] **Hyperparameters logged** — Key hyperparameters ($\sigma=0.4$, `n_anchors=2000`, `n_pairs=1000`, `latent_dim=8`) are hardcoded in `main()` but not saved to a config file or results artifact. **MISSING:** No experiment config is serialized alongside results.
- [ ] **Data preprocessing deterministic and versioned** — Preprocessing uses `np.random.seed(42)` (`preprocess_weinreb.py:150`) but also uses the global NumPy RNG (`np.random.shuffle`, `preprocess_weinreb.py:150`), which is legacy and not reproducible across NumPy versions. **WARNING:** Should use `np.random.default_rng(42)` consistently.
- [ ] **Results include variance estimates** — Standard deviation of energy ratios is reported (`energy_ratio_std`), but: (a) no bootstrap confidence intervals, (b) the experiment is run with a single random seed (no multi-seed repetition). **MISSING:** The H2 experiment runs 5 seeds × 3 $\sigma$ values; the main experiment runs one configuration only.
- [ ] **Baselines are appropriate and fairly implemented** — Riemannian null baseline is appropriate and structurally correct (same decoder, $W=0$). **MISSING:** No comparison to external trajectory inference methods (PRESCIENT, WOT, CellRank, Palantir, Monocle3).

---

## Additional Findings

### F1: StandardScaler fit independently in each script

**Severity: WARNING**

Every script (`weinreb_experiment.py:663`, `experiment_h1_geometric.py:37`, `experiment_h2_directional.py:104`, `experiment_h3_discriminative.py:110`, `train_vae_ablation.py:57`, `weinreb_vae.py:924`) calls `StandardScaler().fit_transform(X_pca)` independently. Because the scaler statistics (mean, variance) are computed from the *entire* dataset each time, and the dataset is identical, this is *numerically* deterministic. However:

1. The scaler is not saved/loaded — if the dataset changes (e.g., filtering), different scripts will silently diverge.
2. The comment at `weinreb_experiment.py:662` says "Same normalisation as phase 1 — MUST be identical" but this is enforced only by convention, not by loading the saved scaler.

**Recommendation:** Serialize the scaler alongside the Phase 1 checkpoint (e.g., via `joblib.dump`) and load it in all downstream scripts.

---

### F2: `weinreb_lineage_triples.npy` — provenance unknown

**Severity: CRITICAL**

`weinreb_experiment.py:636` and `weinreb_vae.py:894` both reference `data/weinreb_lineage_triples.npy`. The preprocessing script `preprocess_weinreb.py` generates `data/weinreb_train_triples.npy` and `data/weinreb_test_triples.npy` but **never** generates `weinreb_lineage_triples.npy`. This file may be:

- (a) A legacy artifact from a previous preprocessing version that combines all clones (no split), in which case validation runs on training data.
- (b) A symlink or copy of the train triples, in which case the issue is merely confusing naming.

Either way, the provenance is unverified and the file name is inconsistent with the preprocessing output.

**Recommendation:** Determine the provenance of this file. If it contains all clones, replace with `weinreb_test_triples.npy`. If it is the train split, label it clearly.

---

### F3: No external baseline methods

**Severity: MISSING**

The experiment compares Randers vs Riemannian (internal ablation). No external trajectory inference baselines are included:

| Method | Reference | Status |
|--------|-----------|--------|
| PRESCIENT | Yeo et al., Nature Biotech. 2021 | Not compared |
| Waddington-OT | Schiebinger et al., Cell 2019 | Not compared |
| CellRank | Lange et al., Nature Methods 2022 | Not compared |
| Palantir | Setty et al., Nature Biotech. 2019 | Not compared |
| FateID | Herman et al., Nature Methods 2018 | Not compared |

**Recommendation:** At minimum, compare against PRESCIENT (which also uses a learned potential + drift model on the Weinreb dataset) and WOT (which uses optimal transport on the same timepoint structure). Both are available as Python packages.

---

### F4: Single-dataset evaluation — no generalization evidence

**Severity: MISSING**

All experiments use the Weinreb et al. (2020) hematopoiesis dataset exclusively. The claim that "Finsler metric outperforms Riemannian for directed/asymmetric processes" is validated on exactly one biological system. Standard practice in computational biology requires validation on at least 2–3 independent datasets.

**Recommendation:** Consider validation on:
- Schiebinger et al. (2019) reprogramming dataset (has clonal barcodes + timepoints)
- Pijuan-Sala et al. (2019) mouse gastrulation dataset (has temporal structure)
- Synthetic datasets with known ground-truth trajectories (for controlled ablation)

---

### F5: No multiple testing correction

**Severity: NOTE**

The experiment reports per-fate correct-fate fractions (`correct_frac_Monocyte`, `correct_frac_Neutrophil`, etc.) without Bonferroni or FDR correction. With only 2 target fates this is unlikely to cause false positives, but if `TARGET_FATES` is extended, this becomes a concern.

**Recommendation:** Apply Bonferroni correction when reporting per-fate significance, or use the overall Wilcoxon test as the primary test (which is already done).

---

### F6: Wind magnitude safety check is a soft warning, not a hard constraint

**Severity: NOTE**

At `weinreb_experiment.py:699–701`, the code checks `|W|_H < 0.95` and prints a warning if violated. The Randers metric requires $|W|_H < 1$ for strong convexity (Bao, Chern & Shen, Ch. 11). Violation produces a degenerate metric. The check is a print statement, not an assertion or automatic $\sigma$ adjustment.

**Recommendation:** Either assert the constraint or implement automatic $\sigma$ reduction when violated.

---

### F7: Only TARGET_FATES = ["Monocyte", "Neutrophil"] tested

**Severity: WARNING**

The Weinreb dataset contains many cell fates (e.g., Basophil, Eosinophil, Erythrocyte, Megakaryocyte, Mast, Lymphoid). The experiment tests only the two dominant granulocyte lineages. While Monocyte and Neutrophil are the primary decision point in this dataset, the restriction limits the generality of the claim.

**Recommendation:** Report results for all fates with sufficient day-6 cell counts. If the metric fails to discriminate rare fates, this should be discussed as a limitation.

---

## Suggested Experiments

1. **Train/test split validation:** Re-run `weinreb_experiment.py` using `weinreb_test_triples.npy` with velocity anchors drawn only from train-clone cells. Report train vs test performance gap.

2. **Multi-seed robustness:** Run the main experiment with 5 different anchor subsampling seeds (as H2 already does for the directional test) and report mean ± std of the energy ratio and p-value.

3. **$\sigma$ sensitivity sweep:** Repeat the energy ratio validation for $\sigma \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0\}$ and report which range produces significant results.

4. **External baseline comparison:** Run PRESCIENT and WOT on the same dataset with matched preprocessing. Compare fate-prediction accuracy (e.g., AUROC for fate classification from day-2 cells).

5. **Second dataset validation:** Apply the pipeline to the Schiebinger et al. (2019) reprogramming dataset.

6. **Sampled counterfactual endpoints:** Instead of using fate centroids, sample individual wrong-fate day-6 cells as counterfactuals and report the distribution of ratios.

7. **Geodesic-based energy validation:** Replace the two-segment chord approximation with AVBD-solved geodesics (as H3 does) in the main experiment and confirm consistency.

8. **Permutation test:** Randomly permute the wind field assignments across cells and show that the discriminative signal disappears — this would confirm the result is not an artifact of the metric's spatial structure alone.
