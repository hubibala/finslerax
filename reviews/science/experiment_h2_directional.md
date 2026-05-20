# Science Audit: `experiment_h2_directional.py`

**Auditor:** Science Auditor Agent  
**Date:** 2026-05-15  
**Source:** [examples/experiment_h2_directional.py](examples/experiment_h2_directional.py)  
**Spec refs:** `spec/RESEARCH_LOG.md` § 3.1, `spec/MATH_SPEC.md` § 1 (Randers metric), § 2 (Geodesic Spray)

---

## Summary

**Overall scientific rigor assessment: needs revision**

H2 tests a central claim of the paper — that the Randers (Finsler) metric captures directional asymmetry of developmental trajectories by exhibiting $L_{\text{fwd}} < L_{\text{bwd}}$ along the same geodesic. The experiment is well-structured (sweep over $\sigma$ and multiple seeds, chunked BVP solving, Wilcoxon signed-rank test). However, it suffers from **missing multiple-comparison correction**, **no effect-size reporting**, **absence of a Riemannian null baseline within this script**, and a **potential circularity** between the velocity field used to construct the wind and the direction used to define "forward." These issues together mean the statistical significance reported is likely inflated and the claim of directional asymmetry is not yet rigorous enough for publication.

---

## Claims Audit

### Claim 1: "Forward Finsler arc length is systematically shorter than backward arc length along the same geodesic for biologically forward (day2→day4, day4→day6) segments"

- **Evidence provided:** Wilcoxon signed-rank test (`scipy.stats.wilcoxon`, `alternative='less'`) across up to 800 clonal triples × 2 segments = 1600 paired observations, repeated for 5 seeds × 3 $\sigma$ values.
- **Literature context:** Directional asymmetry $F(x,v) \neq F(x,-v)$ is the defining property of Finsler vs Riemannian metrics. In the Randers navigation model (Bao, Chern, Shen, *An Introduction to Riemann-Finsler Geometry*, Springer 2000; Randers 1941, Phys. Rev. 59:195), the arc length asymmetry arises from the wind term $\beta(v)$. A proper test must confirm the asymmetry vanishes when wind is removed, i.e., a Riemannian null is essential (this is done in H3 but not within this script).
- **Verdict:** WARNING
- **Issues:**
  1. **No multiple-comparison correction.** The experiment runs 15 independent Wilcoxon tests ($3\sigma \times 5$ seeds) and reports each p-value individually. No Bonferroni, Holm, or Benjamini-Hochberg correction is applied. The summary section reports "fraction of runs with $p<0.05$" which is an informal meta-analytic device but not a statistically valid one; at $\alpha=0.05$ with 15 tests, $\sim$0.54 false positives are expected under $H_0$.
  2. **No effect size.** The ratio $L_\text{fwd}/L_\text{bwd}$ is reported but no standardised effect size (e.g., rank-biserial correlation $r = 1 - 2U/(n_1 n_2)$, or Cohen's $d$ on log-ratios) is computed. A tiny but consistent asymmetry can be statistically significant with $n=1600$ without being scientifically meaningful.
  3. **No confidence intervals.** The summary reports `mean±std` across seeds, but does not bootstrap or compute CIs on the ratio or fraction metrics.
- **Recommended Actions:**
  - Apply Benjamini-Hochberg FDR correction across the 15 tests.
  - Report rank-biserial correlation or matched-pairs effect size alongside each p-value.
  - Report 95% bootstrap CIs on the mean ratio.

### Claim 2: "The asymmetry is robust across kernel bandwidth $\sigma$ and anchor-sampling seed"

- **Evidence provided:** A sweep over `SIGMAS = [0.2, 0.4, 0.6]` and `SEEDS = [42, 101, 256, 1024, 2048]`. The summary table reports `ratio mean±std` and `frac Lf<Lb` aggregated across seeds for each $\sigma$.
- **Literature context:** Kernel bandwidth sensitivity is a well-known issue in non-parametric wind estimation. Silverman's rule-of-thumb or cross-validation are standard approaches for $\sigma$ selection (Silverman 1986, *Density Estimation*). Three $\sigma$ values is a minimal sensitivity check, not a systematic sweep.
- **Verdict:** WEAKNESS
- **Issues:**
  1. **Too few $\sigma$ values.** Three points cannot distinguish a plateau from a monotone trend. A proper robustness study needs at least 5–8 values spanning an order of magnitude (e.g., 0.1 to 1.0).
  2. **No $\sigma$-selection rationale.** The choice of $\{0.2, 0.4, 0.6\}$ is not justified. There is no cross-validation or information-theoretic criterion for choosing the operating point.
  3. **Seed variation conflates two sources:** anchor-sampling randomness and solver randomness (AVBD uses a data-dependent key via `jax.random.fold_in`). These should be ablated independently.
- **Recommended Actions:**
  - Expand $\sigma$ sweep to at least `np.logspace(-1, 0.3, 8)`.
  - Report a recommended $\sigma$ with a principled selection criterion.
  - Separate anchor seed from solver key seed.

### Claim 3: "The experiment uses held-out test triples (no data leakage from velocity estimation to evaluation)"

- **Evidence provided:** The file loads `data/weinreb_test_triples.npy`, which `preprocess_weinreb.py` constructs from the 20% held-out clone split. The velocity pseudo-field is computed only from the 80% train clones.
- **Literature context:** This is the correct design (train velocity on clones A, test geodesic asymmetry on clones B). However, the **wind field anchors** in `attach_datadriven_randers_metric` are drawn from the *entire* dataset (`dataset.X`, `dataset.V`), which includes train-clone velocity estimates. If any test-clone cells have zero velocity (they should, per the preprocessing code), the anchors are filtered to `valid_mask = vel_norms > 1e-6`, which removes them. But the kernel smoother still uses train-clone anchors to impute wind at test-clone latent positions — this is **acceptable** (analogous to fitting a model on train data and predicting on test), but should be stated explicitly.
- **Verdict:** OK (with caveat)
- **Recommended Action:** Add a comment or docstring making the train-anchor / test-evaluation split explicit. Verify in the preprocessing pipeline that zero-velocity cells (test clones) are indeed excluded from anchors.

### Claim 4: "The AVBD solver with 60 iterations and 15 path steps produces converged geodesics"

- **Evidence provided:** `solver = AVBDSolver(iterations=60)` with `n_steps=15` and `train_mode=True` (fixed scan, no convergence check). The standalone function `arc_length_forward_backward` (lines 66–73) uses `n_steps=25` but is **not called** — the `batch_arc_lengths` inner function (line 131) uses `n_steps=15`.
- **Literature context:** BVP geodesic solvers require convergence verification. Standard practice is to report the residual energy or constraint violation (both available in `Trajectory` as `energy` and `constraint_violation`). Neither is logged.
- **Verdict:** WARNING
- **Issues:**
  1. **Dead code misleads.** `arc_length_forward_backward` (lines 66–73) with `n_steps=25` is defined but never called. The actual computation uses `n_steps=15` inside `batch_arc_lengths`. This is confusing and could lead a reviewer to believe higher resolution was used.
  2. **No convergence diagnostics.** `train_mode=True` runs a fixed number of scan iterations with no early-stopping or convergence check. The `Trajectory` output contains `energy` and `constraint_violation` fields but they are ignored.
  3. **15 path points is coarse** for a midpoint-rule arc-length integral in a possibly highly curved latent space. Numerical integration error could be asymmetric (direction-dependent) if the geodesic curvature varies, confounding the forward/backward comparison.
- **Recommended Actions:**
  - Remove or flag the dead `arc_length_forward_backward` function.
  - Log mean/max `constraint_violation` and `energy` from the solver output.
  - Ablate `n_steps ∈ {10, 15, 25, 50}` to confirm ratio stability.

### Claim 5: "The forward/backward asymmetry is a property of the Finsler metric, not a solver artifact"

- **Evidence provided:** None within this script. The Riemannian null baseline (wind=0) is tested in H3, not H2.
- **Literature context:** The midpoint-rule arc length $\sum F(\frac{x_i+x_{i+1}}{2}, x_{i+1}-x_i)$ is *not* the same as $\sum F(\frac{x_i+x_{i+1}}{2}, x_i-x_{i+1})$ for a Randers metric, by construction ($F(x,v) \neq F(x,-v)$). So the directional asymmetry is partly tautological *given a wind field*. The scientifically important question is whether the *magnitude* of asymmetry correlates with biological directionality — which requires the Riemannian null as a calibration.
- **Verdict:** MISSING
- **Recommended Action:** Include the Riemannian null baseline (as done in H3) within this script, or at minimum cross-reference H3 results and verify that the same test triples are used.

---

## Reproducibility Checklist

- [x] **Random seeds fixed** — `SEEDS = [42, 101, 256, 1024, 2048]` for anchor sampling; VAE loaded from deterministic checkpoint with `PRNGKey(42)`.
- [ ] **Hyperparameters logged** — Key hyperparameters (`n_anchors=2000`, `iterations=60`, `n_steps=15`, `chunk_size=50`, `MAX_PAIRS=800`) are hardcoded but not written to a structured log file (JSON/YAML). Partial: they are printed but only to stdout.
- [ ] **Data preprocessing deterministic and versioned** — Preprocessing is in a separate script (`preprocess_weinreb.py`) with fixed seed 42. However, `StandardScaler` is re-fit in this script (line 98–99) rather than loaded from the preprocessing artifacts, introducing a potential inconsistency if the subset of data changes.
- [ ] **Results include variance estimates** — `mean±std` across 5 seeds is reported in the summary table. However, no CIs and no effect sizes.
- [ ] **Baselines are appropriate and fairly implemented** — **No baseline is included in this script.** The Riemannian null is deferred to H3. This makes H2 a one-sided test without calibration.

---

## Methodological Concerns

### MC-1: Potential Tautology in the Test (CRITICAL)

The wind field $W(z)$ is constructed from RNA pseudo-velocities pointing from earlier to later timepoints (`preprocess_weinreb.py`, line 142–163). The forward direction (day2→day4→day6) is *aligned with* the velocity field by construction. Therefore:

$$F(z, v_{\text{fwd}}) = \sqrt{v^T H v + (W \cdot v)^2} - W \cdot v < \sqrt{v^T H v + (W \cdot v)^2} + W \cdot v \approx F(z, -v_{\text{fwd}})$$

whenever $W \cdot v_{\text{fwd}} > 0$, which is expected since $W$ was derived from forward clonal displacement.

This means the test is partly measuring **whether the velocity field was correctly computed**, not whether the Finsler metric captures biology. To distinguish these:
- **Control 1:** Permute the wind field (shuffle anchor assignments) and show asymmetry vanishes.
- **Control 2:** Use anti-temporal triples (day6→day4→day2) and show ratio inverts.
- **Control 3:** Use cross-fate triples (monocyte progenitor → neutrophil descendant) and show no systematic asymmetry.

### MC-2: Velocity Normalization Double-Application (WARNING)

In `preprocess_weinreb.py`, pseudo-velocities are normalised to unit variance per component (line 173–175). In this experiment, `X_pca` is further `StandardScaler`-normalised (line 98–99), but velocities are only divided by `scaler.scale_` (line 99). This is correct *only if* the PCA coordinates fed to the VAE during training used the same scaler. The scaler is fit anew here rather than loaded from a saved artifact, which creates a fragile dependency.

### MC-3: Filtering Bias from `valid` Mask (WARNING)

Lines 161–162 filter out pairs where `Lf ≤ 0` or `Lb ≤ 0` or non-finite values. If the solver systematically fails on certain trajectory types (e.g., long-distance, cross-lineage), this silent exclusion biases the sample toward "easy" pairs that are more likely to show the expected asymmetry. The fraction of excluded pairs should be reported.

---

## Suggested Experiments

| # | Experiment | Purpose | Priority |
|---|-----------|---------|----------|
| 1 | **Permuted-wind null** | Shuffle anchor velocity assignments, re-run H2 → expect ratio≈1.0, $p\gg0.05$ | High |
| 2 | **Anti-temporal control** | Reverse triples (day6→day4→day2) → expect ratio > 1.0 (inverted asymmetry) | High |
| 3 | **Riemannian null within H2** | Set `use_wind=False` on same triples → expect ratio≈1.0 | High |
| 4 | **Solver convergence ablation** | Vary `iterations ∈ {20, 60, 120, 200}` and `n_steps ∈ {10, 15, 25, 50}` → confirm ratio stability | Medium |
| 5 | **$\sigma$ cross-validation** | Proper leave-one-out or hold-out $\sigma$ selection instead of grid search | Medium |
| 6 | **Effect size reporting** | Add rank-biserial correlation + bootstrap 95% CI for main result | Medium |
| 7 | **Cross-fate triples** | Use triples where day6 cell is from a different fate than day2 cell → asymmetry should weaken | Low |
| 8 | **Multiple-comparison correction** | Apply BH-FDR across the $3\times5=15$ Wilcoxon tests | High |

---

## Findings Summary

| # | Finding | Severity | Section |
|---|---------|----------|---------|
| F1 | No multiple-comparison correction across 15 Wilcoxon tests | **CRITICAL** | Claim 1 |
| F2 | Forward/backward asymmetry is partly tautological given velocity-derived wind — missing permutation null | **CRITICAL** | MC-1 |
| F3 | No effect size (rank-biserial, Cohen's $d$) reported | **WARNING** | Claim 1 |
| F4 | No Riemannian baseline within this script | **WARNING** | Claim 5 |
| F5 | Dead code (`arc_length_forward_backward`) with different `n_steps` than actual computation | **WARNING** | Claim 4 |
| F6 | No solver convergence diagnostics logged | **WARNING** | Claim 4 |
| F7 | Only 3 $\sigma$ values; no principled bandwidth selection | **WARNING** | Claim 2 |
| F8 | `StandardScaler` re-fit instead of loaded from checkpoint | **WARNING** | MC-2 |
| F9 | Silent filtering of failed BVP pairs (fraction not reported) | **WARNING** | MC-3 |
| F10 | Hyperparameters printed to stdout but not to structured log | **NOTE** | Reproducibility |
| F11 | Train/test clone split correctly separates velocity estimation from evaluation | **STRONG** | Claim 3 |
| F12 | Multi-seed sweep with per-seed and aggregate reporting | **STRONG** | Claim 2 |
