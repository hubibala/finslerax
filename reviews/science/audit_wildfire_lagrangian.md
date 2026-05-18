# Science Audit: Experiment Plan — Lagrangian Wildfire Propagation on Real Terrain
**Auditor:** Science Auditor Agent  
**Date:** May 18, 2026  
**Plan audited:** [reviews/science/experiment_plan_wildfire_lagrangian.md](experiment_plan_wildfire_lagrangian.md)  
**Source paper:** Gahtan, Shpund & Bronstein. *Wildfire Simulation with Differentiable Randers-Finsler Eikonal Solvers.* arXiv:2603.00035 [cs.CE], Feb 2026.  
**Reference code verified:** `randers-finsler-eikonal/differentiable-eikonal-wildfire/experiments/wildfire/per_scene.py`

---

## Summary

Overall scientific rigor assessment: **needs revision before implementation**.

The experiment plan is well-structured and scientifically ambitious. The Finsler-on-mesh novelty is genuine and the hypothesis is falsifiable. However, **two blocking issues must be resolved before `experiment_wildfire_flat.py` (W1.5) or `experiment_wildfire_mesh.py` (W2.3) can be coded**: (F1) the evaluation protocol is fundamentally incompatible with AVBD output — Gahtan evaluates on all burned pixels using a full distance field, while AVBD produces arrival times at only K sparse path endpoints; (F2) the IoU@50 threshold definition in the plan is subtly wrong relative to Gahtan's actual implementation.

Six additional warnings require resolution before Phase W1 results can be compared against the 0.824 baseline in a publication, and two design decisions (temporal vs. random split; absolute vs. relative arrival time normalization) must be documented explicitly.

---

## Claims Audit

### Claim 1: "AVBD achieves per-scene Pearson correlation ≥ 0.70 vs. Gahtan's 0.824"

- **Evidence provided:** Motivated by K=200–500 BVP information gap. Target explicitly set below Gahtan.
- **Literature context:** Gahtan Figure 7 / Table 9 reports per-scene mean correlation of 0.824 ± 0.044 (std across fires, NOT across seeds) from a single training seed. The std values in `plot_per_scene_results.py` (0.089–0.206 per scene) reflect fire-to-fire variability, not run-to-run variance.
- **Verdict:** WEAKNESS
- **Recommendation:** The plan states "Reduces from Gahtan (1 run)" as justification for 3 seeds, implying Gahtan uses 1 seed to produce its mean ± std. This is incorrect: Gahtan's ±0.044 std is the *inter-scene* spread of per-scene means (not variance over seeds). HAMTools 3-seed averaging will produce cleaner per-scene estimates than Gahtan, making the comparison slightly favourable to HAMTools. Correct this interpretation in the plan and paper. Additionally, Gahtan's learning rate is `lr=1e-3` (per `per_scene.py:ExperimentConfig.learning_rate`), not `5e-4` as stated in the plan's training table — this "Matches Gahtan" claim is wrong. Reconcile or justify the lower LR choice.

---

### Claim 2: "Comparison against 0.824 is fair given AVBD uses K=200–500 BVPs vs. Gahtan's full-field eikonal"

- **Evidence provided:** The plan acknowledges the information asymmetry explicitly and lowers the target to ≥ 0.70.
- **Literature context:** Gahtan's eikonal solve produces a predicted arrival time at every grid pixel simultaneously; gradient signal per training step scales as $O(N \times M)$ where $N \times M$ is the grid size (e.g., 500×500 = 250,000 pixels). AVBD yields gradient signal from only $K \leq 500$ path endpoints. The per-step information ratio is $\geq 500:1$ in favour of Gahtan.
- **Verdict:** WEAKNESS
- **Recommendation:** The plan frames the gap as a "runtime tradeoff" but the issue is more fundamental: it is an *information* tradeoff per gradient step, not a compute tradeoff. A compute-matched comparison (run both methods for the same wall-clock GPU time) would be more scientifically rigorous and would show whether the Lagrangian approach converges at all within a comparable budget. Add a compute-matched comparison (Figure W1.6 already plans a runtime scatter — extend it to explicitly address convergence rate).

---

### Claim 3: "AVBD with n_steps=15 is adequate for wildfire domains"

- **Evidence provided:** "Comparable to Phase 1." Phase 1 used an 8×8 unit square; wildfire scenes are 500×500 pixels at 30m resolution (15 km × 15 km).
- **Literature context:** Phase 1 Code Review ([reviews/code/experiment_gahtan_phase1.md](../code/experiment_gahtan_phase1.md)) already flagged that `step_size=0.05, iterations=50` in Phase 1 is under-converged. Wildfire geodesics spanning 100–300 pixels (3–9 km) at 30m resolution require far more than 15 steps: a straight-line path across 300 pixels discretised to 15 steps has inter-node spacing of 20 pixels (600 m). At 15 steps the metric is effectively sampled only at 15 positions, grossly under-sampling the covariate field.
- **Verdict:** FLAW
- **Recommendation:** Before fixing n_steps=15, run a convergence test: compute arc length error as a function of n_steps (5, 15, 30, 60, 120) on a synthetic 500×500 raster with known analytical Randers geodesic length. Report the step count at which arc length error falls below 1%. For typical wildfire scenes, n_steps ≥ 60 is likely required. JIT compilation makes this affordable.

---

### Claim 4: "Randers (with wind) will outperform Riemannian ($b \equiv 0$) by a statistically significant margin"

- **Evidence provided:** Consistent with Gahtan Section 5 (synthetic) showing ~5–10% gap estimated.
- **Literature context:** Gahtan's synthetic experiments (Section 5) show the Randers advantage for directional anisotropy, but the real wildfire experiments (Section 6.1) do not include an explicit Riemannian ablation — they do not report Gahtan's own Riemannian baseline. The "5–10% estimated" gap in the plan is speculative.
- **Verdict:** WEAKNESS
- **Recommendation:** The W1 "Done" criterion states "statistically significant margin." With 5 initial scenes × 3 seeds = 15 data points, a paired Wilcoxon signed-rank test (W1 correlation vs. W1-Riemannian correlation, paired by scene × seed) is appropriate. Power at n=15 for a 1% true effect is approximately 0.45 — insufficient for a "significant" claim. Either increase initial validation to all 19 scenes before making this claim, or lower the requirement to "directionally consistent" and defer significance testing to the full-19-scene run.

---

### Claim 5: "3D terrain mesh achieves ≥ 2% correlation improvement on steep-terrain scenes"

- **Evidence provided:** Geometric argument: at slope $\alpha = 30°$, surface distance is $1/\cos(30°) \approx 1.155\times$ the flat-projected distance (+15.5%).
- **Literature context:** The argument is geometrically correct. However, Pearson correlation is scale-invariant: a systematic multiplicative bias in predicted arrival times (e.g., all predictions scaled by $1/\cos(\alpha)$) does *not* change the Pearson r, because correlation is invariant to linear scaling. Therefore the "projection distortion" argument for why 3D improves *correlation* specifically requires an additional condition: the projection error must be spatially non-uniform (i.e., different magnification at different pixels), introducing a non-linear distortion that hurts correlation. On a slope that is uniformly tilted, projection distortion is uniform and does not hurt correlation at all.
- **Verdict:** FLAW
- **Recommendation:** The geometric justification for W2 must be revised. The argument should be: (a) non-uniform slope causes non-uniform projection distortion, which introduces spatially-varying multiplicative errors in predicted arrival times, reducing Pearson r; (b) on a uniform slope, the Pearson r argument breaks down, but RMSE and absolute arrival time accuracy still benefit. Report RMSE alongside Pearson r for the W2 comparison, and re-stratify by **slope variability** (std of slope within the burned area) rather than mean slope.

---

### Claim 6: "Temporal stratified sampling by decile ensures coverage of early, mid, and late arrivals"

- **Evidence provided:** Stated as design choice; no validation.
- **Literature context:** Decile stratification ensures the gradient signal covers the full arrival-time distribution. However, for fires with power-law burned-area growth (common in wildfire physics — fire grows exponentially until hitting a barrier), decile stratification over-samples the late-fire phase (large area, slow spread) relative to the critical early-fire phase (small area, rapid spread). Early spread is geometrically most constrained and most sensitive to the metric.
- **Verdict:** WEAKNESS
- **Recommendation:** Consider **logarithmic-time stratification** (equal mass in each log-decile of arrival time) rather than linear-decile stratification, which better captures the rapid early-spread dynamics. Include as an ablation (replace "Temporal stratification: Uniform vs. decile-stratified" with "Uniform vs. linear-decile vs. log-decile").

---

## Reproducibility Checklist

- [x] Random seeds fixed — seeds 42 (split), 0/1/2 (training) are specified
- [ ] **Hyperparameters logged** — learning rate discrepancy (plan: 5e-4, Gahtan code: 1e-3); AVBD n_steps inadequate for domain; batch size (plan: 16, Gahtan: 32) differs — none of these discrepancies are acknowledged
- [ ] **Data preprocessing deterministic and versioned** — no git commit hash or content-hash for the Sim2Real-Fire dataset download; `download_data.py` version not pinned
- [ ] **Results include variance estimates** — 3-seed std is planned; however the test set size (15% of fires per scene) is not reported, making it impossible to assess whether per-fire variance swamps per-seed variance
- [x] Baselines are appropriate — Gahtan per-scene eikonal is the right baseline; Riemannian ablation included
- [ ] **Significance tests specified** — W1 Randers ≥ Riemannian requires paired Wilcoxon or t-test; W2 slope-stratified improvement requires specification (currently none)
- [ ] **Evaluation pixel set defined** — see FLAW F1 below; "test pixels" is ambiguous given AVBD cannot produce full-field predictions

---

## Detailed Findings

### F1 — FLAW (BLOCKER): Evaluation Protocol Incompatible with AVBD Output

**Location:** Plan § "Phase W1 Reproducibility Spec" (metric definition) and Data Pipeline step 7.

**Issue:** Gahtan computes Pearson r using:
```python
# per_scene.py:L175-184
valid = valid_mask & np.isfinite(pred_T) & np.isfinite(true_T) & (pred_T < 1e5)
# valid_mask = np.isfinite(arrival)  → ALL GT-burned pixels
correlation = np.corrcoef(pred_valid, true_valid)[0, 1]
```
Their eikonal solver produces `pred_T` at **every grid pixel** (via fast sweeping). HAMTools AVBD produces arrival times at **only K path endpoints** (K ≤ 500). For a 500×500 scene with 50,000 burned pixels and K=500, HAMTools evaluates correlation on 1% of the pixels Gahtan evaluates. Pearson r on 500 points is not directly comparable to Pearson r on 50,000 points from the same distribution, and systematic sampling bias (the K points are stratified by decile, not random) further breaks comparability.

**Recommended Action:** Choose one of the following protocols and document it explicitly:
1. **Dense evaluation mode (preferred):** At test time, for each GT-burned pixel in the test set, run one AVBD path from ignition to that pixel. Compute correlation on all reached test pixels. Cost: $O(N_\text{test})$ AVBD calls per fire — likely 500–5000 extra calls. Computationally expensive but fully comparable to Gahtan.
2. **Interpolation mode:** Run K_train AVBD paths for training; run K_test (e.g., 1000) additional paths to GT-burned test pixels (stratified by spatial grid) and use IDW interpolation to the full test set. Report interpolation residuals.
3. **Sparse evaluation mode (least preferred):** Report correlation only on the K observation points, with explicit statement that this is not directly comparable to Gahtan's protocol. Rename to "Pearson r (K-sparse)" to avoid confusion.

Do NOT use "test pixels" to refer to the K training-time observations — the plan currently conflates training observations with evaluation pixels.

---

### F2 — FLAW (BLOCKER): IoU@50 Threshold Definition is Wrong

**Location:** Plan § "Baseline Comparison" and Checklist Q5.

**Issue:** The plan states IoU@50 uses "the 50th percentile of GT arrival times." Gahtan's code uses:
```python
# per_scene.py:L199-205
t_threshold = max_time * frac   # frac=0.50
pred_burned = pred_valid <= t_threshold
true_burned = true_valid <= t_threshold
```
The threshold is **50% of the maximum GT arrival time** (i.e., halfway through the fire's duration), not the 50th percentile of the GT arrival time distribution. These quantities differ: because fire spread is typically sub-linear in time (rapid early spread, slower later spread), the 50th percentile of the arrival time distribution is *earlier* than $0.5 \times t_\text{max}$. Using the wrong threshold makes IoU@50 non-comparable to Gahtan.

**Recommended Action:** Use `t_threshold = T_arr_gt.max() * 0.5` (matching Gahtan exactly). Update the plan and implementation spec. Report all four IoU levels (IoU@25, @50, @75, @100) as Gahtan does.

---

### F3 — WARNING: Learning Rate Discrepancy with Gahtan

**Location:** Plan § "Training Pipeline" table, row "Optimizer."

**Issue:** The plan claims lr=5e-4 "matches Gahtan," but Gahtan's `ExperimentConfig.learning_rate = 1e-3` (per `per_scene.py:L106`). The plan halves the learning rate without justification.

**Recommended Action:** Either use Gahtan's lr=1e-3 for a fair comparison or justify the lower rate (e.g., "JAX optimizers require lower lr due to different gradient scaling"). Do not claim "Matches Gahtan" for a hyperparameter that differs by 2×.

---

### F4 — WARNING: Statistical Power Insufficient for W1 Significance Claim

**Location:** Plan § "Phase W1 Definition of Done": "Randers ≥ Riemannian by a statistically significant margin."

**Issue:** 5 initial scenes × 3 seeds = 15 (scene, seed) pairs. For the paired Wilcoxon signed-rank test (most appropriate given correlation values are bounded), the critical effect size detectable at power=0.80, α=0.05 requires n ≥ 25 pairs for a 1% absolute correlation difference. The plan will be underpowered for the initial 5-scene run.

**Recommended Action:** Change the W1 "Done" criterion to "directionally consistent (Randers mean > Riemannian mean over all initial scenes) and statistically significant on the full 19-scene run (Wilcoxon signed-rank, α=0.05)." This avoids overstating a preliminary result.

---

### F5 — WARNING: Phase W2 Slope Argument Fails for Pearson r on Uniform Slopes

**Location:** Plan § "Phase W2 Hypothesis" and "Geometric rationale."

**Issue:** The claim that 3D surface distance ($d_\text{surface} = d_\text{flat}/\cos\alpha$) improves Pearson r rests on the implicit assumption that slope is spatially non-uniform. On a uniformly tilted plane, $d_\text{surface} = c \cdot d_\text{flat}$ for a scene-wide constant $c = 1/\cos\bar{\alpha}$. This introduces a uniform multiplicative bias in predicted arrival times, but Pearson r is scale-invariant: $r(\hat{T}, T) = r(c \hat{T}, T)$. Therefore, 3D modeling improves absolute timing accuracy and RMSE, but **not Pearson r**, on scenes with spatially uniform slope.

**Recommended Action:** Stratify the W2 evaluation by **slope variability** (IQR or std of slope within the burned area) rather than mean slope. The prediction should be: ≥ 2% correlation improvement on scenes where slope variability > 10°. Additionally, report RMSE improvement alongside Pearson r, as RMSE will show the 3D benefit even on uniform slopes.

---

### F6 — WARNING: AVBD n_steps=15 Drastically Under-Resolves Wildfire Geodesics

**Location:** Plan § "Training Pipeline" table, row "AVBD n_steps."

**Issue:** Phase 1 Code Review ([reviews/code/experiment_gahtan_phase1.md](../code/experiment_gahtan_phase1.md)) flagged that n_steps=15 is under-converged even for an 8×8 domain. Wildfire scenes are 500×500 pixels at 30m resolution (15 km × 15 km). A geodesic from ignition to the fire perimeter could span 300 pixels = 9 km. With n_steps=15, each path segment covers 20 pixels (600 m), evaluating the covariate field at only 15 spatial locations. At 30m resolution, the Randers metric field can vary at every pixel — a 600m quadrature interval is grossly inadequate. Arc length errors from left-endpoint quadrature (flagged as a BUG in [reviews/code/experiment_gahtan_phase1.md:L21](../code/experiment_gahtan_phase1.md)) scale with step size; at 600m steps the error will dominate the signal.

**Recommended Action:** Conduct a step-count convergence study on one synthetic wildfire scene before setting n_steps for training. Expected required n_steps: 60–120 for 1% arc length error on 500-pixel domains. Report this as a calibration result.

---

### F7 — MISSING: Weather Variable Normalization Unspecified

**Location:** Plan § "Data Pipeline per Training Fire," step 6.

**Issue:** Step 6 states "normalize spatial covariates per-scene using training-set statistics (mean/std from training fires only)." Spatial covariates (elevation, slope, aspect) are scene-level constants and do not vary across fires — normalizing them on training fires is equivalent to normalizing on all fires (no leakage). However, **weather variables** ($T_\text{air}, q, u, v$) vary per fire and carry fire-specific information. If weather statistics are computed on all fires (train + val + test), this constitutes data leakage. The plan does not specify whether weather normalization is train-only.

**Recommended Action:** Explicitly state: "Weather variables are standardized using mean and std computed from training-fire weather data only. These statistics are frozen and applied to val and test fires without recomputation." Verify the implementation enforces this.

---

### F8 — MISSING: Fire Split Temporal vs. Random Must Be Decided Before Implementation

**Location:** Plan § Science Auditor Checklist Q1.

**Issue:** Gahtan's code (per `per_scene.py:ExperimentConfig.seed=42`, no timestamp-based sorting) uses a **random shuffle** of fires before splitting 70/15/15. The plan defers this decision to the Science Auditor. This is a blocking design decision because it determines whether the random-split val/test fires can include fires that are temporally earlier than some training fires — a subtle but scientifically important form of information leakage (later fires share fuel consumption history with earlier fires in the same scene).

**Recommended Action (see Q1 answer below):** Use random split as the primary protocol (seed=42, matching Gahtan) for direct comparability. Run one temporal-split experiment as a secondary robustness check to quantify the magnitude of temporal autocorrelation bias. Report both results; label them clearly.

---

### F9 — MISSING: Dataset Version Not Pinned

**Location:** Plan § "Phase W1 Reproducibility Spec," row "Dataset version."

**Issue:** "Sim2Real-Fire v1 (download_data.py)" is not reproducible. The Sim2Real-Fire Google Drive distribution has no versioned release tags (as of May 2026). If Li et al. update the dataset or if the `download_data.py` script changes, experiments cannot be reproduced from the plan alone.

**Recommended Action:** Record (a) the SHA-256 hash of each downloaded data package, (b) the download date, (c) the git commit of `differentiable-eikonal-wildfire` from which `download_data.py` is taken. Store these in a `data/wildfire_checksums.json` file committed alongside the experiment script.

---

### F10 — MISSING: AVBD Convergence Criterion Not Specified

**Location:** Plan § "Training Pipeline" table, rows "AVBD n_steps" and "AVBD iterations."

**Issue:** The plan fixes AVBD n_steps=15 and iterations=50 but specifies no convergence criterion. For training, under-converged paths give biased arc lengths and noisy gradients. For evaluation, under-converged paths may give systematically wrong arrival time predictions. The existing code ([src/ham/bio/train_geodesic.py:L13](../../src/ham/bio/train_geodesic.py)) uses `iterations=15` for "fast settings" — but this context is training, not evaluation.

**Recommended Action:** Add a convergence check: if the L2 change in path vertices between successive AVBD iterations is > ε_path (e.g., ε_path = 0.01 × pixel_spacing), flag as unconverged and either increase iterations or skip the fire. Report the fraction of unconverged paths per scene in the training log.

---

### F11 — MISSING: Ignition Point Identification Fragility

**Location:** Plan § "Data Pipeline per Training Fire," step 2.

**Issue:** "`find_ignition_point(masks)` → centroid of `masks[0]`" fails for fires with multi-pixel initial burn regions. Gahtan's code uses a 3×3 source mask around the identified ignition pixel (`source_mask[max(0,si-1):min(H,si+2), max(0,sj-1):min(W,sj+2)] = True`), which is more robust. More critically, for real fires (the ~50 real-fire events in Sim2Real-Fire), initial burn extent can be large, and `masks[0]` may represent multiple disconnected burning pixels (spot fires). A centroid of these gives a misleading ignition location that may lie in unburned terrain.

**Recommended Action:** Use the same 3×3 source mask strategy as Gahtan. For real fires, filter out events where `masks[0].sum() > 25` pixels (anomalously large initial burn) and flag these for separate analysis.

---

### STRONG-1: Falsifiability of Phase W2 Hypothesis

The W2 hypothesis is explicitly stated as falsifiable: if 3D mesh does not improve correlation on steep-terrain scenes, this is reported as an informative negative result. This is exemplary scientific practice and should be preserved in the paper framing. Negative results about when surface geometry matters (or does not matter) for fire spread modeling are directly relevant to the community.

---

### STRONG-2: Three-Seed Averaging

Using 3 training seeds where Gahtan uses 1 is methodologically stronger. This addresses run-to-run variance, which Gahtan does not report. The plan correctly interprets Gahtan's ±0.044 as inter-scene variance — the multi-seed design will additionally quantify training instability, which may be high for AVBD with K=500.

---

### STRONG-3: Decoupled Arrival Time Extraction Warning

The explicit WARNING about ±30 min quantization error from hourly frames is good scientific practice. Its inclusion as a documented limitation (not hidden) is the correct treatment.

---

## Suggested Experiments

1. **Convergence calibration (required before W1.5):** AVBD n_steps sensitivity on a synthetic 500×500 Randers scene with analytical solution. Report arc length error vs. n_steps. This determines the minimum n_steps for reliable training.

2. **Compute-matched comparison (strongly recommended for W1):** Run both HAMTools (AVBD, K=500) and Gahtan (fast sweeping) for the same wall-clock training time on one scene. Show learning curves (correlation vs. wall-clock seconds). This is the scientifically cleanest comparison of the two frameworks.

3. **Temporal split robustness check (recommended for W1):** Run W1 with temporal split (train on first 70% by ignition timestamp) on 5 scenes. Report the correlation gap between random-split and temporal-split results. If the gap is < 1%, temporal autocorrelation bias is negligible.

4. **Multi-source fire filter (required for fair evaluation):** Screen Sim2Real-Fire events for multi-spot ignition in `masks[0]` (sum > 25). Report what fraction is excluded. Evaluate separately on single vs. multi-source fires.

5. **Log-decile stratified sampling (Phase W1 ablation):** Test linear-decile vs. log-decile BVP sampling. Log-decile oversamples the rapid early-spread phase and may improve correlation at low IoU thresholds (IoU@25).

6. **Slope variability stratification for W2 (required before W2.3):** Redefine W2 slope bins as (a) low slope variability, (b) high slope variability, rather than (a) flat mean slope, (b) steep mean slope. This is the correct covariate for predicting when 3D modeling helps.

7. **RMSE and absolute timing as W2 secondary metrics (required for W2):** Add relative RMSE to the W2 evaluation table. Pearson r improvement may be < 2% even when absolute timing improves substantially on steep terrain.

---

## Answers to Science Auditor Checklist Questions

### Q1: Train/val/test contamination — temporal or random split?

**Answer: Use random split (seed=42) as primary; temporal split as secondary robustness check.**

Gahtan's `per_scene.py` uses random shuffle (seed=42) without timestamp-based sorting. Fires within a scene share the same terrain/fuel raster but have independent weather events and independent ignition locations. The primary temporal correlation risk is **weather seasonality** (summer fires behave differently from spring fires) and **fuel consumption** (early fires deplete fuel for later fires). For FARSITE-simulated fires (the bulk of Sim2Real-Fire), fuel consumption across fires is likely not modeled — each fire likely starts from the same initial fuel map. In this case, random vs. temporal split makes little practical difference for simulated fires.

For real fires (~50 events), temporal ordering matters more. The recommended protocol:
- **Primary evaluation (simulated fires):** Random split, seed=42, matching Gahtan. Reports directly comparable to 0.824.
- **Secondary evaluation (all fires, simulated + real):** Temporal split, train on first 70% by ignition date. Reports a more realistic deployment-scenario estimate.
- Explicitly document both in the paper and report both numbers.

### Q2: Multi-source fairness — is single-source setup fair?

**Answer: Fair for primary comparison, but requires screening for multi-source events.**

Gahtan's baseline 0.824 is from a single-source eikonal (one source pixel per fire). Sim2Real-Fire FARSITE simulations use single ignition points. The comparison is fair for simulated fires. However:
- The ~50 real wildfire events in Sim2Real-Fire (`real_fire_eval.py`) may include fires with multiple spot ignitions. Single-source AVBD and single-source fast sweeping will both underperform on these, so comparability is maintained — but absolute performance will be lower for multi-source real fires.
- Screen `masks[0].sum()` > 25 as a proxy for multi-source ignition. Report this fraction. Do not exclude these events from benchmarking, but flag them in the evaluation table.
- The multi-source ablation in Gahtan Table 3 uses synthetic data with controlled multi-source configurations; it does not apply to the real-data comparison and should not be cited as a fairness concern.

### Q3: Arrival time normalization — per-fire or per-scene?

**Answer: Per-fire normalization (divide each fire's arrival times by that fire's $T_\text{max}$). This matches Gahtan.**

Gahtan's relative RMSE is `rmse / max_time` where `max_time` is the maximum GT arrival time within that fire (`per_scene.py:L141`). The Pearson r computation uses raw (un-normalized) arrival times but is scale-invariant, so normalization does not change the Pearson r result — it only affects RMSE.

For `ArrivalTimeLoss`, the BVP supervision uses arrival times normalised to $[0, 1]$ per fire. This means the model learns fire *spread patterns*, not absolute fire *speed*. The limitation is:
- The model cannot distinguish a fast 6-hour fire from a slow 72-hour fire if their spatial spread patterns are identical.
- Report this explicitly as a scope limitation: "the model predicts the relative temporal order of fire arrival, not absolute fire spread rate."
- Do NOT use per-scene normalization (pooling arrival times from fires of different duration): this conflates the fire progression scale across events.

### Q4: Correlation metric — which pixels?

**Answer: All GT-burned pixels (finite arrival time). But AVBD cannot produce these without a dense evaluation protocol. This is FLAW F1 and is a blocker.**

From Gahtan's code:
```python
valid_mask = np.isfinite(arrival)  # ALL pixels burned in GT
valid = valid_mask & np.isfinite(pred_T) & np.isfinite(true_T) & (pred_T < 1e5)
correlation = np.corrcoef(pred_valid, true_valid)[0, 1]
```

The evaluation set is ALL GT-burned pixels. Gahtan's eikonal produces `pred_T` at every pixel. HAMTools AVBD produces `pred_T` at K path endpoints only.

**Required protocol:** Choose one of the three options in FLAW F1. Option 1 (dense evaluation: run K_test AVBD paths to all test-set burned pixels) is required for a paper-quality comparison to Gahtan. For the initial validation (5 scenes), running all AVBD paths to test pixels is feasible (K_test ≈ 7500 paths per fire × 15 test fires per scene × 5 scenes = 562,500 AVBD calls — expensive but finite). This should be the default evaluation protocol.

### Q5: IoU computation — what is the threshold?

**Answer: 50% of the maximum GT arrival time in that fire, NOT the 50th percentile of the GT arrival time distribution.**

From Gahtan's code:
```python
for frac in [0.25, 0.5, 0.75, 1.0]:
    t_threshold = max_time * frac   # max_time = max(GT_arrival_time for finite pixels)
    pred_burned = pred_valid <= t_threshold
    true_burned = true_valid <= t_threshold
    iou = intersection / union
```

IoU@50 binarises both GT and predicted arrival times at `t = 0.5 × max(GT arrival time)`. This measures: "how well does the model predict which half of the burned area burned in the first half of the fire's duration?" Since wildfire spread is typically accelerating early and decelerating late (Rothermel spread model), the first 50% of duration often corresponds to much less than 50% of the total burned area — the threshold is chronologically midpoint, not spatially midpoint.

**Common misinterpretation to avoid:** IoU@50 is NOT "IoU on pixels within the 50th spatial percentile of burned distance from ignition." It is purely temporal.

The plan's target of ≥ 0.40 vs. Gahtan's 0.609 is plausible given the sparse evaluation issue, but must be measured using exactly this threshold definition.

---

## Reproducibility Checklist (Final State)

- [ ] Random seeds fixed — **partially**: seeds specified, but split method (temporal vs. random) not decided
- [ ] Hyperparameters logged — **NO**: learning rate (5e-4 vs Gahtan's 1e-3), batch size (16 vs 32), n_steps (15, insufficient) all differ from Gahtan without documentation
- [ ] Data preprocessing deterministic and versioned — **NO**: no content hash for Sim2Real-Fire download
- [ ] Results include variance estimates — **partially**: 3-seed mean ± std planned; test set sizes (15% of fires per scene) not stated
- [ ] Baselines appropriate and fairly implemented — **partially**: Gahtan eikonal is the right baseline; but evaluation protocol incompatibility (F1, F2) makes direct comparison invalid without fixes

---

*Science Auditor sign-off is withheld until FLAW F1 (evaluation protocol) and FLAW F2 (IoU threshold) are resolved in the plan, and MISSING F7 (weather normalization), F8 (split method), and F9 (dataset version) are documented.*
