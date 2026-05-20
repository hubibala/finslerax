# Science Audit: Wildfire Arrival-Time Experiment — Training Pathologies
**Auditor:** Science Auditor Agent
**Date:** 2026-05-18
**Files audited:**
- `examples/experiment_wildfire_flat.py` (lines 350–580)
- `src/ham/data/wildfire.py` (lines 108–220, 360–450)
- `src/ham/training/losses.py` (lines 570–625)

---

## Summary
**Overall scientific rigor: Major Concerns.**

Two bugs invalidate the quantitative claims. `IoU@50 = 0.0` is a measurement artefact caused by a unit-scale mismatch in the evaluation pipeline, not a reflection of model quality. The monotonic degradation of `val_r` is consistent with a known failure mode of shape-only loss training under `stop_gradient`. A third issue — overlap-prone validation sampling — inflates early-epoch `val_r` and amplifies the apparent degradation. NaN propagation from epoch 9 onward is a numerical instability in the IFT adjoint rather than a genuine divergence of the physics model.

None of the observed failure modes are fundamental to the Finsler approach; all are correctable engineering issues.

---

## Claims Audit

### Claim 1: "IoU@50 = 0.0 reflects model performance on fire perimeter prediction"

- **Evidence provided:** Single test run (10 epochs, 32 fires, 100 eval pixels/fire).
- **Root-cause analysis:**

  At load time (`wildfire.py:395`):
  ```
  arrival_times = arrival_times_hours / t_max      # ∈ [0, 1]
  ```
  In `evaluate_fire` (`experiment_wildfire_flat.py:534–541`):
  ```python
  gt_arrival = scenario.arrival_times[r, c]        # already ∈ [0, 1]
  gt_ref     = max(gt_arrival)                      # ≤ 1.0
  pred_norm  = pred_arrivals / gt_ref               # pred_arrivals in metres ≈ 100–10,000
                                                    # → pred_norm ≈ 100–10,000
  ```
  `iou_at_50` (`wildfire.py:196`):
  ```python
  pred_bin = (pred_raster <= 0.5) & burned         # NEVER true when pred_norm >> 1
  ```
  Because `gt_ref ≤ 1.0`, dividing physical arc-lengths (in metres) by a number ≤ 1 does not
  rescale them to [0, 1]. The threshold of 0.5 is never crossed; `pred_bin` is the empty set;
  IoU = 0 by construction.

- **Literature context:** The standard practice for isochrone IoU is to normalise both
  predicted and ground-truth arrival fields to the same relative time range [0, 1] before
  thresholding (see Finney & McHugh 2014, *Int. J. Wildland Fire* 23:1077–1092).
  The fix is one line: divide `pred_arrivals` by `max(pred_arrivals)` rather than by `gt_ref`.

- **Verdict:** **CRITICAL** — Invalidates the IoU metric entirely. All published IoU@50 = 0.0
  results are artefacts of the normalisation error, not evidence of model failure.

- **Recommended action:** Replace
  ```python
  pred_norm = pred_arrivals / gt_ref
  ```
  with
  ```python
  pred_max  = float(np.max(pred_arrivals)) if np.max(pred_arrivals) > 1e-8 else 1.0
  pred_norm = pred_arrivals / pred_max        # ∈ [0, 1] relative arc-length
  ```
  Both arrays are then on the same 0-to-1 relative scale; the 0.5 threshold is interpretable
  as "the first half of the fire's temporal span."

---

### Claim 2: "val_r degrades monotonically from 0.54 to 0.38 during training"

- **Evidence provided:** 10-epoch training curve (loss ↓, val_r ↓).
- **Diagnosis — shape-only gradient via `stop_gradient`:**

  In `ArrivalTimeLoss.__call__` (`losses.py:612–619`):
  ```python
  scale = jax.lax.stop_gradient(mean_obs / jnp.maximum(mean_pred, 1e-8))
  t_pred_aligned = t_pred * scale
  mse = jnp.mean((t_pred_aligned - t_obs) ** 2)
  ```
  `stop_gradient` on `scale` means gradient information about **absolute geodesic length** is
  severed. The optimiser sees only the MSE objective on the scaled shape of `t_pred`.
  The minimum of
  $$\mathcal{L} = \frac{1}{K}\sum_i \bigl(\hat{s}\, T_i^{\rm pred} - T_i^{\rm obs}\bigr)^2,
  \quad \hat{s} = \text{stop\_grad}\!\left(\frac{\bar{T}^{\rm obs}}{\bar{T}^{\rm pred}}\right)$$
  is achieved by any $T^{\rm pred}$ whose *shape* (relative ordering after centering) matches
  $T^{\rm obs}$. As training collapses variance in `t_pred` toward the mean, the spatial
  gradient of the learned metric flattens and Pearson r at held-out `val_pixels` drops.

- **Diagnosis — partial confirmation by the loss plateau:**
  Loss stabilises at ≈ 0.155 (epochs 3–8) while val_r continues falling. This is consistent
  with the model learning a locally flat metric that achieves low MSE at `obs_pixels` (by
  clustering predictions near the mean) while degrading the global arrival-wave structure.

- **Literature context:** Shape-only training without an absolute-scale anchor is known to
  cause "rank collapse" in metric learning; see Musgrave et al. (2020), *ECCV*,
  arXiv:2003.08505. An anchor loss (e.g., fixing the scale of one reference observation
  outside the `stop_gradient`) prevents this.

- **Verdict:** **WARNING** — The loss design is physically motivated but incomplete.
  Severing the absolute-scale gradient is intentional (to decouple shape from units), but
  the absence of any scale-regularising term allows the metric to degenerate.

- **Recommended action:** Add a lightweight scale-regularisation term
  $\lambda \cdot (\log \bar{T}^{\rm pred} - \log c_{\rm ref})^2$ where $c_{\rm ref}$ is the
  expected physical geodesic length (e.g., `pixel_spacing_m * n_obs_steps`). This keeps the
  metric from collapsing while preserving the shape-focused MSE.

---

### Claim 3: "val_pixels are independent of obs_pixels"

- **Evidence provided:** Comment in `wildfire.py:410`:
  > "Held-out validation pixels: draw a fresh stratified sample with a different seed so
  >  val_pixels are independent of the training observation set."

- **Analysis:** `stratified_sample_observations` divides the burned area into 10 equal-width
  time bins. With `k_train_obs = 50` the function draws ~5 pixels/bin for obs. With
  `n_samples = min(100, 2*k_train_obs) = 100` it draws ~10 pixels/bin for val. Both calls
  operate on the **same candidate pool** within each bin.

  Two overlap modes exist:
  1. **Large fires:** Different seeds draw different pixels → genuine independence. ✓
  2. **Small fires** (few burned pixels): When `total_candidates ≤ n_samples` the code
     falls back to **`replace=True`** sampling (`wildfire.py:134`):
     ```python
     idx = rng.choice(total_candidates, size=n_samples, replace=True)
     ```
     With replacement, the full pixel pool is used for both obs and val; the probability that
     a specific obs pixel appears in val ≈ 1 − (1 − 1/N)^100 → ~1 for small fires. In the
     extreme case of a 50-pixel fire, every val_pixel is drawn from the 50 obs pixels, giving
     complete leakage.

- **Verdict:** **WARNING** — Independence holds only for large fires. The validation
  split degrades to near-identity for small fires, inflating early val_r and creating a
  false impression of severe degradation as training progresses.

- **Recommended action:** Generate obs and val splits simultaneously without replacement
  from the full burned pool, ensuring strict disjointness:
  ```python
  all_pixels = stratified_sample_observations(arrival_times_hours, n_obs + n_val, seed)
  obs_pixels  = all_pixels[:n_obs]
  val_pixels  = all_pixels[n_obs:]
  ```

---

### Claim 4: "NaN at epoch 9–10 indicates training instability"

- **Evidence provided:** `loss=nan`, `val_r=nan` at epochs 9–10.
- **Analysis:** The IFT adjoint for AVBD calls `jnp.linalg.lstsq(..., rcond=None)`.
  With `rcond=None`, JAX/NumPy uses the default machine-precision threshold
  $\text{rcond} = \max(M,N) \cdot \varepsilon \cdot \sigma_{\max}$. For near-singular
  path Jacobians (which arise when predicted trajectories cluster along a degenerate
  geodesic) this threshold is too permissive: a near-zero singular value passes the filter
  and its reciprocal produces an enormous gradient component, causing NaN on the next
  parameter update step.

  The gradient collapse at epoch 9 (rather than earlier) is consistent with the loss
  plateau: as the metric flattens (Claim 2 above), path curvature drops, path Jacobians
  become increasingly rank-deficient, until a single step triggers overflow.

- **Verdict:** **WARNING** — Not a fundamental physics failure. The NaN is a numerical
  consequence of two interacting bugs (metric flattening + permissive rcond).

- **Recommended action:** Set `rcond=1e-6` (or at minimum `rcond=1e-4`) in all `lstsq`
  calls within the IFT/AVBD adjoint. Additionally, gradient clipping (`optax.clip_by_global_norm`)
  before parameter updates provides a second line of defence.

---

## Reproducibility Checklist

- [x] Random seeds fixed — `seed` propagated through scenario loading; `rng = np.random.default_rng(seed)` used consistently.
- [x] Hyperparameters logged — `cfg` dict printed at run start.
- [ ] **MISSING** — `t_max` (per-fire normalisation constant) is not stored in the serialised `WildfireScenario`; re-loading a saved scenario does not allow reconstruction of the absolute time scale needed for correct IoU evaluation.
- [ ] **MISSING** — Results include only mean ± std over fires; no per-epoch variance across independent runs (different global seeds) is reported. It is impossible to separate random-seed sensitivity from genuine model convergence.
- [ ] **MISSING** — No baseline comparison. Euclidean distance from the ignition pixel is a parameter-free isochrone baseline that should be trivially beaten; it is absent from results tables.
- [ ] **MISSING** — No ablation: Finsler (Randers) vs. Riemannian-only metric on this task.

---

## Statistical Validity

| Quantity | Value | Assessment |
|----------|-------|------------|
| Training observations | 50 × 32 = 1,600 | Borderline for a spatially conditioned deep metric |
| Validation fires | 32 (implicit; same set) | No held-out fire split; leakage risk |
| Test eval pixels | 100 per fire | ~1% of a typical burned area; result is sensitive to spatial sampling |
| Test Pearson r | 0.5537 ± 0.1603 | SE ≈ 0.028; 95% CI ≈ [0.50, 0.61] — technically significant but physically weak |
| IoU@50 | 0.0000 | **Artefact** (see Claim 1) — uninformative |

The coefficient of variation of 29% (σ/μ = 0.1603/0.5537) indicates high inter-scenario
variance, suggesting either (a) strong scene-specific overfitting, or (b) genuine variability
in how well the Finsler metric captures terrain-driven fire dynamics across scenes. Without a
baseline r to compare against, the 0.55 figure cannot be interpreted as evidence of model skill.

---

## Biological / Physical Plausibility

The learning curve pattern (training loss ↓, val_r ↓ simultaneously) has a clear physical
interpretation: the model is overfitting the spatial arrangement of `obs_pixels` at the
expense of the global propagation wave. In wildfire spread, the arrival-time field is
governed by a smooth Hamilton–Jacobi eikonal equation; a well-trained Finsler metric should
produce smooth, spatially coherent geodesic fields. Flattening the metric's anisotropy toward
a near-isotropic Riemannian structure (as implied by the variance collapse) would reduce
directional sensitivity to wind and slope — the very signal the Finsler approach is designed
to capture.

This is physically testable: if the Randers wind vector $W$ learned by the metric shrinks
toward zero over epochs, the hypothesis is confirmed.

---

## Suggested Experiments

1. **Euclidean-distance baseline:** Compute IoU@50 and Pearson r using geodesic distance on
   a flat (Euclidean) 2D grid. This takes zero training time and provides the lower bound
   that the Finsler model must beat. Without this baseline, the 0.55 Pearson r is
   uninterpretable.

2. **Wind-vector norm monitoring:** Log $\|W(x)\|$ averaged over the scene at each epoch. A
   decreasing trend would confirm metric collapse under shape-only training and motivate
   scale-regularisation (Recommended action for Claim 2).

3. **Fixed scale experiment:** Remove `stop_gradient` on the scale factor for 3 epochs with
   a very small LR and compare val_r curves. This isolates the effect of the shape-only
   gradient.

4. **IoU@50 after bug fix:** After normalising `pred_arrivals` by `max(pred_arrivals)`,
   report IoU@50 at epoch 1 (before overfitting) and epoch 8 (after plateau). The bug fix
   alone may reveal non-trivial IoU performance.

5. **Strict train/val fire split:** Reserve 8 of the 32 fires for validation only (never seen
   during training). Current setup validates on training fires, making val_r a measure of
   per-fire generalisation rather than out-of-distribution performance.
