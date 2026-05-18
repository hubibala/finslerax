# Science & Code Review: experiment_wildfire_flat.py (post-performance-commit)
**Auditor:** Science Auditor + Math/Code pass  
**Date:** 2026-05-18  
**Scope:** `ArrivalTimeLoss` mean-alignment change, `metric_fn` Zermelo formula, coordinate pipeline, val/test consistency

---

## Summary

Overall: **needs revision**. The Zermelo metric formula and projection helpers are mathematically correct. The mean-alignment trick introduced in `ArrivalTimeLoss` has a gradient-flow defect that is likely responsible for the plateau and the val/test gap. A second methodological issue—validating on in-sample observation pixels—causes early stopping to select on an optimistic criterion. Both need fixing before the r ≥ 0.70 target is achievable.

Current results: `test_r = 0.4918 ± 0.1956`, `IoU@50 = 0.0435`, target `r ≥ 0.70`.

---

## Finding 1 — Gradient flows through the scale factor in `ArrivalTimeLoss`

**Severity: RISK / WARNING**  
**File:** `src/ham/training/losses.py:612–617`

```python
if t_obs.shape[0] > 1:
    mean_obs = jnp.mean(t_obs)
    mean_pred = jnp.mean(t_pred)
    t_pred_aligned = t_pred * (mean_obs / jnp.maximum(mean_pred, 1e-8))
```

The scale factor $s = \bar{t}_{\text{obs}} / \bar{t}_{\text{pred}}$ is a function of the metric parameters $\theta$. The effective gradient is:

$$\frac{\partial L}{\partial \theta} = \frac{2}{K} \sum_i (s \cdot t_{\text{pred},i} - t_{\text{obs},i}) \left( s \cdot \frac{\partial t_{\text{pred},i}}{\partial \theta} + t_{\text{pred},i} \cdot \frac{\partial s}{\partial \theta} \right)$$

where $\partial s / \partial \theta = -s / \bar{t}_{\text{pred}} \cdot (1/K) \sum_j \partial t_{\text{pred},j} / \partial \theta$.  
This creates **implicit all-to-all gradient coupling** among all $K$ observations in the batch. At initialization, when geodesic arc lengths are $O(\text{km})$ and $t_{\text{obs}} \in [0,1]$, the scale is $s \approx 0.5 / \bar{t}_{\text{pred}} \approx 5 \times 10^{-4}$ and the scale gradient term is $O(s / \bar{t}_{\text{pred}}) \approx 5 \times 10^{-7}\,\text{m}^{-1}$. While the magnitude may be modest, the cross-observation coupling introduces noise into per-observation gradients and prevents the loss from directly optimizing Pearson r.

The training loss with this alignment is approximately:
$$L_{\text{aligned}} \approx \text{Var}(t_{\text{obs}}) \cdot (1 - r^2) + O(\text{higher order})$$
so it does proxy $r$, but only at the first-order approximation. The $O$ term contains the gradient-coupling contributions that introduce noise.

**Recommended Action:** Apply `stop_gradient` to the scale factor so gradients flow only through the relative predictions, not through the mean correction:

```python
if t_obs.shape[0] > 1:
    mean_obs = jnp.mean(t_obs)
    mean_pred = jnp.mean(t_pred)
    scale = jax.lax.stop_gradient(mean_obs / jnp.maximum(mean_pred, 1e-8))
    t_pred_aligned = t_pred * scale
else:
    t_pred_aligned = t_pred
```

Or, for an exact scale-invariant loss, replace MSE with the Pearson correlation loss directly:
$$L = 1 - r(t_{\text{pred}}, t_{\text{obs}})$$

---

## Finding 2 — Val r computed on in-sample observation pixels

**Severity: BUG / FLAW**  
**File:** `examples/experiment_wildfire_flat.py`, `_val_pearson_r` function

```python
def _val_pearson_r(metric, solver, scenario, cfg) -> float:
    ...
    pred = _predict_arrivals_chunked(..., scenario.obs_pixels, ...)
    gt   = np.asarray(scenario.obs_arrival_times, dtype=np.float64)
    return pearson_r(pred, gt)
```

`scenario.obs_pixels` are the same $K=50$ stratified pixels used as training targets in `ArrivalTimeLoss`. Computing validation Pearson r on the training observation set is in-sample evaluation, not validation. Early stopping (`if mean_val_r > best_val_r`) selects the checkpoint that best fits training points, not the checkpoint that generalizes.

This directly explains the 0.11-point gap between `val_r ≈ 0.60` and `test_r ≈ 0.49`.

**Recommended Action:** Hold out a separate pixel set per scenario for validation (e.g., draw a fresh random sample of 100–200 burned pixels at load time with a different seed, stored in `scenario.val_pixels`), and compute `_val_pearson_r` on that instead of `obs_pixels`.

---

## Finding 3 — `metric_fn` Zermelo formula is correct (spec/code reconciliation)

**Severity: STRONG**  
**Files:** `src/ham/models/wildfire.py:453–476`, `spec/MATH_SPEC.md § 5`

The spec formula uses $W$ as a **vector wind** with $\langle W, v \rangle_h = W^\top G v$, but the code uses $b$ as the corresponding **covector** $b = GW$, so:

$$\langle W, v \rangle_h = (G^{-1} b)^\top G\, v = b^\top v \quad \checkmark$$
$$\|W\|^2_h = (G^{-1}b)^\top G\, (G^{-1}b) = b^\top G^{-1} b \quad \checkmark$$
$$\lambda = 1 - b^\top G^{-1} b \quad \checkmark$$

The implementation:
$$F = \frac{\sqrt{\lambda\, v^\top G v + (b^\top v)^2} - b^\top v}{\lambda}$$
is 1-homogeneous in $v$ (verified analytically) and equals the Zermelo formula in covector parametrization. **This is correct.**

`project_spd` uses the closed-form 2×2 eigendecomposition (no `jnp.linalg.eigh`) — correct and vmap-safe.  
`project_b_norm` enforces $\|b\|_{G^{-1}} < \text{max\_norm}$ — correct causality bound.

---

## Finding 4 — `project_b_norm` uses hardcoded epsilon inconsistently

**Severity: STYLE**  
**File:** `src/ham/models/wildfire.py:100`

```python
scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
```

The rest of the codebase uses `GRAD_EPS = 1e-12` from `ham.utils.math` for gradient-safe divisions. Using `1e-8` here is less precise. Consistent use of `NORM_EPS = 1e-8` (from math.py) with an explicit import would be cleaner.

**Recommended Action:** Replace `1e-8` with `NORM_EPS` from `ham.utils.math`.

---

## Finding 5 — Zero-velocity perturbation introduces a fixed-direction bias

**Severity: RISK**  
**File:** `src/ham/models/wildfire.py:453–455`

```python
v_sq_raw = jnp.sum(v ** 2)
is_zero = v_sq_raw < GRAD_EPS
v_safe = jnp.where(is_zero, v + jnp.sqrt(GRAD_EPS), v)
```

Adding $\sqrt{\epsilon} \approx 10^{-6}$ to **both components** of a zero vector means the "safe" direction is always $(1, 1) / \sqrt{2}$ (the diagonal). Any gradient computation at $v \approx 0$ will pick up a bias in this direction. Since midpoints of near-stationary path segments may trigger this branch, it can introduce systematic directional artifacts.

**Recommended Action:** Replace with `safe_norm`-style perturbation: add to the squared norm rather than to the vector, or use `jnp.where(is_zero, jnp.ones_like(v) * jnp.sqrt(GRAD_EPS / 2), v)`.

---

## Finding 6 — `iou_at_50` normalization is inconsistent between pred and gt

**Severity: WARNING**  
**Files:** `examples/experiment_wildfire_flat.py:~525`, `src/ham/data/wildfire.py:180–191`

The predicted arrivals are normalized by `max(pred_arrivals)`:
```python
pred_norm = pred_arrivals / float(p_finite.max())
```
The ground-truth `arrival_times` are normalized by `t_max` (the max time-step in the raw masks). The 0.5 threshold therefore cuts different absolute ranges for pred vs. gt. The resulting `IoU@50 = 0.0435` is not meaningful as a standalone metric for this reason — it reflects the normalization mismatch as much as prediction quality.

**Recommended Action:** Normalize both pred and gt by the same quantity (e.g., `gt.max()` over the full burned area), or report IoU at a percentile threshold (e.g., first 50% of burned pixels by arrival rank) rather than a fixed 0.5 value.

---

## Finding 7 — Unused import `compute_slope_std`

**Severity: STYLE**  
**File:** `examples/experiment_wildfire_flat.py:~67`

```python
from ham.data.wildfire import (
    ...
    compute_slope_std,
    ...
)
```

`compute_slope_std` is imported but never called in the script.

---

## Reproducibility Checklist

- [x] Random seeds fixed (`seed=0` in quick run, `seed=42` for splits)
- [x] Hyperparameters logged in `get_config()`
- [x] Data preprocessing deterministic (SceneNormalizer fitted on training set only)
- [ ] **Results reported without variance** (quick run: 1 seed, so std=0.0 is a degenerate report)
- [ ] **Baselines missing**: no Riemannian (`--no_wind`) run shown; target r ≥ 0.70 not yet met
- [ ] **Val metric is in-sample** (Finding 2 above)

---

## Suggested Experiments / Fixes (Priority Order)

1. **Fix Finding 2 first** (held-out val pixels). This is free to fix and will immediately give honest model selection.
2. **Apply `stop_gradient` to scale (Finding 1)**. This should stabilize gradients and is the most likely cause of the plateau at val_r ≈ 0.60.
3. **Run the Riemannian ablation** (`--no_wind`) to quantify the drift contribution.
4. **Increase epochs** (`n_epochs=100` in full config) and run 3 seeds to report mean ± std properly.
5. **Consider Pearson-r loss** as a direct scale-invariant objective if MSE-based alignment continues to underperform.
