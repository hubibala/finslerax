# Math Review: losses_arrival_time_scale

**Reviewer:** Math Reviewer Agent  
**Date:** 2026-05-18  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Commit under review:** dd204da (scale alignment) + uncommitted `stop_gradient` patch  
**Source file:** [src/ham/training/losses.py](../../src/ham/training/losses.py)

---

## Summary

**Major Issues.** The `ArrivalTimeLoss.__call__` method introduced in commit dd204da contains two independent CRITICAL mathematical errors. First, applying `jax.lax.stop_gradient` to the scale factor $s$ makes the loss scale-invariant: the metric parameters $\theta$ receive no gradient signal about the absolute magnitude of arrival times, which directly causes IoU@50 = 0 (the isochrone threshold comparison requires correct absolute values). Second, the numerical floor of $10^{-8}$ on `mean_pred` is eleven orders of magnitude below the expected physical range of geodesic arc-lengths ($\sim 10^2$–$10^4$ m); when the metric approaches collapse, the scale factor reaches $O(10^7)$ and amplifies the gradient accordingly, causing the observed NaN at epoch 9. The correct fix requires either (a) a fixed physical calibration constant replacing the dynamic per-batch scale, or (b) removing `stop_gradient` while raising the floor to a physically meaningful value such as $1.0$ m.

---

## Formula-by-Formula Audit

### 1. Discrete arc-length $T_i^{\mathrm{pred}}$

- **Spec Reference:** spec/MATH_SPEC.md § 1.2 (Energy Functional); class docstring references Gahtan et al. (2026) § 5.
- **Literature Reference:** Bao, Chern, Shen, *Introduction to Riemann-Finsler Geometry* (2000), Ch. 6 (arc-length integral). Using $F$ (not $E$) for arc-length is the 1-homogeneous convention.
- **Implementation (lines 592–603):**
  ```python
  segments = jnp.diff(path, axis=0)           # (T, D)  Δx_k = x_{k+1} − x_k
  midpoints = (path[:-1] + path[1:]) / 2.0   # (T, D)  x_{k+1/2}
  step_costs = jax.vmap(metric.metric_fn)(midpoints, segments)
  return jnp.sum(step_costs)
  ```
  Implements $T_i^{\mathrm{pred}} = \sum_{k=0}^{N-1} F\!\left(x_{k+1/2},\, x_{k+1}-x_k\right)$.
  By 1-homogeneity of $F$, $F(x, \lambda v) = \lambda F(x, v)$ for $\lambda > 0$, so using un-normalized segment vectors is correct: the sum equals $\int_0^1 F(\gamma, \dot\gamma)\,dt$ to $O(h^2)$ via midpoint quadrature.
- **Verdict:** CORRECT
- **Notes:** The midpoint quadrature gives second-order accuracy in step size, which is appropriate.

---

### 2. Mean-ratio scale factor $s$ and its gradient  *(CRITICAL)*

- **Spec Reference:** No arrival-time scale normalization appears in spec/MATH_SPEC.md. The spec formulates the loss directly as MSE without a scaling step.
- **Implementation (lines 613–617):**
  ```python
  mean_obs  = jnp.mean(t_obs)
  mean_pred = jnp.mean(t_pred)
  scale = jax.lax.stop_gradient(mean_obs / jnp.maximum(mean_pred, 1e-8))
  t_pred_aligned = t_pred * scale
  ```
- **Mathematical statement of the problem.**  
  Let $s = \mathrm{sg}\!\left(\bar{t}^{\mathrm{obs}} / \max(\bar{T}^{\mathrm{pred}}, \varepsilon)\right)$ where $\mathrm{sg}$ denotes `stop_gradient`. The loss is:

  $$\mathcal{L} = \frac{1}{K}\sum_{i=1}^K \bigl(s\cdot T_i^{\mathrm{pred}}(\theta) - t_i^{\mathrm{obs}}\bigr)^2$$

  Because `stop_gradient` severs the path from $\theta$ through $\bar{T}^{\mathrm{pred}}$ to $s$, the gradient is:

  $$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{2s}{K}\sum_{i=1}^K \bigl(s\cdot T_i^{\mathrm{pred}} - t_i^{\mathrm{obs}}\bigr)\frac{\partial T_i^{\mathrm{pred}}}{\partial \theta}$$

  This gradient is exactly zero when $s\cdot T_i^{\mathrm{pred}} = t_i^{\mathrm{obs}}$ for all $i$, i.e., when $T_i^{\mathrm{pred}} \propto t_i^{\mathrm{obs}}$. The constant of proportionality is $\bar{T}^{\mathrm{pred}}/\bar{t}^{\mathrm{obs}}$, which is unconstrained by the loss. The metric therefore converges to a state where **the arrival-time ordering is correct but the absolute scale is arbitrary**. Since IoU@50 evaluates the isochrone $\{x : T^{\mathrm{pred}}(x) \leq T_{50}^{\mathrm{obs}}\}$ against ground-truth using the same absolute threshold, and $T^{\mathrm{pred}} \in [10^2, 10^4]$ m while $T_{50}^{\mathrm{obs}} \in [0,1]$, the predicted isochrone is always empty and IoU = 0.

- **Verdict:** CRITICAL  
- **File/Line:** [src/ham/training/losses.py](../../src/ham/training/losses.py#L616), line 616  
- **Recommended Action:** Replace the dynamic per-batch scale (lines 613–620) with a fixed calibration constant `self.t_scale` (units: same as `T_pred`, e.g. meters):

  ```python
  # __init__ addition:
  #   t_scale: float = eqx.field(static=True)
  #   (set to characteristic arc-length of the domain, e.g. 1e4 for 10 km fire domain)

  # __call__ replacement for lines 604–622:
  t_pred = jax.vmap(single_arrival_time)(x_obs)
  t_pred_normalized = t_pred / self.t_scale   # now dimensionless, same units as t_obs ∈ [0,1]
  mse = jnp.mean((t_pred_normalized - t_obs) ** 2)
  return mse * self.weight
  ```

  This retains gradient flow through the absolute scale of $T_i^{\mathrm{pred}}$.  
  If a fixed calibration constant is unavailable, the alternative is to **remove** `stop_gradient` (allowing gradient flow through $s$) while raising the floor (see issue 3 below).

---

### 3. Numerical floor `1e-8` on `mean_pred`  *(CRITICAL)*

- **Spec Reference:** spec/MATH_SPEC.md § 6.1 (Epsilon Regularization) — discusses stabilization of $F_\epsilon$, but does not address arc-length normalization floors.
- **Implementation (line 616):**
  ```python
  scale = jax.lax.stop_gradient(mean_obs / jnp.maximum(mean_pred, 1e-8))
  ```
- **Mathematical statement of the problem.**  
  For typical wildfire scenarios, geodesic arc-lengths satisfy $T_i^{\mathrm{pred}} \in [10^2, 10^4]$ m, giving $\bar{T}^{\mathrm{pred}} \in [10^2, 10^4]$. The floor $\varepsilon = 10^{-8}$ is $10^{10}$ to $10^{12}$ times smaller than any physically realized value. When the metric undergoes early-training collapse (spray coefficients driven toward zero by competing losses), $\bar{T}^{\mathrm{pred}} \to 0$ until the floor clamps it:

  $$s \approx \frac{\bar{t}^{\mathrm{obs}}}{10^{-8}} \approx \frac{0.5}{10^{-8}} = 5\times 10^7$$

  Even with `stop_gradient`, $s$ enters as a constant coefficient in the gradient:

  $$\left\|\frac{\partial \mathcal{L}}{\partial \theta}\right\| \approx \frac{2s \cdot \bar{t}^{\mathrm{obs}}}{K} \cdot \left\|\frac{\partial \bar{T}^{\mathrm{pred}}}{\partial \theta}\right\| = O\!\left(10^7\right)\cdot \left\|\frac{\partial \bar{T}^{\mathrm{pred}}}{\partial \theta}\right\|$$

  This amplification causes a gradient overflow (NaN) consistent with the observed failure at epoch 9.

- **Verdict:** CRITICAL  
- **File/Line:** [src/ham/training/losses.py](../../src/ham/training/losses.py#L616), line 616  
- **Recommended Action:** If the mean-ratio normalization pattern is retained (without `stop_gradient`), raise the floor to a physically meaningful value:

  $$\varepsilon_{\mathrm{floor}} \geq 1.0 \text{ m}$$

  This bounds $s \leq \bar{t}^{\mathrm{obs}} / 1.0 \leq 1.0$, preventing any gradient amplification. For example:

  ```python
  scale = mean_obs / jnp.maximum(mean_pred, 1.0)   # floor at 1 metre; no stop_gradient
  ```

  Using the fixed-calibration approach (recommended in issue 2) makes this floor question moot.

---

### 4. Scale-invariance and its implication for Pearson correlation  *(WARNING)*

- **Spec Reference:** No corresponding section in spec/MATH_SPEC.md.
- **Implementation (lines 613–621):** The full loss with `stop_gradient(s)` is:

  $$\mathcal{L} = s^2 \cdot \frac{1}{K}\sum_i (T_i^{\mathrm{pred}})^2 \;-\; 2s \cdot \frac{1}{K}\sum_i T_i^{\mathrm{pred}} t_i^{\mathrm{obs}} \;+\; \frac{1}{K}\sum_i (t_i^{\mathrm{obs}})^2$$

  The last term is constant w.r.t. $\theta$. Minimizing $\mathcal{L}$ over many steps (as $s$ is implicitly updated each batch) is therefore approximately equivalent to maximizing:

  $$\frac{\sum_i T_i^{\mathrm{pred}} t_i^{\mathrm{obs}}}{\sum_i (T_i^{\mathrm{pred}})^2 / (2s)}$$

  which is a scale-weighted inner product — closely related to but not identical to the Pearson correlation coefficient $r(T^{\mathrm{pred}}, t^{\mathrm{obs}})$. The key difference: Pearson normalizes by both standard deviations (mean-subtracted), while this formulation does not subtract means. In either case, the loss is **invariant to multiplying all $T_i^{\mathrm{pred}}$ by a global constant**, meaning the metric's overall amplitude is unconstrained.

- **Verdict:** WARNING  
- **File/Line:** [src/ham/training/losses.py](../../src/ham/training/losses.py#L613-L621), lines 613–621  
- **Notes:** If a fully scale-invariant loss is desired (e.g., learning the wave-front shape without caring about speed), an explicit Pearson correlation loss is mathematically cleaner:

  $$\mathcal{L}_{\mathrm{corr}} = 1 - r(T^{\mathrm{pred}}, t^{\mathrm{obs}})$$

  However, this cannot be used for IoU evaluation. If IoU is the downstream metric, a scale-invariant loss is incorrect by construction.

---

### 5. Python-level branch on `t_obs.shape[0]` in a JAX-traced function  *(NOTE)*

- **Implementation (line 613):**
  ```python
  if t_obs.shape[0] > 1:
  ```
- **Verdict:** NOTE  
- **Notes:** `t_obs.shape[0]` is a static Python integer resolved at trace time; JAX handles this correctly. The `else` branch is dead code at batch size $K=1$ and never traced otherwise. No mathematical error, but the asymmetric treatment (scale-aligned for $K>1$, raw for $K=1$) means the loss surface is qualitatively different between training (large $K$) and single-point evaluation, which may cause unexpected evaluation behavior.

---

### 6. Comment accuracy  *(NOTE)*

- **Implementation (lines 606–612):**
  ```
  # stop_gradient on the scale factor prevents implicit all-to-all gradient
  # coupling across observations — gradients flow only through relative predictions.
  ```
- **Verdict:** NOTE  
- **Notes:** The stated rationale is technically accurate (stop_gradient does prevent the all-to-all coupling), but is incomplete and misleading: it omits that stop_gradient simultaneously eliminates all gradient signal about absolute scale. A correct comment would read: "Note: stop_gradient also prevents the metric from learning the absolute magnitude of arrival times; use only if absolute scale is calibrated externally."

---

## Open Questions

1. **Calibration constant for `t_scale`:** What is the expected physical range of $\bar{T}^{\mathrm{pred}}$ in meters for the wildfire domain? This determines the correct value of `t_scale`. It depends on the spatial grid resolution and the Finsler metric parameterization (e.g., is `metric_fn` returning raw meters or a dimensionless ratio?).

2. **IoU evaluation code:** Does the IoU@50 evaluation apply the same scale normalization as the loss, or does it compare raw `t_pred` directly against `t_obs`? If the evaluation also normalizes by `mean_pred`, then IoU may have been broken even before the scale-alignment commit, and the root cause may be elsewhere in the pipeline.

3. **Gradient coupling without `stop_gradient`:** If `stop_gradient` is removed (as recommended), the gradient through $s = \bar{t}^{\mathrm{obs}} / \bar{T}^{\mathrm{pred}}$ introduces a cross-prediction coupling term $-T_j^{\mathrm{pred}} \cdot \bar{t}^{\mathrm{obs}} / (K \cdot (\bar{T}^{\mathrm{pred}})^2)$ per observation $j$. This is bounded and correct, but may slow convergence when $K$ is large. Is this acceptable in the Gahtan experiment setup?

4. **NaN at epoch 9 specifically:** The floor clamp at $10^{-8}$ would cause gradient explosion only after `mean_pred` has already collapsed. What loss or regularizer drives `mean_pred` toward zero in the first place? Auditing the other loss terms active in experiment phase that includes `ArrivalTimeLoss` is recommended.
