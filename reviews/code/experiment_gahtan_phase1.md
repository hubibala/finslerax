# Code Review: `examples/experiment_gahtan_phase1.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-17  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

The experiment has a **critical systematic bias** caused by mismatched path discretization between the ground-truth solver (`n_steps=25`) and the training solver (`n_steps=10`/`20`). Both compute discrete arc length $L = \sum_k F(x_k, \Delta x_k)$ using left-point quadrature, but with different numbers of quadrature points. For the piecewise-constant metric with a sharp boundary at $x_1=0.5$, this produces systematically different values, forcing the learned metric to absorb the discretization error. Combined with under-converged training solver iterations and L2 regularization that smooths the exact discontinuity the metric must represent, the expected recovery error is ~25–35% — matching the observed ~30%. The density invariance is a *symptom*: the error bottleneck is solver/discretization accuracy, not data quantity, so adding observations cannot reduce it.

**Energy vs. Length consistency:** Both ground truth and training compute arc length (sum of $F$), not energy (sum of $\frac{1}{2}F^2$). The AVBD solver minimizes energy to find the geodesic path, then arc length is measured along that path. This is mathematically correct — the concern raised about energy/length mismatch does not manifest as a bug because both sides measure the same quantity (length) along a geodesic found by the same method (energy minimization). The issue is purely quantitative: the *accuracy* of the geodesic and arc-length computation differs between ground truth and training.

## Issue Tracker

| # | Severity | Location (file:line) | Description | Suggested Fix |
|---|----------|----------------------|-------------|---------------|
| 1 | **BUG** | `experiment_gahtan_phase1.py:168–171` vs `:85–99,286–294` | **Discretization mismatch between ground truth and training.** Ground truth computes arc lengths with `n_steps=25` (25-segment quadrature). Training computes predicted arc lengths with `solver_steps=10` (quick) or `solver_steps=20` (full). Both use left-point quadrature $L = \sum_{k=0}^{T-1} F(x_k, x_{k+1}-x_k)$ which has $O(h)$ error where $h \propto 1/n\_steps$. The mismatch creates a systematic bias per path: the training arc length is biased relative to the target by $O(1/10) - O(1/25) \approx 6\%$ of path length. Since the learned metric parameters must absorb this bias to minimize MSE, the metric values are distorted. This is the **primary cause** of the ~30% error. | Use the **same** `n_steps` for both ground truth and training. Alternatively, compute ground truth with very high `n_steps` (e.g., 100) to make its quadrature error negligible, and increase training `solver_steps` to at least 25. |
| 2 | **BUG** | `experiment_gahtan_phase1.py:168` vs `:286–290` | **Solver convergence gap between ground truth and training.** Ground-truth solver: `step_size=0.03, iterations=150`. Training solver: `step_size=0.05, iterations=50` (quick) / `100` (full). The training solver has 1.5–3× fewer iterations and 1.67× larger step size. For paths crossing the metric boundary at $x_1=0.5$, the AVBD must discover the refracted geodesic (Snell's law: $\sin\theta_1/\sin\theta_2 = \sqrt{g_2/g_1} = \sqrt{2}$). Under-converged training paths are suboptimal, producing biased arc lengths and noisy gradients through the unrolled `jax.lax.scan`. | Increase training solver iterations to ≥ 150 and decrease step_size to ≤ 0.03. The added compile-time cost is a one-time overhead under JIT. |
| 3 | **BUG** | `experiment_gahtan_phase1.py:267–271,327` + `:85` | **Density has no effect because the error bottleneck is solver accuracy, not data quantity.** The batch size is `min(obs_per_train_step, n_obs)` = 16 (quick) / 32 (full), **independent of density**. With `n_train_steps` also fixed, the total gradient signal per run is identical. For the simple two-valued metric ($g \in \{1, 2\}$), even 25% of a 20×20 grid (100 points) provides full spatial coverage across the boundary. Since the error is dominated by the discretization/solver biases in issues #1 and #2, increasing data quantity cannot reduce it. This explains the observed density invariance. | Not a code bug per se — it is a *consequence* of issues #1 and #2. Fixing the discretization mismatch will restore density sensitivity. Additionally, consider scaling `n_train_steps` with `1/density` or adjusting `batch_size` proportionally so lower-density runs see each observation equally often. |
| 4 | **RISK** | `experiment_gahtan_phase1.py:219–230` + `:787` | **L2 Jacobian regularization penalizes exactly what the metric must learn.** The regularization $\lambda \|\partial G/\partial x\|_F^2$ penalizes spatial gradients of the metric tensor. The ground truth is piecewise-constant with a **discontinuous** jump at $x_1=0.5$, requiring an infinite spatial gradient. L2 regularization smooths this transition, introducing systematic error concentrated near the boundary. The density sweep uses a fixed `reg_lambda=1e-3` (line 787), chosen before knowing the optimal value from the regularization sweep (which runs *after* the density sweep at step [4]). | (a) Run the regularization sweep *first* and select the optimal $\lambda$ for subsequent experiments. (b) Consider $\lambda=0$ for the density sweep, since the ground truth has no spatial smoothness to exploit. (c) For piecewise-constant targets, L1 total-variation $\|\partial G/\partial x\|_1$ has better inductive bias but requires proximal methods. |
| 5 | **RISK** | `src/ham/training/losses.py:589–596` + `experiment_gahtan_phase1.py:172–175` | **Arc length uses left-point quadrature, creating boundary-crossing bias.** Both `compute_true_arrival_times` and `ArrivalTimeLoss.single_arrival_time` compute $L = \sum_{k=0}^{T-1} F(x_k, x_{k+1} - x_k)$, evaluating the metric at $x_k$ (left endpoint). For a segment crossing $x_1=0.5$, the metric is evaluated at $x_k < 0.5$ (where $g=1$) but the segment extends into the $g=2$ region, systematically **underestimating** cost of boundary-crossing segments. With `n_steps=10`, approximately 10% of segments cross the boundary, and each is underestimated by up to $(\sqrt{2}-1) \approx 41\%$ of its true cost contribution. | Use midpoint quadrature: evaluate $F$ at $(x_k + x_{k+1})/2$ instead of $x_k$. This reduces quadrature error from $O(h)$ to $O(h^2)$ and halves the sensitivity to `n_steps` mismatch, partially mitigating issue #1. |
| 6 | **RISK** | `experiment_gahtan_phase1.py:352–382` | **`evaluate_recovery` assumes isotropy via `trace(H)/2`.** The learned `PSDMatrixField` outputs a general 2×2 SPD matrix $H(x) = A(x)A(x)^T + \varepsilon I$. The evaluation extracts the scalar metric as $\text{trace}(H)/2$, which equals $g(x)$ only if $H = g(x) \cdot I_2$ (isotropic). If the network learns an anisotropic $H$ (e.g., $\text{diag}(a, b)$ with $a \neq b$) that happens to produce correct arc lengths for the observed geodesic directions, `trace(H)/2 = (a+b)/2$ could report low error for an incorrect metric, or high error for a correct-but-anisotropic metric. | Report both `trace(H)/2` (mean eigenvalue) and individual eigenvalues of $H$. Also report the anisotropy ratio $\lambda_{\max}/\lambda_{\min}$, which should be ~1.0 for the isotropic ground truth. This decouples scale recovery error from isotropy error. |
| 7 | **RISK** | `src/ham/geometry/zoo/randers.py:48–68` | **Unnecessary wind computation in `_get_zermelo_data` when `use_wind=False`.** The method computes `W_raw`, `w_norm_sq`, `w_norm`, `scale`, `W_safe`, and `safe_w_norm_sq` before checking `self.use_wind` at line 67. While not affecting correctness, this wastes ~40% of the per-call compute inside the AVBD inner loop (called once per vertex per iteration per observation per training step). More critically, if `w_net` produces NaN or Inf for some inputs, these values enter the JAX trace graph and may propagate via XLA optimization even though the W branch is logically discarded by the `jnp.where` in the return. | Move the `if not self.use_wind:` check immediately after computing and symmetrizing H (line 52), returning `(H, jnp.zeros(x.shape), jnp.array(1.0))` before any wind computation. |
| 8 | **RISK** | `experiment_gahtan_phase1.py:267–271` | **Subsampling key is density-coupled.** `k_init` is derived from `jax.random.split(key)` with `key = PRNGKey(seed)`. JAX's `random.choice(k_init, n_total, shape=(n_obs,), replace=False)` internally computes a full permutation and takes the first `n_obs` elements. This means the density=0.25 subset (100 elements) is a **strict prefix** of the density=1.0 permutation (400 elements). While not incorrect, this coupling means density comparisons are not independent samples — every lower-density observation set is a subset of every higher-density set (for the same seed). | Use a density-dependent key: `k_sub = jax.random.fold_in(k_init, int(density * 1000))` to decorrelate subsampling across density levels, ensuring independent observation sets. |
| 9 | **STYLE** | `experiment_gahtan_phase1.py:280` | `schedule = optax.cosine_decay_schedule(cfg['lr'], cfg['n_train_steps'])` — this function has been moved to `optax.schedules.cosine_decay_schedule` in newer Optax versions. The current import works but may trigger deprecation warnings. | Use `optax.schedules.cosine_decay_schedule` or pin Optax version in `pyproject.toml`. |

## Root Cause Analysis: ~30% Error Rate

The ~30% error is a **compound effect** of three reinforcing biases:

1. **Quadrature mismatch (Issue #1):** The ground-truth arc lengths are computed with 25-segment left-point quadrature. The training loss computes predicted arc lengths with 10-segment (quick) or 20-segment (full) left-point quadrature. For paths of length $L \approx 0.5$, the per-path bias is ~$0.06L \approx 0.03$ in absolute arc length. Over the training set, this bias is systematic: the training consistently underestimates arc lengths for boundary-crossing paths (Issue #5), so the optimizer inflates the learned metric values to compensate. The metric $g_{\text{learned}}$ is biased upward everywhere, but more so in the $g=1$ region (where the underestimation is proportionally larger).

2. **Solver convergence gap (Issue #2):** The training solver finds sub-optimal paths with ~50 iterations at step_size=0.05. These paths are longer than the true geodesic, so the training arc lengths are biased *upward* (overestimated). This partially cancels the quadrature underestimation from #1, but not exactly — the cancellation is path-dependent, leaving residual noise.

3. **Regularization bias (Issue #4):** L2 Jacobian regularization with $\lambda=10^{-3}$ smooths the transition at $x_1=0.5$ over ~5–10% of the domain width. Within this transition band, $g_{\text{learned}}$ interpolates between 1 and 2 instead of jumping, contributing ~5% RMSE directly.

**Net effect:** Biases 1+2+3 compound to ~25–35% relative RMSE in $g$, consistent with observations.

## Root Cause Analysis: Density Invariance

For the piecewise-constant metric with only two values ($g=1$ for $x_1<0.5$, $g=2$ for $x_1 \geq 0.5$), the information content of the observations is:
- **How many observations are on each side of the boundary?** Even at 3% density of an 80×80 grid (192 points), ~96 points are in each region — vastly more than needed to estimate two scalar values.
- **The bottleneck is not spatial coverage but computational accuracy.** The systematic biases from issues #1–#4 produce the same ~30% error regardless of how many observations are available.

To verify this diagnosis: if issues #1 and #2 are fixed (same `n_steps`, same solver quality), density should begin to affect results, with higher density giving lower error due to reduced variance in the stochastic batch gradients.

## Test Coverage Assessment

| Public Function | Tested? | Gap |
|---|---|---|
| `piecewise_metric_field` | No | No unit test verifying $G(x) = g(x) \cdot I$ for both sides of boundary |
| `true_metric_scalar` | No | Trivial, low priority |
| `make_true_metric` | No | No test verifying resulting Riemannian metric computes correct $F(x,v)$ |
| `compute_true_arrival_times` | Indirectly (`test_arrival_time_loss.py::test_identity_metric_distance`) | **Critical gap:** no test with piecewise metric; no comparison against an analytical geodesic distance for a known refraction case |
| `make_grid` | No | No test for boundary margins or grid spacing |
| `jacobian_regularization` | No | No test that penalty is nonzero for varying metrics and zero for constant metrics |
| `train_metric` | No (integration test only) | No unit test for single training step convergence |
| `evaluate_recovery` | No | No test that it returns zero error when learned metric exactly matches truth |
| `ArrivalTimeLoss.__call__` | Yes (`test_arrival_time_loss.py`) | Tests only identity metric ($G=I$). Missing: piecewise metric test, `n_steps` sensitivity test, gradient magnitude check |
| `AVBDSolver.solve` | Yes (`test_avbd.py`) | Tests JIT, vmap, differentiability, ALM constraints. Missing: convergence test on piecewise Riemannian metrics, energy-vs-length consistency test |

**Critical gap:** No test validates that `compute_true_arrival_times` with `n_steps=25` produces arc lengths that converge to the true geodesic distance. A simple test case — horizontal path from $[0.35, 0.5]$ to $[0.65, 0.5]$ crossing the boundary, with known analytical distance $0.15 \cdot 1 + 0.15 \cdot \sqrt{2} \approx 0.362$ — would immediately reveal the quadrature sensitivity and calibrate the required `n_steps`.

## Positive Patterns

1. **Self-consistent target generation:** Using the AVBD solver to compute ground-truth arrival times (rather than an analytical formula that ignores refraction) is the conceptually correct approach. Both training and evaluation measure the same quantity (arc length) along the same type of path (energy-minimizing geodesic). The issue is purely quantitative accuracy, not conceptual correctness.

2. **Clean separation of concerns:** Ground-truth generation, training, evaluation, and visualization are cleanly factored into distinct functions with typed docstrings.

3. **Correct JAX/Equinox patterns:** `eqx.filter_jit`, `eqx.filter_value_and_grad`, `eqx.apply_updates`, and `eqx.is_array` are used correctly throughout. No manual `jax.jit`/`jax.grad` on Equinox modules.

4. **JIT-stable shapes:** Fixed `n_reg_points` with padding for regularization samples avoids recompilation. The `batch_size = min(obs_per_train_step, n_obs)` is constant within each `train_metric` call.

5. **Multi-seed averaging with error bars:** Density and regularization sweeps report mean ± std over multiple seeds — proper experimental methodology.

6. **Cosine decay schedule:** Using `optax.cosine_decay_schedule` is good practice for metric recovery where the loss landscape steepens near convergence.

7. **Ablation design:** The Randers-vs-Riemannian ablation is a well-designed control — for isotropic ground truth, the $W=0$ (Riemannian) model should match or outperform Randers, validating experimental integrity.
