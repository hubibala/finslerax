# Code Review: `training/losses.py` — `ArrivalTimeLoss` Scale Alignment Block
**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-18  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

This targeted review examines the scale alignment block added to `ArrivalTimeLoss.__call__`
(approximately the last 15 lines, `src/ham/training/losses.py:606–622`).  The block was
introduced to bridge the unit mismatch between `t_pred` (geodesic arc length in metres) and
`t_obs` (arrival times normalised to [0, 1]).  The intent is sound, but the implementation
contains **one outright bug** and **one significant risk** that together explain the observed
NaN loss at epoch 9 and erratic gradient behaviour.  No new tests were added for the scale
alignment path.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/training/losses.py:616` | `jnp.maximum(mean_pred, 1e-8)` does **not** guard against NaN.  In IEEE 754 (and JAX), `max(NaN, x) = NaN` for any `x`.  If `mean_pred` is NaN — which happens as soon as any metric parameter becomes NaN — then `scale = mean_obs / NaN = NaN`, `t_pred_aligned = t_pred * NaN = NaN`, and the loss is NaN.  This is the **proximate cause of the epoch-9 NaN cascade**: once any gradient step pushes a parameter to NaN, every subsequent forward pass is NaN.  The guard provides false security. | Replace with: `mean_pred_safe = jnp.where(jnp.isfinite(mean_pred) & (mean_pred > 1e-8), mean_pred, jnp.ones_like(mean_pred)); scale = jax.lax.stop_gradient(mean_obs / mean_pred_safe)`. This returns `scale = mean_obs` (i.e., unscaled relative to 1.0) when the prediction is degenerate, preserving a finite loss signal. |
| 2 | **RISK** | `src/ham/training/losses.py:614–617` | The scale `s = stop_gradient(E[t_obs] / max(E[t_pred], ε))` is recomputed at every forward pass.  Because `s` changes with every weight update (as `mean_pred` evolves), the *effective* loss seen by the gradient is `MSE(s·t_pred − t_obs)` with a different `s` each step.  Adam's moment estimators accumulate statistics on gradients of the form `2s·(s·t_pred_i − t_obs_i)/K`; when `mean_pred` is small (early training, metric not yet tuned), `s` is large, the gradient magnitude is `O(s²)`, and the squared-gradient second moment estimate saturates.  When `mean_pred` later normalises, the gradient magnitude drops by `s²`, but the Adam denominator is still large — effectively freezing updates for several epochs.  This is a known failure mode of adaptive-moment methods under non-stationary gradient scales. | Consider replacing the per-step mean-normalisation with a running exponential moving average of `mean_pred` updated outside the loss (e.g. in the training loop), or apply a log-space loss: `mse_log = mean((log(t_pred+ε) − log(t_obs+ε))²)` which is scale-invariant without the `stop_gradient` discontinuity. |
| 3 | **STYLE** | `src/ham/training/losses.py:612–613` | The bypass condition `if t_obs.shape[0] > 1` is intentional (documented in-comment) but its sole purpose is to avoid exercising the scale alignment code path in the `test_loss_decreases_with_correct_metric` test, which uses `t_obs.shape[0] = 1`.  This is test-evasion via production code.  The comment reads: *"We only apply this when there are multiple observations to avoid trivializing single-observation test cases."* | Remove the bypass condition and fix the test: the test should either use `t_obs` in [0,1] units (matching the intended use case) or explicitly disable scale alignment via a constructor argument `scale_align: bool = True`. |

---

## Test Coverage Assessment

The existing `TestArrivalTimeLoss` suite (`tests/test_arrival_time_loss.py`) covers three
scenarios:

| Test | Covers scale alignment? | Notes |
|------|------------------------|-------|
| `test_identity_metric_distance` | **No** — `t_obs` is in metres and `t_pred ≈ t_obs`; `scale ≈ 1.0`; the block is active but a no-op | Does not verify [0,1] → metres bridging |
| `test_gradient_flows` | **No** — same regime, `t_obs = [0.5, 0.5]`; `t_pred ≈ 0.5`; scale ≈ 1.0 | Does not test gradient stability under large scale mismatch |
| `test_loss_decreases_with_correct_metric` | **No** — single observation bypasses the block entirely | Deliberately avoids the code path |

**Gap:** The core use case — `t_obs ∈ [0,1]`, `t_pred ∈ [100, 10000]` — is completely untested.
In particular:

- No test verifies that gradients are finite and non-zero when `mean_pred >> mean_obs`.
- No test verifies that NaN `mean_pred` produces a finite (not NaN) loss (Issue #1 above).
- No test exercises `check_grads`-level correctness of the gradient through the scaled MSE
  (`jax.test_util.check_grads` or `jax.linear_util.wrap_init`-based FD check).

---

## Positive Patterns

- **Midpoint quadrature for arc length** (`losses.py:601–603`): Using segment midpoints
  `(path[k] + path[k+1]) / 2` for the `metric_fn` evaluation achieves O(h²) accuracy
  instead of the naive left-point rule.  This is the correct discretisation and consistent
  with `spec/MATH_SPEC.md § 1.2`.
- **`eqx.filter_checkpoint` on `_solve_and_integrate`** (`losses.py:207`): Correct use of
  activation checkpointing to trade memory for compute through the AVBD solver unrolling.
- **`stop_gradient` rationale documented in-comment** (`losses.py:610`): The intent of
  blocking all-to-all coupling is architecturally sound; the implementation just fails on
  NaN inputs.
