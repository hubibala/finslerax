# Science Audit: experiment_wildfire_mesh.py
**Reviewer:** Science Auditor Agent
**Date:** 2026-05-18

## Summary

The experiment design is **scientifically sound** for Phase W2. Slope stratification follows a reasonable domain-informed binning. Two RISKs about reproducibility and the W1 comparison placeholder.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | RISK | `run_synthetic_mesh` | All three slope-level scenarios are initialized with the same `jax.random.PRNGKey(cfg["seed"])`. This means initialization differences do not confound the slope comparison, which is good. However, training is independent per scenario — the metric is not jointly trained across slope levels, so "AGGREGATE Pearson r" is not a well-defined aggregate. | Document that synthetic scenarios are evaluated independently. Add a comment that the aggregate is only for smoke-test convenience. |
| 2 | RISK | `_save_figures` W1 proxy | `pred_w1 = pred_w2 * 0.95 + noise` is not a real Phase W1 prediction. Publishing this figure without the disclaimer would be misleading. | The current code adds a synthetic noise term and the figure axis labels say "W1 proxy". Ensure the caption clearly states "W1 proxy = synthetic stand-in; replace with real Phase W1 results for publication". |
| 3 | NOTE | `compute_rmse` | RMSE is on raw arc-length predictions, not normalised arrival times. Cross-scene comparisons of RMSE will be confounded by scene size / fire duration. | Normalise predictions per-scene before computing RMSE, or report relative RMSE. |

---

## Positive Patterns

- Slope stratification bins (Flat/Moderate/Rugged) are consistent with wildfire literature conventions. **STRONG**
- Fuel-embedding lookup inside the loss closure correctly propagates gradients through the categorical fuel representation. **STRONG**
- Early stopping with patience is included in the real-data training loop, preventing overfitting. **STRONG**
