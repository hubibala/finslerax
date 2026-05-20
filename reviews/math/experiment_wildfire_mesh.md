# Math Review: experiment_wildfire_mesh.py
**Reviewer:** Math Reviewer Agent
**Date:** 2026-05-18
**Spec Version:** MATH_SPEC.md (Randers §§ 1–2, 5), ARCH_SPEC.md § 3

## Summary

The mathematical formulation is **correct and consistent** with Phase W1 and the Randers-Zermelo framework. The covariate vector format `[elev, slope, sin(aspect), cos(aspect), canopy]` matches the Phase W1 feature encoding in `CovariateConditionedRanders._get_covariates`. One WARNING about elevation units in the docstring.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | WARNING | experiment_wildfire_mesh.py:`build_scene_mesh` return type | `face_elev_m` is labeled "mean face elevation [m]" in the docstring, but in the real-data case `scenario.elev_raster` is z-scored (unitless). The returned value is in normalized units, not metres. | Rename to `face_elev_norm_or_raw` or update docstring to clarify units depend on the caller's normalisation of `scenario.elev_raster`. |
| 2 | WARNING | experiment_wildfire_mesh.py:`_save_figures` (W1 proxy) | The "W1 proxy" is constructed as `pred_w1 = pred_w2 * 0.95 + N(0, σ_noise)` — this is not a real Phase W1 prediction. The figure title and legend do not make this clear. | Add a figure subtitle or legend note: "W1 proxy = synthetic baseline (not a trained flat-grid model)." |
| 3 | NOTE | experiment_wildfire_mesh.py:`compute_rmse` | RMSE is computed on raw (unnormalised) arc-length predictions, not on normalised arrival times. This makes RMSE values scene-dependent and not directly comparable across scenes with different time scales. | Document in the function docstring that callers should normalise predictions before comparing. |

---

## Positive Patterns

- Face covariate vector `[elev_norm, slope_rad, sin(aspect), cos(aspect), canopy]` is exactly the 5-component local feature expected by both `CovariateMeshRanders.local_mlp` (in_size=5+fuel_emb_dim) and `CovariateConditionedRanders._get_covariates`. This ensures weight transfer between Phase W1 and W2 models would be semantically meaningful. **STRONG**
- Fuel embedding lookup `metric.fuel_embedding[fuel_codes_clipped]` is placed inside the differentiable loss closure, ensuring gradients flow through the embedding table. **STRONG**
- Slope stratification bins (Flat ≤3°, Moderate 3–8°, Rugged >8°) in degrees from `jnp.degrees(jnp.std(face_slopes_rad))` are physically meaningful. **STRONG**
