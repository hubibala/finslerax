"""Per-fire prediction + paper metrics, and the few-shot recalibrated variant."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .ingest import (
    FireRecord,
    SceneBundle,
    SceneNormalizer,
    bind_fire_model,
    bind_scene_model,
)
from .medium import fire_metrics, solve_arrival_encoder
from .recover import ScaledWindMetric, recalibrate


def predict_fire(model, bundle: SceneBundle, fire: FireRecord,
                 normalizer: SceneNormalizer, wind_scale: float | None = None):
    """Zero-shot arrival field (hours) for one fire; optional wind multiplier."""
    fm = bind_fire_model(bind_scene_model(model, bundle, normalizer), fire, normalizer)
    fm = fm.precompute_metric_field()
    metric = ScaledWindMetric(fm, wind_scale) if wind_scale is not None else fm
    # Solver frame is pixels (spacing 1.0; see ingest.bind_scene_model).
    return np.asarray(
        solve_arrival_encoder(metric, jnp.asarray(fire.ignition_rc), bundle.shape, 1.0)
    )


def eval_fire(model, bundle, fire, normalizer) -> dict:
    T = predict_fire(model, bundle, fire, normalizer)
    return fire_metrics(T, fire.T_obs, fire.burned)


def eval_scene(model, bundle, fires, normalizer) -> dict:
    """Median paper metrics over a list of fires."""
    rows = [eval_fire(model, bundle, f, normalizer) for f in fires]
    return {
        k: float(np.nanmedian([r[k] for r in rows]))
        for k in ("corr", "iou50", "rel_rmse")
    } | {"n": len(rows)}


def eval_fire_recalibrated(model, bundle, fire, normalizer,
                           k_hours: float) -> tuple[dict, float, float]:
    """Metrics after the (s, c) gauge refit from the fire's first ``k_hours``.

    Returns ``(metrics, s, c)``. The observation mask is the early-isochrone
    set — exactly what a live forecast has after ``k_hours`` of satellite
    passes.
    """
    obs_mask = fire.burned & (fire.T_obs <= k_hours)
    if obs_mask.sum() < 5:
        return {"corr": np.nan, "iou50": np.nan, "rel_rmse": np.nan}, np.nan, np.nan

    def solve_c(c):
        return predict_fire(model, bundle, fire, normalizer, wind_scale=c)

    r = recalibrate(solve_c, fire.T_obs, obs_mask)
    return fire_metrics(r.T_pred, fire.T_obs, fire.burned), r.s, r.c
