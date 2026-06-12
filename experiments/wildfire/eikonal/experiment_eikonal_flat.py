#!/usr/bin/env python3
"""
Phase W1 Training Script: CovariateConditionedRanders on Sim2Real-Fire (Flat Grid)
===================================================================================

Trains a :class:`~ham.models.wildfire.CovariateConditionedRanders` Finsler metric
on the Sim2Real-Fire dataset (Gahtan et al., 2026).  The AVBD solver computes
geodesics from the ignition point to observed fire-front pixels; the metric is
updated so geodesic arc lengths match normalised pixel arrival times.

**Scientific question (Phase W1):**
    Can the Lagrangian covariate-conditioned Randers approach achieve per-scene
    Pearson correlation ≥ 0.70, and does the Finsler drift (wind term) provide a
    statistically significant improvement over the isotropic Riemannian baseline?

**Key outputs (saved to ``output_dir/figs/``):**
    - ``phaseW1_correlation_comparison.{pdf,png}``  — per-scene correlation bar chart
    - ``phaseW1_loss_convergence.{pdf,png}``        — train loss + val r vs epoch
    - ``phaseW1_runtime_tradeoff.{pdf,png}``        — runtime vs correlation scatter

**Reference:**
    Gahtan, Shpund & Bronstein (2026).  *Wildfire Simulation with Differentiable
    Randers-Finsler Eikonal Solvers.*  arXiv:2603.00035, Sections 4–6.

**HAMTools spec references:**
    spec/MATH_SPEC.md §§ 1–2, 5 (Randers / Zermelo metric)
    spec/ARCH_SPEC.md § 3 (CovariateConditionedRanders)
    spec/ARCH_SPEC.md § 4 (AVBDSolver)

Usage::

    # Synthetic smoke-test (no dataset required):
    python examples/experiment_wildfire_flat.py --synthetic --quick

    # Real data:
    python examples/experiment_wildfire_flat.py \\
        --data_root /data/sim2real_fire \\
        --scenes 0014_00426 0023_00512 \\
        --output_dir results/phaseW1

    # Without wind drift (Riemannian ablation):
    python examples/experiment_wildfire_flat.py --synthetic --quick --no_wind
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ham.data.wildfire import (
    SceneNormalizer,
    WildfireScenario,
    load_wildfire_scenario,
    stratified_sample_observations,
    train_val_test_split,
)

# config.update("jax_enable_x64", True)
# ---------------------------------------------------------------------------
# HAMTools imports
# ---------------------------------------------------------------------------
from ham.geometry.manifolds import EuclideanSpace
from ham.models.wildfire import CovariateConditionedRanders
from ham.solvers.eikonal import EikonalSolver
from ham.training.losses import curriculum_alpha

# ---------------------------------------------------------------------------
# Sim2Real-Fire loader (bundled in ham.data.sim2real_loader)
# ---------------------------------------------------------------------------
try:
    from ham.data.sim2real_loader import Sim2RealFireLoader

    HAS_GAHTAN_LOADER = True
except ImportError:
    HAS_GAHTAN_LOADER = False


# ===========================================================================
# Configuration
# ===========================================================================


def get_config(quick: bool = False) -> dict:
    """Return experiment hyperparameters.

    Args:
        quick: If True, return a reduced config sufficient for smoke-testing
            the full pipeline in minutes rather than hours.

    Returns:
        dict with all hyperparameters.
    """
    return dict(
        quick=quick,
        hidden_dim=128,
        fuel_emb_dim=4,
        cnn_channels=64,
        n_epochs=50 if quick else 100,
        lr=1e-3,
        lr_schedule="adam",  # fixed lr Adam (matches Gahtan)
        grad_clip=1.0,  # matches Gahtan grad_clip=1.0
        early_stopping_patience=20,  # patience in epochs (applied every val_interval epochs)
        batch_size_fires=32,  # matches Gahtan batch_size=32
        sequential_fires=False,
        eikonal_iters=50,
        lambda_tv=0.005,  # single TV weight, matches Gahtan lambda_G=lambda_B=0.005
        k_train_obs=500,  # obs pixels sampled per fire by the loader (for val_pixels field)
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
        train_seeds=[0] if quick else [0, 1, 2],
    )


# ===========================================================================
# Model helpers
# ===========================================================================


def make_metric(
    cfg: dict,
    manifold: EuclideanSpace,
    key: jax.Array,
    use_wind: bool = True,
) -> CovariateConditionedRanders:
    """Instantiate an unbound CovariateConditionedRanders metric.

    Args:
        cfg:       Configuration dict from :func:`get_config`.
        manifold:  2-D Euclidean manifold.
        key:       JAX PRNG key for MLP initialisation.
        use_wind:  If False, sets drift b=0 (Riemannian ablation).

    Returns:
        Unbound :class:`~ham.models.wildfire.CovariateConditionedRanders`.
    """
    return CovariateConditionedRanders(
        manifold,
        key,
        hidden_dim=cfg["hidden_dim"],
        fuel_emb_dim=cfg["fuel_emb_dim"],
        cnn_channels=cfg["cnn_channels"],
        use_wind=use_wind,
    )


def bind_scenario_to_metric(
    metric: CovariateConditionedRanders,
    scenario: WildfireScenario,
) -> CovariateConditionedRanders:
    """Return a copy of *metric* with scene covariates baked in.

    All rasters are converted to float64 JAX arrays.  The returned model
    shares MLP weights with *metric* but has frozen scene rasters.

    Args:
        metric:   Unbound (or previously bound) metric.
        scenario: Fire scenario providing rasters and weather.

    Returns:
        Bound :class:`~ham.models.wildfire.CovariateConditionedRanders`.
    """
    return metric.bind_scene(
        elev=jnp.asarray(scenario.elev_raster, dtype=jnp.float32),
        slope=jnp.asarray(scenario.slope_raster, dtype=jnp.float32),
        aspect=jnp.asarray(scenario.aspect_raster, dtype=jnp.float32),
        canopy=jnp.asarray(scenario.canopy_raster, dtype=jnp.float32),
        fuel_codes=jnp.asarray(scenario.fuel_code_raster, dtype=jnp.int32),
        weather_vec=jnp.asarray(scenario.weather_vec, dtype=jnp.float32),
        pixel_spacing_m=float(scenario.pixel_spacing_m),
        origin_xy=jnp.asarray(scenario.origin_xy, dtype=jnp.float32),
    )


def bind_scenario_terrain(
    metric: CovariateConditionedRanders,
    scenario: WildfireScenario,
) -> CovariateConditionedRanders:
    """Return a copy of *metric* with terrain rasters baked in but no weather.

    Used as the outer step before a vmap over fire batches:  terrain is shared
    across all fires in a scene, so binding it once avoids re-tracing per fire.

    Args:
        metric:   Unbound metric.
        scenario: Any fire scenario from the scene (rasters are scene-level).

    Returns:
        Terrain-bound metric (``weather_vec`` still None).
    """
    return metric.bind_scene_rasters(
        elev=jnp.asarray(scenario.elev_raster, dtype=jnp.float32),
        slope=jnp.asarray(scenario.slope_raster, dtype=jnp.float32),
        aspect=jnp.asarray(scenario.aspect_raster, dtype=jnp.float32),
        canopy=jnp.asarray(scenario.canopy_raster, dtype=jnp.float32),
        fuel_codes=jnp.asarray(scenario.fuel_code_raster, dtype=jnp.int32),
        pixel_spacing_m=float(scenario.pixel_spacing_m),
        origin_xy=jnp.asarray(scenario.origin_xy, dtype=jnp.float32),
    )


def _bilinear_interp_point(grid: jax.Array, pt_pixel: jax.Array) -> jax.Array:
    H, W = grid.shape
    py, px = pt_pixel[0], pt_pixel[1]
    px = jnp.clip(px, 0.0, W - 1.001)
    py = jnp.clip(py, 0.0, H - 1.001)
    x0 = jnp.floor(px).astype(jnp.int32)
    y0 = jnp.floor(py).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, W - 1)
    y1 = jnp.minimum(y0 + 1, H - 1)
    fx = px - x0
    fy = py - y0
    return (
        grid[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + grid[y0, x1] * fx * (1.0 - fy)
        + grid[y1, x0] * (1.0 - fx) * fy
        + grid[y1, x1] * fx * fy
    )


def compute_dense_gahtan_loss(
    bound_metric,
    solver: EikonalSolver,
    source_world,
    arrival_times_gt,
    grid_extent: tuple,
    grid_shape: tuple,
):
    """Computes dense MSE loss and TV regularization on the Randers-Finsler metric.

    This matches the exact loss formulation of Gahtan's baseline:
    - Data Loss: MSE = 0.5 * mean( (T_pred - T_gt) ** 2 ) over valid pixels.
    - Regularisation: TV on G and B parameters.
    """
    t_pred_dense, _, _ = solver.solve(
        bound_metric, source_world, grid_extent, grid_shape
    )

    # Valid mask (where GT is finite and pred converged)
    valid_mask = jnp.isfinite(arrival_times_gt) & (t_pred_dense < 1e5)
    n_obs = jnp.sum(valid_mask)

    # Compute MSE data loss
    diff = t_pred_dense - arrival_times_gt
    sq_err = jnp.where(valid_mask, diff**2, 0.0)
    data_loss = 0.5 * jnp.sum(sq_err) / jnp.maximum(n_obs, 1.0)

    # TV regularization over raw metric parameters (to encourage smooth fields)
    metric_field = bound_metric.metric_field
    tv_loss = jnp.mean(jnp.abs(metric_field[1:, :] - metric_field[:-1, :])) + jnp.mean(
        jnp.abs(metric_field[:, 1:] - metric_field[:, :-1])
    )

    return data_loss, tv_loss


def make_batched_train_step(
    terrain_metric: CovariateConditionedRanders,
    solver: EikonalSolver,
    optimizer,
    grid_extent: tuple,
    grid_shape: tuple,
    pixel_spacing_m: float,
    sequential: bool = False,
):
    """Return a JIT-compiled training step over a batch of fires.

    All B fires are processed inside a single ``filter_jit`` call.
    """

    @eqx.filter_jit
    def batched_step(
        metric, opt_state, weather_batch, source_batch, arrival_times_gt_batch
    ):
        """Single JIT dispatch: vmap/lax.map over B fires, then one optimizer update."""

        def total_loss(m):
            def fire_loss(args):
                w, s, gt = args
                bound = m.bind_weather(w).precompute_metric_field()
                data_l, tv_l = compute_dense_gahtan_loss(
                    bound, solver, s, gt, grid_extent, grid_shape
                )
                return data_l + 0.005 * tv_l

            stacked = (weather_batch, source_batch, arrival_times_gt_batch)
            if sequential:
                per_fire_losses = jax.lax.map(fire_loss, stacked)
            else:
                per_fire_losses = jax.vmap(jax.checkpoint(fire_loss))(stacked)
            return jnp.mean(per_fire_losses)

        loss_val, grads = eqx.filter_value_and_grad(total_loss)(metric)
        updates, new_opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_inexact_array),
            opt_state,
            eqx.filter(metric, eqx.is_inexact_array),
        )
        return eqx.apply_updates(metric, updates), new_opt_state, loss_val

    return batched_step


def make_solver(cfg: dict) -> EikonalSolver:
    """Instantiate an Eikonal solver.

    Args:
        cfg: Configuration dict from :func:`get_config`.

    Returns:
        :class:`~ham.solvers.eikonal.EikonalSolver`.
    """
    return EikonalSolver(
        max_iters=cfg["eikonal_iters"],
        tol=1e-6,
    )


# ===========================================================================
# Coordinate helpers
# ===========================================================================


def _pixels_to_world(pixels: np.ndarray, pixel_spacing_m: float) -> np.ndarray:
    """Convert (row, col) pixel indices to (x, y) world coordinates.

    The convention follows the data loader:
        x_world = col * pixel_spacing_m   (horizontal axis)
        y_world = row * pixel_spacing_m   (vertical axis)

    Args:
        pixels:          Shape (K, 2) int — rows in column 0, cols in column 1.
        pixel_spacing_m: Metres per pixel.

    Returns:
        Shape (K, 2) float64 — ``[x_m, y_m]`` pairs.
    """
    pix = np.asarray(pixels, dtype=np.float32)
    x_world = pix[:, 1] * float(pixel_spacing_m)
    y_world = pix[:, 0] * float(pixel_spacing_m)
    return np.stack([x_world, y_world], axis=1)


def _ignition_to_world(ignition_pixel: np.ndarray, pixel_spacing_m: float) -> jax.Array:
    """Convert ignition (row, col) pixel to world (x, y) JAX array.

    Args:
        ignition_pixel:  Shape (2,) — ``[row, col]``.
        pixel_spacing_m: Metres per pixel.

    Returns:
        Shape (2,) float64 JAX array ``[x_m, y_m]``.
    """
    row, col = float(ignition_pixel[0]), float(ignition_pixel[1])
    return jnp.array(
        [col * float(pixel_spacing_m), row * float(pixel_spacing_m)], dtype=jnp.float32
    )


# ===========================================================================
# Pearson r
# ===========================================================================


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1-D arrays.

    Args:
        a: Predicted values, shape (N,).
        b: Ground-truth values, shape (N,).

    Returns:
        Correlation in ``[-1, 1]``; ``0.0`` if either array is constant or
        too short.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2:
        return 0.0
    a_mean = a.mean()
    b_mean = b.mean()
    num = np.sum((a - a_mean) * (b - b_mean))
    denom = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2))
    if denom < 1e-8:
        return 0.0
    return float(num / denom)


def spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays.

    Rank-based and therefore invariant to any monotone rescaling of either
    input.  Directly comparable to Gahtan et al. (2026) Appendix E, which
    reports Spearman ≈ 0.695 for cross-scene transfer and ≈ 0.696 for
    simulation-to-real transfer.

    Args:
        a: Predicted values, shape (N,).
        b: Ground-truth values, shape (N,).

    Returns:
        Correlation in ``[-1, 1]``; ``0.0`` if either array is constant or
        too short.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2:
        return 0.0
    rank_a = np.argsort(np.argsort(a)).astype(np.float32)
    rank_b = np.argsort(np.argsort(b)).astype(np.float32)
    return pearson_r(rank_a, rank_b)


# ===========================================================================
# Chunked arrival-time prediction
# ===========================================================================


@eqx.filter_jit
def _predict_arrivals_dense(
    bound_metric: CovariateConditionedRanders,
    solver: EikonalSolver,
    source_world: jax.Array,
    grid_extent: tuple,
    grid_shape: tuple,
) -> jax.Array:
    """JIT-compiled dense arc-length prediction over the grid."""
    t_pred, _, _ = solver.solve(bound_metric, source_world, grid_extent, grid_shape)
    return t_pred


def _predict_arrivals_eval(
    bound_metric: CovariateConditionedRanders,
    solver: EikonalSolver,
    source_world: jax.Array,
    eval_pixels: np.ndarray,
    pixel_spacing_m: float,
    grid_extent: tuple,
    grid_shape: tuple,
) -> np.ndarray:
    """Predict geodesic arc lengths for *eval_pixels* by querying the dense grid.

    Args:
        bound_metric:    Fully bound metric (rasters + weather baked in).
        solver:          Eikonal solver.
        source_world:    Ignition world coordinate, shape (2,).
        eval_pixels:     Shape (N, 2) int — ``[row, col]`` pairs.
        pixel_spacing_m: Metres per pixel.

    Returns:
        Shape (N,) float64 — predicted arrival arc lengths.
    """
    t_pred_dense = _predict_arrivals_dense(
        bound_metric, solver, source_world, grid_extent, grid_shape
    )

    # We can just extract exactly at the integer pixels:
    rows = eval_pixels[:, 0]
    cols = eval_pixels[:, 1]
    pred = np.array(t_pred_dense)[rows, cols]

    return np.array(pred, dtype=np.float32)


# ===========================================================================
# Per-fire training step
# ===========================================================================


def train_one_fire(
    metric: CovariateConditionedRanders,
    solver: EikonalSolver,
    scenario: WildfireScenario,
    cfg: dict,
    opt_state,
    optimizer,
    key: jax.Array,
):
    obs_world = jnp.asarray(
        _pixels_to_world(scenario.obs_pixels, scenario.pixel_spacing_m),
        dtype=jnp.float32,
    )  # (K, 2)
    t_obs = jnp.asarray(scenario.obs_arrival_times, dtype=jnp.float32)  # (K,)
    source = _ignition_to_world(scenario.ignition_pixel, scenario.pixel_spacing_m)
    alpha = jnp.asarray(0.0, dtype=jnp.float32)
    grid_shape = scenario.arrival_times.shape
    grid_extent = (
        0.0,
        grid_shape[1] * scenario.pixel_spacing_m,
        0.0,
        grid_shape[0] * scenario.pixel_spacing_m,
    )

    @eqx.filter_jit
    def _loss(m: CovariateConditionedRanders) -> jax.Array:
        bound = bind_scenario_to_metric(m, scenario)
        bound = bound.precompute_metric_field()
        return compute_sparse_eikonal_loss(  # type: ignore[name-defined]
            bound,
            solver,
            source,
            obs_world,
            t_obs,
            alpha,
            grid_extent,
            grid_shape,
            scenario.pixel_spacing_m,
        )

    loss_val, grads = eqx.filter_value_and_grad(_loss)(metric)
    updates, new_opt_state = optimizer.update(grads, opt_state, metric)
    new_metric = eqx.apply_updates(metric, updates)
    return new_metric, new_opt_state, float(loss_val)


# ===========================================================================
# Per-fire evaluation
# ===========================================================================


def evaluate_fire(
    metric: CovariateConditionedRanders,
    solver: EikonalSolver,
    scenario: WildfireScenario,
    cfg: dict,
    eval_pixels: np.ndarray | None = None,
) -> dict:
    """Evaluate one fire: Pearson r, Spearman r, and calibrated IoU@50.

    Metrics are designed to be comparable with Gahtan et al. (2026):

    * **Pearson r** — linear correlation of arrival times (Gahtan §6 primary
      metric; within-scene target ≈ 0.824).
    * **Spearman r** — rank correlation; invariant to any monotone rescaling
      of predictions.  Directly comparable to Gahtan Appendix E (cross-scene
      ≈ 0.695, sim-to-real ≈ 0.696).
    * **IoU@50 (calibrated)** — Gahtan-compatible spatial overlap metric.
      A post-hoc scalar ``s = mean(gt) / mean(pred)`` maps arc-lengths to the
      GT time scale; the calibrated predictions are then thresholded at 0.5
      over the full burned area.  For sparse eval (quick mode), unsampled
      burned pixels default to 1.0 ("late"), so IoU degrades with coverage
      but is never structurally zero.  Coverage is reported as
      ``eval_coverage`` for interpretability.

    Args:
        metric:      Trained metric (unbound; scene is bound internally).
        solver:      AVBD solver.
        scenario:    Fire scenario providing GT arrival times.
        cfg:         Configuration dict (provides ``avbd_n_steps``).
        eval_pixels: Shape (N, 2) int ``[row, col]`` pixels to evaluate.
            If None, all burned pixels in the scenario are used.

    Returns:
        dict with keys ``pearson_r``, ``spearman_r``, ``iou_50``,
        ``eval_coverage``, ``n_eval_pixels``.
    """
    if eval_pixels is None:
        rows, cols = np.where(scenario.burned_mask)
        eval_pixels = np.stack([rows, cols], axis=1)
        if cfg.get("quick", True) and len(eval_pixels) > 100:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(eval_pixels), size=100, replace=False)
            eval_pixels = eval_pixels[idx]

    n_burned = int(np.sum(scenario.burned_mask))
    if len(eval_pixels) == 0:
        return dict(
            pearson_r=0.0,
            spearman_r=0.0,
            iou_50=0.0,
            eval_coverage=0.0,
            n_eval_pixels=0,
        )

    bound_metric = bind_scenario_to_metric(metric, scenario)
    bound_metric = bound_metric.precompute_metric_field()
    source = _ignition_to_world(scenario.ignition_pixel, scenario.pixel_spacing_m)

    grid_shape = scenario.arrival_times.shape
    grid_extent = (
        0.0,
        grid_shape[1] * scenario.pixel_spacing_m,
        0.0,
        grid_shape[0] * scenario.pixel_spacing_m,
    )

    pred_arrivals = _predict_arrivals_eval(
        bound_metric,
        solver,
        source,
        eval_pixels,
        scenario.pixel_spacing_m,
        grid_extent,
        grid_shape,
    )

    gt_arrival = np.array(
        [scenario.arrival_times[int(r), int(c)] for r, c in eval_pixels],
        dtype=np.float32,
    )

    r_pearson = pearson_r(pred_arrivals, gt_arrival)
    r_spearman = spearman_r(pred_arrivals, gt_arrival)
    coverage = len(eval_pixels) / max(n_burned, 1)

    # --- Calibrated IoU@50 (Gahtan-compatible) ----------------------------
    # Map arc-lengths to the GT time scale with a single post-hoc scalar
    # s = mean(gt_finite) / mean(pred_finite).  This is NOT on the gradient
    # path — it is only used for evaluation.  The calibrated predictions are
    # then placed into the full burned-area raster; unsampled pixels default
    # to 1.0 ("arrives in the second half"), which is conservative but not
    # structurally zero.  The threshold of 0.5 then matches Gahtan's
    # definition: "pixels that burn in the first half of the fire's duration."
    valid = np.isfinite(pred_arrivals) & np.isfinite(gt_arrival)
    if np.sum(valid) >= 2:
        p_valid = pred_arrivals[valid]
        g_valid = gt_arrival[valid]
        mean_pred = float(np.mean(p_valid))
        mean_gt = float(np.mean(g_valid))
        s = mean_gt / max(mean_pred, 1e-8)  # post-hoc calibration scalar

        # Build full-raster prediction (H, W); default = 1.0 ("late")
        pred_raster = np.ones_like(scenario.arrival_times)
        cal_values = np.clip(s * pred_arrivals, 0.0, 1.0)
        for (row, col), t in zip(eval_pixels, cal_values):
            pred_raster[int(row), int(col)] = float(t)

        burned = scenario.burned_mask
        pred_bin = (pred_raster <= 0.5) & burned
        gt_bin = (scenario.arrival_times <= 0.5) & burned
        intersect = int(np.sum(pred_bin & gt_bin))
        union = int(np.sum(pred_bin | gt_bin))
        iou50 = float(intersect / max(union, 1))
    else:
        iou50 = 0.0

    return dict(
        pearson_r=r_pearson,
        spearman_r=r_spearman,
        iou_50=iou50,
        eval_coverage=coverage,
        n_eval_pixels=len(eval_pixels),
    )


# ===========================================================================
# Validation pass (dense, matching Gahtan metrics)
# ===========================================================================


def _val_metrics(
    metric: CovariateConditionedRanders,
    solver: EikonalSolver,
    scenario: WildfireScenario,
) -> dict:
    """Dense validation: Pearson r AND relative RMSE over all burned pixels.

    Returns a dict with keys ``pearson_r`` and ``relative_rmse``,
    matching the per-epoch metrics reported by Gahtan's ``per_scene.py``.
    """
    # Build full dense GT arrival time raster (H, W), inf where unburned.
    gt_full = np.asarray(scenario.arrival_times, dtype=np.float32)  # (H, W)

    bound_metric = bind_scenario_to_metric(metric, scenario)
    bound_metric = bound_metric.precompute_metric_field()
    source = _ignition_to_world(scenario.ignition_pixel, scenario.pixel_spacing_m)

    grid_shape = gt_full.shape
    grid_extent = (
        0.0,
        grid_shape[1] * scenario.pixel_spacing_m,
        0.0,
        grid_shape[0] * scenario.pixel_spacing_m,
    )

    pred_dense = np.array(
        _predict_arrivals_dense(bound_metric, solver, source, grid_extent, grid_shape),
        dtype=np.float32,
    )  # (H, W)

    # Valid mask: GT finite and solver converged (< 1e5)
    valid = np.isfinite(gt_full) & (pred_dense < 1e5)
    if valid.sum() < 10:
        return dict(pearson_r=0.0, relative_rmse=float("inf"))

    pred_v = pred_dense[valid]
    gt_v = gt_full[valid]

    r = pearson_r(pred_v, gt_v)

    # Relative RMSE = RMSE / max_gt  (Gahtan definition)
    rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))
    max_gt = float(gt_v.max())
    rel_rmse = rmse / max_gt if max_gt > 1e-6 else float("inf")

    return dict(pearson_r=r, relative_rmse=rel_rmse)


def _val_pearson_r(
    metric: CovariateConditionedRanders,
    solver: EikonalSolver,
    scenario: WildfireScenario,
    cfg: dict = None,
) -> float:
    """Backwards-compat wrapper — returns Pearson r only."""
    return _val_metrics(metric, solver, scenario)["pearson_r"]


# ===========================================================================
# Per-scene training loop
# ===========================================================================


def train_scene(
    data_root: str,
    scene_id: str,
    cfg: dict,
    seed: int,
    use_wind: bool = True,
    use_sequential_fires: bool = True,
    output_dir: str = "results",
) -> dict | None:
    """Full per-scene training loop for Phase W1.

    1. Loads all fire events for the scene via :class:`Sim2RealFireLoader`.
    2. Splits into train / val / test (70/15/15).
    3. Fits :class:`~ham.data.wildfire.SceneNormalizer` on training fires.
    4. Trains :class:`~ham.models.wildfire.CovariateConditionedRanders` with
       AVBD arrival-time supervision.  Gradient accumulation is performed over
       ``batch_size_fires`` fires before each optimizer step.
    5. Evaluates best-val checkpoint on the test split.

    Args:
        data_root: Path to the Sim2Real-Fire dataset root directory.
        scene_id:  Scene identifier, e.g. ``"0014_00426"``.
        cfg:       Configuration dict from :func:`get_config`.
        seed:      RNG seed for sampling and weight initialisation.
        use_wind:  If False, trains the Riemannian ablation (b=0).

    Returns:
        Result dict with keys ``scene_id``, ``test_pearson_r_mean``,
        ``test_pearson_r_std``, ``test_iou50``, ``train_loss_history``,
        ``val_r_history``, ``runtime_per_epoch_s``.
        Returns ``None`` if the scene has no events.

    Raises:
        ImportError: If the Gahtan loader is not available.
        FileNotFoundError: If *data_root* does not exist.
    """
    if not HAS_GAHTAN_LOADER:
        raise ImportError(
            "Sim2RealFireLoader not available.  Install the Gahtan loader or "
            "run with --synthetic to use a synthetic smoke test."
        )
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"Dataset not found at {data_root}.  Download with:\n"
            f"  python experiments/wildfire/download_data.py --output_dir {data_root}"
        )

    print(f"\n{'=' * 60}")
    print(
        f"Scene {scene_id}  seed={seed}  wind={'yes' if use_wind else 'no (Riemannian)'}"
    )
    print(f"{'=' * 60}")

    t_scene_start = time.time()

    loader = Sim2RealFireLoader(data_root)

    # List events from the loader's discovered scene registry
    try:
        event_ids = loader.scenes[scene_id]["events"]
    except (AttributeError, KeyError):
        try:
            event_ids = loader.list_events(scene_id)
        except AttributeError:
            try:
                event_ids = loader.get_event_ids(scene_id)
            except AttributeError:
                # Final fallback: read Satellite_Images_Mask subfolders directly
                mask_dir = os.path.join(data_root, scene_id, "Satellite_Images_Mask")
                weather_dir = os.path.join(data_root, scene_id, "Weather_Data")
                if os.path.isdir(mask_dir):
                    event_ids = [
                        d
                        for d in sorted(os.listdir(mask_dir))
                        if os.path.isdir(os.path.join(mask_dir, d))
                        and os.path.exists(os.path.join(weather_dir, f"{d}.txt"))
                    ]
                else:
                    event_ids = []

    if not event_ids:
        print(f"  WARNING: No events found for scene {scene_id}, skipping.")
        return None

    scenarios_keys = [(scene_id, eid) for eid in event_ids]
    train_list, val_list, test_list = train_val_test_split(
        scenarios_keys,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
    )
    if cfg.get("quick", False):
        train_list = train_list[:96]
        val_list = val_list[:32]
        test_list = test_list[:32]
    print(
        f"  Events: {len(scenarios_keys)} total | "
        f"{len(train_list)} train / {len(val_list)} val / {len(test_list)} test"
    )

    # Fit normaliser on (up to 20) training fires to keep loading fast
    print("  Fitting SceneNormalizer on training fires...")
    n_workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        raw_train_for_norm = list(
            pool.map(
                lambda p: loader.load_scenario(p[0], p[1]),
                train_list[:20],
            )
        )
    normalizer = SceneNormalizer.fit(raw_train_for_norm)

    # Load all processed scenarios in parallel (I/O-bound: PIL + rasterio release GIL)
    print(f"  Loading & preprocessing scenarios (n_workers={n_workers})...")

    def _load_one(pair):
        sid, eid = pair
        return load_wildfire_scenario(
            loader,
            sid,
            eid,
            normalizer,
            k_train_obs=cfg["k_train_obs"],
            seed=seed,
        )

    def _load(pairs):
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            # preserve ordering
            return list(pool.map(_load_one, pairs))

    train_scenarios = _load(train_list)
    val_scenarios = _load(val_list)
    test_scenarios = _load(test_list)
    print(
        f"  Loaded {len(train_scenarios)} / {len(val_scenarios)} / "
        f"{len(test_scenarios)} train/val/test scenarios"
    )

    # Metric + solver + optimizer
    key = jax.random.PRNGKey(seed)
    manifold = EuclideanSpace(2)
    metric = make_metric(cfg, manifold, key, use_wind=use_wind)
    solver = make_solver(cfg)

    n_batches_per_epoch = max(1, len(train_scenarios) // cfg["batch_size_fires"])
    # Adam with fixed lr matching Gahtan; grad clip matches Gahtan's grad_clip=1.0
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg["grad_clip"]),
        optax.adam(cfg["lr"]),
    )
    # opt_state is (re-)initialised after binding terrain below

    # Training state — early stopping on val correlation (matches Gahtan)
    best_val_corr: float = -1.0
    best_metric = metric
    patience_counter: int = 0
    train_loss_history: list = []
    train_data_loss_history: list = []
    train_reg_loss_history: list = []
    val_r_history: list = []
    val_rel_rmse_history: list = []
    epoch_runtimes: list = []

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Bind terrain once per scene — shared across all fires in the scene.
    # Only weather_vec differs per fire; that's handled inside vmap.
    # terrain_metric is used as the *base* structure for the batched step:
    # it carries the terrain rasters + initial MLP weights.  The optimizer
    # updates only the MLP leaves (filter=eqx.is_array); rasters are
    # non-differentiable constants carried along.
    # ------------------------------------------------------------------
    terrain_metric = bind_scenario_terrain(metric, train_scenarios[0])
    # Re-initialise optimizer state on the terrain-bound metric so its
    # pytree structure matches what the batched step will return.
    opt_state = optimizer.init(eqx.filter(terrain_metric, eqx.is_inexact_array))
    metric = terrain_metric  # from here on, metric always carries terrain
    grid_shape = train_scenarios[0].arrival_times.shape
    grid_extent = (
        0.0,
        grid_shape[1] * train_scenarios[0].pixel_spacing_m,
        0.0,
        grid_shape[0] * train_scenarios[0].pixel_spacing_m,
    )

    batched_step = make_batched_train_step(
        terrain_metric,
        solver,
        optimizer,
        grid_extent=grid_extent,
        grid_shape=grid_shape,
        pixel_spacing_m=train_scenarios[0].pixel_spacing_m,
        sequential=use_sequential_fires,
    )

    # Pre-stack all training arrays for fast batch slicing.
    # Dense arrival time rasters are padded to (H, W) and stacked.
    H_grid, W_grid = grid_shape

    def _pad_arrival(sc):
        at = np.asarray(sc.arrival_times, dtype=np.float32)
        # Replace inf with a large sentinel so JAX can handle the stacked array.
        return np.where(np.isfinite(at), at, np.float32(1e5))

    _arrival_all = jnp.asarray(
        np.stack([_pad_arrival(sc) for sc in train_scenarios], axis=0),
        dtype=jnp.float32,
    )  # (N_train, H, W)
    _sources_all = jnp.asarray(
        np.stack(
            [
                _ignition_to_world(sc.ignition_pixel, sc.pixel_spacing_m)
                for sc in train_scenarios
            ],
            axis=0,
        ),
        dtype=jnp.float32,
    )  # (N_train, 2)
    _weather_all = jnp.asarray(
        np.stack([sc.weather_vec for sc in train_scenarios], axis=0), dtype=jnp.float32
    )  # (N_train, weather_dim)

    B = cfg["batch_size_fires"]  # vmap batch size
    print(
        f"  Batched training (dense MSE): B={B} fires/step, "
        f"{n_batches_per_epoch} steps/epoch, grid={H_grid}×{W_grid}"
    )

    for epoch in range(cfg["n_epochs"]):
        t_epoch = time.time()

        # Shuffle training fires
        perm = rng.permutation(len(train_scenarios))
        epoch_losses: list = []
        epoch_data_losses: list = []
        epoch_reg_losses: list = []

        for batch_start in range(0, len(perm) - B + 1, B):
            batch_idx = perm[batch_start : batch_start + B]

            weather_b = _weather_all[batch_idx]  # (B, weather_dim)
            source_b = _sources_all[batch_idx]  # (B, 2)
            arrival_b = _arrival_all[batch_idx]  # (B, H, W)

            metric, opt_state, loss_val = batched_step(
                metric, opt_state, weather_b, source_b, arrival_b
            )
            # The batched loss is data + 0.005*tv; approximate decomposition for logging
            epoch_losses.append(float(loss_val))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_loss_history.append(mean_loss)
        epoch_runtimes.append(time.time() - t_epoch)

        # Validation every 5 epochs (quick) or every epoch (full) — matches Gahtan cadence
        val_interval = 5 if cfg.get("quick", False) else 5
        if (epoch + 1) % val_interval == 0 or epoch == 0:
            val_results = [_val_metrics(metric, solver, sc) for sc in val_scenarios]
            mean_val_r = (
                float(np.mean([r["pearson_r"] for r in val_results]))
                if val_results
                else 0.0
            )
            mean_val_rmse = (
                float(np.mean([r["relative_rmse"] for r in val_results]))
                if val_results
                else float("inf")
            )
        # else: reuse previous values so early-stopping logic is unaffected
        val_r_history.append(mean_val_r)
        val_rel_rmse_history.append(mean_val_rmse)

        # Gahtan-equivalent per-epoch print:
        #   Epoch  X: loss=Y.Y (data=Y.Y, reg=Y.Y), val_corr=Y.YYY, val_rel_rmse=Y.YYY, lr=Y.Ye-Y
        current_lr = cfg["lr"]  # fixed Adam lr
        print(
            f"  Epoch {epoch + 1:3d}/{cfg['n_epochs']}: "
            f"loss={mean_loss:.1f}  "
            f"val_corr={mean_val_r:.3f}  val_rel_rmse={mean_val_rmse:.3f}  "
            f"lr={current_lr:.1e}  time={epoch_runtimes[-1]:.1f}s"
        )

        # Early stopping on val correlation (higher = better), matching Gahtan
        if mean_val_r > best_val_corr:
            best_val_corr = mean_val_r
            best_metric = metric
            patience_counter = 0
        else:
            patience_counter += val_interval
            if patience_counter >= cfg["early_stopping_patience"]:
                print(
                    f"  Early stopping at epoch {epoch + 1}  (best_val_corr={best_val_corr:.4f})"
                )
                break

    # Test evaluation — Gahtan-equivalent metrics
    print(f"\n  Evaluating on {len(test_scenarios)} test fires...")
    test_metrics_list: list = []
    for sc in test_scenarios:
        res = evaluate_fire(best_metric, solver, sc, cfg)
        test_metrics_list.append(res)

    def _safe_mean(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else 0.0

    def _safe_std(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    test_rs = [m["pearson_r"] for m in test_metrics_list]
    test_sprs = [m["spearman_r"] for m in test_metrics_list]
    test_ious = [m["iou_50"] for m in test_metrics_list]
    test_coverages = [m["eval_coverage"] for m in test_metrics_list]

    # Relative RMSE from evaluate_fire (needs dense pred — re-run _val_metrics on test set)
    test_rel_rmses = []
    for sc in test_scenarios:
        vm = _val_metrics(best_metric, solver, sc)
        test_rel_rmses.append(vm["relative_rmse"])

    test_r_mean = _safe_mean(test_rs)
    test_r_std = _safe_std(test_rs)
    test_spr_mean = _safe_mean(test_sprs)
    test_iou50 = _safe_mean(test_ious)
    mean_coverage = _safe_mean(test_coverages)
    test_rel_rmse_mean = _safe_mean(test_rel_rmses)
    test_rel_rmse_std = _safe_std(test_rel_rmses)
    runtime_per_epoch = float(np.mean(epoch_runtimes)) if epoch_runtimes else 0.0

    # Print in Gahtan summary format
    print(f"\n  {'=' * 60}")
    print(f"  Scene {scene_id} Results:")
    print(f"    Test correlation:   {test_r_mean:.3f} ± {test_r_std:.3f}")
    print(f"    Test relative RMSE: {test_rel_rmse_mean:.3f} ± {test_rel_rmse_std:.3f}")
    print(f"    Test IoU@50:        {test_iou50:.3f}")
    print(f"    Test Spearman r:    {test_spr_mean:.3f}")
    print(f"    Best val corr:      {best_val_corr:.3f}")
    print(f"    Epoch runtime:      {runtime_per_epoch:.1f} s/epoch")
    print(f"    Total time:         {time.time() - t_scene_start:.1f} s")

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"w1_{scene_id}_seed{seed}.eqx")
    eqx.tree_serialise_leaves(ckpt_path, best_metric)
    print(f"    Saved checkpoint: {ckpt_path}")

    return dict(
        scene_id=scene_id,
        seed=seed,
        use_wind=use_wind,
        test_pearson_r_mean=test_r_mean,
        test_pearson_r_std=test_r_std,
        test_spearman_r_mean=test_spr_mean,
        test_relative_rmse_mean=test_rel_rmse_mean,
        test_relative_rmse_std=test_rel_rmse_std,
        test_iou50=test_iou50,
        eval_coverage=mean_coverage,
        best_val_corr=best_val_corr,
        train_loss_history=train_loss_history,
        val_r_history=val_r_history,
        val_rel_rmse_history=val_rel_rmse_history,
        runtime_per_epoch_s=runtime_per_epoch,
    )


# ===========================================================================
# Multi-scene experiment runner
# ===========================================================================


def run_experiment(
    data_root: str,
    scene_ids: list,
    output_dir: str,
    cfg: dict,
    use_wind: bool = True,
    use_sequential_fires: bool = True,
) -> list:
    """Run Phase W1 for all scenes × seeds and produce publication figures.

    Args:
        data_root:  Path to the Sim2Real-Fire dataset root.
        scene_ids:  List of scene identifier strings.
        output_dir: Directory for result figures.
        cfg:        Configuration dict from :func:`get_config`.
        use_wind:   If False, run the Riemannian ablation.

    Returns:
        List of result dicts, one per (scene × seed) pair.
    """
    all_results: list = []

    for scene_id in scene_ids:
        for seed in cfg["train_seeds"]:
            result = train_scene(
                data_root,
                scene_id,
                cfg,
                seed,
                use_wind=use_wind,
                use_sequential_fires=use_sequential_fires,
                output_dir=output_dir,
            )
            if result is not None:
                all_results.append(result)

    if all_results:
        _save_figures(all_results, output_dir, cfg)

    return all_results


# ===========================================================================
# Figures
# ===========================================================================


def _save_figures(results: list, output_dir: str, cfg: dict) -> None:
    """Generate and save Phase W1 publication figures.

    Args:
        results:    List of result dicts from :func:`run_experiment`.
        output_dir: Root directory; figures saved to ``<output_dir>/figs/``.
        cfg:        Configuration dict (for metadata).
    """
    fig_dir = os.path.join(output_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Per-scene correlation bar chart --------------------------------
    scene_ids = sorted({r["scene_id"] for r in results})
    r_means = []
    r_stds = []
    for sid in scene_ids:
        vals = [r["test_pearson_r_mean"] for r in results if r["scene_id"] == sid]
        r_means.append(float(np.mean(vals)))
        r_stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(scene_ids) * 0.8), 4))
    x = np.arange(len(scene_ids))
    ax.bar(x, r_means, yerr=r_stds, capsize=4, color="#4477AA", label="HAMTools W1")
    ax.axhline(
        0.70, color="red", linestyle="--", linewidth=1.2, label="Target (r=0.70)"
    )
    ax.axhline(
        0.824,
        color="orange",
        linestyle=":",
        linewidth=1.2,
        label="Gahtan et al. mean (0.824)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(scene_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r (test)")
    ax.set_title("Phase W1 — Per-scene arrival-time correlation")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(fig_dir, f"phaseW1_correlation_comparison.{ext}"), dpi=150
        )
    plt.close(fig)

    # ---- Loss convergence (first scene, first seed) ----------------------
    first = next(
        (r for r in results if r["train_loss_history"]),
        None,
    )
    if first is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        epochs = np.arange(1, len(first["train_loss_history"]) + 1)
        ax1.plot(epochs, first["train_loss_history"], color="#4477AA")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train loss (MSE)")
        ax1.set_title(f"Loss convergence — scene {first['scene_id']}")

        ax2.plot(
            epochs[: len(first["val_r_history"])],
            first["val_r_history"],
            color="#CC4444",
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Val Pearson r")
        ax2.set_title(f"Validation r — scene {first['scene_id']}")
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(
                os.path.join(fig_dir, f"phaseW1_loss_convergence.{ext}"), dpi=150
            )
        plt.close(fig)

    # ---- Runtime vs. correlation scatter --------------------------------
    rts = [r["runtime_per_epoch_s"] for r in results]
    cors = [r["test_pearson_r_mean"] for r in results]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(rts, cors, s=40, alpha=0.7, color="#4477AA")
    ax.set_xlabel("Runtime (s / epoch)")
    ax.set_ylabel("Test Pearson r")
    ax.set_title("Phase W1 — Runtime–correlation tradeoff")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"phaseW1_runtime_tradeoff.{ext}"), dpi=150)
    plt.close(fig)

    print(f"\n  Figures saved to {fig_dir}/")


# ===========================================================================
# Synthetic smoke test
# ===========================================================================


def _make_synthetic_scenario(seed: int = 0) -> WildfireScenario:
    """Generate a 20×20 synthetic fire scenario for pipeline validation.

    Terrain: ``elevation = 100 + 10 * sin(pi * row / 20)``, flat slope.
    Fire:    Euclidean wave from ignition at (10, 10):
             ``arrival_time[i,j] = sqrt((i-10)^2 + (j-10)^2) * 0.03``
             normalised to [0, 1].

    Args:
        seed: RNG seed for observation sampling.

    Returns:
        :class:`~ham.data.wildfire.WildfireScenario`.
    """
    H, W = 20, 20
    pixel_spacing_m = 1.0
    ign_row, ign_col = 10, 10

    # Terrain rasters
    rows_g, cols_g = np.mgrid[0:H, 0:W]
    elev_raster = (100.0 + 10.0 * np.sin(np.pi * rows_g / H)).astype(np.float32)
    slope_raster = np.zeros((H, W), dtype=np.float32)
    aspect_raster = np.zeros((H, W), dtype=np.float32)
    canopy_raster = np.zeros((H, W), dtype=np.float32)
    fuel_code_raster = np.full((H, W), 5, dtype=np.int32)
    weather_vec = np.zeros(4, dtype=np.float32)
    origin_xy = np.zeros(2, dtype=np.float32)

    # Arrival times
    dr = rows_g - ign_row
    dc = cols_g - ign_col
    arrival_hours = np.sqrt(dr**2 + dc**2).astype(np.float32) * 0.03
    arrival_hours[ign_row, ign_col] = 0.0

    t_max = float(arrival_hours.max())
    if t_max < 1e-8:
        t_max = 1.0
    arrival_norm = arrival_hours / t_max
    burned_mask = np.ones((H, W), dtype=bool)

    # Observation sampling (50 pixels, stratified)
    obs_pixels = stratified_sample_observations(arrival_hours, n_samples=50, seed=seed)
    obs_arrival_times = arrival_norm[obs_pixels[:, 0], obs_pixels[:, 1]]

    # Normaliser (fit on this single "scene")
    elev_std = float(elev_raster.std()) or 1.0
    normalizer_elev = (elev_raster - elev_raster.mean()) / elev_std
    normalizer_slope = slope_raster  # already zero
    normalizer_canopy = canopy_raster

    ignition_world = np.array(
        [float(ign_col) * pixel_spacing_m, float(ign_row) * pixel_spacing_m],
        dtype=np.float32,
    )

    return WildfireScenario(
        scene_id="synthetic",
        event_id="synth_00001",
        ignition_pixel=np.array([ign_row, ign_col], dtype=np.int64),
        ignition_world=ignition_world,
        arrival_times=arrival_norm,
        arrival_times_hours=arrival_hours,
        obs_pixels=obs_pixels,
        obs_arrival_times=obs_arrival_times,
        elev_raster=normalizer_elev,
        slope_raster=normalizer_slope,
        aspect_raster=aspect_raster,
        canopy_raster=normalizer_canopy,
        fuel_code_raster=fuel_code_raster,
        weather_vec=weather_vec,
        pixel_spacing_m=pixel_spacing_m,
        origin_xy=origin_xy,
        burned_mask=burned_mask,
    )


def run_synthetic(cfg: dict, output_dir: str, use_wind: bool = True) -> dict:
    """Run the synthetic smoke test end-to-end.

    Generates one 20×20 synthetic scenario, trains for ``cfg['n_epochs']``
    epochs on 50 observations, and evaluates on 200 random eval pixels.

    Args:
        cfg:        Configuration dict (typically ``get_config(quick=True)``).
        output_dir: Directory for any saved figures.
        use_wind:   Whether to include the Randers wind drift.

    Returns:
        Result dict (same schema as :func:`train_scene`).
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC SMOKE TEST  (no real dataset required)")
    print("=" * 60)

    scenario = _make_synthetic_scenario(seed=cfg["seed"])
    print(
        f"  Scenario: 20×20 grid, ignition=(10,10), "
        f"{len(scenario.obs_pixels)} training obs"
    )

    key = jax.random.PRNGKey(cfg["seed"])
    manifold = EuclideanSpace(2)
    metric = make_metric(cfg, manifold, key, use_wind=use_wind)
    solver = make_solver(cfg)

    n_epochs = cfg["n_epochs"]
    total_steps = max(n_epochs, 1)
    warmup_steps = min(10, max(1, total_steps // 10))
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=1e-5, end_value=cfg["lr"], transition_steps=warmup_steps
            ),
            optax.cosine_decay_schedule(
                init_value=cfg["lr"], decay_steps=max(1, total_steps - warmup_steps)
            ),
        ],
        boundaries=[warmup_steps],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # guard against IFT gradient spikes
        optax.adam(lr_schedule),
    )
    opt_state = optimizer.init(eqx.filter(metric, eqx.is_array))

    obs_world = jnp.asarray(
        _pixels_to_world(scenario.obs_pixels, scenario.pixel_spacing_m),
        dtype=jnp.float32,
    )
    t_obs = jnp.asarray(scenario.obs_arrival_times, dtype=jnp.float32)
    source = _ignition_to_world(scenario.ignition_pixel, scenario.pixel_spacing_m)

    train_loss_history: list = []

    print(f"  Training for {n_epochs} epochs...")
    t0 = time.time()

    for epoch in range(n_epochs):
        alpha = curriculum_alpha(
            epoch,
            warmup_epochs=cfg["curriculum_warmup_epochs"],
            ramp_epochs=cfg["curriculum_ramp_epochs"],
        )
        alpha = jnp.asarray(alpha, dtype=jnp.float32)

        def _loss(m):
            bound = bind_scenario_to_metric(m, scenario)
            bound = bound.precompute_metric_field()
            grid_shape = scenario.arrival_times.shape
            grid_extent = (
                0.0,
                grid_shape[1] * scenario.pixel_spacing_m,
                0.0,
                grid_shape[0] * scenario.pixel_spacing_m,
            )
            return compute_sparse_eikonal_loss(  # type: ignore[name-defined]
                bound,
                solver,
                source,
                obs_world,
                t_obs,
                alpha,
                grid_extent,
                grid_shape,
                scenario.pixel_spacing_m,
            )

        loss_val, grads = eqx.filter_value_and_grad(_loss)(metric)
        updates, opt_state = optimizer.update(grads, opt_state, metric)
        metric = eqx.apply_updates(metric, updates)

        train_loss_history.append(float(loss_val))
        print(
            f"  Epoch {epoch + 1:3d}/{n_epochs}: loss={float(loss_val):.6f}  alpha={alpha:.2f}"
        )

    train_time = time.time() - t0

    # Evaluate on 200 random pixels
    rng = np.random.default_rng(cfg["seed"])
    H, W = scenario.arrival_times.shape
    all_pixels = np.array([[r, c] for r in range(H) for c in range(W)], dtype=np.int64)
    eval_idx = rng.choice(
        len(all_pixels), size=min(200, len(all_pixels)), replace=False
    )
    eval_pixels = all_pixels[eval_idx]

    print(f"\n  Evaluating on {len(eval_pixels)} pixels...")
    result = evaluate_fire(metric, solver, scenario, cfg, eval_pixels=eval_pixels)

    print(
        f"\n  SYNTHETIC RESULTS\n"
        f"    Pearson r   = {result['pearson_r']:.4f}\n"
        f"    Spearman r  = {result['spearman_r']:.4f}\n"
        f"    IoU@50      = {result['iou_50']:.4f}   (coverage={result['eval_coverage']:.1%})\n"
        f"    Train time  = {train_time:.1f}s for {n_epochs} epochs"
    )

    if output_dir:
        fig_dir = os.path.join(output_dir, "figs")
        os.makedirs(fig_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(
            np.arange(1, len(train_loss_history) + 1),
            train_loss_history,
            color="#4477AA",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title(
            f"Synthetic smoke test — loss convergence  (r={result['pearson_r']:.3f})"
        )
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(fig_dir, f"phaseW1_synthetic_loss.{ext}"), dpi=150)
        plt.close(fig)
        print(f"  Figure saved to {fig_dir}/phaseW1_synthetic_loss.{{pdf,png}}")

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"w1_synthetic_seed{cfg['seed']}.eqx")
        eqx.tree_serialise_leaves(ckpt_path, metric)
        print(f"  Checkpoint saved to {ckpt_path}")

    return dict(
        scene_id="synthetic",
        seed=cfg["seed"],
        use_wind=use_wind,
        test_pearson_r_mean=result["pearson_r"],
        test_pearson_r_std=0.0,
        test_spearman_r_mean=result["spearman_r"],
        test_iou50=result["iou_50"],
        eval_coverage=result["eval_coverage"],
        train_loss_history=train_loss_history,
        val_r_history=[],
        runtime_per_epoch_s=train_time / max(n_epochs, 1),
    )


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase W1: CovariateConditionedRanders training on Sim2Real-Fire",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "sim2real_fire"
        ),
        help="Path to Sim2Real-Fire dataset root directory.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        metavar="SCENE_ID",
        help="Scene IDs to train on (space-separated). Defaults to all scenes found in data_root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/phaseW1",
        help="Output directory for figures and result logs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Reduced config (5 epochs, 15 AVBD steps, 50 obs) "
            "to verify the pipeline without long runtime."
        ),
    )
    parser.add_argument(
        "--no_wind",
        action="store_true",
        help="Riemannian ablation: disable the Randers drift term (b=0).",
    )
    parser.add_argument(
        "--batch_fires",
        type=int,
        default=None,
        metavar="B",
        help=(
            "Number of fires per vmapped training step (default: from config, 16). "
            "Larger values use more memory but compile a bigger XLA kernel — "
            "tune up for GPU, down for low-RAM machines."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "Use jax.lax.map instead of jax.vmap for the per-fire training loop. "
            "Reduces peak memory from O(B×grid) to O(grid) — required on "
            "memory-constrained TPUs (e.g. Colab v2-8 with 32 GB HBM). "
            "Slower than vmap but otherwise numerically identical."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help="Override training seeds (default: from config).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Run the synthetic smoke test on a generated 20×20 grid — "
            "no real dataset required.  Combine with --quick for fast CI."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu", "tpu"],
        help="JAX device to use (default: cpu).",
    )

    args = parser.parse_args()
    from ham.utils import configure_device

    configure_device(args.device)
    cfg = get_config(quick=args.quick)
    use_wind = not args.no_wind

    if args.batch_fires is not None:
        cfg["batch_size_fires"] = args.batch_fires
    if args.sequential:
        cfg["sequential_fires"] = True
    if args.seeds is not None:
        cfg["train_seeds"] = args.seeds

    os.makedirs(args.output_dir, exist_ok=True)

    if args.synthetic:
        run_synthetic(cfg, output_dir=args.output_dir, use_wind=use_wind)
        return

    # Auto-discover scene IDs from data_root if not specified
    scene_ids = args.scenes
    if scene_ids is None:
        scene_ids = [
            d
            for d in sorted(os.listdir(args.data_root))
            if os.path.isdir(os.path.join(args.data_root, d)) and not d.startswith(".")
        ]
        if not scene_ids:
            print(f"No scene folders found in {args.data_root}")
            return
        print(f"Auto-detected scenes: {scene_ids}")

    all_results = run_experiment(
        data_root=args.data_root,
        scene_ids=scene_ids,
        output_dir=args.output_dir,
        cfg=cfg,
        use_wind=use_wind,
    )

    if all_results:
        r_vals = [r["test_pearson_r_mean"] for r in all_results]
        spr_vals = [r.get("test_spearman_r_mean", 0.0) for r in all_results]
        iou_vals = [r["test_iou50"] for r in all_results]
        cov_vals = [r.get("eval_coverage", 1.0) for r in all_results]
        print(
            f"\n{'=' * 60}\n"
            f"  AGGREGATE RESULTS  ({len(all_results)} runs)\n"
            f"  Mean Pearson r  : {np.mean(r_vals):.4f} ± {np.std(r_vals):.4f}\n"
            f"  Mean Spearman r : {np.mean(spr_vals):.4f}   (Gahtan target ≈ 0.695)\n"
            f"  Mean IoU@50     : {np.mean(iou_vals):.4f}   (coverage={np.mean(cov_vals):.1%})\n"
            f"{'=' * 60}"
        )
    else:
        print("  No results collected — check data root and scene IDs.")


if __name__ == "__main__":
    main()
