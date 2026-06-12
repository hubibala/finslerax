import argparse
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

# Ensure HAMTools imports work
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from experiments.wildfire.eikonal.experiment_eikonal_flat import (
    _ignition_to_world,
    bind_scenario_to_metric,
    get_config,
    make_metric,
    make_solver,
)
from ham.data.sim2real_loader import Sim2RealFireLoader
from ham.data.wildfire import (
    SceneNormalizer,
    load_wildfire_scenario,
    train_val_test_split,
)
from ham.geometry.manifolds import EuclideanSpace


def compute_all_metrics_gahtan(
    pred_T: np.ndarray, true_T: np.ndarray, valid_mask: np.ndarray
):
    """Computes Pearson-r and IoU@50 matching Gahtan's paper."""
    valid = valid_mask & np.isfinite(pred_T) & np.isfinite(true_T) & (pred_T < 1e5)
    if valid.sum() < 10:
        return {
            "relative_rmse": float("nan"),
            "rmse": float("nan"),
            "correlation": float("nan"),
            "mae": float("nan"),
            "iou_50": 0.0,
        }

    pred_valid = pred_T[valid]
    true_valid = true_T[valid]

    if np.std(pred_valid) > 1e-8 and np.std(true_valid) > 1e-8:
        correlation = np.corrcoef(pred_valid, true_valid)[0, 1]
    else:
        correlation = float("nan")

    # Apply calibration scalar 's' to predicted arc lengths to match GT time scale
    mean_pred = float(np.mean(pred_valid))
    mean_gt = float(np.mean(true_valid))
    s = mean_gt / max(mean_pred, 1e-8)
    pred_calibrated = pred_valid * s

    rmse = np.sqrt(np.mean((pred_calibrated - true_valid) ** 2))
    max_time = true_valid.max()
    relative_rmse = rmse / max_time if max_time > 1e-6 else float("nan")
    mae = np.mean(np.abs(pred_calibrated - true_valid))

    metrics = {
        "relative_rmse": float(relative_rmse),
        "rmse": float(rmse),
        "pearson_r": float(correlation),
        "mae": float(mae),
        "calibration_scalar": float(s),
    }

    for frac in [0.25, 0.5, 0.75, 1.0]:
        t_threshold = max_time * frac
        pred_burned = pred_calibrated <= t_threshold
        true_burned = true_valid <= t_threshold
        intersection = (pred_burned & true_burned).sum()
        union = (pred_burned | true_burned).sum()
        iou = intersection / union if union > 0 else 0.0
        metrics[f"iou_{int(frac * 100)}"] = float(iou)

    return metrics


def _predict_arrivals_dense(metric, solver, source, grid_extent, grid_shape):
    """Runs the Eikonal solver over the full grid."""
    T, _, _ = solver.solve(metric, jnp.array([source]), grid_extent, grid_shape)
    return T


def evaluate_fire_gahtan(metric, solver, scenario):
    """Binds the scenario and evaluates Eikonal arrival times."""
    bound_metric = bind_scenario_to_metric(metric, scenario).precompute_metric_field()

    # Ignition pixel to world coordinates
    source = _ignition_to_world(scenario.ignition_pixel, scenario.pixel_spacing_m)

    # The Eikonal solver assumes coords[0] varies along the first dimension of the grid.
    # But CovariateConditionedRanders expects coords[0] to be x (cols) and coords[1] to be y (rows).
    # So we must pass the solver a grid of shape (cols, rows).
    grid_shape = scenario.arrival_times.shape
    pixel_spacing_m = scenario.pixel_spacing_m
    solver_grid_shape = (grid_shape[1], grid_shape[0])
    solver_grid_extent = (
        0.0,
        (grid_shape[1] - 1) * pixel_spacing_m,
        0.0,
        (grid_shape[0] - 1) * pixel_spacing_m,
    )

    # Predict arrivals for all pixels
    pred_dense = _predict_arrivals_dense(
        bound_metric, solver, source, solver_grid_extent, solver_grid_shape
    )
    pred_dense = np.array(pred_dense).T
    true_T = np.array(scenario.arrival_times)
    valid_mask = scenario.burned_mask

    return compute_all_metrics_gahtan(pred_dense, true_T, valid_mask)


def resolve_scene_id(loader, short_id):
    for s_id in loader.scenes.keys():
        if s_id.startswith(short_id):
            return s_id
    return short_id


def main():
    parser = argparse.ArgumentParser(description="Evaluate W1 model on real fire data")
    parser.add_argument(
        "--data_root", type=str, required=True, help="Path to the real fire dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained .eqx checkpoint",
    )
    parser.add_argument(
        "--scene_id", type=str, default="0001", help="Scene ID to evaluate on"
    )
    parser.add_argument(
        "--train_data_root",
        type=str,
        default=None,
        help="Path to original train data (to fit Normalizer). Defaults to data_root.",
    )
    args = parser.parse_args()

    cfg = get_config(quick=False)
    key = jax.random.PRNGKey(42)
    solver = make_solver(cfg)

    # We must fit the SceneNormalizer exactly as it was fit during training.
    # Typically, it's fit on the training events of the scene.
    train_data_root = args.train_data_root if args.train_data_root else args.data_root
    train_loader = Sim2RealFireLoader(train_data_root)
    train_scene_id = resolve_scene_id(train_loader, args.scene_id)

    if not train_loader.scenes:
        # If train_loader found no scenes, it might be because train_data_root is the scene itself.
        parent_dir = os.path.dirname(train_data_root)
        train_loader = Sim2RealFireLoader(parent_dir)
        train_scene_id = resolve_scene_id(
            train_loader, os.path.basename(train_data_root)
        )

    try:
        event_ids = train_loader.scenes[train_scene_id]["events"]
    except (AttributeError, KeyError):
        mask_dir = os.path.join(
            train_data_root, train_scene_id, "Satellite_Images_Mask"
        )
        if not os.path.exists(mask_dir):
            mask_dir = os.path.join(train_data_root, "Satellite_Images_Mask")
        event_ids = [
            d
            for d in sorted(os.listdir(mask_dir))
            if os.path.isdir(os.path.join(mask_dir, d))
        ]

    scenarios_keys = [(train_scene_id, eid) for eid in event_ids]
    train_list, _, _ = train_val_test_split(
        scenarios_keys,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
    )

    print(
        f"Fitting normalizer on {len(train_list)} training fires from {train_scene_id}..."
    )
    raw_train = []
    for sid, eid in train_list:
        raw_train.append(train_loader.load_scenario(sid, eid))
    normalizer = SceneNormalizer.fit(raw_train)
    print("Normalizer fit successfully.")

    # Load Evaluation Data
    eval_loader = Sim2RealFireLoader(args.data_root)
    eval_scene_id = resolve_scene_id(eval_loader, args.scene_id)
    try:
        eval_event_ids = eval_loader.scenes[eval_scene_id]["events"]
    except (AttributeError, KeyError):
        mask_dir = os.path.join(args.data_root, eval_scene_id, "Satellite_Images_Mask")
        if not os.path.exists(mask_dir):
            mask_dir = os.path.join(args.data_root, "Satellite_Images_Mask")
        eval_event_ids = [
            d
            for d in sorted(os.listdir(mask_dir))
            if os.path.isdir(os.path.join(mask_dir, d))
        ]

    print(f"Loading model from {args.checkpoint}...")
    manifold = EuclideanSpace(2)
    metric_template = make_metric(cfg, manifold, key, use_wind=True)

    from experiments.wildfire.eikonal.experiment_eikonal_flat import (
        bind_scenario_terrain,
    )

    # We must bind the metric to a scenario before deserializing, because the model
    # was saved as a bound metric (with terrain rasters baked in).
    # IMPORTANT: We must bind to the *training* scene so the raster shapes match the saved checkpoint!
    train_ref_scen = load_wildfire_scenario(
        train_loader, train_scene_id, event_ids[0], normalizer, k_train_obs=10, seed=42
    )
    metric_template = bind_scenario_terrain(metric_template, train_ref_scen)
    metric = eqx.tree_deserialise_leaves(args.checkpoint, metric_template)

    print(f"Evaluating on {len(eval_event_ids)} unseen events...")

    results = []
    for eid in eval_event_ids:
        scen = load_wildfire_scenario(
            eval_loader, eval_scene_id, eid, normalizer, k_train_obs=1000, seed=42
        )
        metrics = evaluate_fire_gahtan(metric, solver, scen)
        results.append(metrics)
        print(
            f"Event {eid}: Pearson-r = {metrics['pearson_r']:.3f}, IoU@50 = {metrics['iou_50']:.3f}"
        )

    avg_r = np.nanmean([r["pearson_r"] for r in results])
    avg_iou = np.nanmean([r["iou_50"] for r in results])

    print("\n--- FINAL RESULTS ---")
    print(f"Average Pearson-r: {avg_r:.4f}")
    print(f"Average IoU@50:    {avg_iou:.4f}")


if __name__ == "__main__":
    main()
