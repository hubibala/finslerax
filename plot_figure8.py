import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Tuple, Optional
import jax

# Setup python path to load the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ham.data.sim2real_loader import Sim2RealFireLoader
from experiments.wildfire.eikonal.experiment_eikonal_flat import get_config, make_metric
from ham.data.wildfire import SceneNormalizer, load_wildfire_scenario
from experiments.wildfire.evaluate_real_fire import resolve_scene_id, evaluate_fire_gahtan
from ham.geometry.manifolds import EuclideanSpace
import equinox as eqx

def get_bounding_box(mask: np.ndarray, padding: int = 15) -> Tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return 0, mask.shape[0], 0, mask.shape[1]
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    rmin = max(0, rmin - padding)
    rmax = min(mask.shape[0], rmax + padding + 1)
    cmin = max(0, cmin - padding)
    cmax = min(mask.shape[1], cmax + padding + 1)
    
    return rmin, rmax, cmin, cmax

def create_fire_visualization(
    gt_arrival: np.ndarray,
    pred_arrival: np.ndarray,
    fire_id: str,
    correlation: float,
    output_path: str,
    vmax: Optional[float] = None,
    n_contours: int = 12,
):
    """Create a 3-panel visualization for a single fire matching Gahtan's paper."""
    valid_gt = np.isfinite(gt_arrival) & (gt_arrival < 1e5)
    rmin, rmax, cmin, cmax = get_bounding_box(valid_gt, padding=15)
    
    gt_crop = gt_arrival[rmin:rmax, cmin:cmax]
    pred_crop = pred_arrival[rmin:rmax, cmin:cmax]
    valid_gt_crop = valid_gt[rmin:rmax, cmin:cmax]
    
    pred_in_gt_region = pred_crop.copy()
    pred_in_gt_region[~valid_gt_crop] = np.nan
    
    error = np.full_like(gt_crop, np.nan)
    valid = valid_gt_crop & np.isfinite(pred_crop) & (pred_crop < 1e5)
    error[valid] = pred_in_gt_region[valid] - gt_crop[valid]
    
    if vmax is None:
        vmax = np.percentile(gt_crop[valid_gt_crop], 99) if valid_gt_crop.any() else 100
    
    gt_masked = np.ma.masked_where(~valid_gt_crop, gt_crop)
    pred_masked = np.ma.masked_where(~valid_gt_crop, pred_in_gt_region)
    error_masked = np.ma.masked_where(~valid, error)
    error_max = np.percentile(np.abs(error[valid]), 95) if valid.any() else 10
    
    # Figure with custom widths
    fig = plt.figure(figsize=(13, 4))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 0.05, 1, 0.05], wspace=0.08)
    
    cmap_arrival = plt.cm.viridis.copy()
    cmap_arrival.set_bad(color='white', alpha=0)
    cmap_error = plt.cm.RdBu_r.copy()
    cmap_error.set_bad(color='white', alpha=0)
    
    contour_levels = np.linspace(0, vmax, n_contours + 2)[1:-1]
    
    # GT
    ax_gt = fig.add_subplot(gs[0, 0])
    im_gt = ax_gt.imshow(gt_masked, cmap=cmap_arrival, vmin=0, vmax=vmax)
    ax_gt.contour(gt_crop, levels=contour_levels, colors='white', linewidths=0.6, alpha=0.8)
    ax_gt.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    for spine in ax_gt.spines.values():
        spine.set_visible(False)
    
    # Pred
    ax_pred = fig.add_subplot(gs[0, 1])
    im_pred = ax_pred.imshow(pred_masked, cmap=cmap_arrival, vmin=0, vmax=vmax)
    ax_pred.contour(pred_in_gt_region, levels=contour_levels, colors='white', linewidths=0.6, alpha=0.8)
    ax_pred.set_title('Predicted', fontsize=12, fontweight='bold')
    ax_pred.set_xticks([]); ax_pred.set_yticks([])
    for spine in ax_pred.spines.values():
        spine.set_visible(False)
    
    # Shared colorbar
    ax_cbar1 = fig.add_subplot(gs[0, 2])
    cbar1 = plt.colorbar(im_pred, cax=ax_cbar1)
    cbar1.set_label('Time (hours)', fontsize=10)
    
    # Error
    ax_err = fig.add_subplot(gs[0, 3])
    im_err = ax_err.imshow(error_masked, cmap=cmap_error, vmin=-error_max, vmax=error_max)
    ax_err.set_title('Error (Pred - GT)', fontsize=12, fontweight='bold')
    ax_err.set_xticks([]); ax_err.set_yticks([])
    for spine in ax_err.spines.values():
        spine.set_visible(False)
    
    # Error colorbar
    ax_cbar2 = fig.add_subplot(gs[0, 4])
    cbar2 = plt.colorbar(im_err, cax=ax_cbar2)
    cbar2.set_label('Error (hours)', fontsize=10)
    
    fig.suptitle(f'{fire_id} ($\\rho$={correlation:.2f})', fontsize=13, fontweight='bold')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved Figure 8 style plot to {output_path}")

def main():
    cfg = get_config(quick=False)
    key = jax.random.PRNGKey(42)
    
    data_root = r"C:\Users\hubib\Documents\Research\GeometricWorldModel\HAM\data\sim2real_fire_real"
    train_data_root = r"C:\Users\hubib\Documents\Research\GeometricWorldModel\HAM\data\sim2real_fire\0014_00426"
    checkpoint = r"C:\Users\hubib\Documents\Research\wildfire_checkpoints\w1_0014_00426_seed0.eqx"
    scene_id = "0014"
    
    from experiments.wildfire.evaluate_real_fire import make_solver
    solver = make_solver(cfg)
    
    # Load and fit Normalizer
    train_loader = Sim2RealFireLoader(train_data_root)
    train_scene_id = resolve_scene_id(train_loader, "0014")
    if not train_loader.scenes:
        parent_dir = os.path.dirname(train_data_root)
        train_loader = Sim2RealFireLoader(parent_dir)
        train_scene_id = resolve_scene_id(train_loader, os.path.basename(train_data_root))
        
    event_ids = train_loader.scenes[train_scene_id]["events"]
    raw_train = [train_loader.load_scenario(train_scene_id, eid) for eid in event_ids[:50]]
    normalizer = SceneNormalizer.fit(raw_train)
    
    # Load model
    manifold = EuclideanSpace(2)
    metric_template = make_metric(cfg, manifold, key, use_wind=True)
    from experiments.wildfire.eikonal.experiment_eikonal_flat import bind_scenario_terrain
    train_ref_scen = load_wildfire_scenario(train_loader, train_scene_id, event_ids[0], normalizer, k_train_obs=10, seed=42)
    metric_template = bind_scenario_terrain(metric_template, train_ref_scen)
    metric = eqx.tree_deserialise_leaves(checkpoint, metric_template)
    
    # Evaluate
    eval_loader = Sim2RealFireLoader(data_root)
    eval_scene_id = resolve_scene_id(eval_loader, scene_id)
    eval_event_ids = eval_loader.scenes[eval_scene_id]["events"]
    
    os.makedirs("results_real_fire", exist_ok=True)
    
    for eid in eval_event_ids:
        scen = load_wildfire_scenario(eval_loader, eval_scene_id, eid, normalizer, k_train_obs=1000, seed=42)
        
        from experiments.wildfire.evaluate_real_fire import bind_scenario_to_metric, _ignition_to_world, _predict_arrivals_dense, compute_all_metrics_gahtan
        bound_metric = bind_scenario_to_metric(metric, scen).precompute_metric_field()
        source = _ignition_to_world(scen.ignition_pixel, scen.pixel_spacing_m)
        
        grid_shape = scen.arrival_times.shape
        pixel_spacing_m = scen.pixel_spacing_m
        solver_grid_shape = (grid_shape[1], grid_shape[0])
        solver_grid_extent = (0.0, (grid_shape[1] - 1) * pixel_spacing_m, 0.0, (grid_shape[0] - 1) * pixel_spacing_m)
            
        pred_dense = _predict_arrivals_dense(bound_metric, solver, source, solver_grid_extent, solver_grid_shape)
        pred_dense = np.array(pred_dense).T
        true_T = np.array(scen.arrival_times)
        valid_mask = scen.burned_mask
        
        metrics = compute_all_metrics_gahtan(pred_dense, true_T, valid_mask)
        print(f"Event {eid}: Pearson-r = {metrics['pearson_r']:.3f}, IoU@50 = {metrics['iou_50']:.3f}")
        
        # Plot Gahtan style
        s_scalar = metrics['calibration_scalar']
        out_path = os.path.join("results_real_fire", f"gahtan_figure_{eval_scene_id}_{eid}.png")
        
        # Prepare inputs exactly like the Gahtan script
        # 1. Mask pred_dense outside of valid_mask with nan
        pred_calibrated = pred_dense * s_scalar
        
        create_fire_visualization(
            gt_arrival=true_T,
            pred_arrival=pred_calibrated,
            fire_id=f"{eval_scene_id}/{eid}",
            correlation=metrics['pearson_r'],
            output_path=out_path,
            n_contours=12
        )

if __name__ == "__main__":
    main()
