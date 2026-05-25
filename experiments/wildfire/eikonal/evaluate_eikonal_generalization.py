import os
import math
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure HAMTools imports work
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ham.geometry.manifolds import EuclideanSpace
from ham.models.wildfire import CovariateConditionedRanders
from ham.data.wildfire import SceneNormalizer, train_val_test_split, load_wildfire_scenario
from ham.solvers.eikonal import EikonalSolver
from ham.data.sim2real_loader import Sim2RealFireLoader

# Import from experiment_eikonal_flat
from experiments.wildfire.eikonal.experiment_eikonal_flat import (
    get_config, make_metric, evaluate_fire, make_solver, _predict_arrivals_eval
)

class CKA(object):
    def __init__(self):
        pass 
    
    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

def load_model(filepath, cfg, key, ref_scen):
    manifold = EuclideanSpace(2)
    metric = make_metric(cfg, manifold, key, use_wind=True)
    from experiments.wildfire.eikonal.experiment_eikonal_flat import bind_scenario_terrain
    if ref_scen is not None:
        metric = bind_scenario_terrain(metric, ref_scen)
    model = eqx.tree_deserialise_leaves(filepath, metric)
    return model

def resolve_scene_id(loader, short_id):
    for s_id in loader.scenes.keys():
        if s_id.startswith(short_id):
            return s_id
    return short_id

def compute_all_metrics_gahtan(pred_T: np.ndarray, true_T: np.ndarray, valid_mask: np.ndarray):
    valid = valid_mask & np.isfinite(pred_T) & np.isfinite(true_T) & (pred_T < 1e5)
    if valid.sum() < 10:
        return {'relative_rmse': float('nan'), 'rmse': float('nan'), 'correlation': float('nan'), 'mae': float('nan'), 'iou_50': 0.0}
    
    pred_valid = pred_T[valid]
    true_valid = true_T[valid]
    
    if np.std(pred_valid) > 1e-8 and np.std(true_valid) > 1e-8:
        correlation = np.corrcoef(pred_valid, true_valid)[0, 1]
    else:
        correlation = float('nan')
        
    # Apply calibration scalar 's' to predicted arc lengths to match GT time scale
    mean_pred = float(np.mean(pred_valid))
    mean_gt = float(np.mean(true_valid))
    s = mean_gt / max(mean_pred, 1e-8)
    pred_calibrated = pred_valid * s
        
    rmse = np.sqrt(np.mean((pred_calibrated - true_valid) ** 2))
    max_time = true_valid.max()
    relative_rmse = rmse / max_time if max_time > 1e-6 else float('nan')
    mae = np.mean(np.abs(pred_calibrated - true_valid))
    
    metrics = {
        'relative_rmse': float(relative_rmse),
        'rmse': float(rmse),
        'correlation': float(correlation),
        'mae': float(mae),
        'calibration_scalar': float(s)
    }
    
    for frac in [0.25, 0.5, 0.75, 1.0]:
        t_threshold = max_time * frac
        pred_burned = pred_valid <= t_threshold
        true_burned = true_valid <= t_threshold
        intersection = (pred_burned & true_burned).sum()
        union = (pred_burned | true_burned).sum()
        iou = intersection / union if union > 0 else 0.0
        metrics[f'iou_{int(frac*100)}'] = float(iou)
        
    return metrics

def evaluate_fire_gahtan(metric, solver, scenario):
    from experiments.wildfire.eikonal.experiment_eikonal_flat import bind_scenario_to_metric, _predict_arrivals_dense, _ignition_to_world
    bound_metric = bind_scenario_to_metric(metric, scenario).precompute_metric_field()
    source = _ignition_to_world(scenario.ignition_pixel, scenario.pixel_spacing_m)
    grid_shape = scenario.arrival_times.shape
    grid_extent = (0.0, grid_shape[1] * scenario.pixel_spacing_m, 0.0, grid_shape[0] * scenario.pixel_spacing_m)
    
    pred_dense = _predict_arrivals_dense(bound_metric, solver, source, grid_extent, grid_shape)
    pred_dense = np.array(pred_dense)
    true_T = np.array(scenario.arrival_times)
    valid_mask = scenario.burned_mask
    
    return compute_all_metrics_gahtan(pred_dense, true_T, valid_mask)


def main():
    checkpoints_dir = r"C:\Users\hubib\Documents\Research\wildfire_checkpoints"
    data_root = r"C:\Users\hubib\Documents\Research\Gahtan-Eikonal-Wildfire\differentiable-eikonal-wildfire\sim2real_fire_data\simulation_data"
    output_dir = "results_generalization"
    os.makedirs(output_dir, exist_ok=True)
    
    cfg = get_config(quick=True)
    cfg["quick"] = False
    cfg["eikonal_iters"] = 150  # Increase iterations to ensure convergence on all fires
    
    key = jax.random.PRNGKey(42)
    solver = make_solver(cfg)
    loader = Sim2RealFireLoader(data_root)
    
    scenes = ["0014_00426", "0005", "0003"]
    test_scenarios = {}
    
    for eval_scene_short in scenes:
        eval_scene = resolve_scene_id(loader, eval_scene_short)
        print(f"\nLoading test scenarios for {eval_scene}...")
        
        try:
            event_ids = loader.scenes[eval_scene]["events"]
        except (AttributeError, KeyError):
            mask_dir = os.path.join(data_root, eval_scene, "Satellite_Images_Mask")
            weather_dir = os.path.join(data_root, eval_scene, "Weather_Data")
            if os.path.isdir(mask_dir):
                event_ids = [d for d in sorted(os.listdir(mask_dir)) if os.path.isdir(os.path.join(mask_dir, d)) and os.path.exists(os.path.join(weather_dir, f"{d}.txt"))]
            else:
                event_ids = []
                        
        scenarios_keys = [(eval_scene, eid) for eid in event_ids]
        train_list, val_list, test_list = train_val_test_split(
            scenarios_keys, train_ratio=cfg["train_ratio"], val_ratio=cfg["val_ratio"], seed=cfg["seed"]
        )
        
        print(f"Fitting normalizer on full training set for {eval_scene_short}...")
        raw_train = []
        for sid, eid in train_list:
            raw_train.append(loader.load_scenario(sid, eid))
        normalizer = SceneNormalizer.fit(raw_train)
        
        test_scenarios[eval_scene_short] = []
        for sid, eid in test_list[:10]:
            scen = load_wildfire_scenario(loader, sid, eid, normalizer, k_train_obs=1000, seed=42)
            test_scenarios[eval_scene_short].append(scen)
            
    models = {}
    print("\nLoading models...")
    for scene in scenes:
        models[scene] = []
        ref_scen = test_scenarios[scene][0] if len(test_scenarios[scene]) > 0 else None
        
        for seed in [0, 1, 2]:
            filepath = os.path.join(checkpoints_dir, f"w1_{scene}_seed{seed}.eqx")
            if os.path.exists(filepath):
                models[scene].append(load_model(filepath, cfg, key, ref_scen))
            else:
                print(f"Warning: Model {filepath} not found.")

    results = {
        "In-Distribution": {"correlation": [], "iou_50": [], "relative_rmse": [], "mae": []},
        "Out-of-Distribution": {"correlation": [], "iou_50": [], "relative_rmse": [], "mae": []}
    }
            
    for source_scene in scenes:
        for eval_scene in scenes:
            dist_type = "In-Distribution" if source_scene == eval_scene else "Out-of-Distribution"
            print(f"\nEvaluating models trained on {source_scene} against {eval_scene} ({dist_type})")
            
            scen_list = test_scenarios[eval_scene]
            
            scene_corrs = []
            scene_ious = []
            scene_rrmses = []
            scene_maes = []
            
            for m_idx, m in enumerate(models[source_scene]):
                seed_corrs = []
                seed_ious = []
                seed_rrmses = []
                seed_maes = []
                for scen in scen_list:
                    res = evaluate_fire_gahtan(m, solver, scen)
                    if not np.isnan(res['correlation']):
                        seed_corrs.append(res['correlation'])
                        seed_ious.append(res['iou_50'])
                        seed_rrmses.append(res['relative_rmse'])
                        seed_maes.append(res['mae'])
                
                avg_corr = np.nanmean(seed_corrs)
                avg_iou = np.nanmean(seed_ious)
                avg_rrmse = np.nanmean(seed_rrmses)
                avg_mae = np.nanmean(seed_maes)
                print(f"  Model seed {m_idx}: Corr = {avg_corr:.3f}, IoU@50 = {avg_iou:.3f}, RelRMSE = {avg_rrmse:.3f}, MAE = {avg_mae:.3f}")
                scene_corrs.append(avg_corr)
                scene_ious.append(avg_iou)
                scene_rrmses.append(avg_rrmse)
                scene_maes.append(avg_mae)
                
            results[dist_type]["correlation"].extend(scene_corrs)
            results[dist_type]["iou_50"].extend(scene_ious)
            results[dist_type]["relative_rmse"].extend(scene_rrmses)
            results[dist_type]["mae"].extend(scene_maes)
            
    print("\nPlotting Performance...")
    labels = ["In-Distribution", "Out-of-Distribution"]
    
    def safe_mean(arr): return np.nanmean(arr) if len(arr) > 0 else 0.0
    def safe_std(arr): return np.nanstd(arr) if len(arr) > 0 else 0.0
    
    metrics_to_plot = {
        'Pearson R': ('correlation', safe_mean, safe_std),
        'IoU@50': ('iou_50', safe_mean, safe_std),
        'Rel RMSE': ('relative_rmse', safe_mean, safe_std),
    }
    
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, (label, (key, mean_fn, std_fn)) in enumerate(metrics_to_plot.items()):
        means = [mean_fn(results["In-Distribution"][key]), mean_fn(results["Out-of-Distribution"][key])]
        stds = [std_fn(results["In-Distribution"][key]), std_fn(results["Out-of-Distribution"][key])]
        ax.bar(x + (idx - 1)*width, means, width, yerr=stds, label=label, capsize=5)
        
    ax.set_ylabel('Score')
    ax.set_title('Cross-Scene Generalization Performance (Exact Gahtan Metrics)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig(os.path.join(output_dir, "generalization_performance.png"))
    plt.close()
    
    print("\n--- Computing CKA Similarity ---")
    ref_scen = test_scenarios[scenes[0]][0]
    
    flat_models = []
    model_labels = []
    for scene in scenes:
        for seed in [0,1,2]:
            flat_models.append(models[scene][seed])
            model_labels.append(f"{scene} s{seed}")
            
    n_models = len(flat_models)
    cka_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        m1 = flat_models[i]
        m1_bound = m1.bind_scene(
            elev=jnp.asarray(ref_scen.elev_raster, dtype=jnp.float32),
            slope=jnp.asarray(ref_scen.slope_raster, dtype=jnp.float32),
            aspect=jnp.asarray(ref_scen.aspect_raster, dtype=jnp.float32),
            canopy=jnp.asarray(ref_scen.canopy_raster, dtype=jnp.float32),
            fuel_codes=jnp.asarray(ref_scen.fuel_code_raster, dtype=jnp.int32),
            weather_vec=jnp.asarray(ref_scen.weather_vec, dtype=jnp.float32),
            pixel_spacing_m=float(ref_scen.pixel_spacing_m),
            origin_xy=jnp.asarray(ref_scen.origin_xy, dtype=jnp.float32),
        ).precompute_metric_field()
        
        feat1 = m1_bound.metric_field.reshape(-1, 5)
        
        for j in range(i, n_models):
            m2 = flat_models[j]
            m2_bound = m2.bind_scene(
                elev=jnp.asarray(ref_scen.elev_raster, dtype=jnp.float32),
                slope=jnp.asarray(ref_scen.slope_raster, dtype=jnp.float32),
                aspect=jnp.asarray(ref_scen.aspect_raster, dtype=jnp.float32),
                canopy=jnp.asarray(ref_scen.canopy_raster, dtype=jnp.float32),
                fuel_codes=jnp.asarray(ref_scen.fuel_code_raster, dtype=jnp.int32),
                weather_vec=jnp.asarray(ref_scen.weather_vec, dtype=jnp.float32),
                pixel_spacing_m=float(ref_scen.pixel_spacing_m),
                origin_xy=jnp.asarray(ref_scen.origin_xy, dtype=jnp.float32),
            ).precompute_metric_field()
            
            feat2 = m2_bound.metric_field.reshape(-1, 5)
            
            idx = np.random.choice(feat1.shape[0], min(2000, feat1.shape[0]), replace=False)
            f1 = feat1[idx]
            f2 = feat2[idx]
            
            cka = CKA()
            sim = cka.linear_CKA(f1, f2)
            cka_matrix[i, j] = sim
            cka_matrix[j, i] = sim
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(cka_matrix, annot=True, xticklabels=model_labels, yticklabels=model_labels, cmap="viridis", fmt=".2f")
    plt.title("CKA Similarity of local_cnn Representations")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cka_similarity.png"))
    plt.close()
    
    print(f"\nEvaluation complete. Results saved in {output_dir}/")

if __name__ == "__main__":
    main()
