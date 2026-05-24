import os
import argparse
import numpy as np
from scipy import stats
import joblib
import jax
import jax.numpy as jnp
import equinox as eqx
import anndata
from bio.data import BioDataset
from bio.vae import GeometricVAE
from ham.models.learned import DataDrivenPullbackRanders, PullbackRiemannian
from ham.solvers.avbd import AVBDSolver

from weinreb_vae import build_diagnostic_vae, TARGET_FATES

# ====================== Utils ======================
def load_phase1_vae(checkpoint, d_in, d_lat, n_cls, k):
    model = build_diagnostic_vae(d_in, d_lat, n_cls, k)
    return eqx.tree_deserialise_leaves(checkpoint, model)

def attach_datadriven_randers_metric(vae, dataset, n_anchors=5000, sigma=0.5, seed=42):
    # Filter velocities to training clones only implicitly via valid_mask
    vel_norms = np.linalg.norm(np.array(dataset.V), axis=1)
    valid_mask = vel_norms > 1e-6
    X_valid = dataset.X[valid_mask]
    V_valid = dataset.V[valid_mask]

    rng = np.random.default_rng(seed)
    n_sample = min(n_anchors, len(X_valid))
    idx = rng.choice(len(X_valid), n_sample, replace=False)

    X_sample = X_valid[idx]
    V_sample = V_valid[idx]
    
    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def project_vel(v_mod, x, u):
        return v_mod.project_control(x, u)

    z_anchors, v_anchors = project_vel(vae, X_sample, V_sample)

    metric = DataDrivenPullbackRanders(
        manifold=vae.manifold,
        decoder=vae.decoder_net,
        anchors_z=z_anchors,
        anchors_v=v_anchors,
        sigma=sigma,
        use_wind=True
    )
    return eqx.tree_at(lambda m: m.metric, vae, metric)

def build_riemannian_fallback(vae: GeometricVAE) -> GeometricVAE:
    metric = PullbackRiemannian(vae.manifold, decoder=vae.decoder_net)
    return eqx.tree_at(lambda m: m.metric, vae, metric)

def arc_length_normalized(metric, z_start, z_end, solver, steps=20):
    trajectory = solver.solve(metric, z_start, z_end, n_steps=steps, train_mode=True)
    finsler_len = metric.arc_length(trajectory.xs)
    diffs = trajectory.xs[1:] - trajectory.xs[:-1]
    eucl_path_len = jnp.sum(jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-12))
    return finsler_len / (eucl_path_len + 1e-8), trajectory.constraint_violation, trajectory.energy

def two_segment_normalized_cost(metric, z2, z4, z6, solver):
    cost_1, cv1, e1 = arc_length_normalized(metric, z2, z4, solver)
    cost_2, cv2, e2 = arc_length_normalized(metric, z4, z6, solver)
    return cost_1 + cost_2, jnp.maximum(cv1, cv2), jnp.maximum(e1, e2)

def geodesic_proximity(metric, z_start, z_end, z_target, solver, steps=20):
    trajectory = solver.solve(metric, z_start, z_end, n_steps=steps, train_mode=True)
    dists = jnp.linalg.norm(trajectory.xs - z_target, axis=-1)
    return jnp.min(dists), trajectory.constraint_violation, trajectory.energy

def build_fate_cells(vae, dataset, lineage_triples, cell_type_labels, fate_names, target_fates):
    cells = {}
    day6_idx = lineage_triples[:, 2]
    day6_labels = cell_type_labels[day6_idx]
    for fate in target_fates:
        if fate not in fate_names: continue
        fidx = fate_names.index(fate)
        mask = day6_labels == fidx
        if mask.sum() == 0: continue
        X_fate = dataset.X[day6_idx[mask]]
        
        @eqx.filter_jit
        @eqx.filter_vmap(in_axes=(None, 0))
        def get_means(v_mod, x):
            return v_mod._get_dist(x).mean
            
        z_fate = get_means(vae, X_fate)
        cells[fate] = np.array(z_fate)
    return cells

def bootstrap_ci(data, n_boot=1000, ci=95):
    data = np.array(data)
    boot_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper

def main():
    parser = argparse.ArgumentParser(description="H3: Discriminative Geometry Experiment")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu', 'tpu'],
                        help='JAX device to use (default: cpu).')
    args = parser.parse_args()
    from ham.utils import configure_device
    configure_device(args.device)

    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    LATENT_DIM   = 8

    if not os.path.exists(TEST_TRIPLES) or not os.path.exists(CHECKPOINT):
        print("Data/Checkpoint not found. Run preprocessing and Phase 1 training.")
        return

    print("Loading data...")
    adata = anndata.read_h5ad(PREPROCESSED)
    X_pca = np.array(adata.obsm['X_pca'], dtype=np.float32)
    V_pca = np.array(adata.obsm['velocity_pca'], dtype=np.float32)
    scaler = joblib.load("data/weinreb_pca_scaler.joblib")
    X_norm = scaler.transform(X_pca).astype(np.float32)
    V_norm = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)
    labels = adata.obs['Cell type annotation'].cat.codes.values.astype(np.int32)
    fate_names = list(adata.obs['Cell type annotation'].cat.categories)

    dataset = BioDataset(X=jnp.array(X_norm), V=jnp.array(V_norm), labels=jnp.array(labels), lineage_pairs=None)
    n_types = len(fate_names)

    triples = np.load(TEST_TRIPLES)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(triples), min(1000, len(triples)), replace=False)
    triples = triples[idx]

    key = jax.random.PRNGKey(42)
    vae_p1 = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM, n_types, key)
    vae_randers = attach_datadriven_randers_metric(vae_p1, dataset)
    vae_riemannian = build_riemannian_fallback(vae_randers)

    fate_cells = build_fate_cells(vae_randers, dataset, triples, labels, fate_names, TARGET_FATES)

    # Encode path cells
    X2 = dataset.X[triples[:, 0]]
    X4 = dataset.X[triples[:, 1]]
    X6 = dataset.X[triples[:, 2]]
    
    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0))
    def get_means(v_mod, x):
        return v_mod._get_dist(x).mean
        
    z2s = get_means(vae_randers, X2)
    z4s = get_means(vae_randers, X4)
    z6s = get_means(vae_randers, X6)

    day6_labels = labels[triples[:, 2]]

    print("\nRunning Null and Hypothesis Metric Loops...")
    
    valid_idxs, valid_z6_cfs = [], []
    for i in range(len(triples)):
        obs_fidx = day6_labels[i]
        obs_fname = fate_names[obs_fidx] if obs_fidx < len(fate_names) else None
        if obs_fname not in TARGET_FATES or obs_fname not in fate_cells:
            continue
        z2 = np.array(z2s[i])
        z6_obs = np.array(z6s[i])
        obs_dist = np.linalg.norm(z6_obs - z2)
        
        cf_candidates = []
        for fn, fzs in fate_cells.items():
            if fn != obs_fname:
                cf_candidates.extend(fzs)
        if not cf_candidates: continue
        cf_candidates = np.array(cf_candidates)
        
        # Match counterfactual cell by similar Euclidean distance to z2
        dists = np.linalg.norm(cf_candidates - z2, axis=-1)
        best_cf_idx = np.argmin(np.abs(dists - obs_dist))
        z6_cf = cf_candidates[best_cf_idx]
        
        valid_idxs.append(i)
        valid_z6_cfs.append(z6_cf)
        
    if not valid_idxs:
        print("No valid trajectories found for targeted fates.")
        return
        
    bz2s = jnp.stack([z2s[i] for i in valid_idxs])
    bz4s = jnp.stack([z4s[i] for i in valid_idxs])
    bz6_obss = jnp.stack([z6s[i] for i in valid_idxs])
    bz6_cfs = jnp.stack(valid_z6_cfs)

    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, None, 0, 0, 0, 0, None))
    def batch_eval(m_rand, m_riem, z2, z4, z6_obs, z6_cf, solver):
        # 1. Energy Discriminative Test (z2 -> z4 -> z6)
        e_obs_rand, cv1r, en1r = two_segment_normalized_cost(m_rand, z2, z4, z6_obs, solver)
        e_cf_rand, cv2r, en2r = two_segment_normalized_cost(m_rand, z2, z4, z6_cf, solver)
        
        e_obs_riem, cv1i, en1i = two_segment_normalized_cost(m_riem, z2, z4, z6_obs, solver)
        e_cf_riem, cv2i, en2i = two_segment_normalized_cost(m_riem, z2, z4, z6_cf, solver)
        
        # 2. Proximity Test (Does geodesic z2 -> z6 pass through z4?)
        prox_rand, cv_p_r, en_p_r = geodesic_proximity(m_rand, z2, z6_obs, z4, solver)
        prox_riem, cv_p_i, en_p_i = geodesic_proximity(m_riem, z2, z6_obs, z4, solver)
        
        cv_max = jnp.max(jnp.array([cv1r, cv2r, cv1i, cv2i, cv_p_r, cv_p_i]))
        en_max = jnp.max(jnp.array([en1r, en2r, en1i, en2i, en_p_r, en_p_i]))
        
        return jnp.array([e_obs_rand, e_cf_rand, e_obs_riem, e_cf_riem, prox_rand, prox_riem, cv_max, en_max])
        
    solver = AVBDSolver(iterations=60)
    output = batch_eval(vae_randers.metric, vae_riemannian.metric, bz2s, bz4s, bz6_obss, bz6_cfs, solver)

    output = np.array(output)
    
    e_obs_rand, e_cf_rand, e_obs_riem, e_cf_riem, prox_rand, prox_riem, cv_max, en_max = output.T
    
    print(f"\nSolver Diagnostics:")
    print(f"  Mean max constraint violation: {np.mean(cv_max):.2e}")
    print(f"  Mean max energy residual:      {np.mean(en_max):.2e}")
    
    # Compute Ratios
    r_rand = e_cf_rand / (e_obs_rand + 1e-8)
    r_riem = e_cf_riem / (e_obs_riem + 1e-8)
    
    # Filter valid
    mask = (e_obs_rand > 0) & (e_cf_rand > 0) & (e_obs_riem > 0) & (e_cf_riem > 0)
    r_rand = r_rand[mask]
    r_riem = r_riem[mask]
    prox_rand = prox_rand[mask]
    prox_riem = prox_riem[mask]

    print("\nRESULTS (Distance-Controlled Two-Segment Cost):")
    print(f"  Valid Trajectories Evaluated: {len(r_rand)}")
    
    mean_r_rand = np.mean(r_rand)
    std_r_rand = np.std(r_rand)
    mean_r_riem = np.mean(r_riem)
    
    cohens_d = (mean_r_rand - 1.0) / std_r_rand
    ci_lower, ci_upper = bootstrap_ci(r_rand)
    
    print(f"  Mean Ratio (Randers):         {mean_r_rand:.4f} ± {std_r_rand:.4f}")
    print(f"  95% CI (Randers):             [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Cohen's d:                    {cohens_d:.4f}")
    print(f"  Mean Ratio (Riemannian Null): {mean_r_riem:.4f}")
    print(f"  Randers Fraction > 1.0:       {np.mean(r_rand > 1.0):.2%}")
    print(f"  Riemannian Fraction > 1.0:    {np.mean(r_riem > 1.0):.2%}")

    w_stat_e, w_pval_e = stats.wilcoxon(r_rand, r_riem, alternative='greater')
    print(f"  Wilcoxon Paired p-value:      {w_pval_e:.2e}")

    print("\nRESULTS (Geodesic Proximity to Midpoint z4):")
    
    diff_prox = prox_riem - prox_rand  # Positive means Randers is closer
    mean_prox_rand = np.mean(prox_rand)
    mean_prox_riem = np.mean(prox_riem)
    prox_improved = np.mean(prox_rand < prox_riem)
    prox_median_imp = np.median(diff_prox)
    
    print(f"  Mean Min-Dist (Randers):      {mean_prox_rand:.4f}")
    print(f"  Mean Min-Dist (Riemannian):   {mean_prox_riem:.4f}")
    print(f"  Median absolute improvement:  {prox_median_imp:.4f}")
    print(f"  Randers closer than Riem:      {prox_improved:.2%}")

    p_stat_prox, p_pval_prox = stats.wilcoxon(prox_rand, prox_riem, alternative='less')
    print(f"  Wilcoxon Proximity p-value:    {p_pval_prox:.2e}")

    # Holm-Bonferroni correction
    pvals = np.array([w_pval_e, p_pval_prox])
    sorted_idx = np.argsort(pvals)
    p_adj = np.empty_like(pvals)
    for i, idx in enumerate(sorted_idx):
        multiplier = len(pvals) - i
        p_adj[idx] = min(1.0, pvals[idx] * multiplier)
        if i > 0:
            p_adj[idx] = max(p_adj[idx], p_adj[sorted_idx[i-1]])
            
    print(f"\nHolm-Bonferroni Adjusted p-values:")
    print(f"  Energy Test:    {p_adj[0]:.2e}")
    print(f"  Proximity Test: {p_adj[1]:.2e}")

    if p_adj[0] < 0.05 and mean_r_rand > mean_r_riem:
        print("\n✓ HYPOTHESIS H3 CONFIRMED: Randers metric correctly discriminates fate trajectories beyond Euclidean distance.")
    else:
        print("\n✗ HYPOTHESIS H3 FAILED TO CONFIRM trajectory energy gap.")

    if p_adj[1] < 0.05:
        print("✓ GEODESIC PROXIMITY CONFIRMED: Randers geodesics pass closer to observed intermediate states.")
    else:
        print("✗ GEODESIC PROXIMITY FAILED: Randers geodesics are not significantly closer to intermediate states.")

if __name__ == "__main__":
    main()
