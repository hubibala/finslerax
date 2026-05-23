import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import anndata

from ham.bio.data import BioDataset
from ham.solvers.geodesic import ExponentialMap

from weinreb_vae import build_diagnostic_vae, encode_all, TARGET_FATES
from weinreb_experiment import attach_datadriven_randers_metric, load_phase1_vae
from experiment_h3_discriminative import build_riemannian_fallback

# Extended fates for H4 as requested by reviewer
EXTENDED_FATES = ["Monocyte", "Neutrophil", "Erythroid", "Megakaryocyte"]

def bootstrap_ci(data, n_boot=1000, ci=95):
    data = np.array(data)
    boot_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper

def get_leave_one_out_centroids(Z_day6, labels_day6, fate_names, test_idx_day6):
    """Compute fate centroids excluding the specific day-6 cell from the test triple."""
    # We precompute sum and count for each fate
    fate_sums = {}
    fate_counts = {}
    for fname in EXTENDED_FATES:
        if fname not in fate_names: continue
        fidx = fate_names.index(fname)
        mask = labels_day6 == fidx
        if mask.sum() > 5:
            fate_sums[fname] = Z_day6[mask].sum(axis=0)
            fate_counts[fname] = mask.sum()
            
    # For each test cell, return its specific centroids
    test_centroids = []
    for test_idx in test_idx_day6:
        test_fidx = labels_day6[test_idx]
        test_fname = fate_names[test_fidx]
        
        centroids = {}
        for fname, fsum in fate_sums.items():
            fcount = fate_counts[fname]
            if fname == test_fname:
                # Leave one out
                centroids[fname] = (fsum - Z_day6[test_idx]) / (fcount - 1 + 1e-8)
            else:
                centroids[fname] = fsum / fcount
        test_centroids.append(centroids)
    return test_centroids

def main():
    parser = argparse.ArgumentParser(description="H4: Forward Simulation Experiment")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu', 'tpu'],
                        help='JAX device to use (default: cpu).')
    args = parser.parse_args()
    from ham.utils import configure_device
    configure_device(args.device)

    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    LATENT_DIM   = 8
    N_EVAL       = 600

    print("=" * 80)
    print("H4 — FORWARD SIMULATION (Exponential Map)")
    print("=" * 80)

    if not os.path.exists(TEST_TRIPLES) or not os.path.exists(CHECKPOINT):
        print("Required files not found.")
        return

    # Load data
    adata = anndata.read_h5ad(PREPROCESSED)
    X_pca = np.array(adata.obsm['X_pca'], dtype=np.float32)
    V_pca = np.array(adata.obsm['velocity_pca'], dtype=np.float32)
    labels = adata.obs['Cell type annotation'].cat.codes.values.astype(np.int32)
    fate_names = list(adata.obs['Cell type annotation'].cat.categories)

    scaler = joblib.load("data/weinreb_pca_scaler.joblib")
    X_norm = scaler.transform(X_pca).astype(np.float32)
    V_norm = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)

    dataset = BioDataset(X=jnp.array(X_norm), V=jnp.array(V_norm), labels=jnp.array(labels), lineage_pairs=None)
    test_triples = np.load(TEST_TRIPLES)[:N_EVAL]

    # Load models
    key = jax.random.PRNGKey(42)
    vae_p1 = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM, len(fate_names), key)
    vae_randers = attach_datadriven_randers_metric(vae_p1, dataset, n_anchors=2000, sigma=0.4)
    vae_null = build_riemannian_fallback(vae_randers)

    Z_all = encode_all(vae_randers, jnp.array(X_norm))
    
    idx_day2 = test_triples[:, 0]
    idx_day6 = test_triples[:, 2]
    Z_start = Z_all[idx_day2]
    
    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def project_vel(v_mod, x, v):
        return v_mod.project_control(x, v)
        
    V0_start = project_vel(vae_randers, jnp.array(X_norm[idx_day2]), jnp.array(V_norm[idx_day2]))
    V0_start = np.array(V0_start)

    true_fates = [fate_names[labels[i]] for i in idx_day6]
    
    # Filter to extended fates
    valid_mask = np.array([f in EXTENDED_FATES for f in true_fates])
    Z_start_v = Z_start[valid_mask]
    V0_start_v = V0_start[valid_mask]
    true_fates_v = [true_fates[i] for i in np.where(valid_mask)[0]]
    idx_day6_v = idx_day6[valid_mask]
    
    print(f"Evaluating on {len(Z_start_v)} valid trajectories (Fates: {EXTENDED_FATES}).")
    
    # Precompute leave-one-out centroids
    loo_centroids = get_leave_one_out_centroids(Z_all, labels, fate_names, idx_day6_v)

    # 1. Deterministic Ablation (v0 magnitude)
    print("\n--- Deterministic Initial Velocity (v0) Ablation ---")
    v0_scales = [0.1, 0.5, 1.0, 1.5, 2.0]
    shooter = ExponentialMap(step_size=0.015, max_steps=120)
    
    for scale in v0_scales:
        v0_scaled = V0_start_v * (scale / (np.linalg.norm(V0_start_v, axis=1, keepdims=True) + 1e-8))
        
        accs = {}
        for name, model in [("Null", vae_null), ("Randers", vae_randers)]:
            @eqx.filter_jit
            def run_batch_simulation(m, zs, vs):
                def single_shoot(z, v):
                    return shooter.shoot(m.metric, z, v)
                return jax.vmap(single_shoot)(zs, vs)
                
            Z_final = run_batch_simulation(model, Z_start_v, v0_scaled)
            
            hits = 0
            for i, zf in enumerate(Z_final):
                tf = true_fates_v[i]
                cents = loo_centroids[i]
                if tf not in cents: continue
                d_true = np.linalg.norm(zf - cents[tf])
                d_cf = min([np.linalg.norm(zf - c) for f, c in cents.items() if f != tf])
                if d_true < d_cf:
                    hits += 1
            accs[name] = hits / max(1, len(Z_start_v))
            
        print(f"  v0_scale = {scale:4.1f} | Null Acc: {accs['Null']:6.1%} | Randers Acc: {accs['Randers']:6.1%} | Delta: {accs['Randers']-accs['Null']:+6.1%}")

    # 2. Stochastic Shooting
    print("\n--- Stochastic Shooting Evaluation (K=20, v0_scale=1.0) ---")
    K = 20
    noise_sigma = 0.05
    scale = 1.0
    v0_base = V0_start_v * (scale / (np.linalg.norm(V0_start_v, axis=1, keepdims=True) + 1e-8))
    
    rng = np.random.default_rng(42)
    
    results = {"Null": [], "Randers": []}
    
    for name, model in [("Null", vae_null), ("Randers", vae_randers)]:
        print(f"  Simulating {name}...")
        
        @eqx.filter_jit
        def run_stochastic(m, zs, vs):
            def single_shoot(z, v):
                return shooter.shoot(m.metric, z, v)
            return jax.vmap(single_shoot)(zs, vs)
            
        is_correct = []
        
        for i in range(len(Z_start_v)):
            z0 = Z_start_v[i]
            v0 = v0_base[i]
            tf = true_fates_v[i]
            cents = loo_centroids[i]
            
            if tf not in cents: 
                is_correct.append(False)
                continue
            
            # Generate K noisy v0s
            v0s = v0 + rng.normal(0, noise_sigma * np.linalg.norm(v0), size=(K, LATENT_DIM))
            z0s = jnp.tile(z0, (K, 1))
            
            zfs = run_stochastic(model, z0s, jnp.array(v0s))
            zfs = np.array(zfs)
            
            # Compute hits among K trajectories
            hits = 0
            for zf in zfs:
                d_true = np.linalg.norm(zf - cents[tf])
                d_cf = min([np.linalg.norm(zf - c) for f, c in cents.items() if f != tf])
                if d_true < d_cf:
                    hits += 1
            
            # Soft assignment: if probability > 0.5, it's correct
            prob = hits / K
            is_correct.append(prob > 0.5)
            
        results[name] = np.array(is_correct)
        acc = np.mean(is_correct)
        print(f"  {name} Macro Accuracy: {acc:.1%}")

    null_correct = results["Null"]
    rand_correct = results["Randers"]
    
    acc_null = np.mean(null_correct)
    acc_rand = np.mean(rand_correct)
    ci_null = bootstrap_ci(null_correct)
    ci_rand = bootstrap_ci(rand_correct)
    
    print("\n" + "="*60)
    print("H4 ROBUSTNESS SUMMARY — Stochastic Predictive Shooting")
    print("-" * 60)
    print(f"  Randers:    {acc_rand:.1%}  95% CI: [{ci_rand[0]:.1%}, {ci_rand[1]:.1%}]")
    print(f"  Riemannian: {acc_null:.1%}  95% CI: [{ci_null[0]:.1%}, {ci_null[1]:.1%}]")
    print(f"  Delta:      {acc_rand - acc_null:+.1%}")
    
    # McNemar's Test
    import statsmodels.stats.contingency_tables as smct
    table = [[np.sum(null_correct & rand_correct), np.sum(null_correct & ~rand_correct)],
             [np.sum(~null_correct & rand_correct), np.sum(~null_correct & ~rand_correct)]]
    mcnemar = smct.mcnemar(table, exact=False, correction=True)
    p_val = mcnemar.pvalue
    print(f"  McNemar's p-value: {p_val:.2e}")
    
    if p_val < 0.05 and acc_rand > acc_null:
        print("\n✓ H4 SUPPORTED: Randers geometry significantly improves predictive shooting.")
    else:
        print("\n✗ H4 NOT SUPPORTED: No significant improvement over Riemannian null.")
        
if __name__ == "__main__":
    main()