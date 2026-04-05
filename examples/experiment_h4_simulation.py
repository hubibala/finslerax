"""
experiment_h4_simulation_exp_map.py
===================================
H4 — Forward Predictive Simulation using Exponential Map

Uses deterministic geodesic shooting with learned wind as initial velocity.
Much more stable than SDE for initial testing.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import anndata

from ham.bio.data import BioDataset
from ham.solvers.geodesic import ExponentialMap

# Reuse helpers
from weinreb_vae import build_diagnostic_vae, encode_all, TARGET_FATES
from experiment_h2_directional import attach_datadriven_randers_metric, load_phase1_vae
from experiment_h3_discriminative import build_riemannian_fallback


def main():
    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    LATENT_DIM   = 8
    N_EVAL       = 600
    N_TRAJ       = 1          # deterministic = 1 trajectory per start

    print("=" * 70)
    print("H4 — FORWARD SIMULATION (Exponential Map)")
    print("Using learned wind as initial velocity for geodesic shooting")
    print("=" * 70)

    # Load data
    adata = anndata.read_h5ad(PREPROCESSED)
    X_pca = np.array(adata.obsm['X_pca'], dtype=np.float32)
    V_pca = np.array(adata.obsm['velocity_pca'], dtype=np.float32)
    labels = adata.obs['Cell type annotation'].cat.codes.values.astype(np.int32)
    fate_names = list(adata.obs['Cell type annotation'].cat.categories)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_pca).astype(np.float32)
    V_norm = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)

    dataset = BioDataset(X=jnp.array(X_norm), V=jnp.array(V_norm), labels=jnp.array(labels), lineage_pairs=None)

    test_triples = np.load(TEST_TRIPLES)[:N_EVAL]

    # Load models
    key = jax.random.PRNGKey(42)
    vae_p1 = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM, len(fate_names), key)
    vae_randers = attach_datadriven_randers_metric(vae_p1, dataset, n_anchors=2000, sigma=0.4)
    vae_null = build_riemannian_fallback(vae_randers)

    Z_all = encode_all(vae_randers, jnp.array(X_norm))

    # Fate centroids
    fate_centroids = {}
    for fname in TARGET_FATES:
        if fname not in fate_names: continue
        fidx = fate_names.index(fname)
        mask = labels == fidx
        if mask.sum() > 10:
            fate_centroids[fname] = jnp.array(Z_all[mask].mean(axis=0))

    # Prepare starts
    idx_day2 = test_triples[:, 0]
    idx_day6 = test_triples[:, 2]
    Z_start = Z_all[idx_day2]
    true_fate_names = [fate_names[labels[i]] for i in idx_day6]

    valid_mask = np.array([f in fate_centroids for f in true_fate_names])
    Z_start_v = Z_start[valid_mask]
    true_fates_v = [true_fate_names[i] for i in np.where(valid_mask)[0]]

    print(f"Evaluating on {len(Z_start_v)} valid trajectories.")

    # Exponential Map shooter
    shooter = ExponentialMap(step_size=0.015, max_steps=120)

    print("\n" + "="*60)
    print("RESULTS — Predictive Accuracy (Exponential Map Shooting)")
    print(f"{'Model':>12} {'Accuracy':>10} {'Delta':>8}")
    print("-" * 40)

    results = {}

    for name, vae_model in [("Randers", vae_randers), ("Null", vae_null)]:
        @eqx.filter_jit
        def run_batch_simulation(model, zs):
            def single_shoot(z):
                # Use wind as velocity if available
                if hasattr(model.metric, 'w_net'):
                    w = model.metric.w_net(z)
                else:
                    w = jnp.zeros_like(z)
                
                w_norm = jnp.linalg.norm(w) + 1e-8
                v0 = w * (0.8 / w_norm)  
                return shooter.shoot(model.metric, z, v0)

            return jax.vmap(single_shoot)(zs)

        print(f"  Computing {name} predictions...")
        Z_final = run_batch_simulation(vae_model, Z_start_v)

        hits = 0
        for i, z_final in enumerate(Z_final):
            true_fate = true_fates_v[i]
            d_true = float(jnp.linalg.norm(z_final - fate_centroids[true_fate]))
            d_cf = min(float(jnp.linalg.norm(z_final - c))
                       for f, c in fate_centroids.items() if f != true_fate)

            if d_true < d_cf:
                hits += 1

        acc = hits / len(Z_start_v)
        print(f"  {name:10s} → {acc:.1%}")
        results[name] = acc

    delta = results["Randers"] - results["Null"]
    print(f"  Delta = {delta:+.1%}")

    if delta > 0.03:
        print("\n✓ H4 SUPPORTED: Randers wind improves predictive shooting.")
    else:
        print("\n✗ H4 Not clearly supported in deterministic shooting.")

    print("\nH4 (Exponential Map) completed.")

    # ── Visualization ────────────────────────────────────────────────────────
    print("\nGenerating trajectory visualization...")
    # Clean Z_all for PCA
    valid_idx = np.all(np.isfinite(Z_all), axis=1)
    Z_clean = Z_all[valid_idx]
    pca2 = PCA(n_components=2).fit(Z_clean)
    z2d_all = pca2.transform(Z_all)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # Background latent dots
    ax.scatter(z2d_all[:, 0], z2d_all[:, 1], s=1, alpha=0.1, color='lightgray', label='Latent Space')
    
    # Target centroids
    colors_fates = {"Monocyte": "steelblue", "Neutrophil": "tomato", "Erythroid": "forestgreen", "Megakaryocyte": "darkorange"}
    for fname, centroid in fate_centroids.items():
        c2d = pca2.transform(np.array(centroid)[None])[0]
        color = colors_fates.get(fname, "black")
        ax.scatter(*c2d, s=150, marker='X', color=color, edgecolor='white', label=f"{fname} Target", zorder=10)

    # Pick several samples for each target fate to show trajectory 'bundles'
    N_VIZ = 4
    print(f"  Tracing {N_VIZ} example paths per fate ...")
    for fname, color in colors_fates.items():
        # Check if we have this fate in our evaluation set
        potential_indices = [i for i, f in enumerate(true_fates_v) if f == fname]
        if not potential_indices: continue
        
        # Plot up to N_VIZ exemplars
        for i, idx_v in enumerate(potential_indices[:N_VIZ]):
            z0 = Z_start_v[idx_v]
            
            # Trace geodesic under Randers wind
            w = vae_randers.metric.w_net(z0)
            w_norm = jnp.linalg.norm(w) + 1e-8
            v0 = w * (0.8 / w_norm)  
            
            # Trace full trajectory for plotting
            traj_r, _ = shooter.trace(vae_randers.metric, z0, v0)
            t2d = pca2.transform(np.array(traj_r))
            
            # Only label the first path to avoid legend clutter
            lbl = f"{fname} Paths" if i == 0 else None
            ax.plot(t2d[:, 0], t2d[:, 1], color=color, lw=1.8, alpha=0.5, zorder=5, label=lbl)
            ax.scatter(t2d[0, 0], t2d[0, 1], color=color, s=20, marker='o', edgecolors='black', alpha=0.6, zorder=6)
            ax.scatter(t2d[-1, 0], t2d[-1, 1], color=color, s=50, marker='*', edgecolors='black', alpha=0.8, zorder=7)

    ax.set_title("H4 Forward Prediction: Day-2 Progenitors Shot Toward Day-6 Fates", fontsize=12)
    ax.set_xlabel("PC1 (Latent)"); ax.set_ylabel("PC2 (Latent)")
    ax.legend(fontsize=8, loc='upper right', frameon=True, framealpha=0.9)
    
    out_img = "h4_fate_trajectories.png"
    plt.savefig(out_img, dpi=200, bbox_inches='tight')
    print(f"Saved visualization → {out_img}")

if __name__ == "__main__":
    main()