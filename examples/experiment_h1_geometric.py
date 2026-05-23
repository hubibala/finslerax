import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import joblib
import jax
import equinox as eqx
import anndata

def main():
    parser = argparse.ArgumentParser(description="H1: Geometric Topology Experiment")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu', 'tpu'],
                        help='JAX device to use (default: cpu).')
    args = parser.parse_args()
    from ham.utils import configure_device
    configure_device(args.device)

    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    LATENT_DIM   = 8

    print("="*60)
    print("H1 - GEOMETRIC TOPOLOGY EXPERIMENT")
    print("Hypothesis: Does the pullback metric JᵀJ encode fate topology?")
    print("="*60)

    if not os.path.exists(CHECKPOINT):
        print(f"Error: Checkpoint {CHECKPOINT} not found.")
        return

    print("Loading data...")
    adata = anndata.read_h5ad(PREPROCESSED)
    X_pca = np.array(adata.obsm['X_pca'], dtype=np.float32)
    labels = adata.obs['Cell type annotation'].cat.codes.values.astype(np.int32)
    fate_names = list(adata.obs['Cell type annotation'].cat.categories)
    n_types = len(fate_names)

    # Note: X_pca is already unit-variance scaled via preprocess_weinreb.py
    # but the VAE diagnostic used standard scaler originally. Let's replicate exact inputs:
    scaler = joblib.load("data/weinreb_pca_scaler.joblib")
    X_norm = scaler.transform(X_pca).astype(np.float32)

    from weinreb_vae import build_diagnostic_vae, encode_all, compute_pullback_det, knn_preservation_score, TARGET_FATES

    def load_phase1_vae(checkpoint, d_in, d_lat, n_cls, k):
        model = build_diagnostic_vae(d_in, d_lat, n_cls, k)
        return eqx.tree_deserialise_leaves(checkpoint, model)

    def get_untrained_vae(d_in, d_lat, n_cls, k):
        return build_diagnostic_vae(d_in, d_lat, n_cls, k)

    print("Loading VAE...")
    key = jax.random.PRNGKey(42)
    vae_riem = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM, n_types, key)

    print("\nEvaluating Geometric Properties over multiple subsamples:")
    
    seeds = [42, 43, 44, 45, 46]
    results_sil = []
    results_knn = []
    
    results_sil_untrained = []
    results_knn_untrained = []
    
    results_sil_pca = []

    for seed in seeds:
        print(f"\n--- Subsample Seed {seed} ---")
        rng = np.random.default_rng(seed)
        sample_size = min(15000, X_norm.shape[0])
        sample_idx = rng.choice(X_norm.shape[0], sample_size, replace=False)
        X_sample = X_norm[sample_idx]
        L_sample = labels[sample_idx]
        
        # Trained VAE
        Z = encode_all(vae_riem, X_sample)
        sil_score = silhouette_score(Z, L_sample, metric="euclidean", sample_size=min(5000, len(Z)))
        knn_score = knn_preservation_score(Z, X_sample, k=15)
        results_sil.append(sil_score)
        results_knn.append(knn_score)
        
        # Untrained VAE
        vae_untrained = get_untrained_vae(X_norm.shape[1], LATENT_DIM, n_types, jax.random.PRNGKey(seed))
        Z_untrained = encode_all(vae_untrained, X_sample)
        sil_score_u = silhouette_score(Z_untrained, L_sample, metric="euclidean", sample_size=min(5000, len(Z)))
        knn_score_u = knn_preservation_score(Z_untrained, X_sample, k=15)
        results_sil_untrained.append(sil_score_u)
        results_knn_untrained.append(knn_score_u)
        
        # PCA Only
        sil_score_pca = silhouette_score(X_sample, L_sample, metric="euclidean", sample_size=min(5000, len(X_sample)))
        results_sil_pca.append(sil_score_pca)
        
        print(f"  Trained VAE  | Sil: {sil_score:.4f}, KNN: {knn_score:.4f}")
        print(f"  Untrained    | Sil: {sil_score_u:.4f}, KNN: {knn_score_u:.4f}")
        print(f"  PCA Only     | Sil: {sil_score_pca:.4f}")

    print("\nAggregate Results (Mean ± Std):")
    mean_sil, std_sil = np.mean(results_sil), np.std(results_sil)
    mean_knn, std_knn = np.mean(results_knn), np.std(results_knn)
    mean_sil_u, std_sil_u = np.mean(results_sil_untrained), np.std(results_sil_untrained)
    mean_knn_u, std_knn_u = np.mean(results_knn_untrained), np.std(results_knn_untrained)
    mean_sil_pca, std_sil_pca = np.mean(results_sil_pca), np.std(results_sil_pca)

    print(f"  Trained VAE  | Sil: {mean_sil:.4f} ± {std_sil:.4f}, KNN: {mean_knn:.4f} ± {std_knn:.4f}")
    print(f"  Untrained    | Sil: {mean_sil_u:.4f} ± {std_sil_u:.4f}, KNN: {mean_knn_u:.4f} ± {std_knn_u:.4f}")
    print(f"  PCA Only     | Sil: {mean_sil_pca:.4f} ± {std_sil_pca:.4f}")

    # Generate one visualization using the first seed's data
    print("\nGenerating Pullback Metric Heatmap (Seed 42)...")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(X_norm.shape[0], min(15000, X_norm.shape[0]), replace=False)
    X_sample = X_norm[sample_idx]
    L_sample = labels[sample_idx]
    Z = encode_all(vae_riem, X_sample)
    
    try:
        GX, GY, logdet, pca2 = compute_pullback_det(vae_riem, Z, n_grid=20)
        print(f"  ✓ Log Determinant Heatmap successfully generated. Max: {np.max(logdet):.2f}, Min: {np.min(logdet):.2f}")
        
        plt.figure(figsize=(10, 8))
        im = plt.contourf(GX, GY, logdet, levels=20, cmap="viridis")
        plt.colorbar(im, label="log det G(z)")
        z2d = pca2.transform(Z)
        plt.scatter(z2d[:,0], z2d[:,1], c=L_sample, s=1, alpha=0.3, cmap='tab20')
        plt.title(f"Pullback Metric JᵀJ Determinant Heatmap\nKNN Score: {mean_knn:.2f} | Sil Score: {mean_sil:.2f}")
        plt.xlabel("Latent PC1")
        plt.ylabel("Latent PC2")
        plt.savefig("h1_geometric_topology.png", dpi=200, bbox_inches='tight')
        print(f"  Saved visualization to h1_geometric_topology.png")
    except Exception as e:
        print(f"  ✗ Metric computation failed: {e}")

    print("\nCONCLUSION:")
    if mean_sil > mean_sil_u and mean_knn > mean_knn_u:
        print("  ✓ H1 SUPPORTED: Trained VAE extracts meaningful geometric topology.")
        print(f"    (Sil {mean_sil:.4f} > {mean_sil_u:.4f}; KNN {mean_knn:.4f} > {mean_knn_u:.4f})")
    else:
        print("  ✗ H1 NOT SUPPORTED: Trained VAE does not meaningfully outperform untrained baseline.")

if __name__ == "__main__":
    main()
