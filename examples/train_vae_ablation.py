"""
train_vae_ablation.py
=====================
Trains a VAE identical to weinreb_vae.py phase 1 but with vel_weight=0.0,
producing data/weinreb_vae_ablation.eqx for ablation comparisons.

This lets H2/H3 compare:
  - Randers + full VAE (vel_weight=1.0)  → data/weinreb_vae_phase1.eqx
  - Randers + ablated VAE (vel_weight=0.0) → data/weinreb_vae_ablation.eqx
"""

import os, time, sys
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import anndata
from sklearn.preprocessing import StandardScaler

# Add examples dir to sys.path so we can import from weinreb_vae.py
sys.path.insert(0, os.path.dirname(__file__))
from weinreb_vae import (
    build_knn_triplet_indices,
    train_vae,
    BioDataset,
)

TARGET_CHECKPOINT = "data/weinreb_vae_ablation.eqx"

def main():
    print("=" * 60)
    print("ABLATION TRAINING — vel_weight = 0.0")
    print(f"Output: {TARGET_CHECKPOINT}")
    print("=" * 60)

    preprocessed_path = "data/weinreb_preprocessed.h5ad"
    triples_path      = "data/weinreb_lineage_triples.npy"

    for p in [preprocessed_path, triples_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run preprocess_weinreb.py first.")

    if os.path.exists(TARGET_CHECKPOINT):
        print(f"Checkpoint already exists at {TARGET_CHECKPOINT}. Delete it to retrain.")
        return

    print("Loading data ...")
    adata = anndata.read_h5ad(preprocessed_path)
    X_pca = np.array(adata.obsm["X_pca"], dtype=np.float32)
    V_pca = np.array(adata.obsm["velocity_pca"], dtype=np.float32)

    ct_series  = adata.obs["Cell type annotation"].astype("category")
    fate_names = list(ct_series.cat.categories)
    labels_np  = ct_series.cat.codes.values.astype(np.int32)
    n_types    = len(fate_names)

    scaler  = StandardScaler()
    X_pca_n = scaler.fit_transform(X_pca).astype(np.float32)
    V_pca_n = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)

    lineage_triples = np.load(triples_path).astype(np.int32)

    dataset = BioDataset(
        X=jnp.array(X_pca_n),
        V=jnp.array(V_pca_n),
        labels=jnp.array(labels_np),
        lineage_pairs=None,
    )

    print("Pre-computing KNN triplets ...")
    knn_trip_idx = build_knn_triplet_indices(
        X_pca_n, labels_np, k=15, n_triplets=30_000, seed=42
    )

    print("\nTraining ablated VAE (vel_weight=0.0) ...")
    t0  = time.time()
    key = jax.random.PRNGKey(2026)
    vae, history = train_vae(
        dataset          = dataset,
        triplet_indices  = knn_trip_idx,
        lineage_triples  = lineage_triples,
        labels_np        = labels_np,
        n_cell_types     = n_types,
        key              = key,
        latent_dim       = 8,
        epochs           = 120,
        batch_size       = 512,
        kl_cycle_len     = 20,
        kl_beta_max      = 5e-4,
        triplet_weight   = 1.0,
        triplet_margin   = 1.0,
        coherence_weight = 0.3,
        cls_weight       = 0.15,
        vel_weight       = 0.0,   # ← KEY ABLATION
    )
    print(f"\nTraining done in {(time.time() - t0) / 60:.1f} min")
    eqx.tree_serialise_leaves(TARGET_CHECKPOINT, vae)
    print(f"Saved ablation checkpoint → {TARGET_CHECKPOINT}")

if __name__ == "__main__":
    main()
