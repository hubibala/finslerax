from ham.training.losses import GeodesicSprayLoss
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx
import os
import optax
from typing import Callable, Tuple, Any

# Import our custom modules
from ham.bio.data import DataLoader, BioDataset
from ham.bio.vae import GeometricVAE
from ham.geometry.surfaces import Hyperboloid
from ham.models.learned import NeuralRanders
from ham.solvers.avbd import AVBDSolver
from ham.vis.hyperbolic import plot_poincare_disk, project_to_poincare
import ham.utils.download_weinreb as downloader

from ham.training.losses import (
    ReconstructionLoss, 
    KLDivergenceLoss, 
    ContrastiveAlignmentLoss, 
    MetricAnchorLoss, 
    MetricSmoothnessLoss
)
from ham.training.pipeline import TrainingPhase, HAMPipeline

# --- Helper Utilities ---

def get_filter_fn(selector):
    """Creates an equinox filter spec from a selector function."""
    def filter_spec(model):
        base_mask = jax.tree_util.tree_map(lambda _: False, model)
        targets = selector(model)
        def make_true(n):
            return jax.tree_util.tree_map(
                lambda leaf: True if eqx.is_array(leaf) else False, n
            )
        if isinstance(targets, tuple):
            true_mask = tuple(make_true(t) for t in targets)
        else:
            true_mask = make_true(targets)
        return eqx.tree_at(selector, base_mask, replace=true_mask)
    return filter_spec

from ham.vis.hyperbolic import project_to_poincare, project_vector_to_poincare

def generate_wind_grid(model, resolution=20):
    """
    Evaluates the learned Wind field on a 2D slice of the Poincaré disk.
    This creates a 'Weather Map' of the biological forces for visualization.
    """
    # 1. Create Grid on Poincaré Disk (radius 0.95 to stay within manifold)
    x = np.linspace(-0.9, 0.9, resolution)
    y = np.linspace(-0.9, 0.9, resolution)
    X, Y = np.meshgrid(x, y)
    grid_points_disk_2d = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    # Filter points outside unit circle
    radii_2d = np.linalg.norm(grid_points_disk_2d, axis=1)
    mask = radii_2d < 0.95
    valid_points_disk_2d = grid_points_disk_2d[mask]
    
    # 2. Pad to intrinsic dimension and lift to Hyperboloid
    int_dim = model.manifold.intrinsic_dim
    # We evaluate on the first two dimensions, padding others with zero
    padding = np.zeros((valid_points_disk_2d.shape[0], int_dim - 2))
    valid_points_disk_hi = np.concatenate([valid_points_disk_2d, padding], axis=1)
    
    # Lift formula: x = 2r / (1-r^2), x0 = (1+r^2)/(1-r^2)
    r_sq = np.sum(valid_points_disk_hi**2, axis=1, keepdims=True)
    denom = 1.0 - r_sq
    x0 = (1.0 + r_sq) / denom
    x_spatial = (2.0 * valid_points_disk_hi) / denom
    
    grid_points_hyp = jnp.array(np.concatenate([x0, x_spatial], axis=1))
    
    # 3. Evaluate Wind W(z) at these points
    def get_wind(z):
        _, W, _ = model.metric._get_zermelo_data(z)
        return W
    
    wind_vectors_hyp = jax.vmap(get_wind)(grid_points_hyp)
    
    # 4. Project Vectors back to Disk (intrinsic space)
    wind_vectors_disk_hi = np.array(project_vector_to_poincare(grid_points_hyp, wind_vectors_hyp))
    
    # For 2D visualization, we return the 2D grid and the 2D wind components
    return valid_points_disk_2d, wind_vectors_disk_hi[:, :2]

# --- Main Execution ---

def main():
    # 1. Data Setup
    data_path = "data/weinreb_raw.h5ad"
    if not os.path.exists(data_path):
        print("Downloading raw data...")
        downloader.process_weinreb()

    print("Loading Weinreb Clonal Data...")
    loader = DataLoader(path=data_path, mode='real')
    loader.preprocess(n_top_genes=100, pca_components=50) 
    dataset = loader.get_jax_data(use_pca=True)
    
    print(f"Dataset: {dataset.X.shape[0]} cells")
    if dataset.lineage_pairs is None:
        raise ValueError("No lineage pairs found! Check download.")
    print(f"Ground Truth Lineages: {dataset.lineage_pairs.shape[0]} pairs")

    # 2. Model Setup
    key = jax.random.PRNGKey(2025)
    latent_dim = 6
    
    manifold = Hyperboloid(intrinsic_dim=latent_dim)
    # Using 'tanh' as verified by test_geodesic_learning
    metric = NeuralRanders(manifold, key, hidden_dim=64) 
    vae = GeometricVAE(dataset.X.shape[1], latent_dim, metric, key)
    
    # 3. Define Pipeline Phases (Modular approach)
    # Phase 1: Manifold Learning (Encoder/Decoder)
    p1 = TrainingPhase(
        name="Manifold Learning",
        epochs=80,
        optimizer=optax.adam(1e-3),
        losses=[
            ReconstructionLoss(weight=1.0),
            KLDivergenceLoss(weight=1e-4),
            MetricAnchorLoss(weight=1.0), # Stability regularization
        ],
        filter_spec=get_filter_fn(lambda m: (m.encoder_net, m.decoder_net)),
        requires_pairs=False
    )
    
    # Phase 2: Biological Flow (Randers Metric)
    # This phase focuses on learning Westward/Eastward biological drift
    # Phase 2: Clonal Lineage Alignment
    # Focuses on long-range directions from parent-child clonal pairs
    p2 = TrainingPhase(
        name="Clonal Alignment",
        epochs=40,
        optimizer=optax.adam(1e-4),
        losses=[
            MetricAnchorLoss(weight=10.0),          # Anchor geometry to Identity
            MetricSmoothnessLoss(weight=0.1),       # Laminar flow
        ],
        filter_spec=get_filter_fn(lambda m: m.metric),
        requires_pairs=True
    )
    
    # Phase 3: Physics Consistency (RNA Velocity)
    # Focuses on local consistency using scVelo velocity estimates
    p3 = TrainingPhase(
        name="Physics Consistency",
        epochs=30,
        optimizer=optax.adam(1e-4),
        losses=[
            GeodesicSprayLoss(weight=1),          # Physical geodesics (acceleration penalty)
            MetricSmoothnessLoss(weight=1),       # Smooth vector field
        ],
        filter_spec=get_filter_fn(lambda m: m.metric),
        requires_pairs=False                        # Correct! Uses X and RNA Velocity V
    )
    
    # 4. Execute Pipeline
    print("\nStarting Modular Training Pipeline...")
    pipeline = HAMPipeline(vae)
    trained_vae = pipeline.fit(dataset, [p1, p2, p3], batch_size=256, seed=2025)

    # 5. Visualization
    print("\nGenerating Windy Tree Visualization...")
    
    # Encode all cells for plotting
    print("Encoding cells...")
    z_all = jax.vmap(lambda x: trained_vae.encode(x, jax.random.PRNGKey(0)))(dataset.X)
    z_disk = np.array(project_to_poincare(z_all))
    
    # Grid for wind field evaluation
    grid_pts, grid_wind = generate_wind_grid(trained_vae, resolution=25)
    
    # Scatter plot
    c = dataset.labels
    fig, ax = plt.subplots(figsize=(10, 10))
    # Cells (latent embedding)
    ax.scatter(z_disk[:,0], z_disk[:,1], c=c, s=5, alpha=0.4, cmap='viridis')
    # Wind (learned vector field)
    ax.quiver(grid_pts[:,0], grid_pts[:,1], grid_wind[:,0], grid_wind[:,1], 
              color='red', scale=20, width=0.002, alpha=0.8)
    
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.axis('off')
    out_file = "weinreb_geodesic_final.png"
    plt.savefig(out_file, dpi=200)
    print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    main()
