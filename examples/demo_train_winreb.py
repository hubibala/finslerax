import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx
import os

from ham.bio.data import DataLoader, BioDataset
from ham.bio.vae import GeometricVAE
from ham.bio.train_joint import GeometricTrainer
from ham.geometry.surfaces import Hyperboloid
from ham.models.learned import NeuralRanders
from ham.vis.hyperbolic import plot_poincare_disk
import ham.utils.download_weinreb as downloader

def generate_wind_grid(model, resolution=20):
    """
    Evaluates the learned Wind field on a uniform grid over the Poincaré disk.
    This creates a 'Weather Map' of the biological forces.
    """
    # 1. Create Grid on Poincaré Disk (radius 0.95 to stay within manifold)
    x = np.linspace(-0.9, 0.9, resolution)
    y = np.linspace(-0.9, 0.9, resolution)
    X, Y = np.meshgrid(x, y)
    grid_points_disk = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    # Filter points outside unit circle
    radii = np.linalg.norm(grid_points_disk, axis=1)
    mask = radii < 0.95
    valid_points_disk = grid_points_disk[mask]
    
    # 2. Lift to Hyperboloid
    # x_spatial = 2*y / (1 - |y|^2)
    # x0 = (1 + |y|^2) / (1 - |y|^2)
    r_sq = np.sum(valid_points_disk**2, axis=1, keepdims=True)
    denom = 1.0 - r_sq
    
    x0 = (1.0 + r_sq) / denom
    x_spatial = (2.0 * valid_points_disk) / denom
    
    # Hyperboloid points: (x0, x1, x2)
    grid_points_hyp = np.concatenate([x0, x_spatial], axis=1)
    grid_points_hyp = jnp.array(grid_points_hyp)
    
    # 3. Evaluate Wind W(z) at these points
    def get_wind(z):
        # Returns W in tangent space of z (Ambient coordinates)
        # We capture 'model' from the outer scope
        _, W, _ = model.metric._get_zermelo_data(z)
        return W
    
    # FIX: Call get_wind(z) only, not get_wind(model, z)
    wind_vectors_hyp = jax.vmap(get_wind)(grid_points_hyp)
    
    # 4. Project Vectors back to Disk for plotting
    x0 = grid_points_hyp[:, 0:1]
    x_s = grid_points_hyp[:, 1:]
    v0 = wind_vectors_hyp[:, 0:1]
    v_s = wind_vectors_hyp[:, 1:]
    
    denom_proj = (1.0 + x0)**2
    # Pushforward formula for Stereographic projection (Hyperboloid -> Poincare)
    wind_vectors_disk = (v_s * (1.0 + x0) - x_s * v0) / denom_proj
    
    return valid_points_disk, np.array(wind_vectors_disk)

def main():
    # 1. Load Data
    data_path = "data/weinreb_raw.h5ad"
    if not os.path.exists(data_path):
        downloader.download_weinreb()
        downloader.process_weinreb()

    print("Loading Weinreb Dataset...")
    loader = DataLoader(path=data_path, mode='real')
    loader.preprocess(n_top_genes=2000, pca_components=50)
    dataset = loader.get_jax_data(use_pca=True)
    
    # 2. Initialize Model
    key = jax.random.PRNGKey(2025)
    data_dim = dataset.X.shape[1]
    latent_dim = 2
    
    manifold = Hyperboloid(intrinsic_dim=latent_dim)
    
    # NeuralRanders: Hidden dim 64 is good for capacity
    metric = NeuralRanders(manifold, key, hidden_dim=64)
    
    vae = GeometricVAE(data_dim, latent_dim, metric, key)
    
    # 3. Train with Custom Loop for Annealing
    trainer = GeometricTrainer(vae, learning_rate=1e-3, seed=42)
    
    print("\n=== Phase 1: Manifold Learning (with KL Annealing) ===")
    
    epochs_manifold = 80
    batch_size = 128
    data_x = dataset.X
    data_v = dataset.V # Not used for VAE but passed
    num_samples = data_x.shape[0]
    
    # We manually run the loop to control KL weight
    # Note: GeometricVAE.loss_fn usually sums KL. 
    # We should ideally pass a weight, but for now we assume standard training.
    # The trick for Hyperbolic VAEs is to train LONGER.
    
    trained_model = trainer.train(dataset, batch_size=128, epochs_manifold=200, epochs_metric=0)

    print("\n=== Phase 2: Wind Learning (50 epochs) ===")
    # Train the wind
    trained_model = trainer.train(dataset, batch_size=128, epochs_manifold=0, epochs_metric=100)
    
    # 4. Visualization (The Good Part)
    print("Generating High-Fidelity Plot...")
    
    # A. Encode Cells
    def encode_batch(model, x):
        return model._get_dist(x).mean

    batch_size_inf = 1000
    N = dataset.X.shape[0]
    zs = []
    for i in range(0, N, batch_size_inf):
        batch_x = dataset.X[i:i+batch_size_inf]
        z_batch = jax.vmap(lambda x: encode_batch(trained_model, x))(batch_x)
        zs.append(z_batch)
    z_all = jnp.concatenate(zs, axis=0)
    
    # B. Generate Wind Grid (Weather Map)
    grid_pts, grid_wind = generate_wind_grid(trained_model, resolution=25)
    
    # C. Plot
    subset_idx = np.random.choice(N, min(N, 3000), replace=False)
    z_sub = np.array(z_all[subset_idx])
    c = dataset.labels[subset_idx] if dataset.labels is not None else None
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Plot Cells (Background)
    # Use existing helper for projection logic, but customize here
    from ham.vis.hyperbolic import project_to_poincare
    z_sub_disk = np.array(project_to_poincare(z_sub))
    
    sc = ax.scatter(z_sub_disk[:, 0], z_sub_disk[:, 1], c=c, cmap='viridis', s=10, alpha=0.6, edgecolors='none')
    
    # 2. Plot Streamlines / Quiver (Foreground)
    # Quiver is often clearer than streams for discrete flow fields
    ax.quiver(grid_pts[:, 0], grid_pts[:, 1], 
              grid_wind[:, 0], grid_wind[:, 1],
              color='red', alpha=0.8, 
              pivot='mid', units='width', width=0.002, scale=15,
              headwidth=4, headlength=5)

    # Boundary
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--', alpha=0.5)
    ax.add_artist(circle)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title("HAM: Differentiation Flow on Hyperbolic Manifold")
    ax.axis('off')
    
    output_path = "weinreb_final.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    main()