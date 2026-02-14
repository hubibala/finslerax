import os
import jax
import jax.numpy as jnp
import scvelo as scv
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx

# HAM Imports
from ham.geometry import Sphere
from ham.models.learned import NeuralRanders
from ham.bio.vae import GeometricVAE
from ham.bio.train_joint import GeometricTrainer

def load_pancreas_data():
    """
    Loads and preprocesses the Pancreas dataset.
    """
    print("Loading Pancreas dataset...")
    # 1. Load Data
    adata = scv.datasets.pancreas()
    
    # 2. Standard Preprocessing
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    
    # 3. Compute Velocity (Dynamical or Stochastic)
    # We use stochastic for speed/robustness in this demo
    scv.tl.velocity(adata, mode='stochastic')
    
    # 4. Extract Matrices
    # X: Spliced counts (normalized)
    # V: Velocity vectors
    # We need to ensure V matches X in terms of genes
    
    # Use the arrays directly. 
    # Note: NaN handling is crucial.
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    V = adata.layers['velocity']
    
    # Basic imputation for NaNs in Velocity (common in scvelo)
    V = np.nan_to_num(V, nan=0.0)
    
    # Normalize inputs for Neural Network stability
    # Simple Z-score or MinMax is usually good.
    # Here we divide by max absolute value to keep in [-1, 1] roughly
    scale_factor = np.max(np.abs(X))
    X_norm = X / scale_factor
    V_norm = V / scale_factor 
    
    # Convert to JAX arrays
    return jnp.array(X_norm), jnp.array(V_norm), adata

def plot_latent_sphere(model, x_data, adata, filename="pancreas_sphere.png"):
    """
    Visualizes the spherical latent space using a Mollweide projection.
    """
    print("Projecting to latent space...")
    key = jax.random.PRNGKey(0)
    
    # Encode all cells
    # We use the mean of the distribution for visualization
    def get_z(x):
        dist = model._get_dist(x)
        return dist.mean # This is on the sphere
    
    z_lat = jax.vmap(get_z)(x_data)
    z_np = np.array(z_lat)
    
    # Mollweide Projection (3D -> 2D Map)
    # z = (x, y, z). Convert to (longitude, latitude)
    # long: arctan2(y, x) [-pi, pi]
    # lat: arcsin(z) [-pi/2, pi/2]
    long = np.arctan2(z_np[:, 1], z_np[:, 0])
    lat = np.arcsin(z_np[:, 2])
    
    # Get cluster colors
    clusters = adata.obs['clusters']
    
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, projection="mollweide")
    
    # Scvelo usually has colors in adata.uns['clusters_colors']
    # For simplicity, we just scatter with default c
    
    # Convert categorical to integer for coloring
    codes = clusters.cat.codes
    scatter = ax.scatter(long, lat, c=codes, cmap='tab20', s=10, alpha=0.7)
    
    plt.title("Pancreas Latent Space (Learned Finsler Manifold)")
    plt.grid(True)
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")

def run_experiment():
    # 1. Setup
    key = jax.random.PRNGKey(42)
    k_model, k_metric = jax.random.split(key)
    
    # 2. Data
    data_x, data_v, adata = load_pancreas_data()
    print(f"Data Shape: {data_x.shape}")
    
    # 3. Model Initialization
    input_dim = data_x.shape[1]
    latent_dim = 3 # S^2 is 2D intrinsic, but embedded in 3D
    
    print("Initializing Geometry (Sphere + Neural Randers)...")
    manifold = Sphere(radius=1.0)
    metric = NeuralRanders(manifold, k_metric, hidden_dim=64, use_fourier=True)
    
    print("Initializing Relativistic VAE...")
    vae = GeometricVAE(input_dim, latent_dim, metric, k_model)
    
    # 4. Training
    trainer = GeometricTrainer(vae, learning_rate=1e-3)
    
    # We run for e.g. 500 epochs
    trained_vae = trainer.train(data_x, data_v, batch_size=128, epochs=500, log_interval=50)
    
    # 5. Analysis
    plot_latent_sphere(trained_vae, data_x, adata)
    
    # 6. Save Model
    eqx.tree_serialise_leaves("pancreas_vae.eqx", trained_vae)
    print("Model saved.")

if __name__ == "__main__":
    run_experiment()