import os
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import anndata
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

from ham.nn.ebm import PseudotimePotential

def main():
    print("Loading data...")
    adata = anndata.read_h5ad("data/weinreb_diffusion.h5ad")
    
    X_diff = adata.obsm['X_diffmap']
    X_umap = adata.obsm['X_umap']
    
    # --------------------------------------------------------------------------
    # 1. Train Bi-directional KNN Mappings
    # --------------------------------------------------------------------------
    print("Training Bi-directional KNN Regressors...")
    # Forward: 10D -> 2D
    knn_fwd = KNeighborsRegressor(n_neighbors=5)
    
    # Inverse: 2D -> 10D
    knn_inv = KNeighborsRegressor(n_neighbors=5)
    
    # Subsample for faster KNN fitting
    n_samples = len(X_diff)
    sub_idx = np.random.choice(n_samples, size=min(30000, n_samples), replace=False)
    
    knn_fwd.fit(X_diff[sub_idx], X_umap[sub_idx])
    knn_inv.fit(X_umap[sub_idx], X_diff[sub_idx])
    
    # --------------------------------------------------------------------------
    # 2. Load the trained Pseudotime Potential
    # --------------------------------------------------------------------------
    print("Loading Trained Potential...")
    # We didn't explicitly save the model weights earlier, so we will retrain it quickly
    # Alternatively, we can just run a quick 50 epoch training here to get the model.
    # It takes very little time.
    import optax
    from ham.training.losses_ebm import MSELoss
    from ham.training.pipeline import HAMPipeline, TrainingPhase
    
    key = jax.random.PRNGKey(42)
    model = PseudotimePotential(dim=10, hidden_dim=64, depth=3, key=key)
    pipeline = HAMPipeline(model)
    phase = TrainingPhase(
        name="Train_Potential",
        epochs=30, # 30 is enough for convergence to 0.00
        optimizer=optax.adam(1e-3),
        losses=[MSELoss(weight=1.0)],
        filter_spec=lambda m: jax.tree_util.tree_map(eqx.is_array, m)
    )
    
    class DummyDataset:
        def __init__(self, X, dpt):
            self.X = jnp.array(X)
            self.V = jnp.zeros_like(self.X)
            self.labels = jnp.array(dpt)
            self.lineage_pairs = None
    
    dpt = adata.obs['dpt_pseudotime'].values
    dataset = DummyDataset(X_diff, dpt)
    trained_model = pipeline.fit(dataset, [phase], batch_size=256)
    
    # Define Wind function (gradient of potential)
    @jax.jit
    def energy_fn(x):
        return trained_model(x)
        
    @jax.jit
    def wind_fn(x):
        # W(x) = \nabla E(x)
        return jax.grad(energy_fn)(x)
        
    vmap_energy = jax.jit(jax.vmap(energy_fn))
    vmap_wind = jax.jit(jax.vmap(wind_fn))
    
    # --------------------------------------------------------------------------
    # 3. Create UMAP Grid and Project
    # --------------------------------------------------------------------------
    print("Evaluating Geometry on UMAP Grid...")
    u1_min, u1_max = X_umap[:, 0].min() - 1, X_umap[:, 0].max() + 1
    u2_min, u2_max = X_umap[:, 1].min() - 1, X_umap[:, 1].max() + 1
    
    # 50x50 grid
    grid_res = 50
    u1_lin = np.linspace(u1_min, u1_max, grid_res)
    u2_lin = np.linspace(u2_min, u2_max, grid_res)
    U1, U2 = np.meshgrid(u1_lin, u2_lin)
    
    U_grid = np.column_stack([U1.ravel(), U2.ravel()])
    
    # Project grid to 10D
    X_grid_10d = knn_inv.predict(U_grid)
    
    # Evaluate Energy and Wind in 10D
    E_10d = vmap_energy(X_grid_10d)
    W_10d = vmap_wind(X_grid_10d)
    
    # Push wind forward to 2D UMAP space using finite differences through KNN
    eps = 1e-3
    U_wind_step = knn_fwd.predict(X_grid_10d + eps * W_10d)
    W_2d = (U_wind_step - U_grid) / eps
    
    E_grid = E_10d.reshape(U1.shape)
    W1_grid = W_2d[:, 0].reshape(U1.shape)
    W2_grid = W_2d[:, 1].reshape(U2.shape)
    
    # --------------------------------------------------------------------------
    # 4. Plotting
    # --------------------------------------------------------------------------
    print("Plotting Heatmap and Wind Field...")
    plt.figure(figsize=(12, 10))
    
    # Plot Energy Heatmap
    contour = plt.contourf(U1, U2, E_grid, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='Pseudotime Potential E(x)')
    
    # Plot Background Cells
    subsample = 10
    plt.scatter(X_umap[::subsample, 0], X_umap[::subsample, 1], 
                s=1, c='white', alpha=0.3, label='Weinreb Data')
                
    # Plot Wind Quiver
    stride = 2
    plt.quiver(U1[::stride, ::stride], U2[::stride, ::stride], 
               W1_grid[::stride, ::stride], W2_grid[::stride, ::stride], 
               color='red', alpha=0.7, scale=20, width=0.003, label='Wind $\\nabla DPT(x)$ (Projected)')
               
    # Plot Geodesic if available
    try:
        traj_10d = np.load("results_weinreb_diffusion/trajectory_10d.npy")
        traj_umap = knn_fwd.predict(traj_10d)
        plt.plot(traj_umap[:, 0], traj_umap[:, 1], color='cyan', linewidth=3, label='Geodesic')
        plt.scatter(traj_umap[0, 0], traj_umap[0, 1], c='cyan', s=100, edgecolors='black', zorder=5)
        plt.scatter(traj_umap[-1, 0], traj_umap[-1, 1], c='magenta', s=100, edgecolors='black', zorder=5)
    except FileNotFoundError:
        pass
        
    plt.title("Weinreb DPT-Finsler Geometry (Rigorous KNN Projection)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    
    out_path = "results_weinreb_diffusion/weinreb_diffusion_heatmap_umap.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
