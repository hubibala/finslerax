import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx
import os
import optax

# Import our custom modules
from ham.bio.data import DataLoader, BioDataset
from ham.bio.vae import GeometricVAE
from ham.geometry.surfaces import Hyperboloid
from ham.models.learned import NeuralRanders
from ham.solvers.avbd import AVBDSolver
from ham.vis.hyperbolic import plot_poincare_disk
import ham.utils.download_weinreb as downloader

# --- The Geodesic Trainer (Integrated) ---
class GeodesicFlowTrainer:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        self.solver = AVBDSolver(step_size=0.1, iterations=15)

    @eqx.filter_jit
    def train_step_joint(self, model, opt_state, x_p, x_c, key, geo_weight=0.1):
        # x_p, x_c: Batches of Parent and Child gene expression vectors (Raw Data)
        
        # 1. Split keys for VAE sampling
        x_all = jnp.concatenate([x_p, x_c], axis=0)
        keys = jax.random.split(key, x_all.shape[0])

        def loss_fn(m):
            # --- A. Geography (VAE Loss) ---
            # Reconstruct both parents and children to learn the manifold map
            vae_losses, vae_stats = jax.vmap(m.loss_fn)(x_all, x_all, keys)
            loss_vae = jnp.mean(vae_losses)
            
            # --- B. Physics (Geodesic Action) ---
            # 1. Get Latent Codes (Gradient flows through Encoder!)
            dist_p = jax.vmap(m._get_dist)(x_p)
            dist_c = jax.vmap(m._get_dist)(x_c)
            z_p = dist_p.mean
            z_c = dist_c.mean
            
            # 2. Solve Inverse Dynamics (AVBD Solver)
            # Find the path of least action connecting z_p -> z_c
            solve_fn = jax.vmap(lambda s, e: self.solver.solve(m.metric, s, e, n_steps=6, train_mode=True))
            trajectories = solve_fn(z_p, z_c)
            
            # 3. Action Minimization (The "Wind" Loss)
            loss_action = jnp.mean(trajectories.energy)
            
            # --- C. Regularization (Stability & Smoothness) ---
            
            # Helper to extract fields pure-functionally
            def get_fields(x):
                M, W, _ = m.metric._get_zermelo_data(x)
                return M, W

            # 1. Anchor Regularization (Keep M approx Identity, W small)
            def reg_magnitudes(x):
                M, W = get_fields(x)
                dim = M.shape[-1]
                loss_M = jnp.sum((M - jnp.eye(dim))**2)
                loss_W = jnp.sum(W**2)
                return loss_M, loss_W

            # Sample points along the trajectory
            # Shape: (Batch * Steps, Dim)
            sample_pts = trajectories.xs.reshape(-1, trajectories.xs.shape[-1])
            
            loss_M_val, loss_W_val = jax.vmap(reg_magnitudes)(sample_pts)
            loss_anchor = jnp.mean(loss_M_val) + 0.001 * jnp.mean(loss_W_val)

            # 2. Jacobian Regularization (The Smoother)
            # Penalize dW/dx to ensure laminar flow
            def get_wind_only(x):
                _, W, _ = m.metric._get_zermelo_data(x)
                return W
            
            # Calculate Jacobian matrix dW/dx
            jac_fn = jax.jacfwd(get_wind_only)
            jacobians = jax.vmap(jac_fn)(sample_pts) 
            loss_smooth = jnp.mean(jacobians**2)
            
            # --- D. Total Weighted Loss ---
            # VAE: Base objective
            # Action: Scaled by geo_weight (ramps up during training)
            # Anchor: 10.0 (Strong constraint on geometry)
            # Smooth: 0.1 (Gentle constraint on turbulence)
            
            total_geo_loss = loss_action + 10.0 * loss_anchor + 0.1 * loss_smooth
            
            return loss_vae + geo_weight * total_geo_loss, (loss_vae, loss_action)

        (loss, (l_vae, l_geo)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        
        return new_model, new_opt_state, loss, l_vae, l_geo

    @eqx.filter_jit
    def train_step_manifold(self, model, opt_state, x, key):
        keys = jax.random.split(key, x.shape[0])
        
        def loss_fn(m):
            # 1. Standard VAE Loss (Recon + KL)
            batch_losses, batch_stats = jax.vmap(m.loss_fn)(x, x, keys)
            loss_vae = jnp.mean(batch_losses)
            
            # 2. Metric Regularization (CRITICAL FIX)
            # We must anchor M(z) close to Identity during Phase 1 too!
            # Otherwise, the metric tensor can explode, causing the geometry to fail.
            
            # Get the latent means (where the data actually lives)
            dist = jax.vmap(m._get_dist)(x)
            z_batch = dist.mean
            
            def reg(z):
                # Check metric at the latent point
                M, W, _ = m.metric._get_zermelo_data(z)
                # Force M approx Identity (Canonical Geometry)
                # Force W approx 0 (No wind yet)
                return jnp.mean((M - jnp.eye(M.shape[-1]))**2) + 0.01 * jnp.mean(W**2)

            loss_reg = jnp.mean(jax.vmap(reg)(z_batch))
            
            # Add a small penalty (e.g., 0.1 or 1.0) to keep geometry sane
            return loss_vae + 1.0 * loss_reg, batch_stats
            
        (loss, stats), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        
        updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        
        # Stats for logging
        recon = jnp.mean(stats[0])
        kl = jnp.mean(stats[1])
        
        return new_model, new_opt_state, loss, recon, kl

    @eqx.filter_jit
    def train_step_geodesic(self, model, opt_state, z_p, z_c):
        
        def loss_fn(m):
            # 1. Solve Inverse Dynamics (BVP)
            # Find the path of least action connecting z_p -> z_c
            # We use jax.vmap to run the solver on every pair in the batch
            solve_fn = jax.vmap(lambda s, e: self.solver.solve(m.metric, s, e, n_steps=6, train_mode=True))
            trajectories = solve_fn(z_p, z_c)
            
            # 2. Minimize Action (The logic you tested)
            # Randers Action approx: |v|^2/2 - <W, v>
            # Minimizing this forces W to align with v (the lineage direction)
            loss_action = jnp.mean(trajectories.energy)
            
            # 3. Regularize Mass Matrix M and Wind W
            # This is your exact regularization logic, mapped over the batch
            def reg(x):
                # Note: In the VAE, the metric is stored in m.metric
                M, W, _ = m.metric._get_zermelo_data(x)
                dim = M.shape[-1]
                return jnp.mean((M - jnp.eye(dim))**2) + 0.001 * jnp.mean(W**2)
            
            # We regularize at the start points of the batch
            loss_reg = jnp.mean(jax.vmap(reg)(z_p))
            
            # Use your tuned weights
            return loss_action + 10.0 * loss_reg

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        
        return new_model, new_opt_state, loss

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
    r_sq = np.sum(valid_points_disk**2, axis=1, keepdims=True)
    denom = 1.0 - r_sq
    
    x0 = (1.0 + r_sq) / denom
    x_spatial = (2.0 * valid_points_disk) / denom
    
    # Hyperboloid points: (x0, x1, x2)
    grid_points_hyp = np.concatenate([x0, x_spatial], axis=1)
    grid_points_hyp = jnp.array(grid_points_hyp)
    
    # 3. Evaluate Wind W(z) at these points
    def get_wind(z):
        _, W, _ = model.metric._get_zermelo_data(z)
        return W
    
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
    latent_dim = 2
    
    manifold = Hyperboloid(intrinsic_dim=latent_dim)
    # Using 'tanh' as verified by your test!
    metric = NeuralRanders(manifold, key, hidden_dim=64) 
    vae = GeometricVAE(dataset.X.shape[1], latent_dim, metric, key)
    
    trainer = GeodesicFlowTrainer(vae, learning_rate=1e-3)

    # 3. Phase 1: Manifold Learning (The "Map")
    print("\n=== Phase 1: Learning the Manifold (80 epochs) ===")
    batch_size = 256
    N = dataset.X.shape[0]
    
    for epoch in range(80):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)
        epoch_loss = 0
        
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch_x = dataset.X[idx]
            step_key = jax.random.fold_in(subkey, i)
            trainer.model, trainer.opt_state, loss, _, _ = trainer.train_step_manifold(
                trainer.model, trainer.opt_state, batch_x, step_key
            )
            epoch_loss += loss
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | VAE Loss: {epoch_loss/N*batch_size:.2f}")

    # 4. Phase 2: Geodesic Regression (The "Wind")
    print("\n=== Phase 2: Regressing the Wind (50 epochs) ===")
    pairs = dataset.lineage_pairs
    n_pairs = pairs.shape[0]
    
    # Pre-encode for speed (treating latent space as fixed for this phase)
    print("Pre-encoding lineages...")
    z_all = jax.vmap(lambda x: trainer.model.encode(x, jax.random.PRNGKey(0)))(dataset.X)
    z_all = jax.lax.stop_gradient(z_all)
    
    for epoch in range(50):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_pairs)
        epoch_loss = 0
        
        for i in range(0, n_pairs, batch_size):
            idx = perm[i:i+batch_size]
            p_idx, c_idx = pairs[idx, 0], pairs[idx, 1]
            
            trainer.model, trainer.opt_state, loss = trainer.train_step_geodesic(
                trainer.model, trainer.opt_state, z_all[p_idx], z_all[c_idx]
            )
            epoch_loss += loss
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Geodesic Action Loss: {epoch_loss/n_pairs*batch_size:.4f}")

    # 5. Visualization
    print("Generating Windy Tree...")
    
    # Grid for wind field
    grid_pts, grid_wind = generate_wind_grid(trainer.model, resolution=25)
    
    # Scatter plot
    z_plot = np.array(z_all)
    # Project z_plot to Poincare for plotting if they are on Hyperboloid
    # (Assuming encode returns Hyperboloid coords)
    from ham.vis.hyperbolic import project_to_poincare
    z_disk = np.array(project_to_poincare(z_plot))
    
    c = dataset.labels
    
    fig, ax = plt.subplots(figsize=(10, 10))
    # Cells
    ax.scatter(z_disk[:,0], z_disk[:,1], c=c, s=5, alpha=0.4, cmap='viridis')
    # Wind
    ax.quiver(grid_pts[:,0], grid_pts[:,1], grid_wind[:,0], grid_wind[:,1], 
              color='red', scale=20, width=0.002, alpha=0.8)
    
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.axis('off')
    plt.savefig("weinreb_geodesic_final.png", dpi=200)
    print("Saved weinreb_geodesic_final.png")

if __name__ == "__main__":
    main()