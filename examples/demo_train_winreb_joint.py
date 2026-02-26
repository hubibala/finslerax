import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx
import os
import optax
import functools

# --- Import Core Modules ---
# Ensure these are importable from your project structure
from ham.bio.data import DataLoader
from ham.bio.vae import GeometricVAE
from ham.geometry.surfaces import Hyperboloid
from ham.models.learned import NeuralRanders
from ham.solvers.avbd import AVBDSolver
import ham.utils.download_weinreb as downloader

# --- Visualization Helper ---
def generate_wind_grid(model, resolution=20, bound=1.0):
    """Generates a grid of wind vectors in the Poincaré disk."""
    # Create grid in Poincaré disk
    x = np.linspace(-bound, bound, resolution)
    y = np.linspace(-bound, bound, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Filter points inside the disk
    mask = X**2 + Y**2 < 0.95
    X, Y = X[mask], Y[mask]
    
    # Convert Poincaré -> Hyperboloid coordinates (Model input)
    # z0 = sqrt(1 + r^2) / (1 - r^2) ... standard projection
    r2 = X**2 + Y**2
    denom = 1 - r2
    z0 = (1 + r2) / denom
    z1 = 2 * X / denom
    z2 = 2 * Y / denom
    
    Z_hyp = np.stack([z0, z1, z2], axis=1) # Shape (N, 3)
    
    # Query model for wind at these points
    @jax.jit
    def get_wind(z):
        # NeuralRanders returns (M, W, V)
        # We want W (Wind Vector in Tangent Space)
        _, W, _ = model.metric._get_zermelo_data(z)
        return W

    # Batch process
    W_hyp = jax.vmap(get_wind)(jnp.array(Z_hyp))
    
    # Project vectors back to Poincaré disk for plotting
    # Simple approximation: The x,y components of the hyperbolic vector 
    # map roughly to the disk tangent space at the origin.
    # For visualization, we strictly scale by the conformal factor.
    scale = 2.0 / (1 + z0) # Conformal factor
    U = W_hyp[:, 1] * scale
    V = W_hyp[:, 2] * scale
    
    return np.stack([X, Y], axis=1), np.stack([U, V], axis=1)


# --- The Joint Trainer ---
class JointTrainer:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        # Slightly lower LR for joint stability
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        # Differentiable solver settings
        self.solver = AVBDSolver(step_size=0.1, iterations=10)

    @eqx.filter_jit
    def train_step_joint(self, model, opt_state, x_p, x_c, key, geo_weight=0.1):
        """
        Jointly trains Geography (VAE) and Physics (Geodesics + Smoothness).
        """
        
        # 1. Split keys for VAE sampling
        # Stack parents and children to compute VAE loss efficiently on both
        x_all = jnp.concatenate([x_p, x_c], axis=0)
        keys = jax.random.split(key, x_all.shape[0])

        def loss_fn(m):
            # --- A. Geography (VAE Loss) ---
            # Reconstruct both parents and children
            # This ensures the latent space captures all cell states
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
            # Minimizing Randers Action aligns W with the trajectory v
            loss_action = jnp.mean(trajectories.energy)
            
            # --- C. Regularization (Stability & Smoothness) ---
            
            # Sample points along the trajectory for regularization
            # Shape: (Batch * Steps, Dim)
            sample_pts = trajectories.xs.reshape(-1, trajectories.xs.shape[-1])

            # Helper to extract fields pure-functionally
            def get_fields(x):
                M, W, _ = m.metric._get_zermelo_data(x)
                return M, W

            # 1. Anchor Regularization (Keep M approx Identity, W small)
            def reg_magnitudes(x):
                M, W = get_fields(x)
                dim = M.shape[-1]
                # Force M approx Identity
                loss_M = jnp.sum((M - jnp.eye(dim))**2)
                # Soft penalty on W magnitude
                loss_W = jnp.sum(W**2)
                return loss_M, loss_W
            
            loss_M_val, loss_W_val = jax.vmap(reg_magnitudes)(sample_pts)
            loss_anchor = jnp.mean(loss_M_val) + 0.001 * jnp.mean(loss_W_val)

            # 2. Jacobian Regularization (The Smoother)
            # Penalize dW/dx to ensure laminar flow
            def get_wind_only(x):
                _, W, _ = m.metric._get_zermelo_data(x)
                return W
            
            # Calculate Jacobian matrix dW/dx
            # jacfwd is efficient for small output dimensions (here: Dim=3)
            jac_fn = jax.jacfwd(get_wind_only)
            jacobians = jax.vmap(jac_fn)(sample_pts) 
            loss_smooth = jnp.mean(jacobians**2)
            
            # --- D. Total Weighted Loss ---
            # Combine terms:
            # - Action: 1.0 (Physics goal)
            # - Anchor: 10.0 (Geometric stability)
            # - Smooth: 0.1 (Laminar flow constraint)
            total_geo_loss = loss_action + 10.0 * loss_anchor + 0.1 * loss_smooth
            
            return loss_vae + geo_weight * total_geo_loss, (loss_vae, loss_action)

        (loss, (l_vae, l_geo)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        
        return new_model, new_opt_state, loss, l_vae, l_geo

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

# --- Main Experiment ---
def main():
    # 1. Load Data
    data_path = "data/weinreb_raw.h5ad"
    if not os.path.exists(data_path):
        print("Downloading raw data...")
        downloader.process_weinreb()
        
    print("Loading Weinreb Clonal Data...")
    # Use 'real' mode to get full dataset
    loader = DataLoader(path=data_path, mode='real')
    loader.preprocess(n_top_genes=100, pca_components=50)
    dataset = loader.get_jax_data(use_pca=True)
    
    pairs = dataset.lineage_pairs
    X = dataset.X
    n_pairs = pairs.shape[0]
    
    if n_pairs == 0:
        raise ValueError("No lineage pairs found in dataset!")
    
    print(f"Dataset: {X.shape[0]} cells, {n_pairs} lineage pairs")
    
    # 2. Setup Model (Latent Space = 2D Hyperboloid)
    key = jax.random.PRNGKey(2025)
    latent_dim = 2
    manifold = Hyperboloid(intrinsic_dim=latent_dim)
    
    # Use Tanh activation for smooth vector fields!
    metric = NeuralRanders(manifold, key, hidden_dim=64, use_fourier=True) 
    vae = GeometricVAE(X.shape[1], latent_dim, metric, key)
    
    trainer = JointTrainer(vae, learning_rate=1e-3)
    
    # 3. Joint Training Loop
    epochs = 100
    batch_size = 128
    
    print(f"\nStarting Joint Training ({epochs} epochs)...")
    print("Strategy: Ramp up Physics weight from 0.0 -> 0.5 over first 20 epochs")

    loss_history = []
    
    for epoch in range(epochs):
        # Dynamic Weight Schedule (Warm-up)
        # Epoch 0-20: Ramp from 0 to 0.5
        # Epoch 20+: Constant 0.5
        geo_weight = min(0.5, (epoch / 20.0) * 0.5)
        
        # Shuffle Data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_pairs)
        
        epoch_vae = 0
        epoch_geo = 0
        
        # Iterate over batches of pairs
        for i in range(0, n_pairs, batch_size):
            idx = perm[i:i+batch_size]
            batch_pairs = pairs[idx]
            
            # Get gene expression for parents and children
            x_p = X[batch_pairs[:, 0]]
            x_c = X[batch_pairs[:, 1]]
            
            step_key = jax.random.fold_in(subkey, i)
            
            # Train Step
            trainer.model, trainer.opt_state, _, l_vae, l_geo = trainer.train_step_joint(
                trainer.model, trainer.opt_state, x_p, x_c, step_key, geo_weight
            )
            
            epoch_vae += l_vae
            epoch_geo += l_geo
            
        # Logging
        n_batches = n_pairs / batch_size
        avg_vae = epoch_vae / n_batches
        avg_geo = epoch_geo / n_batches
        loss_history.append((avg_vae, avg_geo))
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | VAE Loss: {avg_vae:.4f} | Physics Action: {avg_geo:.4f} | Weight: {geo_weight:.2f}")

    # 4. Visualization & Save
    print("\nGenerating Wind Field Visualization...")
    
    # Encode all data for plotting (latent coordinates)
    # We do this in batches to avoid OOM if dataset is huge
    z_mean_list = []
    for i in range(0, X.shape[0], 1000):
        batch_x = X[i:i+1000]
        batch_z = jax.vmap(lambda x: trainer.model.encode(x, jax.random.PRNGKey(0)))(batch_x)
        z_mean_list.append(batch_z)
    z_mean = jnp.concatenate(z_mean_list, axis=0)
    
    # Generate Wind Grid
    grid_pts, grid_wind = generate_wind_grid(trainer.model, resolution=25)
    
    # Project to Poincaré Disk for 2D viewing
    from ham.vis.hyperbolic import project_to_poincare
    z_disk = np.array(project_to_poincare(z_mean))
    c = dataset.labels
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Jointly Learned Randers Metric (Epoch {epochs})")
    
    

    # Scatter Cells
    scatter = ax.scatter(z_disk[:,0], z_disk[:,1], c=c, s=15, alpha=0.6, cmap='viridis', edgecolors='none')
    
    # Quiver Wind
    ax.quiver(grid_pts[:,0], grid_pts[:,1], grid_wind[:,0], grid_wind[:,1], 
              color='red', scale=20, width=0.003, alpha=0.7, label='Learned Flow')
    
    # Style
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--')
    ax.add_artist(circle)
    ax.axis('off')
    
    # Save
    out_file = "weinreb_joint_training.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()