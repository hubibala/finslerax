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
        x_all = jnp.concatenate([x_p, x_c], axis=0)
        keys = jax.random.split(key, x_all.shape[0])

        # ONLY TAKES 'm' (The model parameters being updated)
        def loss_fn(m):
            # --- A. Geography (VAE Loss) ---
            vae_losses, vae_stats = jax.vmap(m.loss_fn)(x_all, x_all, keys)
            loss_vae = jnp.mean(vae_losses)
            
            # --- B. Physics (Geodesic Action) ---
            dist_p = jax.vmap(m._get_dist)(x_p)
            dist_c = jax.vmap(m._get_dist)(x_c)
            z_p = dist_p.mean
            z_c = dist_c.mean
            
            # 1. Stop Gradients into the Latent Space for Solver Stability
            z_p_sg = jax.lax.stop_gradient(z_p)
            z_c_sg = jax.lax.stop_gradient(z_c)
            
            # Pack the physics computation into a function for the true branch
            # Notice: No more `m_mod`, we just use `m` directly from the outer scope!
            def compute_physics(operands):
                zp, zc = operands
                
                # 2. Solve Inverse Dynamics using `m` directly
                solve_fn = jax.vmap(lambda s, e: self.solver.solve(m.metric, s, e, n_steps=6, train_mode=True))
                optimal_traj = jax.lax.stop_gradient(solve_fn(zp, zc))
                
                # 3. Action Minimization
                def compute_step_energy(x, v):
                    return m.metric.metric_fn(x, v)**2 
                    
                xs_segments = optimal_traj.xs[:, :-1, :]
                batch_energies = jax.vmap(jax.vmap(compute_step_energy))(xs_segments, optimal_traj.vs)
                loss_act = jnp.mean(jnp.sum(batch_energies, axis=-1))
                
                # C. Regularization
                sample_pts = optimal_traj.xs.reshape(-1, optimal_traj.xs.shape[-1])
                
                def reg_magnitudes(x):
                    M, W, _ = m.metric._get_zermelo_data(x)
                    dim = M.shape[-1]
                    return jnp.sum((M - jnp.eye(dim))**2), jnp.sum(W**2)
                
                loss_M_val, loss_W_val = jax.vmap(reg_magnitudes)(sample_pts)
                loss_anchor = jnp.mean(loss_M_val) + 0.001 * jnp.mean(loss_W_val)

                def get_wind_only(x):
                    _, W, _ = m.metric._get_zermelo_data(x)
                    return W
                
                jacobians = jax.vmap(jax.jacfwd(get_wind_only))(sample_pts) 
                loss_smooth = jnp.mean(jacobians**2)
                
                total_geo = loss_act + 10.0 * loss_anchor + 0.1 * loss_smooth
                return total_geo, loss_act

            # Pack the false branch (Skip physics entirely)
            def skip_physics(operands):
                return 0.0, 0.0

            # ONLY execute the solver if the weight is strictly greater than 0
            # We only pass the standard JAX arrays as operands!
            total_geo_loss, loss_action = jax.lax.cond(
                geo_weight > 0.0,
                compute_physics,
                skip_physics,
                (z_p_sg, z_c_sg)
            )
            
            # --- D. Total Weighted Loss ---
            return loss_vae + geo_weight * total_geo_loss, (loss_vae, loss_action)

        # eqx.filter_value_and_grad perfectly wraps loss_fn(m) now
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
    
   # --- The Fix: Mini-batching and Hard Warmup ---
    x_p = X[pairs[:, 0]]
    x_c = X[pairs[:, 1]]
    step_key = key
    epochs = 100
    batch_size = 1024 # Fits perfectly in T4 GPU VRAM
    num_batches = len(x_p) // batch_size

    print(f"Starting Joint Training ({epochs} epochs)...")
    print(f"Batches per epoch: {num_batches}")
    
    for epoch in range(epochs):
        # 1. HARD Warmup Schedule: Strictly 0.0 for the first 20 epochs!
        if epoch < 20:
            geo_weight = 0.0
        else:
            # Ramps from 0.0 -> 0.5 between epochs 20 and 40
            geo_weight = min(0.5, ((epoch - 20) / 20.0) * 0.5)
            
        epoch_l_vae = 0.0
        epoch_l_geo = 0.0
        
        # Shuffle the data at the start of each epoch
        step_key, subkey = jax.random.split(step_key)
        perms = jax.random.permutation(subkey, len(x_p))
        x_p_shuffled = x_p[perms]
        x_c_shuffled = x_c[perms]

        # 2. Mini-batch Loop
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size
            
            x_p_batch = x_p_shuffled[start_idx:end_idx]
            x_c_batch = x_c_shuffled[start_idx:end_idx]
            
            trainer.model, trainer.opt_state, loss, l_vae, l_geo = trainer.train_step_joint(
                trainer.model, trainer.opt_state, x_p_batch, x_c_batch, step_key, geo_weight
            )
            
            epoch_l_vae += l_vae
            epoch_l_geo += l_geo
            
        # Average the loss over the batches
        avg_vae = epoch_l_vae / num_batches
        avg_geo = epoch_l_geo / num_batches

        if epoch % 5 == 0 or epoch == 20:
            print(f"Epoch {epoch:03d} | VAE: {avg_vae:.4f} | Physics: {avg_geo:.4f} | Weight: {geo_weight:.2f}")

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