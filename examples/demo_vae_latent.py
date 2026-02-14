import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import equinox as eqx
from sklearn.datasets import make_swiss_roll

# JAX Config
from jax import config
config.update("jax_enable_x64", True)

# HAM Imports
from ham.geometry.manifold import Manifold
from ham.geometry.zoo import Euclidean
from ham.bio.vae import GeometricVAE

# --- 1. Define the Latent Topology (The "Paper" we draw the map on) ---
class LatentPlane(Manifold):
    """A simple 2D Euclidean Plane: R^2."""
    @property
    def ambient_dim(self): return 2
    @property
    def intrinsic_dim(self): return 2
    
    def project(self, x): 
        # No constraints, the manifold is the entire space
        return x
        
    def to_tangent(self, x, v): 
        # Tangent space of R^2 is R^2
        return v
    
    def retract(self, x, delta): 
        # Retraction for R^2 is just addition
        return x + delta
        
    def random_sample(self, key, shape):
        return jax.random.normal(key, shape + (2,))

# --- 2. Generate Data (The "Territory") ---
def get_swiss_roll_data(n_samples=2000):
    # Generate 3D Swiss Roll
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.1)
    
    # Scale to be roughly within [-1, 1] for neural stability
    X = X / 10.0
    
    # Convert to JAX
    X = jnp.array(X)
    return X, t

# --- 3. The Main Demo ---
def main():
    print("--- HAM Latent Space Demo: The Swiss Roll ---")
    
    # Setup Data
    print("Generating Swiss Roll Dataset...")
    X, t_labels = get_swiss_roll_data(n_samples=2000)
    data_dim = 3
    
    # Setup Geometry (The Prior)
    # We want to unroll the 3D data onto a 2D Flat Plane
    latent_manifold = LatentPlane()
    latent_metric = Euclidean(latent_manifold)
    
    # Setup Model
    key = jax.random.PRNGKey(42)
    latent_dim = 2
    
    vae = GeometricVAE(
        data_dim=data_dim,
        latent_dim=latent_dim,
        metric=latent_metric,
        key=key
    )
    
    # Optimization Loop
    print("Training Geometric VAE...")
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(vae, eqx.is_array))
    
    @eqx.filter_jit
    def train_step(model, x_batch, state, k):
        def loss_fn(m):
            # VMAP over batch
            losses, (recon, action) = jax.vmap(m.loss_fn, in_axes=(0, None))(x_batch, k)
            return jnp.mean(losses), (jnp.mean(recon), jnp.mean(action))
            
        (loss, (recon, action)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_state = optimizer.update(grads, state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, loss, recon, action

    # Training Loop
    epochs = 20000
    batch_size = 256
    
    # Simple full-batch for demo clarity (or large chunks)
    for i in range(epochs):
        step_key = jax.random.fold_in(key, i)
        vae, opt_state, loss, recon, action = train_step(vae, X, opt_state, step_key)
        
        if i % 200 == 0:
            print(f"Epoch {i:04d} | Loss: {loss:.4f} (Recon: {recon:.4f}, Action: {action:.4f})")

    print("Training Complete.")

    # --- 4. Visualization ---
    print("Visualizing results...")
    
    # Encode all data to get latent representations
    # We ignore the velocity (v) output for this static visualization
    Z, _, _ = jax.vmap(vae.encode, in_axes=(0, None))(X, key)
    
    # Reconstruct
    X_rec = jax.vmap(vae.decode)(Z)
    
    # Convert to numpy for plotting
    X = np.array(X)
    Z = np.array(Z)
    X_rec = np.array(X_rec)
    
    fig = plt.figure(figsize=(15, 6))
    
    # Plot 1: Original 3D Swiss Roll
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:,0], X[:,1], X[:,2], c=t_labels, cmap='Spectral', s=5)
    ax1.set_title("Original 3D Data (Ambient)")
    ax1.view_init(elev=10, azim=80)
    
    # Plot 2: Reconstructed 3D Roll
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(X_rec[:,0], X_rec[:,1], X_rec[:,2], c=t_labels, cmap='Spectral', s=5)
    ax2.set_title("Reconstructed Data")
    ax2.view_init(elev=10, azim=80)
    
    # Plot 3: The Learned 2D Latent Space
    ax3 = fig.add_subplot(133)
    sc = ax3.scatter(Z[:,0], Z[:,1], c=t_labels, cmap='Spectral', s=5)
    ax3.set_title("Learned 2D Manifold (Latent)")
    ax3.set_xlabel("z1")
    ax3.set_ylabel("z2")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()