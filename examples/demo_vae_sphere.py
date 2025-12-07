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
from ham.geometry import Sphere
from ham.geometry.zoo import Euclidean
from ham.bio.vae import GeometricVAE
from ham.vis import setup_3d_plot, plot_sphere

# --- 1. Define the Latent Topology (The "Globe") ---
# INSTEAD OF PLANE, WE USE SPHERE
latent_manifold = Sphere(radius=1.0)
latent_metric = Euclidean(latent_manifold)

# --- 2. Generate Data (Swiss Roll) ---
def get_swiss_roll_data(n_samples=2000):
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.1)
    X = X / 10.0 # Scale to roughly [-1, 1]
    return jnp.array(X), t

# --- 3. The Main Demo ---
def main():
    print("--- HAM Spherical Latent Space Demo ---")
    
    # Setup Data
    X, t_labels = get_swiss_roll_data(n_samples=2000)
    data_dim = 3
    
    # Setup Model
    key = jax.random.PRNGKey(2025)
    
    # CRITICAL CHANGE: latent_dim=3 because S^2 lives in R^3
    # The VAE will learn to use the surface, effectively finding 2 dimensions.
    latent_dim = 3
    
    vae = GeometricVAE(
        data_dim=data_dim,
        latent_dim=latent_dim,
        metric=latent_metric,
        key=key
    )
    
    # Optimization Loop
    print("Training Spherical VAE...")
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(vae, eqx.is_array))
    
    @eqx.filter_jit
    def train_step(model, x_batch, state, k):
        def loss_fn(m):
            losses, (recon, action) = jax.vmap(m.loss_fn, in_axes=(0, None))(x_batch, k)
            return jnp.mean(losses), (jnp.mean(recon), jnp.mean(action))
            
        (loss, (recon, action)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_state = optimizer.update(grads, state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, loss, recon, action

    # Train
    epochs = 10000
    for i in range(epochs):
        step_key = jax.random.fold_in(key, i)
        vae, opt_state, loss, recon, action = train_step(vae, X, opt_state, step_key)
        
        if i % 500 == 0:
            print(f"Epoch {i:04d} | Loss: {loss:.4f} (Recon: {recon:.4f})")

    print("Training Complete.")

    # --- 4. Visualization ---
    print("Visualizing Spherical Latent Space...")
    
    # Encode
    Z, _, _ = jax.vmap(vae.encode, in_axes=(0, None))(X, key)
    X_rec = jax.vmap(vae.decode)(Z)
    
    # Convert to numpy
    X = np.array(X)
    Z = np.array(Z)
    X_rec = np.array(X_rec)
    
    fig = plt.figure(figsize=(15, 6))
    
    # Plot 1: Original
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:,0], X[:,1], X[:,2], c=t_labels, cmap='Spectral', s=5)
    ax1.set_title("Original Swiss Roll")
    
    # Plot 2: Reconstructed
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(X_rec[:,0], X_rec[:,1], X_rec[:,2], c=t_labels, cmap='Spectral', s=5)
    ax2.set_title("Reconstructed from Sphere")
    
    # Plot 3: The Latent Sphere
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Draw wireframe sphere for context
    plot_sphere(ax3, radius=1.0, alpha=0.1)
    
    # Plot the encoded points
    # They should stick to the surface!
    ax3.scatter(Z[:,0], Z[:,1], Z[:,2], c=t_labels, cmap='Spectral', s=5)
    ax3.set_title("Learned Latent Space (S^2)")
    
    # Remove panes for cleaner look
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()