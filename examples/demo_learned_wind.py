import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx
import optax
from jax import config
config.update("jax_enable_x64", True)

from ham.geometry import Sphere
from ham.geometry.zoo import Randers
from ham.models.learned import NeuralRanders
from ham.sim.fields import rossby_haurwitz
from ham.vis import setup_3d_plot, plot_sphere, plot_vector_field, generate_icosphere

def main():
    print("--- HAM Metric Learning (Smoothed) ---")
    print("Iterating: Adding Jacobian Regularization to fix vector lengths.")

    key = jax.random.PRNGKey(2025)
    
    # 1. Ground Truth
    sphere = Sphere(radius=1.0)
    true_wind_fn = rossby_haurwitz(R=3, omega=1.0)
    
    h_true = lambda x: jnp.eye(3)
    w_true = lambda x: 0.8 * true_wind_fn(x) # True magnitude is 0.8
    metric_true = Randers(sphere, h_true, w_true)

    # 2. Data Generation
    print("Generating trajectories...")
    N_samples = 512
    key, sample_key = jax.random.split(key)
    keys = jax.random.split(sample_key, N_samples)
    
    # Sample points and velocities
    X = jax.vmap(sphere.random_sample, in_axes=(0, None))(keys, ())
    V = jax.vmap(w_true)(X)
    
    # 3. Learner
    print("Initializing Neural Randers...")
    key, k_net = jax.random.split(key)
    # We keep Fourier features for the topology, but tame them with regularization
    learner = NeuralRanders(sphere, k_net, hidden_dim=32, use_fourier=True)
    
    optimizer = optax.adam(learning_rate=2e-3)
    opt_state = optimizer.init(eqx.filter(learner, eqx.is_array))
    
    @eqx.filter_jit
    def train_step(model, x, v, state):
        def loss_fn(m):
            # A. Energy Loss (The Physics)
            energies = jax.vmap(m.energy)(x, v)
            loss_energy = jnp.mean(energies)
            
            # B. Metric Regularization (The Anchor)
            H_vals = jax.vmap(m.h_net)(x)
            I = jnp.eye(3)
            loss_h_reg = jnp.mean((H_vals - I)**2)
            
            # C. Jacobian Regularization (The Smoother) !! NEW !!
            # We calculate dW/dx to penalize rapid changes in magnitude/direction
            def get_w(pt):
                _, W, _ = m._get_zermelo_data(pt)
                return W
            
            # Compute Jacobian matrix for each point x
            jac_fn = jax.jacfwd(get_w)
            jacobians = jax.vmap(jac_fn)(x) # Shape (N, 3, 3)
            loss_smooth = jnp.mean(jacobians**2)
            
            # Weighted Sum: 
            # - Energy is main objective
            # - Smoothness (0.1) kills the ripples
            return loss_energy + 1.0 * loss_h_reg + 0.1 * loss_smooth
            
        grads = eqx.filter_grad(loss_fn)(model)
        updates, new_state = optimizer.update(grads, state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, loss_fn(model)

    print("Training with Smoothness Constraint...")
    for i in range(5001): # Increased steps slightly to allow settling
        learner, opt_state, loss = train_step(learner, X, V, opt_state)
        if i % 300 == 0:
            print(f"  Step {i:04d}: Loss = {loss:.4f}")

    # 5. Visualization
    print("Visualizing...")
    fig, ax = setup_3d_plot()
    plot_sphere(ax, alpha=0.1)
    
    grid_pts, _ = generate_icosphere(radius=1.0, subdivisions=2)
    grid_pts = np.array(grid_pts)
    
    vecs_true = np.array(jax.vmap(w_true)(jnp.array(grid_pts)))
    
    def get_learned_wind(x):
        _, W, _ = learner._get_zermelo_data(x)
        return W
    
    vecs_pred = np.array(jax.vmap(get_learned_wind)(jnp.array(grid_pts)))
    
    plot_vector_field(ax, grid_pts, vecs_true, color='cyan', scale=0.15, alpha=0.3, label='Truth')
    plot_vector_field(ax, grid_pts, vecs_pred, color='magenta', scale=0.15, alpha=0.9, label='Learned (Smoothed)')
    
    ax.legend()
    ax.set_title("Inverse Problem: Smoothed\nMagenta should match Cyan lengths")
    plt.show()

if __name__ == "__main__":
    main()