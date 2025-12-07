import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from ham.geometry import Sphere
from ham.geometry.zoo import Randers, Euclidean
from ham.solvers import AVBDSolver
from ham.vis import setup_3d_plot, plot_sphere, plot_vector_field, plot_trajectory, plot_indicatrices, generate_icosphere

def vortex_field(center, strength=1.0, decay=2.0):
    center = center / jnp.linalg.norm(center)
    def flow(x):
        cos_dist = jnp.dot(x, center)
        dist = jnp.arccos(jnp.clip(cos_dist, -1.0, 1.0))
        v_rot = jnp.cross(center, x)
        magnitude = strength * jnp.exp(-decay * (dist**2))
        return magnitude * v_rot
    return flow

def main():
    print("--- HAM Stiffness Fix Demo ---")
    
    # 1. Setup Strong Vortex
    sphere = Sphere(radius=1.0)
    vortex_center = jnp.array([0.0, 1.0, 0.0])
    w_net = vortex_field(vortex_center, strength=1.5, decay=5.0) # Very strong wind to force curvature
    h_net = lambda x: jnp.eye(3)
    metric = Randers(sphere, h_net, w_net)

    # 2. Mission: Cross the Vortex
    start = jnp.array([1.0, 0.0, 0.0])
    end   = jnp.array([-0.99, 0.1, 0.0]) 
    end   = end / jnp.linalg.norm(end)

    # 3. Solver Comparison: Stiff vs. Relaxed
    print("Solving 'Stiff' (High Beta)...")
    solver_stiff = AVBDSolver(step_size=0.05, beta=20.0, iterations=100) # Quick freeze
    traj_stiff = solver_stiff.solve(metric, start, end, n_steps=40)

    print("Solving 'Relaxed' (Low Beta)...")
    # beta=0.5: Very slow hardening. Allows massive lateral movement.
    solver_relaxed = AVBDSolver(step_size=0.05, beta=20, iterations=10000) 
    traj_relaxed = solver_relaxed.solve(metric, start, end, n_steps=40)

    # 4. Visualization
    fig, ax = setup_3d_plot()
    plot_sphere(ax, alpha=0.1)
    
    # Wind
    pts, _ = generate_icosphere(radius=1.0, subdivisions=2)
    wind_vecs = np.array(jax.vmap(w_net)(pts))
    plot_vector_field(ax, pts, wind_vecs, scale=0.2, color='cyan', alpha=0.3)
    
    # Paths
    plot_trajectory(ax, traj_stiff, color='gray', linestyle='--', linewidth=2, label='Stiff Solver (beta=20)')
    plot_trajectory(ax, traj_relaxed, color='red', linewidth=4, label='Relaxed Solver (beta=0.5)')
    
    ax.legend()
    plt.title("Solver Stiffness Comparison\nNotice the Red line finding the 'D' shape!")
    print("Showing plot. The Red line should now curve significantly more than the Gray line.")
    plt.show()

if __name__ == "__main__":
    main()