import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from ham.geometry import Sphere
from ham.geometry.zoo import Randers
from ham.sim.fields import rossby_haurwitz
from ham.solvers import AVBDSolver
from ham.solvers.geodesic import ExponentialMap
from ham.vis import setup_3d_plot, plot_sphere, plot_vector_field, plot_trajectory, plot_indicatrices, generate_icosphere

def main():
    print("--- HAM Visual Proof (Interactive) ---")
    print("Rendering... Please wait.")
    
    # 1. Physics Setup
    sphere = Sphere(radius=1.0)
    wind_flow = rossby_haurwitz(R=4, omega=1.0)
    h_net = lambda x: jnp.eye(3)
    w_net = lambda x: 0.8 * wind_flow(x)
    metric = Randers(sphere, h_net, w_net)
    
    # 2. Visualization
    fig, ax = setup_3d_plot()
    plot_sphere(ax, alpha=0.1)
    
    # Plot Wind
    pts, _ = generate_icosphere(radius=1.0, subdivisions=2)
    wind_vecs = np.array(jax.vmap(w_net)(pts))
    plot_vector_field(ax, pts, wind_vecs, scale=0.15, color='cyan', label='Rossby Wind')
    
    # 3. Solver 1: BVP (The "Pilot") - RED
    start = jnp.array([1.0, 0.0, 0.0])
    end   = jnp.array([0, 0, 1.0]) 
    end   = end / jnp.linalg.norm(end)
    
    n_steps_bvp = 30
    print("Solving BVP (Optimal Path)...")
    solver_bvp = AVBDSolver(step_size=0.05, beta=1.0, iterations=10000000)
    traj_bvp = solver_bvp.solve(metric, start, end, n_steps=n_steps_bvp)
    plot_trajectory(ax, traj_bvp, color='red', linewidth=4, label='BVP: Optimal Path')
    
    # 4. Solver 2: IVP (The "Leaf") - ORANGE
    print("Solving IVP (Passive Drift)...")
    solver_ivp = ExponentialMap(step_size=0.01, max_steps=200)
    
    north = jnp.array([0., 0., 1.])
    v_north = sphere.to_tangent(start, north)
    traj_ivp_drift = solver_ivp.trace(metric, start, v_north)
    plot_trajectory(ax, traj_ivp_drift, color='orange', linestyle='--', linewidth=2, label='IVP: Passive Drift')

    # 5. The SANITY CHECK - GREEN
    # Extract velocity from BVP and shoot blindly
    v_optimal_approx = traj_bvp.vs[0] * n_steps_bvp
    
    print("Solving IVP (Verification Shot)...")
    traj_verification = solver_ivp.trace(metric, start, v_optimal_approx)
    
    plot_trajectory(ax, traj_verification, color='#00FF00', linestyle='-', linewidth=2, label='IVP: Verification')

    # 6. Indicatrices
    plot_indicatrices(ax, metric, traj_bvp.xs[::5], color='purple')
    
    ax.legend()
    plt.title("Interactive HAM Proof\nRotate to inspect Green/Red overlap!")
    
    print("Plot opened. Use your mouse to rotate and zoom.")
    plt.show()

if __name__ == "__main__":
    main()