import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple, Union
from jax import config
config.update("jax_enable_x64", True)

from ham.geometry import Sphere
from ham.geometry.zoo import Randers
from ham.sim.fields import rossby_haurwitz
from ham.solvers import AVBDSolver
from ham.solvers.geodesic import ExponentialMap, GeodesicState
from ham.vis import setup_3d_plot, plot_sphere, plot_vector_field, plot_trajectory, plot_indicatrices, generate_icosphere


class Trajectory(NamedTuple):
    """Simple trajectory container — positions only."""
    xs: jnp.ndarray  # shape (n_steps + 1, ambient_dim)


def pure_advection(p0: jnp.ndarray,
                   flow: callable,
                   dt: float = 0.005,
                   n_steps: int = 800) -> Trajectory:
    """
    Pure passive transport: dx/dt = flow(x)
    """
    p0 = jnp.asarray(p0).reshape(-1)  # ensure (3,)

    def step(p, _):
        delta = flow(p)
        delta = jnp.asarray(delta).reshape(p.shape)  # force same shape as p
        p_next = p + dt * delta
        p_next = p_next / jnp.linalg.norm(p_next)  # Project to sphere
        return p_next, p_next

    _, traj = jax.lax.scan(step, p0, jnp.arange(n_steps))
    
    # Both should now be (n_steps, 3)
    xs = jnp.concatenate([p0[None, :], traj], axis=0)
    
    return Trajectory(xs=xs)

def path_length(trajectory: Union[jnp.ndarray, NamedTuple, tuple]) -> float:
    """
    Compute total Euclidean length of a trajectory.
    Handles array, NamedTuple with .xs / .x field, or tuple.
    """
    # Extract positions
    if isinstance(trajectory, (tuple, NamedTuple)):
        if hasattr(trajectory, 'xs'):
            xs = trajectory.xs
        elif hasattr(trajectory, 'x'):
            xs = trajectory.x
        elif len(trajectory) > 0 and hasattr(trajectory[0], 'shape'):
            xs = trajectory[0]  # first element is usually positions
        else:
            raise ValueError(f"Cannot extract position array from trajectory: {type(trajectory)}")
    else:
        xs = trajectory

    # Compute length
    if not hasattr(xs, 'shape') or xs.shape[0] < 2:
        return 0.0

    diffs = jnp.diff(xs, axis=0)
    return float(jnp.sum(jnp.linalg.norm(diffs, axis=-1)))


def main():
    print("--- HAM Rossby-Haurwitz Vortex Demo (2026) ---")
    print("Showing optimal steering vs passive drift vs verification shot\n")

    # ────────────────────────────────────────────────────────────────
    # 1. Physics Setup
    # ────────────────────────────────────────────────────────────────
    sphere = Sphere(radius=1.0)
    wind_flow = rossby_haurwitz(R=4, omega=1.0, K=0.8)  # K!=omega avoids stagnation at equator
    w_net = lambda x: 0.8 * wind_flow(x)          # scaled wind strength
    h_net = lambda x: jnp.eye(3)
    metric = Randers(sphere, h_net, w_net)

    # Start / end points (almost antipodal)
    start = jnp.array([1.0, 0.0, 0.0])
    end   = jnp.array([0.0, 0.0, 1.0])
    end   = end / jnp.linalg.norm(end)

    # ────────────────────────────────────────────────────────────────
    # 2. Solve optimal path (BVP) — RED
    # ────────────────────────────────────────────────────────────────
    print("Solving optimal Randers geodesic (BVP)...")
    solver_bvp = AVBDSolver(step_size=0.05, beta=3.0, iterations=5000, tol=1e-6)
    traj_bvp = solver_bvp.solve(metric, start, end, n_steps=40)

    # ────────────────────────────────────────────────────────────────
    # 3. Pure passive advection (dx/dt = wind) — ORANGE
    # ────────────────────────────────────────────────────────────────
    print("Integrating pure passive drift (dx/dt = wind)...")
    wind_fn = lambda x: 0.8 * wind_flow(x)
    traj_passive = pure_advection(start, wind_fn, dt=0.005, n_steps=1200)

    # ────────────────────────────────────────────────────────────────
    # 4. Verification shot: start with BVP's initial velocity — GREEN
    # ────────────────────────────────────────────────────────────────
    print("Shooting geodesic from BVP's initial direction (verification)...")
    solver_ivp = ExponentialMap(step_size=0.01, max_steps=600)
    v_optimal_approx = traj_bvp.vs[0] * 40.0   # rough scaling to match path length
    traj_verif = solver_ivp.trace(metric, start, v_optimal_approx)

    # ────────────────────────────────────────────────────────────────
    # 5. Quantitative comparison
    # ────────────────────────────────────────────────────────────────
    print("\nResults:")
    print(f"  Optimal BVP path (red)    → Randers energy: {traj_bvp.energy:>8.4f}   | Euclidean length: {path_length(traj_bvp):>6.4f}")
    print(f"  Passive drift (orange)    → Euclidean length: {path_length(traj_passive):>6.4f}")
    print(f"  Verification shot (green) → Euclidean length: {path_length(traj_verif):>6.4f}")

    # Overlap between optimal and verification shot
    xs_bvp = traj_bvp.xs
    xs_verif = traj_verif.x if hasattr(traj_verif, 'x') else traj_verif[0]
    min_len = min(len(xs_bvp), len(xs_verif))
    overlap_dev = jnp.mean(jnp.linalg.norm(xs_bvp[:min_len] - xs_verif[:min_len], axis=1))
    print(f"  Mean deviation (red vs green first {min_len} points): {overlap_dev:.4f}")

    # ────────────────────────────────────────────────────────────────
    # 6. Visualization
    # ────────────────────────────────────────────────────────────────
    fig, ax = setup_3d_plot(elev=15, azim=120)
    plot_sphere(ax, alpha=0.08)

    # Wind field
    pts, _ = generate_icosphere(radius=1.0, subdivisions=2)
    wind_vecs = np.array(jax.vmap(w_net)(pts))
    plot_vector_field(ax, pts, wind_vecs, scale=0.18, color='cyan', alpha=0.35, label='Rossby–Haurwitz wind')

    # Trajectories
    plot_trajectory(ax, traj_bvp,     color='red',    linewidth=4.0, label='Optimal Randers geodesic (BVP)')
    plot_trajectory(ax, traj_passive, color='orange', linewidth=2.5, linestyle='--', label='Pure passive advection (dx/dt = wind)')
    plot_trajectory(ax, traj_verif,   color='#00FF88', linewidth=2.8, linestyle='-', label='Geodesic shot from BVP initial velocity')

    # Indicatrices (optional)
    plot_indicatrices(ax, metric, traj_bvp.xs[::6], color='purple', alpha=0.6, scale=0.12)

    ax.legend(loc='upper right', fontsize=9)
    plt.title("Rossby–Haurwitz Wind on Sphere\nOptimal steering vs passive drift vs verification shot", fontsize=11)
    print("\nPlot ready. Rotate to see how the optimal path exploits the wind.")
    plt.show()


if __name__ == "__main__":
    main()