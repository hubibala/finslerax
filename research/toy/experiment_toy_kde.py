import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ham.geometry.manifolds import EuclideanSpace
from ham.models.learned import EnergyBasedRanders
from ham.nn.kde import GaussianKDEEnergy
from ham.solvers.avbd import AVBDSolver


class KDEWeinrebModel(eqx.Module):
    ebm: GaussianKDEEnergy
    metric: EnergyBasedRanders
    solver: AVBDSolver
    manifold: EuclideanSpace

    def __init__(self, centers, sigma=1.0, dim=2):
        self.manifold = EuclideanSpace(dim)
        self.ebm = GaussianKDEEnergy(centers, sigma)
        self.metric = EnergyBasedRanders(
            self.manifold, self.ebm, wind_scale=0.15, beta=10.0
        )
        self.solver = AVBDSolver(
            step_size=0.5, grad_clip=1.0, iterations=200, implicit_diff=True
        )


def generate_synthetic_tree(n_points=500):
    """Generates a Y-shaped branching dataset in 2D."""
    np.random.seed(42)
    # Stem: y from -2 to 0, x approx 0
    y_stem = np.random.uniform(-2, 0, n_points // 3)
    x_stem = np.random.normal(0, 0.1, n_points // 3)

    # Branch 1: y from 0 to 2, x from 0 to 2
    y_b1 = np.random.uniform(0, 2, n_points // 3)
    x_b1 = y_b1 + np.random.normal(0, 0.1, n_points // 3)

    # Branch 2: y from 0 to 2, x from 0 to -2
    y_b2 = np.random.uniform(0, 2, n_points // 3)
    x_b2 = -y_b2 + np.random.normal(0, 0.1, n_points // 3)

    X = np.vstack(
        [
            np.column_stack([x_stem, y_stem]),
            np.column_stack([x_b1, y_b1]),
            np.column_stack([x_b2, y_b2]),
        ]
    )
    return jnp.array(X)


def main():
    print("Generating synthetic branching dataset in 2D...")
    centers = generate_synthetic_tree(n_points=600)

    # For a 2D dataset with noise 0.1, sigma=0.2 is reasonable
    model = KDEWeinrebModel(centers, sigma=0.2, dim=2)

    os.makedirs("results_toy_kde", exist_ok=True)
    print("Generating 2D Geometry Plot for KDE...")

    x_lin = jnp.linspace(-3, 3, 60)
    y_lin = jnp.linspace(-3, 3, 60)
    X_grid, Y_grid = jnp.meshgrid(x_lin, y_lin)

    grid_pts = jnp.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1)

    print("Evaluating KDE Energy Landscape...")
    ebm_fn = jax.vmap(model.ebm)
    energies = ebm_fn(grid_pts)
    energies = energies.reshape(X_grid.shape)

    print("Evaluating KDE Wind Field...")

    def get_wind(x):
        _, w_safe, _ = model.metric.zermelo_data(x)
        return w_safe

    wind_fn = jax.vmap(get_wind)
    wind_2d = wind_fn(grid_pts)

    W_x = wind_2d[:, 0].reshape(X_grid.shape)
    W_y = wind_2d[:, 1].reshape(Y_grid.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(X_grid, Y_grid, energies, levels=50, cmap="viridis")
    plt.colorbar(label="KDE Energy E(x)")

    plt.scatter(
        centers[:, 0], centers[:, 1], s=5, c="white", alpha=0.5, label="Synthetic Data"
    )

    stride = 3
    plt.quiver(
        X_grid[::stride, ::stride],
        Y_grid[::stride, ::stride],
        W_x[::stride, ::stride],
        W_y[::stride, ::stride],
        color="red",
        alpha=0.7,
        scale=15,
        width=0.003,
        label="Wind $-\\nabla E(x)$",
    )

    print("Computing Geodesics for empirical pairs...")

    # Stem to Branch 1
    z_start1 = jnp.array([0.0, -2.0])
    z_end1 = jnp.array([2.0, 2.0])

    # Stem to Branch 2
    z_start2 = jnp.array([0.0, -2.0])
    z_end2 = jnp.array([-2.0, 2.0])

    # Branch 1 to Branch 2 (should route through the origin!)
    z_start3 = jnp.array([2.0, 2.0])
    z_end3 = jnp.array([-2.0, 2.0])

    pairs = [
        (z_start1, z_end1, "Stem -> Branch 1"),
        (z_start2, z_end2, "Stem -> Branch 2"),
        (z_start3, z_end3, "Branch 1 -> Branch 2"),
    ]

    colors = ["cyan", "magenta", "orange"]

    # JIT compile solvers for fair timing
    @jax.jit
    def solve_avbd(z_start, z_end):
        return model.solver.solve(
            model.metric, z_start, z_end, train_mode=False, n_steps=200
        )

    print("Compiling solvers...")
    _ = solve_avbd(z_start1, z_end1)

    for idx, (z_start, z_end, label) in enumerate(pairs):
        t0 = time.time()
        traj_avbd = solve_avbd(z_start, z_end)
        jax.block_until_ready(traj_avbd.xs)
        t_avbd = time.time() - t0

        print(f"\n[{label}]")
        print(f"  AVBD: Energy = {traj_avbd.energy:.4f}, Time = {t_avbd:.4f}s")

        # Plot AVBD
        plt.plot(
            traj_avbd.xs[:, 0],
            traj_avbd.xs[:, 1],
            color=colors[idx],
            linewidth=3,
            linestyle="-",
            label=f"AVBD: {label}",
        )

        plt.scatter(
            z_start[0],
            z_start[1],
            c=colors[idx],
            s=100,
            marker="o",
            edgecolors="black",
            zorder=5,
        )
        plt.scatter(
            z_end[0],
            z_end[1],
            c=colors[idx],
            s=100,
            marker="X",
            edgecolors="black",
            zorder=5,
        )

    plt.title("Synthetic 2D Branching Geometric Evaluation (KDE Sanity Check)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    out_path = "results_toy_kde/synthetic_kde_geometry.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved synthetic KDE geometric evaluation plot to {out_path}")


if __name__ == "__main__":
    main()
