import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ham.geometry.manifolds import EuclideanSpace
from ham.models.learned import EnergyBasedRanders
from ham.nn.kde import GaussianKDEEnergy
from ham.solvers.avbd import AVBDSolver
from research.weinreb.bio.data import DataLoader


class KDEWeinrebModel(eqx.Module):
    ebm: GaussianKDEEnergy
    metric: EnergyBasedRanders
    solver: AVBDSolver
    manifold: EuclideanSpace

    def __init__(self, centers, sigma=1.0, dim=5):
        self.manifold = EuclideanSpace(dim)
        self.ebm = GaussianKDEEnergy(centers, sigma)
        self.metric = EnergyBasedRanders(
            self.manifold, self.ebm, wind_scale=0.1, beta=10.0
        )
        self.solver = AVBDSolver(
            step_size=0.05, grad_clip=1.0, iterations=200, implicit_diff=True
        )


def main():
    print("Loading preprocessed Weinreb dataset...")
    loader = DataLoader("data/weinreb_preprocessed.h5ad", mode="real")
    dataset = loader.get_jax_data(use_pca=True)

    dim = dataset.X.shape[1]

    # Subsample 2000 points for KDE centers
    np.random.seed(42)
    N_centers = min(2000, dataset.X.shape[0])
    idx = np.random.choice(dataset.X.shape[0], N_centers, replace=False)
    centers = dataset.X[idx]

    # Heuristic bandwidth: since PCA coordinates have unit variance, sigma=2.0 is way too large
    # (it blurs the whole dataset into a single blob). We use a much smaller sigma.
    model = KDEWeinrebModel(centers, sigma=0.05, dim=dim)

    os.makedirs("results_weinreb_ebm", exist_ok=True)
    print(f"Generating {dim}D PCA Geometry Plot for KDE...")

    # 1. Create a 2D grid for PC1 and PC2
    pc1_min, pc1_max = jnp.min(dataset.X[:, 0]), jnp.max(dataset.X[:, 0])
    pc2_min, pc2_max = jnp.min(dataset.X[:, 1]), jnp.max(dataset.X[:, 1])

    pad_pc1 = (pc1_max - pc1_min) * 0.1
    pad_pc2 = (pc2_max - pc2_min) * 0.1

    x_lin = jnp.linspace(pc1_min - pad_pc1, pc1_max + pad_pc1, 50)
    y_lin = jnp.linspace(pc2_min - pad_pc2, pc2_max + pad_pc2, 50)
    X_grid, Y_grid = jnp.meshgrid(x_lin, y_lin)

    means = jnp.mean(dataset.X, axis=0)

    grid_pts_2d = jnp.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1)
    grid_pts_nd = jnp.tile(means, (grid_pts_2d.shape[0], 1))
    grid_pts_nd = grid_pts_nd.at[:, 0:2].set(grid_pts_2d)

    print("Evaluating KDE Energy Landscape...")
    ebm_fn = jax.vmap(model.ebm)
    energies = ebm_fn(grid_pts_nd)
    energies = energies.reshape(X_grid.shape)

    print("Evaluating KDE Wind Field...")

    def get_wind(x):
        _, w_safe, _ = model.metric.zermelo_data(x)
        return w_safe

    wind_fn = jax.vmap(get_wind)
    wind_nd = wind_fn(grid_pts_nd)

    W_x = wind_nd[:, 0].reshape(X_grid.shape)
    W_y = wind_nd[:, 1].reshape(Y_grid.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(X_grid, Y_grid, energies, levels=50, cmap="viridis")
    plt.colorbar(label="KDE Energy E(x)")

    subsample = 5
    plt.scatter(
        dataset.X[::subsample, 0],
        dataset.X[::subsample, 1],
        s=2,
        c="white",
        alpha=0.3,
        label="Weinreb Data (PC1/PC2)",
    )

    stride = 3
    plt.quiver(
        X_grid[::stride, ::stride],
        Y_grid[::stride, ::stride],
        W_x[::stride, ::stride],
        W_y[::stride, ::stride],
        color="red",
        alpha=0.7,
        scale=10,
        width=0.003,
        label="Wind $-\\nabla E(x)$",
    )

    print(f"Computing {dim}D Geodesics for empirical pairs...")
    pairs = dataset.lineage_pairs
    if pairs is not None and len(pairs) > 0:
        sample_pairs = pairs[:5]
        for idx, (p1, p2) in enumerate(sample_pairs):
            z_start = dataset.X[p1]
            z_end = dataset.X[p2]

            traj = model.solver.solve(model.metric, z_start, z_end, n_steps=100)

            path_pc1 = traj.xs[:, 0]
            path_pc2 = traj.xs[:, 1]

            if idx == 0:
                plt.plot(
                    path_pc1,
                    path_pc2,
                    "m-",
                    linewidth=2,
                    label=f"Computed {dim}D Geodesic",
                )
            else:
                plt.plot(path_pc1, path_pc2, "m-", linewidth=2)

            plt.scatter(
                z_start[0],
                z_start[1],
                c="cyan",
                s=50,
                marker="o",
                edgecolors="black",
                zorder=5,
            )
            plt.scatter(
                z_end[0],
                z_end[1],
                c="magenta",
                s=50,
                marker="X",
                edgecolors="black",
                zorder=5,
            )

    plt.title(f"Weinreb {dim}D Geometric Evaluation (KDE Sanity Check)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()

    out_path = "results_weinreb_ebm/weinreb_kde_geometry_pca.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved KDE geometric evaluation plot to {out_path}")


if __name__ == "__main__":
    main()
