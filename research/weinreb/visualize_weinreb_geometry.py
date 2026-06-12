import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from ham.geometry.manifolds import EuclideanSpace
from ham.models.learned import EnergyBasedRanders
from ham.nn.ebm import ScalarEnergyField
from ham.solvers.avbd import AVBDSolver
from ham.training.losses_ebm import DenoisingScoreMatchingLoss
from ham.training.pipeline import HAMPipeline, TrainingPhase
from research.weinreb.bio.data import DataLoader


class WeinrebModel(eqx.Module):
    ebm: ScalarEnergyField
    metric: EnergyBasedRanders
    solver: AVBDSolver
    manifold: EuclideanSpace

    def __init__(self, key, dim=50):
        self.manifold = EuclideanSpace(dim)
        self.ebm = ScalarEnergyField(dim, 128, 4, key, use_fourier=False)
        self.metric = EnergyBasedRanders(
            self.manifold, self.ebm, wind_scale=1.0, beta=100.0
        )
        self.solver = AVBDSolver(
            step_size=0.5, grad_clip=1.0, iterations=100, implicit_diff=True
        )


def main():
    print("Loading preprocessed Weinreb dataset...")
    loader = DataLoader("data/weinreb_preprocessed.h5ad", mode="real")
    dataset = loader.get_jax_data(use_pca=True)

    dim = dataset.X.shape[1]
    key = jax.random.PRNGKey(42)
    model = WeinrebModel(key, dim=dim)

    model_path = "results_weinreb_ebm/ebm_model.eqx"

    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        trained_model = eqx.tree_deserialise_leaves(model_path, model)
    else:
        print("Trained model not found. Training EBM for 20 epochs...")

        def filter_spec(m):
            return jax.tree_util.tree_map(lambda leaf: eqx.is_inexact_array(leaf), m)

        phase_ebm = TrainingPhase(
            name="Weinreb_EBM",
            epochs=60,
            optimizer=optax.adamw(1e-3, weight_decay=1e-4),
            losses=[DenoisingScoreMatchingLoss(sigma=0.05)],
            filter_spec=filter_spec,
            requires_pairs=False,
        )

        pipeline = HAMPipeline(model)
        trained_model = pipeline.fit(dataset, [phase_ebm], batch_size=256, seed=42)
        trained_model = eqx.tree_at(
            lambda m: m.metric.ebm, trained_model, trained_model.ebm
        )

        os.makedirs("results_weinreb_ebm", exist_ok=True)
        eqx.tree_serialise_leaves(model_path, trained_model)
        print(f"Saved trained model to {model_path}")

    print("Generating 2D PCA Geometry Plot...")

    # 1. Create a 2D grid for PC1 and PC2
    pc1_min, pc1_max = jnp.min(dataset.X[:, 0]), jnp.max(dataset.X[:, 0])
    pc2_min, pc2_max = jnp.min(dataset.X[:, 1]), jnp.max(dataset.X[:, 1])

    # Pad limits a bit
    pad_pc1 = (pc1_max - pc1_min) * 0.1
    pad_pc2 = (pc2_max - pc2_min) * 0.1

    x_lin = jnp.linspace(pc1_min - pad_pc1, pc1_max + pad_pc1, 50)
    y_lin = jnp.linspace(pc2_min - pad_pc2, pc2_max + pad_pc2, 50)
    X_grid, Y_grid = jnp.meshgrid(x_lin, y_lin)

    # Fill remaining dimensions with the mean of the dataset
    means = jnp.mean(dataset.X, axis=0)

    grid_pts_2d = jnp.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1)
    grid_pts_50d = jnp.tile(means, (grid_pts_2d.shape[0], 1))
    grid_pts_50d = grid_pts_50d.at[:, 0:2].set(grid_pts_2d)

    # 2. Evaluate Energy
    print("Evaluating Energy Landscape...")
    ebm_fn = jax.vmap(trained_model.ebm)
    energies = ebm_fn(grid_pts_50d)
    energies = energies.reshape(X_grid.shape)

    # 3. Evaluate Wind Field
    print("Evaluating Wind Field...")

    def get_wind(x):
        _, w_safe, _ = trained_model.metric.zermelo_data(x)
        return w_safe

    wind_fn = jax.vmap(get_wind)
    wind_50d = wind_fn(grid_pts_50d)

    # We only care about the PC1 and PC2 components of the wind for visualization
    W_x = wind_50d[:, 0].reshape(X_grid.shape)
    W_y = wind_50d[:, 1].reshape(Y_grid.shape)

    # 4. Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X_grid, Y_grid, energies, levels=50, cmap="viridis")
    plt.colorbar(label="Energy E(x)")

    # Subsample scatter to prevent overcrowding
    subsample = 5
    plt.scatter(
        dataset.X[::subsample, 0],
        dataset.X[::subsample, 1],
        s=2,
        c="white",
        alpha=0.3,
        label="Weinreb Data (PC1/PC2)",
    )

    # Quiver plot for wind
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

    # 5. Geodesic Routing
    print("Computing 50D Geodesics for empirical pairs...")
    pairs = dataset.lineage_pairs
    if pairs is not None and len(pairs) > 0:
        sample_pairs = pairs[:10]
        for idx, (p1, p2) in enumerate(sample_pairs):
            z_start = dataset.X[p1]
            z_end = dataset.X[p2]

            # Solve the BVP in 50D
            traj = trained_model.solver.solve(
                trained_model.metric, z_start, z_end, n_steps=100
            )

            # Extract PC1 and PC2 of the trajectory
            path_pc1 = traj.xs[:, 0]
            path_pc2 = traj.xs[:, 1]

            # Plot
            if idx == 0:
                plt.plot(
                    path_pc1, path_pc2, "m-", linewidth=2, label="Computed 50D Geodesic"
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

    plt.title("Weinreb 50D Geometric Evaluation (Projected to PC1-PC2)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()

    out_path = "results_weinreb_ebm/weinreb_geometry_pca.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved geometric evaluation plot to {out_path}")


if __name__ == "__main__":
    main()
