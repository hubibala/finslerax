"""
Experiment script for EBM-Finsler formulation on Weinreb Data.
"""

import os

import equinox as eqx
import jax
import optax

from ham.geometry.manifolds.euclidean_space import EuclideanSpace
from ham.models.learned import EnergyBasedRanders
from ham.nn.ebm import ScalarEnergyField
from ham.solvers.avbd import AVBDSolver
from ham.training.losses_ebm import DenoisingScoreMatchingLoss
from ham.training.pipeline import HAMPipeline, TrainingPhase
from research.weinreb.bio.data import DataLoader
from research.weinreb.metrics import compute_directionality_score


class WeinrebModel(eqx.Module):
    ebm: ScalarEnergyField
    metric: EnergyBasedRanders
    solver: AVBDSolver
    manifold: EuclideanSpace

    def __init__(self, key, dim=50):
        self.manifold = EuclideanSpace(dim)
        # Deep enough to capture the complex landscape
        self.ebm = ScalarEnergyField(dim, 128, 4, key, use_fourier=False)
        self.metric = EnergyBasedRanders(
            self.manifold, self.ebm, wind_scale=0.1, beta=20.0
        )
        self.solver = AVBDSolver(step_size=0.01, grad_clip=1.0, iterations=100)


def main():
    print("Loading preprocessed Weinreb dataset...")
    loader = DataLoader("data/weinreb_preprocessed.h5ad", mode="real")
    dataset = loader.get_jax_data(use_pca=True)

    dim = dataset.X.shape[1]

    key = jax.random.PRNGKey(42)
    model = WeinrebModel(key, dim=dim)

    def filter_spec(m):
        return jax.tree_util.tree_map(lambda leaf: eqx.is_inexact_array(leaf), m)

    phase_ebm = TrainingPhase(
        name="Weinreb_EBM",
        epochs=100,
        optimizer=optax.adamw(1e-3, weight_decay=1e-4),
        losses=[DenoisingScoreMatchingLoss(sigma=0.5)],
        filter_spec=filter_spec,
        requires_pairs=False,
    )

    pipeline = HAMPipeline(model)
    print("Training EBM with Denoising Score Matching...")
    trained_model = pipeline.fit(dataset, [phase_ebm], batch_size=64, seed=42)

    trained_model = eqx.tree_at(
        lambda m: m.metric.ebm, trained_model, trained_model.ebm
    )

    print("Evaluating metrics...")
    pairs = dataset.lineage_pairs
    if pairs is not None and len(pairs) > 0:
        sample_pairs = pairs[:5]
        for p1, p2 in sample_pairs:
            z_start = dataset.X[p1]
            z_end = dataset.X[p2]

            d_score = compute_directionality_score(trained_model, z_start, z_end)
            print(f"Directionality Score (Pair {p1}->{p2}): {d_score:.4f}")

    os.makedirs("results_weinreb_ebm", exist_ok=True)
    eqx.tree_serialise_leaves("results_weinreb_ebm/ebm_model.eqx", trained_model)
    print("Saved trained model to results_weinreb_ebm/ebm_model.eqx")


if __name__ == "__main__":
    main()
