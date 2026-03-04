import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time

from ham.models.learned import NeuralRanders
from ham.bio.vae import GeometricVAE
from ham.bio.data import DataLoader
from ham.training.losses import (
    ReconstructionLoss,
    KLDivergenceLoss,
    ZermeloAlignmentLoss,
    GeodesicSprayLoss,
    ContrastiveAlignmentLoss,
    MetricAnchorLoss,
    MetricSmoothnessLoss
)
from ham.training.pipeline import TrainingPhase, HAMPipeline


def get_filter_fn(selector):
    """
    Returns a filter function for eqx.partition.
    The selector should be a lambda extracting the components to train, e.g.,
    lambda m: (m.encoder_net, m.decoder_net)
    """
    def filter_spec(model):
        base_mask = jax.tree_util.tree_map(lambda _: False, model)
        targets = selector(model)
        
        def make_true(n): 
            return jax.tree_util.tree_map(lambda leaf: True if eqx.is_array(leaf) else False, n)
            
        if isinstance(targets, tuple):
            true_mask = tuple(make_true(t) for t in targets)
        else:
            true_mask = make_true(targets)
            
        return eqx.tree_at(selector, base_mask, replace=true_mask)
    return filter_spec


def main():
    print("Initializing HAM Pipeline Configuration...")
    seed = 2025
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    # 1. Load Data
    print("Loading data...")
    dataset = DataLoader(mode='simulation').get_jax_data(use_pca=False)
    data_dim = dataset.X.shape[1]
    latent_dim = 2

    # 2. Initialize Model Components
    print("Initializing geometry and model...")
    # Just a basic NeuralRanders metric
    # The actual hyper-parameters for the metric depend on the manifold, which is handled inside VAE.
    # We'll instantiate a placeholder manifold to init the metric, or just init it.
    from ham.geometry.surfaces import Hyperboloid
    base_manifold = Hyperboloid(intrinsic_dim=latent_dim)
    metric = NeuralRanders(base_manifold, hidden_dim=64, key=subkey)
    
    key, subkey = jax.random.split(key)
    model = GeometricVAE(data_dim=data_dim, latent_dim=latent_dim, metric=metric, key=subkey)

    # 3. Define Training Phases
    phases = []
    
    # Phase 1: Manifold Pretraining (Train Encoder/Decoder, Freeze Metric)
    phase_1 = TrainingPhase(
        name="Manifold Pretraining",
        epochs=100,
        optimizer=optax.adam(1e-3),
        losses=[
            ReconstructionLoss(weight=1.0),
            KLDivergenceLoss(weight=1e-4),
            ZermeloAlignmentLoss(weight=0.1),
            GeodesicSprayLoss(weight=1e-3)
        ],
        filter_spec=get_filter_fn(lambda m: (m.encoder_net, m.decoder_net)),
        requires_pairs=False
    )
    phases.append(phase_1)

    # Phase 2: Metric Learning (Train Metric, Freeze Encoder/Decoder)
    # Only added if lineage pairs are available
    if dataset.lineage_pairs is not None:
        phase_2 = TrainingPhase(
            name="Metric Learning",
            epochs=50,
            optimizer=optax.adam(1e-3),
            losses=[
                ContrastiveAlignmentLoss(weight=1.0),
                MetricAnchorLoss(weight=1.0),
                MetricSmoothnessLoss(weight=0.1)
            ],
            filter_spec=get_filter_fn(lambda m: m.metric),
            requires_pairs=True
        )
        phases.append(phase_2)

    # 4. Execute Pipeline
    print(f"Executing pipeline with {len(phases)} phases...")
    pipeline = HAMPipeline(model)
    time_start = time.time()
    
    trained_model = pipeline.fit(dataset, phases, batch_size=256, seed=seed)
    
    print(f"Training completed in {time.time() - time_start:.2f} seconds.")

if __name__ == "__main__":
    main()
