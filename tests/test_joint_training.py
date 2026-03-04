import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jax import config

config.update("jax_enable_x64", True)

from ham.bio.data import BioDataset
from ham.bio.vae import GeometricVAE
from ham.geometry.zoo import Riemannian
from ham.geometry.surfaces import Hyperboloid
from ham.training.pipeline import TrainingPhase, HAMPipeline
from ham.training.losses import (
    ReconstructionLoss,
    KLDivergenceLoss,
    ZermeloAlignmentLoss,
    ContrastiveAlignmentLoss,
    MetricAnchorLoss,
    MetricSmoothnessLoss,
)


def get_filter_fn(selector):
    """Returns a filter function for eqx.partition."""
    def filter_spec(model):
        base_mask = jax.tree_util.tree_map(lambda _: False, model)
        targets = selector(model)
        def make_true(n):
            return jax.tree_util.tree_map(
                lambda leaf: True if eqx.is_array(leaf) else False, n
            )
        if isinstance(targets, tuple):
            true_mask = tuple(make_true(t) for t in targets)
        else:
            true_mask = make_true(targets)
        return eqx.tree_at(selector, base_mask, replace=true_mask)
    return filter_spec


class MockMetric(Riemannian, eqx.Module):
    def __init__(self, manifold):
        self.g_net = eqx.nn.Linear(1, 1, key=jax.random.PRNGKey(0))
        self.manifold = manifold

    def inner_product(self, x, u, v=None, w=None):
        if v is None: v = u
        return jnp.sum(u * v, axis=-1)
    
    def spray(self, x, v):
        return jnp.zeros_like(v)
        
    def _get_zermelo_data(self, x):
        dim = x.shape[-1]
        W = jnp.ones(dim) * 0.1
        return jnp.eye(dim), W, jnp.eye(dim)


class TestModularTraining(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(123)
        self.data_dim = 10
        self.latent_dim = 2
        self.N = 50
        
        X = jax.random.normal(self.key, (self.N, self.data_dim))
        V = jax.random.normal(self.key, (self.N, self.data_dim))
        labels = jnp.zeros(self.N)
        lineage_pairs = jnp.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        
        self.dataset = BioDataset(X, V, labels, lineage_pairs)
        
        manifold = Hyperboloid(intrinsic_dim=self.latent_dim)
        metric = MockMetric(manifold)
        
        self.vae = GeometricVAE(self.data_dim, self.latent_dim, metric, self.key)

    def test_phase1_manifold(self):
        """Test Phase 1: VAE training via the modular pipeline."""
        phase = TrainingPhase(
            name="Manifold",
            epochs=2,
            optimizer=optax.adam(1e-3),
            losses=[ReconstructionLoss(weight=1.0), KLDivergenceLoss(weight=1e-4)],
            filter_spec=get_filter_fn(lambda m: (m.encoder_net, m.decoder_net)),
            requires_pairs=False,
        )
        
        old_dec = self.vae.decoder_net.layers[0].weight.copy()
        
        pipeline = HAMPipeline(self.vae)
        trained = pipeline.fit(self.dataset, phases=[phase], batch_size=10, seed=123)
        
        new_dec = trained.decoder_net.layers[0].weight
        self.assertFalse(jnp.allclose(old_dec, new_dec), "Decoder weights should change")

    def test_phase2_metric(self):
        """Test Phase 2: Metric training via the modular pipeline."""
        phase = TrainingPhase(
            name="Metric",
            epochs=2,
            optimizer=optax.adam(1e-3),
            losses=[
                ContrastiveAlignmentLoss(weight=1.0),
                MetricAnchorLoss(weight=1.0),
            ],
            filter_spec=get_filter_fn(lambda m: m.metric),
            requires_pairs=True,
        )
        
        pipeline = HAMPipeline(self.vae)
        trained = pipeline.fit(self.dataset, phases=[phase], batch_size=5, seed=123)
        
        self.assertIsInstance(trained, GeometricVAE)

    def test_full_pipeline(self):
        """Run the full two-phase pipeline for a few epochs."""
        phase1 = TrainingPhase(
            name="Manifold",
            epochs=2,
            optimizer=optax.adam(1e-3),
            losses=[ReconstructionLoss(weight=1.0)],
            filter_spec=get_filter_fn(lambda m: (m.encoder_net, m.decoder_net)),
            requires_pairs=False,
        )
        phase2 = TrainingPhase(
            name="Metric",
            epochs=2,
            optimizer=optax.adam(1e-3),
            losses=[ContrastiveAlignmentLoss(weight=1.0)],
            filter_spec=get_filter_fn(lambda m: m.metric),
            requires_pairs=True,
        )
        
        pipeline = HAMPipeline(self.vae)
        trained = pipeline.fit(self.dataset, phases=[phase1, phase2], batch_size=5, seed=123)
        
        self.assertIsInstance(trained, GeometricVAE)


if __name__ == '__main__':
    unittest.main()