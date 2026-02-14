import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import config

config.update("jax_enable_x64", True)

from ham.bio.data import BioDataset
from ham.bio.vae import GeometricVAE
from ham.bio.train_joint import GeometricTrainer
from ham.geometry.zoo import Riemannian
from ham.geometry.surfaces import Hyperboloid

class MockMetric(Riemannian, eqx.Module):
    # We override the fields to ensure they are tracked by Equinox
    # The parent has g_net, so we must initialize it.
    
    def __init__(self, manifold):
        # Initialize required fields from parent
        self.g_net = eqx.nn.Linear(1, 1, key=jax.random.PRNGKey(0))
        self.manifold = manifold

    def inner_product(self, x, u, v=None, w=None):
        if v is None: v = u
        return jnp.sum(u * v, axis=-1)
    
    def spray(self, x, v):
        return jnp.zeros_like(v)
        
    def _get_zermelo_data(self, x):
        dim = x.shape[-1]
        # Return F=1, W=const, D=Identity
        # Non-zero wind to verify alignment loss
        W = jnp.ones(dim) * 0.1
        return 1.0, W, jnp.eye(dim)

class TestJointTraining(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(123)
        self.data_dim = 10
        self.latent_dim = 2
        self.N = 50
        
        # 1. Create Synthetic Data
        X = jax.random.normal(self.key, (self.N, self.data_dim))
        V = jax.random.normal(self.key, (self.N, self.data_dim))
        labels = jnp.zeros(self.N)
        # Lineage pairs: simple indices
        lineage_pairs = jnp.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        
        self.dataset = BioDataset(X, V, labels, lineage_pairs)
        
        # 2. Initialize Model
        # Ensure Hyperboloid is correctly instantiated
        manifold = Hyperboloid(intrinsic_dim=self.latent_dim)
        metric = MockMetric(manifold)
        
        self.vae = GeometricVAE(self.data_dim, self.latent_dim, metric, self.key)
        self.trainer = GeometricTrainer(self.vae, learning_rate=1e-3)

    def test_phase1_manifold_step(self):
        """Test a single step of Phase 1 (VAE) training."""
        x_batch = self.dataset.X[:10]
        v_batch = self.dataset.V[:10]
        
        # New API: Don't pass model/opt_state
        new_model, new_opt, loss, r, k, a = self.trainer.train_step_manifold(
            x_batch, v_batch, self.key
        )
        
        self.assertTrue(jnp.isfinite(loss))
        
        # Ensure model weights changed
        old_dec = self.trainer.model.decoder_net.layers[0].weight
        new_dec = new_model.decoder_net.layers[0].weight
        self.assertFalse(jnp.allclose(old_dec, new_dec))

    def test_phase2_metric_step(self):
        """Test a single step of Phase 2 (Metric) training."""
        z_parent = jnp.array([[1.5, 0.5, 0.5], [1.2, 0.1, 0.1]]) 
        z_child  = jnp.array([[2.0, 1.0, 1.0], [1.5, 0.5, 0.5]])
        
        manifold = Hyperboloid(intrinsic_dim=2)
        z_parent = manifold.project(z_parent)
        z_child = manifold.project(z_child)
        
        new_model, new_opt, loss = self.trainer.train_step_metric(
            z_parent, z_child
        )
        
        self.assertTrue(jnp.isfinite(loss))

    def test_full_training_loop(self):
        """Run the full train() loop for a few epochs."""
        final_model = self.trainer.train(
            self.dataset, 
            batch_size=10, 
            epochs_manifold=2, 
            epochs_metric=2
        )
        
        self.assertIsInstance(final_model, GeometricVAE)

if __name__ == '__main__':
    unittest.main()