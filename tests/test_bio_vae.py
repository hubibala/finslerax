import jax
import jax.numpy as jnp
import equinox as eqx
import unittest
import numpy as np

from jax import config
config.update("jax_enable_x64", True)

from ham.bio.vae import GeometricVAE, PowerSpherical
from ham.geometry.metric import FinslerMetric
from ham.geometry.manifold import Manifold

class MockManifold(Manifold):
    @property
    def ambient_dim(self): return 3
    @property
    def intrinsic_dim(self): return 3
    def project(self, x): return x / (jnp.linalg.norm(x) + 1e-12)
    def to_tangent(self, x, v): return v - jnp.dot(v, x) * x
    def random_sample(self, key, shape): return jax.random.normal(key, shape + (3,))

class EuclideanMetric(FinslerMetric):
    def metric_fn(self, x, v): return jnp.linalg.norm(v)

class MockLearnedMetric(FinslerMetric):
    """
    A spatially varying metric to ensure non-zero spray.
    F(x, v) = alpha * (1 + x_0^2) * |v|
    """
    alpha: jnp.ndarray
    
    def __init__(self, manifold):
        self.manifold = manifold
        self.alpha = jnp.array(2.0)

    def metric_fn(self, x, v):
        conformal_factor = 1.0 + x[0]**2
        return self.alpha * conformal_factor * jnp.linalg.norm(v)

class TestPowerSpherical(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(42)

    def test_sampling_shape_and_norm(self):
        latent_dim = 5
        mean = jax.random.normal(self.key, (latent_dim,))
        mean = mean / jnp.linalg.norm(mean)
        conc = jnp.array(10.0)
        dist = PowerSpherical(mean, conc)
        
        z_batch = dist.sample(self.key, shape=(100,))
        self.assertEqual(z_batch.shape, (100, latent_dim))
        
        norms = jnp.linalg.norm(z_batch, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5)

    def test_reparameterization(self):
        mean = jnp.array([0.6, 0.8, 0.0]) 
        conc = jnp.array(5.0)

        def sample_projection(m, c):
            dist = PowerSpherical(m, c)
            z = dist.sample(self.key)
            return jnp.dot(z, jnp.array([0.0, 1.0, 0.0]))

        grad_fn = jax.grad(sample_projection, argnums=(0, 1))
        g_mean, g_conc = grad_fn(mean, conc)
        
        self.assertTrue(jnp.any(g_mean != 0), "Gradient w.r.t mean should be non-zero")

class TestGeometricVAE(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(101)
        self.data_dim = 10
        self.latent_dim = 3
        self.manifold = MockManifold()
        self.metric = EuclideanMetric(self.manifold)
        self.x = jax.random.normal(self.key, (self.data_dim,))
        self.v_rna = jax.random.normal(self.key, (self.data_dim,))

    def test_velocity_projection_properties(self):
        vae = GeometricVAE(self.data_dim, self.latent_dim, self.metric, self.key)
        
        # CORRECTED METHOD NAME: project_control
        z_mean, u_lat = vae.project_control(self.x, self.v_rna)
        
        self.assertAlmostEqual(jnp.linalg.norm(z_mean), 1.0, places=5)
        
        # Orthogonality check (u_lat should be tangent to sphere at z_mean)
        dot_prod = jnp.dot(z_mean, u_lat)
        self.assertAlmostEqual(dot_prod, 0.0, places=4)

    def test_joint_gradient_flow(self):
        learnable_metric = MockLearnedMetric(self.manifold)
        vae = GeometricVAE(self.data_dim, self.latent_dim, learnable_metric, self.key)
        
        def loss_wrapper(model):
            loss, _ = model.loss_fn(self.x, self.v_rna, self.key)
            return loss
        
        grads = eqx.filter_grad(loss_wrapper)(vae)
        
        # 1. Check Encoder Gradients
        enc_leaves = jax.tree_util.tree_leaves(grads.encoder_net)
        has_grad_enc = any(jnp.any(g != 0) for g in enc_leaves if isinstance(g, jnp.ndarray))
        self.assertTrue(has_grad_enc, "Encoder should receive gradients from the loss.")
        
        # 2. Check Metric Gradients
        self.assertNotEqual(float(grads.metric.alpha), 0.0, 
            "The Metric parameter 'alpha' must receive gradients (Physics Engine is disconnected).")

if __name__ == '__main__':
    unittest.main()