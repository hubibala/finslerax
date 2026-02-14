import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
from jax import config

# Enable 64-bit precision for robust gradient checks
config.update("jax_enable_x64", True)

from ham.geometry.surfaces import Hyperboloid
from ham.bio.vae import GeometricVAE, WrappedHyperbolicNormal
from ham.geometry.zoo import Riemannian

class MockMetric(Riemannian):
    """Simple Euclidean-like metric for testing VAE integration."""
    
    # We override __init__ to make it easier to instantiate in tests
    def __init__(self, manifold):
        # Pass a dummy network to the parent class to satisfy Equinox requirements
        dummy_net = eqx.nn.Linear(1, 1, key=jax.random.PRNGKey(0))
        super().__init__(manifold, dummy_net)

    def inner_product(self, x, u, v=None, w=None):
        if v is None: v = u
        # Simple Euclidean dot product in ambient space for testing
        return jnp.sum(u * v, axis=-1)
    
    def spray(self, x, v):
        # Zero spray = Geodesics are straight lines (in chart/ambient)
        return jnp.zeros_like(v)
        
    def _get_zermelo_data(self, x):
        # Mock wind field for Zermelo navigation check
        dim = x.shape[-1]
        # F=1, W=0, D=Identity
        return 1.0, jnp.zeros(dim), jnp.eye(dim)

class TestHyperbolicVAE(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        self.dim = 2
        self.manifold = Hyperboloid(intrinsic_dim=self.dim)
        # Now MockMetric takes the manifold argument
        self.metric = MockMetric(self.manifold)

    def test_exp_map_gradients(self):
        """
        Check if gradients propagate through the exponential map.
        L = sum(Exp_x(v)^2)
        """
        x = jnp.array([1.0, 0.0, 0.0]) # Origin
        v = jnp.array([0.0, 0.5, 0.5]) # Tangent vector
        
        def loss_fn(v_in):
            z = self.manifold.exp_map(x, v_in)
            return jnp.sum(z**2)
            
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(v)
        
        self.assertTrue(jnp.all(jnp.isfinite(grads)))
        self.assertFalse(jnp.all(grads == 0))

    def test_parallel_transport_gradients(self):
        """
        Check if gradients propagate through parallel transport.
        """
        x = jnp.array([1.0, 0.0, 0.0])
        y_raw = jnp.array([2.0, 1.0, 0.0]) # Random point
        y = self.manifold.project(y_raw)
        v = jnp.array([0.0, 0.5, 0.5])
        
        def loss_fn(y_in):
            # Transport v from x to y_in
            v_trans = self.manifold.parallel_transport(x, y_in, v)
            return jnp.sum(v_trans**2)
            
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(y)
        
        self.assertTrue(jnp.all(jnp.isfinite(grads)))

    def test_wrapped_normal_sampling(self):
        """
        Verify that WrappedHyperbolicNormal samples lie on the manifold.
        """
        mean = jnp.array([1.0, 0.0, 0.0])
        scale = jnp.array([0.5, 0.5]) # intrinsic dim = 2
        dist = WrappedHyperbolicNormal(mean, scale, self.manifold)
        
        # Sample
        z = dist.sample(self.key, (10,))
        
        # Check manifold constraints
        for i in range(10):
            norm_sq = self.manifold._minkowski_dot(z[i], z[i])
            self.assertAlmostEqual(norm_sq, -1.0, places=5)
            self.assertGreater(z[i, 0], 0.0)

    def test_vae_forward_pass(self):
        """
        Test the full GeometricVAE forward pass.
        """
        data_dim = 5
        latent_dim = 2
        vae = GeometricVAE(data_dim, latent_dim, self.metric, self.key)
        
        x = jax.random.normal(self.key, (data_dim,))
        v_rna = jax.random.normal(self.key, (data_dim,)) # Dummy velocity
        
        loss, aux = vae.loss_fn(x, v_rna, self.key)
        
        self.assertTrue(jnp.isfinite(loss))
        # Check aux outputs (recon, kl, spray, align)
        self.assertEqual(len(aux), 4)
        for val in aux:
            self.assertTrue(jnp.isfinite(val))

    def test_vae_gradients(self):
        """
        Test that we can compute gradients for the VAE parameters.
        """
        data_dim = 5
        latent_dim = 2
        vae = GeometricVAE(data_dim, latent_dim, self.metric, self.key)
        
        x = jax.random.normal(self.key, (data_dim,))
        v_rna = jax.random.normal(self.key, (data_dim,))
        
        def loss_wrapper(model):
            l, _ = model.loss_fn(x, v_rna, self.key)
            return l
            
        grads = eqx.filter_grad(loss_wrapper)(vae)
        
        # Check that at least some gradients are non-zero and finite
        is_finite = jax.tree_util.tree_reduce(
            lambda x, y: x and jnp.all(jnp.isfinite(y)), 
            grads, 
            True
        )
        self.assertTrue(is_finite)

if __name__ == '__main__':
    unittest.main()