import jax
import jax.numpy as jnp
import unittest
import numpy as np

from ham.geometry.manifold import Manifold
from ham.geometry.metric import FinslerMetric

class MockManifold(Manifold):
    """A trivial R^3 manifold for testing."""
    @property
    def ambient_dim(self): return 3
    @property
    def intrinsic_dim(self): return 3
    def project(self, x): return x
    def to_tangent(self, x, v): return v
    def retract(self, x, delta): return x + delta
    def random_sample(self, key, shape): 
        return jax.random.normal(key, shape + (3,))

class EuclideanMetric(FinslerMetric):
    """F(x, v) = |v|"""
    def metric_fn(self, x, v):
        return jnp.linalg.norm(v)

class ScaledEuclideanMetric(FinslerMetric):
    """F(x, v) = 0.5 * |v| (Should behave same as Euclidean structurally)"""
    def metric_fn(self, x, v):
        return 0.5 * jnp.linalg.norm(v)

class TestMetricPhysics(unittest.TestCase):
    
    def setUp(self):
        self.manifold = MockManifold()
        self.euc = EuclideanMetric(self.manifold)
        self.key = jax.random.PRNGKey(42)

    def test_euclidean_spray_is_zero(self):
        """
        In Euclidean space, geodesics are straight lines.
        Acceleration x'' should be 0.
        Spray G should be 0.
        """
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.5, -0.5, 1.0])
        
        spray = self.euc.spray(x, v)
        acc = self.euc.geod_acceleration(x, v)
        
        # Tolerances for float32/64
        np.testing.assert_allclose(spray, jnp.zeros_like(spray), atol=1e-5)
        np.testing.assert_allclose(acc, jnp.zeros_like(acc), atol=1e-5)

    def test_spray_homogeneity(self):
        """
        Spray G(x, v) must be 2-homogeneous in v.
        G(x, lambda * v) == lambda^2 * G(x, v)
        """
        x = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([1.0, 1.0, 1.0])
        lambda_val = 2.5
        
        # We use a locally defined CurvedMetric to ensure non-trivial spray
        class CurvedMetric(FinslerMetric):
            def metric_fn(self, x, v):
                # g_ij = diag(1 + x^2)
                g_diag = 1.0 + x**2
                sq_norm = jnp.sum(g_diag * v**2)
                return jnp.sqrt(sq_norm)
        
        curved = CurvedMetric(self.manifold)
        
        G_v = curved.spray(x, v)
        G_lambda_v = curved.spray(x, lambda_val * v)
        
        # Check scaling
        expected = (lambda_val**2) * G_v
        
        # Tolerances for float32/64
        np.testing.assert_allclose(G_lambda_v, expected, rtol=1e-4, atol=1e-6)

    def test_inner_product_consistency(self):
        """
        Test that inner_product calculates v^T g v correctly.
        """
        x = jnp.array([1.0, 1.0, 1.0])
        v = jnp.array([1.0, 0.0, 0.0])
        w1 = jnp.array([0.0, 1.0, 0.0])
        w2 = jnp.array([0.0, 1.0, 0.0])
        
        # For Euclidean, g_ij is Identity. <w1, w2> should be dot product.
        val = self.euc.inner_product(x, v, w1, w2)
        expected = jnp.dot(w1, w2)
        
        np.testing.assert_allclose(val, expected, atol=1e-5)

    def test_acceleration_sign(self):
        """
        Verify geod_acceleration = -2 * spray
        """
        x = jnp.array([0.5, 0.5, 0.5])
        v = jnp.array([1.0, -1.0, 2.0])
        
        spray = self.euc.spray(x, v)
        acc = self.euc.geod_acceleration(x, v)
        
        np.testing.assert_allclose(acc, -2.0 * spray, atol=1e-6)

if __name__ == '__main__':
    unittest.main()