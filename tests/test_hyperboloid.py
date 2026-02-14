import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

# Use 64-bit precision for geometric checks
config.update("jax_enable_x64", True)

from ham.geometry.surfaces import Hyperboloid

class TestHyperboloid(unittest.TestCase):
    
    def setUp(self):
        self.dim = 2
        self.manifold = Hyperboloid(intrinsic_dim=self.dim)
        self.key = jax.random.PRNGKey(1337)

    def minkowski_dot(self, u, v):
        """Helper to compute <u, v>_L = -u0v0 + u1v1 + ..."""
        return -u[0] * v[0] + jnp.sum(u[1:] * v[1:])

    def test_dimensions(self):
        """Ensure ambient dimension is intrinsic + 1."""
        self.assertEqual(self.manifold.intrinsic_dim, 2)
        self.assertEqual(self.manifold.ambient_dim, 3)

    def test_projection_constraints(self):
        """
        Projecting any random point should land on the hyperboloid:
        -x0^2 + x1^2 + ... = -1
        x0 > 0
        """
        # Generate random ambient points (some valid, some invalid)
        key, subkey = jax.random.split(self.key)
        random_ambient = jax.random.normal(subkey, (10, 3)) * 5.0
        
        projected = self.manifold.project(random_ambient)
        
        for i in range(10):
            p = projected[i]
            norm_sq = self.minkowski_dot(p, p)
            
            # 1. Check Hyperboloid equation <p,p>_L = -1
            self.assertAlmostEqual(norm_sq, -1.0, places=6)
            
            # 2. Check Upper Sheet x0 > 0
            self.assertGreater(p[0], 0.0)

    def test_projection_idempotence(self):
        """Projecting a valid point should not change it."""
        x = jnp.array([1.0, 0.0, 0.0]) # The origin is definitely on the manifold
        p = self.manifold.project(x)
        np.testing.assert_allclose(p, x, atol=1e-8)
        
        # Double projection
        p2 = self.manifold.project(p)
        np.testing.assert_allclose(p2, p, atol=1e-8)

    def test_tangent_space_orthogonality(self):
        """
        Tangent vectors v at x must be Minkowski-orthogonal to x.
        <x, v>_L = 0
        """
        # 1. Get a valid point on manifold
        x = jnp.array([np.cosh(1.0), np.sinh(1.0), 0.0]) 
        
        # 2. Generate random ambient vector
        v_amb = jnp.array([1.0, 2.0, 3.0])
        
        # 3. Project to tangent space
        v_tan = self.manifold.to_tangent(x, v_amb)
        
        # 4. Check orthogonality
        inner = self.minkowski_dot(x, v_tan)
        self.assertAlmostEqual(inner, 0.0, places=6)

    def test_random_sampling(self):
        """
        Random samples must satisfy manifold constraints.
        """
        samples = self.manifold.random_sample(self.key, (100,))
        
        for i in range(100):
            p = samples[i]
            norm_sq = self.minkowski_dot(p, p)
            self.assertAlmostEqual(norm_sq, -1.0, places=5)
            self.assertGreater(p[0], 0.0)

    def test_metric_tensor_signature(self):
        """
        The metric tensor returned should be Minkowski diag(-1, 1, 1).
        """
        x = jnp.array([1.0, 0.0, 0.0])
        g = self.manifold.metric_tensor(x)
        
        expected = jnp.diag(jnp.array([-1.0, 1.0, 1.0]))
        np.testing.assert_allclose(g, expected)

    def test_retraction_stays_on_manifold(self):
        """
        Retracting a tangent vector should result in a valid point on the manifold.
        """
        x = jnp.array([1.0, 0.0, 0.0]) # Origin
        v = jnp.array([0.0, 0.5, 0.5]) # Valid tangent vector at origin (<x, v>_L = 0)
        
        new_x = self.manifold.retract(x, v)
        
        norm_sq = self.minkowski_dot(new_x, new_x)
        self.assertAlmostEqual(norm_sq, -1.0, places=6)

if __name__ == '__main__':
    unittest.main()