import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

# Use 64-bit precision for geometric checks
# config.update("jax_enable_x64", True)

from ham.geometry.manifolds import Hyperboloid

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
            self.assertAlmostEqual(norm_sq, -1.0, places=4)
            
            # 2. Check Upper Sheet x0 > 0
            self.assertGreater(p[0], 0.0)

    def test_projection_idempotence(self):
        """Projecting a valid point should not change it."""
        x = jnp.array([1.0, 0.0, 0.0]) # The origin is definitely on the manifold
        p = self.manifold.project(x)
        np.testing.assert_allclose(p, x, atol=1e-5)
        
        # Double projection
        p2 = self.manifold.project(p)
        np.testing.assert_allclose(p2, p, atol=1e-5)

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
            self.assertAlmostEqual(norm_sq, -1.0, places=4)
            self.assertGreater(p[0], 0.0)

    def test_exp_log_roundtrip(self):
        """
        log_x(exp_x(v)) == v and exp_x(log_x(y)) == y.
        """
        x = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 0.5, 0.5])
        
        y = self.manifold.exp_map(x, v)
        # y should be on the manifold
        norm_sq = self.minkowski_dot(y, y)
        self.assertAlmostEqual(norm_sq, -1.0, places=6)
        
        v_recovered = self.manifold.log_map(x, y)
        np.testing.assert_allclose(v_recovered, v, atol=1e-5)

        # Reverse roundtrip
        y2 = self.manifold.random_sample(self.key, ())
        v2 = self.manifold.log_map(x, y2)
        y2_recovered = self.manifold.exp_map(x, v2)
        np.testing.assert_allclose(y2_recovered, y2, atol=1e-5)

    def test_parallel_transport(self):
        """
        Parallel transport should preserve Minkowski norm and inner products.
        """
        x = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([0.0, 0.5, 0.5])
        y = self.manifold.random_sample(self.key, ())
        
        v_trans = self.manifold.parallel_transport(x, y, v)
        
        # Norm preserved
        self.assertAlmostEqual(self.minkowski_dot(v_trans, v_trans), self.minkowski_dot(v, v), places=6)
        
        # Orthogonal to y
        self.assertAlmostEqual(self.minkowski_dot(y, v_trans), 0.0, places=6)

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