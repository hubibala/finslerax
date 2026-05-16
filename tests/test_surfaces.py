import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

# Use 64-bit precision for geometric checks
config.update("jax_enable_x64", True)

from ham.geometry import Sphere, Torus, Paraboloid, Hyperboloid, EuclideanSpace

class SurfaceTestMixin:
    """Common tests for all manifolds."""
    manifold = None
    
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        
    def test_dimensions(self):
        self.assertIsInstance(self.manifold.ambient_dim, int)
        self.assertIsInstance(self.manifold.intrinsic_dim, int)
        
    def test_jit_compatibility(self):
        x = self.manifold.random_sample(self.key, ())
        v = self.manifold.random_sample(self.key, ())
        v = self.manifold.to_tangent(x, v)
        
        jitted_project = jax.jit(self.manifold.project)
        jitted_exp = jax.jit(self.manifold.exp_map)
        
        np.testing.assert_allclose(jitted_project(x), self.manifold.project(x), atol=1e-10)
        np.testing.assert_allclose(jitted_exp(x, v), self.manifold.exp_map(x, v), atol=1e-10)

    def test_vmap_compatibility(self):
        keys = jax.random.split(self.key, 5)
        pts = jax.vmap(self.manifold.random_sample, in_axes=(0, None))(keys, ())
        
        vmapped_project = jax.vmap(self.manifold.project)
        results = vmapped_project(pts)
        self.assertEqual(results.shape, pts.shape)

    def test_batch_support(self):
        """Test native batch support without explicit vmap."""
        shape = (3, 4)
        k1, k2 = jax.random.split(self.key)
        pts = self.manifold.random_sample(k1, shape)
        self.assertEqual(pts.shape, shape + (self.manifold.ambient_dim,))
        
        # Test project
        proj = self.manifold.project(pts)
        self.assertEqual(proj.shape, pts.shape)
        
        # Test to_tangent
        v = jax.random.normal(k2, pts.shape)
        v_tan = self.manifold.to_tangent(pts, v)
        self.assertEqual(v_tan.shape, pts.shape)

class TestSphere(SurfaceTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.radius = 2.0
        self.manifold = Sphere(intrinsic_dim=2, radius=self.radius)

    def test_sphere_constraints(self):
        pts = self.manifold.random_sample(self.key, (10,))
        norms = jnp.linalg.norm(pts, axis=-1)
        np.testing.assert_allclose(norms, self.radius, atol=1e-10)

    def test_sphere_orthogonality(self):
        x = self.manifold.random_sample(self.key, ())
        v = jax.random.normal(jax.random.split(self.key)[1], x.shape)
        v_tan = self.manifold.to_tangent(x, v)
        dot = jnp.sum(x * v_tan, axis=-1)
        np.testing.assert_allclose(dot, 0.0, atol=1e-10)

    def test_exp_log_roundtrip(self):
        x = self.manifold.random_sample(self.key, ())
        k1, k2 = jax.random.split(self.key)
        v = self.manifold.to_tangent(x, jax.random.normal(k1, x.shape) * 0.1)
        
        y = self.manifold.exp_map(x, v)
        v_rec = self.manifold.log_map(x, y)
        np.testing.assert_allclose(v_rec, v, atol=1e-8)

    def test_parallel_transport(self):
        k1, k2, k3 = jax.random.split(self.key, 3)
        x = self.manifold.random_sample(k1, ())
        y = self.manifold.random_sample(k2, ())
        v = self.manifold.to_tangent(x, jax.random.normal(k3, x.shape))
        
        v_trans = self.manifold.parallel_transport(x, y, v)
        # Check tangency at y
        np.testing.assert_allclose(jnp.sum(y * v_trans, axis=-1), 0.0, atol=1e-8)
        # Check norm preservation
        np.testing.assert_allclose(jnp.linalg.norm(v_trans), jnp.linalg.norm(v), atol=1e-8)

    def test_sphere_zero_tangent(self):
        x = self.manifold.random_sample(self.key, ())
        v = jnp.zeros_like(x)
        y = self.manifold.exp_map(x, v)
        np.testing.assert_allclose(y, x, atol=1e-10)
        v_rec = self.manifold.log_map(x, y)
        np.testing.assert_allclose(v_rec, v, atol=1e-10)

    def test_sphere_near_antipodal(self):
        x = jnp.array([self.radius, 0.0, 0.0])
        # Near antipodal point
        y = jnp.array([-self.radius + 1e-6, 1e-6, 0.0])
        y = self.manifold.project(y)
        
        v_log = self.manifold.log_map(x, y)
        y_rec = self.manifold.exp_map(x, v_log)
        np.testing.assert_allclose(y_rec, y, atol=1e-5)

class TestTorus(SurfaceTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.manifold = Torus(major_R=3.0, minor_r=1.0)

    def test_torus_constraints(self):
        pts = self.manifold.random_sample(self.key, (10,))
        # Idempotence of projection
        proj = self.manifold.project(pts)
        np.testing.assert_allclose(pts, proj, atol=1e-10)

    def test_torus_exp_log_approx(self):
        x = self.manifold.random_sample(self.key, ())
        k1, k2 = jax.random.split(self.key)
        v = self.manifold.to_tangent(x, jax.random.normal(k1, x.shape) * 0.01)
        
        y = self.manifold.exp_map(x, v)
        v_rec = self.manifold.log_map(x, y)
        # For very small v, the approximation should be decent
        np.testing.assert_allclose(v_rec, v, atol=1e-4)

class TestParaboloid(SurfaceTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.manifold = Paraboloid()

    def test_paraboloid_constraints(self):
        pts = self.manifold.random_sample(self.key, (10,))
        expected_z = pts[..., 0]**2 + pts[..., 1]**2
        np.testing.assert_allclose(pts[..., 2], expected_z, atol=1e-10)

    def test_paraboloid_tangent(self):
        x = self.manifold.random_sample(self.key, ())
        v = jax.random.normal(jax.random.split(self.key)[1], x.shape)
        v_tan = self.manifold.to_tangent(x, v)
        
        # Normal vector to z = x^2 + y^2 is (-2x, -2y, 1)
        n = jnp.array([-2*x[0], -2*x[1], 1.0])
        n = n / jnp.linalg.norm(n)
        dot = jnp.sum(n * v_tan, axis=-1)
        np.testing.assert_allclose(dot, 0.0, atol=1e-10)

class TestHyperboloid(SurfaceTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.manifold = Hyperboloid(intrinsic_dim=2)

    def minkowski_dot(self, u, v):
        return -u[..., 0] * v[..., 0] + jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)

    def test_hyperboloid_constraints(self):
        pts = self.manifold.random_sample(self.key, (10,))
        dots = self.minkowski_dot(pts, pts)
        np.testing.assert_allclose(dots, -1.0, atol=1e-10)
        self.assertTrue(jnp.all(pts[..., 0] > 0))

    def test_exp_log_roundtrip(self):
        x = self.manifold.random_sample(self.key, ())
        k1, k2 = jax.random.split(self.key)
        v = self.manifold.to_tangent(x, jax.random.normal(k1, x.shape) * 0.1)
        
        y = self.manifold.exp_map(x, v)
        v_rec = self.manifold.log_map(x, y)
        np.testing.assert_allclose(v_rec, v, atol=1e-8)

    def test_parallel_transport(self):
        k1, k2, k3 = jax.random.split(self.key, 3)
        x = self.manifold.random_sample(k1, ())
        y = self.manifold.random_sample(k2, ())
        v = self.manifold.to_tangent(x, jax.random.normal(k3, x.shape))
        
        v_trans = self.manifold.parallel_transport(x, y, v)
        # Check tangency at y
        np.testing.assert_allclose(self.minkowski_dot(y, v_trans), 0.0, atol=1e-8)
        # Check norm preservation
        norm_v = jnp.sqrt(jnp.maximum(self.minkowski_dot(v, v), 0.0))
        norm_trans = jnp.sqrt(jnp.maximum(self.minkowski_dot(v_trans, v_trans), 0.0))
        np.testing.assert_allclose(norm_trans, norm_v, atol=1e-8)

class TestEuclideanSpace(SurfaceTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.manifold = EuclideanSpace(dim=3)

    def test_euclidean_ops(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        v = y - x
        
        np.testing.assert_allclose(self.manifold.project(x), x, atol=1e-10)
        np.testing.assert_allclose(self.manifold.to_tangent(x, v), v, atol=1e-10)
        np.testing.assert_allclose(self.manifold.exp_map(x, v), y, atol=1e-10)
        np.testing.assert_allclose(self.manifold.log_map(x, y), v, atol=1e-10)
        np.testing.assert_allclose(self.manifold.parallel_transport(x, y, v), v, atol=1e-10)

    def test_euclidean_random(self):
        pts = self.manifold.random_sample(self.key, (10,))
        self.assertEqual(pts.shape, (10, 3))

if __name__ == '__main__':
    unittest.main()
