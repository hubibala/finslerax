import unittest
import jax
import jax.numpy as jnp
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from ham.geometry.surfaces import Sphere, Torus, Paraboloid, EuclideanSpace

class TestSurfaces(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(42)

    def test_sphere_operations(self):
        sphere = Sphere(intrinsic_dim=2, radius=2.0)
        self.assertEqual(sphere.ambient_dim, 3)
        self.assertEqual(sphere.intrinsic_dim, 2)
        
        # Sample
        pts = sphere.random_sample(self.key, (10,))
        norms = jnp.linalg.norm(pts, axis=-1)
        np.testing.assert_allclose(norms, 2.0, atol=1e-5)
        
        # Project
        x = jnp.array([1.0, 1.0, 1.0])
        x_proj = sphere.project(x)
        self.assertAlmostEqual(jnp.linalg.norm(x_proj), 2.0, places=5)
        
        # Retract / Exp Map
        v = jnp.array([1.0, -1.0, 0.0]) # Not tangent
        v_tan = sphere.to_tangent(x_proj, v)
        self.assertAlmostEqual(jnp.dot(v_tan, x_proj), 0.0, places=5)
        
        y = sphere.retract(x_proj, v_tan)
        self.assertAlmostEqual(jnp.linalg.norm(y), 2.0, places=5)
        
        # Log Map
        v_log = sphere.log_map(x_proj, y)
        np.testing.assert_allclose(v_log, v_tan, atol=1e-5)

    def test_sphere_high_dim(self):
        dim = 6
        sphere = Sphere(intrinsic_dim=dim, radius=1.0)
        self.assertEqual(sphere.ambient_dim, 7)
        self.assertEqual(sphere.intrinsic_dim, 6)
        
        pts = sphere.random_sample(self.key, (5,))
        self.assertEqual(pts.shape[-1], 7)
        np.testing.assert_allclose(jnp.linalg.norm(pts, axis=-1), 1.0, atol=1e-5)
        
        x = pts[0]
        v = jax.random.normal(self.key, (7,))
        v_tan = sphere.to_tangent(x, v)
        self.assertAlmostEqual(float(jnp.dot(v_tan, x)), 0.0, delta=1e-5)
        
        y = sphere.exp_map(x, v_tan * 0.1)
        self.assertAlmostEqual(float(jnp.linalg.norm(y)), 1.0, delta=1e-5)

    def test_torus_operations(self):
        torus = Torus(major_R=2.0, minor_r=1.0)
        
        # Sample and project consistency
        pts = torus.random_sample(self.key, (10,))
        proj_pts = jax.vmap(torus.project)(pts)
        np.testing.assert_allclose(pts, proj_pts, atol=1e-5)

    def test_paraboloid_operations(self):
        para = Paraboloid()
        x = jnp.array([1.0, 2.0, 0.0])
        x_proj = para.project(x)
        self.assertAlmostEqual(x_proj[2], x_proj[0]**2 + x_proj[1]**2, places=5)
        
        pts = para.random_sample(self.key, (10,))
        np.testing.assert_allclose(pts[:, 2], pts[:, 0]**2 + pts[:, 1]**2, atol=1e-5)

    def test_euclidean_space(self):
        euc = EuclideanSpace(dim=3)
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.5, 0.5, 0.5])
        
        np.testing.assert_allclose(euc.project(x), x)
        np.testing.assert_allclose(euc.exp_map(x, v), x + v)
        np.testing.assert_allclose(euc.log_map(x, x + v), v)

if __name__ == '__main__':
    unittest.main()
