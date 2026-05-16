import jax
import jax.numpy as jnp
import unittest
import numpy as np
from jax import config

# Ensure 64-bit precision for rigorous geometric testing
config.update("jax_enable_x64", True)

from ham.geometry.manifold import Manifold
from ham.geometry import Euclidean, Riemannian, Randers, DiscreteRanders
from ham.utils.math import GRAD_EPS, PSD_EPS

class FlatPlane(Manifold):
    @property
    def ambient_dim(self): return 2
    @property
    def intrinsic_dim(self): return 2
    def project(self, x): return x
    def to_tangent(self, x, v): return v
    def retract(self, x, delta): return x + delta
    def random_sample(self, key, shape): 
        return jax.random.normal(key, shape + (2,))

class MockMesh(Manifold):
    """Minimal mock for DiscreteRanders testing."""
    @property
    def ambient_dim(self): return 2
    @property
    def intrinsic_dim(self): return 2
    def project(self, x): return x
    def to_tangent(self, x, v): return v
    def retract(self, x, delta): return x + delta
    def random_sample(self, key, shape): 
        return jax.random.normal(key, shape + (2,))
    def get_face_weights(self, x):
        return jnp.array([1.0, 0.0])

class TestMetricZoo(unittest.TestCase):
    def setUp(self):
        self.manifold = FlatPlane()

    def test_euclidean_basic(self):
        metric = Euclidean(self.manifold)
        cost = metric.metric_fn(jnp.zeros(2), jnp.array([3.0, 4.0]))
        np.testing.assert_allclose(float(cost), 5.0, atol=1e-10)

    def test_euclidean_zero(self):
        metric = Euclidean(self.manifold)
        cost = metric.metric_fn(jnp.zeros(2), jnp.zeros(2))
        self.assertEqual(float(cost), 0.0)

class TestRiemannian(unittest.TestCase):
    def setUp(self):
        self.manifold = FlatPlane()
        
    def test_riemannian_scaling(self):
        def g_net(x): return 4.0 * jnp.eye(2)
        metric = Riemannian(self.manifold, g_net)
        cost = metric.metric_fn(jnp.zeros(2), jnp.array([1.0, 0.0]))
        # sqrt((4*4 + PSD_EPS)*1) = sqrt(16 + PSD_EPS)
        expected = jnp.sqrt(16.0 + PSD_EPS)
        np.testing.assert_allclose(float(cost), float(expected), atol=1e-10)

    def test_riemannian_zero(self):
        def g_net(x): return jnp.eye(2)
        metric = Riemannian(self.manifold, g_net)
        cost = metric.metric_fn(jnp.zeros(2), jnp.zeros(2))
        self.assertEqual(float(cost), 0.0)

class TestRanders(unittest.TestCase):
    def setUp(self):
        self.manifold = FlatPlane()
        self.key = jax.random.PRNGKey(0)

    def test_randers_analytical_match(self):
        def h_net(x): return jnp.eye(2)
        def w_net(x): return jnp.array([-0.1, 0.0])
        metric = Randers(self.manifold, h_net, w_net)
        x = jnp.zeros(2)
        v_east = jnp.array([1.0, 0.0])
        
        cost_east = metric.metric_fn(x, v_east)
        
        h_val = 1.0 + PSD_EPS
        wv_h = -0.1 * h_val
        lam = 1.0 - 0.01 * h_val
        # Disc = lam * ||v||_h^2 + <w,v>_h^2 = lam * h_val + wv_h^2 = (1 - 0.01h)*h + 0.01h^2 = h
        expected = (jnp.sqrt(h_val) - wv_h) / lam
        
        np.testing.assert_allclose(float(cost_east), float(expected), atol=1e-10)

    def test_randers_zero_vector(self):
        def h_net(x): return jnp.eye(2)
        def w_net(x): return jnp.array([0.5, 0.0])
        metric = Randers(self.manifold, h_net, w_net)
        cost = metric.metric_fn(jnp.zeros(2), jnp.zeros(2))
        self.assertEqual(float(cost), 0.0)

    def test_randers_jax_transforms(self):
        def h_net(x): return jnp.eye(2)
        def w_net(x): return jnp.array([0.1, 0.1])
        metric = Randers(self.manifold, h_net, w_net)
        x = jnp.array([0.5, 0.5])
        v = jnp.array([1.0, 0.0])
        
        jitted = jax.jit(metric.metric_fn)
        np.testing.assert_allclose(jitted(x, v), metric.metric_fn(x, v))
        
        vmapped = jax.vmap(metric.metric_fn)
        self.assertEqual(vmapped(jnp.tile(x, (5, 1)), jnp.tile(v, (5, 1))).shape, (5,))
        
        def energy(x, v): return 0.5 * metric.metric_fn(x, v)**2
        grad_x = jax.grad(energy, argnums=0)(x, v)
        self.assertTrue(jnp.all(jnp.isfinite(grad_x)))

class TestDiscreteRanders(unittest.TestCase):
    def test_discrete_randers_basic(self):
        mesh = MockMesh()
        face_winds = jnp.array([[0.1, 0.0], [0.0, 0.0]])
        metric = DiscreteRanders(mesh, face_winds)
        cost = metric.metric_fn(jnp.zeros(2), jnp.array([1.0, 0.0]))
        # 0.9 / 0.99 = 0.909090...
        np.testing.assert_allclose(float(cost), 0.90909090909, atol=1e-10)

    def test_discrete_randers_zero(self):
        mesh = MockMesh()
        face_winds = jnp.array([[0.5, 0.0], [0.0, 0.0]])
        metric = DiscreteRanders(mesh, face_winds)
        cost = metric.metric_fn(jnp.zeros(2), jnp.zeros(2))
        self.assertEqual(float(cost), 0.0)

if __name__ == '__main__':
    unittest.main()