"""Tests for learned metric classes in ham.models.learned.

Covers NeuralRanders (convexity, gradients, jit/vmap), NeuralRiemannian,
PullbackRiemannian, PullbackGNet, and KernelWindField.
"""

import jax
import jax.numpy as jnp
import unittest
import equinox as eqx
import numpy as np
from ham.geometry.manifold import Manifold
from ham.models.learned import (
    NeuralRanders, NeuralRiemannian,
    PullbackRiemannian, PullbackGNet, KernelWindField,
)

class MockManifold(Manifold):
    """Trivial R^3 manifold for testing."""
    @property
    def ambient_dim(self): return 3
    @property
    def intrinsic_dim(self): return 3
    def project(self, x): return x
    def to_tangent(self, x, v): return v
    def retract(self, x, v): return x + v
    def random_sample(self, key, shape): 
        return jax.random.normal(key, shape + (3,))


class TestNeuralRanders(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(0)
        self.manifold = MockManifold()
        self.metric = NeuralRanders(self.manifold, self.key, hidden_dim=16)

    def test_zermelo_convexity_enforcement(self):
        """Even with huge wind weights, causality ||W||_h < 1 must hold."""
        x = jnp.zeros(3)
        
        huge_w_net = eqx.tree_at(
            lambda m: m.mlp.layers[-1].weight,
            self.metric.w_net,
            jnp.ones_like(self.metric.w_net.mlp.layers[-1].weight) * 1000.0
        )
        broken_metric = eqx.tree_at(lambda m: m.w_net, self.metric, huge_w_net)
        
        H, W, lam = broken_metric.zermelo_data(x)
        w_norm = jnp.sqrt(jnp.dot(W, jnp.dot(H, W)))
        
        self.assertLess(float(w_norm), 1.0, "Wind vector violated convexity constraint!")
        self.assertGreater(float(lam), 0.0, "Lambda became non-positive!")

    def test_gradients_exist(self):
        """Metric must be differentiable w.r.t input x and parameters."""
        x = jnp.array([0.5, 0.5, 0.5])
        v = jnp.array([1.0, 0.0, 0.0])
        
        # 1. Grad w.r.t Input (Needed for Solver)
        grad_x = jax.grad(self.metric.energy, argnums=0)(x, v)
        self.assertEqual(grad_x.shape, (3,))
        self.assertTrue(jnp.isfinite(grad_x).all())
        
        # 2. Grad w.r.t Weights (Needed for Training)
        def loss(m):
            return m.energy(x, v)
            
        grads = eqx.filter_grad(loss)(self.metric)
        
        # Check gradient flow to both H and W nets
        w_grad = grads.w_net.mlp.layers[0].weight
        self.assertGreater(float(jnp.abs(w_grad).max()), 1e-10,
                           "No gradient flow to Wind Network")
        
        h_grad = grads.h_net.mlp.layers[0].weight
        self.assertGreater(float(jnp.abs(h_grad).max()), 1e-10,
                           "No gradient flow to H Network")

    def test_jit_compatible(self):
        """metric_fn and energy must work under jit."""
        x = jnp.array([0.1, 0.2, 0.3])
        v = jnp.array([1.0, 0.0, 0.0])
        jit_energy = eqx.filter_jit(self.metric.energy)
        result = jit_energy(x, v)
        self.assertTrue(jnp.isfinite(result))
        np.testing.assert_allclose(float(result), float(self.metric.energy(x, v)), atol=1e-5)

    def test_vmap_compatible(self):
        """Energy must work under vmap for batch evaluation."""
        xs = jnp.ones((4, 3)) * 0.5
        vs = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (4, 1))
        result = jax.vmap(self.metric.energy)(xs, vs)
        self.assertEqual(result.shape, (4,))
        self.assertTrue(jnp.all(jnp.isfinite(result)))

    def test_finsler_positivity(self):
        """F(x, v) >= 0 for all v, and F(x, 0) == 0."""
        x = jnp.array([0.1, 0.2, 0.3])
        v = jnp.array([1.0, 0.5, -0.3])
        self.assertGreaterEqual(float(self.metric.metric_fn(x, v)), 0.0)
        self.assertAlmostEqual(float(self.metric.metric_fn(x, jnp.zeros(3))), 0.0, places=4)


class TestNeuralRiemannian(unittest.TestCase):
    """Smoke tests for NeuralRiemannian."""

    def setUp(self):
        self.manifold = MockManifold()
        self.metric = NeuralRiemannian(self.manifold, jax.random.PRNGKey(1), hidden_dim=16)

    def test_metric_fn_positive(self):
        x = jnp.array([0.1, 0.2, 0.3])
        v = jnp.array([1.0, 0.0, 0.0])
        self.assertGreater(float(self.metric.metric_fn(x, v)), 0.0)

    def test_energy_finite(self):
        x = jnp.array([0.5, 0.5, 0.5])
        v = jnp.array([0.0, 1.0, 0.0])
        e = self.metric.energy(x, v)
        self.assertTrue(jnp.isfinite(e))
        self.assertGreater(float(e), 0.0)

    def test_jit_vmap(self):
        xs = jnp.ones((3, 3)) * 0.5
        vs = jnp.eye(3)
        energy_fn = eqx.filter_jit(jax.vmap(self.metric.energy))
        result = energy_fn(xs, vs)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(jnp.all(jnp.isfinite(result)))


class TestPullbackGNet(unittest.TestCase):
    """Tests for the pullback metric tensor computation."""

    def setUp(self):
        self.key = jax.random.PRNGKey(2)
        # Simple linear decoder R^3 -> R^5
        self.decoder = eqx.nn.Linear(3, 5, key=self.key)
        self.g_net = PullbackGNet(decoder=self.decoder, dim=3)

    def test_output_shape_and_spd(self):
        z = jnp.array([0.1, 0.2, 0.3])
        G = self.g_net(z)
        self.assertEqual(G.shape, (3, 3))
        # Symmetry
        self.assertLess(float(jnp.max(jnp.abs(G - G.T))), 1e-10)
        # Positive definite
        eigs = jnp.linalg.eigvalsh(G)
        self.assertGreater(float(jnp.min(eigs)), 0.0)

    def test_jit_compatible(self):
        z = jnp.array([0.1, 0.2, 0.3])
        jit_fn = eqx.filter_jit(self.g_net)
        np.testing.assert_allclose(jit_fn(z), self.g_net(z), atol=1e-5)

    def test_gradient_finite(self):
        z = jnp.array([0.1, 0.2, 0.3])
        grad_z = jax.grad(lambda v: jnp.trace(self.g_net(v)))(z)
        self.assertTrue(jnp.all(jnp.isfinite(grad_z)))


class TestPullbackRiemannian(unittest.TestCase):
    """Tests for PullbackRiemannian metric."""

    def setUp(self):
        self.manifold = MockManifold()
        self.decoder = eqx.nn.Linear(3, 5, key=jax.random.PRNGKey(3))
        self.metric = PullbackRiemannian(self.manifold, decoder=self.decoder)

    def test_energy_positive(self):
        x = jnp.array([0.1, 0.2, 0.3])
        v = jnp.array([1.0, 0.0, 0.0])
        e = self.metric.energy(x, v)
        self.assertTrue(jnp.isfinite(e))
        self.assertGreater(float(e), 0.0)

    def test_jit_vmap(self):
        xs = jnp.ones((3, 3)) * 0.1
        vs = jnp.eye(3)
        energy_fn = eqx.filter_jit(jax.vmap(self.metric.energy))
        result = energy_fn(xs, vs)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(jnp.all(jnp.isfinite(result)))


class TestKernelWindField(unittest.TestCase):
    """Tests for the non-parametric kernel wind smoother."""

    def setUp(self):
        self.anchors_z = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.anchors_v = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        self.kwf = KernelWindField(self.anchors_z, self.anchors_v, sigma=0.5)

    def test_output_shape(self):
        z = jnp.array([0.5, 0.5])
        w = self.kwf(z)
        self.assertEqual(w.shape, (2,))
        self.assertTrue(jnp.all(jnp.isfinite(w)))

    def test_near_anchor_returns_anchor_velocity(self):
        """At an anchor point, the output should be dominated by that anchor's velocity."""
        z = jnp.array([0.0, 0.0])  # First anchor
        w = self.kwf(z)
        # Should be dominated by [1.0, 0.0] — use generous tolerance
        # since other anchors contribute
        self.assertGreater(float(w[0]), 0.5, "Wind should point mostly in anchor direction")

    def test_jit_compatible(self):
        z = jnp.array([0.5, 0.5])
        jit_fn = eqx.filter_jit(self.kwf)
        np.testing.assert_allclose(jit_fn(z), self.kwf(z), atol=1e-5)

    def test_vmap_compatible(self):
        zs = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        result = jax.vmap(self.kwf)(zs)
        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(jnp.all(jnp.isfinite(result)))


if __name__ == '__main__':
    unittest.main()