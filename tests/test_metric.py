"""
Tests for FinslerMetric and derived quantities.

Note on conventions:
- `jax.numpy` (jnp) is used for all arrays that are intended to be JAX-traced.
- `numpy` (np) is used strictly for non-traced assertions and testing utilities.
- All tests use `ATOL` for consistency unless a tighter bound is documented.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ham.geometry import EuclideanSpace

# config.update("jax_enable_x64", True)
from ham.geometry.manifold import Manifold
from ham.geometry.metric import FinslerMetric
from ham.geometry.zoo import Randers
from ham.utils.math import safe_norm

# ---------- Module-Level Tolerance ----------
ATOL = 1e-5

# ---------- Module-Level Test Metrics ----------

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
    """F(x, v) = |v|. Batch-safe via axis=-1."""
    def metric_fn(self, x, v):
        return safe_norm(v, axis=-1)

class CurvedMetric(FinslerMetric):
    """
    Riemannian metric with g_ij = diag(1 + x^2).
    F(x, v) = sqrt(sum_i (1 + x_i^2) v_i^2).
    Produces non-trivial spray because g depends on x.
    """
    def metric_fn(self, x, v):
        g_diag = 1.0 + x**2
        sq_norm = jnp.sum(g_diag * v**2)
        return jnp.sqrt(sq_norm)


class TestSprayAndAcceleration(unittest.TestCase):

    def setUp(self):
        self.manifold = MockManifold()
        self.euc = EuclideanMetric(self.manifold)
        self.curved = CurvedMetric(self.manifold)
        self.key = jax.random.PRNGKey(42)

    def test_euclidean_spray_is_zero(self):
        """
        In Euclidean space, geodesics are straight lines.
        Spray G and acceleration should be 0.
        """
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.5, -0.5, 1.0])

        spray = self.euc.spray(x, v)
        acc = self.euc.geod_acceleration(x, v)

        np.testing.assert_allclose(spray, jnp.zeros_like(spray), atol=ATOL)
        np.testing.assert_allclose(acc, jnp.zeros_like(acc), atol=ATOL)

    def test_spray_homogeneity(self):
        """
        Spray G(x, v) must be 2-homogeneous in v.
        G(x, lambda * v) == lambda^2 * G(x, v)
        """
        x = jnp.array([1.0, 0.0, 0.0])
        v = jnp.array([1.0, 1.0, 1.0])
        lambda_val = 2.5

        G_v = self.curved.spray(x, v)
        G_lambda_v = self.curved.spray(x, lambda_val * v)

        expected = (lambda_val**2) * G_v
        np.testing.assert_allclose(G_lambda_v, expected, rtol=1e-4, atol=ATOL)

    def test_acceleration_sign_euclidean(self):
        """Verify geod_acceleration = -2 * spray (trivial case)."""
        x = jnp.array([0.5, 0.5, 0.5])
        v = jnp.array([1.0, -1.0, 2.0])

        spray = self.euc.spray(x, v)
        acc = self.euc.geod_acceleration(x, v)
        np.testing.assert_allclose(acc, -2.0 * spray, atol=ATOL)

    def test_acceleration_sign_curved(self):
        """
        Verify geod_acceleration = -2 * spray for the curved metric.
        This is the non-trivial case where spray != 0.
        """
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.5, -0.5, 1.0])

        spray = self.curved.spray(x, v)
        acc = self.curved.geod_acceleration(x, v)

        # Spray should be non-zero for this metric
        self.assertGreater(float(jnp.linalg.norm(spray)), 1e-6,
                           "CurvedMetric spray should be non-trivial")
        np.testing.assert_allclose(acc, -2.0 * spray, atol=ATOL)

    def test_spray_jit_vmap(self):
        """Test that spray composes safely with vmap and jit."""
        x = jax.random.normal(self.key, (10, 3))
        v = jax.random.normal(self.key, (10, 3))

        vmap_spray = jax.vmap(self.euc.spray)
        sprays = vmap_spray(x, v)
        self.assertEqual(sprays.shape, (10, 3))

        jit_vmap_spray = jax.jit(vmap_spray)
        sprays_jit = jit_vmap_spray(x, v)
        np.testing.assert_allclose(sprays, sprays_jit, atol=ATOL)

    def test_gradient_safety(self):
        """Test that gradients are stable at v=0 (no NaNs)."""
        x = jnp.array([1.0, 1.0, 1.0])
        v_zero = jnp.zeros(3)

        spray_zero = self.euc.spray(x, v_zero)
        self.assertFalse(jnp.any(jnp.isnan(spray_zero)))

        grad_v = jax.grad(self.euc.energy, argnums=1)(x, v_zero)
        self.assertFalse(jnp.any(jnp.isnan(grad_v)))
        np.testing.assert_allclose(grad_v, jnp.zeros(3), atol=ATOL)


class TestEnergy(unittest.TestCase):

    def setUp(self):
        self.manifold = MockManifold()
        self.euc = EuclideanMetric(self.manifold)
        self.curved = CurvedMetric(self.manifold)

    def test_energy_euclidean(self):
        """E(x, v) = 0.5 * F^2 = 0.5 * ||v||^2 for Euclidean."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([3.0, 4.0, 0.0])  # norm = 5

        energy = self.euc.energy(x, v)
        expected = 0.5 * jnp.sum(v**2)  # 0.5 * 25 = 12.5
        np.testing.assert_allclose(energy, expected, atol=ATOL)

    def test_energy_curved(self):
        """E(x, v) = 0.5 * F^2 for the curved diagonal metric."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([1.0, 1.0, 1.0])

        energy = self.curved.energy(x, v)
        # g = diag(2, 5, 10), E = 0.5 * (2 + 5 + 10) = 8.5
        expected = 0.5 * (2.0 + 5.0 + 10.0)
        np.testing.assert_allclose(energy, expected, atol=ATOL)

    def test_energy_identity(self):
        """Verify E = 0.5 * F^2 identity holds numerically."""
        x = jnp.array([0.3, 0.7, 1.2])
        v = jnp.array([1.5, -0.8, 2.1])

        E = self.curved.energy(x, v)
        F = self.curved.metric_fn(x, v)
        np.testing.assert_allclose(E, 0.5 * F**2, atol=ATOL)


class TestInnerProduct(unittest.TestCase):

    def setUp(self):
        self.manifold = MockManifold()
        self.euc = EuclideanMetric(self.manifold)
        self.curved = CurvedMetric(self.manifold)

    def test_inner_product_consistency(self):
        """For Euclidean, g_ij = I, so <w1, w2> = w1 . w2."""
        x = jnp.array([1.0, 1.0, 1.0])
        v = jnp.array([1.0, 0.0, 0.0])
        w1 = jnp.array([0.0, 1.0, 0.0])
        w2 = jnp.array([0.0, 1.0, 0.0])

        val = self.euc.inner_product(x, v, w1, w2)
        expected = jnp.dot(w1, w2)
        np.testing.assert_allclose(val, expected, atol=ATOL)

    def test_inner_product_curved(self):
        """For CurvedMetric, g_ij = diag(1 + x^2)."""
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([1.0, 1.0, 1.0])
        w1 = jnp.array([1.0, 0.0, 0.5])
        w2 = jnp.array([0.0, 1.0, 2.0])

        val = self.curved.inner_product(x, v, w1, w2)

        expected_g = jnp.diag(1.0 + x**2)
        expected_val = jnp.dot(w1, jnp.dot(expected_g, w2))
        np.testing.assert_allclose(val, expected_val, atol=ATOL)

    def test_fundamental_tensor_positive_definite(self):
        """
        The fundamental tensor g_ij must be positive definite.
        This is a core Finsler axiom (MATH_SPEC § 1.1, condition 3).
        """
        x = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.5, -0.3, 1.0])

        # Compute g_ij = Hess_v(E)
        hess = jax.hessian(self.curved.energy, argnums=1)(x, v)
        eigenvalues = jnp.linalg.eigvalsh(hess)

        # All eigenvalues must be strictly positive
        self.assertTrue(jnp.all(eigenvalues > 0),
                        f"Fundamental tensor must be positive definite; eigenvalues: {eigenvalues}")


class TestArcLength(unittest.TestCase):

    def setUp(self):
        self.manifold = MockManifold()
        self.euc = EuclideanMetric(self.manifold)

    def test_arc_length_degenerate(self):
        """arc_length handles paths with < 2 points gracefully."""
        path = jnp.array([[1.0, 2.0, 3.0]])
        self.assertEqual(float(self.euc.arc_length(path)), 0.0)

    def test_arc_length_straight_line(self):
        """arc_length of a straight line equals Euclidean distance."""
        path = jnp.linspace(jnp.zeros(3), jnp.ones(3), 10)
        length = self.euc.arc_length(path)
        expected = jnp.sqrt(3.0)
        np.testing.assert_allclose(length, expected, atol=ATOL)


class TestRandersMetric(unittest.TestCase):
    """
    Tests for Finslerian (non-Riemannian) metrics.
    The Randers metric F(x,v) = sqrt(v^T M v) + beta . v has v-dependent
    fundamental tensor, which is the central Finsler use case.
    """

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)
        self.h_net = lambda x: jnp.eye(2)
        self.w_net = lambda x: jnp.array([0.3, 0.0])  # Moderate constant wind
        self.metric = Randers(self.plane, self.h_net, self.w_net)

    def test_randers_spray_homogeneity(self):
        """
        Spray G(x, v) must be 2-homogeneous in v for Randers.
        Note: regularization may break exact homogeneity slightly.
        """
        x = jnp.array([1.0, 0.5])
        v = jnp.array([1.0, 1.0])
        lambda_val = 2.0

        G_v = self.metric.spray(x, v)
        G_lambda_v = self.metric.spray(x, lambda_val * v)
        expected = (lambda_val**2) * G_v

        # Slightly looser tolerance because Tikhonov reg can perturb
        np.testing.assert_allclose(G_lambda_v, expected, rtol=1e-3, atol=1e-4)

    def test_randers_fundamental_tensor_positive_definite(self):
        """Randers fundamental tensor must be positive definite for ||W|| < 1."""
        x = jnp.array([0.5, 0.5])
        v = jnp.array([1.0, 0.5])

        hess = jax.hessian(self.metric.energy, argnums=1)(x, v)
        eigenvalues = jnp.linalg.eigvalsh(hess)

        self.assertTrue(jnp.all(eigenvalues > 0),
                        f"Randers fundamental tensor must be PD; eigenvalues: {eigenvalues}")

    def test_randers_v_zero_finite(self):
        """Spray at v=0 must be finite (regularization prevents NaN)."""
        x = jnp.array([1.0, 1.0])
        v_zero = jnp.zeros(2)

        spray = self.metric.spray(x, v_zero)
        self.assertFalse(jnp.any(jnp.isnan(spray)))
        self.assertTrue(jnp.all(jnp.isfinite(spray)))


if __name__ == '__main__':
    unittest.main()
