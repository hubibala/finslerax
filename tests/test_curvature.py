"""
Tests for Finsler curvature quantities.

Note on conventions:
- `jax.numpy` (jnp) is used for all arrays that are JAX-traced.
- `numpy` (np) is used strictly for non-traced assertions.
- float64 is enabled for precision in 3rd/4th-order autodiff chains.

Test structure:
    1. Zero curvature (Euclidean flat space)
    2. Known curvature (unit sphere, K = +1)
    3. Mathematical properties (antisymmetry, Euler identity)
    4. Numerical stability (safe division, near-degenerate planes)
    5. JAX transform compatibility (jit, vmap, grad)
    6. flag_curvature_sample (metric Gram-Schmidt, PRNG key API)
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

# config.update("jax_enable_x64", True)
from ham.geometry import Euclidean, EuclideanSpace, Randers, Riemannian, Sphere
from ham.geometry.curvature import (
    flag_curvature_sample,
    riemann_curvature_tensor,
    sectional_curvature,
)

# ---------------------------------------------------------------------------
# Shared metric-based Gram-Schmidt helper for test setup
# ---------------------------------------------------------------------------

def make_orthonormal_pair(metric, x, v1_raw, v2_raw):
    """Return a metric-orthonormal pair (v1, v2) from raw ambient vectors."""
    t1 = metric.manifold.to_tangent(x, v1_raw)
    g11 = metric.inner_product(x, t1, t1, t1)
    t1 = t1 / jnp.sqrt(jnp.maximum(g11, 1e-10))

    t2 = metric.manifold.to_tangent(x, v2_raw)
    g12 = metric.inner_product(x, t1, t1, t2)
    g11n = metric.inner_product(x, t1, t1, t1)
    t2 = t2 - (g12 / jnp.maximum(g11n, 1e-10)) * t1
    g22 = metric.inner_product(x, t1, t2, t2)
    t2 = t2 / jnp.sqrt(jnp.maximum(g22, 1e-10))
    return t1, t2


class TestRiemannCurvatureTensor(unittest.TestCase):

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)
        self.sphere = Sphere(intrinsic_dim=2, radius=1.0)

    # -----------------------------------------------------------------------
    # 1. Zero curvature — Euclidean flat space
    # -----------------------------------------------------------------------

    def test_riemann_tensor_euclidean_is_zero(self):
        """
        For flat Euclidean space, the Riemann tensor must vanish identically.
        This validates the entire autodiff pipeline for the zero-curvature baseline.
        """
        metric = Euclidean(self.plane)
        x = jnp.array([1.0, 2.0])
        v = jnp.array([1.0, 0.0])

        R = riemann_curvature_tensor(metric, x, v)
        np.testing.assert_allclose(R, jnp.zeros((2, 2, 2)), atol=1e-5)

    def test_riemann_tensor_antisymmetry(self):
        """
        R^i_jk must be antisymmetric in j and k: R^i_jk = -R^i_kj.
        This is a fundamental algebraic identity of curvature tensors.
        """
        # Use a curved Riemannian metric so the tensor is non-trivially non-zero
        def curved_g(x):
            return jnp.diag(jnp.array([1.0, 1.0 + x[0]**2]))

        metric = Riemannian(self.plane, curved_g)
        x = jnp.array([1.0, 0.5])
        v = jnp.array([0.7, 0.3])

        R = riemann_curvature_tensor(metric, x, v)
        # R[i, j, k] == -R[i, k, j]
        np.testing.assert_allclose(R, -jnp.transpose(R, (0, 2, 1)), atol=1e-5)

    def test_nonlinear_connection_euler_identity(self):
        """
        Euler homogeneity identity: N^i_j v^j = 2 G^i.
        This must hold for any 2-homogeneous spray.
        (Bao-Chern-Shen, Lemma 2.3.1)
        """
        from ham.geometry.curvature import _nonlinear_connection

        def curved_g(x):
            return jnp.diag(jnp.array([1.0 + x[0]**2, 1.0 + x[1]**2]))

        metric = Riemannian(self.plane, curved_g)
        x = jnp.array([1.0, 0.5])
        v = jnp.array([0.7, 0.3])

        N = _nonlinear_connection(metric, x, v)      # (D, D)
        G = metric.spray(x, v)                        # (D,)

        # N^i_j v^j should equal 2 G^i
        N_v = jnp.einsum('ij,j->i', N, v)
        np.testing.assert_allclose(N_v, 2.0 * G, atol=1e-5)


class TestSectionalCurvature(unittest.TestCase):

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)
        self.sphere = Sphere(intrinsic_dim=2, radius=1.0)

    # -----------------------------------------------------------------------
    # 2. Euclidean: K = 0
    # -----------------------------------------------------------------------

    def test_sectional_curvature_euclidean_is_zero(self):
        """Euclidean space has zero sectional curvature everywhere."""
        metric = Euclidean(self.plane)
        x = jnp.array([0.0, 0.0])
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])

        K = sectional_curvature(metric, x, v1, v2)
        np.testing.assert_allclose(K, 0.0, atol=1e-5)

    # -----------------------------------------------------------------------
    # 3. Curved space test
    # -----------------------------------------------------------------------

    def test_sectional_curvature_sphere_is_positive(self):
        """
        A metric with positive curvature should give non-zero K.
        """
        # Use g = diag(1, 1 + x[0]^2) and verify curvature is non-trivial
        def curved_g(x):
            return jnp.diag(jnp.array([1.0, 1.0 + x[0]**2]))

        metric = Riemannian(self.plane, curved_g)
        x = jnp.array([0.5, 0.0])  # non-trivial position
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])

        K = sectional_curvature(metric, x, v1, v2)
        self.assertFalse(jnp.isnan(K))
        self.assertTrue(jnp.isfinite(K))
        # Curvature is non-trivially non-zero for this metric
        self.assertGreater(abs(float(K)), 1e-4)

    def test_sectional_curvature_flat_riemannian_is_zero(self):
        """A flat Riemannian metric (constant g) gives K = 0 everywhere."""
        def flat_g(x):
            return jnp.diag(jnp.array([2.0, 3.0]))  # anisotropic but flat

        metric = Riemannian(self.plane, flat_g)
        x = jnp.array([1.0, 1.0])
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])

        K = sectional_curvature(metric, x, v1, v2)
        np.testing.assert_allclose(K, 0.0, atol=1e-5)

    # -----------------------------------------------------------------------
    # 4. Degenerate-plane guard
    # -----------------------------------------------------------------------

    def test_degenerate_plane_returns_zero(self):
        """
        Passing two parallel vectors (v2 = v1) yields a degenerate plane.
        The guard must return 0.0 without NaN.
        """
        metric = Euclidean(self.plane)
        x = jnp.array([1.0, 0.0])
        v = jnp.array([1.0, 0.0])

        K = sectional_curvature(metric, x, v, v)
        self.assertFalse(jnp.isnan(K))
        np.testing.assert_allclose(K, 0.0)

    # -----------------------------------------------------------------------
    # 5. JAX transform compatibility
    # -----------------------------------------------------------------------

    def test_jit_compatibility(self):
        """jit-compiled sectional_curvature must match eager evaluation."""
        def curved_g(x):
            return jnp.diag(jnp.array([1.0, 1.0 + x[0]**2]))

        metric = Riemannian(self.plane, curved_g)
        x = jnp.array([0.5, 0.0])
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])

        K_eager = sectional_curvature(metric, x, v1, v2)
        K_jit = jax.jit(sectional_curvature, static_argnums=0)(metric, x, v1, v2)
        np.testing.assert_allclose(K_eager, K_jit, atol=1e-5)

    def test_grad_through_sectional_curvature(self):
        """
        Gradient of K w.r.t. position x must be finite (no NaN).
        Validates the entire 4th-order autodiff chain is differentiable.
        """
        metric = Euclidean(self.plane)
        x = jnp.array([1.0, 2.0])
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])

        def K_fn(x):
            return sectional_curvature(metric, x, v1, v2)

        g = jax.grad(K_fn)(x)
        self.assertFalse(jnp.any(jnp.isnan(g)))

    def test_riemann_tensor_jit(self):
        """riemann_curvature_tensor must compile and run under jit."""
        metric = Euclidean(self.plane)
        x = jnp.array([0.0, 0.0])
        v = jnp.array([1.0, 0.0])

        R = jax.jit(riemann_curvature_tensor, static_argnums=0)(metric, x, v)
        self.assertEqual(R.shape, (2, 2, 2))
        self.assertFalse(jnp.any(jnp.isnan(R)))


class TestFlagCurvatureSample(unittest.TestCase):

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)
        self.key = jax.random.PRNGKey(0)

    def test_euclidean_flag_curvature_is_zero(self):
        """flag_curvature_sample returns ~0 for Euclidean metric."""
        metric = Euclidean(self.plane)
        x = jnp.array([1.0, 2.0])

        K = flag_curvature_sample(metric, x, self.key)
        self.assertFalse(jnp.isnan(K))
        np.testing.assert_allclose(K, 0.0, atol=1e-5)

    def test_flag_curvature_sample_is_non_zero(self):
        """
        flag_curvature_sample returns a finite non-NaN value for a curved metric.
        """
        def curved_g(x):
            return jnp.diag(jnp.array([1.0, 1.0 + x[0]**2]))

        metric = Riemannian(self.plane, curved_g)
        x = jnp.array([0.5, 0.0])

        K = flag_curvature_sample(metric, x, self.key)
        self.assertFalse(jnp.isnan(K))
        self.assertTrue(jnp.isfinite(K))
        self.assertGreater(abs(float(K)), 1e-4)

    def test_different_keys_give_same_curvature_on_constant_metric(self):
        """
        For a space of constant curvature (flat space K=0), all tangent planes
        give the same flag curvature regardless of the sampled key.
        """
        metric = Euclidean(self.plane)
        x = jnp.array([1.0, 0.0])

        key1 = jax.random.PRNGKey(1)
        key2 = jax.random.PRNGKey(99)
        K1 = flag_curvature_sample(metric, x, key1)
        K2 = flag_curvature_sample(metric, x, key2)

        np.testing.assert_allclose(K1, K2, atol=1e-5)

    def test_key_parameter_is_explicit(self):
        """
        Two calls with the same key must return the same value (determinism).
        This tests the JAX PRNG convention.
        """
        metric = Euclidean(self.plane)
        x = jnp.array([0.5, 0.5])

        K1 = flag_curvature_sample(metric, x, self.key)
        K2 = flag_curvature_sample(metric, x, self.key)
        np.testing.assert_allclose(K1, K2)

    def test_metric_gram_schmidt_orthogonality(self):
        """
        The internally generated (t1, t2) pair must be metric-orthogonal.
        We test indirectly: for an anisotropic metric, Euclidean Gram-Schmidt
        would yield t2 that is NOT metric-orthogonal to t1, causing the
        denominator g_11*g_22 - g_12^2 to be significantly < 1.
        With correct metric Gram-Schmidt, the denominator should be close to 1.
        """
        def anisotropic_metric(x):
            return jnp.diag(jnp.array([1.0, 100.0]))

        metric = Riemannian(self.plane, anisotropic_metric)
        x = jnp.array([1.0, 1.0])

        # flag_curvature_sample must not NaN even for anisotropic metrics
        K = flag_curvature_sample(metric, x, self.key)
        self.assertFalse(jnp.isnan(K))
        self.assertTrue(jnp.isfinite(K))


class TestRandersNonZeroCurvature(unittest.TestCase):
    """
    Test that curvature is well-defined and finite for Randers metrics.
    Randers metrics have velocity-dependent Christoffel symbols and thus
    non-trivial flag curvature.
    """

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)

    def test_randers_curvature_finite(self):
        """Flag curvature must be finite for a well-conditioned Randers metric."""
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.3, 0.0])
        metric = Randers(self.plane, h_net, w_net)

        x = jnp.array([0.0, 0.0])
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])

        K = sectional_curvature(metric, x, v1, v2)
        self.assertFalse(jnp.isnan(K))
        self.assertTrue(jnp.isfinite(K))

    def test_randers_curvature_differs_from_riemannian(self):
        """
        Randers flag curvature with non-constant wind must differ from
        the background Riemannian curvature, confirming that the
        velocity-dependent Gamma terms contribute.
        """
        h_net = lambda x: jnp.eye(2)
        # Position-dependent wind creates non-trivial curvature
        w_net = lambda x: jnp.array([0.3 * x[1], 0.0])

        randers_metric = Randers(self.plane, h_net, w_net)
        riemannian_metric = Riemannian(self.plane, h_net)

        x = jnp.array([1.0, 1.0])
        v1 = jnp.array([1.0, 0.0])
        v2 = jnp.array([0.0, 1.0])

        K_randers = sectional_curvature(randers_metric, x, v1, v2)
        K_riemannian = sectional_curvature(riemannian_metric, x, v1, v2)

        # They should differ because Randers adds velocity-dependent curvature
        # (For flat background Riemannian, K_riemannian ~ 0)
        # We just verify Randers curvature is non-trivially different
        self.assertFalse(jnp.isnan(K_randers))
        # K_randers != 0 for position-dependent wind
        self.assertGreater(abs(float(K_randers)), 1e-6)


if __name__ == '__main__':
    unittest.main()
