"""
Tests for Finsler curvature.

The suite is built around metrics whose curvature is known in closed form, so
that a wrong sign or a swapped index fails rather than merely looking plausible:

    - Euclidean space and a constant Riemannian metric, K = 0.
    - The stereographic round sphere, K = +1.
    - The Poincare half-plane, K = -1.
    - A surface of revolution, K = -1/(1+x^2)^2.
    - ``ProjectivelyFlatRanders``, K = -1/4 — genuinely non-Riemannian, and the
      only case here that can distinguish the two contraction conventions.

Conventions:
    ``jax.numpy`` for traced arrays, ``numpy`` for assertions. Tolerances come
    from ``tests/_precision.tol`` so the file passes in float32 and verifies the
    stronger float64 guarantee under x64.
"""

import unittest
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from _precision import tol

from finslerax.geometry import (
    Euclidean,
    EuclideanSpace,
    ProjectivelyFlatRanders,
    Randers,
    Riemannian,
    curvature_tensor,
    flag_curvature,
    flag_curvature_sample,
    ricci_curvature,
    riemann_curvature_tensor,
    riemannian_curvature,
    sectional_curvature,
)

# Curvature is a 4th-order autodiff chain, so it carries more float32 noise than
# most of the suite; the float64 bound is what pins the mathematics.
CURV_TOL = {"atol32": 3e-3, "rtol32": 3e-3, "atol64": 1e-6, "rtol64": 1e-6}


def sphere_g(x):
    """Unit 2-sphere in stereographic coordinates. K = +1."""
    return (4.0 / (1.0 + jnp.sum(x**2)) ** 2) * jnp.eye(2)


def half_plane_g(x):
    """Poincare half-plane, g = I / y^2. K = -1."""
    return jnp.eye(2) / x[1] ** 2


def revolution_g(x):
    """ds^2 = dx^2 + (1+x^2) dy^2. K = -1/(1+x^2)^2."""
    return jnp.diag(jnp.array([1.0, 1.0 + x[0] ** 2]))


def anisotropic_g(x):
    """A curved, non-diagonal Riemannian metric with no closed-form K."""
    return jnp.array([[1.0 + 0.3 * x[1] ** 2, 0.1 * x[0]], [0.1 * x[0], 1.0]])


def swirl_wind(x):
    """A position-dependent wind, so the Randers metric is genuinely Finslerian."""
    return jnp.array([0.35 * jnp.sin(x[1]), 0.25 * jnp.cos(x[0])])


def fundamental_tensor(metric, x, y):
    """g_ij(x, y) = d^2 E / dy^i dy^j."""
    return jax.hessian(metric.energy, argnums=1)(x, y)


class TestCurvatureTensor(unittest.TestCase):
    """R^i_jk itself: vanishing, antisymmetry, and the Euler identity."""

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)

    def test_euclidean_tensor_vanishes(self):
        """Flat space has an integrable horizontal distribution, so R = 0."""
        R = curvature_tensor(
            Euclidean(self.plane), jnp.array([1.0, 2.0]), jnp.array([1.0, 0.0])
        )
        atol, _ = tol(**CURV_TOL)
        np.testing.assert_allclose(R, jnp.zeros((2, 2, 2)), atol=atol)

    def test_flat_riemannian_tensor_vanishes(self):
        """A constant (but anisotropic) metric is still flat."""
        metric = Riemannian(self.plane, lambda x: jnp.diag(jnp.array([2.0, 3.0])))
        R = curvature_tensor(metric, jnp.array([1.0, 1.0]), jnp.array([1.0, 0.5]))
        atol, _ = tol(**CURV_TOL)
        np.testing.assert_allclose(R, jnp.zeros((2, 2, 2)), atol=atol)

    def test_antisymmetry_in_lower_indices(self):
        """R^i_jk = -R^i_kj, for Finslerian metrics as well as Riemannian ones."""
        cases = [
            Riemannian(self.plane, anisotropic_g),
            Randers(self.plane, anisotropic_g, swirl_wind, wind_mode="raw"),
        ]
        for metric in cases:
            with self.subTest(metric=type(metric).__name__):
                x, y = jnp.array([0.2, -0.4]), jnp.array([1.0, 0.3])
                R = curvature_tensor(metric, x, y)
                scale = float(jnp.max(jnp.abs(R)))
                self.assertGreater(scale, 1e-3, "test metric is too flat to be a test")
                residual = float(jnp.max(jnp.abs(R + jnp.transpose(R, (0, 2, 1)))))
                self.assertLess(residual / scale, 1e-5)

    def test_jacobi_endomorphism_annihilates_the_flagpole(self):
        """R_y(y) = R^i_jk y^j y^k = 0, straight from antisymmetry in j, k."""
        metric = Randers(self.plane, anisotropic_g, swirl_wind, wind_mode="raw")
        x, y = jnp.array([0.2, -0.4]), jnp.array([1.0, 0.3])
        R_y = riemannian_curvature(metric, x, y)
        scale = float(jnp.max(jnp.abs(R_y)))
        self.assertGreater(scale, 1e-3)
        self.assertLess(float(jnp.max(jnp.abs(R_y @ y))) / scale, 1e-5)

    def test_jacobi_endomorphism_is_two_homogeneous(self):
        """R_{cy} = c^2 R_y, since R^i_jk is 1-homogeneous and y appears once."""
        metric = Riemannian(self.plane, anisotropic_g)
        x, y = jnp.array([0.2, -0.4]), jnp.array([1.0, 0.3])
        atol, rtol = tol(**CURV_TOL)
        np.testing.assert_allclose(
            riemannian_curvature(metric, x, 2.0 * y),
            4.0 * riemannian_curvature(metric, x, y),
            atol=atol,
            rtol=rtol,
        )

    def test_berwald_coefficients_satisfy_euler_identity(self):
        """G^i_j y^j = 2 G^i, since the spray is 2-homogeneous in y."""
        metric = Riemannian(self.plane, anisotropic_g)
        x, y = jnp.array([0.3, 0.2]), jnp.array([1.0, -0.7])
        coeffs = jax.jacfwd(metric.spray, argnums=1)(x, y)
        atol, rtol = tol(**CURV_TOL)
        np.testing.assert_allclose(
            coeffs @ y, 2.0 * metric.spray(x, y), atol=atol, rtol=rtol
        )


class TestKnownRiemannianCurvature(unittest.TestCase):
    """Closed-form Riemannian surfaces, magnitude and sign."""

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)
        self.u = jnp.array([1.0, 0.0])
        self.v = jnp.array([0.0, 1.0])

    def test_euclidean_is_flat(self):
        K = sectional_curvature(
            Euclidean(self.plane), jnp.array([0.0, 0.0]), self.u, self.v
        )
        atol, _ = tol(**CURV_TOL)
        np.testing.assert_allclose(K, 0.0, atol=atol)

    def test_stereographic_sphere_is_plus_one(self):
        """The sign here is what pins the contraction convention."""
        metric = Riemannian(self.plane, sphere_g)
        atol, rtol = tol(**CURV_TOL)
        for xv in (0.0, 0.3, 0.7):
            with self.subTest(x=xv):
                K = sectional_curvature(metric, jnp.array([xv, 0.0]), self.u, self.v)
                np.testing.assert_allclose(K, 1.0, atol=atol, rtol=rtol)

    def test_poincare_half_plane_is_minus_one(self):
        metric = Riemannian(self.plane, half_plane_g)
        atol, rtol = tol(**CURV_TOL)
        for xv, yv in ((0.0, 1.0), (0.5, 2.0), (-0.3, 0.7)):
            with self.subTest(x=(xv, yv)):
                K = sectional_curvature(metric, jnp.array([xv, yv]), self.u, self.v)
                np.testing.assert_allclose(K, -1.0, atol=atol, rtol=rtol)

    def test_surface_of_revolution_matches_closed_form(self):
        metric = Riemannian(self.plane, revolution_g)
        atol, rtol = tol(**CURV_TOL)
        for xv in (0.0, 0.5, 1.0):
            with self.subTest(x=xv):
                K = sectional_curvature(metric, jnp.array([xv, 0.0]), self.u, self.v)
                expected = -1.0 / (1.0 + xv**2) ** 2
                np.testing.assert_allclose(K, expected, atol=atol, rtol=rtol)


class TestProjectivelyFlatRanders(unittest.TestCase):
    """The non-Riemannian oracle: constant flag curvature -1/4.

    Every other non-zero curvature assertion in this file is Riemannian, and a
    Riemannian metric cannot distinguish the two contraction conventions from
    each other in the presence of a compensating sign elsewhere. This one can.
    """

    def _metrics(self, dim):
        M = EuclideanSpace(dim=dim)
        zero = jnp.zeros((dim,))
        drift = jnp.array([0.3, *[0.0] * (dim - 1)])
        return [
            ("funk", ProjectivelyFlatRanders(M, a=zero)),
            ("drift", ProjectivelyFlatRanders(M, a=drift)),
            ("drift_neg", ProjectivelyFlatRanders(M, a=drift, epsilon=-1.0)),
        ]

    def test_flag_curvature_is_minus_one_quarter(self):
        """Constant at every point, in every flag, from every flagpole."""
        atol, rtol = tol(**CURV_TOL)
        for dim in (2, 3):
            pad = [0.15] * (dim - 2)
            points = [[0.0] * dim, [0.25, 0.1, *pad], [-0.3, 0.4, *pad]]
            flags = [
                ([1.0, 0.3, *pad], [0.2, 1.0, *pad]),
                ([-0.6, 0.9, *pad], [1.0, 0.1, *pad]),
            ]
            for name, metric in self._metrics(dim):
                for x in points:
                    for y, u in flags:
                        with self.subTest(dim=dim, metric=name, x=x, y=y):
                            K = flag_curvature(
                                metric, jnp.array(x), jnp.array(y), jnp.array(u)
                            )
                            np.testing.assert_allclose(K, -0.25, atol=atol, rtol=rtol)

    def test_ricci_matches_constant_curvature_identity(self):
        """Ric(y) = (n-1) * lambda * F^2(x, y)."""
        atol, rtol = tol(**CURV_TOL)
        for dim in (2, 3):
            pad = [0.15] * (dim - 2)
            for name, metric in self._metrics(dim):
                for x, y in (
                    ([0.0] * dim, [1.0, 0.3, *pad]),
                    ([0.25, 0.1, *pad], [-0.6, 0.9, *pad]),
                ):
                    with self.subTest(dim=dim, metric=name, x=x):
                        x_a, y_a = jnp.array(x), jnp.array(y)
                        ric = ricci_curvature(metric, x_a, y_a)
                        expected = (dim - 1) * (-0.25) * 2.0 * metric.energy(x_a, y_a)
                        np.testing.assert_allclose(ric, expected, atol=atol, rtol=rtol)

    def test_tensor_matches_constant_curvature_coefficients(self):
        """R^i_jk = lambda (delta^i_k l_j - delta^i_j l_k) with l_j = g_jm y^m.

        This is the sharpest statement of the index convention available: it
        fixes which lower slot carries the flagpole, not merely the overall sign.
        """
        atol, rtol = tol(atol32=1e-2, rtol32=1e-2, atol64=1e-6, rtol64=1e-6)
        dim = 2
        metric = ProjectivelyFlatRanders(
            EuclideanSpace(dim=dim), a=jnp.array([0.3, 0.0])
        )
        for x, y in (([0.0, 0.0], [1.0, 0.3]), ([0.25, 0.1], [-0.6, 0.9])):
            with self.subTest(x=x):
                x_a, y_a = jnp.array(x), jnp.array(y)
                R = curvature_tensor(metric, x_a, y_a)
                ell = fundamental_tensor(metric, x_a, y_a) @ y_a
                eye = jnp.eye(dim)
                expected = -0.25 * (
                    jnp.einsum("ik,j->ijk", eye, ell)
                    - jnp.einsum("ij,k->ijk", eye, ell)
                )
                np.testing.assert_allclose(R, expected, atol=atol, rtol=rtol)

    def test_metric_is_not_riemannian(self):
        metric = ProjectivelyFlatRanders(EuclideanSpace(dim=2))
        self.assertFalse(metric.is_riemannian)
        self.assertEqual(metric.flag_curvature_constant, -0.25)

    def test_rejects_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            ProjectivelyFlatRanders(EuclideanSpace(dim=2), epsilon=0.5)


class TestFlagCurvature(unittest.TestCase):
    """Homogeneity, flagpole dependence, and the degenerate-flag guard."""

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)
        self.randers = Randers(self.plane, anisotropic_g, swirl_wind, wind_mode="raw")
        self.x = jnp.array([0.2, -0.4])

    def test_zero_homogeneous_in_both_arguments(self):
        """Scaling the flagpole or the transverse edge must not move K."""
        y, u = jnp.array([1.0, 0.3]), jnp.array([0.2, 1.0])
        base = flag_curvature(self.randers, self.x, y, u)
        atol, rtol = tol(**CURV_TOL)
        for sy, su in ((2.0, 1.0), (1.0, 3.5), (0.5, 0.25)):
            with self.subTest(scale=(sy, su)):
                scaled = flag_curvature(self.randers, self.x, sy * y, su * u)
                np.testing.assert_allclose(scaled, base, atol=atol, rtol=rtol)

    def test_depends_on_the_flagpole_for_a_finsler_metric(self):
        """The reason K takes two arguments: one plane, several values."""
        u = jnp.array([0.0, 1.0])
        values = [
            float(flag_curvature(self.randers, self.x, jnp.array(y), u))
            for y in ([1.0, 0.0], [1.0, 0.5], [1.0, -0.5])
        ]
        spread = max(values) - min(values)
        self.assertGreater(spread, 1e-2, f"expected flagpole dependence, got {values}")

    def test_independent_of_the_flagpole_for_a_riemannian_metric(self):
        """And the reason sectional curvature is well defined when it is."""
        metric = Riemannian(self.plane, anisotropic_g)
        u = jnp.array([0.0, 1.0])
        values = [
            float(flag_curvature(metric, self.x, jnp.array(y), u))
            for y in ([1.0, 0.0], [1.0, 0.5], [1.0, -0.5], [2.0, 0.0])
        ]
        atol, rtol = tol(**CURV_TOL)
        np.testing.assert_allclose(values, values[0], atol=atol, rtol=rtol)

    def test_degenerate_flag_returns_zero(self):
        """Metrically parallel edges span no plane; guard, do not NaN."""
        v = jnp.array([1.0, 0.0])
        K = flag_curvature(Euclidean(self.plane), jnp.array([1.0, 0.0]), v, v)
        self.assertFalse(bool(jnp.isnan(K)))
        np.testing.assert_allclose(K, 0.0)


class TestSectionalCurvatureDegeneration(unittest.TestCase):
    """sectional_curvature must refuse the cases where it is not defined."""

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)
        self.x = jnp.array([1.0, 1.0])
        self.u = jnp.array([1.0, 0.0])
        self.v = jnp.array([0.0, 1.0])

    def test_declared_riemannian_metrics_take_the_direct_path(self):
        for metric in (
            Euclidean(self.plane),
            Riemannian(self.plane, sphere_g),
            Randers(self.plane, anisotropic_g, swirl_wind, use_wind=False),
        ):
            with self.subTest(metric=type(metric).__name__):
                self.assertTrue(metric.is_riemannian)
                K = sectional_curvature(metric, self.x, self.u, self.v)
                self.assertTrue(bool(jnp.isfinite(K)))

    def test_raises_on_a_genuinely_finslerian_metric(self):
        metric = Randers(self.plane, anisotropic_g, swirl_wind, wind_mode="raw")
        self.assertFalse(metric.is_riemannian)
        with self.assertRaises(eqx.EquinoxRuntimeError):
            sectional_curvature(metric, self.x, self.u, self.v)

    def test_check_false_bypasses_the_verification(self):
        """An escape hatch, and it must return the flag curvature at u."""
        metric = Randers(self.plane, anisotropic_g, swirl_wind, wind_mode="raw")
        K = sectional_curvature(metric, self.x, self.u, self.v, check=False)
        expected = flag_curvature(metric, self.x, self.u, self.v)
        np.testing.assert_allclose(K, expected)

    def test_undeclared_riemannian_metric_passes_the_check(self):
        """A metric that is Riemannian in fact but does not say so still works."""

        class QuietRiemannian(Riemannian):
            @property
            def is_riemannian(self) -> bool:
                return False

        metric = QuietRiemannian(self.plane, sphere_g)
        K = sectional_curvature(metric, jnp.array([0.3, 0.0]), self.u, self.v)
        atol, rtol = tol(**CURV_TOL)
        np.testing.assert_allclose(K, 1.0, atol=atol, rtol=rtol)


class TestJaxTransforms(unittest.TestCase):
    """jit, vmap and grad through the whole 4th-order chain."""

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)
        self.metric = Riemannian(self.plane, revolution_g)
        self.x = jnp.array([0.5, 0.0])
        self.u = jnp.array([1.0, 0.0])
        self.v = jnp.array([0.0, 1.0])

    def test_jit_matches_eager(self):
        for fn in (flag_curvature, sectional_curvature):
            with self.subTest(fn=fn.__name__):
                eager = fn(self.metric, self.x, self.u, self.v)
                compiled = jax.jit(fn, static_argnums=0)(
                    self.metric, self.x, self.u, self.v
                )
                atol, rtol = tol(**CURV_TOL)
                np.testing.assert_allclose(eager, compiled, atol=atol, rtol=rtol)

    def test_curvature_tensor_jit(self):
        R = jax.jit(curvature_tensor, static_argnums=0)(self.metric, self.x, self.u)
        self.assertEqual(R.shape, (2, 2, 2))
        self.assertFalse(bool(jnp.any(jnp.isnan(R))))

    def test_vmap_over_positions(self):
        xs = jnp.stack([jnp.array([xv, 0.0]) for xv in (0.0, 0.5, 1.0)])
        Ks = jax.vmap(lambda x: flag_curvature(self.metric, x, self.u, self.v))(xs)
        expected = [-1.0 / (1.0 + xv**2) ** 2 for xv in (0.0, 0.5, 1.0)]
        atol, rtol = tol(**CURV_TOL)
        np.testing.assert_allclose(Ks, expected, atol=atol, rtol=rtol)

    def test_grad_is_finite(self):
        """The chain stays differentiable, which is the point of building it so."""

        def flag_at(x):
            return flag_curvature(self.metric, x, self.u, self.v)

        def ricci_at(x):
            return ricci_curvature(self.metric, x, self.u)

        for name, fn in (("flag", flag_at), ("ricci", ricci_at)):
            with self.subTest(fn=name):
                g = jax.grad(fn)(self.x)
                self.assertFalse(bool(jnp.any(jnp.isnan(g))))


class TestDeprecatedNames(unittest.TestCase):
    """The 1.0 surface keeps working, loudly, until 2.0."""

    def setUp(self):
        self.plane = EuclideanSpace(dim=2)

    def test_riemann_curvature_tensor_warns_and_delegates(self):
        metric = Riemannian(self.plane, revolution_g)
        x, y = jnp.array([0.5, 0.0]), jnp.array([1.0, 0.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = riemann_curvature_tensor(metric, x, y)
        self.assertTrue(any(w.category is DeprecationWarning for w in caught))
        np.testing.assert_allclose(legacy, curvature_tensor(metric, x, y))

    def test_flag_curvature_sample_warns_and_is_finite(self):
        metric = Riemannian(self.plane, revolution_g)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            K = flag_curvature_sample(
                metric, jnp.array([0.5, 0.0]), jax.random.PRNGKey(0)
            )
        self.assertTrue(any(w.category is DeprecationWarning for w in caught))
        self.assertTrue(bool(jnp.isfinite(K)))


if __name__ == "__main__":
    unittest.main()
