"""
Tests for Berwald parallel transport.

Note on conventions:
- `jax.numpy` (jnp) is used for all arrays that are intended to be JAX-traced.
- `numpy` (np) is used strictly for non-traced assertions and testing utilities.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

# Ensure precision for geometric drift checks
from ham.geometry import EuclideanSpace, Sphere
from ham.geometry.transport import BerwaldConnection
from ham.geometry.zoo import Euclidean, Randers, Riemannian
from ham.solvers import ExponentialMap


class TestTransport(unittest.TestCase):
    def setUp(self):
        # Use real manifold implementations from surfaces.py
        self.plane = EuclideanSpace(dim=2)
        self.sphere = Sphere(intrinsic_dim=2, radius=1.0)
        self.key = jax.random.PRNGKey(42)

    def test_euclidean_flat_invariance(self):
        """
        In flat Euclidean space, parallel transport is trivial.
        A vector transported along any path remains constant in coordinates.
        """
        metric = Euclidean(self.plane)

        # Path: Circle in the plane
        t = jnp.linspace(0, 2 * jnp.pi, 50)
        path_x = jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)
        path_v = jnp.stack([-jnp.sin(t), jnp.cos(t)], axis=1)

        vec_start = jnp.array([1.0, 0.0])

        vecs = BerwaldConnection(metric).parallel_transport(path_x, path_v, vec_start)

        expected = jnp.tile(vec_start, (len(path_x), 1))
        np.testing.assert_allclose(vecs, expected, atol=1e-5)

    def test_connection_coefficients_euclidean(self):
        """Euclidean connection coefficients must be identically zero."""
        metric = Euclidean(self.plane)
        conn = BerwaldConnection(metric)
        x = jnp.array([1.0, 2.0])
        v = jnp.array([3.0, 4.0])
        coeff = conn.connection_coefficients(x, v)
        np.testing.assert_allclose(coeff, jnp.zeros((2, 2)), atol=1e-5)

    def test_jit_vmap_grad_compatibility(self):
        """Test JAX transforms (jit, vmap, grad) over the connection object."""
        metric = Euclidean(self.plane)
        conn = BerwaldConnection(metric)

        x_batch = jax.random.normal(self.key, (10, 2))
        v_batch = jax.random.normal(self.key, (10, 2))

        # 1. vmap over batch
        vmap_coeff = jax.vmap(conn.connection_coefficients)
        coeffs = vmap_coeff(x_batch, v_batch)
        self.assertEqual(coeffs.shape, (10, 2, 2))

        # 2. jit over transport
        jit_transport = jax.jit(conn.parallel_transport)
        vec_start = jnp.array([1.0, 0.0])
        res = jit_transport(x_batch, v_batch, vec_start)
        self.assertEqual(res.shape, (10, 2))

        # 3. grad check (differentiability w.r.t vec_start)
        def transport_loss(v0):
            vecs = conn.parallel_transport(x_batch, v_batch, v0)
            return jnp.sum(vecs[-1] ** 2)

        grad_fn = jax.grad(transport_loss)
        g = grad_fn(vec_start)
        self.assertFalse(jnp.any(jnp.isnan(g)))
        # For Euclidean, d/dv0 ||v0||^2 = 2*v0
        np.testing.assert_allclose(g, 2.0 * vec_start, atol=1e-5)

    def test_coefficients_zero_velocity(self):
        """Ensure connection coefficients do not NaN at v=0."""
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5, 0.0])
        metric = Randers(self.plane, h_net, w_net)
        conn = BerwaldConnection(metric)

        x = jnp.array([1.0, 1.0])
        v_zero = jnp.array([0.0, 0.0])

        coeff = conn.connection_coefficients(x, v_zero)
        self.assertFalse(jnp.any(jnp.isnan(coeff)))

    def test_transport_degenerate(self):
        """Test that transport handles single-point paths gracefully."""
        metric = Euclidean(self.plane)
        path_x = jnp.array([[1.0, 2.0]])
        path_v = jnp.array([[0.0, 0.0]])
        vec_start = jnp.array([1.0, 0.0])

        vecs = BerwaldConnection(metric).parallel_transport(path_x, path_v, vec_start)
        self.assertEqual(vecs.shape, (1, 2))
        np.testing.assert_allclose(vecs[0], vec_start)

    def test_coefficients_are_homogeneous_degree_one(self):
        """
        G^i_j is the velocity gradient of a degree-two-homogeneous spray, so it
        is homogeneous of degree one: G^i_j(x, s*y) = s * G^i_j(x, y).

        Euler's theorem then gives G^i_j(x, y) y^j = 2 G^i(x, y), which is what
        makes a geodesic auto-parallel.
        """
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0])
        metric = Randers(self.plane, h_net, w_net, wind_mode="raw")
        conn = BerwaldConnection(metric)

        x = jnp.array([1.0, 0.5])
        y = jnp.array([0.7, 1.3])

        base = conn.connection_coefficients(x, y)
        scaled = conn.connection_coefficients(x, 3.0 * y)
        np.testing.assert_allclose(scaled, 3.0 * base, rtol=1e-5)

        # Euler: contracting with y recovers twice the spray.
        np.testing.assert_allclose(
            jnp.einsum("ij,j->i", base, y), 2.0 * metric.spray(x, y), rtol=1e-4
        )

    def test_coefficients_linear_in_y_iff_berwald(self):
        """
        G^i_j is linear in y exactly on Berwald manifolds. A position-dependent
        Riemannian metric is one, so there G^i_j(x, y) = Gamma^i_jk(x) y^k and
        additivity holds. A Randers metric with non-parallel wind is not, and
        additivity must fail — this is what stops the coefficients from being
        Christoffel symbols of any linear connection on M.
        """
        from ham.geometry.metric import FinslerMetric

        class DiagMetric(FinslerMetric):
            def metric_fn(self, x, v):
                # g = diag(1, 1 + x[0]^2)
                g_diag = jnp.array([1.0, 1.0 + x[0] ** 2])
                return jnp.sqrt(jnp.sum(g_diag * v**2))

        x = jnp.array([1.0, 0.0])  # x[0]=1 → g22 = 2
        y1 = jnp.array([1.0, 1.0])
        y2 = jnp.array([0.4, -0.9])

        riemannian = BerwaldConnection(DiagMetric(self.plane))
        coeff = riemannian.connection_coefficients(x, y1)

        # Position-dependent metric ⇒ genuinely non-zero coefficients.
        self.assertGreater(float(jnp.max(jnp.abs(coeff))), 1e-4)

        # Riemannian ⇒ Berwald ⇒ linear in y.
        np.testing.assert_allclose(
            riemannian.connection_coefficients(x, y1 + y2),
            coeff + riemannian.connection_coefficients(x, y2),
            atol=1e-5,
        )

        # Randers with non-parallel wind ⇒ not Berwald ⇒ additivity fails.
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0])
        randers = BerwaldConnection(Randers(self.plane, h_net, w_net, wind_mode="raw"))
        residual = (
            randers.connection_coefficients(x, y1 + y2)
            - randers.connection_coefficients(x, y1)
            - randers.connection_coefficients(x, y2)
        )
        self.assertGreater(float(jnp.max(jnp.abs(residual))), 1e-3)

    def test_berwald_coefficients_torsion_free(self):
        """
        G^i_jk is symmetric in j, k. Guaranteed by Schwarz's theorem on the
        double jacfwd, but verified explicitly as a guard against refactors.
        """
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0])
        conn = BerwaldConnection(Randers(self.plane, h_net, w_net, wind_mode="raw"))

        gamma = conn.berwald_coefficients(jnp.array([1.0, 0.5]), jnp.array([0.7, 1.3]))
        np.testing.assert_allclose(gamma, jnp.transpose(gamma, (0, 2, 1)), atol=1e-5)

    def test_berwald_coefficients_y_independent_iff_berwald(self):
        """
        Independence of y *is* the definition of a Berwald manifold. A
        position-dependent Riemannian metric qualifies; Randers with sheared
        wind does not. This is the direct dual of the additivity test above.
        """
        from ham.geometry.metric import FinslerMetric

        class DiagMetric(FinslerMetric):
            def metric_fn(self, x, v):
                g_diag = jnp.array([1.0, 1.0 + x[0] ** 2])
                return jnp.sqrt(jnp.sum(g_diag * v**2))

        x = jnp.array([1.0, 0.0])
        y1 = jnp.array([1.0, 1.0])
        y2 = jnp.array([0.4, -0.9])

        riemannian = BerwaldConnection(DiagMetric(self.plane))
        np.testing.assert_allclose(
            riemannian.berwald_coefficients(x, y1),
            riemannian.berwald_coefficients(x, y2),
            atol=1e-4,
        )

        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0])
        randers = BerwaldConnection(Randers(self.plane, h_net, w_net, wind_mode="raw"))
        spread = jnp.max(
            jnp.abs(
                randers.berwald_coefficients(x, y1)
                - randers.berwald_coefficients(x, y2)
            )
        )
        self.assertGreater(float(spread), 1e-3)

    def test_berwald_coefficients_poincare_levi_civita(self):
        """
        On the Poincare half-plane, G^i_jk must reproduce the analytic
        Levi-Civita symbols. This is real coverage of the spray's second
        velocity derivative against closed-form values.
        """
        from ham.geometry.metric import FinslerMetric
        from ham.utils.math import safe_norm

        class PoincareMetric(FinslerMetric):
            """Poincare half-plane metric: F(x, v) = ||v|| / y."""

            def metric_fn(self, x, v):
                y = jnp.maximum(x[1], 1e-10)
                return safe_norm(v) / y

        conn = BerwaldConnection(PoincareMetric(self.plane))
        gamma = conn.berwald_coefficients(jnp.array([0.0, 2.0]), jnp.array([1.0, 1.0]))

        y_val = 2.0
        # Gamma^1_12 = Gamma^1_21 = -1/y, Gamma^2_11 = 1/y, Gamma^2_22 = -1/y.
        # Tolerance 1e-3: the spray carries a Tikhonov term (spray_reg).
        np.testing.assert_allclose(gamma[0, 0, 1], -1.0 / y_val, atol=1e-3)
        np.testing.assert_allclose(gamma[0, 1, 0], -1.0 / y_val, atol=1e-3)
        np.testing.assert_allclose(gamma[1, 0, 0], 1.0 / y_val, atol=1e-3)
        np.testing.assert_allclose(gamma[1, 1, 1], -1.0 / y_val, atol=1e-3)
        # The remaining independent symbols vanish.
        np.testing.assert_allclose(gamma[0, 0, 0], 0.0, atol=1e-3)
        np.testing.assert_allclose(gamma[1, 0, 1], 0.0, atol=1e-3)

    def test_riemannian_sphere_isometry(self):
        """
        Riemannian transport on a Sphere (via projection-based Berwald).
        MUST preserve the norm (isometry) and tangency.

        Uses a non-degenerate initial vector (1, 0, 0) that is NOT
        orthogonal to the path plane, forcing the projection to do
        real work at each step.
        """

        def identity_metric(x):
            return jnp.eye(3)

        metric = Riemannian(self.sphere, identity_metric)

        # Path: Quarter circle (North Pole -> Equator) in xz-plane
        theta = jnp.linspace(0, jnp.pi / 2, 40)
        path_x = jnp.stack(
            [jnp.sin(theta), jnp.zeros_like(theta), jnp.cos(theta)], axis=1
        )
        path_v = jnp.stack(
            [jnp.cos(theta), jnp.zeros_like(theta), -jnp.sin(theta)], axis=1
        )

        # Non-degenerate vector: lies IN the path plane, requires projection
        vec_start = jnp.array([1.0, 0.0, 0.0])
        # Project to tangent space at north pole: remove radial component
        vec_start = self.sphere.to_tangent(path_x[0], vec_start)

        vecs = BerwaldConnection(metric).parallel_transport(path_x, path_v, vec_start)

        # 1. Norm Preservation (Euler drift is O(1/N); N=40 gives ~3% max drift)
        norms = jnp.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, jnp.full_like(norms, norms[0]), atol=5e-2)

        # 2. Tangency: <v, x> = 0 at each point
        dots = jnp.sum(vecs * path_x, axis=1)
        np.testing.assert_allclose(dots, jnp.zeros_like(dots), atol=1e-3)

    def test_randers_horizontal_transport_preserves_norm(self):
        """
        Horizontal parallel translation preserves the Finsler norm.

        This is the defining property of the canonical Finsler translation, and
        the reason holonomy acts on the indicatrix: F is constant along a
        horizontal curve even though the wind W(x) varies along the path, and
        even though the transported vector's Euclidean length does not.
        """
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0])
        metric = Randers(self.plane, h_net, w_net, wind_mode="raw")

        y = jnp.linspace(0, 1, 400)
        path_x = jnp.stack([jnp.zeros_like(y), y], axis=1)
        path_v = jnp.stack([jnp.zeros_like(y), jnp.ones_like(y)], axis=1)
        vec_start = jnp.array([1.0, 0.0])

        conn = BerwaldConnection(metric)
        vecs = conn.parallel_transport(path_x, path_v, vec_start)

        norms = jax.vmap(metric.metric_fn)(path_x, vecs)
        np.testing.assert_allclose(norms, jnp.full_like(norms, norms[0]), rtol=2e-3)

        # The coordinate vector genuinely moves; the norm is what is held fixed.
        self.assertGreater(float(jnp.linalg.norm(vecs[-1] - vec_start)), 0.1)

    def test_horizontal_transport_is_homogeneous(self):
        """
        Horizontal translation is positively homogeneous of degree one:
        P(lambda * Y) = lambda * P(Y). This is what lets it be read as a map
        between indicatrices.
        """
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0])
        metric = Randers(self.plane, h_net, w_net, wind_mode="raw")

        y = jnp.linspace(0, 1, 100)
        path_x = jnp.stack([jnp.zeros_like(y), y], axis=1)
        path_v = jnp.stack([jnp.zeros_like(y), jnp.ones_like(y)], axis=1)

        conn = BerwaldConnection(metric)
        v0 = jnp.array([1.0, 0.3])
        single = conn.parallel_transport(path_x, path_v, v0)[-1]
        scaled = conn.parallel_transport(path_x, path_v, 2.0 * v0)[-1]

        np.testing.assert_allclose(scaled, 2.0 * single, rtol=1e-5)

    def test_geodesic_is_autoparallel(self):
        """
        A geodesic is autoparallel: transporting its own initial velocity along
        it reproduces the velocity field.

        This pins the sign and index convention of the horizontal equation
        against the geodesic equation itself. Euler's theorem on the
        degree-two-homogeneous spray gives N^i_j(x, y) y^j = 2 G^i(x, y), so
        Y = gamma_dot solves the horizontality condition exactly when gamma
        solves gamma_ddot + 2G = 0.
        """
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.3 * jnp.sin(x[1]), 0.2])
        metric = Randers(self.plane, h_net, w_net, wind_mode="raw")

        path_x, path_v = ExponentialMap(max_steps=200).trace(
            metric, jnp.array([0.0, 0.0]), jnp.array([1.0, 0.4]), t_max=1.0
        )

        conn = BerwaldConnection(metric)
        transported = conn.parallel_transport(path_x, path_v, path_v[0])

        np.testing.assert_allclose(transported, path_v, rtol=2e-2, atol=2e-2)

    def test_sphere_holonomy(self):
        """
        Verify parallel transport around a latitude circle on S^2
        reproduces a known holonomy angle.

        Note: Our implementation uses g(x) = I_3 (ambient Euclidean), so
        Gamma^i_jk = 0 and the transport is entirely projection-based.
        This correctly computes the exact Levi-Civita connection via the Gauss
        equation. The true holonomy rotation is the solid angle 2*pi*(1-cos(theta)).
        However, the local frame (phi_hat, theta_hat) used here is negatively
        oriented, causing the measured angle to appear as -Omega, which is exactly
        -2*pi*(1-cos(theta)) ≡ 2*pi*cos(theta) modulo 2*pi.
        """

        def identity_metric(x):
            return jnp.eye(3)

        metric = Riemannian(self.sphere, identity_metric)

        theta = jnp.pi / 4.0
        t = jnp.linspace(0, 2 * jnp.pi, 200)

        path_x = jnp.stack(
            [
                jnp.sin(theta) * jnp.cos(t),
                jnp.sin(theta) * jnp.sin(t),
                jnp.full_like(t, jnp.cos(theta)),
            ],
            axis=1,
        )

        path_v = jnp.stack(
            [
                -jnp.sin(theta) * jnp.sin(t),
                jnp.sin(theta) * jnp.cos(t),
                jnp.zeros_like(t),
            ],
            axis=1,
        )

        vec_start = jnp.array([0.0, 1.0, 0.0])

        vecs = BerwaldConnection(metric).parallel_transport(path_x, path_v, vec_start)
        vec_end = vecs[-1]

        # Tangent plane basis at the start/end point (sin(theta), 0, cos(theta))
        phi_hat = jnp.array([0.0, 1.0, 0.0])
        theta_hat = jnp.array([jnp.cos(theta), 0.0, -jnp.sin(theta)])

        v_end_phi = jnp.dot(vec_end, phi_hat)
        v_end_theta = jnp.dot(vec_end, theta_hat)

        angle = jnp.arctan2(v_end_theta, v_end_phi)

        # The physical rotation is the enclosed solid angle Omega = 2*pi*(1 - cos(theta)).
        # However, our local tangent frame (phi_hat, theta_hat) has a negative orientation
        # relative to the outward normal: (phi_hat x theta_hat) = -r_hat.
        # Thus, the physical rotation Omega appears as -Omega in this frame.
        # Modulo 2*pi, we have -2*pi*(1 - cos(theta)) = 2*pi*cos(theta) - 2*pi ≡ 2*pi*cos(theta).
        expected_angle = 2 * jnp.pi * jnp.cos(theta)

        # We need to account for the fact that angles near 0 and 2pi might wrap differently
        # so we compare the complex phases directly.
        np.testing.assert_allclose(jnp.exp(1j * angle), jnp.exp(1j * expected_angle), atol=1e-2)

    def test_integrator_convergence_order(self):
        """
        Test that the integrator converges at O(1/N) rate (1st-order forward Euler).
        """

        def identity_metric(x):
            return jnp.eye(3)

        metric = Riemannian(self.sphere, identity_metric)
        conn = BerwaldConnection(metric)

        theta = jnp.pi / 4.0
        vec_start = jnp.array([0.0, 1.0, 0.0])

        # Projection-based transport angle
        expected_angle = 2 * jnp.pi * jnp.cos(theta)

        phi_hat = jnp.array([0.0, 1.0, 0.0])
        theta_hat = jnp.array([jnp.cos(theta), 0.0, -jnp.sin(theta)])

        def run_transport(N):
            t = jnp.linspace(0, 2 * jnp.pi, N)
            path_x = jnp.stack(
                [
                    jnp.sin(theta) * jnp.cos(t),
                    jnp.sin(theta) * jnp.sin(t),
                    jnp.full_like(t, jnp.cos(theta)),
                ],
                axis=1,
            )
            path_v = jnp.stack(
                [
                    -jnp.sin(theta) * jnp.sin(t),
                    jnp.sin(theta) * jnp.cos(t),
                    jnp.zeros_like(t),
                ],
                axis=1,
            )

            vec_end = conn.parallel_transport(path_x, path_v, vec_start)[-1]

            v_end_phi = jnp.dot(vec_end, phi_hat)
            v_end_theta = jnp.dot(vec_end, theta_hat)

            # Exact analytic components
            exact_phi = jnp.cos(expected_angle)
            exact_theta = jnp.sin(expected_angle)

            # Euclidean error in tangent plane
            return jnp.sqrt(
                (v_end_phi - exact_phi) ** 2 + (v_end_theta - exact_theta) ** 2
            )

        error_20 = run_transport(20)
        error_40 = run_transport(40)
        error_80 = run_transport(80)

        # Ratio ~ 2.0 for 1st-order convergence
        ratio_20_40 = error_20 / error_40
        ratio_40_80 = error_40 / error_80

        self.assertTrue(
            1.5 < ratio_20_40 < 2.5,
            f"1st-order ratio expected near 2, got {ratio_20_40}",
        )
        self.assertTrue(
            1.5 < ratio_40_80 < 2.5,
            f"1st-order ratio expected near 2, got {ratio_40_80}",
        )

    def test_poincare_half_plane_transport(self):
        """
        Transport a vector along a vertical geodesic in the Poincaré half-plane.

        This is the key test that exercises the Berwald connection ODE with
        analytically non-zero Christoffel symbols. Unlike the sphere tests
        (where g(x)=I_3 gives Gamma=0), here the connection genuinely drives
        the transport.

        Setup:
            Metric: ds^2 = (dx^2 + dy^2) / y^2  (constant negative curvature)
            Christoffel symbols:
                Gamma^1_12 = Gamma^1_21 = -1/y
                Gamma^2_11 = 1/y,  Gamma^2_22 = -1/y

            Path: vertical geodesic y(t) = e^t, x(t) = 0, t in [0, 1]
            Velocity: v(t) = (0, e^t)

        Analytic solution:
            The transport ODE with v = (0, y) gives dX/dt = X (both components),
            so X(t) = X(0) * e^t. For X(0) = (1, 0): X(1) = (e, 0).

            The metric norm ||X||_g = ||X|| / y = e^t / e^t = 1 (preserved).

        Critical check:
            If Gamma were incorrectly zero, the vector would stay at (1, 0),
            and the metric norm would drop to 1/e ≈ 0.368 (wrong).
        """
        from ham.geometry.metric import FinslerMetric
        from ham.utils.math import safe_norm

        class PoincareMetric(FinslerMetric):
            """Poincaré half-plane metric: F(x, v) = ||v|| / y."""

            def metric_fn(self, x, v):
                y = jnp.maximum(x[1], 1e-10)  # Ensure y > 0
                return safe_norm(v) / y

        plane = EuclideanSpace(dim=2)
        metric = PoincareMetric(plane)
        conn = BerwaldConnection(metric)

        # --- 1. Verify the connection coefficients analytically ---
        x_test = jnp.array([0.0, 2.0])  # y = 2
        v_test = jnp.array([1.0, 1.0])
        coeff = conn.connection_coefficients(x_test, v_test)

        y_val = 2.0
        # The half-plane is Riemannian, hence Berwald, so G^i_j = Gamma^i_jk y^k
        # with Gamma^1_12 = Gamma^1_21 = -1/y, Gamma^2_11 = 1/y, Gamma^2_22 = -1/y.
        # Contracting with v = (1, 1) gives:
        expected = jnp.array(
            [
                [-v_test[1] / y_val, -v_test[0] / y_val],
                [v_test[0] / y_val, -v_test[1] / y_val],
            ]
        )
        # Note: tolerance is 1e-3 due to Tikhonov regularization in spray (spray_reg)
        np.testing.assert_allclose(coeff, expected, atol=1e-3)

        # --- 2. Transport along vertical geodesic ---
        N = 200
        t = jnp.linspace(0, 1, N)
        path_x = jnp.stack([jnp.zeros(N), jnp.exp(t)], axis=1)  # (0, e^t)
        path_v = jnp.stack([jnp.zeros(N), jnp.exp(t)], axis=1)  # (0, e^t)

        vec_start = jnp.array([1.0, 0.0])
        vecs = conn.parallel_transport(path_x, path_v, vec_start)

        # Analytic: X(t) = (e^t, 0)
        analytic_vecs = jnp.stack([jnp.exp(t), jnp.zeros(N)], axis=1)

        # The transported vector should grow as e^t in coordinates
        np.testing.assert_allclose(vecs, analytic_vecs, rtol=5e-2)

        # Final vector should be approximately (e, 0)
        np.testing.assert_allclose(vecs[-1], jnp.array([jnp.e, 0.0]), rtol=5e-2)

        # --- 3. Metric norm preservation ---
        # ||X||_g = ||X|| / y = e^t / e^t = 1 at every point
        metric_norms = jax.vmap(metric.metric_fn)(path_x, vecs)
        np.testing.assert_allclose(metric_norms, jnp.ones(N), atol=5e-2)

        # --- 4. Verify that Gamma=0 would be WRONG ---
        # If Gamma were 0, the vector would stay at (1, 0)
        # and the norm at the end would be 1/e ≈ 0.368, not 1.0
        naive_norm = metric.metric_fn(path_x[-1], vec_start)  # ||[1,0]||/e
        self.assertLess(
            float(naive_norm),
            0.5,
            "Sanity check: without transport, norm should drop significantly",
        )


if __name__ == "__main__":
    unittest.main()
