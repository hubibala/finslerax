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
# config.update("jax_enable_x64", True)
from ham.geometry import EuclideanSpace, Sphere
from ham.geometry.transport import BerwaldConnection
from ham.geometry.zoo import Euclidean, Randers, Riemannian


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
        t = jnp.linspace(0, 2*jnp.pi, 50)
        path_x = jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)
        path_v = jnp.stack([-jnp.sin(t), jnp.cos(t)], axis=1)

        vec_start = jnp.array([1.0, 0.0])

        vecs = BerwaldConnection(metric).parallel_transport(path_x, path_v, vec_start)

        expected = jnp.tile(vec_start, (len(path_x), 1))
        np.testing.assert_allclose(vecs, expected, atol=1e-5)

    def test_christoffel_symbols_euclidean(self):
        """Euclidean Christoffel symbols must be identically zero."""
        metric = Euclidean(self.plane)
        conn = BerwaldConnection(metric)
        x = jnp.array([1.0, 2.0])
        v = jnp.array([3.0, 4.0])
        gamma = conn.christoffel_symbols(x, v)
        np.testing.assert_allclose(gamma, jnp.zeros((2, 2, 2)), atol=1e-5)

    def test_jit_vmap_grad_compatibility(self):
        """Test JAX transforms (jit, vmap, grad) over the connection object."""
        metric = Euclidean(self.plane)
        conn = BerwaldConnection(metric)

        x_batch = jax.random.normal(self.key, (10, 2))
        v_batch = jax.random.normal(self.key, (10, 2))

        # 1. vmap over batch
        vmap_gamma = jax.vmap(conn.christoffel_symbols)
        gammas = vmap_gamma(x_batch, v_batch)
        self.assertEqual(gammas.shape, (10, 2, 2, 2))

        # 2. jit over transport
        jit_transport = jax.jit(conn.parallel_transport)
        vec_start = jnp.array([1.0, 0.0])
        res = jit_transport(x_batch, v_batch, vec_start)
        self.assertEqual(res.shape, (10, 2))

        # 3. grad check (differentiability w.r.t vec_start)
        def transport_loss(v0):
            vecs = conn.parallel_transport(x_batch, v_batch, v0)
            return jnp.sum(vecs[-1]**2)

        grad_fn = jax.grad(transport_loss)
        g = grad_fn(vec_start)
        self.assertFalse(jnp.any(jnp.isnan(g)))
        # For Euclidean, d/dv0 ||v0||^2 = 2*v0
        np.testing.assert_allclose(g, 2.0 * vec_start, atol=1e-5)


    def test_christoffel_zero_velocity(self):
        """Ensure connection coefficients do not NaN at v=0."""
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5, 0.0])
        metric = Randers(self.plane, h_net, w_net)
        conn = BerwaldConnection(metric)

        x = jnp.array([1.0, 1.0])
        v_zero = jnp.array([0.0, 0.0])

        gamma = conn.christoffel_symbols(x, v_zero)
        self.assertFalse(jnp.any(jnp.isnan(gamma)))

    def test_transport_degenerate(self):
        """Test that transport handles single-point paths gracefully."""
        metric = Euclidean(self.plane)
        path_x = jnp.array([[1.0, 2.0]])
        path_v = jnp.array([[0.0, 0.0]])
        vec_start = jnp.array([1.0, 0.0])

        vecs = BerwaldConnection(metric).parallel_transport(path_x, path_v, vec_start)
        self.assertEqual(vecs.shape, (1, 2))
        np.testing.assert_allclose(vecs[0], vec_start)


    def test_christoffel_torsion_free(self):
        """
        Berwald connection must be torsion-free: Gamma^i_jk = Gamma^i_kj.
        This is guaranteed by Schwarz's theorem on the double jacfwd,
        but we verify it explicitly as a guard against future refactors.
        """
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0])
        metric = Randers(self.plane, h_net, w_net)
        conn = BerwaldConnection(metric)

        x = jnp.array([1.0, 0.5])
        v = jnp.array([0.7, 1.3])

        gamma = conn.christoffel_symbols(x, v)

        # Check symmetry in last two indices: gamma[i,j,k] == gamma[i,k,j]
        np.testing.assert_allclose(gamma, jnp.transpose(gamma, (0, 2, 1)), atol=1e-5)

    def test_christoffel_nonzero_curved_riemannian(self):
        """
        Verify that a position-dependent Riemannian metric produces
        analytically non-zero Christoffel symbols.
        
        Uses metric g = diag(1, 1+x^2) on R^2.
        For this metric: Gamma^2_11 = x / (1+x^2) (non-zero when x != 0).
        """
        from ham.geometry.metric import FinslerMetric

        class DiagMetric(FinslerMetric):
            def metric_fn(self, x, v):
                # g = diag(1, 1 + x[0]^2)
                g_diag = jnp.array([1.0, 1.0 + x[0]**2])
                return jnp.sqrt(jnp.sum(g_diag * v**2))

        metric = DiagMetric(self.plane)
        conn = BerwaldConnection(metric)

        x = jnp.array([1.0, 0.0])  # x[0]=1 → g22 = 2
        v = jnp.array([1.0, 1.0])

        gamma = conn.christoffel_symbols(x, v)

        # The tensor should be non-trivially non-zero
        max_abs = jnp.max(jnp.abs(gamma))
        self.assertGreater(float(max_abs), 1e-4,
                           "Christoffel symbols should be non-zero for position-dependent metric")

        # Still torsion-free
        np.testing.assert_allclose(gamma, jnp.transpose(gamma, (0, 2, 1)), atol=1e-5)


    def test_riemannian_sphere_isometry(self):
        """
        Riemannian transport on a Sphere (via projection-based Berwald).
        MUST preserve the norm (isometry) and tangency.
        
        Uses a non-degenerate initial vector (1, 0, 0) that is NOT
        orthogonal to the path plane, forcing the projection to do
        real work at each step.
        """
        def identity_metric(x): return jnp.eye(3)
        metric = Riemannian(self.sphere, identity_metric)

        # Path: Quarter circle (North Pole -> Equator) in xz-plane
        theta = jnp.linspace(0, jnp.pi/2, 40)
        path_x = jnp.stack([jnp.sin(theta), jnp.zeros_like(theta), jnp.cos(theta)], axis=1)
        path_v = jnp.stack([jnp.cos(theta), jnp.zeros_like(theta), -jnp.sin(theta)], axis=1)

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


    def test_randers_norm_drift(self):
        """
        Verifies that Berwald transport in a Randers space is NOT an isometry.
        The Finsler norm of the transported vector changes because the
        wind W(x) varies along the path.
        """
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0])
        metric = Randers(self.plane, h_net, w_net)

        y = jnp.linspace(0, 1, 20)
        path_x = jnp.stack([jnp.zeros_like(y), y], axis=1)
        path_v = jnp.stack([jnp.zeros_like(y), jnp.ones_like(y)], axis=1)
        vec_start = jnp.array([1.0, 0.0])

        conn = BerwaldConnection(metric)
        vecs_randers = conn.parallel_transport(path_x, path_v, vec_start)

        norms_randers = jax.vmap(metric.metric_fn)(path_x, vecs_randers)
        initial_norm = norms_randers[0]
        final_norm = norms_randers[-1]

        self.assertNotAlmostEqual(float(initial_norm), float(final_norm), places=3)

    def test_randers_velocity_dependence(self):
        """
        Verify that Randers Berwald transport depends on velocity.
        
        We compare Randers transport against a reference Riemannian transport
        on the SAME path geometry. For Riemannian metrics, the Berwald 
        connection Gamma(x) is velocity-independent, so doubling v with 
        the same discrete dt is purely a rescaling artifact. For Randers,
        the difference must exceed the Riemannian difference.
        """
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0])
        randers_metric = Randers(self.plane, h_net, w_net)
        riemannian_metric = Riemannian(self.plane, h_net)

        vec_start = jnp.array([1.0, 0.0])

        y1 = jnp.linspace(0, 1, 20)
        path_x1 = jnp.stack([jnp.zeros_like(y1), y1], axis=1)
        path_v1 = jnp.stack([jnp.zeros_like(y1), jnp.ones_like(y1)], axis=1)
        path_v2 = path_v1 * 2.0  # Double speed, same geometry

        # Randers transport at two speeds
        randers_conn = BerwaldConnection(randers_metric)
        vecs_randers_1 = randers_conn.parallel_transport(path_x1, path_v1, vec_start)
        vecs_randers_2 = randers_conn.parallel_transport(path_x1, path_v2, vec_start)
        diff_randers = jnp.linalg.norm(vecs_randers_1[-1] - vecs_randers_2[-1])

        # Riemannian transport at two speeds (for reference)
        riem_conn = BerwaldConnection(riemannian_metric)
        vecs_riem_1 = riem_conn.parallel_transport(path_x1, path_v1, vec_start)
        vecs_riem_2 = riem_conn.parallel_transport(path_x1, path_v2, vec_start)
        diff_riem = jnp.linalg.norm(vecs_riem_1[-1] - vecs_riem_2[-1])

        # The Randers difference should strictly exceed the Riemannian difference
        # because Gamma(x, v) genuinely depends on v for Randers
        self.assertGreater(float(diff_randers), float(diff_riem) + 1e-4,
                           "Randers velocity dependence should exceed Riemannian dt-artifact")

    def test_sphere_holonomy(self):
        """
        Verify parallel transport around a latitude circle on S^2
        reproduces a known holonomy angle.
        
        Note: Our implementation uses g(x) = I_3 (ambient Euclidean), so
        Gamma^i_jk = 0 and the transport is entirely projection-based.
        The resulting holonomy angle is 2*pi*cos(theta), which is the 
        complement of the standard solid-angle formula 2*pi*(1-cos(theta)).
        Both are equivalent modulo 2*pi (cos(a) = cos(2*pi - a)).
        """
        def identity_metric(x): return jnp.eye(3)
        metric = Riemannian(self.sphere, identity_metric)

        theta = jnp.pi / 4.0
        t = jnp.linspace(0, 2*jnp.pi, 200)

        path_x = jnp.stack([
            jnp.sin(theta) * jnp.cos(t),
            jnp.sin(theta) * jnp.sin(t),
            jnp.full_like(t, jnp.cos(theta))
        ], axis=1)

        path_v = jnp.stack([
            -jnp.sin(theta) * jnp.sin(t),
             jnp.sin(theta) * jnp.cos(t),
             jnp.zeros_like(t)
        ], axis=1)

        vec_start = jnp.array([0.0, 1.0, 0.0])

        vecs = BerwaldConnection(metric).parallel_transport(path_x, path_v, vec_start)
        vec_end = vecs[-1]

        # Tangent plane basis at the start/end point (sin(theta), 0, cos(theta))
        phi_hat = jnp.array([0.0, 1.0, 0.0])
        theta_hat = jnp.array([jnp.cos(theta), 0.0, -jnp.sin(theta)])

        v_end_phi = jnp.dot(vec_end, phi_hat)
        v_end_theta = jnp.dot(vec_end, theta_hat)

        angle = jnp.arctan2(v_end_theta, v_end_phi)

        # Our projection-based transport produces angle = 2*pi*cos(theta)
        expected_angle = 2 * jnp.pi * jnp.cos(theta)

        # Use cosine comparison to avoid sign/wrapping ambiguities
        np.testing.assert_allclose(jnp.cos(angle), jnp.cos(expected_angle), atol=1e-2)


    def test_integrator_convergence_order(self):
        """
        Test that the integrator converges at O(1/N) rate (1st-order forward Euler).
        """
        def identity_metric(x): return jnp.eye(3)
        metric = Riemannian(self.sphere, identity_metric)
        conn = BerwaldConnection(metric)

        theta = jnp.pi / 4.0
        vec_start = jnp.array([0.0, 1.0, 0.0])

        # Projection-based transport angle
        expected_angle = 2 * jnp.pi * jnp.cos(theta)

        phi_hat = jnp.array([0.0, 1.0, 0.0])
        theta_hat = jnp.array([jnp.cos(theta), 0.0, -jnp.sin(theta)])

        def run_transport(N):
            t = jnp.linspace(0, 2*jnp.pi, N)
            path_x = jnp.stack([
                jnp.sin(theta) * jnp.cos(t),
                jnp.sin(theta) * jnp.sin(t),
                jnp.full_like(t, jnp.cos(theta))
            ], axis=1)
            path_v = jnp.stack([
                -jnp.sin(theta) * jnp.sin(t),
                 jnp.sin(theta) * jnp.cos(t),
                 jnp.zeros_like(t)
            ], axis=1)

            vec_end = conn.parallel_transport(path_x, path_v, vec_start)[-1]

            v_end_phi = jnp.dot(vec_end, phi_hat)
            v_end_theta = jnp.dot(vec_end, theta_hat)

            # Exact analytic components
            exact_phi = jnp.cos(expected_angle)
            exact_theta = jnp.sin(expected_angle)

            # Euclidean error in tangent plane
            return jnp.sqrt((v_end_phi - exact_phi)**2 + (v_end_theta - exact_theta)**2)

        error_20 = run_transport(20)
        error_40 = run_transport(40)
        error_80 = run_transport(80)

        # Ratio ~ 2.0 for 1st-order convergence
        ratio_20_40 = error_20 / error_40
        ratio_40_80 = error_40 / error_80

        self.assertTrue(1.5 < ratio_20_40 < 2.5, f"1st-order ratio expected near 2, got {ratio_20_40}")
        self.assertTrue(1.5 < ratio_40_80 < 2.5, f"1st-order ratio expected near 2, got {ratio_40_80}")


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

        # --- 1. Verify Christoffel symbols analytically ---
        x_test = jnp.array([0.0, 2.0])  # y = 2
        v_test = jnp.array([1.0, 1.0])
        gamma = conn.christoffel_symbols(x_test, v_test)

        y_val = 2.0
        # Expected: Gamma^1_12 = Gamma^1_21 = -1/y, Gamma^2_11 = 1/y, Gamma^2_22 = -1/y
        # Note: tolerance is 1e-3 due to Tikhonov regularization in spray (spray_reg)
        np.testing.assert_allclose(gamma[0, 0, 1], -1.0/y_val, atol=1e-3)  # Gamma^1_12
        np.testing.assert_allclose(gamma[0, 1, 0], -1.0/y_val, atol=1e-3)  # Gamma^1_21
        np.testing.assert_allclose(gamma[1, 0, 0],  1.0/y_val, atol=1e-3)  # Gamma^2_11
        np.testing.assert_allclose(gamma[1, 1, 1], -1.0/y_val, atol=1e-3)  # Gamma^2_22

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
        self.assertLess(float(naive_norm), 0.5,
                        "Sanity check: without transport, norm should drop significantly")


if __name__ == '__main__':
    unittest.main()
