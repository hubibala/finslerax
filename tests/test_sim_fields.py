"""Tests for finslerax.sim.fields (analytic vector fields) and finslerax.utils.device."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from finslerax.sim.fields import (
    get_stream_function_flow,
    harmonic_vortices,
    lamb_oseen_vortex,
    rankine_vortex,
    rossby_haurwitz,
    tilted_rotation,
)
from finslerax.utils.device import configure_device, get_device


def sphere_points(n=32, seed=0):
    pts = jax.random.normal(jax.random.PRNGKey(seed), (n, 3))
    return pts / jnp.linalg.norm(pts, axis=1, keepdims=True)


class TestSphereFields(unittest.TestCase):
    def assert_tangent(self, flow, atol=1e-5):
        pts = sphere_points()
        vs = jax.vmap(flow)(pts)
        self.assertTrue(np.all(np.isfinite(np.array(vs))))
        dots = jnp.abs(jnp.sum(pts * vs, axis=1))
        self.assertLess(float(dots.max()), atol)

    def test_stream_function_flow_tangent_and_divergence_free(self):
        flow = get_stream_function_flow(lambda x: x[2] ** 2 + 0.5 * x[0] * x[1])
        self.assert_tangent(flow)
        # v = grad(psi) x X has identically zero ambient divergence.
        divs = jax.vmap(lambda p: jnp.trace(jax.jacfwd(flow)(p)))(sphere_points())
        self.assertLess(float(jnp.abs(divs).max()), 1e-5)

    def test_tilted_rotation_tangent(self):
        self.assert_tangent(tilted_rotation(alpha_deg=30.0))

    def test_rossby_haurwitz_tangent(self):
        self.assert_tangent(rossby_haurwitz(R=4, omega=1.0, K=0.8))

    def test_harmonic_vortices_tangent(self):
        self.assert_tangent(harmonic_vortices(ell=5, m=3))


class TestPlanarVortices(unittest.TestCase):
    def assert_azimuthal(self, flow, center):
        """The velocity is perpendicular to the radius vector from the center."""
        pts = jax.random.normal(jax.random.PRNGKey(1), (32, 2)) * 2.0
        vs = jax.vmap(flow)(pts)
        self.assertTrue(np.all(np.isfinite(np.array(vs))))
        radial = pts - center
        dots = jnp.abs(jnp.sum(radial * vs, axis=1))
        self.assertLess(float(dots.max()), 1e-5)

    def test_lamb_oseen_azimuthal_and_regular_at_core(self):
        center = jnp.zeros(2)
        flow = lamb_oseen_vortex(center, core_radius=0.5, circulation=1.0)
        self.assert_azimuthal(flow, center)
        # Smoothed core: speed vanishes at the center and is finite everywhere.
        self.assertLess(float(jnp.linalg.norm(flow(center + 1e-4))), 1e-2)

    def test_rankine_profile(self):
        center = jnp.zeros(2)
        rc, gamma = 1.0, 2.0
        flow = rankine_vortex(center, core_radius=rc, circulation=gamma)
        self.assert_azimuthal(flow, center)
        # Solid-body inside: |v|(r) linear in r; irrotational outside: ~ 1/r.
        v_half = float(jnp.linalg.norm(flow(jnp.array([0.5, 0.0]))))
        v_quarter = float(jnp.linalg.norm(flow(jnp.array([0.25, 0.0]))))
        self.assertAlmostEqual(v_half / v_quarter, 2.0, places=3)
        v_2 = float(jnp.linalg.norm(flow(jnp.array([2.0, 0.0]))))
        v_4 = float(jnp.linalg.norm(flow(jnp.array([4.0, 0.0]))))
        self.assertAlmostEqual(v_2 / v_4, 2.0, places=3)

    def test_padding_preserves_dimension(self):
        flow = lamb_oseen_vortex(jnp.zeros(2))
        v = flow(jnp.array([1.0, 0.5, 7.0]))  # 3-D input, vortex in first 2 dims
        self.assertEqual(v.shape, (3,))
        self.assertEqual(float(v[2]), 0.0)


class TestDevice(unittest.TestCase):
    def test_get_device_cpu(self):
        dev = get_device("cpu")
        self.assertEqual(dev.platform, "cpu")

    def test_get_device_invalid_raises(self):
        with self.assertRaises(RuntimeError):
            get_device("nonexistent-backend")

    def test_configure_device_returns_device(self):
        dev = configure_device("cpu")
        self.assertEqual(dev.platform, "cpu")


if __name__ == "__main__":
    unittest.main()
