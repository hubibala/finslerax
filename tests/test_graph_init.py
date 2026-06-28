"""Tests for the kNN-graph geodesic initialiser and arc-length reparametrisation."""

import unittest

import jax.numpy as jnp
import numpy as np

from ham.solvers import (
    build_knn_graph,
    geodesic_graph_init,
    reparametrize_arclength,
)


def ring_cloud(n=64):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(th), np.sin(th)], 1).astype(np.float32)


class TestGeodesicGraphInit(unittest.TestCase):
    def test_shape_and_endpoints(self):
        cloud = ring_cloud()
        p0 = jnp.array([1.0, 0.0])
        p1 = jnp.array([-1.0, 0.0])
        path = geodesic_graph_init(jnp.asarray(cloud), p0, p1, n_steps=20, k=4)
        self.assertEqual(path.shape, (21, 2))
        np.testing.assert_allclose(np.array(path[0]), np.array(p0), atol=1e-5)
        np.testing.assert_allclose(np.array(path[-1]), np.array(p1), atol=1e-5)

    def test_goes_around_not_through(self):
        """On a ring, the graph path must hug the ring (radius ~1), not cut the
        empty centre as a straight chord would."""
        cloud = ring_cloud()
        p0 = jnp.array([1.0, 0.0])
        p1 = jnp.array([-1.0, 0.0])
        path = np.asarray(geodesic_graph_init(jnp.asarray(cloud), p0, p1, n_steps=24, k=4))
        closest_to_centre = float(np.min(np.linalg.norm(path, axis=1)))
        self.assertGreater(closest_to_centre, 0.7)
        # the straight chord, by contrast, passes through the origin
        chord = np.linspace(np.array(p0), np.array(p1), 25)
        self.assertLess(float(np.min(np.linalg.norm(chord, axis=1))), 0.1)

    def test_reuse_prebuilt_graph(self):
        cloud = ring_cloud()
        g = build_knn_graph(cloud, k=4)
        p0 = jnp.array([1.0, 0.0])
        p1 = jnp.array([0.0, 1.0])
        path = geodesic_graph_init(jnp.asarray(cloud), p0, p1, n_steps=12, graph=g)
        self.assertEqual(path.shape, (13, 2))

    def test_disconnected_falls_back_to_chord(self):
        """Two far-apart clusters with small k: no path -> straight-chord fallback."""
        a = np.random.RandomState(0).randn(20, 2).astype(np.float32) * 0.05
        b = a + np.array([100.0, 0.0], np.float32)
        cloud = np.concatenate([a, b])
        path = geodesic_graph_init(
            jnp.asarray(cloud), jnp.asarray(a[0]), jnp.asarray(b[0]), n_steps=8, k=3
        )
        chord = np.linspace(a[0], b[0], 9)
        self.assertLess(float(np.max(np.abs(np.array(path) - chord))), 1e-3)


class TestReparametrizeArclength(unittest.TestCase):
    def test_uniform_segments(self):
        # A path with cubic-bunched parameter; reparam should equalise segments.
        t = np.linspace(0, 1, 11) ** 3
        bunched = jnp.asarray(np.stack([t, np.zeros(11)], 1).astype(np.float32))
        rp = np.asarray(reparametrize_arclength(bunched, 10))
        seg = np.linalg.norm(np.diff(rp, axis=0), axis=1)
        self.assertLess(float(seg.std() / seg.mean()), 1e-3)

    def test_preserves_endpoints_and_curve(self):
        pts = jnp.asarray(
            np.stack([np.linspace(0, 1, 9), np.sin(np.linspace(0, 1, 9))], 1).astype(
                np.float32
            )
        )
        rp = reparametrize_arclength(pts, 16)
        self.assertEqual(rp.shape, (17, 2))
        np.testing.assert_allclose(np.array(rp[0]), np.array(pts[0]), atol=1e-5)
        np.testing.assert_allclose(np.array(rp[-1]), np.array(pts[-1]), atol=1e-5)

    def test_degenerate_zero_length(self):
        pts = jnp.zeros((5, 3))
        rp = reparametrize_arclength(pts, 8)
        self.assertEqual(rp.shape, (9, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(rp))))


if __name__ == "__main__":
    unittest.main()
