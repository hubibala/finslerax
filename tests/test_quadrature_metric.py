"""Tests for the segment-quadrature metric wrapper."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ham.geometry import EuclideanSpace, Riemannian, SegmentQuadratureMetric
from ham.geometry.zoo import Euclidean
from ham.solvers import AVBDSolver

jax.config.update("jax_enable_x64", False)


def bump_metric(D, height=50.0, width=0.05):
    """Conformal metric with a tall, narrow cost bump at the origin (a 'void')."""

    def g_net(z):
        c = 1.0 + height * jnp.exp(-jnp.sum(z**2) / width)
        return c * jnp.eye(D)

    return Riemannian(EuclideanSpace(D), g_net)


class TestSegmentQuadratureMetric(unittest.TestCase):
    def test_nquad1_reproduces_base(self):
        """nquad=1 (start-vertex sample) must equal the base metric exactly."""
        base = bump_metric(2)
        wrapped = SegmentQuadratureMetric(base, nquad=1)
        x = jnp.array([-1.0, 0.3])
        v = jnp.array([2.0, -0.1])
        self.assertAlmostEqual(
            float(wrapped.energy(x, v)), float(base.energy(x, v)), places=5
        )
        self.assertAlmostEqual(
            float(wrapped.metric_fn(x, v)), float(base.metric_fn(x, v)), places=5
        )

    def test_midpoint_simpson_see_the_void(self):
        """A segment straddling the void costs strictly more under nquad=2/3."""
        base = bump_metric(2)
        x = jnp.array([-1.0, 0.0])  # start away from the bump (cheap)
        v = jnp.array([2.0, 0.0])  # midpoint lands on the origin bump
        e1 = float(SegmentQuadratureMetric(base, 1).energy(x, v))
        e2 = float(SegmentQuadratureMetric(base, 2).energy(x, v))
        e3 = float(SegmentQuadratureMetric(base, 3).energy(x, v))
        self.assertGreater(e2, e1)
        self.assertGreater(e3, e1)

    def test_manifold_delegated(self):
        base = bump_metric(3)
        wrapped = SegmentQuadratureMetric(base, nquad=2)
        self.assertIs(wrapped.manifold, base.manifold)

    def test_invalid_nquad(self):
        with self.assertRaises(ValueError):
            SegmentQuadratureMetric(bump_metric(2), nquad=4)

    def test_jit_vmap_grad(self):
        """energy is jit/vmap-able and differentiable through the quadrature."""
        wrapped = SegmentQuadratureMetric(bump_metric(2), nquad=3)
        xs = jnp.array([[-1.0, 0.0], [0.5, 0.5]])
        vs = jnp.array([[2.0, 0.0], [0.1, -0.2]])
        es = jax.jit(jax.vmap(wrapped.energy))(xs, vs)
        self.assertEqual(es.shape, (2,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(es))))
        g = jax.jit(jax.grad(lambda v: wrapped.energy(xs[0], v)))(vs[0])
        self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    def test_wrapped_euclidean_solve_matches_unwrapped(self):
        """On a flat metric (constant conformal factor), wrapping is a no-op for
        the solved geodesic: the straight line is recovered either way."""
        base = Euclidean(EuclideanSpace(3))
        wrapped = SegmentQuadratureMetric(base, nquad=2)
        p0 = jnp.array([0.0, 0.0, 0.0])
        p1 = jnp.array([1.0, 2.0, -1.0])
        solver = AVBDSolver(step_size=0.1, iterations=300, grad_clip=10.0)
        key = jax.random.PRNGKey(0)
        t_base = solver.solve(base, p0, p1, n_steps=10, key=key)
        t_wrap = solver.solve(wrapped, p0, p1, n_steps=10, key=key)
        line = np.linspace(np.array(p0), np.array(p1), 11)
        self.assertLess(float(np.max(np.abs(np.array(t_base.xs) - line))), 1e-2)
        self.assertLess(float(np.max(np.abs(np.array(t_wrap.xs) - line))), 1e-2)


if __name__ == "__main__":
    unittest.main()
