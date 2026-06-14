"""Tests for the global Gauss-Newton geodesic solver and continuation drivers."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ham.geometry import EuclideanSpace, Riemannian
from ham.solvers import (
    AVBDSolver,
    GaussNewtonGeodesic,
    resample_path,
    solve_continuation,
)

jax.config.update("jax_enable_x64", False)
R = 1.0


def ring_metric(D, alpha, w):
    def g_net(z):
        r = jnp.sqrt(z[0] ** 2 + z[1] ** 2 + 1e-12)
        c = ((r - R) ** 2 + jnp.sum(z[2:] ** 2)) / (w**2)
        return jnp.exp(alpha * c) * jnp.eye(D)

    return Riemannian(EuclideanSpace(D), g_net)


def ring_pt(theta, D):
    z = np.zeros(D, np.float32)
    z[0], z[1] = R * np.cos(theta), R * np.sin(theta)
    return jnp.asarray(z)


def arc_rmse(xs, t0, t1, D):
    xs = np.asarray(xs)
    n = xs.shape[0] - 1
    ts = np.linspace(t0, t1, n + 1)
    arc = np.zeros((n + 1, D), np.float32)
    arc[:, 0], arc[:, 1] = R * np.cos(ts), R * np.sin(ts)
    return float(np.sqrt(np.mean(np.sum((xs - arc) ** 2, axis=1))))


class TestGaussNewtonGeodesic(unittest.TestCase):
    def test_euclidean_straight_line(self):
        """With a Euclidean metric the geodesic is the straight line."""
        metric = ring_metric(3, 0.0, 1.0)  # alpha=0 => identity metric
        p0 = jnp.array([0.0, 0.0, 0.0])
        p1 = jnp.array([1.0, 2.0, -1.0])
        solver = GaussNewtonGeodesic(iterations=20)
        traj = solver.solve(metric, p0, p1, n_steps=12)
        line = np.linspace(np.array(p0), np.array(p1), 13)
        self.assertLess(float(np.max(np.abs(np.array(traj.xs) - line))), 1e-3)

    def test_recovers_arc_mild_metric(self):
        """On a mild ring metric, GN recovers the connecting arc from a cold start."""
        D, th = 8, np.deg2rad(120)
        metric = ring_metric(D, 4.0, 0.35)
        solver = GaussNewtonGeodesic(iterations=40, mu0=1e-2)
        traj = solver.solve(metric, ring_pt(0, D), ring_pt(th, D), n_steps=24)
        self.assertLess(arc_rmse(traj.xs, 0, th, D), 0.06)

    def test_dimension_independence(self):
        """In-basin convergence is independent of ambient dimension: padding the ring
        problem with extra zero dimensions leaves the recovered arc unchanged."""
        th = np.deg2rad(120)
        N = 24
        # Warm start near the arc (the solver's guarantee is in-basin convergence).
        rmses = []
        for D in (2, 32, 128):
            metric = ring_metric(D, 4.0, 0.35)
            ts = np.linspace(0, th, N + 1)
            warm = np.zeros((N + 1, D), np.float32)
            warm[:, 0], warm[:, 1] = R * np.cos(ts), R * np.sin(ts)
            warm += np.random.RandomState(0).randn(N + 1, D).astype(np.float32) * 0.03
            solver = GaussNewtonGeodesic(iterations=25, mu0=1e-3)
            traj = solver.solve(
                metric, ring_pt(0, D), ring_pt(th, D), n_steps=N,
                init_path=jnp.asarray(warm),
            )
            rmses.append(arc_rmse(traj.xs, 0, th, D))
        self.assertLess(max(rmses) - min(rmses), 5e-3)
        self.assertLess(max(rmses), 0.05)

    def test_jit_and_vmap(self):
        """solve is jittable and vmappable."""
        D = 4
        metric = ring_metric(D, 2.0, 0.4)
        solver = GaussNewtonGeodesic(iterations=10)
        jit_solve = jax.jit(
            lambda a, b: solver.solve(metric, a, b, n_steps=8)
        )
        traj = jit_solve(ring_pt(0.0, D), ring_pt(0.5, D))
        self.assertEqual(traj.xs.shape, (9, D))
        self.assertTrue(np.all(np.isfinite(np.array(traj.xs))))

    def test_finite_on_n_inner_one(self):
        """n_steps=2 (a single interior vertex) is handled by block-Thomas."""
        D = 3
        metric = ring_metric(D, 1.0, 0.5)
        solver = GaussNewtonGeodesic(iterations=10)
        traj = solver.solve(metric, ring_pt(0.0, D), ring_pt(0.3, D), n_steps=2)
        self.assertEqual(traj.xs.shape, (3, D))
        self.assertTrue(np.all(np.isfinite(np.array(traj.xs))))


class TestContinuation(unittest.TestCase):
    def test_resample_preserves_endpoints(self):
        path = jnp.asarray(np.random.RandomState(0).randn(9, 5).astype(np.float32))
        up = resample_path(path, 16)
        self.assertEqual(up.shape, (17, 5))
        self.assertTrue(np.allclose(np.array(up[0]), np.array(path[0]), atol=1e-5))
        self.assertTrue(np.allclose(np.array(up[-1]), np.array(path[-1]), atol=1e-5))

    def test_continuation_solves_stiff_case(self):
        """Cold GS-GD diverges on a stiff long arc; multilevel+anneal continuation solves it."""
        D, th, w = 64, np.deg2rad(150), 0.30

        # Cold brute-force GS-GD diverges (energy blows up).
        cold = AVBDSolver(step_size=0.05, iterations=2000, grad_clip=10.0)
        cold_traj = cold.solve(
            ring_metric(D, 8.0, w), ring_pt(0, D), ring_pt(th, D), n_steps=32
        )
        cold_E = float(cold_traj.energy)
        self.assertTrue(not np.isfinite(cold_E) or cold_E > 1.0)  # diverged / far from arc

        # Continuation recipe recovers the arc.
        stages = [
            (AVBDSolver(step_size=0.05, iterations=300, grad_clip=10.0),
             ring_metric(D, 2.0, w), 8),
            (AVBDSolver(step_size=0.05, iterations=300, grad_clip=10.0),
             ring_metric(D, 4.0, w), 16),
            (AVBDSolver(step_size=0.05, iterations=500, grad_clip=10.0),
             ring_metric(D, 8.0, w), 32),
        ]
        traj = solve_continuation(stages, ring_pt(0, D), ring_pt(th, D))
        self.assertLess(arc_rmse(traj.xs, 0, th, D), 0.05)

    def test_continuation_with_gn_stage(self):
        """Continuation accepts a GaussNewtonGeodesic polishing stage."""
        D, th, w = 32, np.deg2rad(130), 0.33
        stages = [
            (AVBDSolver(step_size=0.05, iterations=300, grad_clip=10.0),
             ring_metric(D, 2.0, w), 12),
            (GaussNewtonGeodesic(iterations=30, mu0=1e-2),
             ring_metric(D, 5.0, w), 24),
        ]
        traj = solve_continuation(stages, ring_pt(0, D), ring_pt(th, D))
        self.assertLess(arc_rmse(traj.xs, 0, th, D), 0.06)


if __name__ == "__main__":
    unittest.main()
