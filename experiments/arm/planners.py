"""Planners — the continuation-wrapped AVBD workhorse and the 2-DoF eikonal cross-check.

``AVBDPlanner`` solves the geodesic BVP with **barrier-strength continuation**: a
stiff obstacle metric is reached by annealing ``α`` through a schedule, each stage
warm-started (``init_path``) from the previous solution, which keeps the iterate in
a good basin where a single cold solve diverges (the known stiff-metric failure
mode). It optionally polishes the smooth final solve with a global Gauss-Newton
step. ``EikonalPlanner`` solves the Randers arrival-time PDE on a 2-D configuration
grid — feasible only at 2 DoF — as an independent forward-planning ground truth.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.solvers import (
    AVBDSolver,
    EikonalSolver,
    GaussNewtonGeodesic,
    resample_path,
)
from ham.solvers.avbd import Trajectory

MetricFactory = Callable[[float], Any]


class AVBDPlanner(eqx.Module):
    """Continuation-wrapped AVBD geodesic planner.

    Args:
        step_size, beta, iterations, grad_clip: AVBD solver settings.
        polish: If set, refine the final (unconstrained) solve with Gauss-Newton.
        gn_iterations: Gauss-Newton iterations when ``polish``.
    """

    step_size: float = eqx.field(static=True, default=0.05)
    beta: float = eqx.field(static=True, default=2.0)
    iterations: int = eqx.field(static=True, default=300)
    grad_clip: float = eqx.field(static=True, default=10.0)
    polish: bool = eqx.field(static=True, default=False)
    gn_iterations: int = eqx.field(static=True, default=40)

    def plan(
        self,
        make_metric: MetricFactory,
        start: jax.Array,
        goal: jax.Array,
        *,
        n_steps: int = 24,
        alphas: tuple[float, ...] = (0.0,),
        constraints: list[Callable] | None = None,
        init_path: jax.Array | None = None,
    ) -> Trajectory:
        """Plan a path, annealing the barrier strength through ``alphas``.

        Args:
            make_metric: ``α -> FinslerMetric`` (the obstacle barrier strength).
                For an obstacle-free metric, ``α`` is ignored and ``alphas=(0.0,)``.
            start, goal: Boundary points on the metric's manifold.
            n_steps: Path segments.
            alphas: Continuation schedule of barrier strengths (last = target).
            constraints: Optional AVBD equality constraints ``c(x) = 0``.
            init_path: Optional warm-start (e.g. a sampler route, Stage E).
        """
        path = init_path
        traj: Trajectory | None = None
        for alpha in alphas:
            metric = make_metric(alpha)
            if path is not None and path.shape[0] != n_steps + 1:
                path = resample_path(path, n_steps, project=metric.manifold.project)
            solver = AVBDSolver(
                step_size=self.step_size,
                beta=self.beta,
                iterations=self.iterations,
                grad_clip=self.grad_clip,
            )
            traj = solver.solve(
                metric,
                start,
                goal,
                n_steps=n_steps,
                constraints=constraints,
                train_mode=False,
                init_path=path,
            )
            path = traj.xs

        if self.polish and not constraints:
            gn = GaussNewtonGeodesic(iterations=self.gn_iterations)
            traj = gn.solve(
                make_metric(alphas[-1]),
                start,
                goal,
                n_steps=n_steps,
                init_path=path,
                train_mode=False,
            )
        return traj


class EikonalPlanner(eqx.Module):
    """2-DoF eikonal arrival-time field over a configuration grid (cross-check only)."""

    max_iters: int = eqx.field(static=True, default=400)
    tol: float = eqx.field(static=True, default=1e-5)

    def arrival_field(
        self,
        metric: Any,
        source: jax.Array,
        grid_extent: tuple[float, float, float, float],
        grid_shape: tuple[int, int],
    ) -> jax.Array:
        """Time-to-arrival field ``T(θ₁, θ₂)`` from ``source`` (an AsymmetricMetric)."""
        solver = EikonalSolver(max_iters=self.max_iters, tol=self.tol)
        T, _, _ = solver.solve(metric, jnp.atleast_2d(source), grid_extent, grid_shape)
        return T

    @staticmethod
    def sample(
        T: jax.Array,
        grid_extent: tuple[float, float, float, float],
        point: jax.Array,
    ) -> jax.Array:
        """Bilinearly sample the arrival field at a configuration ``point``."""
        xmin, xmax, ymin, ymax = grid_extent
        nx, ny = T.shape
        gx = (point[0] - xmin) / (xmax - xmin) * (nx - 1)
        gy = (point[1] - ymin) / (ymax - ymin) * (ny - 1)
        x0 = jnp.clip(jnp.floor(gx).astype(int), 0, nx - 2)
        y0 = jnp.clip(jnp.floor(gy).astype(int), 0, ny - 2)
        fx, fy = gx - x0, gy - y0
        t00, t10 = T[x0, y0], T[x0 + 1, y0]
        t01, t11 = T[x0, y0 + 1], T[x0 + 1, y0 + 1]
        return (
            t00 * (1 - fx) * (1 - fy)
            + t10 * fx * (1 - fy)
            + t01 * (1 - fx) * fy
            + t11 * fx * fy
        )
