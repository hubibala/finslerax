"""Planners — stationary (HAM eikonal/AVBD) and the time-lifted novelty.

``StationaryPlanner`` is a thin, dimension-agnostic wrapper over HAM's existing
solvers: the (volumetric) eikonal for the *global* time-to-arrival field, and
AVBD for a point-to-point route. It is exact for a *frozen* current.

``TimeLiftedPlanner`` is the contribution. For a time-varying current the cost of
crossing a segment depends on *when* you cross it, so the metric is not local and
neither the eikonal nor AVBD apply directly. We discretize the route into vertices
and thread the clock causally with ``jax.lax.scan``:

    ΔT_k = randers_cost(H(m_k), W(m_k, t_k), Δx_k),   t_{k+1} = t_k + ΔT_k,

and minimize the true arrival time ``T = Σ ΔT_k`` plus the constraint penalties
over the interior vertices (Adam), warm-started from the stationary route. This is
the rigorous form of "AVBD over E(x_i, t_i, v_i)": differentiable, dimension-
agnostic, returns the path, and handles arbitrary ``W(x, t)``.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ham.solvers import AVBDSolver, EikonalSolver, VolumetricEikonalSolver
from ham.utils.config import DEFAULT_JNP_DTYPE

from .constraints import avbd_equality_constraints, constraint_penalty, max_violation
from .medium import randers_cost


def thread_clock(path, medium, glider, t0):
    """Causal travel-time accumulation along a path through ``W(x, t)``.

    Args:
        path: Vertices, shape ``(N + 1, D)``.
        medium: :class:`OceanMedium`.
        glider: :class:`Glider`.
        t0: Departure time (scalar).

    Returns:
        ``(T_total, times)`` where ``T_total = Σ ΔT_k`` is the executed arrival
        time and ``times`` (shape ``(N + 1,)``) is the clock at each vertex.
    """
    seg = path[1:] - path[:-1]
    mid = 0.5 * (path[1:] + path[:-1])

    def step(t_k, k):
        m = mid[k]
        dx = seg[k]
        W = medium.physical_current(m, t_k)
        H = glider.sea_tensor(medium, m)
        dT = randers_cost(H, W, dx)
        return t_k + dT, t_k

    n_seg = seg.shape[0]
    t_final, t_starts = jax.lax.scan(step, jnp.asarray(float(t0)), jnp.arange(n_seg))
    times = jnp.concatenate([t_starts, t_final[None]])
    return t_final - jnp.asarray(float(t0)), times


class PlanResult(NamedTuple):
    path: jax.Array
    arrival_time: jax.Array
    violation: jax.Array


class StationaryPlanner(eqx.Module):
    """Frozen-current planning via HAM's eikonal field and AVBD route."""

    max_iters: int = eqx.field(static=True, default=200)
    tol: float = eqx.field(static=True, default=1e-5)
    avbd_step: float = eqx.field(static=True, default=0.05)
    avbd_beta: float = eqx.field(static=True, default=2.0)
    avbd_iters: int = eqx.field(static=True, default=200)

    def arrival_field(self, metric, source, grid_extent, grid_shape):
        """Global time-to-arrival field (2D eikonal or 3D volumetric)."""
        source = jnp.atleast_2d(source)
        if len(grid_shape) == 3:
            solver = VolumetricEikonalSolver(max_iters=self.max_iters, tol=self.tol)
            T, _, _ = solver.solve(metric, source, grid_extent, grid_shape)
            return T
        solver = EikonalSolver(max_iters=self.max_iters, tol=self.tol)
        T, _, _ = solver.solve(metric, source, grid_extent, grid_shape)
        return T

    def route(self, metric, start, end, n_steps=24, constraints=None, init_path=None):
        """Point-to-point optimal route under a frozen metric (AVBD).

        Equality, per-vertex constraints are passed into AVBD's native ALM array;
        inequality/segment constraints are left to the time-lifted planner.
        """
        eq = avbd_equality_constraints(constraints or [])
        solver = AVBDSolver(
            step_size=self.avbd_step,
            beta=self.avbd_beta,
            iterations=self.avbd_iters,
            grad_clip=10.0,
        )
        traj = solver.solve(
            metric,
            start,
            end,
            n_steps=n_steps,
            constraints=eq or None,
            train_mode=False,
            init_path=init_path,
        )
        return traj.xs


class TimeLiftedPlanner(eqx.Module):
    """Time-dependent route optimizer (clock-threaded action; the novelty)."""

    n_iters: int = eqx.field(static=True, default=400)
    lr: float = eqx.field(static=True, default=0.03)
    penalty_weight: float = eqx.field(static=True, default=50.0)
    penalty_ramp: float = eqx.field(static=True, default=40.0)

    def plan(
        self,
        medium,
        glider,
        start,
        end,
        t0=0.0,
        n_steps=24,
        constraints=None,
        init_path=None,
        key=None,
        n_restarts=0,
    ) -> PlanResult:
        """Minimize true (time-varying) arrival time over interior vertices.

        Args:
            medium, glider: the environment / vehicle.
            start, end: fixed boundary points, shape ``(D,)``.
            t0: departure time.
            n_steps: number of path segments.
            constraints: list of :class:`Constraint` (defaults to the glider's).
            init_path: warm-start path ``(n_steps + 1, D)`` (e.g. the stationary
                route); a straight line is used if omitted.
            key: PRNG key for multi-start perturbations.
            n_restarts: extra randomly-perturbed starts; the best feasible plan
                is returned (mitigates the local-minimum nature of the BVP).
        """
        start = jnp.asarray(start, dtype=DEFAULT_JNP_DTYPE)
        end = jnp.asarray(end, dtype=DEFAULT_JNP_DTYPE)
        constraints = constraints if constraints is not None else glider.constraints()

        if init_path is None:
            init_path = jnp.linspace(start, end, n_steps + 1)
        interior0 = jnp.asarray(init_path, dtype=DEFAULT_JNP_DTYPE)[1:-1]

        # Penalty-continuation schedule: the weight grows geometrically from
        # ``penalty_weight`` to ``penalty_weight · penalty_ramp`` across the run,
        # so early iterations stay stable while late ones enforce the constraints
        # to tolerance (standard penalty method; cf. ham.solvers.continuation).
        w0 = self.penalty_weight
        ramp = self.penalty_ramp if constraints else 1.0

        @jax.jit
        def loss_and_aux(interior, pen_w):
            path = jnp.concatenate([start[None], interior, end[None]], axis=0)
            T_total, times = thread_clock(path, medium, glider, t0)
            pen = constraint_penalty(path, times, constraints)
            return T_total + pen_w * pen, (T_total, path, times)

        grad_fn = jax.jit(jax.grad(lambda z, w: loss_and_aux(z, w)[0]))
        opt = optax.adam(self.lr)
        n = max(1, self.n_iters)

        def run(interior):
            state = opt.init(interior)

            @jax.jit
            def step(interior, state, pen_w):
                g = grad_fn(interior, pen_w)
                updates, state = opt.update(g, state)
                return optax.apply_updates(interior, updates), state

            for i in range(n):
                pen_w = w0 * (ramp ** (i / (n - 1) if n > 1 else 1.0))
                interior, state = step(interior, state, pen_w)
            _, (T_total, path, times) = loss_and_aux(interior, w0 * ramp)
            viol = max_violation(path, times, constraints)
            return path, T_total, viol

        path, T_total, viol = run(interior0)

        if n_restarts > 0:
            key = jax.random.PRNGKey(0) if key is None else key
            best = (path, T_total, viol)
            for _ in range(n_restarts):
                key, sk = jax.random.split(key)
                # Perturb interior (mostly in depth) to seed a different basin.
                noise = 0.3 * jax.random.normal(sk, interior0.shape)
                cand = run(interior0 + noise)
                # Prefer feasible; among feasible, prefer smaller arrival time.
                cand_score = cand[1] + 1e3 * jax.nn.relu(cand[2] - 1e-2)
                best_score = best[1] + 1e3 * jax.nn.relu(best[2] - 1e-2)
                if float(cand_score) < float(best_score):
                    best = cand
            path, T_total, viol = best

        return PlanResult(path=path, arrival_time=T_total, violation=viol)
