"""Evaluation — provider-agnostic path metrics and an independent geodesic oracle.

Every metric is computed from ``(path, robot, metric/dist)`` and never reaches
inside a concrete provider, so the same numbers apply to a synthetic or a real
arm. ``spray_geodesic`` is the HAM-internal cross-check: it re-derives the geodesic
by integrating the spray ODE (``ExponentialMap``) and shooting to the goal — a
different computation from AVBD's discrete BVP, so agreement is real evidence the
metric's whole autodiff chain and both solvers are correct (validation rungs
L3b/L7b). Use it on the smooth intrinsic-angle metric at moderate distance;
shooting is unstable on stiff or long geodesics, where the eikonal (L3) governs.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ham.geometry.manifolds import FlatTorus
from ham.solvers import ExponentialMap


def to_angle_path(manifold: Any, path: jax.Array) -> jax.Array:
    """Joint-angle path ``(T+1, dof)`` from an ambient path (identity off the torus)."""
    if isinstance(manifold, FlatTorus):
        return jax.vmap(manifold.to_angles)(path)
    return path


# ---------------------------------------------------------------------------
# Path metrics (the per-stage report table)
# ---------------------------------------------------------------------------
def kinetic_cost(metric: Any, path: jax.Array) -> jax.Array:
    """Directed Finsler arc length under the kinetic-energy metric (the primary cost)."""
    return metric.arc_length(path)


def cspace_length(manifold: Any, path: jax.Array) -> jax.Array:
    """Configuration-space path length ``Σ ‖log(x_i, x_{i+1})‖`` (wrap-aware)."""
    seg = jax.vmap(manifold.log_map)(path[:-1], path[1:])
    return jnp.sum(jnp.linalg.norm(seg, axis=-1))


def dirichlet_energy(manifold: Any, path: jax.Array) -> jax.Array:
    """Smoothness ``Σ ‖Δ‖²`` (the sampling-paper Dirichlet energy)."""
    seg = jax.vmap(manifold.log_map)(path[:-1], path[1:])
    return jnp.sum(jnp.sum(seg**2, axis=-1))


def task_length(robot: Any, manifold: Any, path: jax.Array) -> jax.Array:
    """End-effector (workspace) path length."""
    q = to_angle_path(manifold, path)
    ee = jax.vmap(robot.fk)(q)
    return jnp.sum(jnp.linalg.norm(ee[1:] - ee[:-1], axis=-1))


def acceleration_energy(manifold: Any, path: jax.Array) -> jax.Array:
    """Discrete ``∫‖q̈‖²`` from second differences of the angle path (jerk proxy)."""
    q = to_angle_path(manifold, path)
    accel = q[2:] - 2 * q[1:-1] + q[:-2]
    return jnp.sum(jnp.sum(accel**2, axis=-1))


def min_clearance(dist: Any, manifold: Any, path: jax.Array) -> jax.Array:
    """Minimum C-space clearance along the path (negative ⇒ collision)."""
    q = to_angle_path(manifold, path)
    return jnp.min(jax.vmap(dist)(q))


def is_collision_free(dist: Any, manifold: Any, path: jax.Array, margin: float = 0.0) -> bool:
    """Whether the whole path stays clear of obstacles."""
    return bool(min_clearance(dist, manifold, path) > margin)


def path_metrics(
    metric: Any,
    robot: Any,
    path: jax.Array,
    dist: Any | None = None,
) -> dict:
    """Assemble the per-stage report dict from a solved path."""
    manifold = metric.manifold
    report = {
        "kinetic_cost": float(kinetic_cost(metric, path)),
        "cspace_length": float(cspace_length(manifold, path)),
        "task_length": float(task_length(robot, manifold, path)),
        "dirichlet_energy": float(dirichlet_energy(manifold, path)),
        "accel_energy": float(acceleration_energy(manifold, path)),
    }
    if dist is not None:
        report["min_clearance"] = float(min_clearance(dist, manifold, path))
    return report


# ---------------------------------------------------------------------------
# Independent geodesic oracle (spray-ODE shooting) — the HAM-internal cross-check
# ---------------------------------------------------------------------------
def spray_geodesic(
    metric: Any,
    start: jax.Array,
    goal: jax.Array,
    *,
    n_steps: int = 64,
    iters: int = 25,
    reg: float = 1e-7,
    damping: float = 1.0,
) -> jax.Array:
    """Geodesic ``start -> goal`` by shooting the spray ODE (intrinsic-angle metric).

    Root-finds the initial velocity ``v0`` so that ``Exp_start(v0) = goal`` with a
    damped Gauss-Newton iteration, then traces the spray. Independent of AVBD.
    Intended for the smooth, full-rank intrinsic-angle representation.

    Returns:
        The geodesic path, shape ``(n_steps + 1, dof)``.
    """
    em = ExponentialMap(max_steps=n_steps)
    manifold = metric.manifold
    dim = start.shape[0]
    v0 = manifold.log_map(start, goal)

    def resid(v):
        return em.shoot(metric, start, v) - goal

    def step(v, _):
        r = resid(v)
        J = jax.jacfwd(resid)(v)
        dv = jnp.linalg.solve(J.T @ J + reg * jnp.eye(dim, dtype=v.dtype), -J.T @ r)
        return v + damping * dv, None

    v0, _ = jax.lax.scan(step, v0, None, length=iters)
    xs, _ = em.trace(metric, start, v0)
    return xs


def spray_endpoint_error(metric: Any, start: jax.Array, goal: jax.Array, **kw) -> float:
    """Residual ``‖Exp_start(v0) - goal‖`` of the shot geodesic (shooting quality)."""
    xs = spray_geodesic(metric, start, goal, **kw)
    return float(jnp.linalg.norm(xs[-1] - goal))
