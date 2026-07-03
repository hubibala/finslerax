"""Task constraints for the arm — exact equality conditions enforced by AVBD's ALM.

AVBD enforces equality ``c(x) = 0`` natively via its Augmented Lagrangian. The
headline task constraint is the *upright cup*: the end-effector must hold a fixed
orientation along the whole path (carry a cup without spilling). Each builder
returns a callable ``c(x) -> scalar`` over an ambient manifold point, ready to
drop into ``AVBDSolver.solve(..., constraints=[c, ...])``.

The residuals are built from the arm's link *directions* (``link_points``
differences), which depend on ``cos``/``sin`` of cumulative joint angles and are
therefore invariant to the ``2π`` wrapping of individual joints on the torus — so
the constraint is smooth across the seam.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from ham.geometry.manifolds import FlatTorus
from ham.utils import NORM_EPS


def _angles(robot, manifold, x: jax.Array) -> jax.Array:
    """Joint angles from an ambient manifold point (identity off the torus)."""
    if isinstance(manifold, FlatTorus):
        return manifold.to_angles(x)
    return x


def upright_constraint(
    robot, manifold, target_angle: float = 0.0
) -> Callable[[jax.Array], jax.Array]:
    """End-effector orientation pinned to ``target_angle`` (the upright-cup condition).

    On the intrinsic-angle representation the orientation ``Σθᵢ`` is a smooth,
    unwrapped function, so the residual is the **linear** ``ee_orientation - target``
    — well conditioned for the ALM (constant gradient). On the Clifford torus,
    where the joint sum wraps, the residual is the seam-safe ``sin(target - φ)``
    built from the last link's direction; the opposite branch (``φ = target + π``)
    is avoided by warm-starting near upright.
    """
    if not isinstance(manifold, FlatTorus):
        def c_linear(x: jax.Array) -> jax.Array:
            return robot.ee_orientation(x) - target_angle

        return c_linear

    d_star = jnp.array([jnp.cos(target_angle), jnp.sin(target_angle)])

    def c(x: jax.Array) -> jax.Array:
        q = manifold.to_angles(x)
        pts = robot.link_points(q)
        d = pts[-1] - pts[-2]
        d = d / (jnp.linalg.norm(d) + NORM_EPS)
        return d[0] * d_star[1] - d[1] * d_star[0]  # cross = sin(target - φ)

    return c


def waypoint_constraint(
    robot, manifold, target_ee: jax.Array
) -> Callable[[jax.Array], jax.Array]:
    """Pass the end-effector through ``target_ee`` (squared-distance equality)."""
    target = jnp.asarray(target_ee)

    def c(x: jax.Array) -> jax.Array:
        q = _angles(robot, manifold, x)
        return jnp.sum((robot.fk(q) - target) ** 2)

    return c


def max_constraint_violation(
    path: jax.Array, constraints: list[Callable[[jax.Array], jax.Array]]
) -> jax.Array:
    """Largest ``|c(x)|`` over interior vertices of ``path`` (0 if no constraints)."""
    if not constraints:
        return jnp.asarray(0.0)
    inner = path[1:-1]
    worst = jnp.asarray(0.0)
    for c in constraints:
        worst = jnp.maximum(worst, jnp.max(jnp.abs(jax.vmap(c)(inner))))
    return worst


def constraint_penalty(
    path: jax.Array, constraints: list[Callable[[jax.Array], jax.Array]]
) -> jax.Array:
    """Quadratic penalty ``Σ mean(c(x)²)`` over interior vertices (penalty-only baseline)."""
    if not constraints:
        return jnp.asarray(0.0)
    inner = path[1:-1]
    total = jnp.asarray(0.0)
    for c in constraints:
        total = total + jnp.mean(jax.vmap(c)(inner) ** 2)
    return total
