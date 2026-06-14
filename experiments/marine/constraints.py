"""Unified constraint layer — the extensibility hook (and the AVBD answer).

``AVBDSolver.solve(..., constraints=[c(x) -> scalar])`` enforces *equality*
``c(x) = 0`` via its Augmented Lagrangian (``beta`` is the penalty stiffness;
it is inert without constraints). That natively covers equality, per-vertex
constraints (waypoints, fixed-depth legs) — use :func:`avbd_equality_constraints`
to extract them.

But the grounded vehicle physics are mostly **inequality**, **per-segment**, and
sometimes **time-aware** (depth envelope, glide-angle kinematics, seafloor
clearance, time-varying no-go zones), and AVBD assumes a *local* metric — which
the history-dependent clock of the time-lifted planner is not. So every
:class:`Constraint` is also enforceable as a differentiable penalty via
:func:`constraint_penalty`, which the time-lifted planner adds to its action.

Adding new physics = construct a new :class:`Constraint`; both planners pick it up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Constraint:
    """A single path constraint.

    Used at Python/trace level (not a traced PyTree), so ``fn`` may freely close
    over arrays and other constraints' details.

    Attributes:
        name: Human-readable identifier.
        kind: ``"eq"`` (``c(x) = 0``) or ``"ineq"`` (``g(x) <= 0``).
        scope: ``"vertex"`` (``fn`` of one point) or ``"segment"`` (``fn`` of
            consecutive points ``x_k, x_{k+1}``).
        time_aware: If True, ``fn`` takes an extra trailing arg ``t`` (the running
            clock at that vertex/segment).
        fn: The constraint function (see ``scope`` / ``time_aware`` for signature).
        weight: Per-constraint penalty weight.
    """

    name: str
    kind: str
    scope: str
    time_aware: bool
    fn: Callable
    weight: float = 1.0


def _term(c: Constraint, value: jax.Array) -> jax.Array:
    """Penalty contribution for a raw constraint value."""
    if c.kind == "eq":
        return value**2
    # inequality g <= 0
    return jax.nn.relu(value) ** 2


def constraint_penalty(
    path: jax.Array,
    times: jax.Array,
    constraints: list[Constraint],
) -> jax.Array:
    """Total differentiable penalty ``Σ weight · mean(term)`` over a path.

    Args:
        path: Vertices, shape ``(N + 1, D)``.
        times: Arrival clock at each vertex, shape ``(N + 1,)`` (``times[0] = t0``).
        constraints: Constraints to enforce.

    Returns:
        Scalar penalty (0.0 if ``constraints`` is empty).
    """
    if not constraints:
        return jnp.asarray(0.0)

    total = jnp.asarray(0.0)
    for c in constraints:
        if c.scope == "vertex":
            if c.time_aware:
                vals = jax.vmap(c.fn)(path, times)
            else:
                vals = jax.vmap(c.fn)(path)
        elif c.scope == "segment":
            x0, x1 = path[:-1], path[1:]
            if c.time_aware:
                vals = jax.vmap(c.fn)(x0, x1, times[:-1])
            else:
                vals = jax.vmap(c.fn)(x0, x1)
        else:  # pragma: no cover - guarded by builders
            raise ValueError(f"unknown scope {c.scope!r}")
        total = total + c.weight * jnp.mean(_term(c, vals))
    return total


def max_violation(
    path: jax.Array, times: jax.Array, constraints: list[Constraint]
) -> jax.Array:
    """Largest constraint violation along a path (0 if all satisfied)."""
    if not constraints:
        return jnp.asarray(0.0)
    worst = jnp.asarray(0.0)
    for c in constraints:
        if c.scope == "vertex":
            vals = (
                jax.vmap(c.fn)(path, times)
                if c.time_aware
                else jax.vmap(c.fn)(path)
            )
        else:
            x0, x1 = path[:-1], path[1:]
            vals = (
                jax.vmap(c.fn)(x0, x1, times[:-1])
                if c.time_aware
                else jax.vmap(c.fn)(x0, x1)
            )
        v = jnp.abs(vals) if c.kind == "eq" else jax.nn.relu(vals)
        worst = jnp.maximum(worst, jnp.max(v))
    return worst


def avbd_equality_constraints(
    constraints: list[Constraint],
) -> list[Callable[[jax.Array], jax.Array]]:
    """Extract the subset enforceable by AVBD's native equality array."""
    out = []
    for c in constraints:
        if c.kind == "eq" and c.scope == "vertex" and not c.time_aware:
            out.append(c.fn)
    return out


# =============================================================================
# Grounded constraint builders
# =============================================================================
def depth_envelope(z_min: float, z_max: float, axis: int = 2) -> list[Constraint]:
    """Operating-depth bounds ``z_min <= x[axis] <= z_max`` (two inequalities)."""
    return [
        Constraint(
            name="depth_max",
            kind="ineq",
            scope="vertex",
            time_aware=False,
            fn=lambda x: x[axis] - z_max,
        ),
        Constraint(
            name="depth_min",
            kind="ineq",
            scope="vertex",
            time_aware=False,
            fn=lambda x: z_min - x[axis],
        ),
    ]


def glide_angle_limit(
    max_angle_deg: float,
    horiz_axes: tuple[int, int] = (0, 1),
    vert_axis: int = 2,
    eps: float = 1e-6,
) -> Constraint:
    """Kinematic glide limit: ``|Δz| <= tan(max_angle) · ‖Δ_horiz‖`` per segment.

    A buoyancy glider cannot dive/climb steeper than its lift/drag envelope
    allows; this is the glider-specific, *segment-wise* constraint.
    """
    tan_max = float(jnp.tan(jnp.deg2rad(max_angle_deg)))
    i, j = horiz_axes

    def fn(x0, x1):
        d = x1 - x0
        d_horiz = jnp.sqrt(d[i] ** 2 + d[j] ** 2 + eps)
        d_vert = jnp.abs(d[vert_axis])
        return d_vert - tan_max * d_horiz

    return Constraint(
        name="glide_angle",
        kind="ineq",
        scope="segment",
        time_aware=False,
        fn=fn,
        weight=1.0,
    )


def seafloor_clearance(
    bathymetry: Callable[[jax.Array], jax.Array], axis: int = 2
) -> Constraint:
    """Stay above the seabed: ``x[axis] <= bathymetry(x)`` (depth below seabed)."""

    def fn(x):
        return x[axis] - bathymetry(x)

    return Constraint(
        name="seafloor",
        kind="ineq",
        scope="vertex",
        time_aware=False,
        fn=fn,
    )


def time_varying_nogo(
    center_fn: Callable[[jax.Array], jax.Array],
    radius: float,
    horiz_axes: tuple[int, int] = (0, 1),
) -> Constraint:
    """Avoid a moving exclusion disk ``‖x_horiz - center(t)‖ >= radius``.

    Demonstrates a time-aware constraint (the centre depends on the running
    clock); ``g(x, t) = radius - ‖x_horiz - center(t)‖ <= 0``.
    """
    i, j = horiz_axes

    def fn(x, t):
        c = center_fn(t)
        d = jnp.sqrt((x[i] - c[0]) ** 2 + (x[j] - c[1]) ** 2 + 1e-9)
        return radius - d

    return Constraint(
        name="nogo",
        kind="ineq",
        scope="vertex",
        time_aware=True,
        fn=fn,
    )
