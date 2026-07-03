"""Learnable C-space fields — Stage-D targets and the L2 adjoint probe.

Distance fields satisfy the :class:`DistanceField` interface (``q -> clearance``);
wind fields are joint-space drifts ``q -> W``. ``MLPDistance`` is the blank
obstacle field; ``ScaledDistance`` wraps a fixed field with one learnable scalar
(the L2 adjoint finite-difference probe).

The wind fields encode the **Hodge/gauge structure** of the inverse problem
(``spec/exact_drift_gauge_equivalence.md``): a drift's Randers 1-form is
``β = -(M W)♭/λ``, so parameterizing the *form* and raising it with the mobility
``W = -M⁻¹ b`` puts each hypothesis in a known Hodge class:

* :class:`PotentialWind` — ``b = ∇U_φ`` (exact ⇒ invisible in path shape,
  identifiable from cost/timing only);
* :class:`StreamWind` — ``b = ⋆dψ`` (co-exact ⇒ bends paths at first order,
  identifiable from path shape);
* :class:`HodgeWind` — the sum, the full identifiable split;
* :class:`BoundedWind` — an unstructured free wind (the naive hypothesis the
  identifiability experiment shows *cannot* work on a pure-gravity drift).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.nn.networks import VectorField


class MLPDistance(eqx.Module):
    """A blank C-space clearance field ``q -> scalar`` parameterized by an MLP."""

    mlp: eqx.nn.MLP

    def __init__(self, dof: int, *, width: int = 32, depth: int = 2, key: jax.Array):
        self.mlp = eqx.nn.MLP(
            in_size=dof,
            out_size="scalar",
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, q: jax.Array) -> jax.Array:
        return self.mlp(q)


class BoundedWind(eqx.Module):
    """A learnable joint-space drift ``q -> W`` bounded to a mild magnitude.

    ``W(q) = scale · tanh(net(q))`` keeps every component in ``(-scale, scale)``,
    so the wind stays inside the Randers causal limit ``‖W‖_M < 1`` throughout
    training — without it an unconstrained MLP wind goes super-causal and the
    inner geodesic solve diverges.
    """

    net: VectorField
    scale: float = eqx.field(static=True)

    def __init__(self, dof: int, *, width: int = 32, depth: int = 2, scale: float = 0.25, key):
        self.net = VectorField(dof, hidden_dim=width, depth=depth, key=key, use_fourier=False)
        self.scale = float(scale)

    def __call__(self, q: jax.Array) -> jax.Array:
        return self.scale * jnp.tanh(self.net(q))


def _scalar_mlp(in_size: int, width: int, depth: int, key: jax.Array) -> eqx.nn.MLP:
    """Scalar MLP with a zero-initialized final layer.

    Fields built from it start *exactly* at the drift-free metric, so solver-in-
    the-loop losses (L-A) begin from well-conditioned inner solves instead of a
    random super-stiff wind.
    """
    mlp = eqx.nn.MLP(
        in_size=in_size, out_size="scalar", width_size=width, depth=depth,
        activation=jax.nn.tanh, key=key,
    )
    zeroed = jax.tree_util.tree_map(jnp.zeros_like, mlp.layers[-1])
    return eqx.tree_at(lambda m: m.layers[-1], mlp, zeroed)


def form_to_wind(robot: Any, q: jax.Array, b: jax.Array) -> jax.Array:
    """Wind realizing the Randers 1-form ``β ≈ b`` at ``q``: ``W = -M⁻¹(q) b``.

    Since ``β = -(M W)♭/λ``, this raising makes ``β = b/λ`` — equal to ``b`` up to
    the second-order ``1/λ`` factor, so the Hodge class of ``b`` (exact/co-exact)
    is the Hodge class of the drift.
    """
    return -jnp.linalg.solve(robot.inertia(q), b)


def star_grad(f: Any, q: jax.Array) -> jax.Array:
    """Components of the co-exact 1-form ``⋆df`` at ``q`` (2-DoF): ``(-∂₂f, ∂₁f)``."""
    g = jax.grad(f)(q)
    return jnp.array([-g[1], g[0]])


class PotentialWind(eqx.Module):
    """Exact-channel drift ``W = -M⁻¹∇U_φ`` — ``β = dU_φ/λ`` closed by construction.

    The structural prior of INVERSE_REVISION §3C: fitting this class from
    cost/timing data recovers the gravity potential on its identifiable channel,
    with the path-shape gauge ambiguity removed by construction. ``potential``
    exposes ``U_φ`` for direct comparison against the true ``gₛ·U``.
    """

    robot: Any
    net: eqx.nn.MLP

    def __init__(self, robot: Any, *, width: int = 32, depth: int = 2, key: jax.Array):
        self.robot = robot
        self.net = _scalar_mlp(robot.dof, width, depth, key)

    def potential(self, q: jax.Array) -> jax.Array:
        return self.net(q)

    def __call__(self, q: jax.Array) -> jax.Array:
        return form_to_wind(self.robot, q, jax.grad(self.net)(q))


class StreamWind(eqx.Module):
    """Rotational-channel drift (2-DoF) ``W = -M⁻¹ ⋆dψ`` — co-exact ``β``, non-zero curl.

    The path-visible complement of :class:`PotentialWind`; ``stream`` exposes ``ψ``.
    """

    robot: Any
    net: eqx.nn.MLP

    def __init__(self, robot: Any, *, width: int = 32, depth: int = 2, key: jax.Array):
        if robot.dof != 2:
            raise ValueError("StreamWind's ⋆dψ construction is 2-DoF only")
        self.robot = robot
        self.net = _scalar_mlp(2, width, depth, key)

    def stream(self, q: jax.Array) -> jax.Array:
        return self.net(q)

    def __call__(self, q: jax.Array) -> jax.Array:
        return form_to_wind(self.robot, q, star_grad(self.net, q))


class HodgeWind(eqx.Module):
    """Structured drift ``β = dU_φ + ⋆dψ`` — the full identifiable split (2-DoF)."""

    grad_part: PotentialWind
    rot_part: StreamWind

    def __init__(self, robot: Any, *, width: int = 32, depth: int = 2, key: jax.Array):
        kg, kr = jax.random.split(key)
        self.grad_part = PotentialWind(robot, width=width, depth=depth, key=kg)
        self.rot_part = StreamWind(robot, width=width, depth=depth, key=kr)

    def __call__(self, q: jax.Array) -> jax.Array:
        return self.grad_part(q) + self.rot_part(q)


class VortexWind(eqx.Module):
    """Ground-truth solenoidal drift: Gaussian vortex stream ``ψ = s·exp(-‖q-c‖²/2σ²)``.

    Purely co-exact (``β = ⋆dψ``, ``dβ = Δψ·vol ≠ 0``) — the first-order
    path-visible drift that makes the G+R regime identifiable from shape.
    """

    robot: Any
    center: jax.Array
    strength: float = eqx.field(static=True)
    width: float = eqx.field(static=True)

    def __init__(self, robot: Any, center, strength: float = 0.1, width: float = 0.7):
        self.robot = robot
        self.center = jnp.asarray(center, dtype=float)
        self.strength = float(strength)
        self.width = float(width)

    def stream(self, q: jax.Array) -> jax.Array:
        r2 = jnp.sum((q - self.center) ** 2)
        return self.strength * jnp.exp(-r2 / (2.0 * self.width**2))

    def __call__(self, q: jax.Array) -> jax.Array:
        return form_to_wind(self.robot, q, star_grad(self.stream, q))


class ScaledDistance(eqx.Module):
    """A fixed distance field rescaled by one learnable scalar (adjoint gradcheck probe)."""

    base: Any
    scale: jax.Array

    def __init__(self, base: Any, scale: Any = 1.0):
        self.base = base
        self.scale = jnp.asarray(scale)

    def __call__(self, q: jax.Array) -> jax.Array:
        return self.scale * self.base(q)
