"""The robot-driven configuration-space metric — a layered Randers/Finsler cost.

Three layers, each grounded and mapped to existing HAM machinery:

1. **Kinetic-energy base** ``F = sqrt(q̇ᵀ M(q) q̇)`` — the natural mechanical metric
   (mass matrix from the robot). On the Clifford ``FlatTorus`` a joint-space
   ``M(q)`` is lifted to the ambient tangent via ``E M Eᵀ``; on the intrinsic
   ``EuclideanSpace`` angle representation it is used directly.
2. **Gravity asymmetry (the novelty)** ``+ β(q)·q̇`` with the Zermelo "wind"
   ``W = -gravity_strength · ∇U(q)`` blowing *downhill*, so descending costs less
   than lifting. The causal soft clamp keeps ``‖W‖_M < 1`` (mild-wind).
3. **Obstacle conformal warp** ``ρ(q) F`` with ``ρ = 1 + α/(dist(q) - δ)₊`` over an
   injected :class:`DistanceField` (Region-Avoiding Metrics, IROS 2023). Conformal
   scaling is realised as ``H -> ρ²H, W -> W/ρ``, which leaves ``‖W‖_H`` — hence
   the mild-wind clamp — invariant and keeps the metric a valid Randers metric
   (so the eikonal cross-check still applies on the 2-D angle grid).

:class:`ArmMetric` is an :class:`~ham.geometry.metric.AsymmetricMetric`, so every
HAM solver (AVBD, eikonal, ``ExponentialMap``, Gauss-Newton) consumes it directly,
and the injected ``dist`` field is a PyTree child — its parameters are visible to
the implicit-diff adjoint, which is what makes the obstacle field learnable
(Stage D).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.manifold import Manifold
from ham.geometry.manifolds import EuclideanSpace, FlatTorus
from ham.geometry.metric import AsymmetricMetric
from ham.utils import GRAD_EPS, WIND_STIFFNESS, causal_wind_scale


def zermelo_cost(H: jax.Array, W: jax.Array, lam: jax.Array, v: jax.Array) -> jax.Array:
    """Zermelo travel-time ``F`` for displacement ``v`` given sea ``H``, wind ``W``, ``λ``.

    The exact body of :meth:`ham.geometry.zoo.Randers.metric_fn`, factored out so
    :class:`ArmMetric` and any segment-cost consumer share one definition.
    """
    v_sq_raw = jnp.sum(v**2, axis=-1)
    is_zero = v_sq_raw < GRAD_EPS
    v_safe = jnp.where(is_zero, v + jnp.sqrt(GRAD_EPS), v)

    Hv = jnp.matmul(H, v_safe)
    HW = jnp.matmul(H, W)
    v_sq_h = jnp.sum(v_safe * Hv, axis=-1)
    W_dot_v = jnp.sum(v_safe * HW, axis=-1)

    discriminant = lam * v_sq_h + W_dot_v**2
    cost = (jnp.sqrt(jnp.maximum(discriminant, GRAD_EPS)) - W_dot_v) / lam
    return jnp.where(is_zero, 0.0, cost)


class ArmMetric(AsymmetricMetric):
    """Layered kinetic-energy + gravity-Randers + obstacle-barrier C-space metric.

    Args:
        manifold: Configuration manifold. ``FlatTorus(n)`` (Clifford, ambient 2n)
            lifts the joint inertia; ``EuclideanSpace(n)`` is the intrinsic-angle
            representation used for the 2-D eikonal grid and the spray oracle.
        robot: Provides ``inertia(q)`` and ``gravity(q)``.
        dist: Optional :class:`DistanceField`; enables the obstacle barrier.
        gravity_strength: Scales the gravity wind ``W = -strength · ∇U``.
        alpha: Barrier strength (annealed by continuation for stiff obstacles).
        delta: Barrier safety margin; ``ρ`` blows up as ``dist -> δ``.
        epsilon: Causal margin, ``max_speed = 1 - epsilon``.
    """

    robot: Any
    dist: Any | None
    gravity_strength: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    delta: float = eqx.field(static=True)
    barrier_width: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    wind_stiffness: float = eqx.field(static=True)
    use_gravity: bool = eqx.field(static=True)
    use_obstacle: bool = eqx.field(static=True)

    def __init__(
        self,
        manifold: Manifold,
        robot: Any,
        dist: Any | None = None,
        *,
        gravity_strength: float = 0.0,
        alpha: float = 0.0,
        delta: float = 0.05,
        barrier_width: float = 0.05,
        epsilon: float = 1e-5,
        wind_stiffness: float = WIND_STIFFNESS,
    ):
        super().__init__(manifold=manifold)
        self.robot = robot
        self.dist = dist
        self.gravity_strength = float(gravity_strength)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.barrier_width = float(barrier_width)
        self.epsilon = float(epsilon)
        self.wind_stiffness = float(wind_stiffness)
        self.use_gravity = self.gravity_strength != 0.0
        self.use_obstacle = dist is not None and self.alpha != 0.0

    def _q_and_frame(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Joint angles ``q`` and the ambient tangent frame ``E`` (identity off the torus)."""
        if isinstance(self.manifold, FlatTorus):
            return self.manifold.to_angles(x), self.manifold.tangent_frame(x)
        return x, jnp.eye(x.shape[-1], dtype=x.dtype)

    def _rho(self, q: jax.Array) -> jax.Array:
        """Conformal barrier ``ρ(q) = 1 + α/gap`` with a smooth softplus gap (1 if disabled).

        The gap ``w·softplus((dist - δ)/w)`` approaches ``(dist - δ)₊`` but stays
        strictly positive and — crucially — keeps a restoring gradient *inside*
        collision, so continuation can pull a colliding init out. A hard
        ``max(dist - δ, 0)`` floor zeros that gradient and traps the path.
        """
        if not self.use_obstacle:
            return jnp.asarray(1.0)
        w = self.barrier_width
        gap = w * jax.nn.softplus((self.dist(q) - self.delta) / w)
        return 1.0 + self.alpha / gap

    def zermelo_data(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        q, E = self._q_and_frame(x)

        M = self.robot.inertia(q)
        H_base = E @ M @ E.T
        H_base = 0.5 * (H_base + H_base.T)

        if self.use_gravity:
            W = E @ (-self.gravity_strength * self.robot.gravity(q))
        else:
            W = jnp.zeros(x.shape, dtype=x.dtype)

        # Causal soft clamp against the base sea (conformal-invariant: ‖W‖_H is
        # unchanged by the ρ-fold below, so clamping here is correct).
        w_norm = jnp.sqrt(jnp.maximum(jnp.sum(W * (H_base @ W)), GRAD_EPS))
        max_speed = 1.0 - self.epsilon
        scale = causal_wind_scale(w_norm, max_speed, self.wind_stiffness)
        W_safe = W * scale

        rho = self._rho(q)
        H = rho**2 * H_base
        W_prime = W_safe / rho
        lam = 1.0 - jnp.sum(W_prime * (H @ W_prime))
        return H, W_prime, jnp.maximum(lam, GRAD_EPS)

    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        H, W, lam = self.zermelo_data(x)
        return zermelo_cost(H, W, lam, v)


def build_arm_metric(
    robot: Any,
    dist: Any | None = None,
    *,
    manifold: Manifold | None = None,
    gravity_strength: float = 0.0,
    alpha: float = 0.0,
    delta: float = 0.05,
    barrier_width: float = 0.05,
    epsilon: float = 1e-5,
) -> ArmMetric:
    """Build an :class:`ArmMetric`, defaulting to the robot's Clifford torus.

    Pass ``manifold=EuclideanSpace(robot.dof)`` for the intrinsic-angle
    representation (the 2-D eikonal grid and the spray oracle).
    """
    manifold = robot.manifold if manifold is None else manifold
    return ArmMetric(
        manifold,
        robot,
        dist,
        gravity_strength=gravity_strength,
        alpha=alpha,
        delta=delta,
        barrier_width=barrier_width,
        epsilon=epsilon,
    )


def angle_manifold(robot: Any) -> EuclideanSpace:
    """Intrinsic-angle configuration manifold ``EuclideanSpace(dof)`` for ``robot``."""
    return EuclideanSpace(robot.dof)
