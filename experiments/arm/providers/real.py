"""Real-data providers — STUBS for the follow-up session.

These lock the signatures of the real :mod:`experiments.arm.interfaces`
implementations so the synthetic→real transition is a provider swap. Filling
them in (URDF FK/inertia via pinocchio, a MotionBenchMaker/MπNets scene, a
learned CDF clearance field, logged teleop demos) is a separate research
session; nothing downstream changes when they land.
"""

from __future__ import annotations

import jax

from ham.geometry.manifold import Manifold

_TODO = "real provider is a follow-up session; implement against the locked interface"


class URDFRobot:
    """Real arm (e.g. Franka) from a URDF: FK, Jacobian, inertia, gravity via pinocchio.

    Bounded joints ⇒ the configuration manifold is a box (``EuclideanSpace`` + a
    joint-limit barrier), not a torus; the metric/solver stack is unchanged.
    """

    def __init__(self, urdf_path: str, *, gravity: float = 9.81):
        raise NotImplementedError(_TODO)

    @property
    def dof(self) -> int:
        raise NotImplementedError(_TODO)

    @property
    def manifold(self) -> Manifold:
        raise NotImplementedError(_TODO)

    def fk(self, q: jax.Array) -> jax.Array:
        raise NotImplementedError(_TODO)

    def jacobian(self, q: jax.Array) -> jax.Array:
        raise NotImplementedError(_TODO)

    def link_points(self, q: jax.Array) -> jax.Array:
        raise NotImplementedError(_TODO)

    def inertia(self, q: jax.Array) -> jax.Array:
        raise NotImplementedError(_TODO)

    def gravity(self, q: jax.Array) -> jax.Array:
        raise NotImplementedError(_TODO)

    def ee_orientation(self, q: jax.Array) -> jax.Array:
        raise NotImplementedError(_TODO)


class MotionBenchMakerScene:
    """A MotionBenchMaker / MπNets scene as a capsule/mesh signed distance field."""

    def __init__(self, scene_id: str):
        raise NotImplementedError(_TODO)

    def sdf(self, point: jax.Array) -> jax.Array:
        raise NotImplementedError(_TODO)


class CDFDistance:
    """Learned configuration-space distance field (RSS 2024 CDF): ``q -> clearance``."""

    def __init__(self, params, robot: URDFRobot):
        raise NotImplementedError(_TODO)

    def __call__(self, q: jax.Array) -> jax.Array:
        raise NotImplementedError(_TODO)


class TeleopDemos:
    """Logged human teleoperation paths, each ``(T + 1, dof)``."""

    def __init__(self, path_glob: str):
        raise NotImplementedError(_TODO)

    def __iter__(self):
        raise NotImplementedError(_TODO)

    def __len__(self):
        raise NotImplementedError(_TODO)
