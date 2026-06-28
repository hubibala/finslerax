"""The injected interfaces — the only contract the rest of the experiment depends on.

``medium.py``, ``constraints.py``, ``planners.py``, ``learn.py``, ``evaluate.py``
and every stage script touch *only* these protocols, never a concrete robot,
scene, or demo source. The synthetic providers (``providers/synthetic.py``) and
the real ones (``providers/real.py``, stubbed) implement the same four
interfaces, so moving from a synthetic arm to a real Franka is a provider swap,
not a rewrite.

All robot methods act on a single configuration ``q`` of shape ``(dof,)`` (joint
angles); ``vmap`` for batches. The metric is robot-driven — ``inertia`` and
``gravity`` come from here — so once the robot is real the geometry is real.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import jax

from ham.geometry.manifold import Manifold


@runtime_checkable
class Robot(Protocol):
    """An arm's kinematics, joint-space inertia, and gravity field."""

    dof: int

    @property
    def manifold(self) -> Manifold:
        """Configuration-space manifold (a flat torus for revolute joints)."""
        ...

    def fk(self, q: jax.Array) -> jax.Array:
        """End-effector position in workspace, shape ``(work_dim,)``."""
        ...

    def jacobian(self, q: jax.Array) -> jax.Array:
        """End-effector Jacobian ``d fk / d q``, shape ``(work_dim, dof)``."""
        ...

    def link_points(self, q: jax.Array) -> jax.Array:
        """Workspace positions of every joint incl. the base, shape ``(dof + 1, work_dim)``."""
        ...

    def inertia(self, q: jax.Array) -> jax.Array:
        """Joint-space mass matrix ``M(q)``, shape ``(dof, dof)``, symmetric PD."""
        ...

    def gravity(self, q: jax.Array) -> jax.Array:
        """Generalized gravity vector ``grad U(q)`` in joint space, shape ``(dof,)``."""
        ...

    def ee_orientation(self, q: jax.Array) -> jax.Array:
        """End-effector orientation (planar: the scalar link angle), for task constraints."""
        ...


@runtime_checkable
class Scene(Protocol):
    """A workspace obstacle set, exposed as a signed distance field."""

    def sdf(self, point: jax.Array) -> jax.Array:
        """Signed distance from a workspace ``point`` to the nearest obstacle (negative inside)."""
        ...


@runtime_checkable
class DistanceField(Protocol):
    """C-space clearance — the seam the obstacle barrier consumes.

    Synthetic (``AnalyticDistance``) computes it from the arm geometry and a
    ``Scene``; the real swap (``CDFDistance``) is a learned ``q -> clearance``
    map. Both are just a callable, so swapping them is one constructor argument.
    """

    def __call__(self, q: jax.Array) -> jax.Array:
        """Clearance at configuration ``q`` (scalar; negative when the arm is in collision)."""
        ...


@runtime_checkable
class DemoSource(Protocol):
    """An iterable of demonstrated paths, each of shape ``(T + 1, dof)``."""

    def __iter__(self) -> Iterator[jax.Array]:
        ...

    def __len__(self) -> int:
        ...
