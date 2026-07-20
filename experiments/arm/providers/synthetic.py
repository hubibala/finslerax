"""Synthetic providers: an analytic planar arm, circle obstacles, demos.

These instantiate the :mod:`experiments.arm.interfaces` protocols with closed-form
physics so the machinery can be proven before real data. The real counterparts
(``providers/real.py``) implement the same interfaces against a URDF / learned
fields, with nothing downstream changed.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.manifolds import FlatTorus


class PlanarArm(eqx.Module):
    """An ``n``-link planar arm with point masses at the link tips.

    Joint ``i`` is the angle of link ``i`` relative to link ``i-1``; the absolute
    link angles are the cumulative sums. With a point mass ``m_i`` at the tip of
    link ``i``, the kinetic energy ``½ q̇ᵀ M(q) q̇`` gives the mass matrix
    ``M(q) = Σ_i m_i J_iᵀ J_i`` (``J_i = ∂p_i/∂q``), and the potential
    ``U(q) = g Σ_i m_i y_i`` gives the joint-space gravity vector ``∇U(q)``. Both
    are computed by autodiff of the kinematics, so they stay consistent by
    construction. Revolute joints ⇒ the configuration space is ``FlatTorus(n)``.

    Physical parameters are static (the arm is fixed; Stage D learns the metric /
    obstacle field, not the arm).
    """

    n_links: int = eqx.field(static=True)
    lengths: tuple = eqx.field(static=True)
    masses: tuple = eqx.field(static=True)
    g: float = eqx.field(static=True)

    def __init__(self, n_links, lengths=None, masses=None, g=1.0):
        self.n_links = int(n_links)
        self.lengths = tuple(
            float(x) for x in (lengths if lengths is not None else [1.0] * n_links)
        )
        self.masses = tuple(
            float(x) for x in (masses if masses is not None else [1.0] * n_links)
        )
        self.g = float(g)
        if len(self.lengths) != self.n_links or len(self.masses) != self.n_links:
            raise ValueError("lengths/masses must each have n_links entries")

    @property
    def dof(self) -> int:
        return self.n_links

    @property
    def manifold(self) -> FlatTorus:
        return FlatTorus(self.n_links)

    def link_points(self, q: jax.Array) -> jax.Array:
        """Joint positions ``(n+1, 2)``, base at the origin out to the end-effector."""
        lengths = jnp.asarray(self.lengths, dtype=q.dtype)
        phi = jnp.cumsum(q)  # absolute link angles
        segs = lengths[:, None] * jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)
        tips = jnp.cumsum(segs, axis=0)  # (n, 2)
        base = jnp.zeros((1, 2), dtype=q.dtype)
        return jnp.concatenate([base, tips], axis=0)

    def fk(self, q: jax.Array) -> jax.Array:
        """End-effector position."""
        return self.link_points(q)[-1]

    def jacobian(self, q: jax.Array) -> jax.Array:
        """End-effector Jacobian (autodiff of :meth:`fk`)."""
        return jax.jacobian(self.fk)(q)

    def ee_orientation(self, q: jax.Array) -> jax.Array:
        """Orientation of the last link — the cumulative joint angle ``Σ θ_i``."""
        return jnp.sum(q)

    def inertia(self, q: jax.Array) -> jax.Array:
        """Mass matrix ``M(q) = Σ_i m_i J_iᵀ J_i`` from the tip Jacobians."""
        masses = jnp.asarray(self.masses, dtype=q.dtype)
        tips = lambda qq: self.link_points(qq)[1:]
        J = jax.jacobian(tips)(q)  # (n, 2, n)
        return jnp.einsum("i,iaj,iak->jk", masses, J, J)

    def potential(self, q: jax.Array) -> jax.Array:
        """Gravitational potential ``U(q) = g Σ_i m_i y_i`` (height of each tip mass)."""
        masses = jnp.asarray(self.masses, dtype=q.dtype)
        heights = self.link_points(q)[1:, 1]
        return self.g * jnp.sum(masses * heights)

    def gravity(self, q: jax.Array) -> jax.Array:
        """Joint-space gravity vector ``∇U(q)`` (autodiff of :meth:`potential`)."""
        return jax.grad(self.potential)(q)


class CircleScene(eqx.Module):
    """Workspace obstacles as a set of circles, exposed as a signed distance field."""

    circles: jax.Array  # (M, 3): (cx, cy, radius)

    def __init__(self, circles):
        self.circles = jnp.asarray(circles, dtype=float)
        if self.circles.ndim != 2 or self.circles.shape[1] != 3:
            raise ValueError("circles must have shape (M, 3): cx, cy, radius")

    def sdf(self, point: jax.Array) -> jax.Array:
        """Signed distance from a workspace point to the nearest circle surface."""
        centers = self.circles[:, :2]
        radii = self.circles[:, 2]
        return jnp.min(jnp.linalg.norm(point - centers, axis=-1) - radii)


class AnalyticDistance(eqx.Module):
    """C-space clearance: the minimum obstacle SDF over points sampled along the arm.

    Provider-agnostic in the scene — it works for any :class:`Scene` SDF — and
    implements the :class:`DistanceField` seam the obstacle barrier consumes.
    """

    robot: PlanarArm
    scene: CircleScene
    n_samples: int = eqx.field(static=True, default=16)

    def __call__(self, q: jax.Array) -> jax.Array:
        pts = self.robot.link_points(q)  # (n+1, 2)
        a, b = pts[:-1], pts[1:]
        ts = jnp.linspace(0.0, 1.0, self.n_samples)
        # Points along every link segment: (n_seg, n_samples, 2).
        samples = a[:, None, :] + ts[None, :, None] * (b - a)[:, None, :]
        flat = samples.reshape(-1, 2)
        return jnp.min(jax.vmap(self.scene.sdf)(flat))


class GroundTruthDemos:
    """A finite set of demonstrated paths, each ``(T + 1, dof)``.

    Holds paths directly (e.g. solved under a *known* metric for Stage-D recovery
    experiments). Use :meth:`from_solver` to generate them from a solve callable.
    """

    def __init__(self, paths):
        self.paths = [jnp.asarray(p, dtype=float) for p in paths]

    @classmethod
    def from_solver(cls, solve_fn, endpoints):
        """Build demos by solving ``solve_fn(start, goal)`` for each endpoint pair.

        Args:
            solve_fn: ``(start, goal) -> path`` of shape ``(T + 1, dof)``.
            endpoints: iterable of ``(start, goal)`` configuration pairs.
        """
        return cls([solve_fn(jnp.asarray(s), jnp.asarray(g)) for s, g in endpoints])

    def __iter__(self):
        return iter(self.paths)

    def __len__(self):
        return len(self.paths)
