"""Flat (Clifford) torus manifold implementation."""

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.manifold import Manifold
from ham.utils import NORM_EPS, safe_norm


class FlatTorus(Manifold):
    r"""The flat n-torus T^n = (S^1)^n via the Clifford embedding in R^{2n}.

    Each angle is placed on its own unit circle,
    ``q = (theta_1, ..., theta_n) -> (cos theta_1, sin theta_1, ..., cos theta_n,
    sin theta_n)``, so an ambient point is a stack of ``n`` unit 2-vectors
    ("blocks"). Unlike the donut :class:`~ham.geometry.manifolds.torus.Torus`
    (T^2 in R^3, which carries spurious Gaussian curvature), the Clifford torus is
    intrinsically flat yet globally non-trivial (pi_1 = Z^n). It is the exact
    configuration space of an ``n``-revolute-joint arm, and serves as a
    flat-but-topologically-rich high-dimensional test manifold (ambient ``2n``,
    intrinsic ``n``).

    Flatness makes every operation closed-form and exact, including ``log_map``
    (the donut's is only approximate). The geodesic length between ``q0`` and
    ``q1`` is the Euclidean norm of the wrapped angle difference — an exact ground
    truth (validation rung L0). A robot-driven metric is layered on top by lifting
    a joint-space tensor ``M(q)`` through :meth:`tangent_frame`.

    Only genuinely revolute joints live on T^n; bounded joints (e.g. a Franka) are
    better modelled as a box (:class:`EuclideanSpace` + limit barrier) on the same
    metric/solver stack.

    Args:
        dim: Number of joints ``n`` (intrinsic dimension); ambient dimension is ``2n``.
    """

    _dim: int = eqx.field(static=True)

    def __init__(self, dim: int):
        if int(dim) < 1:
            raise ValueError(f"FlatTorus dim must be >= 1, got {dim}")
        self._dim = int(dim)

    def __repr__(self) -> str:
        return f"FlatTorus(dim={self._dim})"

    @property
    def ambient_dim(self) -> int:
        return 2 * self._dim

    @property
    def intrinsic_dim(self) -> int:
        return self._dim

    # -- internal block (un)packing: (..., 2n) <-> (..., n, 2) ----------------
    def _blocks(self, x: jax.Array) -> jax.Array:
        return x.reshape((*x.shape[:-1], self._dim, 2))

    def _flat(self, b: jax.Array) -> jax.Array:
        return b.reshape((*b.shape[:-2], 2 * self._dim))

    def _unit_blocks(self, x: jax.Array) -> jax.Array:
        """Per-block unit vectors, with a (1, 0) fallback for zero blocks."""
        b = self._blocks(x)
        norm = safe_norm(b, axis=-1, keepdims=True)
        return jnp.where(
            norm > NORM_EPS,
            b / jnp.maximum(norm, NORM_EPS),
            jnp.zeros_like(b).at[..., 0].set(1.0),
        )

    # -- Manifold interface ---------------------------------------------------
    def project(self, x: jax.Array) -> jax.Array:
        """Normalise each 2-block onto its unit circle."""
        return self._flat(self._unit_blocks(x))

    def to_tangent(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Remove the radial component of each block: ``v_i - (v_i . x_i) x_i``."""
        xhat = self._unit_blocks(x)
        vb = self._blocks(v)
        radial = jnp.sum(vb * xhat, axis=-1, keepdims=True) * xhat
        return self._flat(vb - radial)

    def log_map(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Exact inverse exponential map: per-joint wrapped angle difference.

        ``atan2`` keeps each increment in ``(-pi, pi]`` (seam-safe and invariant
        to the scale of ``y``), carried on the unit circle tangent ``(-x_i1, x_i0)``.
        """
        xhat = self._unit_blocks(x)
        yb = self._blocks(y)
        x0, x1 = xhat[..., 0], xhat[..., 1]
        y0, y1 = yb[..., 0], yb[..., 1]
        cross = x0 * y1 - x1 * y0
        dot = x0 * y0 + x1 * y1
        dtheta = jnp.arctan2(cross, dot)
        e = jnp.stack([-x1, x0], axis=-1)
        return self._flat(dtheta[..., None] * e)

    def exp_map(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Exact exponential map: rotate each block by its tangential increment.

        The increment ``a_i = v_i . e_i`` ignores any radial part of ``v``; the
        rotation is exact, so the result is always on the manifold.
        """
        xhat = self._unit_blocks(x)
        vb = self._blocks(v)
        x0, x1 = xhat[..., 0], xhat[..., 1]
        e = jnp.stack([-x1, x0], axis=-1)
        a = jnp.sum(vb * e, axis=-1, keepdims=True)
        new_b = jnp.cos(a) * xhat + jnp.sin(a) * e
        return self._flat(new_b)

    def retract(self, x: jax.Array, delta: jax.Array) -> jax.Array:
        """Retraction via the exact exponential map (rotation stays on T^n)."""
        return self.exp_map(x, delta)

    def parallel_transport(self, x: jax.Array, y: jax.Array, v: jax.Array) -> jax.Array:
        """Exact holonomy-free transport: preserve each angular component, re-express at ``y``."""
        xhat = self._unit_blocks(x)
        yhat = self._unit_blocks(y)
        vb = self._blocks(v)
        ex = jnp.stack([-xhat[..., 1], xhat[..., 0]], axis=-1)
        ey = jnp.stack([-yhat[..., 1], yhat[..., 0]], axis=-1)
        a = jnp.sum(vb * ex, axis=-1, keepdims=True)
        return self._flat(a * ey)

    def random_sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        """Uniform angles per joint, embedded on the Clifford torus."""
        theta = jax.random.uniform(
            key, (*shape, self._dim), minval=0.0, maxval=2.0 * jnp.pi
        )
        return self.embed_angles(theta)

    # -- Clifford-torus conveniences (exact angle <-> ambient) ----------------
    def to_angles(self, x: jax.Array) -> jax.Array:
        """Joint angles ``(..., n)`` in ``(-pi, pi]`` from ambient points ``(..., 2n)``."""
        b = self._blocks(x)
        return jnp.arctan2(b[..., 1], b[..., 0])

    def embed_angles(self, theta: jax.Array) -> jax.Array:
        """Embed joint angles ``(..., n)`` onto the Clifford torus ``(..., 2n)``."""
        b = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
        return self._flat(b)

    def tangent_frame(self, x: jax.Array) -> jax.Array:
        """Orthonormal frame ``E(x)``, shape ``(2n, n)``, of the tangent space at ``x``.

        Column ``i`` is joint ``i``'s circle tangent placed in its block. ``E(x)``
        lifts a joint velocity to the ambient tangent (``E q_dot``) and a
        joint-space tensor to the ambient metric (``E M E^T``). Single-point only.
        """
        xhat = self._unit_blocks(x)  # (n, 2)
        e = jnp.stack([-xhat[:, 1], xhat[:, 0]], axis=-1)  # circle tangents (n, 2)
        # Scatter e[i] into block i of a (2n, n) matrix.
        frame = jnp.einsum("ic,ij->icj", e, jnp.eye(self._dim, dtype=x.dtype))
        return frame.reshape(2 * self._dim, self._dim)
