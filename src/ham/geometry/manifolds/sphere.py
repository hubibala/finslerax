"""Sphere manifold implementation."""

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.manifold import Manifold
from ham.geometry.manifolds.utils import _safe_arccos
from ham.utils import GRAD_EPS, NORM_EPS, TAYLOR_EPS, safe_norm


class Sphere(Manifold):
    """The n-sphere S^n(r) of radius r, embedded in R^{n+1}.

    The sphere uses the standard round metric inherited from ambient Euclidean space.
    Exponential and logarithmic maps are exact (closed-form geodesic formulae).
    See `spec/MATH_SPEC.md` § 4.

    Args:
        intrinsic_dim: Dimension n of the sphere. Default: 2 (S^2).
        radius: Radius r. Default: 1.0.

    Example:
        See `examples/demo_trajectories.py`.
    """

    radius: float = eqx.field(static=True)
    _intrinsic_dim: int = eqx.field(static=True)

    def __init__(self, intrinsic_dim: int = 2, radius: float = 1.0):
        self._intrinsic_dim = int(intrinsic_dim)
        self.radius = float(radius)

    def __repr__(self) -> str:
        return f"Sphere(radius={self.radius}, intrinsic_dim={self._intrinsic_dim})"

    @property
    def ambient_dim(self) -> int:
        return self._intrinsic_dim + 1

    @property
    def intrinsic_dim(self) -> int:
        return self._intrinsic_dim

    def project(self, x: jax.Array) -> jax.Array:
        """Projects x in R^{n+1} onto S^n(r) by normalizing: pi(x) = r * x / ||x||.

        For zero-length inputs (||x|| < eps), defaults to the north pole (0, ..., r).

        Args:
            x: Point in ambient space, shape `(..., n+1)`.

        Returns:
            Projected point on S^n(r), shape `(..., n+1)`.
        """
        norm = safe_norm(x, axis=-1, keepdims=True)
        safe_x = x / jnp.maximum(norm, NORM_EPS)
        is_zero = norm < NORM_EPS
        pole = jnp.zeros_like(x).at[..., -1].set(1.0)
        direction = jnp.where(is_zero, pole, safe_x)
        return self.radius * direction

    def to_tangent(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Projects ambient vector v onto Tx S^n(r): Pi(v) = v - <x, v>/r^2 * x.

        Args:
            x: Base point on S^n(r), shape `(..., n+1)`.
            v: Ambient vector, shape `(..., n+1)`.

        Returns:
            Tangent vector in Tx S^n(r), shape `(..., n+1)`.
        """
        proj = jnp.einsum("...i,...i->...", x, v)[..., None] / (self.radius**2)
        return v - proj * x

    def exp_map(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Riemannian exponential map on the sphere S^n(r)."""
        norm_v = safe_norm(v, axis=-1, keepdims=True)
        theta = norm_v / self.radius
        safe_theta = jnp.maximum(theta, TAYLOR_EPS)
        sin_theta_over_theta = jnp.where(
            theta < TAYLOR_EPS,
            1.0 - (theta**2) / 6.0,
            jnp.sin(safe_theta) / safe_theta,
        )
        cos_theta = jnp.where(
            theta < TAYLOR_EPS,
            1.0 - (theta**2) / 2.0,
            jnp.cos(safe_theta),
        )
        return cos_theta * x + sin_theta_over_theta * v

    def retract(self, x: jax.Array, delta: jax.Array) -> jax.Array:
        """Retraction via exponential map with re-projection."""
        return self.project(self.exp_map(x, delta))

    def log_map(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Inverse exponential map on S^n(r).

        The ``jnp.clip`` on the cosine argument is intentionally absent here:
        ``_safe_arccos`` already provides a stable custom JVP at x=±1 via
        an epsilon-clamped denominator, so double-regularising would skew
        the gradients for nearly-identical or antipodal point pairs.
        """
        u = jnp.sum(x * y, axis=-1, keepdims=True) / (self.radius**2)
        dist = _safe_arccos(u)
        diff = y - u * x
        norm_diff = safe_norm(diff, axis=-1, keepdims=True)
        scale = jnp.where(
            dist < TAYLOR_EPS,
            1.0 + (dist**2) / 6.0,
            dist / jnp.maximum(norm_diff / self.radius, GRAD_EPS),
        )
        return scale * diff

    def parallel_transport(self, x: jax.Array, y: jax.Array, v: jax.Array) -> jax.Array:
        """Parallel transports v from Tx S^n to Ty S^n along the geodesic x -> y."""
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        denominator = self.radius**2 + xy
        denominator = jnp.maximum(denominator, GRAD_EPS)
        y_dot_v = jnp.sum(y * v, axis=-1, keepdims=True)
        return v - (y_dot_v / denominator) * (x + y)

    def random_sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        """Samples uniformly on S^n(r) via Gaussian projection."""
        z = jax.random.normal(key, (*shape, self.ambient_dim))
        return self.project(z)
