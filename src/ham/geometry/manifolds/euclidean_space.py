"""Euclidean space manifold implementation."""
import jax
import equinox as eqx

from ham.geometry.manifold import Manifold

class EuclideanSpace(Manifold):
    """Flat Euclidean space R^N.

    Args:
        dim: Dimension N.
    """
    _dim: int = eqx.field(static=True)

    def __init__(self, dim: int):
        self._dim = int(dim)

    def __repr__(self) -> str:
        return f"EuclideanSpace(dim={self._dim})"

    @property
    def ambient_dim(self) -> int:
        return self._dim

    @property
    def intrinsic_dim(self) -> int:
        return self._dim

    def project(self, x: jax.Array) -> jax.Array:
        """Identity projection."""
        return x

    def to_tangent(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Identity tangent projection."""
        return v

    def exp_map(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Flat exponential map: x + v."""
        return x + v

    def retract(self, x: jax.Array, delta: jax.Array) -> jax.Array:
        """Flat retraction: x + delta."""
        return x + delta

    def log_map(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Flat log map: y - x."""
        return y - x

    def parallel_transport(self, x: jax.Array, y: jax.Array, v: jax.Array) -> jax.Array:
        """Identity parallel transport."""
        return v

    def random_sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        """Gaussian random sampling."""
        return jax.random.normal(key, shape + (self._dim,))
