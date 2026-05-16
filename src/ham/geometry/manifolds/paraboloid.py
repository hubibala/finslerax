"""Paraboloid manifold implementation."""
import jax
import jax.numpy as jnp

from ham.geometry.manifold import Manifold
from ham.utils import safe_norm

class Paraboloid(Manifold):
    """The paraboloid of revolution z = x^2 + y^2, embedded in R^3.
    
    Note: `exp_map` is an approximate projected retraction.
    """

    def __repr__(self) -> str:
        return "Paraboloid()"

    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 2

    def project(self, x: jax.Array) -> jax.Array:
        """Projects ambient point onto the paraboloid z = x^2 + y^2."""
        z = x[..., 0] ** 2 + x[..., 1] ** 2
        return jnp.concatenate([x[..., :2], z[..., None]], axis=-1)

    def to_tangent(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Projects ambient vector v onto Tx M."""
        n = jnp.concatenate([-2 * x[..., 0:1], -2 * x[..., 1:2], jnp.ones_like(x[..., 0:1])], axis=-1)
        n = n / safe_norm(n, axis=-1, keepdims=True)
        inner = jnp.sum(n * v, axis=-1, keepdims=True)
        return v - inner * n

    def exp_map(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Approximate exp map via projected retraction."""
        return self.retract(x, v)

    def log_map(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Approximate log map via projected ambient secant."""
        return self.to_tangent(x, y - x)

    def parallel_transport(self, x: jax.Array, y: jax.Array, v: jax.Array) -> jax.Array:
        """Approximate parallel transport via orthogonal projection onto Ty M."""
        return self.to_tangent(y, v)

    def retract(self, x: jax.Array, delta: jax.Array) -> jax.Array:
        """Retraction via projected move."""
        xy_new = x[..., :2] + delta[..., :2]
        z_new = jnp.sum(xy_new ** 2, axis=-1, keepdims=True)
        return jnp.concatenate([xy_new, z_new], axis=-1)

    def random_sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        """Samples on the paraboloid from a Gaussian base."""
        xy = jax.random.normal(key, shape + (2,)) * 2.0
        z = jnp.sum(xy ** 2, axis=-1)
        return jnp.concatenate([xy, z[..., None]], axis=-1)
