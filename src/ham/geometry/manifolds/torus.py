"""Torus manifold implementation."""
import jax
import jax.numpy as jnp
import equinox as eqx

from ham.geometry.manifold import Manifold
from ham.utils import NORM_EPS, TAYLOR_EPS, safe_norm

class Torus(Manifold):
    """The 2-torus T^2 embedded in R^3.

    Parametrized by major radius R (distance from center to tube center) and 
    minor radius r (tube radius): 
    (x, y, z) = ((R + r cos v) cos u, (R + r cos v) sin u, r sin v).

    Args:
        major_R: Major radius R. Default: 2.0.
        minor_r: Minor radius r. Default: 1.0.

    Note:
        `log_map` uses an approximate angular parameterization. No exact closed-form 
        Riemannian inverse exponential map is implemented.
    """
    R: float = eqx.field(static=True)
    r: float = eqx.field(static=True)

    def __init__(self, major_R: float = 2.0, minor_r: float = 1.0):
        self.R = float(major_R)
        self.r = float(minor_r)

    def __repr__(self) -> str:
        return f"Torus(R={self.R}, r={self.r})"

    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 2

    def project(self, x: jax.Array) -> jax.Array:
        """Projects ambient point onto the torus surface."""
        rho = safe_norm(x[..., :2], axis=-1)
        dir_xy = jnp.where(
            rho[..., None] > NORM_EPS,
            x[..., :2] / jnp.maximum(rho, NORM_EPS)[..., None],
            jnp.zeros_like(x[..., :2]).at[..., 0].set(1.0),
        )

        dist = rho - self.R
        n_xy = jnp.concatenate([dir_xy * dist[..., None], x[..., 2:3]], axis=-1)
        n_norm = safe_norm(n_xy, axis=-1, keepdims=True)

        n = jnp.where(
            n_norm > NORM_EPS,
            n_xy / jnp.maximum(n_norm, NORM_EPS),
            jnp.zeros_like(x).at[..., 2].set(1.0),
        )

        projected = x - n * (n_norm - self.r)
        fallback = jnp.zeros_like(x).at[..., 0].set(self.R + self.r)
        use_fallback = rho < TAYLOR_EPS
        return jnp.where(use_fallback[..., None], fallback, projected)

    def to_tangent(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Projects ambient vector v onto Tx T^2."""
        rho = safe_norm(x[..., :2], axis=-1)
        dir_xy = jnp.where(
            rho[..., None] > NORM_EPS,
            x[..., :2] / jnp.maximum(rho, NORM_EPS)[..., None],
            jnp.zeros_like(x[..., :2]).at[..., 0].set(1.0),
        )
        dist = rho - self.R
        n_xy = jnp.concatenate([dir_xy * dist[..., None], x[..., 2:3]], axis=-1)
        n_norm = safe_norm(n_xy, axis=-1, keepdims=True)

        n = jnp.where(
            n_norm > NORM_EPS,
            n_xy / jnp.maximum(n_norm, NORM_EPS),
            jnp.zeros_like(x).at[..., 2].set(1.0),
        )
        inner = jnp.sum(n * v, axis=-1, keepdims=True)
        return v - inner * n

    def exp_map(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Approximate exp map via projected retraction."""
        return self.project(x + v)

    def log_map(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Approximate log map via angular parameterization."""
        ux = jnp.atan2(x[..., 1], x[..., 0])
        uy = jnp.atan2(y[..., 1], y[..., 0])
        rho_x = safe_norm(x[..., :2], axis=-1)
        rho_y = safe_norm(y[..., :2], axis=-1)
        vx = jnp.atan2(x[..., 2], rho_x - self.R)
        vy = jnp.atan2(y[..., 2], rho_y - self.R)
        du = (uy - ux + jnp.pi) % (2 * jnp.pi) - jnp.pi
        dv = (vy - vx + jnp.pi) % (2 * jnp.pi) - jnp.pi
        sin_u, cos_u = jnp.sin(ux), jnp.cos(ux)
        sin_v, cos_v = jnp.sin(vx), jnp.cos(vx)
        e_u = jnp.stack([-(self.R + self.r * cos_v) * sin_u, 
                          (self.R + self.r * cos_v) * cos_u, 
                          jnp.zeros_like(ux)], axis=-1)
        e_v = jnp.stack([-self.r * sin_v * cos_u, 
                         -self.r * sin_v * sin_u, 
                          self.r * cos_v], axis=-1)
        return du[..., None] * e_u + dv[..., None] * e_v

    def parallel_transport(self, x: jax.Array, y: jax.Array, v: jax.Array) -> jax.Array:
        """Approximate parallel transport via orthogonal projection onto Ty M."""
        return self.to_tangent(y, v)

    def retract(self, x: jax.Array, delta: jax.Array) -> jax.Array:
        """Retraction via projection."""
        return self.project(x + delta)

    def random_sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        """Samples from uniform angular coordinates."""
        u_key, v_key = jax.random.split(key)
        u = jax.random.uniform(u_key, shape, minval=0, maxval=2 * jnp.pi)
        v = jax.random.uniform(v_key, shape, minval=0, maxval=2 * jnp.pi)
        x = (self.R + self.r * jnp.cos(v)) * jnp.cos(u)
        y = (self.R + self.r * jnp.cos(v)) * jnp.sin(u)
        z = self.r * jnp.sin(v)
        return jnp.stack([x, y, z], axis=-1)
