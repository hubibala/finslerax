import jax._src.cudnn.scaled_matmul_stablehlo
import jax
import jax.numpy as jnp
from .manifold import Manifold
from ..utils.math import safe_norm

class Sphere(Manifold):
    def __init__(self, radius: float = 1.0):
        self.radius = radius

    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 2

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        norm = jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
        return self.radius * x / norm

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Normal is x / ||x||, but since x is already on sphere, normal = x / radius
        proj = jnp.einsum('...i,...i->...', x, v)[..., None] / (self.radius ** 2)
        return v - proj * x

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        # Standard sphere retraction (very accurate for small delta)
        norm_delta = jnp.linalg.norm(delta, axis=-1, keepdims=True)
        safe_norm = jnp.maximum(norm_delta, 1e-8)
        sin_theta_over_theta = jnp.sin(safe_norm) / safe_norm
        cos_theta = jnp.cos(safe_norm)
        return cos_theta * x + sin_theta_over_theta * delta

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        # Gaussian projection + normalization
        gauss = jax.random.normal(key, shape=shape + (self.ambient_dim,))
        return self.project(gauss)

class Torus(Manifold):
    def __init__(self, major_R: float = 2.0, minor_r: float = 1.0):
        self.R = major_R
        self.r = minor_r

    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 2

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        rho = jnp.linalg.norm(x[:2])
        
        # Safe direction (avoid div-by-zero)
        dir_xy = jnp.where(
            rho > 1e-8,
            x[:2] / rho,
            jnp.array([1.0, 0.0])   # arbitrary safe direction
        )
        
        dist = rho - self.R
        n_xy = jnp.concatenate([dir_xy * dist, jnp.array([x[2]])])
        n_norm = jnp.linalg.norm(n_xy)
        
        # Avoid div-by-zero in normal
        n = jnp.where(
            n_norm > 1e-8,
            n_xy / n_norm,
            jnp.zeros(3)
        )
        
        # Single projection step
        projected = x - n * (n_norm - self.r)
        
        # Fallback only when rho is extremely small (use where)
        fallback = jnp.array([self.R + self.r, 0.0, 0.0])
        use_fallback = rho < 1e-6
        return jnp.where(use_fallback, fallback, projected)

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        rho = jnp.linalg.norm(x[:2])
        
        # Safe direction (avoid div-by-zero)
        dir_xy = jnp.where(
            rho > 1e-8,
            x[:2] / rho,
            jnp.zeros(2)   # arbitrary safe direction
        )
        dist = rho - self.R
        n_xy = jnp.concatenate([dir_xy * dist, jnp.array([x[2]])])
        n_norm = jnp.linalg.norm(n_xy)
        n = jnp.where(
            n_norm > 1e-8,
            n_xy / n_norm,
            jnp.zeros(3)
        )
        return v - jnp.dot(n, v) * n

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        # Simple projected retraction — reliable for optimization
        return self.project(x + delta)

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        u_key, v_key = jax.random.split(key)
        u = jax.random.uniform(u_key, shape, minval=0, maxval=2*jnp.pi)
        v = jax.random.uniform(v_key, shape, minval=0, maxval=2*jnp.pi)
        x = (self.R + self.r * jnp.cos(v)) * jnp.cos(u)
        y = (self.R + self.r * jnp.cos(v)) * jnp.sin(u)
        z = self.r * jnp.sin(v)
        return jnp.stack([x, y, z], axis=-1)

class Paraboloid(Manifold):
    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 2

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0], x[1], x[0]**2 + x[1]**2])

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        n = jnp.array([-2*x[0], -2*x[1], 1.0])
        n = n / jnp.linalg.norm(n)
        return v - jnp.dot(n, v) * n

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        # Exact retraction for paraboloid: update xy freely, set z = x'^2 + y'^2
        xy_new = x[:2] + delta[:2]
        z_new = jnp.sum(xy_new**2)
        return jnp.concatenate([xy_new, jnp.array([z_new])])

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        xy = jax.random.normal(key, shape + (2,)) * 2.0  # adjust scale
        z = jnp.sum(xy**2, axis=-1)
        return jnp.concatenate([xy, z[..., None]], axis=-1)