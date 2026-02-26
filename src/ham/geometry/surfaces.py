import jax._src.cudnn.scaled_matmul_stablehlo
import jax
import jax.numpy as jnp
import equinox as eqx
from .manifold import Manifold
from ..utils.math import safe_norm

class Sphere(Manifold):
    radius: float = eqx.field(static=True)

    def __init__(self, radius: float = 1.0):
        self.radius = radius

    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 2

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        mask = norm > 1e-8
        safe_x = x / jnp.where(mask, norm, 1.0)
        direction = jnp.where(mask, safe_x, jnp.array([0.0, 0.0, 1.0]))
        return self.radius * direction

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
    R: float = eqx.field(static=True)
    r: float = eqx.field(static=True)

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
        
        # If n_norm is 0, we are on the skeleton (center of the tube).
        # We must choose a direction to push it to the surface.
        # We pick the "up" direction in the tube's local cross-section.
        n = jnp.where(
            n_norm > 1e-8,
            n_xy / n_norm,
            jnp.array([0.0, 0.0, 1.0])
        )
        
        # Single projection step
        projected = x - n * (n_norm - self.r)
        
        # Fallback only when rho is extremely small (near Z-axis)
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
        
        # Consistent normal with project()
        n = jnp.where(
            n_norm > 1e-8,
            n_xy / n_norm,
            jnp.array([0.0, 0.0, 1.0])
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
class Hyperboloid(Manifold):
    """
    The upper sheet of the hyperboloid in Minkowski space.
    -x0^2 + x1^2 + ... + xn^2 = -1, x0 > 0
    """
    # Store dimension as static metadata for JAX/Equinox
    _intrinsic_dim: int = eqx.field(static=True)

    def __init__(self, intrinsic_dim: int = 2):
        self._intrinsic_dim = intrinsic_dim

    # Satisfy Manifold ABC property requirement
    @property
    def intrinsic_dim(self) -> int:
        return self._intrinsic_dim

    @property
    def ambient_dim(self) -> int:
        return self._intrinsic_dim + 1

    def _minkowski_dot(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Computes the Minkowski inner product: -u0v0 + u1v1 + ..."""
        return -u[..., 0] * v[..., 0] + jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)

    def _minkowski_norm(self, u: jnp.ndarray) -> jnp.ndarray:
        """Computes sqrt(<u,u>_L) for space-like vectors."""
        sq_norm = self._minkowski_dot(u, u)
        return jnp.sqrt(jnp.maximum(sq_norm, 1e-12))

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Project ambient points onto the Hyperboloid.
        """
        sq_norm = self._minkowski_dot(x, x)
        
        # Valid candidates for scaling: Time-like AND Upper Sheet
        is_valid_candidate = (sq_norm < -1e-6) & (x[..., 0] > 0)
        
        # Branch 1: Scaling (for valid candidates)
        safe_sq_norm = jnp.minimum(sq_norm, -1e-6)
        denom = jnp.sqrt(-safe_sq_norm)
        x_scaled = x / denom[..., None]
        
        # Branch 2: Lifting (for space-like or inverted points)
        x_spatial = x[..., 1:]
        x_spatial_sq = jnp.sum(x_spatial**2, axis=-1)
        x0_new = jnp.sqrt(1.0 + x_spatial_sq)
        x_lifted = jnp.concatenate([x0_new[..., None], x_spatial], axis=-1)
        
        mask = is_valid_candidate[..., None]
        return jnp.where(mask, x_scaled, x_lifted)

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Tangent space T_x H = { v | <x, v>_L = 0 }
        inner = self._minkowski_dot(x, v)
        return v + inner[..., None] * x

    def exp_map(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        norm_v = self._minkowski_norm(v)
        norm_v_safe = jnp.maximum(norm_v, 1e-6)
        
        res = (jnp.cosh(norm_v)[..., None] * x + 
               jnp.sinh(norm_v)[..., None] * (v / norm_v_safe[..., None]))
               
        return jnp.where(norm_v[..., None] < 1e-6, x, res)

    def log_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        xy = self._minkowski_dot(x, y)
        xy = jnp.minimum(xy, -1.0 - 1e-6)
        dist = jnp.arccosh(-xy)
        
        u = y + xy[..., None] * x
        norm_u = self._minkowski_norm(u)
        
        res = dist[..., None] * u / jnp.maximum(norm_u, 1e-6)[..., None]
        return jnp.where(dist[..., None] < 1e-6, jnp.zeros_like(x), res)

    def parallel_transport(self, x: jnp.ndarray, y: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        xy = self._minkowski_dot(x, y)
        yv = self._minkowski_dot(y, v)
        denom = 1.0 - xy
        scale = yv / denom
        return v + scale[..., None] * (x + y)

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        return self.exp_map(x, delta)

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        spat_dim = self.intrinsic_dim
        v_spatial = jax.random.normal(key, shape + (spat_dim,))
        norm_v = jnp.linalg.norm(v_spatial, axis=-1, keepdims=True)
        safe_norm = jnp.maximum(norm_v, 1e-6)
        
        x0 = jnp.cosh(norm_v)
        x_rest = jnp.sinh(norm_v) * (v_spatial / safe_norm)
        return jnp.concatenate([x0, x_rest], axis=-1)
    
    def metric_tensor(self, x: jnp.ndarray) -> jnp.ndarray:
        m = jnp.eye(self.ambient_dim)
        m = m.at[0, 0].set(-1.0)
        return m

class EuclideanSpace(Manifold):
    """Flat Euclidean space R^N."""
    _dim: int = eqx.field(static=True)

    def __init__(self, dim: int):
        self._dim = dim

    @property
    def ambient_dim(self) -> int:
        return self._dim

    @property
    def intrinsic_dim(self) -> int:
        return self._dim

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return v

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        return x + delta

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        return jax.random.normal(key, shape + (self._dim,))
        