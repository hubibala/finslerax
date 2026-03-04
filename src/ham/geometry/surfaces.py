import jax._src.cudnn.scaled_matmul_stablehlo
import jax
import jax.numpy as jnp
import equinox as eqx
from .manifold import Manifold
from ..utils.math import safe_norm

@jax.custom_jvp
def _safe_norm_jvp(x):
    return jnp.linalg.norm(x, axis=-1, keepdims=True)

@_safe_norm_jvp.defjvp
def _safe_norm_jvp_def(primals, tangents):
    x, = primals
    x_dot, = tangents
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    out = jnp.where(norm < 1e-12, jnp.zeros_like(norm), jnp.sum(x * x_dot, axis=-1, keepdims=True) / jnp.maximum(norm, 1e-12))
    return norm, out

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
        # Standard sphere retraction (exponential map)
        norm_delta = _safe_norm_jvp(delta)
        # Avoid NaN gradients at norm_delta=0 by using a stable Taylor approximation
        safe_norm = jnp.where(norm_delta < 1e-6, 1.0, norm_delta)
        sin_theta_over_theta = jnp.where(
            norm_delta < 1e-6, 
            1.0 - (norm_delta**2)/6.0, 
            jnp.sin(safe_norm) / safe_norm
        )
        cos_theta = jnp.where(
            norm_delta < 1e-6,
            1.0 - (norm_delta**2)/2.0,
            jnp.cos(safe_norm)
        )
        return cos_theta * x + sin_theta_over_theta * delta

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        # Gaussian projection + normalization
        gauss = jax.random.normal(key, shape=shape + (self.ambient_dim,))
        return self.project(gauss)

    def log_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # P_T_x(y - x) gives the projected secant, which has norm sin(theta)*radius
        v = self.to_tangent(x, y - x)
        norm_v = _safe_norm_jvp(v)
        
        # Calculate theta using arcsin instead of arccos for stable gradients near 0
        # norm_v / radius = sin(theta)
        sin_theta = jnp.clip(norm_v / self.radius, -1.0, 1.0)
        theta = jnp.arcsin(sin_theta)
        
        # Scaling factor: theta * radius / norm_v. If norm_v is 0, scale is 1
        safe_norm = jnp.maximum(norm_v, 1e-7)
        scale = jnp.where(norm_v < 1e-7, 1.0 + (norm_v**2)/(6.0*self.radius**2), (theta * self.radius) / safe_norm)
        return v * scale

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

# =======================================================================
# Safe Math Primitives for Hyperbolic Geometry
# =======================================================================

@jax.custom_jvp
def _safe_minkowski_norm(x, y):
    """
    Computes sqrt(-x0*y0 + x1*y1 + ...) safely.
    x and y are assumed to be the same vector here for norm calculation.
    """
    sq_norm = -x[..., 0] * y[..., 0] + jnp.sum(x[..., 1:] * y[..., 1:], axis=-1)
    return jnp.sqrt(jnp.maximum(sq_norm, 0.0))

@_safe_minkowski_norm.defjvp
def _safe_minkowski_norm_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    
    # Forward pass
    sq_norm = -x[..., 0] * y[..., 0] + jnp.sum(x[..., 1:] * y[..., 1:], axis=-1)
    norm = jnp.sqrt(jnp.maximum(sq_norm, 0.0))
    
    # Backward pass (gradient)
    # d(norm) = <x, x_dot>_L / norm
    is_zero = norm < 1e-12
    safe_norm = jnp.where(is_zero, 1.0, norm)
    
    inner_dot = -x[..., 0] * x_dot[..., 0] + jnp.sum(x[..., 1:] * x_dot[..., 1:], axis=-1)
    
    # If norm is ~0, gradient is clamped to 0 to prevent explosion
    tangent_out = jnp.where(is_zero, 0.0, inner_dot / safe_norm)
    return norm, tangent_out

@jax.custom_jvp
def _safe_arccos(x):
    """Safely computes arccos(x) avoiding infinite gradients at |x|=1."""
    return jnp.arccos(jnp.clip(x, -1.0, 1.0))

@_safe_arccos.defjvp
def _safe_arccos_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    x_clipped = jnp.clip(x, -1.0, 1.0)
    primal_out = jnp.arccos(x_clipped)
    denom = jnp.sqrt(jnp.maximum(1.0 - x_clipped**2, 1e-12))
    tangent_out = -x_dot / denom
    return primal_out, tangent_out


# =======================================================================
# Refactored Hyperboloid Manifold
# =======================================================================

class Hyperboloid(Manifold):
    """
    The upper sheet of the hyperboloid in Minkowski space.
    -x0^2 + x1^2 + ... + xn^2 = -1, x0 > 0
    """
    _intrinsic_dim: int = eqx.field(static=True)

    def __init__(self, intrinsic_dim: int = 2):
        self._intrinsic_dim = intrinsic_dim

    @property
    def intrinsic_dim(self) -> int:
        return self._intrinsic_dim

    @property
    def ambient_dim(self) -> int:
        return self._intrinsic_dim + 1

    def _minkowski_dot(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return -u[..., 0] * v[..., 0] + jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)

    def _minkowski_norm(self, u: jnp.ndarray) -> jnp.ndarray:
        return _safe_minkowski_norm(u, u)

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        sq_norm = self._minkowski_dot(x, x)
        
        is_valid_candidate = (sq_norm < -1e-6) & (x[..., 0] > 0)
        
        safe_sq_norm = jnp.minimum(sq_norm, -1e-6)
        denom = jnp.sqrt(-safe_sq_norm)
        x_scaled = x / denom[..., None]
        
        x_spatial = x[..., 1:]
        x_spatial_sq = jnp.sum(x_spatial**2, axis=-1)
        x0_new = jnp.sqrt(1.0 + x_spatial_sq)
        x_lifted = jnp.concatenate([x0_new[..., None], x_spatial], axis=-1)
        
        mask = is_valid_candidate[..., None]
        return jnp.where(mask, x_scaled, x_lifted)

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        inner = self._minkowski_dot(x, v)
        return v + inner[..., None] * x

    def exp_map(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        norm_v = self._minkowski_norm(v)
        
        # Avoid jnp.where evaluating explosive branches by using Taylor Expansion
        # sinh(x) / x ≈ 1 + x^2 / 6
        safe_norm_v = jnp.where(norm_v < 1e-6, 1.0, norm_v)
        sinh_over_norm = jnp.where(
            norm_v < 1e-6,
            1.0 + (norm_v**2) / 6.0,
            jnp.sinh(safe_norm_v) / safe_norm_v
        )
        
        return jnp.cosh(norm_v)[..., None] * x + sinh_over_norm[..., None] * v

    def log_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        xy = self._minkowski_dot(x, y)
        u = y + xy[..., None] * x
        norm_u = self._minkowski_norm(u)
        
        # distance d(x,y) = arcsinh(||u||)
        dist = jnp.arcsinh(norm_u)
        
        # Taylor trick for the distance scaling: dist / sinh(dist) = dist / norm_u
        safe_norm_u = jnp.maximum(norm_u, 1e-7)
        scale = jnp.where(
            norm_u < 1e-7,
            1.0 - (norm_u**2) / 6.0, # Taylor approx for x/sinh(x) near 0
            dist / safe_norm_u
        )
        
        return scale[..., None] * u

    def parallel_transport(self, x: jnp.ndarray, y: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        xy = self._minkowski_dot(x, y)
        yv = self._minkowski_dot(y, v)
        
        # Ensure denominator is strictly non-zero (it's 1 - xy, and xy <= -1, so denom >= 2)
        denom = jnp.maximum(1.0 - xy, 2.0)
        scale = yv / denom
        return v + scale[..., None] * (x + y)

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        # Clamp delta to prevent float32 overflow in cosh/sinh (cosh(10) ~ 11013)
        norm_delta = self._minkowski_norm(delta)
        max_norm = 10.0
        safe_norm = jnp.maximum(norm_delta, 1e-12)
        scale = jnp.where(norm_delta > max_norm, max_norm / safe_norm, 1.0)
        safe_delta = delta * scale[..., None]
        return self.project(self.exp_map(x, safe_delta))

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        spat_dim = self.intrinsic_dim
        v_spatial = jax.random.normal(key, shape + (spat_dim,))
        norm_v = jnp.linalg.norm(v_spatial, axis=-1, keepdims=True)
        
        safe_norm = jnp.maximum(norm_v, 1e-6)
        x0 = jnp.cosh(norm_v)
        
        # No grad needed here usually, but Taylor safe anyway
        sinh_over_norm = jnp.where(
            norm_v < 1e-6,
            1.0 + (norm_v**2)/6.0,
            jnp.sinh(safe_norm) / safe_norm
        )
        x_rest = sinh_over_norm * v_spatial
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

    def log_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return y - x

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        return jax.random.normal(key, shape + (self._dim,))
        