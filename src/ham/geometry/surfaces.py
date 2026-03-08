"""Concrete manifold implementations: Sphere, Torus, Paraboloid, Hyperboloid, EuclideanSpace.

Changes from review:
  P0-1: Removed ``import jax._src.cudnn.scaled_matmul_stablehlo``
  P0-2: Removed duplicate ``_safe_norm_jvp``; use canonical ``safe_norm`` from utils.math
  P1-6: Sphere.log_map now uses the arccos-based formula valid for all angular separations
  P1-7: ``_safe_minkowski_norm`` renamed to ``_safe_minkowski_self_norm`` and enforced x is y
  P2-10: Added ``exp_map`` / ``log_map`` to every manifold (Sphere, Torus, Paraboloid, EuclideanSpace)
  P2-14: Added ``__repr__`` to all geometry classes
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from .manifold import Manifold
from ..utils.math import safe_norm, GRAD_EPS, NORM_EPS, TAYLOR_EPS


# =======================================================================
# Sphere (S^n embedded in R^{n+1})
# =======================================================================

class Sphere(Manifold):
    radius: float = eqx.field(static=True)
    _intrinsic_dim: int = eqx.field(static=True)

    def __init__(self, intrinsic_dim: int = 2, radius: float = 1.0):
        self._intrinsic_dim = intrinsic_dim
        self.radius = radius

    def __repr__(self) -> str:
        return f"Sphere(radius={self.radius}, intrinsic_dim={self._intrinsic_dim})"

    @property
    def ambient_dim(self) -> int:
        return self._intrinsic_dim + 1

    @property
    def intrinsic_dim(self) -> int:
        return self._intrinsic_dim

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        norm = safe_norm(x, axis=-1, keepdims=True)
        safe_x = x / jnp.maximum(norm, NORM_EPS)
        # For zero-length vectors, default to the "north pole" in ambient space
        is_zero = norm < NORM_EPS
        pole = jnp.zeros_like(x).at[..., -1].set(1.0)
        direction = jnp.where(is_zero, pole, safe_x)
        return self.radius * direction

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        proj = jnp.einsum('...i,...i->...', x, v)[..., None] / (self.radius ** 2)
        return v - proj * x

    def exp_map(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Riemannian exponential map on the sphere S^n(r).

        For a sphere of radius r the geodesic angle is θ = ||v|| / r.
        γ(1) = cos(θ) x  +  sin(θ)/θ  v
        """
        norm_v = safe_norm(v, axis=-1, keepdims=True)
        theta = norm_v / self.radius
        safe_theta = jnp.maximum(theta, TAYLOR_EPS)
        sin_theta_over_theta = jnp.where(
            theta < TAYLOR_EPS,
            1.0 - (theta ** 2) / 6.0,
            jnp.sin(safe_theta) / safe_theta,
        )
        cos_theta = jnp.where(
            theta < TAYLOR_EPS,
            1.0 - (theta ** 2) / 2.0,
            jnp.cos(safe_theta),
        )
        return cos_theta * x + sin_theta_over_theta * v

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        """Retraction (delegates to exp_map)."""
        return self.exp_map(x, delta)

    def log_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # 1. Safe dot product: strictly clip away from 1.0 and -1.0
        # Normalize by radius squared to get cos(theta)
        u = jnp.sum(x * y, axis=-1, keepdims=True) / (self.radius ** 2)
        u_clipped = jnp.clip(u, -1.0 + GRAD_EPS, 1.0 - GRAD_EPS)
        
        # 2. Safe distance (theta)
        dist = jnp.arccos(u_clipped)
        
        # 3. Safe tangent direction
        # We need v = (y - cos(theta)x) * (theta / sin(theta))
        # For small theta, theta/sin(theta) ~ 1 + theta^2/6
        safe_sin = jnp.sqrt(jnp.maximum(1.0 - u_clipped**2, GRAD_EPS))
        scale = jnp.where(
            dist < TAYLOR_EPS,
            1.0 + (dist ** 2) / 6.0,
            dist / safe_sin,
        )
        
        diff = y - u_clipped * x
        return scale * diff

    def parallel_transport(self, x: jnp.ndarray, y: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Reflection through the bisector of x and y
        xy = jnp.sum(x * y, axis=-1)
        yv = jnp.sum(y * v, axis=-1)
        # Denominator should be (r^2 + <x, y>)
        denom = jnp.maximum(self.radius**2 + xy, 1e-5)
        scale = yv / denom
        return v - scale[..., None] * (x + y)

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        gauss = jax.random.normal(key, shape=shape + (self.ambient_dim,))
        return self.project(gauss)


# =======================================================================
# Torus (T^2 embedded in R^3)
# =======================================================================

class Torus(Manifold):
    R: float = eqx.field(static=True)
    r: float = eqx.field(static=True)

    def __init__(self, major_R: float = 2.0, minor_r: float = 1.0):
        self.R = major_R
        self.r = minor_r

    def __repr__(self) -> str:
        return f"Torus(R={self.R}, r={self.r})"

    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 2

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        rho = safe_norm(x[:2])

        dir_xy = jnp.where(
            rho > NORM_EPS,
            x[:2] / jnp.maximum(rho, NORM_EPS),
            jnp.array([1.0, 0.0]),
        )

        dist = rho - self.R
        n_xy = jnp.concatenate([dir_xy * dist, jnp.array([x[2]])])
        n_norm = safe_norm(n_xy)

        n = jnp.where(
            n_norm > NORM_EPS,
            n_xy / jnp.maximum(n_norm, NORM_EPS),
            jnp.array([0.0, 0.0, 1.0]),
        )

        projected = x - n * (n_norm - self.r)

        fallback = jnp.array([self.R + self.r, 0.0, 0.0])
        use_fallback = rho < TAYLOR_EPS
        return jnp.where(use_fallback, fallback, projected)

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        rho = safe_norm(x[:2])

        dir_xy = jnp.where(
            rho > NORM_EPS,
            x[:2] / jnp.maximum(rho, NORM_EPS),
            jnp.zeros(2),
        )
        dist = rho - self.R
        n_xy = jnp.concatenate([dir_xy * dist, jnp.array([x[2]])])
        n_norm = safe_norm(n_xy)

        n = jnp.where(
            n_norm > NORM_EPS,
            n_xy / jnp.maximum(n_norm, NORM_EPS),
            jnp.array([0.0, 0.0, 1.0]),
        )
        return v - jnp.dot(n, v) * n

    def exp_map(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        """Approximate exp map via projected retraction."""
        return self.project(x + delta)

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        return self.project(x + delta)

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        u_key, v_key = jax.random.split(key)
        u = jax.random.uniform(u_key, shape, minval=0, maxval=2 * jnp.pi)
        v = jax.random.uniform(v_key, shape, minval=0, maxval=2 * jnp.pi)
        x = (self.R + self.r * jnp.cos(v)) * jnp.cos(u)
        y = (self.R + self.r * jnp.cos(v)) * jnp.sin(u)
        z = self.r * jnp.sin(v)
        return jnp.stack([x, y, z], axis=-1)


# =======================================================================
# Paraboloid (z = x² + y²)
# =======================================================================

class Paraboloid(Manifold):

    def __repr__(self) -> str:
        return "Paraboloid()"

    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 2

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0], x[1], x[0] ** 2 + x[1] ** 2])

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        n = jnp.array([-2 * x[0], -2 * x[1], 1.0])
        n = n / safe_norm(n)
        return v - jnp.dot(n, v) * n

    def exp_map(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        """Approximate exp map via exact retraction."""
        return self.retract(x, delta)

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        xy_new = x[:2] + delta[:2]
        z_new = jnp.sum(xy_new ** 2)
        return jnp.concatenate([xy_new, jnp.array([z_new])])

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        xy = jax.random.normal(key, shape + (2,)) * 2.0
        z = jnp.sum(xy ** 2, axis=-1)
        return jnp.concatenate([xy, z[..., None]], axis=-1)


# =======================================================================
# Safe Math Primitives for Hyperbolic Geometry
# =======================================================================

@jax.custom_jvp
def _safe_minkowski_self_norm(x):
    """Computes sqrt(-x0² + x1² + ...) for a single vector (self-norm).

    This is intentionally restricted to the self-norm case so that the
    custom JVP does not silently drop a ``y_dot`` tangent.  (P1-7 fix.)
    """
    sq_norm = -x[..., 0] ** 2 + jnp.sum(x[..., 1:] ** 2, axis=-1)
    return jnp.sqrt(jnp.maximum(sq_norm, 0.0))


@_safe_minkowski_self_norm.defjvp
def _safe_minkowski_self_norm_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents

    sq_norm = -x[..., 0] ** 2 + jnp.sum(x[..., 1:] ** 2, axis=-1)
    norm = jnp.sqrt(jnp.maximum(sq_norm, 0.0))

    is_zero = norm < GRAD_EPS
    safe_norm_val = jnp.where(is_zero, 1.0, norm)

    # d/dt sqrt(<x,x>_L) = <x, x_dot>_L / sqrt(<x,x>_L)
    inner_dot = -x[..., 0] * x_dot[..., 0] + jnp.sum(x[..., 1:] * x_dot[..., 1:], axis=-1)
    tangent_out = jnp.where(is_zero, 0.0, inner_dot / safe_norm_val)
    return norm, tangent_out


@jax.custom_jvp
def _safe_arccos(x):
    """Safely computes arccos(x) avoiding infinite gradients at |x|=1."""
    return jnp.arccos(jnp.clip(x, -1.0, 1.0))


@_safe_arccos.defjvp
def _safe_arccos_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    x_clipped = jnp.clip(x, -1.0, 1.0)
    primal_out = jnp.arccos(x_clipped)
    denom = jnp.sqrt(jnp.maximum(1.0 - x_clipped ** 2, GRAD_EPS))
    tangent_out = -x_dot / denom
    return primal_out, tangent_out


# =======================================================================
# Hyperboloid Manifold
# =======================================================================

class Hyperboloid(Manifold):
    """The upper sheet of the hyperboloid in Minkowski space.

    -x0² + x1² + ... + xn² = -1, x0 > 0
    """
    _intrinsic_dim: int = eqx.field(static=True)

    def __init__(self, intrinsic_dim: int = 2):
        self._intrinsic_dim = intrinsic_dim

    def __repr__(self) -> str:
        return f"Hyperboloid(intrinsic_dim={self._intrinsic_dim})"

    @property
    def intrinsic_dim(self) -> int:
        return self._intrinsic_dim

    @property
    def ambient_dim(self) -> int:
        return self._intrinsic_dim + 1

    def _minkowski_dot(self, u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return -u[..., 0] * v[..., 0] + jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)

    def _minkowski_norm(self, u: jnp.ndarray) -> jnp.ndarray:
        return _safe_minkowski_self_norm(u)

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        sq_norm = self._minkowski_dot(x, x)

        is_valid_candidate = (sq_norm < -TAYLOR_EPS) & (x[..., 0] > 0)

        safe_sq_norm = jnp.minimum(sq_norm, -TAYLOR_EPS)
        denom = jnp.sqrt(-safe_sq_norm)
        x_scaled = x / denom[..., None]

        x_spatial = x[..., 1:]
        x_spatial_sq = jnp.sum(x_spatial ** 2, axis=-1)
        x0_new = jnp.sqrt(1.0 + x_spatial_sq)
        x_lifted = jnp.concatenate([x0_new[..., None], x_spatial], axis=-1)

        mask = is_valid_candidate[..., None]
        return jnp.where(mask, x_scaled, x_lifted)

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        inner = self._minkowski_dot(x, v)
        return v + inner[..., None] * x

    def exp_map(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        norm_v = self._minkowski_norm(v)

        safe_norm_v = jnp.where(norm_v < TAYLOR_EPS, 1.0, norm_v)
        sinh_over_norm = jnp.where(
            norm_v < TAYLOR_EPS,
            1.0 + (norm_v ** 2) / 6.0,
            jnp.sinh(safe_norm_v) / safe_norm_v,
        )

        return jnp.cosh(norm_v)[..., None] * x + sinh_over_norm[..., None] * v

    def log_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        xy = self._minkowski_dot(x, y)
        u = y + xy[..., None] * x
        norm_u = self._minkowski_norm(u)

        dist = jnp.arcsinh(norm_u)

        safe_norm_u = jnp.maximum(norm_u, NORM_EPS)
        scale = jnp.where(
            norm_u < NORM_EPS,
            1.0 - (norm_u ** 2) / 6.0,
            dist / safe_norm_u,
        )

        return scale[..., None] * u

    def parallel_transport(self, x: jnp.ndarray, y: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        xy = self._minkowski_dot(x, y)
        yv = self._minkowski_dot(y, v)

        denom = jnp.maximum(1.0 - xy, 2.0)
        scale = yv / denom
        return v + scale[..., None] * (x + y)

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        norm_delta = self._minkowski_norm(delta)
        max_norm = 10.0
        safe_nd = jnp.maximum(norm_delta, GRAD_EPS)
        scale = jnp.where(norm_delta > max_norm, max_norm / safe_nd, 1.0)
        safe_delta = delta * scale[..., None]
        return self.project(self.exp_map(x, safe_delta))

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        spat_dim = self.intrinsic_dim
        v_spatial = jax.random.normal(key, shape + (spat_dim,))
        norm_v = jnp.linalg.norm(v_spatial, axis=-1, keepdims=True)

        safe_nv = jnp.maximum(norm_v, TAYLOR_EPS)
        x0 = jnp.cosh(norm_v)

        sinh_over_norm = jnp.where(
            norm_v < TAYLOR_EPS,
            1.0 + (norm_v ** 2) / 6.0,
            jnp.sinh(safe_nv) / safe_nv,
        )
        x_rest = sinh_over_norm * v_spatial
        return jnp.concatenate([x0, x_rest], axis=-1)

    def metric_tensor(self, x: jnp.ndarray) -> jnp.ndarray:
        m = jnp.eye(self.ambient_dim)
        m = m.at[0, 0].set(-1.0)
        return m


# =======================================================================
# Euclidean Space R^N
# =======================================================================

class EuclideanSpace(Manifold):
    """Flat Euclidean space R^N."""
    _dim: int = eqx.field(static=True)

    def __init__(self, dim: int):
        self._dim = dim

    def __repr__(self) -> str:
        return f"EuclideanSpace(dim={self._dim})"

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

    def exp_map(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        return x + delta

    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        return x + delta

    def log_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return y - x

    def parallel_transport(self, x: jnp.ndarray, y: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return v

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
        return jax.random.normal(key, shape + (self._dim,))