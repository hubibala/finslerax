"""Hyperboloid manifold implementation."""
import jax
import jax.numpy as jnp
import equinox as eqx

from ham.geometry.manifold import Manifold
from ham.utils import GRAD_EPS, TAYLOR_EPS
from ham.geometry.manifolds.utils import _safe_minkowski_self_norm

# Constants for numerical stability
RETRACT_MAX_NORM = 10.0

class Hyperboloid(Manifold):
    """The upper sheet of the hyperboloid in Minkowski space.

    Defined by the constraint -x0² + x1² + ... + xn² = -1, x0 > 0.
    Features exact closed-form exponential and logarithmic maps.
    See `spec/MATH_SPEC.md` § 4.1.

    Args:
        intrinsic_dim: Dimension n. Default: 2.

    Warning:
        Joint training on complex curved manifolds like Hyperboloid with the 
        full VAE pipeline remains numerically sensitive.
    """
    _intrinsic_dim: int = eqx.field(static=True)

    def __init__(self, intrinsic_dim: int = 2):
        self._intrinsic_dim = int(intrinsic_dim)

    def __repr__(self) -> str:
        return f"Hyperboloid(intrinsic_dim={self._intrinsic_dim})"

    @property
    def intrinsic_dim(self) -> int:
        return self._intrinsic_dim

    @property
    def ambient_dim(self) -> int:
        return self._intrinsic_dim + 1

    def _minkowski_dot(self, u: jax.Array, v: jax.Array) -> jax.Array:
        """Minkowski inner product with signature (-1, 1, ..., 1)."""
        return -u[..., 0] * v[..., 0] + jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)

    def _minkowski_norm(self, u: jax.Array) -> jax.Array:
        """Minkowski self-norm using safe primitive."""
        return _safe_minkowski_self_norm(u)

    def project(self, x: jax.Array) -> jax.Array:
        """Projects ambient point onto the hyperboloid upper sheet."""
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

    def to_tangent(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Projects ambient vector v onto Tx M: Pi(v) = v + <x, v>_L x."""
        inner = self._minkowski_dot(x, v)
        return v + inner[..., None] * x

    def exp_map(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Exact Riemannian exponential map on the hyperboloid."""
        norm_v = self._minkowski_norm(v)
        safe_norm_v = jnp.where(norm_v < TAYLOR_EPS, 1.0, norm_v)
        sinh_over_norm = jnp.where(
            norm_v < TAYLOR_EPS,
            1.0 + (norm_v ** 2) / 6.0,
            jnp.sinh(safe_norm_v) / safe_norm_v,
        )
        return jnp.cosh(norm_v)[..., None] * x + sinh_over_norm[..., None] * v

    def log_map(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Exact inverse exponential map on the hyperboloid."""
        xy = self._minkowski_dot(x, y)
        u = y + xy[..., None] * x
        norm_u = self._minkowski_norm(u)
        dist = jnp.arcsinh(norm_u)
        safe_norm_u = jnp.maximum(norm_u, TAYLOR_EPS)
        scale = jnp.where(
            norm_u < TAYLOR_EPS,
            1.0 - (norm_u ** 2) / 6.0,
            dist / safe_norm_u,
        )
        return scale[..., None] * u

    def parallel_transport(self, x: jax.Array, y: jax.Array, v: jax.Array) -> jax.Array:
        """Parallel transports v from Tx to Ty along the hyperboloid geodesic."""
        xy = self._minkowski_dot(x, y)
        yv = self._minkowski_dot(y, v)
        denom = jnp.maximum(1.0 - xy, 2.0)
        scale = yv / denom
        return v + scale[..., None] * (x + y)

    def retract(self, x: jax.Array, delta: jax.Array) -> jax.Array:
        """Clamped retraction via exact exponential map.

        The exponential map of a tangent vector on the hyperboloid is already
        on the hyperboloid exactly, so a redundant ``project`` call is omitted.
        Clamping ensures the step magnitude stays within RETRACT_MAX_NORM,
        preventing numerical overflow for large gradient steps.
        """
        norm_delta = self._minkowski_norm(delta)
        safe_nd = jnp.maximum(norm_delta, GRAD_EPS)
        scale = jnp.where(norm_delta > RETRACT_MAX_NORM, RETRACT_MAX_NORM / safe_nd, 1.0)
        safe_delta = delta * scale[..., None]
        return self.exp_map(x, safe_delta)

    def random_sample(self, key: jax.Array, shape: tuple[int, ...] = ()) -> jax.Array:
        """Samples uniformly on the hyperboloid upper sheet."""
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

    def metric_tensor(self, x: jax.Array) -> jax.Array:
        """Returns the ambient Minkowski metric tensor diag(-1, 1, ..., 1)."""
        dim = self.ambient_dim
        return jnp.eye(dim).at[0, 0].set(-1.0)
