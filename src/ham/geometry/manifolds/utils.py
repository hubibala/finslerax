"""Internal numerical utilities for manifolds."""
import jax
import jax.numpy as jnp
from ham.utils.math import GRAD_EPS

@jax.custom_jvp
def _safe_minkowski_self_norm(x: jax.Array) -> jax.Array:
    """Computes Minkowski self-norm sqrt(-x0² + x1² + ...) safely.

    Args:
        x: Vector in Minkowski space, shape (..., n+1).

    Returns:
        Minkowski self-norm, shape (...).
    """
    sq_norm = -x[..., 0] ** 2 + jnp.sum(x[..., 1:] ** 2, axis=-1)
    return jnp.sqrt(jnp.maximum(sq_norm, 0.0))


@_safe_minkowski_self_norm.defjvp
def _safe_minkowski_self_norm_jvp(primals, tangents):
    """Custom JVP for _safe_minkowski_self_norm. 
    
    Derivative: <x, x_dot>_L / ||x||_L, clamped to 0 when ||x||_L < GRAD_EPS.
    """
    (x,) = primals
    (x_dot,) = tangents
    sq_norm = -x[..., 0] ** 2 + jnp.sum(x[..., 1:] ** 2, axis=-1)
    norm = jnp.sqrt(jnp.maximum(sq_norm, 0.0))
    is_zero = norm < GRAD_EPS
    safe_norm_val = jnp.where(is_zero, 1.0, norm)
    inner_dot = -x[..., 0] * x_dot[..., 0] + jnp.sum(x[..., 1:] * x_dot[..., 1:], axis=-1)
    tangent_out = jnp.where(is_zero, 0.0, inner_dot / safe_norm_val)
    return norm, tangent_out


@jax.custom_jvp
def _safe_arccos(x: jax.Array) -> jax.Array:
    """Computes arccos(x) with stable primal and gradients at x=±1.

    The primal clamps ``x`` to ``[-1, 1]`` before calling ``jnp.arccos``
    to avoid NaN when points drift slightly off the manifold (float64
    rounding can produce ``|<x,y>/r^2| > 1``).  The custom JVP then
    regularises the ``sqrt(1-x^2)`` denominator independently so that
    near-antipodal gradients are well-defined even after the clamp.

    Args:
        x: Input array.

    Returns:
        arccos(x), clipped to avoid NaN for |x| > 1.
    """
    return jnp.arccos(jnp.clip(x, -1.0, 1.0))


@_safe_arccos.defjvp
def _safe_arccos_jvp(primals, tangents):
    """Custom JVP for _safe_arccos. 
    
    Derivative: -x_dot / sqrt(1 - x^2), regularized by GRAD_EPS.
    """
    (x,) = primals
    (x_dot,) = tangents
    denom = jnp.sqrt(jnp.maximum(1.0 - x**2, GRAD_EPS))
    tangent_out = -x_dot / denom
    return _safe_arccos(x), tangent_out
