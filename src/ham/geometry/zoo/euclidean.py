"""Euclidean metric implementation."""
import jax
import jax.numpy as jnp
from ham.geometry.metric import FinslerMetric
from ham.utils.math import safe_norm, GRAD_EPS

class Euclidean(FinslerMetric):
    """Standard Euclidean metric: F(x, v) = ||v||.

    The spray is identically zero and the Berwald connection vanishes.
    See `spec/MATH_SPEC.md` § 4, Euclidean row.

    Args:
        manifold: The topological domain M.

    Example:
        See `examples/demo_vortex.py`, `examples/demo_zermelo.py`.
    """
    
    def __repr__(self) -> str:
        return f"Euclidean(manifold={self.manifold})"

    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Computes F(x, v) = ||v|| using `safe_norm`."""
        v_sq = jnp.sum(v**2, axis=-1)
        is_zero = v_sq < GRAD_EPS
        return jnp.where(is_zero, 0.0, safe_norm(v))
