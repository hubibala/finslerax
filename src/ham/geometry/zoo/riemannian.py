"""Riemannian metric implementation."""
import jax
import jax.numpy as jnp
from typing import Callable, Any

from ham.geometry.metric import FinslerMetric
from ham.geometry.manifold import Manifold
from ham.utils import PSD_EPS, GRAD_EPS

class Riemannian(FinslerMetric):
    """General Riemannian metric: F(x, v) = sqrt( v^T G(x) v ).

    The metric tensor field G(x) is reconstructed from a raw factor A(x) provided by `g_net`.
    This Cholesky-like construction (G = A A^T + eps I) guarantees positive-definiteness.

    Args:
        manifold: The topological domain M.
        g_net: Callable mapping position x (shape `(D,)`) to a metric factor A(x) (shape `(D, D)`).
    """
    g_net: Any

    def __init__(self, manifold: Manifold, g_net: Callable[[jax.Array], jax.Array]):
        """Initializes the Riemannian metric."""
        super().__init__(manifold)
        self.g_net = g_net

    def __repr__(self) -> str:
        return f"Riemannian(manifold={self.manifold})"

    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Computes F(x, v) = sqrt( v^T G(x) v )."""
        v_sq = jnp.sum(v**2, axis=-1)
        is_zero = v_sq < GRAD_EPS
        
        # Symmetrize and construct PSD metric via Cholesky factor A
        A = self.g_net(x)
        G_x = jnp.dot(A, A.T) + PSD_EPS * jnp.eye(A.shape[-1])
        
        quad = jnp.dot(v, jnp.dot(G_x, v))
        return jnp.where(is_zero, 0.0, jnp.sqrt(jnp.maximum(quad, GRAD_EPS)))
