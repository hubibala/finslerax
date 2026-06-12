"""Riemannian metric implementation."""
import jax
import jax.numpy as jnp
from typing import Callable, Any

from ham.geometry.metric import FinslerMetric
from ham.geometry.manifold import Manifold
from ham.utils import PSD_EPS, GRAD_EPS

class Riemannian(FinslerMetric):
    """General Riemannian metric: F(x, v) = sqrt( v^T G(x) v ).

    The metric tensor field G(x) is provided by `g_net`. This implementation 
    assumes `g_net` handles positive-definiteness (e.g., via PSDMatrixField).

    Args:
        manifold: The topological domain M.
        g_net: Callable mapping position x (shape `(D,)`) to a metric tensor G(x) (shape `(D, D)`).
    """
    g_net: Any

    def __init__(self, manifold: Manifold, g_net: Callable[[jax.Array], jax.Array]):
        """Initializes the Riemannian metric."""
        super().__init__(manifold=manifold)
        self.g_net = g_net

    def __repr__(self) -> str:
        return f"Riemannian(manifold={self.manifold})"

    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Computes F(x, v) = sqrt( v^T G(x) v )."""
        v_sq = jnp.sum(v**2, axis=-1)
        is_zero = v_sq < GRAD_EPS
        
        # Get the metric tensor G(x)
        G_x = self.g_net(x)
        
        # Defensive symmetrization
        G_x = 0.5 * (G_x + G_x.T)
        
        quad = jnp.dot(v, jnp.dot(G_x, v))
        return jnp.where(is_zero, 0.0, jnp.sqrt(jnp.maximum(quad, GRAD_EPS)))

    def inner_product(
        self, x: jax.Array, v: jax.Array, w1: jax.Array, w2: jax.Array
    ) -> jax.Array:
        """Riemannian inner product: g_ij(x) = G(x), independent of v."""
        G_x = self.g_net(x)
        G_x = 0.5 * (G_x + G_x.T)
        return jnp.dot(w1, jnp.dot(G_x, w2))
