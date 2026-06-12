"""Discrete Randers metric implementation."""


import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.mesh import TriangularMesh
from ham.geometry.metric import AsymmetricMetric
from ham.utils.math import GRAD_EPS, safe_norm


class DiscreteRanders(AsymmetricMetric):
    """Randers metric on a triangular mesh with per-face wind vectors."""

    face_winds: jnp.ndarray
    epsilon: float = eqx.field(static=True)

    def __init__(
        self, mesh: TriangularMesh, face_winds: jnp.ndarray, epsilon: float = 1e-5
    ):
        """Initializes the Discrete Randers metric."""
        super().__init__(manifold=mesh)
        self.face_winds = face_winds
        self.epsilon = epsilon

    def zermelo_data(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns the Zermelo navigation triple (H, W, lambda)."""
        weights = self.manifold.get_face_weights(x)
        W_raw = jnp.dot(weights, self.face_winds)
        w_norm = safe_norm(W_raw)

        max_speed = 1.0 - self.epsilon
        scale = jnp.where(
            w_norm < 0.5, 1.0, (max_speed * jnp.tanh(w_norm)) / (w_norm + GRAD_EPS)
        )
        W = W_raw * scale
        lam = 1.0 - (w_norm * scale) ** 2

        dim = self.manifold.ambient_dim
        H = jnp.eye(dim)
        return H, W, lam

    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Computes the Randers-Zermelo cost with interpolated per-face wind."""
        v_sq_raw = jnp.sum(v**2, axis=-1)
        is_zero = v_sq_raw < GRAD_EPS
        v_safe = jnp.where(is_zero[..., None], v + jnp.sqrt(GRAD_EPS), v)

        _, W, lam = self.zermelo_data(x)

        v_sq = jnp.sum(v_safe**2, axis=-1)
        W_dot_v = jnp.sum(W * v_safe, axis=-1)
        discriminant = lam * v_sq + W_dot_v**2
        cost = (jnp.sqrt(jnp.maximum(discriminant, GRAD_EPS)) - W_dot_v) / lam
        return jnp.where(is_zero, 0.0, cost)
