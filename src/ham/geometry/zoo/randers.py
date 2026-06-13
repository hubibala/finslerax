"""Randers metric implementation."""

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.manifold import Manifold
from ham.geometry.metric import AsymmetricMetric
from ham.utils import GRAD_EPS


class Randers(AsymmetricMetric):
    """Rigorous Randers Metric using Zermelo Navigation.

    A Randers metric is an asymmetric Finsler metric defined by a Riemannian
    "sea" (H) and a drift "wind" field (W). The wind is squashed via tanh
    so that ||W||_H < 1 - epsilon, preserving strong convexity.
    """

    h_net: Any
    w_net: Any
    epsilon: float = eqx.field(static=True)
    use_wind: bool = eqx.field(static=True)

    def __init__(
        self,
        manifold: Manifold,
        h_net: Callable[[jax.Array], jax.Array],
        w_net: Callable[[jax.Array], jax.Array],
        epsilon: float = 1e-5,
        use_wind: bool = True,
    ):
        """Initializes the Randers metric."""
        super().__init__(manifold=manifold)
        self.h_net = h_net
        self.w_net = w_net
        self.epsilon = float(epsilon)
        self.use_wind = bool(use_wind)

    def __repr__(self) -> str:
        return f"Randers(manifold={self.manifold}, epsilon={self.epsilon})"

    def zermelo_data(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns the Riemannian tensor H, the Wind vector W, and the Lambda scalar."""
        # Get the sea metric H(x)
        H = self.h_net(x)
        # Defensive symmetrization
        H = 0.5 * (H + H.T)

        W_raw = self.w_net(x)
        W_raw = self.manifold.to_tangent(x, W_raw)

        w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))
        w_norm = jnp.sqrt(jnp.maximum(w_norm_sq, GRAD_EPS))

        # Smooth causal squash applied for ALL wind magnitudes.  The previous
        # ``jnp.where(w_norm < 0.5, 1.0, ...)`` gate introduced a C0 JUMP at the
        # boundary: the squashed magnitude dropped from 0.5 (scale=1) to
        # max_speed*tanh(0.5) ~= 0.462 (scale<1), so the wind field -- and hence
        # F and its gradients -- was discontinuous, violating Finsler regularity
        # (review finding W-RAND).  Applying ``max_speed * tanh(w_norm)/w_norm``
        # everywhere keeps F in C^1 and still guarantees the Zermelo weak-wind
        # bound ||W_safe||_H = max_speed*tanh(w_norm) < max_speed < 1 (strong
        # convexity).  Reference: spec/MATH_SPEC.md section 5.
        max_speed = 1.0 - self.epsilon
        scale = (max_speed * jnp.tanh(w_norm)) / (w_norm + GRAD_EPS)
        W_safe = W_raw * scale

        safe_w_norm_sq = jnp.dot(W_safe, jnp.dot(H, W_safe))
        lambda_factor = 1.0 - safe_w_norm_sq

        if not self.use_wind:
            return H, jnp.zeros_like(W_safe), jnp.array(1.0, dtype=H.dtype)

        return H, W_safe, lambda_factor

    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Computes the Randers-Zermelo cost."""
        v_sq_raw = jnp.sum(v**2, axis=-1)
        is_zero = v_sq_raw < GRAD_EPS
        v_safe = jnp.where(is_zero[..., None], v + jnp.sqrt(GRAD_EPS), v)

        H, W, lam = self.zermelo_data(x)

        Hv = jnp.matmul(H, v_safe)
        HW = jnp.matmul(H, W)

        v_sq_h = jnp.sum(v_safe * Hv, axis=-1)
        W_dot_v = jnp.sum(v_safe * HW, axis=-1)

        discriminant = lam * v_sq_h + W_dot_v**2
        cost = (jnp.sqrt(jnp.maximum(discriminant, GRAD_EPS)) - W_dot_v) / lam
        return jnp.where(is_zero, 0.0, cost)
