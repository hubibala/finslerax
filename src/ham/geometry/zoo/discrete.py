"""Discrete Randers metric implementation."""


import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.mesh import TriangularMesh
from ham.geometry.metric import AsymmetricMetric
from ham.utils.math import GRAD_EPS, WIND_STIFFNESS, causal_wind_scale, safe_norm


class DiscreteRanders(AsymmetricMetric):
    """Randers metric on a triangular mesh with per-face wind vectors.

    The Zermelo bound ``||W|| < 1 - epsilon`` is enforced via the smooth,
    identity-preserving causal clamp (``wind_mode="soft"``, the default) or
    bypassed for trusted fields (``wind_mode="raw"``); see
    :class:`ham.geometry.zoo.Randers` and :func:`ham.utils.causal_wind_scale`.
    """

    face_winds: jnp.ndarray
    epsilon: float = eqx.field(static=True)
    wind_stiffness: float = eqx.field(static=True)
    wind_mode: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: TriangularMesh,
        face_winds: jnp.ndarray,
        epsilon: float = 1e-5,
        wind_mode: str = "soft",
        wind_stiffness: float = WIND_STIFFNESS,
    ):
        """Initializes the Discrete Randers metric."""
        super().__init__(manifold=mesh)
        self.face_winds = face_winds
        self.epsilon = epsilon
        self.wind_mode = str(wind_mode)
        self.wind_stiffness = float(wind_stiffness)
        if self.wind_mode not in ("soft", "raw"):
            raise ValueError(
                f"wind_mode must be 'soft' or 'raw', got {self.wind_mode!r}"
            )

    def zermelo_data(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns the Zermelo navigation triple (H, W, lambda)."""
        weights = self.manifold.get_face_weights(x)
        W_raw = jnp.dot(weights, self.face_winds)
        w_norm = safe_norm(W_raw)
        max_speed = 1.0 - self.epsilon

        # Identity sea (H = I), so ||W|| is the Euclidean norm.  See
        # ham.geometry.zoo.Randers.zermelo_data for the soft/raw rationale.
        if self.wind_mode == "raw":
            W = W_raw
            lam = jnp.maximum(1.0 - w_norm**2, GRAD_EPS)
        else:
            scale = causal_wind_scale(w_norm, max_speed, self.wind_stiffness)
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
