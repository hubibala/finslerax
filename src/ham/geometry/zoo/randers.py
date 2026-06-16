"""Randers metric implementation."""

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.manifold import Manifold
from ham.geometry.metric import AsymmetricMetric
from ham.utils import GRAD_EPS, WIND_STIFFNESS, causal_wind_scale


class Randers(AsymmetricMetric):
    """Rigorous Randers Metric using Zermelo Navigation.

    A Randers metric is an asymmetric Finsler metric defined by a Riemannian
    "sea" (H) and a drift "wind" field (W). The Zermelo strong-convexity bound
    ``||W||_H < 1 - epsilon`` is enforced by :func:`ham.utils.causal_wind_scale`.

    Two ``wind_mode`` policies are available:

    ``"soft"`` (default)
        A smooth, identity-preserving causal clamp (the temperature-controlled
        smooth minimum). Physically-valid winds pass through essentially
        unchanged — error ``~exp(-stiffness * (1 - ||W||_H))`` — while winds
        approaching the causal limit are bent within a thin shell of width
        ``~1/stiffness``. Recommended for *learned* winds, whose network output
        is unconstrained and may exceed the causal limit during training.

    ``"raw"``
        Pass the wind through bit-exact (no clamp), flooring only ``lambda`` as
        a NaN guard. Use for *trusted, prescribed* fields that the caller
        guarantees satisfy ``||W||_H < 1`` (e.g. a known ocean current), when
        maximum precision is required.

    Note:
        The historical ``max_speed * tanh(||W||) / ||W||`` squash bent *every*
        wind (its slope at the origin is ``max_speed < 1``), silently distorting
        valid currents — e.g. ``0.5 -> 0.46``. The soft clamp fixes this; see
        :func:`ham.utils.causal_wind_scale`.
    """

    h_net: Any
    w_net: Any
    epsilon: float = eqx.field(static=True)
    wind_stiffness: float = eqx.field(static=True)
    wind_mode: str = eqx.field(static=True)
    use_wind: bool = eqx.field(static=True)

    def __init__(
        self,
        manifold: Manifold,
        h_net: Callable[[jax.Array], jax.Array],
        w_net: Callable[[jax.Array], jax.Array],
        epsilon: float = 1e-5,
        use_wind: bool = True,
        wind_mode: str = "soft",
        wind_stiffness: float = WIND_STIFFNESS,
    ):
        """Initializes the Randers metric.

        Args:
            manifold: The topological domain M.
            h_net: Callable ``x -> H(x)`` returning the SPD sea tensor.
            w_net: Callable ``x -> W(x)`` returning the wind vector.
            epsilon: Causality margin; ``max_speed = 1 - epsilon``. Default 1e-5.
            use_wind: If False, the wind is zeroed (reduces to Riemannian).
            wind_mode: ``"soft"`` (default smooth causal clamp) or ``"raw"``
                (trusted pass-through). See the class docstring.
            wind_stiffness: Sharpness of the ``"soft"`` clamp transition.
                Ignored when ``wind_mode == "raw"``. Defaults to
                :const:`ham.utils.WIND_STIFFNESS`.
        """
        super().__init__(manifold=manifold)
        self.h_net = h_net
        self.w_net = w_net
        self.epsilon = float(epsilon)
        self.use_wind = bool(use_wind)
        self.wind_mode = str(wind_mode)
        self.wind_stiffness = float(wind_stiffness)
        if self.wind_mode not in ("soft", "raw"):
            raise ValueError(
                f"wind_mode must be 'soft' or 'raw', got {self.wind_mode!r}"
            )

    def __repr__(self) -> str:
        return (
            f"Randers(manifold={self.manifold}, epsilon={self.epsilon}, "
            f"wind_mode={self.wind_mode!r})"
        )

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
        max_speed = 1.0 - self.epsilon

        # wind_mode is a static (Python) field, so this branch is JIT-safe.
        if self.wind_mode == "raw":
            # Trusted prescribed field: pass through bit-exact. The caller
            # guarantees ||W||_H < 1; lambda is floored only as a NaN guard and
            # never engages for a valid wind.
            W_safe = W_raw
            lambda_factor = jnp.maximum(1.0 - w_norm_sq, GRAD_EPS)
        else:
            # Default: smooth, identity-preserving causal clamp.  Unlike the
            # historical ``max_speed * tanh(w_norm)/w_norm`` squash -- which bent
            # every wind (slope max_speed<1 at 0) and so silently distorted valid
            # currents -- this is the identity to ~exp(-stiffness*(max_speed -
            # w_norm)) inside the physical region and only bends within a thin
            # shell around the causal boundary, keeping F in C^infinity and
            # guaranteeing ||W_safe||_H < max_speed < 1 (strong convexity).
            # Reference: spec/MATH_SPEC.md section 5.
            scale = causal_wind_scale(w_norm, max_speed, self.wind_stiffness)
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
