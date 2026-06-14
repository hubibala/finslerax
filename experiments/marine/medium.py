"""Physically-grounded synthetic ocean medium + the Randers travel-time cost.

The medium follows the Helmholtz decomposition of a real ocean surface/interior
current into named components (no capability-for-its-own-sake — each term maps to
an oceanographic process):

* **Geostrophic** ``W_g = ∇^⊥ψ`` — the rotational, *divergence-free* mesoscale
  flow (a meandering jet + Gaussian eddies). ``ψ`` is the sea-surface-height /
  stream function; ``W_g = (∂ψ/∂y, -∂ψ/∂x)`` is geostrophy itself. This is why a
  stream-function prior is the physically correct reconstruction model (Stage B).
* **Baroclinic vertical structure** ``F(z) = cos(π z / Z_bc)`` — a first-mode
  profile that *reverses* the geostrophic current at depth, creating the
  depth-riding opportunity that makes the 3D problem interesting.
* **Ekman** ``W_ek`` — a wind-driven, surface-trapped, *divergent* (curl-free)
  ageostrophic drift. It is deliberately NOT representable by ``∇^⊥ψ`` so that the
  stream-function reconstruction has an honest, characterizable blind spot.
* **Time variation** — the jet meander phase advects and the deep favorable layer
  opens/closes with period ``tau``; this is what breaks the stationary eikonal and
  motivates the time-lifted planner (Stage C).

Vertical ocean current is ~mm/s and neglected (``W_z = 0``); depth changes are
vehicle-controlled.

``randers_cost`` is the per-segment Zermelo travel time — the exact Randers cost
of :class:`ham.geometry.zoo.Randers` (including the causal ``tanh`` squash),
factored out so the history-dependent time-lifted planner can thread the clock
without instantiating a metric per segment. ``tests/test_marine.py`` asserts it is
numerically identical to ``Randers.metric_fn``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.zoo import Randers
from ham.utils import GRAD_EPS


# =============================================================================
# Randers (Zermelo) per-segment travel-time cost
# =============================================================================
def randers_cost(
    H: jax.Array,
    W_raw: jax.Array,
    v: jax.Array,
    epsilon: float = 1e-5,
    eps: float = GRAD_EPS,
) -> jax.Array:
    """Zermelo time to traverse displacement ``v`` through current ``W_raw``.

    Mirrors :meth:`ham.geometry.zoo.Randers.zermelo_data` (causal ``tanh`` squash
    enforcing ``‖W‖_H < 1``) and :meth:`Randers.metric_fn` exactly, for a flat
    (Euclidean) manifold where ``to_tangent`` is the identity.

    Args:
        H: Riemannian sea tensor at the segment midpoint, shape ``(D, D)``.
        W_raw: Raw physical current at the midpoint, shape ``(D,)``.
        v: Displacement (segment) vector, shape ``(D,)``.
        epsilon: Causality margin (``max_speed = 1 - epsilon``).
        eps: Numerical floor.

    Returns:
        Scalar travel time ``F(midpoint, v) >= 0``.
    """
    H = 0.5 * (H + H.T)

    w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))
    w_norm = jnp.sqrt(jnp.maximum(w_norm_sq, eps))
    max_speed = 1.0 - epsilon
    scale = (max_speed * jnp.tanh(w_norm)) / (w_norm + eps)
    W = W_raw * scale

    safe_w_norm_sq = jnp.dot(W, jnp.dot(H, W))
    lam = 1.0 - safe_w_norm_sq

    v_sq_raw = jnp.sum(v**2)
    is_zero = v_sq_raw < eps
    v_safe = jnp.where(is_zero, v + jnp.sqrt(eps), v)

    Hv = jnp.dot(H, v_safe)
    HW = jnp.dot(H, W)
    v_sq_h = jnp.dot(v_safe, Hv)
    W_dot_v = jnp.dot(v_safe, HW)

    discriminant = lam * v_sq_h + W_dot_v**2
    cost = (jnp.sqrt(jnp.maximum(discriminant, eps)) - W_dot_v) / lam
    return jnp.where(is_zero, 0.0, cost)


# =============================================================================
# Ocean medium
# =============================================================================
class OceanMedium(eqx.Module):
    """A time-varying, depth-stratified synthetic ocean current field (3D).

    All quantities are non-dimensional: lengths in basin units, speeds normalized
    so the glider's through-water speed is ``1`` (set on the :class:`Glider`).
    Domain convention: ``x = (east, north, depth)`` with depth increasing
    downward, ``depth ∈ [0, z_max]`` (0 = surface).

    Attributes are static floats so the medium is a valid (leaf-free) PyTree and
    can be closed over inside ``jax.jit`` / ``lax.scan``.
    """

    # Geostrophic jet (meandering, Bickley-like)
    jet_y: float = eqx.field(static=True, default=5.0)
    jet_speed: float = eqx.field(static=True, default=0.5)
    jet_width: float = eqx.field(static=True, default=1.2)
    meander_amp: float = eqx.field(static=True, default=0.9)
    meander_k: float = eqx.field(static=True, default=0.7)
    meander_c: float = eqx.field(static=True, default=0.25)  # phase speed

    # Geostrophic eddies (Gaussian, advecting). Cores are deliberately strong:
    # the slow glider cannot stem them (‖W‖_H > 1), so they are avoided — the
    # featured near-boundary regime, not a bug.
    eddy_strength: float = eqx.field(static=True, default=1.8)
    eddy_radius: float = eqx.field(static=True, default=1.1)
    eddy_drift: float = eqx.field(static=True, default=0.18)

    # Baroclinic two-layer (thermocline) structure: an adverse surface layer over
    # a strong, broad, *reversed* deep layer — the depth-riding opportunity.
    z_max: float = eqx.field(static=True, default=1.0)
    z_thermocline: float = eqx.field(static=True, default=0.45)
    bc_width: float = eqx.field(static=True, default=0.08)
    bc_reversal: float = eqx.field(static=True, default=1.1)  # deep reversal gain
    bc_base: float = eqx.field(static=True, default=0.0)  # window-closed level
    bc_pulse: float = eqx.field(static=True, default=1.0)  # window-open level
    tau: float = eqx.field(static=True, default=8.0)  # favorable-window period

    # Ekman (divergent, surface-trapped, wind-driven)
    ekman_strength: float = eqx.field(static=True, default=0.12)
    ekman_depth: float = eqx.field(static=True, default=0.25)
    ekman_omega: float = eqx.field(static=True, default=0.15)  # wind rotation rate

    # Covariate "slow lens" (cold/dense water -> more drag -> slower vehicle)
    lens_x: float = eqx.field(static=True, default=5.0)
    lens_y: float = eqx.field(static=True, default=5.0)
    lens_radius: float = eqx.field(static=True, default=2.0)
    lens_drag: float = eqx.field(static=True, default=0.3)

    # ---- geostrophic stream function ψ(x, y, t) ----
    def streamfunction(self, xy: jax.Array, t: jax.Array) -> jax.Array:
        """Sea-surface-height stream function ψ at horizontal point ``xy``."""
        x, y = xy[0], xy[1]

        # Meandering zonal jet: ψ_jet = -U L tanh((y - y_c(x,t)) / L)
        y_c = self.jet_y + self.meander_amp * jnp.sin(
            self.meander_k * (x - self.meander_c * t)
        )
        psi_jet = -self.jet_speed * self.jet_width * jnp.tanh(
            (y - y_c) / self.jet_width
        )

        # Two counter-rotating eddies that drift eastward with time.
        def eddy(cx0, cy0, sign):
            cx = cx0 + self.eddy_drift * t
            d2 = (x - cx) ** 2 + (y - cy0) ** 2
            return sign * self.eddy_strength * jnp.exp(-d2 / (2.0 * self.eddy_radius**2))

        psi_e = eddy(3.0, 8.0, 1.0) + eddy(7.0, 2.0, -1.0)
        return psi_jet + psi_e

    def _baroclinic_amp(self, z: jax.Array, t: jax.Array) -> jax.Array:
        """Depth × time modulation of the geostrophic current.

        Two-layer profile: ``+1`` (full surface current) above the thermocline,
        smoothly transitioning to ``-reversal·window`` below it. The deep
        *reversal* strength is gated by a periodic favorable window (the deep
        layer that opens and closes), so timing the dive matters (Stage C).
        """
        deep_frac = jax.nn.sigmoid((z - self.z_thermocline) / self.bc_width)
        window = self.bc_base + self.bc_pulse * jnp.clip(
            jnp.sin(2.0 * jnp.pi * t / self.tau), 0.0, 1.0
        )
        reversal = self.bc_reversal * window
        return (1.0 - deep_frac) * 1.0 + deep_frac * (-reversal)

    def geostrophic(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Divergence-free geostrophic current ``W_g = ∇^⊥ψ · F(z, t)``."""
        xy = x[:2]
        z = x[2]
        grad_psi = jax.grad(self.streamfunction)(xy, t)  # (2,) = (∂ψ/∂x, ∂ψ/∂y)
        # ∇^⊥ψ = (∂ψ/∂y, -∂ψ/∂x)
        w_h = jnp.array([grad_psi[1], -grad_psi[0]])
        amp = self._baroclinic_amp(z, t)
        return jnp.array([amp * w_h[0], amp * w_h[1], 0.0])

    def ekman(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Curl-free, divergent, surface-trapped wind drift (ageostrophic).

        Built as ``E0 · e^{-z/d} · ∇φ`` with a radial potential ``φ`` so the field
        has nonzero divergence and is *not* representable by a stream function —
        the honest blind spot of the Stage-B geostrophic reconstruction.
        """
        x0, y0 = x[0], x[1]
        z = x[2]
        # Slowly rotating wind direction modulates a divergent radial drift.
        ang = self.ekman_omega * t
        cx = self.lens_x + 1.5 * jnp.cos(ang)
        cy = self.lens_y + 1.5 * jnp.sin(ang)
        r = jnp.array([x0 - cx, y0 - cy])
        decay = jnp.exp(-z / self.ekman_depth)
        # ∇(½‖r‖²) = r  -> divergent (∇·W_ek ≠ 0)
        w_h = self.ekman_strength * decay * r / (1.0 + jnp.sum(r**2))
        return jnp.array([w_h[0], w_h[1], 0.0])

    def physical_current(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Total raw ocean current ``W(x, t)`` (geostrophic + Ekman), shape (3,)."""
        return self.geostrophic(x, t) + self.ekman(x, t)

    # ---- covariate -> vehicle speed ----
    def speed_factor(self, x: jax.Array) -> jax.Array:
        """Multiplicative drag factor in ``(0, 1]`` from a cold/dense "lens".

        ``s(x) = s_max · speed_factor(x)``; the lens lowers achievable speed,
        raising the Riemannian sea ``H = I / s²`` (the demo_eikonal_fronts lens).
        """
        d2 = (x[0] - self.lens_x) ** 2 + (x[1] - self.lens_y) ** 2
        lens = jnp.exp(-d2 / self.lens_radius**2)
        return 1.0 - self.lens_drag * lens

    def temperature(self, x: jax.Array) -> jax.Array:
        """Synthetic covariate (proxy temperature) driving ``speed_factor``."""
        return self.speed_factor(x)


class FrozenMedium(eqx.Module):
    """Wraps a time-varying medium so it reports the current frozen at ``t0``.

    Used to build the *frozen-field* planner that ignores the evolution of the
    current — the baseline the time-aware planner is compared against in Stage C.
    """

    base: OceanMedium
    t0: float = eqx.field(static=True)

    def physical_current(self, x, t):
        return self.base.physical_current(x, jnp.asarray(self.t0))

    def speed_factor(self, x):
        return self.base.speed_factor(x)


def build_snapshot_metric(medium: OceanMedium, glider, t: float) -> Randers:
    """Freeze the medium at time ``t`` into a stationary Randers metric.

    Reuses all existing HAM machinery (eikonal / AVBD) for the *frozen* problem.
    The ``tanh`` squash inside :class:`Randers` models the vehicle's navigability
    limit (it cannot stem a current stronger than its top speed).
    """
    t_arr = jnp.asarray(float(t))
    manifold = EuclideanSpace(glider.dim)

    def h_net(x):
        return glider.sea_tensor(medium, x)

    def w_net(x):
        return medium.physical_current(x, t_arr)

    return Randers(manifold, h_net, w_net, epsilon=glider.epsilon)
