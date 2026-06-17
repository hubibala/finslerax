"""Ground-truth Waddington landscape — the "Medium" of the single-cell study.

Everything in this module is *known by construction*: the developmental
potential ``φ``, the tunable non-conservative flux ``κ·f_curl``, the total drift
``f = -∇φ + κ·f_curl``, the diffusion ``D``, and — crucially — the **exact Hodge
split** of ``f`` into its gradient (reversible) and solenoidal (irreversible)
parts. This is the synthetic advantage the real Weinreb data cannot offer
(`spec/single_cell_synthetic_PLAN.md` §5).

Design (a standard double-well-along-a-channel Waddington with a pitchfork):

* **Potential** ``φ(x) = -c·x₁ + b·(x₂⁴/4 - s(x₁)·x₂²/2)`` with the pitchfork
  control ``s(x₁) = s₀·tanh((x₁ - x_b)/w)``.  For ``x₁ ≪ x_b`` the ``x₂``
  potential is a single confining well at ``x₂ = 0`` (the progenitor channel);
  for ``x₁ ≫ x_b`` it is a symmetric double well at ``x₂ = ±√s₀`` (two terminal
  fates).  The ``-c·x₁`` tilt is the constant forward developmental push along
  ``x₁``.  The conservative drift ``-∇φ`` rolls cells downhill toward a fate.

* **Flux** ``f_curl = ∇^⊥ψ = (∂ψ/∂x₂, -∂ψ/∂x₁)`` for a stream function ``ψ``
  peaked at the saddle ``(x_b, 0)``.  Because it is the skew gradient of a
  scalar it is **exactly divergence-free** (``∇·f_curl ≡ 0``), so the Helmholtz
  / Hodge decomposition of ``f`` is, by construction,

      gradient (curl-free) part   = -∇φ
      solenoidal (div-free) part  = κ·∇^⊥ψ

  with no numerical projection needed.  This is what makes flux recovery
  *directly scorable* and is the whole reason the experiment is synthetic-first.

The ``κ`` axis (``0 → 2``) is the conservative→non-conservative knob that the
headline claim (Stage C) is a *function of*: a Randers metric with an exact
gradient one-form only tilts cost; only the non-exact (``κ>0``) part bends
geodesic shape irreversibly (PLAN §1).

**Mild-wind caveat.** The induced Zermelo wind must stay navigable
(``‖W‖_H < 1``).  :meth:`Landscape.max_drift_norm` reports the worst-case drift
magnitude over a region so callers can pick a ``wind_scale`` that keeps the
corridor navigable, exactly as ``experiments/marine`` does.

This module is autodiff-first: ``φ`` and ``ψ`` are the only hand-written
scalars; every field (``∇φ``, ``∇^⊥ψ``, divergence, curl) is obtained from
``jax.grad``/``jax.jacobian`` so the analytic Hodge identities are testable
rather than asserted.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class Landscape(eqx.Module):
    """A 2-D branching Waddington landscape with a tunable irreversible flux.

    All parameters are static floats so the landscape is a leaf-free PyTree and
    can be closed over inside ``jax.jit`` / ``lax.scan`` (mirrors
    ``experiments.marine.OceanMedium``).

    Convention: ``x = (x₁, x₂)`` where ``x₁`` is the developmental (pseudotime)
    axis and ``x₂`` is the lateral fate-decision axis.  The two terminal valleys
    live at ``x₂ ≈ ±√s₀`` for ``x₁`` past the bifurcation ``x_b``.
    """

    # Potential shape
    c: float = eqx.field(static=True, default=0.6)  # forward tilt along x1
    b: float = eqx.field(static=True, default=1.0)  # double-well stiffness
    s0: float = eqx.field(static=True, default=1.0)  # well half-separation^2
    x_b: float = eqx.field(static=True, default=0.0)  # bifurcation location
    w: float = eqx.field(static=True, default=0.8)  # pitchfork sharpness

    # Non-conservative flux
    kappa: float = eqx.field(static=True, default=1.0)  # flux fraction (the axis)
    psi_amp: float = eqx.field(static=True, default=1.0)  # stream-fn amplitude
    psi_sigma: float = eqx.field(static=True, default=1.0)  # eddy radius at saddle

    # Diffusion (isotropic)
    D: float = eqx.field(static=True, default=0.05)

    # -------------------------------------------------------------------------
    # Scalars (the only hand-written fields)
    # -------------------------------------------------------------------------
    def potential(self, x: jax.Array) -> jax.Array:
        """Developmental potential ``φ(x)`` (scalar)."""
        x1, x2 = x[0], x[1]
        s = self.s0 * jnp.tanh((x1 - self.x_b) / self.w)
        well = self.b * (x2**4 / 4.0 - s * x2**2 / 2.0)
        return -self.c * x1 + well

    def streamfunction(self, x: jax.Array) -> jax.Array:
        """Stream function ``ψ(x)`` peaked at the saddle (scalar).

        A Gaussian bump centred on the bifurcation ``(x_b, 0)`` so the induced
        flux circulates around the commitment point — cells circle as they
        commit, which is what breaks reversibility of the fate decision.
        """
        x1, x2 = x[0], x[1]
        d2 = (x1 - self.x_b) ** 2 + x2**2
        return self.psi_amp * jnp.exp(-d2 / (2.0 * self.psi_sigma**2))

    # -------------------------------------------------------------------------
    # Derived fields (autodiff)
    # -------------------------------------------------------------------------
    def grad_potential(self, x: jax.Array) -> jax.Array:
        """``∇φ(x)`` — the conservative direction (uphill)."""
        return jax.grad(self.potential)(x)

    def curl_flux(self, x: jax.Array) -> jax.Array:
        """Divergence-free flux ``∇^⊥ψ = (∂ψ/∂x₂, -∂ψ/∂x₁)``.

        Skew gradient of ``ψ``; exactly solenoidal (``∇·(∇^⊥ψ) ≡ 0``).
        """
        g = jax.grad(self.streamfunction)(x)
        return jnp.array([g[1], -g[0]])

    def drift(self, x: jax.Array) -> jax.Array:
        """Total SDE drift ``f(x) = -∇φ(x) + κ·∇^⊥ψ(x)``."""
        return -self.grad_potential(x) + self.kappa * self.curl_flux(x)

    def hodge_split(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Exact Hodge split of the drift at ``x``.

        Returns ``(grad_part, sol_part)`` with
        ``grad_part = -∇φ`` (curl-free, reversible) and
        ``sol_part = κ·∇^⊥ψ`` (divergence-free, the irreversible flux).
        Their sum is :meth:`drift`.  This split is exact by construction — no
        discrete Helmholtz solve — which is the synthetic ground truth that
        flux-recovery is scored against (PLAN §5.2, §8).
        """
        grad_part = -self.grad_potential(x)
        sol_part = self.kappa * self.curl_flux(x)
        return grad_part, sol_part

    # -------------------------------------------------------------------------
    # Diagnostics used by tests and by the mild-wind regime check
    # -------------------------------------------------------------------------
    def divergence(self, x: jax.Array, field) -> jax.Array:
        """Divergence ``∇·field`` at ``x`` (``field`` is ``x -> ℝ²``)."""
        J = jax.jacobian(field)(x)
        return jnp.trace(J)

    def scalar_curl(self, x: jax.Array, field) -> jax.Array:
        """2-D scalar curl ``∂field_y/∂x - ∂field_x/∂y`` at ``x``."""
        J = jax.jacobian(field)(x)
        return J[1, 0] - J[0, 1]

    def fate_of(self, x: jax.Array) -> jax.Array:
        """Terminal-fate label of a *late* state by its ``x₂`` sign.

        ``0`` = lower valley (``x₂ < 0``), ``1`` = upper valley (``x₂ > 0``).
        Only meaningful past the bifurcation; used to label clones by their
        terminal valley (PLAN §5.3).
        """
        return (x[1] > 0).astype(jnp.int32)

    def terminal_states(self) -> jax.Array:
        """The two analytic fate attractors at a representative late ``x₁``.

        Returns shape ``(2, 2)`` — rows are the lower/upper valley fixed points
        ``(x₁_late, ∓√s₀)`` where the double-well minima sit.
        """
        x1_late = self.x_b + 3.0 * self.w
        s = self.s0 * jnp.tanh((x1_late - self.x_b) / self.w)
        root = jnp.sqrt(jnp.maximum(s, 0.0))
        return jnp.array([[x1_late, -root], [x1_late, root]])

    def max_drift_norm(self, points: jax.Array) -> jax.Array:
        """Worst-case Euclidean drift magnitude over ``points`` (shape ``(K, 2)``).

        Used to size a navigable Zermelo ``wind_scale``: with a flat sea
        ``H = I`` the causal bound is ``wind_scale · max‖f‖ < 1`` (PLAN §5.2).
        """
        norms = jax.vmap(lambda p: jnp.linalg.norm(self.drift(p)))(points)
        return jnp.max(norms)


# =============================================================================
# Least-action (Onsager–Machlup / Freidlin–Wentzell) path utilities — H3
# =============================================================================
def om_action(path: jax.Array, landscape: Landscape, dt: float) -> jax.Array:
    """Discrete Onsager–Machlup / Freidlin–Wentzell action of a fixed-time path.

    For the SDE ``dx = f(x) dt + √(2D) dW`` the FW rate functional is

        S[x] = (1/4D) ∫ ‖ẋ - f(x)‖² dt

    (Dynamo's least-action ``S = ½∫(v-f)ᵀD⁻¹(v-f)dt`` with isotropic
    ``D⁻¹ = (1/2D)I``; PLAN §3, baseline table row "Dynamo LAP").  The minimiser
    is the most-probable transition path — the object HAM's Randers geodesic is
    claimed to recover (H3).

    Args:
        path: Waypoints, shape ``(N+1, 2)``, at uniform time spacing ``dt``.
        landscape: The ground-truth landscape providing ``f``.
        dt: Physical time between consecutive waypoints.

    Returns:
        Scalar action ``S`` (midpoint quadrature of ``f``).
    """
    xs0, xs1 = path[:-1], path[1:]
    v = (xs1 - xs0) / dt
    mid = 0.5 * (xs0 + xs1)
    f_mid = jax.vmap(landscape.drift)(mid)
    integrand = jnp.sum((v - f_mid) ** 2, axis=-1)
    return jnp.sum(integrand) * dt / (4.0 * landscape.D)


def least_action_path(
    landscape: Landscape,
    x0: jax.Array,
    x1: jax.Array,
    *,
    n_steps: int = 40,
    dt: float = 0.15,
    iters: int = 1500,
    lr: float = 5e-3,
    init_path: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Numeric minimum-action path between two fixed endpoints.

    Adam descent (with gradient clipping) on the interior vertices of a
    discretised path minimising :func:`om_action`; endpoints are pinned.  The
    OM action functional is stiff, so a plain fixed-step gradient descent
    diverges (the same critical-slowing / fixed-step pathology that motivates
    continuation for the Randers BVP, memory ``avbd-long-geodesic-diagnosis``);
    Adam + clipping is the robust workhorse.  Provides the *true* min-action
    reference for the LAP↔Randers check (H3) — computable only because we know
    ``f`` and ``D`` (PLAN §9).

    Returns ``(path, action)``.
    """
    import optax

    x0 = jnp.asarray(x0, dtype=jnp.float32)
    x1 = jnp.asarray(x1, dtype=jnp.float32)
    if init_path is None:
        path = jnp.linspace(x0, x1, n_steps + 1)
    else:
        path = jnp.asarray(init_path, dtype=jnp.float32)

    def action_of_interior(interior):
        full = jnp.concatenate([x0[None], interior, x1[None]], axis=0)
        return om_action(full, landscape, dt)

    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
    interior = path[1:-1]
    opt_state = opt.init(interior)
    grad_fn = jax.value_and_grad(action_of_interior)

    @jax.jit
    def body(carry, _):
        interior, opt_state = carry
        val, g = grad_fn(interior)
        updates, opt_state = opt.update(g, opt_state)
        interior = optax.apply_updates(interior, updates)
        return (interior, opt_state), val

    (interior, _), _ = jax.lax.scan(body, (interior, opt_state), None, length=iters)
    full = jnp.concatenate([x0[None], interior, x1[None]], axis=0)
    return full, om_action(full, landscape, dt)
