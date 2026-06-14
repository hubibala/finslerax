"""Evaluation — executed-plan timing, recovery metrics, independent validation.

Nothing here optimizes; it measures. The executed-plan simulator reuses the exact
clock-threading of the planner, so "execute plan P under the true evolving
current" and "the time the planner believes P takes" are the same computation —
that is what makes the frozen-vs-time-aware comparison fair.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .planners import thread_clock


def executed_arrival_time(path, medium, glider, t0=0.0):
    """True time to traverse a geometric ``path`` under the evolving current."""
    T_total, _ = thread_clock(jnp.asarray(path, dtype=jnp.float32), medium, glider, t0)
    return T_total


def time_saved(reference_time, plan_time):
    """Fractional time saved by ``plan`` vs ``reference`` (positive = better)."""
    reference_time = float(reference_time)
    return (reference_time - float(plan_time)) / max(reference_time, 1e-9)


# =============================================================================
# Independent time-optimal solution for a *spatially uniform* current W(t)
# =============================================================================
def uniform_shooting_time(
    current_of_t, s_max, start, end, t0=0.0, t_hi=200.0, iters=80
):
    """Exact min arrival time for a spatially-uniform (time-varying) current.

    For uniform ``W(t)`` the Pontryagin optimum is a constant heading
    ``u = R/‖R‖`` with ``R(T) = D - ∫_{t0}^{t0+T} W dτ``, and the minimum time
    solves ``s_max · T = ‖R(T)‖``. Solved here by bisection on
    ``g(T) = s_max·T - ‖R(T)‖`` (monotone increasing), giving an optimizer-free
    ground truth for the time-lifted planner (Stage-C validation).

    Args:
        current_of_t: callable ``t -> W`` (shape ``(D,)``), space-independent.
        s_max: through-water speed.
        start, end: boundary points ``(D,)``.
        t0: departure time.
        t_hi: upper bracket for T.
        iters: bisection iterations.
    """
    start = jnp.asarray(start, dtype=jnp.float32)
    end = jnp.asarray(end, dtype=jnp.float32)
    D = end - start

    def integral_W(T, n=200):
        taus = jnp.linspace(0.0, T, n)
        Ws = jax.vmap(lambda tt: current_of_t(t0 + tt))(taus)
        return jnp.trapezoid(Ws, taus, axis=0)

    def g(T):
        R = D - integral_W(T)
        return s_max * T - jnp.linalg.norm(R)

    lo = jnp.asarray(1e-4)
    hi = jnp.asarray(float(t_hi))

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        cond = g(mid) > 0.0
        hi = jnp.where(cond, mid, hi)
        lo = jnp.where(cond, lo, mid)
        return (lo, hi)

    lo, hi = jax.lax.fori_loop(0, iters, body, (lo, hi))
    return 0.5 * (lo + hi)


# =============================================================================
# Current-field reconstruction metrics (Stage B)
# =============================================================================
def recovery_metrics(pred_current_fn, true_current_fn, points):
    """Cosine similarity + RMSE of a reconstructed current over ``points``.

    Args:
        pred_current_fn: callable ``x -> W_pred`` (shape ``(D,)``).
        true_current_fn: callable ``x -> W_true`` (shape ``(D,)``).
        points: evaluation points, shape ``(K, D)``.

    Returns:
        dict with ``cosine`` (mean cosine similarity of the horizontal current)
        and ``rmse``.
    """
    points = jnp.asarray(points, dtype=jnp.float32)
    Wp = jax.vmap(pred_current_fn)(points)
    Wt = jax.vmap(true_current_fn)(points)
    # Compare horizontal components (the geostrophic/observable part).
    Wp_h = Wp[:, :2]
    Wt_h = Wt[:, :2]
    dots = jnp.sum(Wp_h * Wt_h, axis=-1)
    np_ = jnp.linalg.norm(Wp_h, axis=-1)
    nt = jnp.linalg.norm(Wt_h, axis=-1)
    cosine = jnp.mean(dots / (np_ * nt + 1e-8))
    rmse = jnp.sqrt(jnp.mean(jnp.sum((Wp_h - Wt_h) ** 2, axis=-1)))
    return {"cosine": float(cosine), "rmse": float(rmse)}


def navigability_map(medium, glider, points, t=0.0):
    """Causality scalar ``lam = 1 - ‖W‖²_H`` over ``points`` (negative = stalls).

    ``lam <= 0`` marks genuinely non-navigable water (current exceeds the
    vehicle's top speed) — the regime the ``tanh`` squash caps and the reason a
    slow glider must route *with* the current.
    """
    points = jnp.asarray(points, dtype=jnp.float32)
    t = jnp.asarray(float(t))

    def lam(x):
        W = medium.physical_current(x, t)
        H = glider.sea_tensor(medium, x)
        return 1.0 - jnp.dot(W, jnp.dot(H, W))

    return jax.vmap(lam)(points)
