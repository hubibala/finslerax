"""Forecast (belief) models — what the glider *thinks* the ocean will do.

Stage C planned against the true future current: perfect foreknowledge. In
reality a vehicle plans against a forecast that is accurate now and degrades with
lead time, and corrects course as new data arrives (Stage D, closed loop).

A `Forecast` is issued at a time ``t_now`` and returns a *belief medium* — an
object with the same ``physical_current(x, t)`` / ``speed_factor(x)`` interface as
:class:`OceanMedium`, so the planner consumes it unchanged. Only the *current* is
treated as uncertain; the vehicle's drag map (``speed_factor`` → ``H``) is a known
property of the vehicle, so it passes through from the truth.

Three models, spanning the honest range:

* :class:`PerfectForecast` — knows the future exactly (the Stage-C assumption; the
  best attainable, a lower bound on arrival time).
* :class:`PersistenceForecast` — "the current will stay as it is now" (no skill; the
  worst forecast, an upper bound). Implemented with :class:`FrozenMedium`.
* :class:`DecayingForecast` — exact at issue time, blending toward persistence with
  lead time over a skill horizon ``skill`` (the realistic case).
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from .medium import FrozenMedium, OceanMedium


class DecayingBelief(eqx.Module):
    """Forecast skill decays with lead time toward persistence.

    ``W_belief(x, t) = W_now(x) + α(lead)·[W_true(x, t) - W_now(x)]`` with
    ``W_now(x) = W_true(x, t_now)``, ``lead = max(t - t_now, 0)`` and
    ``α = exp(-lead / skill)``. At ``t = t_now`` the belief equals the observed
    current; for short lead it tracks the true future; for long lead it reverts to
    persistence.
    """

    base: OceanMedium
    t_now: float = eqx.field(static=True)
    skill: float = eqx.field(static=True)

    def physical_current(self, x, t):
        now = self.base.physical_current(x, jnp.asarray(self.t_now))
        fut = self.base.physical_current(x, t)
        lead = jnp.maximum(t - self.t_now, 0.0)
        alpha = jnp.exp(-lead / self.skill)
        return now + alpha * (fut - now)

    def speed_factor(self, x):
        return self.base.speed_factor(x)


@dataclass(frozen=True)
class PerfectForecast:
    """The glider knows the future current exactly (Stage-C assumption)."""

    def issue(self, true_medium, t_now):
        return true_medium


@dataclass(frozen=True)
class PersistenceForecast:
    """The current is assumed to stay frozen at its observed (issue-time) value."""

    def issue(self, true_medium, t_now):
        return FrozenMedium(true_medium, float(t_now))


@dataclass(frozen=True)
class DecayingForecast:
    """Realistic forecast: exact now, reverting to persistence over ``skill``."""

    skill: float = 2.5

    def issue(self, true_medium, t_now):
        return DecayingBelief(true_medium, float(t_now), self.skill)


def forecast_error(true_medium, belief, points, lead_times, t_now):
    """Mean current-forecast error vs lead time (a forecast-skill diagnostic).

    Returns an array aligned with ``lead_times`` of the mean ``‖W_belief - W_true‖``
    over ``points`` at ``t = t_now + lead``.
    """
    points = jnp.asarray(points, dtype=jnp.float32)
    out = []
    for lead in lead_times:
        t = jnp.asarray(float(t_now) + float(lead))
        wb = jax.vmap(lambda p, t=t: belief.physical_current(p, t))(points)
        wt = jax.vmap(lambda p, t=t: true_medium.physical_current(p, t))(points)
        out.append(float(jnp.mean(jnp.linalg.norm(wb - wt, axis=-1))))
    return out
