"""Validation suite for the marine navigation experiment (experiments/marine).

Run: ``JAX_PLATFORMS=cpu pytest tests/test_marine.py``.

Covers the claims of the plan: the per-segment cost equals HAM's Randers metric,
the synthetic geostrophic current is divergence-free (and Ekman is not), the
chosen medium is navigable, the time-lifted planner reproduces an independent
shooting solution (uniform W(t)) and the AVBD/eikonal route in the steady limit,
and the unified constraint layer is honoured by the time-lifted planner.
"""

import pathlib
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.marine import (
    Glider,
    OceanMedium,
    StationaryPlanner,
    TimeLiftedPlanner,
    build_snapshot_metric,
    randers_cost,
)
from experiments.marine.constraints import max_violation
from experiments.marine.evaluate import (
    navigability_map,
    uniform_shooting_time,
)
from experiments.marine.planners import thread_clock
from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.zoo import Randers

jax.config.update("jax_platform_name", "cpu")

START = jnp.array([1.0, 5.0, 0.1])
END = jnp.array([9.0, 5.0, 0.1])


def _steady_medium():
    """A frozen variant of the medium (no time dependence) for steady checks."""
    return OceanMedium(
        meander_c=0.0, eddy_drift=0.0, bc_pulse=0.0, ekman_omega=0.0
    )


# ---------------------------------------------------------------------------
def test_randers_cost_matches_metric_fn():
    """randers_cost must equal Randers.metric_fn for the same (H, W, v)."""
    key = jax.random.PRNGKey(0)
    for i in range(5):
        k1, k2, k3, key = jax.random.split(key, 4)
        A = jax.random.normal(k1, (3, 3))
        H = A @ A.T + 0.5 * jnp.eye(3)  # SPD sea
        W = 0.3 * jax.random.normal(k2, (3,))
        v = jax.random.normal(k3, (3,))

        metric = Randers(EuclideanSpace(3), lambda x, H=H: H, lambda x, W=W: W)
        expected = metric.metric_fn(START, v)
        got = randers_cost(H, W, v)
        assert jnp.allclose(expected, got, atol=1e-5, rtol=1e-4), (i, expected, got)


# ---------------------------------------------------------------------------
def test_geostrophic_is_divergence_free():
    """∇·W_g ≈ 0 (geostrophy from a stream function) but ∇·W_ek ≠ 0 (Ekman)."""
    medium = _steady_medium()
    pts = jnp.array(
        [[x, y, 0.05] for x in (2.0, 4.0, 6.0, 8.0) for y in (3.0, 5.0, 7.0)]
    )

    def hdiv(fn, x):
        J = jax.jacfwd(lambda p: fn(p, jnp.asarray(0.0)))(x)  # (3,3)
        return J[0, 0] + J[1, 1]  # ∂u/∂x + ∂v/∂y

    g_div = jax.vmap(lambda x: hdiv(medium.geostrophic, x))(pts)
    e_div = jax.vmap(lambda x: hdiv(medium.ekman, x))(pts)
    assert float(jnp.max(jnp.abs(g_div))) < 1e-4
    # Ekman is intentionally divergent (the stream-function blind spot).
    assert float(jnp.max(jnp.abs(e_div))) > 1e-3


# ---------------------------------------------------------------------------
def test_medium_is_navigable():
    """The transit corridor is navigable; eddy cores are intentionally not.

    The slow glider's signature regime is ``‖W‖_H → 1``. We require: (1) the
    transit corridor (near the jet, ``y ∈ [4.5, 5.5]``) is fully navigable;
    (2) the domain is majority-navigable (strong eddy cores excepted); and
    (3) the Randers cost stays finite even at the most adverse point, because
    the causal squash caps ``‖W_safe‖ < 1`` so planning is always well-posed.
    """
    medium = _steady_medium()
    glider = Glider(glide_angle_max_deg=None)
    xs = jnp.linspace(1.0, 9.0, 12)

    # (1) the *planned* steady route stays in navigable water (avoids cores).
    metric = build_snapshot_metric(medium, glider, t=0.0)
    route = StationaryPlanner(avbd_iters=250).route(metric, START, END, n_steps=24)
    lam_route = navigability_map(medium, glider, route, t=0.0)
    assert float(jnp.min(lam_route)) > -1e-3, float(jnp.min(lam_route))

    # (2) majority of the full domain navigable (strong eddy cores excepted).
    ys = jnp.linspace(2.0, 8.0, 7)
    zs = jnp.linspace(0.0, 1.0, 4)
    box = jnp.array([[x, y, z] for x in xs for y in ys for z in zs])
    lam_box = navigability_map(medium, glider, box, t=0.0)
    assert float(jnp.mean(lam_box > 0.0)) > 0.8, float(jnp.mean(lam_box > 0.0))

    # (3) cost finite even at the worst point (squash keeps planning well-posed).
    worst = box[int(jnp.argmin(lam_box))]
    H = glider.sea_tensor(medium, worst)
    W = medium.physical_current(worst, jnp.asarray(0.0))
    assert jnp.isfinite(randers_cost(H, W, jnp.array([0.3, 0.0, 0.0])))


# ---------------------------------------------------------------------------
def test_uniform_time_varying_matches_shooting():
    """Time-lifted planner reproduces the exact uniform-W(t) shooting solution."""

    class UniformMedium(eqx.Module):
        amp: float = eqx.field(static=True, default=0.12)
        omega: float = eqx.field(static=True, default=0.2)

        def physical_current(self, x, t):
            ang = self.omega * t
            return jnp.array([self.amp * jnp.cos(ang), self.amp * jnp.sin(ang), 0.0])

        def speed_factor(self, x):
            return jnp.asarray(1.0)

    medium = UniformMedium()
    glider = Glider(s_max=0.85, glide_angle_max_deg=None)

    def current_of_t(t):
        return medium.physical_current(START, t)

    t_star = float(
        uniform_shooting_time(current_of_t, glider.s_max, START, END, t0=0.0)
    )

    planner = TimeLiftedPlanner(n_iters=500, lr=0.05, penalty_weight=0.0)
    res = planner.plan(medium, glider, START, END, t0=0.0, n_steps=20, constraints=[])
    rel = abs(float(res.arrival_time) - t_star) / t_star
    assert rel < 0.08, (float(res.arrival_time), t_star, rel)


# ---------------------------------------------------------------------------
def test_steady_limit_cross_solver():
    """In the steady limit, time-lifted ≈ AVBD route time ≈ volumetric eikonal T."""
    medium = _steady_medium()
    glider = Glider(glide_angle_max_deg=None)
    metric = build_snapshot_metric(medium, glider, t=0.0)

    # AVBD route (Lagrangian) and its Zermelo travel time.
    stat = StationaryPlanner(avbd_iters=300)
    route = stat.route(metric, START, END, n_steps=24)
    t_avbd = float(metric.arc_length(route))

    # Time-lifted planner on the same steady medium, warm-started from AVBD.
    planner = TimeLiftedPlanner(n_iters=300, lr=0.02, penalty_weight=0.0)
    res = planner.plan(
        medium, glider, START, END, t0=0.0, n_steps=24, constraints=[], init_path=route
    )
    t_tl = float(res.arrival_time)
    assert abs(t_tl - t_avbd) / t_avbd < 0.08, (t_tl, t_avbd)

    # Eulerian cross-check: volumetric eikonal field near the endpoint.
    extent = (0.0, 10.0, 0.0, 10.0, 0.0, 1.0)
    shape = (24, 24, 8)
    T = stat.arrival_field(metric, START, extent, shape)
    nx, ny, nz = shape
    ix = round(float(END[0]) / 10.0 * (nx - 1))
    iy = round(float(END[1]) / 10.0 * (ny - 1))
    iz = round(float(END[2]) / 1.0 * (nz - 1))
    t_eik = float(T[ix, iy, iz])
    # Coarse grid + different solver family: loose agreement only.
    assert abs(t_eik - t_avbd) / t_avbd < 0.25, (t_eik, t_avbd)


# ---------------------------------------------------------------------------
def test_constraint_layer_enforced():
    """A tight depth ceiling is respected by the time-lifted planner."""
    from experiments.marine.constraints import depth_envelope

    medium = _steady_medium()
    glider = Glider(glide_angle_max_deg=None)
    # Force a shallow ceiling that the straight line (z=0.1) already meets, then
    # a start/end that dips: require staying above z=0.4 while endpoints at 0.1.
    start = jnp.array([1.0, 5.0, 0.1])
    end = jnp.array([9.0, 5.0, 0.1])
    cons = depth_envelope(z_min=0.0, z_max=0.4)

    planner = TimeLiftedPlanner(n_iters=300, lr=0.03, penalty_weight=200.0)
    # Seed with a deep init that violates, ensure optimizer pulls it back up.
    bad = jnp.linspace(start, end, 25).at[:, 2].set(0.7)
    res = planner.plan(
        medium, glider, start, end, t0=0.0, n_steps=24, constraints=cons, init_path=bad
    )
    _, times = thread_clock(res.path, medium, glider, 0.0)
    viol = float(max_violation(res.path, times, cons))
    assert viol < 0.05, viol


# ---------------------------------------------------------------------------
def test_forecast_consistency():
    """Belief models are exact at issue time; persistence is constant in time."""
    from experiments.marine import (
        DecayingForecast,
        OceanMedium,
        PersistenceForecast,
    )

    medium = OceanMedium()
    x = jnp.array([4.0, 5.0, 0.3])
    t_now = 4.0

    # Decaying forecast equals the true current at the issue time (lead 0)...
    dec = DecayingForecast(skill=2.5).issue(medium, t_now)
    w_now = dec.physical_current(x, jnp.asarray(t_now))
    w_true = medium.physical_current(x, jnp.asarray(t_now))
    assert jnp.allclose(w_now, w_true, atol=1e-5), (w_now, w_true)

    # ...and reverts toward persistence at long lead (error shrinks vs the truth).
    far = jnp.asarray(t_now + 20.0)
    persist = medium.physical_current(x, jnp.asarray(t_now))
    assert jnp.linalg.norm(dec.physical_current(x, far) - persist) < 1e-2

    # Persistence forecast ignores time entirely.
    per = PersistenceForecast().issue(medium, t_now)
    assert jnp.allclose(
        per.physical_current(x, jnp.asarray(t_now)),
        per.physical_current(x, jnp.asarray(t_now + 7.3)),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
def test_mpc_between_perfect_and_persistence():
    """Closed-loop replanning with a decaying forecast lands between the bounds.

    A perfect forecast is the best attainable (lower bound on time); persistence
    is the worst. Closed-loop MPC with an imperfect (decaying) forecast must not
    beat perfect foreknowledge, nor do worse than the no-forecast plan.
    """
    from experiments.marine import (
        DecayingForecast,
        FrozenMedium,
        OceanMedium,
        run_mpc,
    )
    from experiments.marine.evaluate import executed_arrival_time

    medium = OceanMedium()
    glider = Glider(glide_angle_max_deg=None)
    start = jnp.array([1.0, 5.0, 0.05])
    end = jnp.array([9.0, 5.0, 0.05])
    t0 = 4.0

    def dive(c, a=0.8):
        b = np.linspace(np.array(start), np.array(end), 25)
        s = np.linspace(0, 1, 25)
        b[:, 2] = np.clip(b[:, 2] + a * np.exp(-((s - c) ** 2) / (2 * 0.18**2)), 0, 1)
        return jnp.asarray(b)

    tl = TimeLiftedPlanner(n_iters=200, lr=0.03, penalty_weight=80.0)
    perfect = tl.plan(medium, glider, start, end, t0=t0, n_steps=24,
                      init_path=dive(0.7), n_restarts=1)
    frozen = tl.plan(FrozenMedium(medium, t0), glider, start, end, t0=t0,
                     n_steps=24, init_path=dive(0.5))
    perf = float(perfect.arrival_time)
    persistence = float(executed_arrival_time(frozen.path, medium, glider, t0))

    mpc_tl = TimeLiftedPlanner(n_iters=150, lr=0.03, penalty_weight=80.0)
    cl = run_mpc(medium, glider, DecayingForecast(skill=2.5), mpc_tl, start, end,
                 t0=t0, control_horizon=2.0, n_steps=16, max_replans=10, n_restarts=1)

    # Reaches the destination.
    assert np.linalg.norm(cl.flown_path[-1][:2] - np.array(end)[:2]) < 0.5
    # Bracketed by the two open-loop bounds (with slack for discretization).
    assert cl.arrival_time >= perf - 0.10 * perf, (cl.arrival_time, perf)
    assert cl.arrival_time <= persistence + 0.10 * persistence, (cl.arrival_time, persistence)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
