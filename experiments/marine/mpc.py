"""Stage D — closed-loop receding-horizon replanning (model-predictive control).

Stage C is open-loop: it commits to one plan computed against a known future. Real
glider operations are closed-loop — the vehicle surfaces periodically, takes in new
data (its position and a refreshed current forecast), re-plans the *remainder* of
the route, flies a short interval, and repeats. This is exactly model-predictive
control, and it is the honest way to handle forecast uncertainty: you no longer need
a perfect long-horizon forecast, only a decent short-horizon one, corrected as you go.

The controller reuses the existing pieces unchanged:

* a :class:`~experiments.marine.forecast.Forecast` produces the vehicle's *belief*
  medium at each replan (what it currently thinks the ocean will do);
* :class:`~experiments.marine.planners.TimeLiftedPlanner` re-plans from the current
  position and clock, **warm-started** from the previous plan for speed and temporal
  coherence;
* the *true* medium is used only to advance the glider — it is never seen by the
  planner, only experienced.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from ham.utils.config import DEFAULT_JNP_DTYPE, DEFAULT_NP_DTYPE

from .planners import thread_clock


class MPCResult(NamedTuple):
    """Outcome of a closed-loop run."""

    flown_path: np.ndarray       # (K, D) positions the glider actually occupied
    arrival_time: float          # ELAPSED travel time (matches planner/evaluate)
    plans: list                  # list of (N+1, D) planned paths, one per replan
    issue_times: list            # the absolute clock at each replan


def _advance(path, true_medium, glider, t_start, horizon):
    """Fly along ``path`` under the TRUE current for ``horizon`` of clock time.

    Returns ``(new_pos, new_t, reached_end)``. If the whole remaining path can be
    traversed within ``horizon``, the glider reaches the destination and ``new_t``
    is the true arrival time; otherwise it stops at the interpolated position.
    """
    path = jnp.asarray(path, dtype=DEFAULT_JNP_DTYPE)
    _, times = thread_clock(path, true_medium, glider, t_start)
    times = np.asarray(times)
    path_np = np.asarray(path)

    if times[-1] - t_start <= horizon + 1e-9:
        return path_np[-1], float(times[-1]), True

    target = t_start + horizon
    k = int(np.searchsorted(times, target) - 1)
    k = max(0, min(k, len(path_np) - 2))
    f = (target - times[k]) / (times[k + 1] - times[k] + 1e-9)
    pos = path_np[k] + f * (path_np[k + 1] - path_np[k])
    return pos, float(target), False


def _resample(path, pos, m):
    """Warm-start for the next solve: remaining geometry from ``pos``, ``m`` points.

    Takes the tail of ``path`` from the vertex nearest ``pos`` and resamples it by
    chord length, preserving the shape (e.g. a planned dive) for temporal coherence.
    """
    path = np.asarray(path)
    k = int(np.argmin(np.linalg.norm(path - pos, axis=1)))
    tail = np.vstack([pos, path[k:]])
    seg = np.linalg.norm(np.diff(tail, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return np.linspace(pos, path[-1], m)
    s = np.linspace(0.0, cum[-1], m)
    return np.stack([np.interp(s, cum, tail[:, d]) for d in range(tail.shape[1])], axis=1)


def run_mpc(
    true_medium,
    glider,
    forecast,
    planner,
    start,
    end,
    t0: float = 0.0,
    control_horizon: float = 1.5,
    n_steps: int = 20,
    max_replans: int = 20,
    arrival_radius: float = 0.4,
    constraints: list | None = None,
    n_restarts: int = 1,
) -> MPCResult:
    """Run the receding-horizon controller until the glider reaches ``end``.

    Args:
        true_medium: the real ocean (used only to advance the glider).
        glider: the vehicle.
        forecast: a :class:`Forecast` issued afresh at each replan.
        planner: a :class:`TimeLiftedPlanner`.
        start, end: boundary points ``(D,)``.
        t0: departure clock.
        control_horizon: clock time flown between replans (e.g. surfacing interval).
        n_steps: path segments per plan.
        max_replans: safety cap on iterations.
        arrival_radius: horizontal distance at which the destination counts as reached.
        constraints: optional :class:`Constraint` list (defaults to the glider's).
        n_restarts: per-replan multi-starts (lets a replan discover a new dive once
            the refreshed belief reveals the favorable window).
    """
    constraints = constraints if constraints is not None else glider.constraints()
    end_np = np.asarray(end, dtype=DEFAULT_NP_DTYPE)
    pos = np.asarray(start, dtype=DEFAULT_NP_DTYPE)
    t = float(t0)

    flown = [pos.copy()]
    plans, issue_times = [], []
    remaining = None

    for _ in range(max_replans):
        belief = forecast.issue(true_medium, t)
        res = planner.plan(
            belief, glider, jnp.asarray(pos), jnp.asarray(end_np), t0=t,
            n_steps=n_steps, constraints=constraints,
            init_path=None if remaining is None else jnp.asarray(remaining),
            n_restarts=n_restarts,
        )
        path = np.asarray(res.path)
        plans.append(path)
        issue_times.append(t)

        pos, t, reached = _advance(path, true_medium, glider, t, control_horizon)
        flown.append(np.asarray(pos).copy())

        if reached or np.linalg.norm(pos[:2] - end_np[:2]) < arrival_radius:
            if not reached:
                # Close the small residual hop to the destination under the truth.
                hop = jnp.asarray(np.stack([pos, end_np]))
                _, tt = thread_clock(hop, true_medium, glider, t)
                t = float(tt[-1])
                flown.append(end_np.copy())
            break

        remaining = _resample(path, pos, n_steps + 1)

    # Report ELAPSED travel time, matching TimeLiftedPlanner / executed_arrival_time.
    return MPCResult(np.asarray(flown), float(t - t0), plans, issue_times)
