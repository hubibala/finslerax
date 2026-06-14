"""Stage C — time-dependent depth-riding (the novelty).

The stationary eikonal cannot represent ``W(x, t)``; the time-lifted planner can.
We compare two plans by *executing both under the true evolving current*:

* **frozen-field plan** — optimized against the current frozen at departure
  (``W(x, t0)``), the implicit assumption of any stationary planner;
* **time-aware plan** — optimized against the true clock-threaded cost.

The favorable deep layer opens and closes with period ``tau`` and the eddies/jet
advect, so the frozen plan's route is stale by the time the glider flies it; the
time-aware plan times its dive to the open window and routes around the *future*
eddy positions.

Run:  python -m experiments.marine.run_stage_c_timedependent
Writes experiments/marine/visualizations/stage_c_timedependent.png
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import executed_arrival_time, time_saved
from .medium import FrozenMedium, OceanMedium
from .planners import TimeLiftedPlanner
from .vehicle import Glider

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

START = jnp.array([1.0, 5.0, 0.05])
END = jnp.array([9.0, 5.0, 0.05])
# Depart while the deep favorable window is CLOSED (it reopens at t = tau = 8): a
# frozen planner sees no reason to dive, but the window opens mid-transit.
T0 = 4.0
N_STEPS = 28


def _dive_init(n_steps, center=0.5, amp=0.8):
    """Straight line with a Gaussian depth bump centred at route fraction ``center``."""
    base = np.linspace(np.asarray(START), np.asarray(END), n_steps + 1)
    s = np.linspace(0, 1, n_steps + 1)
    base[:, 2] = np.clip(base[:, 2] + amp * np.exp(-((s - center) ** 2) / (2 * 0.18**2)), 0, 1)
    return jnp.asarray(base)


def main():
    medium = OceanMedium()  # fully time-varying (meander + eddy drift + window)
    glider = Glider(glide_angle_max_deg=None)
    tl = TimeLiftedPlanner(n_iters=600, lr=0.03, penalty_weight=80.0)

    print("Planning frozen-field route (assumes W frozen at t0)...")
    frozen_medium = FrozenMedium(medium, T0)
    # Offer the frozen planner the same diving option; with the window closed at
    # t0 it sees no benefit and returns a (near-)surface route.
    frozen = tl.plan(
        frozen_medium, glider, START, END, t0=T0, n_steps=N_STEPS,
        init_path=_dive_init(N_STEPS, center=0.5), n_restarts=1,
    )
    # Frozen planner *believes* this time; reality differs:
    frozen_belief = float(frozen.arrival_time)
    frozen_executed = float(executed_arrival_time(frozen.path, medium, glider, T0))

    print("Planning time-aware route (true clock-threaded cost)...")
    # Warm-start from a *late* dive (the window opens in the second half of the
    # transit, once the clock passes t = tau).
    aware = tl.plan(
        medium, glider, START, END, t0=T0, n_steps=N_STEPS,
        init_path=_dive_init(N_STEPS, center=0.7), n_restarts=2,
    )
    aware_executed = float(aware.arrival_time)

    saved = time_saved(frozen_executed, aware_executed)
    print(f"  frozen plan — believed time     : {frozen_belief:.3f}")
    print(f"  frozen plan — executed (true)   : {frozen_executed:.3f}")
    print(f"  time-aware plan — executed      : {aware_executed:.3f}")
    print(f"  time-aware advantage            : {100 * saved:.1f}% faster")

    # --- figure ---
    fz = np.asarray(frozen.path)
    aw = np.asarray(aware.path)

    fig = plt.figure(figsize=(14, 5))

    def current_slice(ax, t, title):
        axx = np.linspace(0, 10, 22)
        axy = np.linspace(0, 10, 22)
        QX, QY = np.meshgrid(axx, axy, indexing="ij")
        W = np.array([
            [np.asarray(medium.physical_current(jnp.array([x, y, 0.05]), jnp.asarray(float(t))))[:2]
             for y in axy] for x in axx
        ])
        spd = np.linalg.norm(W, axis=-1)
        ax.contourf(QX, QY, spd, levels=16, cmap="Blues", alpha=0.7)
        ax.quiver(QX, QY, W[..., 0], W[..., 1], color="navy", alpha=0.6, scale=12)
        ax.plot(fz[:, 0], fz[:, 1], "--", color="gray", lw=2, label="frozen plan")
        ax.plot(aw[:, 0], aw[:, 1], "-", color="crimson", lw=2.5, label="time-aware")
        ax.plot(*np.asarray(START[:2]), "ko", ms=8)
        ax.plot(*np.asarray(END[:2]), "k*", ms=14)
        ax.set_title(title), ax.set_xlabel("east"), ax.set_ylabel("north")
        ax.set_aspect("equal")

    ax1 = fig.add_subplot(1, 3, 1)
    current_slice(ax1, T0, f"Surface current @ t0={T0:.0f}")
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    current_slice(ax2, T0 + 0.5 * aware_executed, "Surface current @ mid-transit\n(eddies/jet have moved)")

    ax3 = fig.add_subplot(1, 3, 3)
    s_fz = np.linspace(0, 1, len(fz))
    s_aw = np.linspace(0, 1, len(aw))
    ax3.plot(s_fz, fz[:, 2], "--", color="gray", lw=2,
             label=f"frozen (exec {frozen_executed:.2f})")
    ax3.plot(s_aw, aw[:, 2], "-", color="crimson", lw=2.5,
             label=f"time-aware (exec {aware_executed:.2f})")
    ax3.invert_yaxis()
    ax3.set_title(f"Depth profile — time-aware {100 * saved:.0f}% faster\n(dives when the deep window is open)")
    ax3.set_xlabel("fraction of route"), ax3.set_ylabel("depth")
    ax3.legend(loc="lower right", fontsize=8)

    fig.suptitle("Stage C — time-dependent Zermelo routing (frozen-field vs time-aware)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "stage_c_timedependent.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
