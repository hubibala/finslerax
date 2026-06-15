"""Stage D — closed-loop replanning (MPC) under an imperfect forecast.

Stage C assumed perfect foreknowledge of the current. Here the glider only ever
sees a *forecast* that is accurate now and decays with lead time, and it re-plans
the remainder of the route each time it surfaces. The figure places the closed-loop
result between the two open-loop bounds and shows why it works.

Run:  python -m experiments.marine.run_stage_d_mpc
Writes experiments/marine/visualizations/stage_d_mpc.png
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import time_saved
from .forecast import DecayingForecast, PersistenceForecast, forecast_error
from .medium import FrozenMedium, OceanMedium
from .mpc import run_mpc
from .planners import TimeLiftedPlanner
from .vehicle import Glider

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

START = jnp.array([1.0, 5.0, 0.05])
END = jnp.array([9.0, 5.0, 0.05])
T0 = 4.0          # depart while the deep window is closed (reopens at tau = 8)
SKILL = 2.5


def _dive(center, n=28, amp=0.8):
    b = np.linspace(np.array(START), np.array(END), n + 1)
    s = np.linspace(0, 1, n + 1)
    b[:, 2] = np.clip(b[:, 2] + amp * np.exp(-((s - center) ** 2) / (2 * 0.18**2)), 0, 1)
    return jnp.asarray(b)


def main():
    medium = OceanMedium()
    glider = Glider(glide_angle_max_deg=None)

    # --- open-loop ideal (perfect foreknowledge = a lower bound on time) --
    tl = TimeLiftedPlanner(n_iters=500, lr=0.03, penalty_weight=80.0)
    print("Open-loop perfect-foreknowledge ideal + persistence reference...")
    perfect = tl.plan(medium, glider, START, END, t0=T0, n_steps=28,
                      init_path=_dive(0.7), n_restarts=2)
    # The frozen (persistence) plan is kept only as a visual reference in panel 3.
    frozen = tl.plan(FrozenMedium(medium, T0), glider, START, END, t0=T0, n_steps=28,
                     init_path=_dive(0.5))
    perf_t = float(perfect.arrival_time)

    # --- closed-loop MPC: same controller, forecast skill is the only variable
    print("Closed-loop MPC (decaying forecast vs persistence forecast)...")
    mpc_tl = TimeLiftedPlanner(n_iters=300, lr=0.03, penalty_weight=80.0)
    cl_dec = run_mpc(medium, glider, DecayingForecast(skill=SKILL), mpc_tl, START, END,
                     t0=T0, control_horizon=1.5, n_steps=20, n_restarts=1)
    cl_per = run_mpc(medium, glider, PersistenceForecast(), mpc_tl, START, END,
                     t0=T0, control_horizon=1.5, n_steps=20, n_restarts=1)
    skill_gain = time_saved(cl_per.arrival_time, cl_dec.arrival_time)

    print(f"  open-loop perfect (unattainable ideal) : {perf_t:.2f}")
    print(f"  closed-loop, decaying forecast         : {cl_dec.arrival_time:.2f}")
    print(f"  closed-loop, persistence forecast      : {cl_per.arrival_time:.2f}")
    print(f"  value of forecast skill (decaying vs persistence): {100 * skill_gain:+.1f}%")

    # --- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(14, 4.5))

    # (1) the unattainable ideal + the closed-loop pair (forecast skill isolated)
    ax1 = fig.add_subplot(1, 3, 1)
    labels = ["open-loop perfect\n(unattainable ideal)", "closed-loop\ndecaying forecast",
              "closed-loop\npersistence forecast"]
    vals = [perf_t, cl_dec.arrival_time, cl_per.arrival_time]
    colors = ["#2f9e44", "#e8833a", "#9aa3ad"]
    ax1.barh(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax1.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=10)
    ax1.set_xlim(0, max(vals) * 1.18)
    ax1.invert_yaxis()
    ax1.set_xlabel("elapsed travel time")
    ax1.set_title(f"Forecast skill saves {100 * skill_gain:.0f}%\n(same controller, closed loop)")

    # (2) forecast skill: error vs lead time
    ax2 = fig.add_subplot(1, 3, 2)
    leads = np.linspace(0, 2 * float(medium.tau), 40)
    belief = DecayingForecast(skill=SKILL).issue(medium, T0)
    grid = np.array([[x, y, 0.05] for x in np.linspace(2, 8, 6) for y in np.linspace(3, 7, 5)])
    err = forecast_error(medium, belief, grid, leads, T0)
    ax2.plot(leads, err, color="#2f6db5", lw=2)
    ax2.axvline(SKILL, color="#9aa3ad", ls="--", lw=1)
    ax2.text(SKILL + 0.1, max(err) * 0.9, "skill horizon", fontsize=9, color="#6b7480")
    ax2.set_xlabel("forecast lead time")
    ax2.set_ylabel("mean current error ‖ΔW‖")
    ax2.set_title("Why it works: the forecast is good\nnow and decays with lead time")

    # (3) flown depth profile vs the open-loop plans
    ax3 = fig.add_subplot(1, 3, 3)
    fp = np.array(cl_dec.flown_path)
    pp = np.array(perfect.path)
    sp = np.array(frozen.path)
    ax3.plot(sp[:, 0], sp[:, 2], color="#d6456b", lw=2, ls=":", label="persistence (open loop)")
    ax3.plot(pp[:, 0], pp[:, 2], color="#2f9e44", lw=2, ls="--", label="perfect (open loop)")
    ax3.plot(fp[:, 0], fp[:, 2], color="#e8833a", lw=2.5, marker="o", ms=3,
             label="flown (closed loop)")
    ax3.invert_yaxis()
    ax3.set_xlabel("east"), ax3.set_ylabel("depth")
    ax3.set_title("Flown path: dives to escape the\nadverse surface, as data arrives")
    ax3.legend(loc="lower right", fontsize=8)

    fig.suptitle("Stage D — closed-loop replanning under an imperfect forecast (MPC)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "stage_d_mpc.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
