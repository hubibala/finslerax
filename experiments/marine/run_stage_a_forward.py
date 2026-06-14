"""Stage A — forward stationary planning + cross-solver validation + depth-riding.

Builds the frozen medium, computes the global time-to-arrival field with the
volumetric eikonal, plans a route with AVBD, and shows the 3D payoff already in
the steady case: a depth-free plan dives to ride the reversed deep current and
beats a depth-locked (surface) plan.

Run:  python -m experiments.marine.run_stage_a_forward
Writes experiments/marine/visualizations/stage_a_forward.png
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .constraints import depth_envelope
from .evaluate import navigability_map, time_saved
from .medium import OceanMedium, build_snapshot_metric
from .planners import StationaryPlanner, TimeLiftedPlanner
from .vehicle import Glider

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

# Eastward mission: the surface jet (u = ∂ψ/∂y = -U·sech² < 0) flows *westward*,
# so an eastward transit fights it at the surface; the reversed deep layer flows
# eastward and assists — diving to ride it pays off.
START = jnp.array([1.0, 5.0, 0.05])
END = jnp.array([9.0, 5.0, 0.05])
EXTENT = (0.0, 10.0, 0.0, 10.0, 0.0, 1.0)


def main():
    # Steady medium with an explicitly *open* favorable deep layer (bc_base high,
    # no time variation) — isolates the depth-riding payoff in the stationary case.
    medium = OceanMedium(
        meander_c=0.0, eddy_drift=0.0, bc_base=0.8, bc_pulse=0.0, ekman_omega=0.0
    )
    glider = Glider(glide_angle_max_deg=None)
    metric = build_snapshot_metric(medium, glider, t=0.0)

    print("Solving volumetric eikonal arrival field...")
    stat = StationaryPlanner(max_iters=120, tol=1e-5, avbd_iters=300)
    shape = (40, 40, 12)
    T = np.asarray(stat.arrival_field(metric, START, EXTENT, shape))

    print("Planning routes (depth-locked vs depth-free)...")
    tl = TimeLiftedPlanner(n_iters=500, lr=0.03, penalty_weight=80.0)
    # Depth-locked (surface) baseline.
    surf = tl.plan(
        medium, glider, START, END, t0=0.0, n_steps=28,
        constraints=depth_envelope(0.0, 0.1),
    )
    # Depth-free: warm-start from a *diving* guess (a half-sine depth bump) so the
    # local optimizer can discover the deep favorable layer. This is the honest
    # mitigation for the local-minimum nature of the BVP (see README caveats).
    n_steps = 28
    base = np.linspace(np.asarray(START), np.asarray(END), n_steps + 1)
    bump = 0.85 * np.sin(np.linspace(0, np.pi, n_steps + 1))
    dive_init = base.copy()
    dive_init[:, 2] = np.clip(base[:, 2] + bump, 0.0, 1.0)
    deep = tl.plan(
        medium, glider, START, END, t0=0.0, n_steps=n_steps,
        constraints=depth_envelope(0.0, 1.0), init_path=jnp.asarray(dive_init),
        n_restarts=1,
    )

    ts = time_saved(surf.arrival_time, deep.arrival_time)
    print(f"  surface-locked arrival time : {float(surf.arrival_time):.3f}")
    print(f"  depth-riding  arrival time : {float(deep.arrival_time):.3f}")
    print(f"  depth-riding time saved    : {100 * ts:.1f}%")

    # --- figure ---
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    k = shape[2] // 2
    axx = np.linspace(EXTENT[0], EXTENT[1], shape[0])
    axy = np.linspace(EXTENT[2], EXTENT[3], shape[1])
    X, Y = np.meshgrid(axx, axy, indexing="ij")
    cf = ax1.contourf(X, Y, T[:, :, k], levels=24, cmap="turbo")
    ax1.contour(X, Y, T[:, :, k], levels=12, colors="white", linewidths=0.5, alpha=0.6)
    qx = axx[::4]
    qy = axy[::4]
    QX, QY = np.meshgrid(qx, qy, indexing="ij")
    z_mid = (k / (shape[2] - 1)) * EXTENT[5]
    W = np.array([
        [np.asarray(medium.physical_current(jnp.array([x, y, z_mid]), jnp.asarray(0.0)))[:2]
         for y in qy] for x in qx
    ])
    ax1.quiver(QX, QY, W[..., 0], W[..., 1], color="white", alpha=0.7, scale=12)
    ax1.plot(*np.asarray(START[:2]), "wo", ms=9, mec="k", label="start")
    ax1.plot(*np.asarray(END[:2]), "w*", ms=15, mec="k", label="end")
    ax1.set_title(f"Volumetric eikonal arrival field\n(mid-depth z={z_mid:.2f})")
    ax1.set_xlabel("east"), ax1.set_ylabel("north"), ax1.legend(loc="upper right")
    fig.colorbar(cf, ax=ax1, shrink=0.8, label="arrival time")

    ax2 = fig.add_subplot(1, 3, 2)
    pts = jnp.array([[x, y, 0.05] for x in axx for y in axy])
    lam = np.asarray(navigability_map(medium, glider, pts, t=0.0)).reshape(len(axx), len(axy))
    cm = ax2.contourf(X, Y, lam, levels=np.linspace(-1, 1, 21), cmap="RdBu", extend="both")
    ax2.contour(X, Y, lam, levels=[0.0], colors="k", linewidths=2)
    ax2.set_title("Navigability  lam = 1 - ||W||^2_H\n(red < 0: non-navigable cores)")
    ax2.set_xlabel("east"), ax2.set_ylabel("north")
    fig.colorbar(cm, ax=ax2, shrink=0.8, label="λ")

    ax3 = fig.add_subplot(1, 3, 3)
    sp = np.asarray(surf.path)
    dp = np.asarray(deep.path)
    ax3.plot(sp[:, 0], sp[:, 2], "--", color="gray", lw=2, label=f"surface-locked (T={float(surf.arrival_time):.2f})")
    ax3.plot(dp[:, 0], dp[:, 2], "-", color="crimson", lw=2.5, label=f"depth-riding (T={float(deep.arrival_time):.2f})")
    ax3.invert_yaxis()
    ax3.set_title(f"Side view — depth-riding saves {100 * ts:.0f}%\n(dives to the reversed deep current)")
    ax3.set_xlabel("east"), ax3.set_ylabel("depth"), ax3.legend(loc="lower right")

    fig.suptitle("Stage A — forward Zermelo planning through a 3D ocean current", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "stage_a_forward.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
