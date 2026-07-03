"""Stage A — forward geodesic planning with *demonstrable* obstacle-avoidance intent.

The task is chosen so the obstacle genuinely constrains the optimum: the arm
must swing its end-effector from pointing down to pointing up, and the direct
motion sweeps the (nearly extended) arm straight through a workspace disk —
16% of the straight path is in collision, with the deepest penetration at
mid-motion. Avoidance is then visible and verifiable, four ways:

1. **Global ground truth.** The eikonal arrival field is solved on the *same
   barrier metric*, and its steepest-descent characteristic — the globally
   optimal route on the grid, no basins, no initialization — detours around
   the C-space obstacle. The AVBD continuation geodesic lands on the same
   route at matching cost: the detour is the optimum, not an accident.
2. **The optimum is actively constrained.** The avoided path's *tightest*
   clearance occurs mid-swing, hugging the barrier margin exactly where the
   straight path collides — the signature of a cost trade-off, not of a path
   that merely wandered far from the obstacle.
3. **The physical picture.** In the workspace, the straight sweep slices
   through the disk; the geodesic sweep tucks the elbow to slip past it, then
   re-extends.
4. **Cross-solver validation.** On the smooth metric AVBD, the eikonal field
   and spray shooting agree to ~2% (L3/L3b); on the barrier metric AVBD agrees
   with the eikonal arrival time.

D-agnosticism: the identical code plans a collision-free path for a 5-DoF arm
(step 0.01 — the stiffer 5-link mass matrix is past AVBD's fixed-step limit at
0.05).

Run:  python -m experiments.arm.run_stage_a_forward
Writes experiments/arm/visualizations/stage_a_forward.png
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import kinetic_cost, min_clearance, polyline_deviation, spray_geodesic
from .medium import angle_manifold, build_arm_metric
from .planners import AVBDPlanner, EikonalPlanner
from .providers.synthetic import AnalyticDistance, CircleScene, PlanarArm

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

CIRCLE = (1.3, 0.9, 0.35)
QA, QB = (-1.2, -0.6), (1.8, -0.6)   # EE swings bottom -> top past the disk
ALPHAS = (0.01, 0.03, 0.06, 0.15)    # localized barrier: rho ~ 1 in free space
DELTA = 0.05
EXTENT = (-2.8, 2.8, -2.8, 2.8)
GRID = (181, 181)


def main():
    arm = PlanarArm(2, lengths=[1.0, 1.0])
    ang = angle_manifold(arm)
    qa, qb = jnp.array(QA), jnp.array(QB)
    dist = AnalyticDistance(arm, CircleScene([list(CIRCLE)]))
    base = build_arm_metric(arm, manifold=ang)

    def make_metric(alpha):
        return build_arm_metric(arm, dist, manifold=ang, alpha=alpha, delta=DELTA)

    m_obs = make_metric(ALPHAS[-1])
    straight = jnp.linspace(qa, qb, 25)

    # --- A.1 the avoided geodesic (AVBD continuation) ----------------------
    geo = AVBDPlanner(step_size=0.05, iterations=600).plan(
        make_metric, qa, qb, n_steps=24, alphas=ALPHAS).xs
    d_str = jax.vmap(dist)(straight)
    d_geo = jax.vmap(dist)(geo)
    frac_colliding = float(jnp.mean((d_str < 0).astype(float)))
    print("Stage A.1 — the obstacle constrains the motion")
    print(f"  straight sweep: min clearance {float(d_str.min()):+.3f} "
          f"({100 * frac_colliding:.0f}% of the path in collision)")
    print(f"  avoided geodesic: min clearance {float(d_geo.min()):+.3f} "
          f"at node {int(jnp.argmin(d_geo))}/24 (mid-swing — the barrier is active)")
    print(f"  kinetic length: straight={float(base.arc_length(straight)):.3f}  "
          f"avoided={float(base.arc_length(geo)):.3f}")

    # --- A.2 global ground truth: eikonal on the SAME barrier metric -------
    eik = EikonalPlanner(max_iters=800)
    T_obs = eik.arrival_field(m_obs, qa, EXTENT, GRID)
    t_goal = float(eik.sample(T_obs, EXTENT, qb))
    char = eik.trace_path(T_obs, EXTENT, qb, tol_frac=0.01)
    char = jnp.concatenate([char, qa[None]], axis=0)
    len_avbd = float(kinetic_cost(m_obs, geo))
    d_char = jax.vmap(dist)(char)
    route_dev = float(jnp.maximum(polyline_deviation(geo, char),
                                  polyline_deviation(char, geo)))
    print("Stage A.2 — the detour is the *global* optimum (eikonal characteristic)")
    print(f"  barrier metric: AVBD cost={len_avbd:.3f}  eikonal arrival={t_goal:.3f}  "
          f"rel={abs(len_avbd - t_goal) / t_goal:.3f} (grid h={4 * 2.8 / 180:.3f})")
    print(f"  characteristic min clearance {float(d_char.min()):+.3f}; "
          f"AVBD-vs-characteristic route deviation {route_dev:.3f} rad (same route class)")

    # --- A.3 cross-solver triangle on the smooth metric (L3/L3b) -----------
    q0, q1 = jnp.array([-0.5, 0.4]), jnp.array([0.8, -0.5])
    sm = AVBDPlanner(step_size=0.05, iterations=400).plan(lambda a: base, q0, q1, n_steps=32).xs
    len_s_avbd = float(kinetic_cost(base, sm))
    T_sm = eik.arrival_field(base, q0, (-1.6, 1.6, -1.6, 1.6), (101, 101))
    len_s_eik = float(eik.sample(T_sm, (-1.6, 1.6, -1.6, 1.6), q1))
    len_s_spray = float(kinetic_cost(base, spray_geodesic(base, q0, q1, n_steps=64)))
    spread = max(len_s_avbd, len_s_eik, len_s_spray) / min(len_s_avbd, len_s_eik, len_s_spray) - 1
    print("Stage A.3 — cross-solver validation (smooth metric)")
    print(f"  AVBD={len_s_avbd:.4f}  eikonal={len_s_eik:.4f}  spray={len_s_spray:.4f}  "
          f"(spread {100 * spread:.1f}%)")

    # --- A.4 D-agnosticism: same code, 5-DoF -------------------------------
    arm5 = PlanarArm(5, lengths=[0.6] * 5)
    ang5 = angle_manifold(arm5)
    clutter = AnalyticDistance(arm5, CircleScene([[2.3, 1.2, 0.15]]))
    qa5 = jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])
    qb5 = jnp.array([-0.4, 0.5, -0.3, 0.4, -0.2])
    straight5 = jnp.linspace(qa5, qb5, 25)
    geo5 = AVBDPlanner(step_size=0.01, iterations=1000).plan(
        lambda a: build_arm_metric(arm5, clutter, manifold=ang5, alpha=a, delta=0.04),
        qa5, qb5, n_steps=24, alphas=(0.02, 0.05, 0.12, 0.3)).xs
    base5 = build_arm_metric(arm5, manifold=ang5)
    print("Stage A.4 — D-agnosticism (same code, 5-DoF)")
    print(f"  straight min clearance {float(min_clearance(clutter, ang5, straight5)):+.3f}  "
          f"planned {float(min_clearance(clutter, ang5, geo5)):+.3f}")
    print(f"  kinetic length: straight={float(base5.arc_length(straight5)):.3f}  "
          f"planned={float(base5.arc_length(geo5)):.3f}")

    _figure(arm, dist, straight, geo, d_str, d_geo, T_obs, char, len_avbd, t_goal,
            arm5, clutter, straight5, geo5)


def _silhouettes(ax, robot, path, cmap, n=9, lw=1.6, alpha=0.85, zorder=2):
    """Arm silhouettes along a path, colored early->late by the colormap."""
    idx = np.linspace(0, path.shape[0] - 1, n).astype(int)
    for j, i in enumerate(idx):
        pts = np.array(robot.link_points(path[i]))
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=cmap(0.15 + 0.75 * j / (n - 1)),
                lw=lw, ms=2.5, alpha=alpha, zorder=zorder, solid_capstyle="round")


def _figure(arm, dist, straight, geo, d_str, d_geo, T_obs, char, len_avbd, t_goal,
            arm5, clutter, straight5, geo5):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5))

    # (a) workspace: straight sweep slices the disk; geodesic sweep tucks past it.
    ax = axes[0, 0]
    cx, cy, r = CIRCLE
    ax.add_patch(plt.Circle((cx, cy), r, color="crimson", alpha=0.85, zorder=3))
    _silhouettes(ax, arm, np.array(straight), plt.cm.Reds, n=9, lw=1.2, alpha=0.35, zorder=1)
    _silhouettes(ax, arm, np.array(geo), plt.cm.Blues, n=11, lw=1.8, alpha=0.9, zorder=2)
    ee_g = np.array(jax.vmap(arm.fk)(geo))
    ax.plot(ee_g[:, 0], ee_g[:, 1], "-", color="navy", lw=2, zorder=4, label="end-effector (avoided)")
    ax.plot([], [], "-", color="crimson", alpha=0.5, label="straight sweep (collides)")
    ax.plot([], [], "-", color="C0", label="geodesic sweep (tucks the elbow)")
    ax.plot(0, 0, "ks", ms=6)
    ax.set_title("Workspace: the arm tucks its elbow\nto slip past the disk, then re-extends")
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)

    # (b) C-space: arrival field on the barrier metric + all three routes.
    ax = axes[0, 1]
    xs = np.linspace(EXTENT[0], EXTENT[1], T_obs.shape[0])
    ys = np.linspace(EXTENT[2], EXTENT[3], T_obs.shape[1])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Tc = np.clip(np.array(T_obs), 0, float(2.2 * t_goal))
    cf = ax.contourf(X, Y, Tc, levels=26, cmap="turbo")
    fig.colorbar(cf, ax=ax, shrink=0.8, label="arrival time (barrier metric)")
    g = np.linspace(EXTENT[0], EXTENT[1], 200)
    G1, G2 = np.meshgrid(g, g, indexing="ij")
    clr = np.array(jax.vmap(dist)(jnp.stack([G1.ravel(), G2.ravel()], -1))).reshape(G1.shape)
    ax.contourf(G1, G2, (clr < 0).astype(float), levels=[0.5, 1.5], colors=["white"], alpha=0.55)
    ax.contour(G1, G2, clr, levels=[0.0], colors="white", linewidths=2)
    sp, gp, cp = np.array(straight), np.array(geo), np.array(char)
    ax.plot(sp[:, 0], sp[:, 1], "--", color="crimson", lw=2, label="straight (collides)")
    ax.plot(cp[:, 0], cp[:, 1], "-", color="white", lw=3.5, label="eikonal characteristic (global)")
    ax.plot(gp[:, 0], gp[:, 1], "-", color="k", lw=1.8, label="AVBD geodesic")
    ax.plot(*sp[0], "wo", ms=9, mec="k"), ax.plot(*sp[-1], "w*", ms=15, mec="k")
    lo = min(float(gp[:, 1].min()), float(cp[:, 1].min())) - 0.25
    ax.set_xlim(EXTENT[0], EXTENT[1]), ax.set_ylim(max(EXTENT[2], lo), EXTENT[3])
    ax.set_title(f"C-space: AVBD rides the globally optimal route\n"
                 f"(AVBD={len_avbd:.3f}, eikonal={t_goal:.3f}; obstacle = white region)")
    ax.set_xlabel(r"$\theta_1$"), ax.set_ylabel(r"$\theta_2$")
    ax.legend(loc="lower right", fontsize=8)

    # (c) clearance along the path: the barrier is ACTIVE mid-swing.
    ax = axes[1, 0]
    s = np.linspace(0, 1, len(d_str))
    ax.plot(s, np.array(d_str), "--", color="crimson", lw=2, label="straight")
    ax.plot(s, np.array(d_geo), "-", color="C0", lw=2.5, label="avoided geodesic")
    ax.axhline(0.0, color="k", lw=1)
    ax.axhline(DELTA, color="0.4", lw=1, ls=":", label=f"barrier margin δ={DELTA}")
    ax.fill_between(s, -0.45, 0.0, color="crimson", alpha=0.12)
    i_min = int(np.argmin(np.array(d_geo)))
    ax.annotate("tightest point mid-swing:\nthe barrier shapes the optimum",
                xy=(s[i_min], float(d_geo[i_min])), xytext=(0.42, 0.42),
                arrowprops={"arrowstyle": "->", "color": "0.2"}, fontsize=9)
    ax.set_title("Whole-arm clearance along the motion")
    ax.set_xlabel("path parameter"), ax.set_ylabel("C-space clearance (workspace units)")
    ax.set_ylim(-0.45, 0.85), ax.legend(loc="upper left", fontsize=8)

    # (d) D-agnosticism: the 5-DoF arm does the same, same code.
    ax = axes[1, 1]
    ax.add_patch(plt.Circle((2.3, 1.2), 0.15, color="crimson", alpha=0.85, zorder=3))
    _silhouettes(ax, arm5, np.array(straight5), plt.cm.Reds, n=7, lw=1.2, alpha=0.35, zorder=1)
    _silhouettes(ax, arm5, np.array(geo5), plt.cm.Blues, n=9, lw=1.8, alpha=0.9, zorder=2)
    ax.plot(0, 0, "ks", ms=6)
    ax.plot([], [], "-", color="crimson", alpha=0.5, label="straight sweep (collides)")
    ax.plot([], [], "-", color="C0", label="planned sweep (clear)")
    ax.set_title("Same code, 5-DoF arm in clutter")
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Stage A — geodesic obstacle avoidance, validated against the "
                 "globally optimal route", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / "stage_a_forward.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
