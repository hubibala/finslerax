"""Stage A — solver validation on a Waddington-stiff asymmetric metric.

Given the *true* Randers metric (built from the known drift via Zermelo), this
stage validates the solver stack and the asymmetry/LAP physics before any
learning enters (PLAN Stage A, H3):

* exact geodesic via AVBD continuation (+ GN polish) vs the analytic minimum
  *action* (Onsager–Machlup) path — the LAP↔Randers correspondence;
* the directionality signal ``cost(late→early)/cost(early→late) > 1`` and that it
  grows with the flux ``κ``;
* a solver cost/convergence phase diagram vs path resolution ``N``.

Run:  python -m experiments.single_cell_synthetic.run_stage_a_solver
Writes visualizations/stage_a_solver.png
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import directionality_score, path_discrepancy, solver_diagnostics
from .landscape import Landscape, least_action_path
from .metric import build_true_metric
from .solvers import exact_geodesic

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)


def _grid_points(rng, n=400):
    return jnp.asarray(rng.uniform([-2.5, -1.5], [2.5, 1.5], size=(n, 2)), jnp.float32)


def main():
    rng = np.random.default_rng(0)
    pts = _grid_points(rng)

    # --- LAP correspondence + geodesic on the kappa=1 metric ---
    ls = Landscape(kappa=1.0)
    metric = build_true_metric(ls, pts, margin=0.8)
    term = ls.terminal_states()
    A = jnp.array([-2.0, 0.0], jnp.float32)
    B = term[1]
    geo = exact_geodesic(metric, A, B, n_steps=32)
    lap, lap_action = least_action_path(ls, A, B, n_steps=32)
    disc = path_discrepancy(np.asarray(geo.xs), np.asarray(lap))
    print(f"LAP<->Randers path discrepancy: {disc:.3f}  (true min-action {float(lap_action):.2f})")

    # --- directionality vs kappa, on the transverse commitment route ---
    # A constant-x₁ route across the fate axis has *no* gradient-tilt asymmetry,
    # so directionality is exactly 1 at κ=0 and *only* the rotational flux can
    # break it — the cleanest isolation of the non-conservative signal.
    TA = jnp.array([0.8, -0.8], jnp.float32)
    TB = jnp.array([0.8, 0.8], jnp.float32)
    kappas = [0.0, 0.25, 0.5, 1.0, 2.0]
    dirs = []
    for k in kappas:
        lk = Landscape(kappa=k)
        mk = build_true_metric(lk, pts, margin=0.8)
        d = directionality_score(mk, TA, TB, n_steps=24)
        dirs.append(d)
        print(f"  kappa={k:.2f}  directionality cost(rev)/cost(fwd) = {d:.3f}")

    # --- solver phase diagram: cost / convergence vs N ---
    Ns = [8, 16, 32, 48, 64]
    diags = [solver_diagnostics(metric, A, B, n_steps=n) for n in Ns]
    for n, dg in zip(Ns, diags):
        print(f"  N={n:3d}  arc_len={dg['arc_length']:.3f}  finite={dg['finite']}  wall={dg['wall_s']:.2f}s")

    # --- figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    gx = np.linspace(-2.6, 2.6, 60)
    gy = np.linspace(-1.6, 1.6, 60)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    phi = np.array([[float(ls.potential(jnp.array([x, y]))) for y in gy] for x in gx])
    ax.contourf(X, Y, phi, levels=24, cmap="terrain", alpha=0.8)
    g = np.asarray(geo.xs)
    lp = np.asarray(lap)
    ax.plot(g[:, 0], g[:, 1], "-", color="crimson", lw=2.5, label="Randers geodesic")
    ax.plot(lp[:, 0], lp[:, 1], "--", color="white", lw=2, label="true min-action (OM)")
    ax.plot(*np.asarray(A), "wo", ms=9, mec="k")
    ax.plot(*np.asarray(B), "w*", ms=15, mec="k")
    ax.set_title(f"Geodesic vs OM least-action\n(discrepancy {disc:.3f})")
    ax.set_xlabel("x₁ (developmental)"), ax.set_ylabel("x₂ (fate)"), ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(kappas, dirs, "o-", color="navy")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_title("Directionality grows with flux κ\n(transverse commitment route)")
    ax.set_xlabel("κ (flux fraction)"), ax.set_ylabel("cost(against)/cost(with) (>1)")

    ax = axes[2]
    ax.plot(Ns, [d["wall_s"] for d in diags], "s-", color="darkgreen")
    ax.set_title("Solver cost vs path resolution N\n(continuation keeps it finite)")
    ax.set_xlabel("N (path segments)"), ax.set_ylabel("wall-clock (s)")

    fig.suptitle("Stage A — solver validation on the true Waddington Randers metric", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "stage_a_solver.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
