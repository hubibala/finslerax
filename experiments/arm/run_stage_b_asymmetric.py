"""Stage B — asymmetric (gravity-Randers) C-space cost: the headline novelty.

A gravity-loaded arm has a Finsler configuration-space cost: lifting the links
costs more than lowering them. Two consequences, both shown here:

* **Asymmetry.** The geodesic cost A→B differs from B→A by tens of percent —
  the metric is genuinely directional (no symmetric Riemannian metric can do
  this). Actionable: whenever the *direction* of a motion is free (task
  sequencing, pick order, return legs), the Randers model says which way is
  cheap.
* **...yet the plan itself barely moves.** The gravity drift's 1-form
  ``β = gₛ·dU/λ`` is *exact*, so it is projectively (near-)invisible: the
  gravity-aware and gravity-blind geodesics coincide as point sets to
  ``O(gₛ²)``. The drift lives in the **ledger** (directed costs), not on the
  **map** (path shape). This is not a weakness of the planner — it is a
  theorem, and Stage D turns it into the identifiability structure of the
  inverse problem.

Run:  python -m experiments.arm.run_stage_b_asymmetric
Writes experiments/arm/visualizations/stage_b_asymmetric.png
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import kinetic_cost, polyline_deviation
from .medium import angle_manifold, build_arm_metric
from .planners import AVBDPlanner

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

GS = 0.12


def main():
    from .providers.synthetic import PlanarArm

    arm = PlanarArm(2, lengths=[1.0, 1.0], g=1.0)
    ang = angle_manifold(arm)
    m_sym = build_arm_metric(arm, manifold=ang, gravity_strength=0.0)
    m_asym = build_arm_metric(arm, manifold=ang, gravity_strength=GS)

    # A and B chosen so the direct path swings the arm up (raising potential),
    # while a detour keeps it low — the gravity-aware plan finds the detour.
    A = jnp.array([-1.1, 1.4])
    B = jnp.array([1.1, 1.4])
    planner = AVBDPlanner(step_size=0.04, iterations=500)

    sym = planner.plan(lambda a: m_sym, A, B, n_steps=32).xs
    asym = planner.plan(lambda a: m_asym, A, B, n_steps=32).xs
    asym_rev = planner.plan(lambda a: m_asym, B, A, n_steps=32).xs

    # The ledger: directed cost A->B != B->A under the gravity metric.
    cost_ab = float(kinetic_cost(m_asym, asym))
    cost_ba = float(kinetic_cost(m_asym, asym_rev))
    # The map: the exact drift is projectively invisible, so the gravity-aware
    # plan's shape (and hence its true cost) matches the gravity-blind plan's.
    true_sym = float(kinetic_cost(m_asym, sym))
    shape_dev = float(jnp.maximum(polyline_deviation(asym, sym),
                                  polyline_deviation(sym, asym)))

    print("Stage B — asymmetric gravity-Randers cost")
    print(f"  ledger: directed cost A->B = {cost_ab:.4f}   B->A = {cost_ba:.4f}   "
          f"direction choice saves {100 * abs(cost_ab - cost_ba) / max(cost_ab, cost_ba):.1f}%")
    print(f"  map:    plan-shape deviation aware vs blind = {shape_dev:.4f} rad "
          f"(projective invariance; true-cost gap {100 * (true_sym - cost_ab) / true_sym:.1f}%)")

    _figure(arm, ang, A, B, sym, asym, shape_dev, cost_ab, cost_ba)


def _figure(arm, ang, A, B, sym, asym, shape_dev, cost_ab, cost_ba):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) potential landscape over C-space + both plans (which coincide).
    ax = axes[0]
    g = np.linspace(-2.2, 2.2, 140)
    G1, G2 = np.meshgrid(g, g, indexing="ij")
    U = np.array(jax.vmap(arm.potential)(jnp.stack([G1.ravel(), G2.ravel()], -1))).reshape(G1.shape)
    cf = ax.contourf(G1, G2, U, levels=24, cmap="coolwarm")
    sp, ap = np.array(sym), np.array(asym)
    ax.plot(sp[:, 0], sp[:, 1], "-", color="k", lw=3.5, alpha=0.5, label="gravity-blind plan")
    ax.plot(ap[:, 0], ap[:, 1], "--", color="lime", lw=2.0,
            label=f"gravity-aware plan (dev {shape_dev:.3f} rad)")
    ax.plot(*np.array(A), "ko", ms=9), ax.plot(*np.array(B), "k*", ms=15)
    ax.set_title("The map: plans coincide as point sets\n(exact drift = projective invariance)")
    ax.set_xlabel(r"$\theta_1$"), ax.set_ylabel(r"$\theta_2$"), ax.legend(loc="lower center")
    fig.colorbar(cf, ax=ax, shrink=0.8, label="potential")

    # (b) the gravity drift field W(q) = -M⁻¹∇U — the source of the asymmetry.
    ax = axes[1]
    gq = np.linspace(-2.0, 2.0, 16)
    Q1, Q2 = np.meshgrid(gq, gq, indexing="ij")
    W = np.array(jax.vmap(
        lambda q: -jnp.linalg.solve(arm.inertia(q), arm.gravity(q))
    )(jnp.stack([Q1.ravel(), Q2.ravel()], -1)))
    ax.quiver(Q1, Q2, W[:, 0].reshape(Q1.shape), W[:, 1].reshape(Q1.shape), color="0.3", alpha=0.8)
    ax.plot(ap[:, 0], ap[:, 1], "-", color="lime", lw=2.5, label="plan")
    ax.set_title("Gravity drift field (downhill wind)\n— moving with it is cheaper")
    ax.set_xlabel(r"$\theta_1$"), ax.set_ylabel(r"$\theta_2$"), ax.legend(loc="lower center")

    # (c) the ledger: directed costs differ although the path does not.
    ax = axes[2]
    bars = ax.bar(["A→B\n(into the lift)", "B→A\n(with the drift)"],
                  [cost_ab, cost_ba], color=["crimson", "green"])
    ax.set_title(f"The ledger: same point set, directed costs differ\n"
                 f"direction choice saves {100 * abs(cost_ab - cost_ba) / max(cost_ab, cost_ba):.0f}%")
    ax.set_ylabel("gravity-metric cost")
    for b, v in zip(bars, [cost_ab, cost_ba]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")

    fig.suptitle("Stage B — the gravity drift is big in the ledger, invisible on the map", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "stage_b_asymmetric.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
