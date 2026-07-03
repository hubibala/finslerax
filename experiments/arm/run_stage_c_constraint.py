"""Stage C — exact task constraints via the Augmented Lagrangian (the upright cup).

A 3-link arm carries a cup: the end-effector orientation must stay level along the
whole motion. AVBD enforces this equality ``c(q) = 0`` natively with its ALM.
We show: (1) the unconstrained geodesic lets the cup tilt; (2) the ALM holds it to
tolerance; (3) the ALM beats a fixed-penalty baseline at the same smoothness.

Run:  python -m experiments.arm.run_stage_c_constraint
Writes experiments/arm/visualizations/stage_c_constraint.png
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from ham.solvers import AVBDSolver

from .constraints import max_constraint_violation, upright_constraint
from .medium import angle_manifold, build_arm_metric
from .providers.synthetic import PlanarArm

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)


def _penalty_plan(metric, ang, qs, qg, c, mu_hi, n_steps, iters=1500, lr=8e-3):
    """Penalty-only baseline with penalty continuation (a fair, well-tuned penalty).

    Even with a geometric ``mu`` ramp and gradient clipping, a pure penalty
    plateaus at a residual ``~1/mu`` violation (the classic ill-conditioning),
    which the ALM's dual variables remove.
    """
    inner0 = jnp.linspace(qs, qg, n_steps + 1)[1:-1]

    def loss(inner, mu):
        path = jnp.concatenate([qs[None], inner, qg[None]], axis=0)
        vs = jax.vmap(ang.log_map)(path[:-1], path[1:])
        energy = jnp.sum(jax.vmap(metric.energy)(path[:-1], vs))
        return energy + mu * jnp.mean(jax.vmap(c)(inner) ** 2)

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = opt.init(inner0)
    grad = jax.jit(jax.grad(loss))
    inner = inner0
    for i in range(iters):
        mu = 20.0 * (mu_hi / 20.0) ** (i / (iters - 1))
        upd, state = opt.update(grad(inner, mu), state)
        inner = optax.apply_updates(inner, upd)
    return jnp.concatenate([qs[None], inner, qg[None]], axis=0)


def main():
    # The redundant 3-link's minimum-energy motion naturally tilts the cup (a
    # null-space shortcut); the ALM cancels exactly that drift. The constrained
    # solve needs the smaller step (0.02): at 0.05 the primal update overshoots
    # against the dual ascent and the ALM diverges into a high-energy basin.
    arm = PlanarArm(3, lengths=[1.0, 0.8, 0.6], g=1.0)
    ang = angle_manifold(arm)
    metric = build_arm_metric(arm, manifold=ang, gravity_strength=0.02)
    qs = jnp.array([1.2, -0.6, -0.6])  # cup level (Σθ = 0)
    qg = jnp.array([-0.9, 1.1, -0.2])  # cup level (Σθ = 0)
    c = upright_constraint(arm, ang, target_angle=0.0)
    n_steps = 24

    uncon = AVBDSolver(step_size=0.05, iterations=500).solve(
        metric, qs, qg, n_steps=n_steps, train_mode=False)
    alm = AVBDSolver(step_size=0.02, beta=1.2, iterations=800, tol=1e-4).solve(
        metric, qs, qg, n_steps=n_steps, constraints=[c], train_mode=False)
    pen = _penalty_plan(metric, ang, qs, qg, c, mu_hi=2000.0, n_steps=n_steps)

    v_uncon = float(max_constraint_violation(uncon.xs, [c]))
    v_alm = float(max_constraint_violation(alm.xs, [c]))
    v_pen = float(max_constraint_violation(pen, [c]))
    print("Stage C — upright-cup ALM constraint")
    print(f"  max tilt violation: unconstrained={v_uncon:.4f}  penalty(tuned)={v_pen:.4f}  ALM={v_alm:.2e}")
    print(f"  ALM within tol(1e-2)? {v_alm < 1e-2}   ALM beats penalty? {v_alm < v_pen}")
    print(f"  ALM path energy = {float(alm.energy):.4f}  (unconstrained = {float(uncon.energy):.4f})")

    _figure(arm, uncon.xs, alm.xs, pen, v_uncon, v_alm, v_pen)


def _figure(arm, uncon, alm, pen, v_uncon, v_alm, v_pen):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    s = np.linspace(0, 1, uncon.shape[0])

    # (a) end-effector orientation (cup tilt) along each path.
    ax = axes[0]
    for path, c_, lab in [(uncon, "crimson", "unconstrained"),
                          (pen, "orange", "penalty-only"),
                          (alm, "C0", "ALM (exact)")]:
        ori = np.array(jax.vmap(arm.ee_orientation)(path))
        ax.plot(s, np.rad2deg(ori), "-", color=c_, lw=2, label=lab)
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_title("Cup orientation along the path")
    ax.set_xlabel("path parameter"), ax.set_ylabel("tilt (deg)"), ax.legend()

    # (b) the arm carrying the cup (ALM) at sampled poses.
    ax = axes[1]
    for q in (alm[0], alm[len(alm) // 3], alm[2 * len(alm) // 3], alm[-1]):
        pts = np.array(arm.link_points(q))
        ax.plot(pts[:, 0], pts[:, 1], "-o", lw=1.5, ms=3, alpha=0.7)
        ee = pts[-1]
        ax.plot([ee[0] - 0.15, ee[0] + 0.15], [ee[1], ee[1]], "-", color="k", lw=3)  # level cup
    ax.set_title("ALM motion keeps the cup level"), ax.set_aspect("equal")
    ax.set_xlabel("x"), ax.set_ylabel("y")

    # (c) violation comparison (log scale).
    ax = axes[2]
    bars = ax.bar(["unconstrained", "penalty\n(tuned)", "ALM"],
                  [v_uncon, v_pen, max(v_alm, 1e-6)], color=["crimson", "orange", "C0"])
    ax.set_yscale("log"), ax.set_ylabel("max tilt violation (|sin Δ| or rad)")
    ax.set_title(f"ALM enforces to {v_alm:.1e}\n(penalty plateaus, ill-conditioned)")
    for b, v in zip(bars, [v_uncon, v_pen, v_alm]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1e}", ha="center", va="bottom")

    fig.suptitle("Stage C — exact upright-cup constraint via Augmented Lagrangian", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "stage_c_constraint.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
