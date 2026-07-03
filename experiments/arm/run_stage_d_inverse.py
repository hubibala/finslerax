"""Stage D — the exact-drift gauge: what demonstrations can and cannot teach.

The prior Stage D tried to recover the gravity drift from demonstrated *paths*
and found it "weakly identified". That was not a numerical soft spot — it is a
theorem (``spec/exact_drift_gauge_equivalence.md``): the gravity drift's Randers
1-form ``β = gₛ·dU/λ`` is **exact**, and adding an exact form to a cost changes
the value of every path but the choice of path not at all (Finsler projective
invariance = potential-based reward shaping, verbatim). Path shape is
structurally blind to gravity; only the drift's *rotational* part bends paths.
The complementary channel — *cost/timing* — carries exactly the missing piece
(Bucataru–Muzsnay parametrization-rigidity), and the cleanest such observation
is the experiment's own headline: the directed round-trip asymmetry
``cost(γ) − cost(γ⁻¹) = 2∫β = 2gₛ(U(B) − U(A))``.

Stage D therefore measures the *identifiability structure*, on two ground-truth
regimes — **G** (pure gravity, exact) and **G+R** (gravity + solenoidal vortex):

1. **Scaling law.** Point-set deviation of drifted geodesics grows ∝ strength²
   for the exact drift (a second-order Zermelo leak; zero in the additive
   Randers form) but ∝ strength for the rotational drift — the theorem, made
   visible.
2. **A-priori diagnostic.** ``shape_identifiability`` (a discrete projective-
   metrizability test) predicts from the demos alone, before any fit, which
   regime carries shape information.
3. **Channel experiment.** With the *same* potential hypothesis class on G, the
   pure-shape loss L-B (EL residual projected onto the path normal — vertex
   spacing is timing in disguise and is deliberately excluded) has only the
   second-order leak to work with: it overfits few demos and *degrades* as
   demos are added, while the timing loss L-T sits at R² ≈ 1.000 throughout.
4. **Convex recovery.** Round-trip cost asymmetries observe ``∫β`` itself, so a
   linear least-squares recovers the potential on G (seedless, global, high R²)
   and the *entire* drift form on G+R — with the Hodge *split* of the recovered
   form only softly determined by a finite demo graph (harmonic ambiguity),
   reported as such.
5. **De-circularization (L7b).** The eikonal arrival field of the recovered
   metric (a solver independent of AVBD) reproduces a held-out demo's cost.

Run:  python -m experiments.arm.run_stage_d_inverse
Writes experiments/arm/visualizations/stage_d_inverse.png
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ham.solvers import AVBDSolver, ExponentialMap

from .evaluate import (
    form_cosine,
    kinetic_cost,
    polyline_deviation,
    shape_identifiability,
)
from .fields import PotentialWind, VortexWind, star_grad
from .learn import (
    cost_matching_loss,
    el_residual_loss,
    fit_field,
    recover_form_lsq,
    recover_potential_lsq,
    segment_asymmetry,
    segment_costs,
)
from .medium import angle_manifold, build_arm_metric
from .planners import EikonalPlanner
from .providers.synthetic import PlanarArm

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

GS = 0.15                      # gravity strength (regime G)
VORTEX = {"center": (0.2, 0.3), "strength": 0.12, "width": 0.7}  # rotational part (G+R)
N_DEMOS = 12
N_STEPS = 16


def _demos(metric, n, key):
    """Demonstrations = geodesics of the true metric between random endpoint pairs."""
    gen = AVBDSolver(step_size=0.05, iterations=400)
    pairs = jax.random.uniform(key, (n, 2, 2), minval=-1.2, maxval=1.2)
    return [gen.solve(metric, p[0], p[1], n_steps=N_STEPS, train_mode=False).xs for p in pairs]


def _observe(metric, demos):
    """The two observation channels: per-segment costs of each demo run forward/backward."""
    costs = [segment_costs(metric, d) for d in demos]
    asyms = [segment_asymmetry(metric, d) for d in demos]
    return costs, asyms


def main():
    arm = PlanarArm(2, lengths=[1.0, 1.0])
    ang = angle_manifold(arm)
    base = build_arm_metric(arm, manifold=ang)
    vortex = VortexWind(arm, **VORTEX)
    mG = build_arm_metric(arm, manifold=ang, gravity_strength=GS)
    mGR = build_arm_metric(arm, manifold=ang, gravity_strength=GS, wind_field=vortex)

    demosG = _demos(mG, N_DEMOS, jax.random.PRNGKey(0))
    demosGR = _demos(mGR, N_DEMOS, jax.random.PRNGKey(0))
    costsG, asymsG = _observe(mG, demosG)
    regionG = jnp.concatenate(demosG, axis=0)

    print("Stage D — the exact-drift gauge (G: pure gravity, G+R: gravity + vortex)")

    # ------------------------------------------------------------------ 1.
    # Scaling law: exact drift bends point sets at 2nd order, rotational at 1st.
    A0, v0 = jnp.array([-0.9, 0.2]), jnp.array([1.6, 0.8])
    em, em_ref = ExponentialMap(max_steps=64), ExponentialMap(max_steps=96)
    ref, _ = em_ref.trace(base, A0, v0)
    strengths = np.array([0.05, 0.1, 0.2, 0.3])
    dev_g, dev_r, asym_g = [], [], []
    for s in strengths:
        mg = build_arm_metric(arm, manifold=ang, gravity_strength=float(s))
        pg, _ = em.trace(mg, A0, v0)
        dev_g.append(float(polyline_deviation(pg, ref)))
        asym_g.append(float(2.0 * jnp.abs(jnp.sum(segment_asymmetry(mg, pg)))))
        vw = VortexWind(arm, center=VORTEX["center"], strength=float(s) / 4.0,
                        width=VORTEX["width"])
        pv, _ = em.trace(build_arm_metric(arm, manifold=ang, wind_field=vw), A0, v0)
        dev_r.append(float(polyline_deviation(pv, ref)))
    print("  1. scaling (same start/direction, point-set deviation from base geodesic):")
    for s, dg, dr in zip(strengths, dev_g, dev_r):
        print(f"     strength={s:.2f}: gravity dev={dg:.4f} (dev/s²={dg / s**2:.2f})   "
              f"vortex(s/4) dev={dr:.4f}")

    # ------------------------------------------------------------------ 2.
    # A-priori identifiability diagnostic — before any recovery is attempted.
    diag_solver = AVBDSolver(step_size=0.05, iterations=400)
    sG = shape_identifiability(base, demosG, diag_solver)
    sGR = shape_identifiability(base, demosGR, diag_solver)
    print(f"  2. a-priori shape-identifiability: G={sG:.4f}  G+R={sGR:.4f}  "
          f"(ratio {sGR / max(sG, 1e-9):.1f}x — predicts 3. before fitting)")

    # ------------------------------------------------------------------ 3.
    # Channel experiment on G: same hypothesis (potential), two channels.
    def make_metric(field):
        return build_arm_metric(arm, manifold=ang, wind_field=field)

    def potential_r2(U_fn, region):
        """R² between a recovered potential and the true gₛ·U (additive gauge removed)."""
        u_hat = jax.vmap(U_fn)(region)
        u_true = GS * jax.vmap(arm.potential)(region)
        u_hat = u_hat - jnp.mean(u_hat)
        u_true = u_true - jnp.mean(u_true)
        return 1.0 - float(jnp.sum((u_hat - u_true) ** 2) / jnp.sum(u_true**2))

    ks = [3, 6, 9, 12]
    seeds = [1, 2, 3]
    shape_runs = {s: [] for s in seeds}  # the gauge direction: pure seed noise
    timing_curve = []
    for k in ks:
        for s in seeds:
            f0 = PotentialWind(arm, width=32, depth=2, key=jax.random.PRNGKey(s))
            # normal_only: vertex spacing is timing in disguise (constant-F-speed
            # sampling), so the shape channel projects the residual onto the
            # path normal — the honest unparametrized observation model.
            fb, _ = fit_field(f0, demosG[:k], make_metric, loss_fn=el_residual_loss,
                              steps=600, lr=5e-3, normal_only=True)
            shape_runs[s].append(potential_r2(fb.potential, regionG))
        f0 = PotentialWind(arm, width=32, depth=2, key=jax.random.PRNGKey(1))
        ft, _ = fit_field(f0, demosG[:k], make_metric, loss_fn=cost_matching_loss,
                          steps=600, lr=5e-3, demo_costs=costsG[:k])
        timing_curve.append(potential_r2(ft.potential, regionG))
        span = [shape_runs[s][-1] for s in seeds]
        print(f"  3. #demos={k:2d}: potential R²  shape={min(span):+.2f}..{max(span):+.2f} "
              f"(3 seeds)  timing={timing_curve[-1]:+.3f}")
    pot_final = ft  # timing-channel fit on all demos (used for L7b)

    # ------------------------------------------------------------------ 4a.
    # Convex potential recovery from round-trip cost asymmetries (regime G).
    g = jnp.linspace(-1.4, 1.4, 6)
    C1, C2 = jnp.meshgrid(g, g, indexing="ij")
    centers = jnp.stack([C1.ravel(), C2.ravel()], -1)
    starts = jnp.concatenate([d[:-1] for d in demosG], axis=0)
    ends = jnp.concatenate([d[1:] for d in demosG], axis=0)
    deltas = jnp.concatenate(asymsG, axis=0)
    U_hat, _ = recover_potential_lsq(starts, ends, deltas, centers, width=0.7)
    r2 = potential_r2(U_hat, regionG)
    print(f"  4a. cost-asymmetry LSQ on G: potential R² = {r2:.4f} (convex, no seeds)")

    # ------------------------------------------------------------------ 4b.
    # The same estimator on G+R recovers the *entire* drift form; its Hodge
    # split is only softly determined by a finite demo graph — reported as such.
    regionGR = jnp.concatenate(demosGR, axis=0)
    startsR = jnp.concatenate([d[:-1] for d in demosGR], axis=0)
    endsR = jnp.concatenate([d[1:] for d in demosGR], axis=0)
    deltasR = jnp.concatenate([segment_asymmetry(mGR, d) for d in demosGR], axis=0)
    U_hatR, psi_hatR, b_hatR = recover_form_lsq(startsR, endsR, deltasR, centers, width=0.7)
    b_true_full = (GS * jax.vmap(arm.gravity)(regionGR)
                   + jax.vmap(lambda q: star_grad(vortex.stream, q))(regionGR))
    cos_full = form_cosine(jax.vmap(b_hatR)(regionGR), b_true_full)
    r2_ugr = potential_r2(U_hatR, regionGR)
    cos_rot = form_cosine(jax.vmap(lambda q: star_grad(psi_hatR, q))(regionGR),
                          jax.vmap(lambda q: star_grad(vortex.stream, q))(regionGR))
    print(f"  4b. cost-asymmetry LSQ on G+R: full-form cosine = {cos_full:.3f}  "
          f"(soft Hodge split: U R²={r2_ugr:.2f}, rot cos={cos_rot:.2f})")

    # ------------------------------------------------------------------ 5.
    # L7b — de-circularization: eikonal on the timing-recovered metric
    # reproduces a held-out demo's cost (eikonal is independent of AVBD).
    ho = _demos(mG, 1, jax.random.PRNGKey(99))[0]
    learned_metric = make_metric(pot_final)
    eik = EikonalPlanner(max_iters=500)
    extent = (-1.6, 1.6, -1.6, 1.6)
    T = eik.arrival_field(learned_metric, ho[0], extent, (101, 101))
    t_eik = float(eik.sample(T, extent, ho[-1]))
    t_true = float(kinetic_cost(mG, ho))
    print(f"  5. L7b de-circularization: held-out true cost={t_true:.4f}  "
          f"eikonal(learned)={t_eik:.4f}  rel-err={abs(t_eik - t_true) / t_true:.3f}")

    _figure(arm, base, demosG, demosGR, diag_solver, strengths, dev_g, dev_r,
            sG, sGR, ks, shape_runs, timing_curve, vortex, U_hat, r2, b_hatR,
            regionGR, cos_full)


def _figure(arm, base, demosG, demosGR, diag_solver, strengths, dev_g, dev_r,
            sG, sGR, ks, shape_runs, timing_curve, vortex, U_hat, r2, b_hatR,
            regionGR, cos_full):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))

    # (a) the gauge in pictures: G demos trace base geodesics, G+R demos leave them.
    ax = axes[0, 0]
    for i, (dG, dGR) in enumerate(zip(demosG[:4], demosGR[:4])):
        refs = diag_solver.solve(base, dG[0], dG[-1], n_steps=dG.shape[0] - 1,
                                 train_mode=False).xs
        rp, gp, rr = np.array(refs), np.array(dG), np.array(dGR)
        ax.plot(rp[:, 0], rp[:, 1], "-", color="0.75", lw=4, zorder=1,
                label="base geodesic" if i == 0 else None)
        ax.plot(gp[:, 0], gp[:, 1], "-", color="C0", lw=1.6, zorder=3,
                label="demo, gravity only (G)" if i == 0 else None)
        ax.plot(rr[:, 0], rr[:, 1], "--", color="crimson", lw=1.6, zorder=2,
                label="demo, +vortex (G+R)" if i == 0 else None)
    ax.set_title("Exact drift hides in path shape;\nrotational drift does not")
    ax.set_xlabel(r"$\theta_1$"), ax.set_ylabel(r"$\theta_2$"), ax.legend(loc="best", fontsize=8)

    # (b) the scaling law behind (a).
    ax = axes[0, 1]
    ax.loglog(strengths, dev_g, "-o", color="C0", label="gravity (exact drift)")
    ax.loglog(strengths / 4.0, dev_r, "-s", color="crimson", label="vortex (rotational)")
    for expo, x0, y0, c in [(2, strengths, dev_g, "C0"), (1, strengths / 4.0, dev_r, "crimson")]:
        xs = np.array([x0[0], x0[-1]])
        ax.loglog(xs, y0[0] * (xs / x0[0]) ** expo, ":", color=c, alpha=0.6,
                  label=f"slope {expo}")
    ax.set_title("Point-set bending vs drift strength\n(1st order rotational, 2nd order exact)")
    ax.set_xlabel("drift strength"), ax.set_ylabel("max deviation (rad)")
    ax.legend(fontsize=8), ax.grid(True, which="both", alpha=0.25)

    # (c) a-priori diagnostic predicts which regime is shape-identifiable.
    ax = axes[0, 2]
    bars = ax.bar(["G\n(pure gravity)", "G+R\n(+vortex)"], [sG, sGR],
                  color=["C0", "crimson"])
    for b, v in zip(bars, [sG, sGR]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylabel("shape-identifiability score")
    ax.set_title("A-priori diagnostic (no fit needed):\ndemos vs base geodesics, point-set")

    # (d) the channel experiment: same hypothesis, two observation channels.
    ax = axes[1, 0]
    for i, run in enumerate(shape_runs.values()):
        ax.plot(ks, run, "-o", color="C0", alpha=0.55, ms=4,
                label="L-B (path shape), 3 seeds" if i == 0 else None)
    ax.plot(ks, timing_curve, "-s", color="green", lw=2.5,
            label="L-T (segment costs)")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylim(-1.2, 1.1)
    ax.set_title("Potential recovery vs #demos (regime G)\n"
                 "shape overfits the 2nd-order leak — more data makes it worse")
    ax.set_xlabel("# demonstrations"), ax.set_ylabel("potential R²")
    ax.legend(fontsize=8, loc="lower left")

    # (e) convex potential recovery from round-trip asymmetries.
    ax = axes[1, 1]
    g = np.linspace(-1.3, 1.3, 120)
    G1, G2 = np.meshgrid(g, g, indexing="ij")
    pts = jnp.stack([G1.ravel(), G2.ravel()], -1)
    Ut = np.array(GS * jax.vmap(arm.potential)(pts)).reshape(G1.shape)
    Uh = np.array(jax.vmap(U_hat)(pts)).reshape(G1.shape)
    cf = ax.contourf(G1, G2, Ut - Ut.mean(), levels=18, cmap="coolwarm")
    ax.contour(G1, G2, Uh - Uh.mean(), levels=12, colors="k", linewidths=0.9)
    for d in demosG:
        dp = np.array(d)
        ax.plot(dp[:, 0], dp[:, 1], "-", color="0.35", lw=0.7, alpha=0.7)
    ax.set_title(f"Potential from cost asymmetry (R²={r2:.3f})\nfill: true gₛU · lines: recovered")
    ax.set_xlabel(r"$\theta_1$"), ax.set_ylabel(r"$\theta_2$")
    fig.colorbar(cf, ax=ax, shrink=0.8)

    # (f) the estimator generalizes: the full drift form on G+R, where observed.
    ax = axes[1, 2]
    qs = regionGR[::4]
    bt = np.array(GS * jax.vmap(arm.gravity)(qs)
                  + jax.vmap(lambda q: star_grad(vortex.stream, q))(qs))
    bl = np.array(jax.vmap(b_hatR)(qs))
    P = np.array(qs)
    ax.quiver(P[:, 0], P[:, 1], bt[:, 0], bt[:, 1], color="0.15", alpha=0.9,
              width=0.008, label="true drift form β")
    ax.quiver(P[:, 0], P[:, 1], bl[:, 0], bl[:, 1], color="crimson", alpha=0.9,
              width=0.0035, label="recovered from asymmetry")
    ax.set_title(f"G+R: full drift form from round-trip costs\n"
                 f"(cosine {cos_full:.3f} at demo support, convex LSQ)")
    ax.set_xlabel(r"$\theta_1$"), ax.set_ylabel(r"$\theta_2$"), ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Stage D — the exact-drift gauge: path shape is blind to the "
                 "gravity potential; cost/timing recovers it exactly", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "stage_d_inverse.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
