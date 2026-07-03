"""The validation ladder as a runnable gate — machinery hardening before the science.

Run ``python -m experiments.arm.validate`` to check each rung and print a table.
The posture mirrors ``experiments/marine``: the experiment ships its own
independent ground-truth cross-checks, not just pretty paths. The L7 rungs gate
the *identifiability structure* of Stage D (the exact-drift gauge), not blind
recovery: L7 asserts path-shape inversion succeeds/fails exactly where the
projective-invariance theorem says it must, L7c that the convex cost-asymmetry
estimator recovers the gravity potential, and L7d that the a-priori diagnostic
predicts both — before any fit.

Each rung returns ``(passed, detail)``. Exit code is nonzero if any rung fails,
so this doubles as a CI-style gate the stage scripts can depend on.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from ham.solvers import AVBDSolver, GaussNewtonGeodesic

from .constraints import max_constraint_violation, upright_constraint
from .evaluate import (
    form_cosine,
    kinetic_cost,
    min_clearance,
    shape_identifiability,
    spray_geodesic,
)
from .fields import PotentialWind, ScaledDistance, VortexWind, star_grad
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
from .planners import AVBDPlanner, EikonalPlanner
from .providers.synthetic import AnalyticDistance, CircleScene, PlanarArm


def _arm():
    return PlanarArm(2, lengths=[1.0, 1.0])


# ---------------------------------------------------------------------------
def l0_flat_torus() -> tuple[bool, str]:
    """Clifford torus: exp∘log round-trip is exact and length is closed-form."""
    T = _arm().manifold
    key = jax.random.PRNGKey(0)
    k0, k1 = jax.random.split(key)
    x, y = T.random_sample(k0), T.random_sample(k1)
    err = float(jnp.max(jnp.abs(T.exp_map(x, T.log_map(x, y)) - y)))
    return err < 1e-5, f"exp(log) round-trip err={err:.2e}"


def l1_robot() -> tuple[bool, str]:
    """Robot sanity: M(q) symmetric-PD and Jacobian matches finite differences."""
    arm = _arm()
    q = jnp.array([0.4, -0.7])
    M = arm.inertia(q)
    sym = float(jnp.max(jnp.abs(M - M.T)))
    pd = float(jnp.min(jnp.linalg.eigvalsh(M)))
    J = arm.jacobian(q)
    # FD step / tolerance scale with working precision: float32 roundoff makes
    # a 1e-5 step pure noise (central-diff error ~ eps/h ≈ 1e-2 there).
    h, jtol = (1e-5, 1e-4) if jax.config.jax_enable_x64 else (1e-3, 1e-3)
    J_fd = jnp.stack(
        [(arm.fk(q.at[i].add(h)) - arm.fk(q.at[i].add(-h))) / (2 * h) for i in range(2)],
        axis=1,
    )
    jerr = float(jnp.max(jnp.abs(J - J_fd)))
    ok = sym < 1e-6 and pd > 1e-6 and jerr < jtol
    return ok, f"M sym={sym:.1e} min-eig={pd:.3f} Jac-FD err={jerr:.1e}"


def l2_adjoint() -> tuple[bool, str]:
    """Implicit adjoint matches finite differences through the metric (Stage-D gate)."""
    arm = _arm()
    ang = angle_manifold(arm)
    q0, q1 = jnp.array([-0.5, 0.4]), jnp.array([0.8, -0.5])
    center = jnp.array([1.4, 0.2])

    class _Smooth:
        def __call__(self, q):
            ee = arm.fk(q)
            return jnp.linalg.norm(ee - center) - 0.3

    base = _Smooth()

    def energy(scale):
        m = build_arm_metric(arm, ScaledDistance(base, scale), manifold=ang, alpha=0.05)
        return AVBDSolver(
            step_size=0.05, iterations=400, energy_tol=1e-12, implicit_diff=True
        ).solve(m, q0, q1, n_steps=12).energy

    s0 = jnp.array(1.0)
    g = float(jax.grad(energy)(s0))
    fd = float((energy(s0 + 1e-4) - energy(s0 - 1e-4)) / 2e-4)
    rel = abs(g - fd) / (abs(fd) + 1e-9)
    return rel < 0.05, f"adjoint={g:.4f} FD={fd:.4f} rel={rel:.3f}"


def l3_eikonal() -> tuple[bool, str]:
    """AVBD geodesic length agrees with the 2-DoF eikonal arrival field."""
    arm = _arm()
    ang = angle_manifold(arm)
    metric = build_arm_metric(arm, manifold=ang)
    q0, q1 = jnp.array([-0.5, 0.4]), jnp.array([0.8, -0.5])
    len_avbd = float(
        kinetic_cost(metric, AVBDSolver(step_size=0.05, iterations=400).solve(
            metric, q0, q1, n_steps=32, train_mode=False).xs)
    )
    extent = (-1.6, 1.6, -1.6, 1.6)
    T = EikonalPlanner(max_iters=500).arrival_field(metric, q0, extent, (91, 91))
    t_goal = float(EikonalPlanner.sample(T, extent, q1))
    rel = abs(t_goal - len_avbd) / len_avbd
    return rel < 0.05, f"eikonal={t_goal:.4f} AVBD={len_avbd:.4f} rel={rel:.3f}"


def l3b_spray() -> tuple[bool, str]:
    """AVBD ≈ spray-shot geodesic ≈ Gauss-Newton on the smooth metric."""
    arm = _arm()
    ang = angle_manifold(arm)
    metric = build_arm_metric(arm, manifold=ang)
    q0, q1 = jnp.array([-0.5, 0.4]), jnp.array([0.8, -0.5])
    la = float(kinetic_cost(metric, AVBDSolver(step_size=0.05, iterations=400).solve(
        metric, q0, q1, n_steps=32, train_mode=False).xs))
    xs = spray_geodesic(metric, q0, q1, n_steps=64, iters=30)
    ls = float(kinetic_cost(metric, xs))
    lg = float(kinetic_cost(metric, GaussNewtonGeodesic(iterations=60).solve(
        metric, q0, q1, n_steps=32, train_mode=False).xs))
    endpt = float(jnp.linalg.norm(xs[-1] - q1))
    ok = endpt < 1e-3 and abs(la - ls) / ls < 0.02 and abs(la - lg) / lg < 0.02
    return ok, f"AVBD={la:.4f} spray={ls:.4f} GN={lg:.4f} shoot-err={endpt:.1e}"


def l4_continuation() -> tuple[bool, str]:
    """Barrier continuation stays collision-free where a cold single-shot diverges.

    α=0.6 is the stiff end of the *honest* barrier range: hard wall, still
    localized. (Pushing α further is self-defeating — the softplus wall's
    hardness is set by its width, so ever-larger α only inflates free space
    until cutting through the wall beats detouring, and even continuation
    tunnels. Clearance is densified, so tunneling cannot hide.)
    """
    arm = _arm()
    ang = angle_manifold(arm)
    qa, qb = jnp.array([-1.2, -0.6]), jnp.array([1.8, -0.6])
    dist = AnalyticDistance(arm, CircleScene([[1.3, 0.9, 0.35]]))

    def make_metric(alpha):
        return build_arm_metric(arm, dist, manifold=ang, alpha=alpha, delta=0.05)

    single = AVBDSolver(step_size=0.05, iterations=1000).solve(
        make_metric(0.6), qa, qb, n_steps=48, train_mode=False)
    cont = AVBDPlanner(step_size=0.05, iterations=600).plan(
        make_metric, qa, qb, n_steps=48, alphas=(0.03, 0.09, 0.24, 0.6))
    cc = float(min_clearance(dist, ang, cont.xs))
    cs = float(min_clearance(dist, ang, single.xs))
    ok = cc > 0.15 and cs < 0.0 and float(cont.energy) < 0.01 * float(single.energy)
    return ok, f"cont clr={cc:.3f} E={float(cont.energy):.3g} | single clr={cs:.3f} E={float(single.energy):.3g}"


def l5_asymmetry() -> tuple[bool, str]:
    """Gravity drift sign (lifting > descending), mild-wind cap, gs=0 symmetric."""
    arm = _arm()
    ang = angle_manifold(arm)
    m = build_arm_metric(arm, manifold=ang, gravity_strength=0.1)
    q = jnp.array([0.4, -0.6])
    up = arm.gravity(q) / jnp.linalg.norm(arm.gravity(q))
    f_up, f_dn = float(m.metric_fn(q, up)), float(m.metric_fn(q, -up))
    _, _, lam = m.zermelo_data(q)
    ok = f_up > f_dn and 0.0 < float(lam) <= 1.0
    return ok, f"F_up={f_up:.3f} F_dn={f_dn:.3f} lam={float(lam):.3f}"


def l6_alm_constraint() -> tuple[bool, str]:
    """ALM holds the cup orientation to tolerance, beyond the unconstrained drift."""
    arm = PlanarArm(3, lengths=[1.0, 0.8, 0.6])
    ang = angle_manifold(arm)
    m = build_arm_metric(arm, manifold=ang, gravity_strength=0.02)
    qs, qg = jnp.array([1.2, -0.6, -0.6]), jnp.array([-0.9, 1.1, -0.2])
    c = upright_constraint(arm, ang, 0.0)
    u = AVBDSolver(step_size=0.05, iterations=500).solve(m, qs, qg, n_steps=24, train_mode=False)
    a = AVBDSolver(step_size=0.02, beta=1.2, iterations=800, tol=1e-4).solve(
        m, qs, qg, n_steps=24, constraints=[c], train_mode=False)
    vu = float(max_constraint_violation(u.xs, [c]))
    va = float(max_constraint_violation(a.xs, [c]))
    return (vu > 0.02 and va < 1e-2 and va < 0.2 * vu), f"uncon={vu:.3f} ALM={va:.1e}"


def _stage_d_regimes(gs: float = 0.15, n: int = 5):
    """Shared Stage-D fixtures: base metric, G / G+R truths and their demos."""
    arm = _arm()
    ang = angle_manifold(arm)
    base = build_arm_metric(arm, manifold=ang)
    vortex = VortexWind(arm, center=(0.2, 0.3), strength=0.12, width=0.7)
    mG = build_arm_metric(arm, manifold=ang, gravity_strength=gs)
    mGR = build_arm_metric(arm, manifold=ang, gravity_strength=gs, wind_field=vortex)
    gen = AVBDSolver(step_size=0.05, iterations=400)
    pairs = jax.random.uniform(jax.random.PRNGKey(0), (n, 2, 2), minval=-1.2, maxval=1.2)
    demosG = [gen.solve(mG, p[0], p[1], n_steps=16, train_mode=False).xs for p in pairs]
    demosGR = [gen.solve(mGR, p[0], p[1], n_steps=16, train_mode=False).xs for p in pairs]
    return arm, ang, base, vortex, mG, mGR, gen, demosG, demosGR


def _potential_r2(arm, U_fn, region, gs: float = 0.15) -> float:
    """R² between a recovered potential and the true gs·U (additive gauge removed)."""
    u_hat = jax.vmap(U_fn)(region)
    u_true = gs * jax.vmap(arm.potential)(region)
    u_hat = u_hat - jnp.mean(u_hat)
    u_true = u_true - jnp.mean(u_true)
    return 1.0 - float(jnp.sum((u_hat - u_true) ** 2) / jnp.sum(u_true**2))


def l7_channel_structure() -> tuple[bool, str]:
    """The observation channel decides recovery, as the gauge theorem predicts.

    Same hypothesis class (a scalar potential), same demos (regime G), two
    losses: the geometry loss L-B — whose only signal is the second-order
    Zermelo leak plus vertex spacing — must recover far less of the potential
    than the timing loss L-T, which parametrization-rigidity makes identifiable.
    """
    arm, ang, _, _, mG, _, _, demosG, _ = _stage_d_regimes()

    def make_metric(field):
        return build_arm_metric(arm, manifold=ang, wind_field=field)

    region = jnp.concatenate(demosG, axis=0)
    costs = [segment_costs(mG, d) for d in demosG]
    f0 = PotentialWind(arm, width=32, depth=2, key=jax.random.PRNGKey(1))
    fb, _ = fit_field(f0, demosG, make_metric, loss_fn=el_residual_loss,
                      steps=600, lr=5e-3, normal_only=True)
    ft, _ = fit_field(f0, demosG, make_metric, loss_fn=cost_matching_loss,
                      steps=600, lr=5e-3, demo_costs=costs)
    r2_b = _potential_r2(arm, fb.potential, region)
    r2_t = _potential_r2(arm, ft.potential, region)
    ok = r2_t > 0.8 and r2_t - r2_b > 0.2
    return ok, f"potential R²: shape={r2_b:+.2f}  timing={r2_t:+.2f}"


def l7c_cost_asymmetry_recovery() -> tuple[bool, str]:
    """Round-trip cost asymmetries recover the gravity potential (convex LSQ, seedless)."""
    arm, _, _, _, mG, _, _, demosG, _ = _stage_d_regimes()
    starts = jnp.concatenate([d[:-1] for d in demosG], axis=0)
    ends = jnp.concatenate([d[1:] for d in demosG], axis=0)
    deltas = jnp.concatenate([segment_asymmetry(mG, d) for d in demosG], axis=0)
    g = jnp.linspace(-1.4, 1.4, 6)
    C1, C2 = jnp.meshgrid(g, g, indexing="ij")
    centers = jnp.stack([C1.ravel(), C2.ravel()], -1)
    U_hat, _ = recover_potential_lsq(starts, ends, deltas, centers, width=0.7)
    r2 = _potential_r2(arm, U_hat, jnp.concatenate(demosG, axis=0))
    return r2 > 0.9, f"potential R²={r2:.4f}"


def l7d_identifiability_diagnostic() -> tuple[bool, str]:
    """The a-priori diagnostic separates the regimes before any fit (sign-of-effect)."""
    _, _, base, _, _, _, gen, demosG, demosGR = _stage_d_regimes()
    sG = shape_identifiability(base, demosG, gen)
    sGR = shape_identifiability(base, demosGR, gen)
    return (sGR > 3.0 * sG and sG < 0.02), f"G={sG:.4f} G+R={sGR:.4f} ratio={sGR / max(sG, 1e-9):.1f}"


def l7e_full_form_recovery() -> tuple[bool, str]:
    """On G+R the asymmetry estimator recovers the entire drift 1-form (convex)."""
    arm, _, _, vortex, _, mGR, _, _, demosGR = _stage_d_regimes()
    gs = 0.15
    starts = jnp.concatenate([d[:-1] for d in demosGR], axis=0)
    ends = jnp.concatenate([d[1:] for d in demosGR], axis=0)
    deltas = jnp.concatenate([segment_asymmetry(mGR, d) for d in demosGR], axis=0)
    g = jnp.linspace(-1.4, 1.4, 6)
    C1, C2 = jnp.meshgrid(g, g, indexing="ij")
    centers = jnp.stack([C1.ravel(), C2.ravel()], -1)
    _, _, b_hat = recover_form_lsq(starts, ends, deltas, centers, width=0.7)
    region = jnp.concatenate(demosGR, axis=0)
    b_true = (gs * jax.vmap(arm.gravity)(region)
              + jax.vmap(lambda q: star_grad(vortex.stream, q))(region))
    cos = form_cosine(jax.vmap(b_hat)(region), b_true)
    return cos > 0.9, f"full-form cosine={cos:.3f}"


LADDER = [
    ("L0  FlatTorus seam", l0_flat_torus),
    ("L1  Robot sanity", l1_robot),
    ("L2  Implicit adjoint", l2_adjoint),
    ("L3  Eikonal cross-check", l3_eikonal),
    ("L3b Spray oracle", l3b_spray),
    ("L4  Continuation", l4_continuation),
    ("L5  Gravity asymmetry", l5_asymmetry),
    ("L6  ALM constraint", l6_alm_constraint),
    ("L7  Channel structure", l7_channel_structure),
    ("L7c Cost-asymmetry recovery", l7c_cost_asymmetry_recovery),
    ("L7d A-priori diagnostic", l7d_identifiability_diagnostic),
    ("L7e Full-form recovery", l7e_full_form_recovery),
]


def main() -> int:
    print(f"Robot-arm validation ladder  (x64={jax.config.jax_enable_x64})\n" + "=" * 72)
    all_ok = True
    for name, fn in LADDER:
        try:
            ok, detail = fn()
        except Exception as exc:  # report any rung failure verbatim
            ok, detail = False, f"ERROR: {type(exc).__name__}: {exc}"
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:24s} {detail}")
    print("=" * 72)
    print("ALL RUNGS PASSED" if all_ok else "SOME RUNGS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
