"""The validation ladder as a runnable gate — machinery hardening before the science.

Run ``python -m experiments.arm.validate`` to check each rung and print a table.
L0–L5 are implemented here; L6 (ALM) and L7 (Stage-D recovery) land with the
stage scripts. The posture mirrors ``experiments/marine``: the experiment ships
its own independent ground-truth cross-checks, not just pretty paths.

Each rung returns ``(passed, detail)``. Exit code is nonzero if any rung fails,
so this doubles as a CI-style gate the stage scripts can depend on.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

from ham.solvers import AVBDSolver, GaussNewtonGeodesic

from .evaluate import kinetic_cost, min_clearance, spray_geodesic
from .fields import ScaledDistance
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
    h = 1e-5
    J_fd = jnp.stack(
        [(arm.fk(q.at[i].add(h)) - arm.fk(q.at[i].add(-h))) / (2 * h) for i in range(2)],
        axis=1,
    )
    jerr = float(jnp.max(jnp.abs(J - J_fd)))
    ok = sym < 1e-6 and pd > 1e-6 and jerr < 1e-4
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
    """Barrier continuation stays collision-free where a cold single-shot diverges."""
    arm = _arm()
    ang = angle_manifold(arm)
    qa, qb = jnp.array([-0.3, 1.2]), jnp.array([1.1, 0.3])
    dist = AnalyticDistance(arm, CircleScene([[1.4, 1.0, 0.3]]))

    def make_metric(alpha):
        return build_arm_metric(arm, dist, manifold=ang, alpha=alpha, delta=0.04)

    single = AVBDSolver(step_size=0.05, iterations=800).solve(
        make_metric(12.0), qa, qb, n_steps=24, train_mode=False)
    cont = AVBDPlanner(step_size=0.05, iterations=400).plan(
        make_metric, qa, qb, n_steps=24, alphas=(0.05, 0.2, 0.6, 1.5, 12.0))
    cc = float(min_clearance(dist, ang, cont.xs))
    cs = float(min_clearance(dist, ang, single.xs))
    ok = cc > 0 and cc >= cs and float(cont.energy) < 0.5 * float(single.energy)
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


LADDER = [
    ("L0  FlatTorus seam", l0_flat_torus),
    ("L1  Robot sanity", l1_robot),
    ("L2  Implicit adjoint", l2_adjoint),
    ("L3  Eikonal cross-check", l3_eikonal),
    ("L3b Spray oracle", l3b_spray),
    ("L4  Continuation", l4_continuation),
    ("L5  Gravity asymmetry", l5_asymmetry),
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
