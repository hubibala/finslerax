"""Validation suite for the robot-arm geodesic experiment (experiments/arm).

Run: ``JAX_PLATFORMS=cpu pytest tests/test_arm.py`` (and again with
``JAX_ENABLE_X64=1``). Mirrors the validation ladder in
``spec/robot_arm_geodesic_PLAN.md`` §5 — one test per rung — so the science
stages run on a proven machine. This file grows a rung per chunk.

Rungs covered so far:
    L1  — Robot sanity: FK, M(q) symmetric-PD, Jacobian/gravity vs finite-diff.
    L2  — Implicit-adjoint gradcheck (the Stage-D gate): AVBD block-tridiag
          adjoint vs finite differences through the metric.
    L3  — Eikonal cross-check: AVBD geodesic length vs the grid arrival field.
    L3b — Spray oracle: AVBD ≈ ExponentialMap shooting ≈ Gauss-Newton on the
          smooth metric (four independent geodesic derivations agree).
    L4  — Continuation: a stiff barrier annealed with warm-starts converges
          collision-free where a cold single-shot diverges.
    L5  — Gravity asymmetry, mild-wind cap, drift sign.
    Provider seam: the synthetic providers structurally satisfy the protocols.
"""

import pathlib
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from _precision import tol, x64_enabled

from experiments.arm import (
    AnalyticDistance,
    CircleScene,
    GroundTruthDemos,
    PlanarArm,
)
from experiments.arm.constraints import (
    max_constraint_violation,
    upright_constraint,
)
from experiments.arm.evaluate import kinetic_cost, min_clearance, spray_geodesic
from experiments.arm.fields import MLPDistance, ScaledDistance
from experiments.arm.interfaces import DemoSource, DistanceField, Robot, Scene
from experiments.arm.medium import angle_manifold, build_arm_metric
from experiments.arm.planners import AVBDPlanner, EikonalPlanner
from ham.geometry.manifolds import EuclideanSpace, FlatTorus
from ham.geometry.zoo import Randers
from ham.solvers import AVBDSolver, GaussNewtonGeodesic

jax.config.update("jax_platform_name", "cpu")


# ===========================================================================
# L1 — Robot sanity (the metric is robot-driven, so a bad M(q) corrupts all)
# ===========================================================================
def test_l1_fk_known_poses():
    """Forward kinematics matches hand-computed poses."""
    atol, rtol = tol(atol32=1e-5, atol64=1e-9)
    arm = PlanarArm(3, lengths=[1.0, 1.0, 1.0])
    # Straight along +x.
    np.testing.assert_allclose(arm.fk(jnp.zeros(3)), [3.0, 0.0], atol=atol, rtol=rtol)
    # Base joint vertical -> straight up.
    np.testing.assert_allclose(
        arm.fk(jnp.array([jnp.pi / 2, 0.0, 0.0])), [0.0, 3.0], atol=atol, rtol=rtol
    )
    # An L-shape: second joint bends +90 deg.
    lp = arm.link_points(jnp.array([0.0, jnp.pi / 2, 0.0]))
    np.testing.assert_allclose(lp[-1], [1.0, 2.0], atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        arm.ee_orientation(jnp.array([0.3, 0.2, -0.1])), 0.4, atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("dof", [2, 3, 5])
def test_l1_inertia_symmetric_pd(dof):
    """The mass matrix is symmetric positive-definite at every configuration."""
    atol, _ = tol(atol32=1e-5, atol64=1e-10)
    arm = PlanarArm(dof)
    key = jax.random.PRNGKey(dof)
    for _ in range(5):
        key, sk = jax.random.split(key)
        q = jax.random.uniform(sk, (dof,), minval=-jnp.pi, maxval=jnp.pi)
        M = arm.inertia(q)
        np.testing.assert_allclose(M, M.T, atol=atol)
        assert float(jnp.min(jnp.linalg.eigvalsh(M))) > 1e-7


def test_l1_inertia_one_link_closed_form():
    """A 1-link arm has constant inertia ``m L^2``."""
    atol, rtol = tol(atol32=1e-5, atol64=1e-9)
    arm = PlanarArm(1, lengths=[2.0], masses=[3.0])
    M = arm.inertia(jnp.array([0.9]))
    np.testing.assert_allclose(M, [[12.0]], atol=atol, rtol=rtol)


# Finite-difference step / tolerance scale with precision: a float32 central
# difference is roundoff-limited (~eps_machine/h), so it needs a coarser step
# and a looser tolerance than float64.
_FD_STEP = 1e-6 if x64_enabled() else 1e-3
_FD_ATOL = 1e-7 if x64_enabled() else 5e-3


def test_l1_jacobian_vs_finite_diff():
    """The end-effector Jacobian matches a central finite difference of FK."""
    arm = PlanarArm(4)
    q = jnp.array([0.4, -0.7, 1.1, 0.2])
    J = arm.jacobian(q)
    J_fd = np.zeros_like(np.array(J))
    for j in range(arm.dof):
        dq = jnp.zeros(arm.dof).at[j].set(_FD_STEP)
        J_fd[:, j] = np.array(arm.fk(q + dq) - arm.fk(q - dq)) / (2 * _FD_STEP)
    np.testing.assert_allclose(J, J_fd, atol=_FD_ATOL)


def test_l1_gravity_vs_finite_diff():
    """The gravity vector matches a finite difference of the potential, and points uphill."""
    arm = PlanarArm(3, g=2.0)
    q = jnp.array([0.4, -0.7, 1.1])
    g_an = arm.gravity(q)
    g_fd = np.array(
        [
            float(
                arm.potential(q.at[j].add(_FD_STEP))
                - arm.potential(q.at[j].add(-_FD_STEP))
            )
            / (2 * _FD_STEP)
            for j in range(arm.dof)
        ]
    )
    np.testing.assert_allclose(g_an, g_fd, atol=_FD_ATOL)
    # From the horizontal pose, raising the base joint lifts all masses.
    assert float(arm.gravity(jnp.zeros(3))[0]) > 0.0


def test_l1_distance_field_sign_and_grad():
    """C-space clearance is negative in collision, positive when free, differentiable."""
    arm = PlanarArm(3)
    scene = CircleScene([[1.5, 0.0, 0.4]])  # circle on the straight-out arm
    dist = AnalyticDistance(arm, scene)
    assert float(dist(jnp.zeros(3))) < 0.0  # straight arm pierces the circle
    assert float(dist(jnp.array([jnp.pi / 2, 0.0, 0.0]))) > 0.0  # folded up, clear
    grad = jax.grad(dist)(jnp.array([0.1, 0.1, 0.1]))
    assert grad.shape == (3,) and bool(jnp.all(jnp.isfinite(grad)))


# ===========================================================================
# Metric correctness anchors — the layered cost matches HAM's Randers, and the
# Clifford and intrinsic-angle representations agree.
# ===========================================================================
def test_metric_matches_randers():
    """ArmMetric (angle rep, no barrier) equals HAM's Randers for the same (H, W)."""
    atol, rtol = tol(atol32=1e-5, atol64=1e-9)
    arm = PlanarArm(2)
    ang = angle_manifold(arm)
    metric = build_arm_metric(arm, manifold=ang, gravity_strength=0.1)
    q = jnp.array([0.4, -0.6])
    v = jnp.array([0.7, 0.3])
    H = arm.inertia(q)
    W = -0.1 * jnp.linalg.solve(H, arm.gravity(q))  # mobility-raised gravity force
    ref = Randers(
        EuclideanSpace(2), lambda z, H=H: H, lambda z, W=W: W, wind_mode="soft"
    )
    np.testing.assert_allclose(
        metric.metric_fn(q, v), ref.metric_fn(q, v), atol=atol, rtol=rtol
    )


def test_gravity_form_is_exact():
    """The gravity drift's Randers 1-form is exactly gₛ·dU (the theorem's premise).

    ``β = -(H W)/λ`` with ``W = -gₛ·M⁻¹∇U`` gives ``λβ = gₛ∇U`` — the mobility
    raising in :mod:`experiments.arm.medium` is what makes the drift an *exact*
    form, hence a pure gauge of path shape (projective invariance).
    """
    atol, rtol = tol(atol32=1e-5, atol64=1e-9)
    arm = PlanarArm(2)
    ang = angle_manifold(arm)
    gs = 0.15
    metric = build_arm_metric(arm, manifold=ang, gravity_strength=gs)
    for q in [jnp.array([0.4, -0.7]), jnp.array([-1.0, 0.3]), jnp.array([0.9, 1.1])]:
        H, W, _lam = metric.zermelo_data(q)
        np.testing.assert_allclose(
            -(H @ W), gs * arm.gravity(q), atol=atol, rtol=max(rtol, 1e-4)
        )


def test_roundtrip_asymmetry_measures_potential():
    """cost(γ) − cost(γ⁻¹) = 2gₛ(U(B) − U(A)) — the note's clean estimator, on any path."""
    from experiments.arm.learn import segment_asymmetry

    arm = PlanarArm(2, g=1.0)
    ang = angle_manifold(arm)
    gs = 0.1
    metric = build_arm_metric(arm, manifold=ang, gravity_strength=gs)
    key = jax.random.PRNGKey(3)
    A, B = jax.random.uniform(key, (2, 2), minval=-1.0, maxval=1.0)
    # An arbitrary (non-geodesic!) wiggly path — the identity is path-independent.
    t = jnp.linspace(0.0, 1.0, 33)[:, None]
    path = A + t * (B - A) + 0.2 * jnp.sin(jnp.pi * t) * jnp.array([1.0, -1.0])
    asym = 2.0 * float(jnp.sum(segment_asymmetry(metric, path)))
    dU = 2.0 * gs * float(arm.potential(B) - arm.potential(A))
    np.testing.assert_allclose(asym, dU, rtol=0.05)


def test_projective_invariance_scaling():
    """Point-set bending: second order for the exact gravity drift, first order for a vortex.

    Traces the spray from the same start/direction under drifted and base
    metrics; the exact drift's deviation quadruples when gₛ doubles (2nd order,
    projective invisibility at 1st order), and a matched-strength vortex bends
    the path more than an order of magnitude harder.
    """
    from experiments.arm.evaluate import polyline_deviation
    from experiments.arm.fields import VortexWind
    from ham.solvers import ExponentialMap

    arm = PlanarArm(2)
    ang = angle_manifold(arm)
    A, v0 = jnp.array([-0.9, 0.2]), jnp.array([1.6, 0.8])
    base = build_arm_metric(arm, manifold=ang)
    ref, _ = ExponentialMap(max_steps=96).trace(base, A, v0)
    em = ExponentialMap(max_steps=64)

    def dev(metric):
        pg, _ = em.trace(metric, A, v0)
        return float(polyline_deviation(pg, ref))

    d1 = dev(build_arm_metric(arm, manifold=ang, gravity_strength=0.1))
    d2 = dev(build_arm_metric(arm, manifold=ang, gravity_strength=0.2))
    assert 3.0 < d2 / d1 < 5.5  # doubling gs quadruples the bending (slope 2)
    vw = VortexWind(arm, center=(0.2, 0.3), strength=0.1, width=0.7)
    dv = dev(build_arm_metric(arm, manifold=ang, wind_field=vw))
    assert dv > 10.0 * d1  # first-order rotational bending dominates


def test_clifford_and_angle_representations_agree():
    """The lifted Clifford metric equals the intrinsic-angle metric on lifted velocities."""
    atol, rtol = tol(atol32=1e-5, atol64=1e-9)
    arm = PlanarArm(3)
    ang = angle_manifold(arm)
    # gs=0.03 keeps the 3-link comfortably in the mild-wind regime (λ≈0.4); a
    # larger gs sits near the causal boundary where float32 amplifies round-off.
    m_ang = build_arm_metric(arm, manifold=ang, gravity_strength=0.03)
    m_cliff = build_arm_metric(arm, gravity_strength=0.03)  # default: FlatTorus
    T = arm.manifold
    q = jnp.array([0.4, -0.6, 0.9])
    v = jnp.array([0.5, 0.2, -0.3])
    x = T.embed_angles(q)
    v_amb = T.tangent_frame(x) @ v
    np.testing.assert_allclose(
        m_cliff.metric_fn(x, v_amb), m_ang.metric_fn(q, v), atol=atol, rtol=rtol
    )


# ===========================================================================
# L5 — gravity asymmetry (the headline novelty); mild-wind cap; drift sign.
# ===========================================================================
def test_l5_drift_sign_and_mild_wind():
    """Lifting costs more than descending, the cap holds (0 < λ ≤ 1), gs=0 is symmetric."""
    arm = PlanarArm(2)
    ang = angle_manifold(arm)
    metric = build_arm_metric(arm, manifold=ang, gravity_strength=0.1)
    key = jax.random.PRNGKey(0)
    for _ in range(6):
        key, sk = jax.random.split(key)
        q = jax.random.uniform(sk, (2,), minval=-jnp.pi, maxval=jnp.pi)
        grad = arm.gravity(q)
        if float(jnp.linalg.norm(grad)) < 1e-3:
            continue
        up = grad / jnp.linalg.norm(grad)  # uphill direction
        assert float(metric.metric_fn(q, up)) > float(metric.metric_fn(q, -up))
        _, _, lam = metric.zermelo_data(q)
        assert 0.0 < float(lam) <= 1.0 + 1e-9

    # Removing gravity makes the metric symmetric.
    sym = build_arm_metric(arm, manifold=ang, gravity_strength=0.0)
    q = jnp.array([0.5, -0.7])
    up = arm.gravity(q) / jnp.linalg.norm(arm.gravity(q))
    np.testing.assert_allclose(
        sym.metric_fn(q, up), sym.metric_fn(q, -up), atol=1e-6
    )


def test_l5_path_cost_asymmetric():
    """A→B (lifting) integrates to a higher cost than B→A (descending)."""
    arm = PlanarArm(2, g=1.0)
    ang = angle_manifold(arm)
    metric = build_arm_metric(arm, manifold=ang, gravity_strength=0.12)
    low = jnp.array([-jnp.pi / 2, 0.0])  # arm hanging down (low potential)
    high = jnp.array([jnp.pi / 2, 0.0])  # arm raised (high potential)
    assert float(arm.potential(high)) > float(arm.potential(low))
    path = jnp.linspace(low, high, 40)
    up_cost = float(metric.arc_length(path))
    down_cost = float(metric.arc_length(path[::-1]))
    assert up_cost > down_cost, (up_cost, down_cost)


def test_upright_constraint_residual():
    """The upright constraint is zero at the target orientation, nonzero off it, wrap-safe."""
    arm = PlanarArm(3)
    T = arm.manifold
    q = jnp.array([0.3, 0.4, -0.2])  # ee orientation φ = 0.5
    phi = float(arm.ee_orientation(q))
    c = upright_constraint(arm, T, target_angle=phi)
    assert abs(float(c(T.embed_angles(q)))) < 1e-6
    off = q.at[2].add(0.3)
    assert abs(float(c(T.embed_angles(off)))) > 1e-2
    assert bool(jnp.all(jnp.isfinite(jax.grad(c)(T.embed_angles(off)))))
    # Adding 2π to a joint is the same physical config -> same residual (seam-safe).
    wrapped = q.at[0].add(2 * jnp.pi)
    np.testing.assert_allclose(
        c(T.embed_angles(wrapped)), c(T.embed_angles(q)), atol=1e-6
    )
    viol = max_constraint_violation(jnp.stack([T.embed_angles(q)] * 3), [c])
    assert float(viol) < 1e-6


def test_l5_barrier_inflates_near_obstacles():
    """The conformal barrier raises cost near obstacles and blows up in collision."""
    arm = PlanarArm(2)
    ang = angle_manifold(arm)
    scene = CircleScene([[1.0, 0.5, 0.3]])
    dist = AnalyticDistance(arm, scene)
    base = build_arm_metric(arm, manifold=ang, gravity_strength=0.1)
    warped = build_arm_metric(
        arm, dist, manifold=ang, gravity_strength=0.1, alpha=0.05, delta=0.05
    )
    v = jnp.array([0.5, 0.3])
    q_collide = jnp.array([0.5, 0.2])  # arm intersects the circle
    q_free = jnp.array([-1.2, -1.0])
    assert float(dist(q_collide)) < 0.0
    infl_collide = float(warped.metric_fn(q_collide, v) / base.metric_fn(q_collide, v))
    infl_free = float(warped.metric_fn(q_free, v) / base.metric_fn(q_free, v))
    assert infl_collide > 100.0  # barrier wall
    assert 1.0 < infl_free < infl_collide  # mild, monotone in clearance


# ===========================================================================
# L2 — implicit-adjoint gradcheck (THE Stage-D gate: a wrong adjoint silently
# poisons inverse learning).
# ===========================================================================
class _SmoothEEDist(eqx.Module):
    """Smooth end-effector-to-circle clearance (no min-over-segments kinks).

    A clean, differentiable field so the finite-difference gradcheck isn't
    confounded by the subgradient kinks of the realistic min-based field.
    """

    arm: PlanarArm
    cx: float = eqx.field(static=True)
    cy: float = eqx.field(static=True)
    r: float = eqx.field(static=True)

    def __call__(self, q):
        ee = self.arm.fk(q)
        return jnp.sqrt((ee[0] - self.cx) ** 2 + (ee[1] - self.cy) ** 2) - self.r


def test_l2_implicit_adjoint_gradcheck():
    """The block-tridiagonal adjoint matches finite differences through the metric."""
    arm = PlanarArm(2)
    ang = angle_manifold(arm)
    q0, q1 = jnp.array([-0.5, 0.4]), jnp.array([0.8, -0.5])
    base = _SmoothEEDist(arm, cx=1.4, cy=0.2, r=0.3)

    def path_energy(scale):
        metric = build_arm_metric(
            arm, ScaledDistance(base, scale), manifold=ang, alpha=0.05, delta=0.05
        )
        solver = AVBDSolver(
            step_size=0.05, iterations=400, energy_tol=1e-12, implicit_diff=True
        )
        return solver.solve(metric, q0, q1, n_steps=12).energy

    s0 = jnp.array(1.0)
    grad_adj = jax.grad(path_energy)(s0)
    fd_step = 1e-4 if x64_enabled() else 1e-3
    fd = (path_energy(s0 + fd_step) - path_energy(s0 - fd_step)) / (2 * fd_step)
    rel = abs(float(grad_adj) - float(fd)) / (abs(float(fd)) + 1e-9)
    assert rel < (0.02 if x64_enabled() else 0.10), (float(grad_adj), float(fd), rel)


def test_l2_adjoint_finite_through_learnable_fields():
    """Adjoint grads are finite (and nonzero) through the realistic min-field and an MLP field."""
    arm = PlanarArm(2)
    ang = angle_manifold(arm)
    q0, q1 = jnp.array([-0.5, 0.4]), jnp.array([0.8, -0.5])

    real = AnalyticDistance(arm, CircleScene([[1.4, 0.2, 0.3]]))

    def energy_scale(scale):
        metric = build_arm_metric(arm, ScaledDistance(real, scale), manifold=ang, alpha=0.05)
        return AVBDSolver(step_size=0.05, iterations=250, implicit_diff=True).solve(
            metric, q0, q1, n_steps=10
        ).energy

    assert bool(jnp.isfinite(jax.grad(energy_scale)(jnp.array(1.0))))

    mlp = MLPDistance(2, width=16, depth=2, key=jax.random.PRNGKey(0))

    def energy_mlp(field):
        metric = build_arm_metric(arm, field, manifold=ang, alpha=0.02)
        return AVBDSolver(step_size=0.05, iterations=200, implicit_diff=True).solve(
            metric, q0, q1, n_steps=10
        ).energy

    grads = eqx.filter_grad(energy_mlp)(mlp)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_array(g)]
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(bool(jnp.any(g != 0)) for g in leaves)


# ===========================================================================
# L3 / L3b — independent geodesic cross-checks (the validation triangle).
# ===========================================================================
def test_l3_eikonal_cross_check():
    """AVBD geodesic length agrees with the 2-DoF eikonal arrival field."""
    arm = PlanarArm(2)
    ang = angle_manifold(arm)
    metric = build_arm_metric(arm, manifold=ang, gravity_strength=0.0)
    q0, q1 = jnp.array([-0.5, 0.4]), jnp.array([0.8, -0.5])

    avbd = AVBDSolver(step_size=0.05, iterations=400).solve(
        metric, q0, q1, n_steps=32, train_mode=False
    )
    len_avbd = float(kinetic_cost(metric, avbd.xs))

    eik = EikonalPlanner(max_iters=500, tol=1e-6)
    extent = (-1.6, 1.6, -1.6, 1.6)
    T = eik.arrival_field(metric, q0, extent, (91, 91))
    t_goal = float(eik.sample(T, extent, q1))
    assert abs(t_goal - len_avbd) / len_avbd < 0.05, (t_goal, len_avbd)


def test_l3b_spray_oracle_triangle():
    """AVBD ≈ spray-shot geodesic ≈ Gauss-Newton on the smooth metric."""
    arm = PlanarArm(2)
    ang = angle_manifold(arm)
    metric = build_arm_metric(arm, manifold=ang, gravity_strength=0.0)
    q0, q1 = jnp.array([-0.5, 0.4]), jnp.array([0.8, -0.5])

    len_avbd = float(
        kinetic_cost(metric, AVBDSolver(step_size=0.05, iterations=400).solve(
            metric, q0, q1, n_steps=32, train_mode=False).xs)
    )
    xs_spray = spray_geodesic(metric, q0, q1, n_steps=64, iters=30)
    len_spray = float(kinetic_cost(metric, xs_spray))
    len_gn = float(
        kinetic_cost(metric, GaussNewtonGeodesic(iterations=60).solve(
            metric, q0, q1, n_steps=32, train_mode=False).xs)
    )

    # The spray actually reaches the goal (shooting converged).
    assert float(jnp.linalg.norm(xs_spray[-1] - q1)) < 1e-3
    assert abs(len_avbd - len_spray) / len_spray < 0.02
    assert abs(len_avbd - len_gn) / len_gn < 0.02


# ===========================================================================
# L4 — barrier continuation converges where a cold single-shot diverges.
# ===========================================================================
def test_l4_continuation_beats_single_shot():
    """Annealing a stiff barrier (warm-started) stays collision-free; a cold solve diverges.

    α=0.6 is the stiff end of the honest barrier range: hard wall, still
    localized. Beyond α≈1 the barrier defeats itself — the softplus wall's
    hardness is fixed by its width, so larger α only inflates free space until
    cutting through the wall beats detouring and even continuation tunnels.
    Clearance is densified, so a tunneling segment cannot pass as clear.
    """
    arm = PlanarArm(2)
    ang = angle_manifold(arm)
    qa, qb = jnp.array([-1.2, -0.6]), jnp.array([1.8, -0.6])
    dist = AnalyticDistance(arm, CircleScene([[1.3, 0.9, 0.35]]))
    # Endpoints are clear, but the straight C-space line dips into collision.
    assert float(dist(qa)) > 0 and float(dist(qb)) > 0
    assert float(min_clearance(dist, ang, jnp.linspace(qa, qb, 49))) < 0

    def make_metric(alpha):
        return build_arm_metric(arm, dist, manifold=ang, alpha=alpha, delta=0.05)

    single = AVBDSolver(step_size=0.05, iterations=1000).solve(
        make_metric(0.6), qa, qb, n_steps=48, train_mode=False
    )
    cont = AVBDPlanner(step_size=0.05, iterations=600).plan(
        make_metric, qa, qb, n_steps=48, alphas=(0.03, 0.09, 0.24, 0.6)
    )
    clr_single = float(min_clearance(dist, ang, single.xs))
    clr_cont = float(min_clearance(dist, ang, cont.xs))
    assert clr_cont > 0.15          # continuation: genuinely clear (densified)
    assert clr_single < 0.0         # cold single-shot dives through the wall
    assert float(cont.energy) < 0.01 * float(single.energy)


def test_3dof_no_tunneling():
    """The 3-DoF plan is clear on the *densified* path — the tunneling regression.

    At 24 segments this task tunnels: the barrier repels vertices until one
    long chord spans the obstacle, and the midpoint-rule energy never sees it
    (vertex clearance even reports +0.6). 48 segments keep every segment well
    below the obstacle body thickness.
    """
    arm3 = PlanarArm(3, lengths=[0.9, 0.8, 0.7])
    ang3 = angle_manifold(arm3)
    dist3 = AnalyticDistance(arm3, CircleScene([[1.6, 0.9, 0.3]]))
    qa3, qb3 = jnp.array([-0.9, -0.3, -0.3]), jnp.array([1.3, -0.4, 0.2])
    assert float(min_clearance(dist3, ang3, jnp.linspace(qa3, qb3, 49))) < -0.2

    geo3 = AVBDPlanner(step_size=0.02, iterations=1200, resample_between=True).plan(
        lambda a: build_arm_metric(arm3, dist3, manifold=ang3, alpha=a, delta=0.04),
        qa3, qb3, n_steps=48, alphas=(0.02, 0.05, 0.12, 0.3)).xs
    assert float(min_clearance(dist3, ang3, geo3)) > 0.2
    seg = jnp.linalg.norm(geo3[1:] - geo3[:-1], axis=-1)
    assert float(seg.max()) < 0.6  # no chord long enough to straddle the body


# ===========================================================================
# L6 — exact ALM equality constraint (the upright cup).
# ===========================================================================
def test_l6_alm_constraint_enforced():
    """The ALM holds the cup orientation to tolerance, far beyond the unconstrained drift."""
    arm = PlanarArm(3, lengths=[1.0, 0.8, 0.6])
    ang = angle_manifold(arm)
    metric = build_arm_metric(arm, manifold=ang, gravity_strength=0.02)
    qs, qg = jnp.array([1.2, -0.6, -0.6]), jnp.array([-0.9, 1.1, -0.2])
    c = upright_constraint(arm, ang, target_angle=0.0)

    uncon = AVBDSolver(step_size=0.05, iterations=500).solve(
        metric, qs, qg, n_steps=24, train_mode=False)
    # step 0.02: the constrained primal update overshoots the dual ascent at 0.05.
    alm = AVBDSolver(step_size=0.02, beta=1.2, iterations=800, tol=1e-4).solve(
        metric, qs, qg, n_steps=24, constraints=[c], train_mode=False)
    v_uncon = float(max_constraint_violation(uncon.xs, [c]))
    v_alm = float(max_constraint_violation(alm.xs, [c]))
    assert v_uncon > 0.02       # the unconstrained motion tilts the cup
    assert v_alm < 1e-2         # the ALM enforces upright to tolerance
    assert v_alm < 0.2 * v_uncon  # ... a large reduction


# ===========================================================================
# Stage A avoidance intent — the obstacle constrains the optimum, verifiably.
# ===========================================================================
def test_stage_a_avoidance_is_the_global_optimum():
    """The avoidance detour matches the eikonal characteristic on the barrier metric.

    Three claims: (1) the straight motion collides deeply mid-path; (2) the
    AVBD geodesic clears with its *tightest* point mid-swing (the barrier
    actively shapes the optimum); (3) the eikonal arrival field on the same
    barrier metric — a global method, no initialization — agrees with the AVBD
    cost to grid accuracy, so the detour is the optimum, not a solver accident.
    """
    arm = PlanarArm(2, lengths=[1.0, 1.0])
    ang = angle_manifold(arm)
    qa, qb = jnp.array([-1.2, -0.6]), jnp.array([1.8, -0.6])
    dist = AnalyticDistance(arm, CircleScene([[1.3, 0.9, 0.35]]))

    def make_metric(alpha):
        return build_arm_metric(arm, dist, manifold=ang, alpha=alpha, delta=0.05)

    straight = jnp.linspace(qa, qb, 25)
    d_str = jax.vmap(dist)(straight)
    assert float(d_str.min()) < -0.2 and 8 <= int(jnp.argmin(d_str)) <= 16

    geo = AVBDPlanner(step_size=0.05, iterations=600).plan(
        make_metric, qa, qb, n_steps=24, alphas=(0.01, 0.03, 0.06, 0.15)).xs
    d_geo = jax.vmap(dist)(geo)
    assert float(min_clearance(dist, ang, geo)) > 0.15  # densified — no tunneling
    assert 4 <= int(jnp.argmin(d_geo)) <= 20  # tightest point mid-swing

    m_obs = make_metric(0.15)
    extent = (-2.8, 2.8, -2.8, 2.8)
    eik = EikonalPlanner(max_iters=800)
    T = eik.arrival_field(m_obs, qa, extent, (121, 121))
    t_goal = float(eik.sample(T, extent, qb))
    len_avbd = float(m_obs.arc_length(geo))
    assert abs(len_avbd - t_goal) / t_goal < 0.06  # global agreement (grid h≈0.05)


# ===========================================================================
# L7 — the exact-drift gauge / identifiability structure (Stage D).
#
# The gravity drift's β = gₛ·dU/λ is exact, so it is a *gauge* of path shape
# (projective invariance = potential-based reward shaping). The tests gate the
# structure, not blind recovery: the point-set diagnostic separates the pure-
# gravity regime G from the +vortex regime G+R before any fit (L7d); the
# timing channel recovers the potential where geometry cannot (L7); and the
# convex round-trip-asymmetry estimator recovers gₛ·U on G (L7c) and the whole
# drift form on G+R (L7e). See spec/exact_drift_gauge_equivalence.md.
# ===========================================================================
def _stage_d_fixtures(n=5, gs=0.15):
    from experiments.arm.fields import VortexWind

    arm = PlanarArm(2, lengths=[1.0, 1.0])
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


def _r2_vs_true_potential(arm, U_fn, region, gs=0.15):
    u_hat = jax.vmap(U_fn)(region)
    u_true = gs * jax.vmap(arm.potential)(region)
    u_hat = u_hat - jnp.mean(u_hat)
    u_true = u_true - jnp.mean(u_true)
    return 1.0 - float(jnp.sum((u_hat - u_true) ** 2) / jnp.sum(u_true**2))


def test_hodge_split_separates_known_channels():
    """The region LSQ Hodge split recovers the components of a known mixed form."""
    from experiments.arm.evaluate import form_cosine, hodge_split_form

    key = jax.random.PRNGKey(0)
    qs = jax.random.uniform(key, (160, 2), minval=-1.2, maxval=1.2)

    def U(q):  # smooth non-linear potential
        return jnp.sin(q[0]) * jnp.cos(0.7 * q[1])

    def psi(q):  # smooth stream
        return jnp.exp(-jnp.sum((q - jnp.array([0.3, -0.2])) ** 2))

    def b_true_exact(q):
        return jax.grad(U)(q)

    def b_true_rot(q):
        g = jax.grad(psi)(q)
        return jnp.array([-g[1], g[0]])

    b = jax.vmap(lambda q: b_true_exact(q) + b_true_rot(q))(qs)
    g = jnp.linspace(-1.4, 1.4, 6)
    C1, C2 = jnp.meshgrid(g, g, indexing="ij")
    centers = jnp.stack([C1.ravel(), C2.ravel()], -1)
    b_exact, b_rot = hodge_split_form(b, qs, centers, width=0.7)
    assert form_cosine(jax.vmap(b_exact)(qs), jax.vmap(b_true_exact)(qs)) > 0.9
    assert form_cosine(jax.vmap(b_rot)(qs), jax.vmap(b_true_rot)(qs)) > 0.9


def test_l7d_shape_identifiability_diagnostic():
    """The a-priori diagnostic separates G from G+R before any fit is attempted."""
    from experiments.arm.evaluate import shape_identifiability

    _, _, base, _, _, _, gen, demosG, demosGR = _stage_d_fixtures()
    sG = shape_identifiability(base, demosG, gen)
    sGR = shape_identifiability(base, demosGR, gen)
    assert sG < 0.02          # pure gravity: demos already trace base geodesics
    assert sGR > 3.0 * sG     # the rotational part bends demos at first order


def test_l7c_cost_asymmetry_recovers_potential():
    """Round-trip cost asymmetries recover gₛ·U by convex LSQ (seedless, global)."""
    from experiments.arm.learn import recover_potential_lsq, segment_asymmetry

    arm, _, _, _, mG, _, _, demosG, _ = _stage_d_fixtures()
    starts = jnp.concatenate([d[:-1] for d in demosG], axis=0)
    ends = jnp.concatenate([d[1:] for d in demosG], axis=0)
    deltas = jnp.concatenate([segment_asymmetry(mG, d) for d in demosG], axis=0)
    g = jnp.linspace(-1.4, 1.4, 6)
    C1, C2 = jnp.meshgrid(g, g, indexing="ij")
    centers = jnp.stack([C1.ravel(), C2.ravel()], -1)
    U_hat, _ = recover_potential_lsq(starts, ends, deltas, centers, width=0.7)
    r2 = _r2_vs_true_potential(arm, U_hat, jnp.concatenate(demosG, axis=0))
    assert r2 > 0.9


def test_l7e_full_form_recovery_on_gr():
    """On G+R the same convex estimator recovers the entire drift 1-form."""
    from experiments.arm.evaluate import form_cosine
    from experiments.arm.fields import star_grad
    from experiments.arm.learn import recover_form_lsq, segment_asymmetry

    arm, _, _, vortex, _, mGR, _, _, demosGR = _stage_d_fixtures()
    starts = jnp.concatenate([d[:-1] for d in demosGR], axis=0)
    ends = jnp.concatenate([d[1:] for d in demosGR], axis=0)
    deltas = jnp.concatenate([segment_asymmetry(mGR, d) for d in demosGR], axis=0)
    g = jnp.linspace(-1.4, 1.4, 6)
    C1, C2 = jnp.meshgrid(g, g, indexing="ij")
    centers = jnp.stack([C1.ravel(), C2.ravel()], -1)
    _, _, b_hat = recover_form_lsq(starts, ends, deltas, centers, width=0.7)
    region = jnp.concatenate(demosGR, axis=0)
    b_true = (0.15 * jax.vmap(arm.gravity)(region)
              + jax.vmap(lambda q: star_grad(vortex.stream, q))(region))
    assert form_cosine(jax.vmap(b_hat)(region), b_true) > 0.9


def test_l7_timing_channel_recovers_where_geometry_cannot():
    """Same potential hypothesis, same demos: L-T identifies gₛ·U, L-B does not."""
    from experiments.arm.fields import PotentialWind
    from experiments.arm.learn import (
        cost_matching_loss,
        el_residual_loss,
        fit_field,
        segment_costs,
    )

    arm, ang, _, _, mG, _, _, demosG, _ = _stage_d_fixtures()

    def make_metric(field):
        return build_arm_metric(arm, manifold=ang, wind_field=field)

    region = jnp.concatenate(demosG, axis=0)
    costs = [segment_costs(mG, d) for d in demosG]
    f0 = PotentialWind(arm, width=32, depth=2, key=jax.random.PRNGKey(1))
    ft, _ = fit_field(f0, demosG, make_metric, loss_fn=cost_matching_loss,
                      steps=600, lr=5e-3, demo_costs=costs)
    # normal_only quotients out vertex spacing (= timing in disguise), leaving
    # pure path shape — the channel the gauge theorem says is blind here.
    fb, _ = fit_field(f0, demosG, make_metric, loss_fn=el_residual_loss,
                      steps=600, lr=5e-3, normal_only=True)
    r2_t = _r2_vs_true_potential(arm, ft.potential, region)
    r2_b = _r2_vs_true_potential(arm, fb.potential, region)
    assert r2_t > 0.8                # timing: parametrization-rigidity
    assert r2_t - r2_b > 0.2         # the shape channel starves in comparison


# ===========================================================================
# Provider seam — synthetic providers satisfy the interface protocols
# ===========================================================================
def test_provider_protocol_satisfaction():
    """Synthetic providers structurally satisfy the injected protocols."""
    arm = PlanarArm(3)
    scene = CircleScene([[1.5, 0.0, 0.4]])
    dist = AnalyticDistance(arm, scene)
    demos = GroundTruthDemos([jnp.zeros((11, 3))])
    assert isinstance(arm, Robot)
    assert isinstance(scene, Scene)
    assert isinstance(dist, DistanceField)
    assert isinstance(demos, DemoSource)
    assert isinstance(arm.manifold, FlatTorus)
    assert arm.dof == arm.manifold.intrinsic_dim == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
