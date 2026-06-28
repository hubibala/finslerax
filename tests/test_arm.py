"""Validation suite for the robot-arm geodesic experiment (experiments/arm).

Run: ``JAX_PLATFORMS=cpu pytest tests/test_arm.py`` (and again with
``JAX_ENABLE_X64=1``). Mirrors the validation ladder in
``spec/robot_arm_geodesic_PLAN.md`` §5 — one test per rung — so the science
stages run on a proven machine. This file grows a rung per chunk.

Rungs covered so far:
    L1 — Robot sanity: FK vs known poses, M(q) symmetric-PD, Jacobian vs
         finite-difference, gravity vs finite-difference of the potential.
    Provider seam: the synthetic providers structurally satisfy the protocols.
"""

import pathlib
import sys

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
from experiments.arm.interfaces import DemoSource, DistanceField, Robot, Scene
from experiments.arm.medium import angle_manifold, build_arm_metric
from ham.geometry.manifolds import EuclideanSpace, FlatTorus
from ham.geometry.zoo import Randers

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
    W = -0.1 * arm.gravity(q)
    ref = Randers(
        EuclideanSpace(2), lambda z, H=H: H, lambda z, W=W: W, wind_mode="soft"
    )
    np.testing.assert_allclose(
        metric.metric_fn(q, v), ref.metric_fn(q, v), atol=atol, rtol=rtol
    )


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
