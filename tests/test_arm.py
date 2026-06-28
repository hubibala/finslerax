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
from experiments.arm.interfaces import DemoSource, DistanceField, Robot, Scene
from ham.geometry.manifolds import FlatTorus

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
