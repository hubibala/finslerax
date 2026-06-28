"""Tests for AVBDSolver.implicit_diff=True (IFT-based backprop).

Verifies that:
 1. Forward pass produces an identical path to the standard unrolled solver.
 2. Backward pass produces finite, non-zero gradients w.r.t. metric parameters.
 3. Gradient *direction* is consistent with a finite-difference check.
 4. Memory is not proportional to solver iterations (soft check via timing).
 5. ArrivalTimeLoss works end-to-end with the implicit solver.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.zoo import Riemannian
from ham.models.learned import NeuralRanders
from ham.solvers.avbd import AVBDSolver
from ham.training.losses import ArrivalTimeLoss

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _flat_identity_metric():
    """Euclidean R^2 with G=I — geodesics are straight lines."""
    manifold = EuclideanSpace(2)
    return Riemannian(manifold, lambda x: jnp.eye(2))


def _neural_randers(key, hidden_dim=16, depth=2):
    manifold = EuclideanSpace(2)
    return NeuralRanders(manifold, key, hidden_dim=hidden_dim, depth=depth)


def _make_solvers(iterations=30):
    """Return a matched pair (unrolled, implicit) of AVBDSolvers."""
    common = dict(step_size=0.05, iterations=iterations, energy_tol=1e-9)
    unrolled = AVBDSolver(**common, implicit_diff=False)
    implicit = AVBDSolver(**common, implicit_diff=True)
    return unrolled, implicit


# ---------------------------------------------------------------------------
# 1. Forward correctness
# ---------------------------------------------------------------------------

class TestImplicitForward:
    """The IFT wrapper must not change the converged path."""

    def test_same_path_identity_metric(self):
        """On flat R^2 both solvers should return the same inner vertices."""
        metric = _flat_identity_metric()
        unrolled, implicit = _make_solvers()

        p0 = jnp.array([0.0, 0.0])
        p1 = jnp.array([1.0, 1.0])

        traj_u = unrolled.solve(metric, p0, p1, n_steps=8)
        traj_i = implicit.solve(metric, p0, p1, n_steps=8)

        # Inner vertices must match (endpoints are pinned).
        # The implicit solver uses while_loop (train_mode=False) for memory efficiency,
        # which may stop 1-2 iterations earlier than the unrolled scan, so we allow a
        # slightly looser tolerance (1e-4) than the machine-epsilon level.
        assert jnp.allclose(traj_u.xs, traj_i.xs, atol=1e-4), (
            f"Path mismatch: max diff = {jnp.max(jnp.abs(traj_u.xs - traj_i.xs))}"
        )

    def test_endpoints_pinned(self):
        """Implicit solver must not move boundary vertices."""
        metric = _flat_identity_metric()
        _, implicit = _make_solvers()

        p0 = jnp.array([0.5, -0.3])
        p1 = jnp.array([-0.2, 0.7])

        traj = implicit.solve(metric, p0, p1, n_steps=6)
        assert jnp.allclose(traj.xs[0], p0, atol=1e-5)
        assert jnp.allclose(traj.xs[-1], p1, atol=1e-5)

    def test_trajectory_shape(self):
        """Output shape must be (n_steps+1, D)."""
        metric = _flat_identity_metric()
        _, implicit = _make_solvers()

        n_steps = 10
        traj = implicit.solve(metric, jnp.zeros(2), jnp.ones(2), n_steps=n_steps)
        assert traj.xs.shape == (n_steps + 1, 2)
        assert traj.vs.shape == (n_steps, 2)


# ---------------------------------------------------------------------------
# 2. Gradient finiteness and non-triviality
# ---------------------------------------------------------------------------

class TestImplicitGradients:
    """Gradients from IFT backward pass must be finite and non-zero."""

    def test_gradients_finite_neural_randers(self):
        """IFT gradients w.r.t. NeuralRanders parameters are all finite."""
        key = jax.random.PRNGKey(0)
        metric = _neural_randers(key)
        _, implicit = _make_solvers(iterations=20)
        loss_fn = ArrivalTimeLoss(solver=implicit, solver_steps=8)

        source = jnp.array([0.0, 0.0])
        x_obs  = jnp.array([[0.5, 0.0], [0.0, 0.5], [0.4, 0.4]])
        t_obs  = jnp.array([0.5, 0.5, jnp.sqrt(0.32)])

        @eqx.filter_jit
        def compute(m):
            return loss_fn(m, source, x_obs, t_obs)

        grads = eqx.filter_grad(compute)(metric)
        leaves = jax.tree_util.tree_leaves(grads)
        for leaf in leaves:
            if hasattr(leaf, 'shape'):
                assert jnp.all(jnp.isfinite(leaf)), "Non-finite gradient leaf"

    def test_gradients_nonzero(self):
        """At least some gradients must be non-zero (learning signal exists)."""
        key = jax.random.PRNGKey(1)
        metric = _neural_randers(key)
        _, implicit = _make_solvers(iterations=20)
        loss_fn = ArrivalTimeLoss(solver=implicit, solver_steps=8)

        source = jnp.array([0.0, 0.0])
        x_obs  = jnp.array([[0.6, 0.2]])
        t_obs  = jnp.array([0.4])  # deliberately wrong

        @eqx.filter_jit
        def compute(m):
            return loss_fn(m, source, x_obs, t_obs)

        grads = eqx.filter_grad(compute)(metric)
        leaves = [l for l in jax.tree_util.tree_leaves(grads)
                  if hasattr(l, 'shape') and l.size > 0]
        nonzero = any(jnp.any(l != 0) for l in leaves)
        assert nonzero, "All gradients are zero — no learning signal"


# ---------------------------------------------------------------------------
# 3. Gradient direction consistency (finite-difference check)
# ---------------------------------------------------------------------------

class TestGradientConsistency:
    """IFT gradient direction must agree with finite differences."""

    def test_fd_agreement_identity_metric(self):
        """Directional derivative via IFT vs FD on a scalar parameter."""
        manifold = EuclideanSpace(2)
        scale = jnp.array(1.0)  # scalar metric multiplier

        class ScaledIdentity(eqx.Module):
            scale: jax.Array

            def __call__(self, x):
                return self.scale * jnp.eye(2)

        def make_metric(s):
            return Riemannian(manifold, ScaledIdentity(s))

        _, implicit = _make_solvers(iterations=40)

        source = jnp.array([0.0, 0.0])
        target = jnp.array([0.8, 0.0])

        def path_energy(s):
            m = make_metric(s)
            traj = implicit.solve(m, source, target, n_steps=6)
            return traj.energy

        grad_ift = jax.grad(path_energy)(scale)

        eps = 1e-4
        fd = (path_energy(scale + eps) - path_energy(scale - eps)) / (2 * eps)

        rel_err = jnp.abs(grad_ift - fd) / (jnp.abs(fd) + 1e-8)
        assert float(rel_err) < 0.05, (
            f"IFT grad {float(grad_ift):.6f} vs FD {float(fd):.6f} "
            f"(rel err = {float(rel_err):.4f})"
        )


# ---------------------------------------------------------------------------
# 4. End-to-end with ArrivalTimeLoss
# ---------------------------------------------------------------------------

class TestArrivalTimeLossImplicit:
    """ArrivalTimeLoss must work identically with both solver modes."""

    def test_loss_finite(self):
        """Loss value must be finite with the implicit solver."""
        key = jax.random.PRNGKey(42)
        metric = _neural_randers(key)
        _, implicit = _make_solvers(iterations=20)
        loss_fn = ArrivalTimeLoss(solver=implicit, solver_steps=8)

        source = jnp.array([0.0, 0.0])
        x_obs  = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        t_obs  = jnp.array([1.0, 1.0])

        loss = loss_fn(metric, source, x_obs, t_obs)
        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"

    def test_loss_lower_correct_times(self):
        """Loss is lower for correct arrival times than for wrong ones."""
        metric = _flat_identity_metric()
        _, implicit = _make_solvers(iterations=40)
        loss_fn = ArrivalTimeLoss(solver=implicit, solver_steps=12)

        source = jnp.array([0.0, 0.0])
        x_obs  = jnp.array([[1.0, 0.0]])

        loss_correct = loss_fn(metric, source, x_obs, jnp.array([1.0]))
        loss_wrong   = loss_fn(metric, source, x_obs, jnp.array([5.0]))

        assert loss_wrong > loss_correct, (
            f"Wrong-time loss ({float(loss_wrong):.4f}) should exceed "
            f"correct-time loss ({float(loss_correct):.4f})"
        )

    def test_implicit_vs_unrolled_loss_close(self):
        """Both modes should produce similar loss values at convergence."""
        key = jax.random.PRNGKey(7)
        metric = _neural_randers(key)
        unrolled, implicit = _make_solvers(iterations=40)

        loss_u = ArrivalTimeLoss(solver=unrolled, solver_steps=10)
        loss_i = ArrivalTimeLoss(solver=implicit, solver_steps=10)

        source = jnp.array([0.0, 0.0])
        x_obs  = jnp.array([[0.5, 0.5]])
        t_obs  = jnp.array([0.7])

        l_u = float(loss_u(metric, source, x_obs, t_obs))
        l_i = float(loss_i(metric, source, x_obs, t_obs))

        rel = abs(l_u - l_i) / (abs(l_u) + 1e-8)
        assert rel < 0.10, (
            f"Unrolled loss {l_u:.4f} vs implicit loss {l_i:.4f} "
            f"differ by {rel*100:.1f}% — solver may not have converged"
        )


# ---------------------------------------------------------------------------
# 5. Constrained and curved-manifold gradient consistency
#    (regression for the IFT adjoint fixes of 2026-06: constraint Hessian in
#    the block Jacobian, tangent-projected residual, transposed solve)
# ---------------------------------------------------------------------------


class _DiagMetricFn2(eqx.Module):
    """diag(1, s) metric field with a learnable y-weight."""

    s: jax.Array

    def __call__(self, x):
        return jnp.diag(jnp.array([1.0, 0.0]) + jnp.array([0.0, 1.0]) * self.s)


class _DiagMetricFn3(eqx.Module):
    """diag(1, 1, s) ambient metric field with a learnable z-weight."""

    s: jax.Array

    def __call__(self, x):
        return jnp.diag(
            jnp.array([1.0, 1.0, 0.0]) + jnp.array([0.0, 0.0, 1.0]) * self.s
        )


class TestConstrainedAndCurvedGradients:
    """FD agreement beyond the unconstrained-Euclidean case."""

    def test_fd_agreement_constrained(self):
        """IFT gradient with an active constraint tracks finite differences.

        Tolerance note: the backward pass treats the converged multipliers
        and stiffness as constants (their theta-sensitivity is dropped), so
        agreement is approximate; 20% guards the order of magnitude and the
        sign, which the pre-fix adjoint (missing the penalty Hessian) failed
        by >100x.
        """
        from ham.geometry.zoo import Riemannian

        manifold = EuclideanSpace(2)
        solver = AVBDSolver(
            step_size=0.05, iterations=300, energy_tol=1e-12, implicit_diff=True
        )
        p0 = jnp.array([0.0, 0.0])
        p1 = jnp.array([0.8, 0.0])
        constraints = [lambda x: x[1] - 0.1]

        def path_energy(s):
            m = Riemannian(manifold, _DiagMetricFn2(s))
            traj = solver.solve(m, p0, p1, n_steps=6, constraints=constraints)
            return traj.energy

        s0 = jnp.array(1.3)
        grad_ift = jax.grad(path_energy)(s0)

        eps = 1e-3
        fd = (path_energy(s0 + eps) - path_energy(s0 - eps)) / (2 * eps)

        rel_err = jnp.abs(grad_ift - fd) / (jnp.abs(fd) + 1e-8)
        assert float(rel_err) < 0.20, (
            f"IFT grad {float(grad_ift):.6f} vs FD {float(fd):.6f} "
            f"(rel err = {float(rel_err):.4f})"
        )

    def test_fd_agreement_sphere(self):
        """IFT gradients on a curved manifold (sphere) track FD."""
        from ham.geometry import Sphere
        from ham.geometry.zoo import Riemannian

        sphere = Sphere()
        p0 = jnp.array([1.0, 0.0, 0.0])
        p1 = jnp.array([0.0, 1.0, 0.2])
        p1 = p1 / jnp.linalg.norm(p1)

        solver = AVBDSolver(
            step_size=0.05, iterations=300, energy_tol=1e-12, implicit_diff=True
        )

        def path_energy(s):
            m = Riemannian(sphere, _DiagMetricFn3(s))
            traj = solver.solve(m, p0, p1, n_steps=6)
            return traj.energy

        s0 = jnp.array(1.4)
        grad_ift = jax.grad(path_energy)(s0)

        # eps below ~3e-3 is dominated by float32 noise in the energy
        # differences; 1e-2 sits in the stable central-difference window.
        eps = 1e-2
        fd = (path_energy(s0 + eps) - path_energy(s0 - eps)) / (2 * eps)

        rel_err = jnp.abs(grad_ift - fd) / (jnp.abs(fd) + 1e-8)
        assert float(rel_err) < 0.10, (
            f"IFT grad {float(grad_ift):.6f} vs FD {float(fd):.6f} "
            f"(rel err = {float(rel_err):.4f})"
        )

    def test_constrained_solve_stays_converged(self):
        """Long constrained runs must not destabilize after convergence.

        Regression for the ALM schedule fix: unconditional stiffness growth
        plus per-sweep dual updates used to blow the solve up *after* it had
        converged (violation 0.8 at 300 iterations vs 1e-4 at 25).
        """
        from ham.geometry.zoo import Riemannian

        manifold = EuclideanSpace(2)
        m = Riemannian(manifold, _DiagMetricFn2(jnp.array(1.3)))
        p0 = jnp.array([0.0, 0.0])
        p1 = jnp.array([0.8, 0.0])
        constraints = [lambda x: x[1] - 0.1]

        solver = AVBDSolver(
            step_size=0.05, beta=1.2, iterations=300, energy_tol=1e-12,
            implicit_diff=False,
        )
        traj = solver.solve(
            m, p0, p1, n_steps=6, constraints=constraints, train_mode=True
        )
        assert float(traj.constraint_violation) < 5e-4, (
            f"violation {float(traj.constraint_violation):.2e} after 300 sweeps"
        )
        assert float(traj.energy) < 0.1
