"""Tests for ArrivalTimeLoss used in metric recovery experiments."""
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from jax import config

config.update("jax_enable_x64", True)

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.zoo import Riemannian
from ham.models.learned import NeuralRanders
from ham.solvers.avbd import AVBDSolver
from ham.training.losses import ArrivalTimeLoss


def _identity_metric():
    """Create a Riemannian metric with G(x)=I on R^2."""
    manifold = EuclideanSpace(2)
    g_net = lambda x: jnp.eye(2)
    return Riemannian(manifold, g_net)


class TestArrivalTimeLoss:
    """Tests for the ArrivalTimeLoss class."""

    def test_identity_metric_distance(self):
        """On flat R^2 with G=I, geodesic distance should equal Euclidean distance."""
        metric = _identity_metric()
        solver = AVBDSolver(step_size=0.05, iterations=80)
        loss_fn = ArrivalTimeLoss(solver=solver, solver_steps=15)

        source = jnp.array([0.0, 0.0])
        x_obs = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # True Euclidean distances
        t_obs = jnp.array([1.0, 1.0, jnp.sqrt(2.0)])

        loss = loss_fn(metric, source, x_obs, t_obs)
        # Loss should be near zero since predicted distances match
        assert loss < 0.01, f"Loss too high for identity metric: {loss}"

    def test_gradient_flows(self):
        """Gradients of ArrivalTimeLoss w.r.t. NeuralRanders parameters should be finite."""
        manifold = EuclideanSpace(2)
        key = jax.random.PRNGKey(42)
        metric = NeuralRanders(manifold, key, hidden_dim=16, depth=2)
        solver = AVBDSolver(step_size=0.05, iterations=50)
        loss_fn = ArrivalTimeLoss(solver=solver, solver_steps=10)

        source = jnp.array([0.0, 0.0])
        x_obs = jnp.array([[0.5, 0.0], [0.0, 0.5]])
        t_obs = jnp.array([0.5, 0.5])

        @eqx.filter_jit
        def compute_loss(m):
            return loss_fn(m, source, x_obs, t_obs)

        loss = compute_loss(metric)
        grads = eqx.filter_grad(compute_loss)(metric)

        assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
        # Check that at least some gradients are non-zero and finite
        grad_leaves = jax.tree_util.tree_leaves(grads)
        finite_grads = [jnp.all(jnp.isfinite(g)) for g in grad_leaves if g.size > 0]
        assert all(finite_grads), "Some gradients are not finite"
        nonzero_grads = [jnp.any(g != 0) for g in grad_leaves if g.size > 0]
        assert any(nonzero_grads), "All gradients are zero — no learning signal"

    def test_loss_decreases_with_correct_metric(self):
        """Loss should be lower for a metric that produces correct distances."""
        metric = _identity_metric()
        solver = AVBDSolver(step_size=0.05, iterations=80)
        loss_fn = ArrivalTimeLoss(solver=solver, solver_steps=15)

        source = jnp.array([0.0, 0.0])
        x_obs = jnp.array([[1.0, 0.0]])

        # Correct arrival time
        t_correct = jnp.array([1.0])
        loss_correct = loss_fn(metric, source, x_obs, t_correct)

        # Wrong arrival time
        t_wrong = jnp.array([5.0])
        loss_wrong = loss_fn(metric, source, x_obs, t_wrong)

        assert loss_wrong > loss_correct, (
            f"Loss with wrong T ({loss_wrong}) should exceed loss with correct T ({loss_correct})"
        )
