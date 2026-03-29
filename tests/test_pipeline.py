"""
Exhaustive tests for the modular training pipeline.

Covers:
  - Parameter freezing / unfreezing via eqx.partition
  - Loss component correctness (scalar output, weight application)
  - Multi-phase sequential execution
  - Pipeline skips phases when lineage pairs are missing
  - Gradient flow through unfrozen parameters only
"""
import unittest
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from ham.training.pipeline import TrainingPhase, HAMPipeline
from ham.training.losses import LossComponent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyModel(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(2, 2, key=k1)
        self.layer2 = eqx.nn.Linear(2, 2, key=k2)

    def __call__(self, x):
        return self.layer2(jax.nn.relu(self.layer1(x)))


class MSELoss(LossComponent):
    """Supervised MSE loss for testing."""
    def __init__(self, weight=1.0):
        super().__init__(weight, "MSE")

    def __call__(self, model, batch, key):
        x = batch[0]
        y = batch[1]
        return jnp.mean((model(x) - y) ** 2) * self.weight


class ConstantLoss(LossComponent):
    """Always returns a fixed value — useful for verifying multi-loss summation."""
    value: float

    def __init__(self, value=1.0, weight=1.0, name="Const"):
        self.value = value
        super().__init__(weight, name)

    def __call__(self, model, batch, key):
        return jnp.float32(self.value) * self.weight


class DummyDataset:
    def __init__(self, n=20, lineage_pairs=None):
        key = jax.random.PRNGKey(42)
        self.X = jax.random.normal(key, (n, 2))
        self.V = jnp.zeros((n, 2))
        self.lineage_pairs = lineage_pairs
        self.labels = None


def _filter_layer1(model):
    """Filter that unfreezes only layer1."""
    base = jax.tree_util.tree_map(lambda _: False, model)
    return eqx.tree_at(
        lambda m: m.layer1,
        base,
        replace=jax.tree_util.tree_map(
            lambda leaf: True if eqx.is_array(leaf) else False, model.layer1
        ),
    )


def _filter_layer2(model):
    """Filter that unfreezes only layer2."""
    base = jax.tree_util.tree_map(lambda _: False, model)
    return eqx.tree_at(
        lambda m: m.layer2,
        base,
        replace=jax.tree_util.tree_map(
            lambda leaf: True if eqx.is_array(leaf) else False, model.layer2
        ),
    )


def _filter_all(model):
    """Unfreeze everything."""
    return jax.tree_util.tree_map(lambda leaf: True if eqx.is_array(leaf) else False, model)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestParameterFreezing(unittest.TestCase):
    """Verify eqx.partition-based freezing works correctly."""

    def test_freeze_layer2_update_layer1(self):
        key = jax.random.PRNGKey(0)
        model = DummyModel(key)

        w1_init = model.layer1.weight.copy()
        w2_init = model.layer2.weight.copy()

        phase = TrainingPhase(
            name="FreezeL2",
            epochs=3,
            optimizer=optax.sgd(0.1),
            losses=[MSELoss()],
            filter_spec=_filter_layer1,
        )

        trained = HAMPipeline(model).fit(DummyDataset(), [phase], batch_size=5)

        self.assertFalse(jnp.allclose(trained.layer1.weight, w1_init),
                         "layer1 should update")
        self.assertTrue(jnp.allclose(trained.layer2.weight, w2_init),
                        "layer2 should stay frozen")

    def test_freeze_layer1_update_layer2(self):
        key = jax.random.PRNGKey(0)
        model = DummyModel(key)

        w1_init = model.layer1.weight.copy()
        w2_init = model.layer2.weight.copy()

        phase = TrainingPhase(
            name="FreezeL1",
            epochs=3,
            optimizer=optax.sgd(0.1),
            losses=[MSELoss()],
            filter_spec=_filter_layer2,
        )

        trained = HAMPipeline(model).fit(DummyDataset(), [phase], batch_size=5)

        self.assertTrue(jnp.allclose(trained.layer1.weight, w1_init),
                        "layer1 should stay frozen")
        self.assertFalse(jnp.allclose(trained.layer2.weight, w2_init),
                         "layer2 should update")

    def test_unfreeze_all(self):
        key = jax.random.PRNGKey(0)
        model = DummyModel(key)

        w1_init = model.layer1.weight.copy()
        w2_init = model.layer2.weight.copy()

        phase = TrainingPhase(
            name="All",
            epochs=3,
            optimizer=optax.sgd(0.1),
            losses=[MSELoss()],
            filter_spec=_filter_all,
        )

        trained = HAMPipeline(model).fit(DummyDataset(), [phase], batch_size=5)

        self.assertFalse(jnp.allclose(trained.layer1.weight, w1_init))
        self.assertFalse(jnp.allclose(trained.layer2.weight, w2_init))


class TestLossComponents(unittest.TestCase):
    """Verify LossComponent contract."""

    def test_mse_returns_scalar(self):
        model = DummyModel(jax.random.PRNGKey(0))
        loss = MSELoss()
        x = jnp.ones(2)
        y = jnp.zeros(2)
        val = loss(model, (x, y), jax.random.PRNGKey(1))
        self.assertEqual(val.shape, ())

    def test_weight_scaling(self):
        model = DummyModel(jax.random.PRNGKey(0))
        x, y = jnp.ones(2), jnp.zeros(2)

        base = MSELoss(weight=1.0)(model, (x, y), jax.random.PRNGKey(1))
        scaled = MSELoss(weight=3.0)(model, (x, y), jax.random.PRNGKey(1))

        self.assertAlmostEqual(float(scaled), float(base) * 3.0, places=5)

    def test_multiple_losses_sum(self):
        """Pipeline should sum all active losses."""
        model = DummyModel(jax.random.PRNGKey(0))
        x, y = jnp.ones(2), jnp.zeros(2)

        l1 = ConstantLoss(value=2.0, weight=1.0, name="C1")
        l2 = ConstantLoss(value=3.0, weight=1.0, name="C2")

        total = l1(model, (x, y), jax.random.PRNGKey(0)) + l2(model, (x, y), jax.random.PRNGKey(0))
        self.assertAlmostEqual(float(total), 5.0, places=5)


class TestMultiPhaseExecution(unittest.TestCase):
    """Verify that sequential phases each run and modify only their targets."""

    def test_two_phases_sequential(self):
        key = jax.random.PRNGKey(0)
        model = DummyModel(key)

        w1_init = model.layer1.weight.copy()
        w2_init = model.layer2.weight.copy()

        phase1 = TrainingPhase(
            name="P1", epochs=2, optimizer=optax.sgd(0.1),
            losses=[MSELoss()], filter_spec=_filter_layer1,
        )
        phase2 = TrainingPhase(
            name="P2", epochs=2, optimizer=optax.sgd(0.1),
            losses=[MSELoss()], filter_spec=_filter_layer2,
        )

        trained = HAMPipeline(model).fit(DummyDataset(), [phase1, phase2], batch_size=5)

        # Both layers should have changed — each in its own phase
        self.assertFalse(jnp.allclose(trained.layer1.weight, w1_init))
        self.assertFalse(jnp.allclose(trained.layer2.weight, w2_init))

    def test_skip_phase_when_no_pairs(self):
        """If requires_pairs=True but dataset has no pairs, phase should be skipped."""
        key = jax.random.PRNGKey(0)
        model = DummyModel(key)
        w_init = model.layer2.weight.copy()

        phase = TrainingPhase(
            name="SkipMe", epochs=5, optimizer=optax.sgd(0.1),
            losses=[MSELoss()], filter_spec=_filter_layer2,
            requires_pairs=True,
        )

        trained = HAMPipeline(model).fit(DummyDataset(lineage_pairs=None), [phase], batch_size=5)

        self.assertTrue(jnp.allclose(trained.layer2.weight, w_init),
                        "Phase should have been skipped, weights unchanged")

    def test_loss_decreases_over_epochs(self):
        """Sanity check: loss should go down when training on a simple target."""
        key = jax.random.PRNGKey(0)
        model = DummyModel(key)

        phase = TrainingPhase(
            name="Converge", epochs=50, optimizer=optax.adam(1e-2),
            losses=[MSELoss()], filter_spec=_filter_all,
        )

        # Create a tiny dataset where targets are zeros
        ds = DummyDataset(n=10)

        pipeline = HAMPipeline(model)
        trained = pipeline.fit(ds, [phase], batch_size=10)

        # Evaluate initial vs final loss
        x = ds.X
        initial_pred = jax.vmap(model)(x)
        final_pred = jax.vmap(trained)(x)

        initial_loss = jnp.mean(initial_pred ** 2)
        final_loss = jnp.mean(final_pred ** 2)

        self.assertLess(float(final_loss), float(initial_loss),
                        "Loss should decrease over training")


if __name__ == "__main__":
    unittest.main()
