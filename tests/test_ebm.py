"""Tests for the EBM stack: nn.ebm, nn.kde, and training.losses_ebm."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ham.nn.ebm import QuadraticHead, ScalarEnergyField
from ham.nn.kde import GaussianKDEEnergy
from ham.training.losses_ebm import (
    ContrastiveDivergenceLoss,
    DenoisingScoreMatchingLoss,
    MSELoss,
    sgld_sample_single,
    sgld_step,
)

KEY = jax.random.PRNGKey(0)
D = 3


class TestScalarEnergyField(unittest.TestCase):
    def test_scalar_output_and_finite_grad(self):
        ebm = ScalarEnergyField(dim=D, hidden_dim=16, depth=2, key=KEY)
        x = jnp.array([0.3, -0.7, 1.1])
        e = ebm(x)
        self.assertEqual(e.shape, ())
        self.assertTrue(np.isfinite(float(e)))
        g = jax.grad(ebm)(x)
        self.assertEqual(g.shape, (D,))
        self.assertTrue(np.all(np.isfinite(np.array(g))))

    def test_fourier_variant_and_vmap(self):
        ebm = ScalarEnergyField(
            dim=D, hidden_dim=16, depth=2, key=KEY, use_fourier=True
        )
        xs = jax.random.normal(KEY, (8, D))
        es = jax.vmap(ebm)(xs)
        self.assertEqual(es.shape, (8,))
        self.assertTrue(np.all(np.isfinite(np.array(es))))

    def test_fourier_requires_even_hidden_dim(self):
        with self.assertRaises(AssertionError):
            ScalarEnergyField(dim=D, hidden_dim=15, depth=2, key=KEY, use_fourier=True)


class TestQuadraticHead(unittest.TestCase):
    def test_output_shape_and_grad(self):
        head = QuadraticHead(in_features=4, key=KEY)
        x = jnp.arange(4.0)
        y = head(x)
        self.assertEqual(y.shape, (1,))
        g = jax.grad(lambda z: head(z).sum())(x)
        self.assertTrue(np.all(np.isfinite(np.array(g))))


class TestGaussianKDEEnergy(unittest.TestCase):
    def test_energy_lower_near_data(self):
        centers = jax.random.normal(KEY, (32, D)) * 0.1  # cluster near origin
        kde = GaussianKDEEnergy(centers, sigma=0.5)
        e_near = kde(jnp.zeros(D))
        e_far = kde(5.0 * jnp.ones(D))
        self.assertLess(float(e_near), float(e_far))

    def test_grad_points_toward_data(self):
        """-grad E (the score) at a point away from the data points back at it."""
        centers = jnp.zeros((16, D))
        kde = GaussianKDEEnergy(centers, sigma=1.0)
        x = jnp.array([2.0, 0.0, 0.0])
        score = -jax.grad(kde)(x)
        # Score should point from x toward the origin (negative x-direction).
        self.assertLess(float(score[0]), 0.0)


class TestSGLD(unittest.TestCase):
    def _bowl(self, x):
        return 0.5 * jnp.sum(x**2)

    def test_noiseless_step_descends(self):
        x = jnp.array([1.0, -2.0, 0.5])
        x_next = sgld_step(x, self._bowl, step_size=0.1, noise_scale=0.0, key=KEY)
        self.assertLess(float(self._bowl(x_next)), float(self._bowl(x)))

    def test_sampler_shape_and_finiteness(self):
        x0 = jnp.ones(D) * 3.0
        x = sgld_sample_single(
            x0, self._bowl, num_steps=50, step_size=0.05, noise_scale=0.01, key=KEY
        )
        self.assertEqual(x.shape, (D,))
        self.assertTrue(np.all(np.isfinite(np.array(x))))
        # A quadratic bowl concentrates samples near the origin.
        self.assertLess(float(jnp.linalg.norm(x)), float(jnp.linalg.norm(x0)))


class TestEBMLosses(unittest.TestCase):
    def test_cd_loss_finite_and_differentiable(self):
        ebm = ScalarEnergyField(dim=D, hidden_dim=16, depth=2, key=KEY)
        loss_fn = ContrastiveDivergenceLoss(sgld_steps=10)
        batch = (jnp.array([0.2, -0.1, 0.4]),)

        def scalar_loss(model):
            return loss_fn(model, batch, KEY)

        val = scalar_loss(ebm)
        self.assertTrue(np.isfinite(float(val)))
        import equinox as eqx

        grads = eqx.filter_grad(scalar_loss)(ebm)
        leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        self.assertTrue(all(np.all(np.isfinite(np.array(g))) for g in leaves))
        self.assertTrue(any(float(jnp.abs(g).max()) > 0 for g in leaves))

    def test_dsm_loss_zero_for_perfect_score(self):
        """For E(x) = |x - x0|^2 / (2 sigma^2), the model score equals the DSM
        target exactly, so the loss vanishes for any noise draw."""
        sigma = 0.3
        x0 = jnp.array([0.5, -1.0, 0.25])

        def perfect_energy(x):
            return jnp.sum((x - x0) ** 2) / (2.0 * sigma**2)

        loss_fn = DenoisingScoreMatchingLoss(sigma=sigma)
        val = loss_fn(perfect_energy, (x0,), KEY)
        self.assertLess(float(val), 1e-8)

    def test_mse_loss_zero_at_target(self):
        loss_fn = MSELoss()
        model = lambda x: jnp.sum(x)
        inputs = jnp.array([1.0, 2.0, 3.0])
        batch = (inputs, None, jnp.array(6.0))
        self.assertAlmostEqual(float(loss_fn(model, batch, KEY)), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
