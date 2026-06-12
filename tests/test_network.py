"""Tests for neural network building blocks (VectorField, PSDMatrixField).

Verifies shapes, mathematical properties (symmetry, positive-definiteness), 
and compatibility with JAX transformations (jit, vmap, grad).
"""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ham.nn.networks import PSDMatrixField, RandomFourierFeatures, VectorField
from ham.utils.math import PSD_EPS


class TestRandomFourierFeatures(unittest.TestCase):
    """Standalone tests for the RFF embedding."""

    def setUp(self):
        self.key = jax.random.PRNGKey(42)

    def test_output_shape(self):
        M = 16
        rff = RandomFourierFeatures(in_dim=3, mapping_size=M, scale=1.0, key=self.key)
        feat = rff(jnp.array([1.0, 2.0, 3.0]))
        self.assertEqual(feat.shape, (2 * M,))

    def test_output_range(self):
        """cos/sin outputs must be bounded in [-1, 1]."""
        rff = RandomFourierFeatures(in_dim=3, mapping_size=32, scale=1.0, key=self.key)
        feat = rff(jnp.array([1.0, 2.0, 3.0]))
        self.assertTrue(jnp.all(feat >= -1.0))
        self.assertTrue(jnp.all(feat <= 1.0))

    def test_jit_compatible(self):
        rff = RandomFourierFeatures(in_dim=3, mapping_size=16, scale=1.0, key=self.key)
        x = jnp.array([0.1, 0.2, 0.3])
        jit_fn = eqx.filter_jit(lambda m, v: m(v))
        np.testing.assert_allclose(jit_fn(rff, x), rff(x), atol=1e-5)


class TestNetworks(unittest.TestCase):

    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        self.dim = 3

    def test_vector_field_transforms(self):
        """Verify VectorField works with JIT, vmap, and grad."""
        vf = VectorField(dim=self.dim, hidden_dim=16, depth=2, key=self.key)
        x = jnp.array([0.1, 0.2, 0.3])

        # 1. JIT
        @eqx.filter_jit
        def jit_vf(model, val):
            return model(val)

        np.testing.assert_allclose(jit_vf(vf, x), vf(x), atol=1e-5)

        # 2. Vmap (Map over data, not model)
        xs = jnp.ones((5, self.dim))
        vmap_vf = eqx.filter_vmap(lambda m, val: m(val), in_axes=(None, 0))
        out_batched = vmap_vf(vf, xs)
        self.assertEqual(out_batched.shape, (5, self.dim))

        # 3. Grad (Differentiate w.r.t model parameters)
        def loss(model, val):
            return jnp.sum(model(val)**2)

        grad_fn = eqx.filter_grad(loss)
        g_model = grad_fn(vf, x)

        # Verify gradients flow to MLP weights
        leaves = jax.tree_util.tree_leaves(g_model.mlp)
        self.assertTrue(len(leaves) > 0)
        self.assertTrue(jnp.all(jnp.isfinite(leaves[0])))

    def test_vector_field_grad_wrt_input(self):
        """Gradient w.r.t. input x must be finite (needed for spray computation)."""
        vf = VectorField(dim=self.dim, hidden_dim=16, depth=2, key=self.key)
        x = jnp.array([0.5, -0.3, 0.1])
        grad_x = jax.grad(lambda v: jnp.sum(vf(v)))(x)
        self.assertEqual(grad_x.shape, (self.dim,))
        self.assertTrue(jnp.all(jnp.isfinite(grad_x)))

    def test_psd_matrix_properties(self):
        """The output G must be Symmetric and Positive Definite."""
        psd_net = PSDMatrixField(dim=self.dim, hidden_dim=16, depth=2, key=self.key)

        # Test at multiple inputs
        test_inputs = [
            jnp.array([0.5, -0.5, 0.0]),
            jnp.zeros(self.dim),
            jnp.ones(self.dim) * 100.0,
        ]
        for x in test_inputs:
            G = psd_net(x)
            self.assertEqual(G.shape, (self.dim, self.dim))

            # Symmetry: G == G.T
            diff_sym = jnp.max(jnp.abs(G - G.T))
            self.assertLess(float(diff_sym), 1e-12, "Matrix is not exactly symmetric")

            # Positive Definite: minimum eigenvalue >= PSD_EPS floor
            eigs = jnp.linalg.eigvalsh(G)
            min_eig = float(jnp.min(eigs))
            self.assertGreaterEqual(min_eig, 0.9 * PSD_EPS,
                                    f"Min eigenvalue {min_eig} below regularisation floor")

    def test_psd_jit_vmap_grad(self):
        """PSDMatrixField must work under jit, vmap, and grad."""
        psd_net = PSDMatrixField(dim=self.dim, hidden_dim=16, depth=2, key=self.key)
        x = jnp.array([0.1, 0.2, 0.3])

        # JIT
        jit_fn = eqx.filter_jit(lambda m, v: m(v))
        G_jit = jit_fn(psd_net, x)
        np.testing.assert_allclose(G_jit, psd_net(x), atol=1e-5)

        # Vmap
        xs = jnp.ones((4, self.dim))
        batched = eqx.filter_vmap(lambda m, v: m(v), in_axes=(None, 0))(psd_net, xs)
        self.assertEqual(batched.shape, (4, self.dim, self.dim))

        # Grad (through trace of G)
        def scalar_fn(model, v):
            return jnp.trace(model(v))
        grad_model = eqx.filter_grad(scalar_fn)(psd_net, x)
        leaves = jax.tree_util.tree_leaves(grad_model.mlp)
        self.assertTrue(all(jnp.all(jnp.isfinite(l)) for l in leaves))

    def test_psd_with_fourier(self):
        """PSDMatrixField with use_fourier=True should still be SPD."""
        psd_net = PSDMatrixField(dim=self.dim, hidden_dim=16, depth=2,
                                 key=self.key, use_fourier=True)
        x = jnp.array([0.1, 0.2, 0.3])
        G = psd_net(x)
        self.assertEqual(G.shape, (self.dim, self.dim))
        eigs = jnp.linalg.eigvalsh(G)
        self.assertGreaterEqual(float(jnp.min(eigs)), 0.9 * PSD_EPS)

    def test_fourier_features_impact(self):
        """Ensure RFF embedding actually changes the output behavior."""
        vf_base = VectorField(3, 32, 2, self.key, use_fourier=False)
        vf_four = VectorField(3, 32, 2, self.key, use_fourier=True)

        x = jnp.array([0.1, 0.2, 0.3])
        out_base = vf_base(x)
        out_four = vf_four(x)

        # They should differ significantly due to frequency mapping
        self.assertFalse(jnp.allclose(out_base, out_four),
                         "Fourier and non-Fourier outputs should differ")

    def test_even_hidden_dim_assertion(self):
        """VectorField with odd hidden_dim and use_fourier=True should raise."""
        with self.assertRaises(AssertionError):
            VectorField(3, 33, 2, self.key, use_fourier=True)


if __name__ == '__main__':
    unittest.main()
