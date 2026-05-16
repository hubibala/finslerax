"""Tests for neural network building blocks (VectorField, PSDMatrixField).

Verifies shapes, mathematical properties (symmetry, positive-definiteness), 
and compatibility with JAX transformations (jit, vmap, grad).
"""

import jax
import jax.numpy as jnp
import unittest
import equinox as eqx
import numpy as np

from ham.nn.networks import VectorField, PSDMatrixField, RandomFourierFeatures

class TestNetworks(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        self.dim = 3

    def test_rff_properties(self):
        """Verify Random Fourier Features output range and shape."""
        M = 16
        rff = RandomFourierFeatures(in_dim=self.dim, mapping_size=M, scale=1.0, key=self.key)
        x = jnp.array([1.0, 2.0, 3.0])
        feat = rff(x)
        
        # 1. Shape
        self.assertEqual(feat.shape, (2 * M,))
        
        # 2. Range [-1, 1]
        self.assertTrue(jnp.all(feat >= -1.0))
        self.assertTrue(jnp.all(feat <= 1.0))

    def test_vector_field_transforms(self):
        """Verify VectorField works with JIT, vmap, and grad."""
        vf = VectorField(dim=self.dim, hidden_dim=16, depth=2, key=self.key)
        x = jnp.array([0.1, 0.2, 0.3])
        
        # 1. JIT
        @eqx.filter_jit
        def jit_vf(model, val):
            return model(val)
            
        np.testing.assert_allclose(jit_vf(vf, x), vf(x), atol=1e-7)
        
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
        # filter_grad returns a grad of the same structure as the first argument
        leaves = jax.tree_util.tree_leaves(g_model.mlp)
        self.assertTrue(len(leaves) > 0)
        self.assertTrue(jnp.all(jnp.isfinite(leaves[0])))

    def test_psd_matrix_properties(self):
        """
        CRITICAL: The output G must be Symmetric and Positive Definite.
        """
        psd_net = PSDMatrixField(dim=self.dim, hidden_dim=16, depth=2, key=self.key)
        x = jnp.array([0.5, -0.5, 0.0])
        
        # 1. JIT evaluation
        @eqx.filter_jit
        def get_G(model, val):
            return model(val)
            
        G = get_G(psd_net, x)
        self.assertEqual(G.shape, (self.dim, self.dim))
        
        # 2. Symmetry: G == G.T
        diff_sym = jnp.max(jnp.abs(G - G.T))
        self.assertLess(diff_sym, 1e-12, "Matrix is not exactly symmetric")
        
        # 3. Positive Definite: All eigenvalues >= eps
        eigs = jnp.linalg.eigvalsh(G)
        min_eig = jnp.min(eigs)
        # Expected eps is 1e-4
        self.assertGreaterEqual(min_eig, 0.9e-4)

    def test_fourier_features_impact(self):
        """Ensure RFF embedding actually changes the output behavior."""
        vf_base = VectorField(3, 32, 2, self.key, use_fourier=False)
        vf_four = VectorField(3, 32, 2, self.key, use_fourier=True)
        
        x = jnp.array([0.1, 0.2, 0.3])
        out_base = vf_base(x)
        out_four = vf_four(x)
        
        # They should differ significantly due to frequency mapping
        diff = jnp.max(jnp.abs(out_base - out_four))
        self.assertGreater(diff, 1e-3)

if __name__ == '__main__':
    unittest.main()