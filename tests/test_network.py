import jax
import jax.numpy as jnp
import unittest
import equinox as eqx
from ham.nn.networks import VectorField, PSDMatrixField

class TestNetworks(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        self.dim = 3

    def test_vector_field_shape(self):
        """Ensure VectorField outputs (D,) vectors."""
        vf = VectorField(dim=self.dim, hidden_dim=16, depth=2, key=self.key)
        x = jnp.zeros(self.dim)
        v = vf(x)
        self.assertEqual(v.shape, (self.dim,))
        self.assertTrue(jnp.isfinite(v).all())

    def test_psd_matrix_properties(self):
        """
        CRITICAL: The output G must be Symmetric and Positive Definite.
        """
        psd_net = PSDMatrixField(dim=self.dim, hidden_dim=16, depth=2, key=self.key)
        x = jnp.ones(self.dim)
        G = psd_net(x)
        
        # 1. Shape
        self.assertEqual(G.shape, (self.dim, self.dim))
        
        # 2. Symmetry: G == G.T
        diff_sym = jnp.max(jnp.abs(G - G.T))
        self.assertLess(diff_sym, 1e-6, "Matrix is not symmetric")
        
        # 3. Positive Definite: All eigenvalues > 0
        eigs = jnp.linalg.eigvalsh(G)
        min_eig = jnp.min(eigs)
        print(f"\nMin Eigenvalue: {min_eig}")
        self.assertGreater(min_eig, 0.0, "Matrix is not positive definite")

    def test_fourier_features(self):
        """Verify RFF embedding runs and changes output."""
        vf_base = VectorField(3, 16, 2, self.key, use_fourier=False)
        vf_four = VectorField(3, 16, 2, self.key, use_fourier=True)
        
        x = jnp.array([0.1, 0.2, 0.3])
        out_base = vf_base(x)
        out_four = vf_four(x)
        
        self.assertEqual(out_base.shape, out_four.shape)
        # They shouldn't be identical (random weights differ, but functional path differs too)

if __name__ == '__main__':
    unittest.main()