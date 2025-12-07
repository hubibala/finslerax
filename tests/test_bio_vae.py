import jax
import jax.numpy as jnp
import unittest
import optax
import equinox as eqx

from ham.geometry import Sphere
from ham.geometry.zoo import Euclidean
from ham.bio.vae import GeometricVAE
from ham.sim.fields import rossby_haurwitz, get_stream_function_flow

class TestBioVAE(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(2025)
        
        # 1. Define the "Virtual Organism" (Ground Truth Physics)
        # We use a Sphere to represent the differentiation manifold (Waddington's Landscape)
        self.manifold = Sphere(radius=1.0)
        
        # We use the Rossby-Haurwitz wave from ham.sim.fields as the "True Flow"
        # This creates complex, wavy trajectories on the sphere
        self.true_flow = rossby_haurwitz(R=3, omega=1.0)
        
        # 2. Generate Synthetic Data
        # Sample 100 random cells on the manifold
        self.N_cells = 100
        raw_points = jax.random.normal(self.key, (self.N_cells, 3))
        self.cells_z = jax.vmap(self.manifold.project)(raw_points)
        
        # Calculate their "true" biological velocities (RNA velocity)
        self.cells_v = jax.vmap(self.true_flow)(self.cells_z)
        
        # Map to "Gene Space" (Ambient R^10)
        # We simulate a linear projection from Latent(3) -> Genes(10)
        self.gene_matrix = jax.random.normal(self.key, (3, 10))
        self.cells_x = self.cells_z @ self.gene_matrix  # X = Z * W
        
        # 3. Initialize VAE
        # We use a Euclidean metric on the Sphere as the prior
        self.metric = Euclidean(self.manifold)
        self.vae = GeometricVAE(
            data_dim=10, 
            latent_dim=3, 
            metric=self.metric, 
            key=self.key
        )

    def test_forward_shapes(self):
        """
        Smoke test: Does the model process a single cell correctly?
        """
        x = self.cells_x[0]
        z, v, logvar = self.vae.encode(x, self.key)
        
        self.assertEqual(z.shape, (3,))
        self.assertEqual(v.shape, (3,))
        
        # Check Manifold Adherence
        norm_z = jnp.linalg.norm(z)
        self.assertAlmostEqual(norm_z, 1.0, places=5, msg="Latent point not on Sphere")
        
        # Check Tangency
        ortho = jnp.dot(z, v)
        self.assertAlmostEqual(ortho, 0.0, places=5, msg="Velocity not tangent to Sphere")

    def test_overfitting_trajectory(self):
        """
        Optimization Test: Can we train the VAE to reconstruct the synthetic organism?
        """
        optimizer = optax.adam(learning_rate=1e-3)
        opt_state = optimizer.init(eqx.filter(self.vae, eqx.is_array))
        
        @eqx.filter_jit
        def train_step(model, x, opt_state, key):
            def batch_loss(m):
                losses, _ = jax.vmap(m.loss_fn, in_axes=(0, None))(x, key)
                return jnp.mean(losses)
            
            grads = eqx.filter_grad(batch_loss)(model)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, batch_loss(model)

        # Train for 50 steps
        model = self.vae
        initial_loss = 0.0
        final_loss = 0.0
        
        print("\nTraining BioVAE on Rossby-Haurwitz Data:")
        for i in range(201):
            k_step = jax.random.fold_in(self.key, i)
            model, opt_state, loss = train_step(model, self.cells_x, opt_state, k_step)
            if i == 0: initial_loss = loss
            if i == 50: final_loss = loss
            if i % 10 == 0:
                print(f"  Step {i}: Loss = {loss:.4f}")
                
        self.assertLess(final_loss, initial_loss * 0.8, "VAE failed to learn (Loss did not decrease significantly)")
        
        # Test Forecasting
        # Can we predict the future state of cell 0?
        traj = model.predict_trajectory(self.cells_x[0], self.key, steps=10)
        self.assertEqual(traj.shape, (11, 10), "Trajectory shape mismatch (Steps+1, GeneDim)")

if __name__ == '__main__':
    unittest.main()