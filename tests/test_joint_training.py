import jax
import jax.numpy as jnp
import unittest
import equinox as eqx
import optax

from ham.geometry import Sphere
from ham.models.learned import NeuralRanders
from ham.bio.vae import GeometricVAE

class TestJointTraining(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(2025)
        self.manifold = Sphere(radius=1.0)
        
        # Initialize Learnable Metric
        self.neural_metric = NeuralRanders(self.manifold, self.key)
        
        # Initialize VAE
        self.vae = GeometricVAE(
            data_dim=5, 
            latent_dim=3, 
            metric=self.neural_metric, 
            key=self.key
        )
        
        # Dummy Data
        self.data = jax.random.normal(self.key, (10, 5))

    def test_metric_is_updated(self):
        """
        Verify that a training step actually updates the NeuralRanders parameters.
        """
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(self.vae, eqx.is_array))
        
        # Snapshot weights before
        w_before = self.vae.metric.w_net.mlp.layers[0].weight
        h_before = self.vae.metric.h_net.mlp.layers[0].weight
        
        @eqx.filter_jit
        def step(model, state):
            grads = eqx.filter_grad(lambda m: jnp.mean(jax.vmap(m.loss_fn, in_axes=(0,None))(self.data, self.key)[0]))(model)
            updates, new_state = optimizer.update(grads, state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_state
            
        # Run one step
        vae_new, _ = step(self.vae, opt_state)
        
        # Snapshot weights after
        w_after = vae_new.metric.w_net.mlp.layers[0].weight
        h_after = vae_new.metric.h_net.mlp.layers[0].weight
        
        # Check diff
        w_diff = jnp.sum(jnp.abs(w_after - w_before))
        h_diff = jnp.sum(jnp.abs(h_after - h_before))
        
        print(f"\nWeight updates -- Wind: {w_diff:.2e}, Metric: {h_diff:.2e}")
        
        self.assertGreater(w_diff, 0.0, "Wind network weights did not change!")
        self.assertGreater(h_diff, 0.0, "Metric tensor weights did not change!")

if __name__ == '__main__':
    unittest.main()