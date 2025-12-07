import jax
import jax.numpy as jnp
import unittest
import equinox as eqx
from ham.geometry.manifold import Manifold
from ham.models.learned import NeuralRanders

class MockManifold(Manifold):
    """Trivial R^3 manifold for testing."""
    @property
    def ambient_dim(self): return 3
    @property
    def intrinsic_dim(self): return 3
    def project(self, x): return x
    def to_tangent(self, x, v): return v
    def random_sample(self, key, shape): 
        return jax.random.normal(key, shape + (3,))

class TestNeuralRanders(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(0)
        self.manifold = MockManifold()
        # Initialize Neural Metric
        self.metric = NeuralRanders(self.manifold, self.key, hidden_dim=16)

    def test_zermelo_convexity_enforcement(self):
        """
        Safety Check: Even if the network predicts a massive wind W,
        the metric logic must squash it to |W| < 1 to prevent negative energy.
        """
        x = jnp.zeros(3)
        
        # We manually inject huge weights into the wind network to simulate explosion
        # This is a bit of a hack to test the robustness wrapper
        huge_w_net = eqx.tree_at(
            lambda m: m.mlp.layers[-1].weight,
            self.metric.w_net,
            jnp.ones_like(self.metric.w_net.mlp.layers[-1].weight) * 1000.0
        )
        broken_metric = eqx.tree_at(lambda m: m.w_net, self.metric, huge_w_net)
        
        # Check internal Zermelo data
        H, W, lam = broken_metric._get_zermelo_data(x)
        
        # Calculate norm of W in H
        w_norm = jnp.sqrt(jnp.dot(W, jnp.dot(H, W)))
        
        print(f"\nClamped Wind Norm: {w_norm}")
        self.assertLess(w_norm, 1.0, "Wind vector violated convexity constraint!")
        self.assertGreater(lam, 0.0, "Lambda became non-positive!")

    def test_gradients_exist(self):
        """
        The metric must be differentiable w.r.t input x (for Geodesics) 
        and w.r.t parameters (for Learning).
        """
        x = jnp.array([0.5, 0.5, 0.5])
        v = jnp.array([1.0, 0.0, 0.0])
        
        # 1. Grad w.r.t Input (Needed for Solver)
        grad_x = jax.grad(self.metric.energy, argnums=0)(x, v)
        self.assertTrue(jnp.isfinite(grad_x).all())
        
        # 2. Grad w.r.t Weights (Needed for Training)
        def loss(m):
            return m.energy(x, v)
            
        grads = eqx.filter_grad(loss)(self.metric)
        
        # Check that gradients propagate to H and W nets
        # We check the first layer weight gradient
        w_grad = grads.w_net.mlp.layers[0].weight
        self.assertFalse(jnp.all(w_grad == 0), "No gradient flow to Wind Network")

if __name__ == '__main__':
    unittest.main()