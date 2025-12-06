import unittest
import jax
import jax.numpy as jnp
from jax import config
import numpy as np

# Use Double Precision for sensitive ODE tests
config.update("jax_enable_x64", True)

from ham.geometry import Sphere
from ham.geometry.zoo import Euclidean, Randers
from ham.solvers.geodesic import ExponentialMap
from ham.utils.math import safe_norm

class TestGeodesicSolver(unittest.TestCase):
    
    def setUp(self):
        # Increased resolution: 1000 steps prevents "projection drag"
        # 100 steps was causing ~1% energy loss on the Sphere.
        self.solver = ExponentialMap(step_size=0.002, max_steps=1000)
        self.key = jax.random.PRNGKey(42)

    def test_euclidean_ballistic(self):
        """In flat space, a geodesic is a straight line."""
        # Locally flat test
        class Plane:
            def project(self, x): return x
            def to_tangent(self, x, v): return v
            
        metric = Euclidean(Plane())
        x0 = jnp.array([0.0, 0.0])
        v0 = jnp.array([1.0, 0.5])
        
        x_final = self.solver.shoot(metric, x0, v0)
        expected = x0 + v0
        np.testing.assert_allclose(x_final, expected, atol=1e-5)

    def test_sphere_great_circle(self):
        """
        On a Unit Sphere:
        Start at (1, 0, 0)
        Velocity (0, 0, pi/2) -> Should reach North Pole (0, 0, 1) at t=1
        """
        sphere = Sphere(radius=1.0)
        metric = Euclidean(sphere)
        
        x0 = jnp.array([1.0, 0.0, 0.0])
        # Speed required to travel pi/2 in t=1
        speed = jnp.pi / 2.0
        v0 = jnp.array([0.0, 0.0, speed]) 
        
        x_final = self.solver.shoot(metric, x0, v0)
        
        # Check position
        expected = jnp.array([0.0, 0.0, 1.0])
        
        print(f"\n[Great Circle] Target: {expected}")
        print(f"[Great Circle] Arrival: {x_final}")
        
        np.testing.assert_allclose(x_final, expected, atol=1e-3)

    def test_energy_conservation_randers(self):
        """
        CRITICAL TEST: Energy E(x, v) is constant along a geodesic.
        """
        sphere = Sphere(1.0)
        # Swirl wind
        def w_net(x): return 0.5 * jnp.array([-x[1], x[0], 0.0])
        def h_net(x): return jnp.eye(3)
        
        metric = Randers(sphere, h_net, w_net)
        
        x0 = jnp.array([1.0, 0.0, 0.0])
        v0 = jnp.array([0.0, 0.5, 0.5])
        v0 = sphere.to_tangent(x0, v0)
        
        xs, vs = self.solver.trace(metric, x0, v0)
        energies = jax.vmap(metric.energy)(xs, vs)
        
        e_start = energies[0]
        e_end = energies[-1]
        max_deviation = jnp.max(jnp.abs(energies - e_start))
        
        print(f"\n[Randers Energy] Start: {e_start:.6f} | End: {e_end:.6f}")
        print(f"[Randers Energy] Max Dev: {max_deviation:.2e}")
        
        # Should be much tighter now (< 1e-4 is easy, likely < 1e-5)
        self.assertLess(max_deviation, 1e-4)

    def test_manifold_adherence(self):
        """Ensure the solver stays on the Sphere."""
        sphere = Sphere(1.0)
        metric = Euclidean(sphere)
        
        x0 = jnp.array([0.6, 0.8, 0.0])
        v0 = jnp.array([0.0, 0.0, 1.0])
        
        xs, _ = self.solver.trace(metric, x0, v0)
        radii = safe_norm(xs, axis=1)
        max_err = jnp.max(jnp.abs(radii - 1.0))
        
        print(f"\n[Manifold] Max Drift: {max_err:.2e}")
        self.assertLess(max_err, 1e-6)

if __name__ == '__main__':
    unittest.main()