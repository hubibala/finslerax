import unittest
import jax
import jax.numpy as jnp
from jax import config
import numpy as np

# Use Double Precision for sensitive ODE tests
config.update("jax_enable_x64", True)

from ham.geometry import Sphere
from ham.geometry.zoo import Euclidean, Randers, Riemannian
from ham.solvers.geodesic import ExponentialMap
from ham.utils.math import safe_norm

class TestGeodesicSolver(unittest.TestCase):
    
    def setUp(self):
        # Default resolution for tests
        self.solver = ExponentialMap(step_size=0.002, max_steps=1000)
        self.key = jax.random.PRNGKey(42)

    def test_euclidean_ballistic(self):
        """In flat space, a geodesic is a straight line."""
        # Locally flat test
        from ham.geometry.manifold import Manifold
        class Plane(Manifold):
            @property
            def ambient_dim(self) -> int: return 2
            @property
            def intrinsic_dim(self) -> int: return 2
            def project(self, x): return x
            def to_tangent(self, x, v): return v
            def retract(self, x, v): return x + v
            def random_sample(self, key, shape): return jnp.zeros(shape + (2,))
            
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
        expected = jnp.array([0.0, 0.0, 1.0])
        
        np.testing.assert_allclose(x_final, expected, atol=1e-3)

    def test_energy_conservation_randers(self):
        """CRITICAL TEST: Energy E(x, v) is constant along a geodesic."""
        sphere = Sphere(radius=1.0)
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
        max_deviation = jnp.max(jnp.abs(energies - e_start))
        
        # Should be much tighter now with intermediate projections
        self.assertLess(max_deviation, 5e-3)

    def test_manifold_adherence(self):
        """Ensure the solver stays on the Sphere."""
        sphere = Sphere(radius=1.0)
        metric = Euclidean(sphere)
        
        x0 = jnp.array([0.6, 0.8, 0.0])
        v0 = jnp.array([0.0, 0.0, 1.0])
        
        xs, _ = self.solver.trace(metric, x0, v0)
        radii = safe_norm(xs, axis=1)
        max_err = jnp.max(jnp.abs(radii - 1.0))
        
        self.assertLess(max_err, 1e-6)

    def test_zero_velocity(self):
        """A geodesic with zero initial velocity should stay at x0."""
        sphere = Sphere(radius=1.0)
        metric = Euclidean(sphere)
        x0 = jnp.array([1.0, 0.0, 0.0])
        v0 = jnp.zeros(3)
        xf = self.solver.shoot(metric, x0, v0)
        np.testing.assert_allclose(xf, x0, atol=1e-5)

    def test_differentiability(self):
        """Verify we can differentiate through the solver."""
        sphere = Sphere(radius=1.0)
        target = jnp.array([0.0, 0.0, 1.0])
        
        def loss(wind_speed):
            # A Randers metric with variable wind
            def w_net(x): 
                return wind_speed * jnp.array([-x[1], x[0], 0.0])
            def h_net(x): 
                return jnp.eye(3)
            
            metric = Randers(sphere, h_net, w_net)
            x0 = jnp.array([1.0, 0.0, 0.0])
            v0 = jnp.array([0.0, 0.5, 0.5])
            v0 = sphere.to_tangent(x0, v0)
            
            # Shoot endpoint
            xf = self.solver.shoot(metric, x0, v0)
            return jnp.sum((xf - target)**2)
        
        # Use a non-zero start for params
        grad = jax.grad(loss)(0.5)
        self.assertTrue(jnp.all(jnp.isfinite(grad)))
        self.assertNotEqual(float(jnp.sum(jnp.abs(grad))), 0.0)

    def test_high_velocity_clamping(self):
        """Verify that high velocity clamping works and is configurable."""
        # Set a tight clamp
        solver = ExponentialMap(max_velocity=1.0, max_steps=100)
        sphere = Sphere(radius=1.0)
        metric = Euclidean(sphere)
        x0 = jnp.array([1.0, 0.0, 0.0])
        # Velocity that would travel pi units if not clamped
        v0 = jnp.array([0.0, 0.0, jnp.pi]) 
        
        xf = solver.shoot(metric, x0, v0)
        # With speed clamp=1.0 and t_max=1.0, it should travel ~1.0 radian
        dist = jnp.arccos(jnp.clip(jnp.dot(x0, xf), -1.0, 1.0))
        self.assertLess(dist, 1.5) 
        self.assertGreater(dist, 0.5)

if __name__ == '__main__':
    unittest.main()