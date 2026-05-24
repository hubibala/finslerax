"""Integration tests for boundary-value problem (BVP) solvers on manifolds.

Verifies that AVBDSolver correctly finds energy-minimizing paths on various 
topologies (Torus, Sphere, Paraboloid) and respects physical invariants 
like Zermelo asymmetry.
"""

import unittest
import jax
import jax.numpy as jnp
from jax import config
import equinox as eqx

# Enforce High Precision
# config.update("jax_enable_x64", True)

from ham.solvers.avbd import AVBDSolver
from ham.geometry import Sphere, Torus, Paraboloid
from ham.geometry.zoo import Euclidean, Randers
from ham.geometry.manifolds.euclidean_space import EuclideanSpace
from ham.utils.math import safe_norm

class SolverTestBase(unittest.TestCase):
    """Base class for solver tests providing common utilities."""
    def setUp(self):
        # Professional defaults: 50 iterations is sufficient for these manifolds
        self.solver = AVBDSolver(
            step_size=0.05, 
            beta=1.5, 
            iterations=50, 
            energy_tol=1e-6
        )

    def assertBoundaryConditions(self, traj, start, end, tol=1e-5):
        """Verify that the trajectory strictly respects start and end points."""
        self.assertTrue(jnp.allclose(traj.xs[0], start, atol=tol), 
                        f"Start point mismatch: {traj.xs[0]} vs {start}")
        self.assertTrue(jnp.allclose(traj.xs[-1], end, atol=tol), 
                        f"End point mismatch: {traj.xs[-1]} vs {end}")

class TestEuclideanSolver(SolverTestBase):
    """Verifies the solver on flat space (the trivial baseline)."""
    
    def test_euclidean_straight_line(self):
        """Geodesic in R^n must be a straight line."""
        space = EuclideanSpace(dim=3)
        metric = Euclidean(space)
        
        start = jnp.array([0.0, 0.0, 0.0])
        end = jnp.array([1.0, 2.0, 3.0])
        n_steps = 10
        
        traj = self.solver.solve(metric, start, end, n_steps=n_steps)
        self.assertBoundaryConditions(traj, start, end)
        
        # Check linearity: x(t) = t * end
        t = jnp.linspace(0, 1, n_steps + 1)[:, None]
        expected_xs = t * end
        max_deviation = jnp.max(jnp.abs(traj.xs - expected_xs))
        
        self.assertLess(max_deviation, 1e-4, 
                        f"Euclidean path deviated from straight line by {max_deviation:.2e}")

class TestTorusSolver(SolverTestBase):
    """Verifies solver performance on non-trivial topologies (Torus)."""

    def test_torus_topology(self):
        """Geodesic on a Torus must wrap around the tube, not cut through the hole."""
        torus = Torus(major_R=2.0, minor_r=1.0)
        metric = Euclidean(torus)
        
        # Start (Outer Equator) -> End (Inner Equator)
        start = jnp.array([3.0, 0.0, 0.0]) 
        end   = jnp.array([1.0, 0.0, 0.0]) 
        
        traj = self.solver.solve(metric, start, end, n_steps=20)
        self.assertBoundaryConditions(traj, start, end)
        
        # 1. Check Manifold Constraint Satisfaction
        # (sqrt(x^2+y^2) - R)^2 + z^2 = r^2
        xy_norm = safe_norm(traj.xs[:, :2], axis=1)
        violation = jnp.abs((xy_norm - torus.R)**2 + traj.xs[:, 2]**2 - torus.r**2)
        max_err = jnp.max(violation)
        
        self.assertLess(max_err, 0.1, f"Torus constraint violation {max_err:.2e} exceeds 0.1")
        
        # 2. Check Topology (Did it wrap over the tube?)
        max_z = jnp.max(jnp.abs(traj.xs[:, 2]))
        self.assertGreater(max_z, 0.5, 
                           f"Path failed to wrap around the Torus tube (max_z={max_z:.2f})")

class TestSphereSolver(SolverTestBase):
    """Verifies solver on curved manifolds with asymmetric (Randers) metrics."""

    def test_sphere_zermelo(self):
        """Test Randers asymmetry: Downwind travel must be cheaper than upwind."""
        sphere = Sphere(radius=1.0)
        
        # Use eqx.nn.Lambda for JAX-friendly closure wrapping
        h_net = eqx.nn.Lambda(lambda x: jnp.eye(3))
        w_net = eqx.nn.Lambda(lambda x: jnp.array([0.5, 0.0, 0.0]))
        
        metric = Randers(sphere, h_net, w_net)
        
        # Points on Equator (analytical sphere points)
        p_west = jnp.array([-0.5, 0.0, 0.8660254037844386])
        p_east = jnp.array([ 0.5, 0.0, 0.8660254037844386])
        
        # Solve in both directions
        traj_down = self.solver.solve(metric, p_west, p_east, n_steps=15)
        traj_up   = self.solver.solve(metric, p_east, p_west, n_steps=15)
        
        # Verify finite and positive energy
        self.assertTrue(jnp.isfinite(traj_down.energy), "Downwind energy is not finite")
        self.assertTrue(jnp.isfinite(traj_up.energy), "Upwind energy is not finite")
        self.assertGreater(traj_down.energy, 0, "Downwind energy must be positive")
        
        # Core Randers Invariant: Downwind < Upwind
        self.assertLess(traj_down.energy, traj_up.energy, 
                        f"Randers asymmetry failed: down={traj_down.energy:.4f}, up={traj_up.energy:.4f}")

class TestParaboloidSolver(SolverTestBase):
    """Verifies implicit constraint handling and manifold projection."""

    def test_paraboloid_implicit(self):
        """Test constraint enforcement on Paraboloid z = x^2 + y^2."""
        para = Paraboloid()
        metric = Euclidean(para)
        
        # Define the implicit surface as an ALM constraint
        def para_c(x): return x[2] - (x[0]**2 + x[1]**2)
        
        start = jnp.array([-1.0, 0.0, 1.0])
        end   = jnp.array([ 1.0, 0.0, 1.0])
        
        # Test with training mode (scan)
        traj = self.solver.solve(metric, start, end, n_steps=20, constraints=[para_c])
        self.assertBoundaryConditions(traj, start, end)
        
        # Dynamic midpoint check
        mid_idx = len(traj.xs) // 2
        mid_z = traj.xs[mid_idx, 2]
        self.assertLess(mid_z, 0.2, f"Path did not dip to follow surface at midpoint (z={mid_z:.4f})")

    def test_inference_mode_convergence(self):
        """Verify that train_mode=False (while_loop) converges and respects boundaries."""
        para = Paraboloid()
        metric = Euclidean(para)
        start = jnp.array([-1.0, 0.0, 1.0])
        end   = jnp.array([ 1.0, 0.0, 1.0])
        
        traj = self.solver.solve(metric, start, end, n_steps=10, train_mode=False)
        self.assertBoundaryConditions(traj, start, end)
        self.assertTrue(jnp.isfinite(traj.energy))

if __name__ == '__main__':
    unittest.main()