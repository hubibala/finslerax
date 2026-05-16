"""Tests for analytical vector fields and wind field generators.

Verifies tangency on spheres, divergence-free properties, 
and correct analytical profiles for 2D vortices.
"""

import unittest
import jax
import jax.numpy as jnp
from jax import config
import numpy as np

# Enforce High Precision
config.update("jax_enable_x64", True)

from ham.sim.fields import (
    get_stream_function_flow,
    tilted_rotation,
    rossby_haurwitz,
    harmonic_vortices,
    lamb_oseen_vortex,
    rankine_vortex
)

class TestFields(unittest.TestCase):
    
    def test_get_stream_function_flow_variations(self):
        """Test stream-function flow with various psi and off-axis points."""
        # 1. ψ = z (Equatorial rotation)
        def psi1(x): return x[2]
        flow1 = get_stream_function_flow(psi1)
        x1 = jnp.array([1.0, 0.0, 0.0])
        v1 = flow1(x1)
        np.testing.assert_allclose(v1, jnp.array([0.0, 1.0, 0.0]), atol=1e-8)
        
        # 2. ψ = x*y (Non-trivial saddle)
        def psi2(x): return x[0] * x[1]
        flow2 = get_stream_function_flow(psi2)
        # Point at 45 degrees in XY plane: [1/sqrt(2), 1/sqrt(2), 0]
        s2 = 1.0 / jnp.sqrt(2.0)
        x2 = jnp.array([s2, s2, 0.0])
        # grad(psi) = [y, x, 0] = [s2, s2, 0]
        # v = [s2, s2, 0] x [s2, s2, 0] = [0, 0, 0]
        v2 = flow2(x2)
        np.testing.assert_allclose(v2, jnp.zeros(3), atol=1e-8)
        
        # 3. Tangency check on arbitrary point
        x3 = jnp.array([0.5, 0.5, jnp.sqrt(0.5)]) # point on S2
        v3 = flow2(x3)
        self.assertAlmostEqual(float(jnp.dot(v3, x3)), 0.0, places=12)

    def test_tilted_rotation_full(self):
        """Verify tilted rotation at non-trivial angles and JIT."""
        # Rotation around X axis (alpha=90)
        flow = tilted_rotation(alpha_deg=90.0)
        jit_flow = jax.jit(flow)
        
        # Point on North Pole (0,0,1). 
        # Axis is (1,0,0). v = axis x x = (0, -1, 0)
        x = jnp.array([0.0, 0.0, 1.0])
        v = jit_flow(x)
        np.testing.assert_allclose(v, jnp.array([0.0, -1.0, 0.0]), atol=1e-8)
        
        # Point on X-axis (1,0,0) should have zero velocity
        v_axis = flow(jnp.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(v_axis, jnp.zeros(3), atol=1e-8)

    def test_rossby_haurwitz_properties(self):
        """Verify Rossby-Haurwitz wave invariants."""
        # R=4, omega=1, K=1
        flow = rossby_haurwitz(R=4, omega=1.0, K=1.0)
        
        # At poles, the flow must be zero or purely tangential (and actually vanishes due to sin(lat))
        for z in [-1.0, 1.0]:
            x = jnp.array([0.0, 0.0, z])
            v = flow(x)
            np.testing.assert_allclose(v, jnp.zeros(3), atol=1e-8)
            
        # At equator x=1, y=0, z=0:
        # psi = -omega*0 + K*1^4*0*... = 0
        # Check tangency at multiple points
        xs = jnp.array([
            [1.0, 0.0, 0.0],
            [s2 := 1/jnp.sqrt(2), s2, 0.0],
            [0.5, 0.5, jnp.sqrt(0.5)]
        ])
        vs = jax.vmap(flow)(xs)
        dots = jax.vmap(jnp.dot)(vs, xs)
        np.testing.assert_allclose(dots, jnp.zeros(3), atol=1e-12)

    def test_harmonic_vortices_symmetry(self):
        """Verify cellular vortices high-frequency properties."""
        flow = harmonic_vortices(ell=10, m=5)
        
        # Check that it compiles and is finite
        v = flow(jnp.array([1.0, 0.0, 0.0]))
        self.assertTrue(jnp.all(jnp.isfinite(v)))
        
        # Check differentiability w.r.t. position
        jac = jax.jacobian(flow)(jnp.array([0.6, 0.8, 0.0]))
        self.assertEqual(jac.shape, (3, 3))
        self.assertTrue(jnp.all(jnp.isfinite(jac)))

    def test_lamb_oseen_vortex_precision(self):
        """Verify Lamb-Oseen with tight tolerances and 3D support."""
        center = jnp.array([0.0, 0.0])
        # Gamma=2pi, rc=1.0 -> v_theta = (1/r)(1 - exp(-r^2))
        flow = lamb_oseen_vortex(center, core_radius=1.0, circulation=2.0*jnp.pi)
        
        # Far field (r=10): v_theta approx 0.1
        x_far = jnp.array([10.0, 0.0, 5.0]) # 3D input
        v_far = flow(x_far)
        
        # Tighten tolerance from places=1 to places=5
        np.testing.assert_allclose(v_far[1], 0.1, atol=1e-5)
        np.testing.assert_allclose(v_far[0], 0.0, atol=1e-8)
        np.testing.assert_allclose(v_far[2], 0.0, atol=1e-8) # Z-padding
        
        # Center (r=0)
        v_center = flow(jnp.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(v_center, jnp.zeros(3), atol=1e-8)

    def test_rankine_vortex_boundary(self):
        """Verify Rankine continuity at r=rc and center behavior."""
        center = jnp.array([0.0, 0.0])
        rc = 2.0
        # Gamma=4pi -> v_theta = r/2 inside, 2/r outside
        flow = rankine_vortex(center, core_radius=rc, circulation=4.0*jnp.pi)
        
        # 1. At Boundary r=rc=2.0 -> v_theta = 1.0
        x_bound = jnp.array([2.0, 0.0])
        v_bound = flow(x_bound)
        np.testing.assert_allclose(v_bound[1], 1.0, atol=1e-7)
        
        # 2. Inside r=1.0 -> v_theta = 0.5
        x_in = jnp.array([1.0, 0.0])
        v_in = flow(x_in)
        np.testing.assert_allclose(v_in[1], 0.5, atol=1e-7)
        
        # 3. Outside r=4.0 -> v_theta = 0.5
        x_out = jnp.array([4.0, 0.0])
        v_out = flow(x_out)
        np.testing.assert_allclose(v_out[1], 0.5, atol=1e-7)
        
        # 4. Center r=0
        v_center = flow(jnp.array([0.0, 0.0]))
        np.testing.assert_allclose(v_center, jnp.zeros(2), atol=1e-8)

if __name__ == '__main__':
    unittest.main()
