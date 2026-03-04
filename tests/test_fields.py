import unittest
import jax
import jax.numpy as jnp

from ham.sim.fields import (
    get_stream_function_flow,
    tilted_rotation,
    rossby_haurwitz,
    harmonic_vortices,
    lamb_oseen_vortex,
    rankine_vortex
)

class TestFields(unittest.TestCase):

    def test_stream_function_flow(self):
        # Psi(x,y,z) = z. Flow should be rotation around z-axis
        def psi(x):
            return x[2]
        flow_fn = get_stream_function_flow(psi)
        
        x = jnp.array([1.0, 0.0, 0.0]) # Point on equator
        v = flow_fn(x)
        
        # grad(z) = [0, 0, 1]
        # v = [0, 0, 1] x [1, 0, 0] = [0, 1, 0]
        self.assertTrue(jnp.allclose(v, jnp.array([0.0, 1.0, 0.0])))

    def test_tilted_rotation(self):
        flow_fn = tilted_rotation(alpha_deg=0.0) # Rotation around z
        x = jnp.array([1.0, 0.0, 0.0])
        v = flow_fn(x)
        self.assertTrue(jnp.allclose(v, jnp.array([0.0, 1.0, 0.0])))

    def test_lamb_oseen_vortex_2d(self):
        center = jnp.array([0.0, 0.0])
        flow_fn = lamb_oseen_vortex(center, core_radius=1.0, circulation=2*jnp.pi)
        
        # Test far field: should behave like point vortex v_theta = Gamma / (2*pi*r)
        # At r=10: v_theta ~ 1/10
        x_far = jnp.array([10.0, 0.0])
        v_far = flow_fn(x_far)
        self.assertAlmostEqual(v_far[1], 0.1, places=1) # v_y should be approx 0.1
        self.assertAlmostEqual(v_far[0], 0.0, places=5)
        
        # Near center, velocity should go to zero (unlike point vortex)
        x_near = jnp.array([1e-5, 0.0])
        v_near = flow_fn(x_near)
        self.assertTrue(jnp.linalg.norm(v_near) < 1e-3)

    def test_rankine_vortex_2d(self):
        center = jnp.array([0.0, 0.0])
        flow_fn = rankine_vortex(center, core_radius=2.0, circulation=4*jnp.pi)
        
        # Inside core (r <= 2.0), rigid body rotation v_theta = (Gamma / (2*pi*r_c^2)) * r
        # = (4*pi / (2*pi*4)) * r = 0.5 * r
        x_in = jnp.array([1.0, 0.0])
        v_in = flow_fn(x_in)
        self.assertAlmostEqual(v_in[1], 0.5, places=5)
        self.assertAlmostEqual(v_in[0], 0.0, places=5)
        
        # Outside core (r > 2.0), irrotational v_theta = Gamma / (2*pi*r)
        # At r=4: v_theta = 4*pi / (2*pi*4) = 0.5
        x_out = jnp.array([4.0, 0.0])
        v_out = flow_fn(x_out)
        self.assertAlmostEqual(v_out[1], 0.5, places=5)
        self.assertAlmostEqual(v_out[0], 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
