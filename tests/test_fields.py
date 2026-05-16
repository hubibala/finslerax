import unittest
import jax
import jax.numpy as jnp
from jax import config
import numpy as np

# Use Double Precision
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
    
    def test_get_stream_function_flow(self):
        """Test basic equatorial rotation ψ = z."""
        def psi(x): return x[2]
        flow = get_stream_function_flow(psi)
        
        # At equator x=1, y=0, z=0: grad(psi)=[0,0,1], v = [0,0,1]x[1,0,0] = [0,1,0]
        x = jnp.array([1.0, 0.0, 0.0])
        v = flow(x)
        np.testing.assert_allclose(v, jnp.array([0.0, 1.0, 0.0]), atol=1e-7)
        
        # Tangency check
        self.assertAlmostEqual(float(jnp.dot(v, x)), 0.0)

    def test_tilted_rotation_jit(self):
        """Verify JIT and tilt angle for tilted_rotation."""
        # Tilted 90 degrees (rotation around X axis)
        flow = tilted_rotation(alpha_deg=90.0)
        jit_flow = jax.jit(flow)
        
        # Point on North Pole (0,0,1). Axis is (1,0,0).
        # v = (1,0,0) x (0,0,1) = (0,-1,0)
        x = jnp.array([0.0, 0.0, 1.0])
        v = jit_flow(x)
        np.testing.assert_allclose(v, jnp.array([0.0, -1.0, 0.0]), atol=1e-7)

    def test_rossby_haurwitz_vmap(self):
        """Verify Rossby-Haurwitz wave and vmap batching."""
        flow = rossby_haurwitz(R=4, omega=1.0, K=0.5)
        batched_flow = jax.vmap(flow)
        
        # Batch of points
        xs = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        vs = batched_flow(xs)
        
        self.assertEqual(vs.shape, (3, 3))
        self.assertTrue(jnp.all(jnp.isfinite(vs)))
        
        # Tangency check
        dots = jax.vmap(jnp.dot)(vs, xs)
        np.testing.assert_allclose(dots, jnp.zeros(3), atol=1e-7)

    def test_harmonic_vortices_grad(self):
        """Verify differentiability through harmonic vortices."""
        def loss(ell):
            flow = harmonic_vortices(ell=ell, m=3)
            v = flow(jnp.array([1.0, 0.0, 0.0]))
            return jnp.sum(v**2)
            
        grad_ell = jax.grad(loss)(5.0)
        self.assertTrue(jnp.isfinite(grad_ell))

    def test_vortex_3d_extension(self):
        """Test Lamb-Oseen and Rankine in 3D."""
        center = jnp.array([0.0, 0.0, 0.0])
        
        for factory in [lamb_oseen_vortex, rankine_vortex]:
            flow = factory(center, core_radius=1.0)
            # Pass 3D point
            x = jnp.array([1.0, 0.0, 5.0])
            v = flow(x)
            
            self.assertEqual(v.shape, (3,))
            self.assertEqual(v[2], 0.0) # Z-component must be zero
            self.assertNotEqual(v[1], 0.0) # Azimuthal velocity at (1,0) is in Y

if __name__ == '__main__':
    unittest.main()
