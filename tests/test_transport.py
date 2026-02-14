import jax
import jax.numpy as jnp
import unittest
import numpy as np
from jax import config

# Ensure precision for geometric drift checks
config.update("jax_enable_x64", True)

from ham.geometry.manifold import Manifold
from ham.geometry.zoo import Euclidean, Riemannian, Randers
from ham.geometry.transport import berwald_transport

# --- Mock Manifolds ---
class FlatPlane(Manifold):
    @property
    def ambient_dim(self): return 2
    @property
    def intrinsic_dim(self): return 2
    def project(self, x): return x
    def to_tangent(self, x, v): return v
    def retract(self, x, delta): return x + delta
    def random_sample(self, key, shape): 
        return jax.random.normal(key, shape + (2,))

class Sphere(Manifold):
    @property
    def ambient_dim(self): return 3
    @property
    def intrinsic_dim(self): return 2
    def project(self, x): return x / jnp.linalg.norm(x)
    def to_tangent(self, x, v):
        x_unit = x / jnp.linalg.norm(x)
        return v - jnp.dot(v, x_unit) * x_unit
    def retract(self, x, delta): return x + delta
    def random_sample(self, key, shape):
        x = jax.random.normal(key, shape + (3,))
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

class TestTransport(unittest.TestCase):
    
    def setUp(self):
        self.plane = FlatPlane()
        self.sphere = Sphere()
        self.key = jax.random.PRNGKey(42)

    # --- 1. Euclidean Transport (Baseline) ---
    def test_euclidean_flat_invariance(self):
        """
        In flat Euclidean space, parallel transport is trivial.
        A vector transported along any path remains constant in coordinates.
        """
        metric = Euclidean(self.plane)
        
        # Path: Circle in the plane
        t = jnp.linspace(0, 2*jnp.pi, 50)
        path_x = jnp.stack([jnp.cos(t), jnp.sin(t)], axis=1)
        path_v = jnp.stack([-jnp.sin(t), jnp.cos(t)], axis=1) # Velocity
        
        # Vector: Constant X=[1, 0]
        vec_start = jnp.array([1.0, 0.0])
        
        vecs = berwald_transport(metric, path_x, path_v, vec_start)
        
        # Check: Every vector in the sequence should be [1, 0]
        expected = jnp.tile(vec_start, (len(path_x), 1))
        np.testing.assert_allclose(vecs, expected, atol=1e-5)

    # --- 2. Riemannian Transport (Curvature Check) ---
    def test_riemannian_sphere_isometry(self):
        """
        Riemannian transport on a Sphere (Levi-Civita).
        MUST preserve the norm (Isometry).
        """
        # We use the generic Riemannian class to ensure it works, 
        # not just the Euclidean subclass
        def sphere_metric(x):
            # Induced metric on S^2 is effectively identity in embedding 
            # (handled by projection constraints)
            return jnp.eye(3)
            
        metric = Riemannian(self.sphere, sphere_metric)
        
        # Path: Quarter circle (North Pole -> Equator)
        theta = jnp.linspace(0, jnp.pi/2, 20)
        path_x = jnp.stack([jnp.sin(theta), jnp.zeros_like(theta), jnp.cos(theta)], axis=1)
        path_v = jnp.stack([jnp.cos(theta), jnp.zeros_like(theta), -jnp.sin(theta)], axis=1)
        
        # Transport tangent vector (pointing in Y)
        vec_start = jnp.array([0.0, 1.0, 0.0])
        
        vecs = berwald_transport(metric, path_x, path_v, vec_start)
        
        # 1. Norm Preservation (The critical Riemannian check)
        norms = jnp.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, jnp.ones_like(norms), atol=1e-5)
        
        # 2. Tangency
        dots = jnp.sum(vecs * path_x, axis=1)
        np.testing.assert_allclose(dots, jnp.zeros_like(dots), atol=1e-5)

    # --- 3. Randers Transport (The Non-Metric Check) ---
    def test_randers_norm_drift(self):
        """
        Verifies that Berwald transport in a Randers space is NOT an isometry.
        
        Scenario:
            - Flat Plane.
            - "Shear Wind": W = (0.5 * y, 0). Wind increases as we go up.
            - Path: Vertical line from (0,0) to (0,1).
            - We transport a vector X that points 'downstream' (Right).
            
        Physics:
            As the wind increases, the "cost" of moving against/with it changes.
            The Berwald connection tracks the affine flow, but the Metric Norm 
            (energy cost) of that vector changes because the background flow changed.
            
        Expected:
            norm(final_vector) != norm(initial_vector)
        """
        h_net = lambda x: jnp.eye(2)
        # Shear wind: Strength depends on Y coordinate
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0])
        
        metric = Randers(self.plane, h_net, w_net)
        
        # Path: Vertical line (0,0) -> (0,1)
        y = jnp.linspace(0, 1, 20)
        path_x = jnp.stack([jnp.zeros_like(y), y], axis=1)
        path_v = jnp.stack([jnp.zeros_like(y), jnp.ones_like(y)], axis=1)
        
        # Transport a vector pointing Right (into the wind direction)
        vec_start = jnp.array([1.0, 0.0])
        
        # 1. Run Randers Transport
        vecs_randers = berwald_transport(metric, path_x, path_v, vec_start)
        
        # Calculate Finsler Norms of the transported vectors
        # We need to map metric.metric_fn over (x, transported_v)
        norms_randers = jax.vmap(metric.metric_fn)(path_x, vecs_randers)
        
        # 2. Assert Drift
        # The norm should change because the wind W(x) changes along the path
        initial_norm = norms_randers[0]
        final_norm = norms_randers[-1]
        
        print(f"Randers Norm Drift: {initial_norm:.4f} -> {final_norm:.4f}")
        
        # It should strictly drift
        self.assertNotAlmostEqual(initial_norm, final_norm, places=3, 
                                  msg="Randers Berwald transport should NOT preserve norm")

    def test_randers_velocity_dependence(self):
        """
        Verify that Randers connection depends on velocity (unlike Riemannian).
        Transport same vector along same path, but with different path speed.
        """
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5 * x[1], 0.0]) # Shear wind
        metric = Randers(self.plane, h_net, w_net)

        # Path Geometry: Vertical line
        vec_start = jnp.array([1.0, 0.0])
        
        # Case 1: Slow Speed
        y1 = jnp.linspace(0, 1, 20)
        path_x1 = jnp.stack([jnp.zeros_like(y1), y1], axis=1)
        path_v1 = jnp.stack([jnp.zeros_like(y1), jnp.ones_like(y1)], axis=1)
        
        # Case 2: Fast Speed (2x)
        # We traverse the same geometry, but v is doubled.
        # In Riemannian, Gamma(x) is constant, so transport is same.
        # In Randers, Gamma(x, v) depends on v.
        path_v2 = path_v1 * 2.0
        
        vecs_1 = berwald_transport(metric, path_x1, path_v1, vec_start)
        vecs_2 = berwald_transport(metric, path_x1, path_v2, vec_start)
        
        # Compare final vectors
        diff = jnp.linalg.norm(vecs_1[-1] - vecs_2[-1])
        print(f"Velocity Dependence Diff: {diff:.4f}")
        
        self.assertGreater(diff, 1e-4, 
                           "Randers transport should depend on the speed of the base curve")

if __name__ == '__main__':
    unittest.main()