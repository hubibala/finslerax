import jax
import jax.numpy as jnp
import unittest
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from ham.geometry.mesh import TriangularMesh

class TestMeshManifold(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(42)

    def test_standard_3d_tetrahedron(self):
        """Standard 3D test case."""
        verts = jnp.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
        faces = jnp.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
        mesh = TriangularMesh(verts, faces)
        
        # Project off-surface point
        p = jnp.array([0.2, 0.2, -0.5])
        proj = mesh.project(p)
        expected = jnp.array([0.2, 0.2, 0.0])
        np.testing.assert_allclose(proj, expected, atol=1e-5)
        
        # Tangent vector projection
        x = jnp.array([0.2, 0.2, 0.0]) # On XY face
        v = jnp.array([1.0, 1.0, 1.0]) # Has Z component
        v_tan = mesh.to_tangent(x, v)
        # Should remove Z component
        np.testing.assert_allclose(v_tan, jnp.array([1.0, 1.0, 0.0]), atol=1e-5)

    def test_high_dim_embedding(self):
        """Verify mesh logic works for a 2D triangle embedded in 4D."""
        verts = jnp.array([
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]
        ])
        faces = jnp.array([[0, 1, 2]])
        mesh = TriangularMesh(verts, faces)
        
        self.assertEqual(mesh.ambient_dim, 4)
        self.assertEqual(mesh.intrinsic_dim, 2)
        
        # 1. Project point
        p_off = jnp.array([0.2, 0.2, 0.0, 10.0])
        proj = mesh.project(p_off)
        expected = jnp.array([0.2, 0.2, 0.0, 0.0])
        np.testing.assert_allclose(proj, expected, atol=1e-5)
        
        # 2. Tangent Projection
        v = jnp.array([1.0, 1.0, 1.0, 1.0])
        v_tan = mesh.to_tangent(expected, v)
        np.testing.assert_allclose(v_tan, jnp.array([1.0, 1.0, 0.0, 0.0]), atol=1e-5)
        
        # 3. Sampling
        samples = mesh.random_sample(self.key, (10,))
        self.assertEqual(samples.shape, (10, 4))
        np.testing.assert_allclose(samples[:, 2:], 0.0, atol=1e-5)

    def test_get_face_index(self):
        """Verify face indexing."""
        verts = jnp.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
        faces = jnp.array([[0,1,2], [0,1,3]]) # XY plane and XZ plane
        mesh = TriangularMesh(verts, faces)
        
        # Point near XY face
        p1 = jnp.array([0.1, 0.1, 0.01])
        self.assertEqual(mesh.get_face_index(p1), 0)
        
        # Point near XZ face
        p2 = jnp.array([0.1, 0.01, 0.1])
        self.assertEqual(mesh.get_face_index(p2), 1)

    def test_get_face_weights_differentiability(self):
        """Verify weights are differentiable."""
        verts = jnp.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
        faces = jnp.array([[0,1,2], [0,1,3]])
        mesh = TriangularMesh(verts, faces)
        
        def loss_fn(x):
            weights = mesh.get_face_weights(x, temperature=1.0)
            return jnp.sum(weights**2)
        
        p = jnp.array([0.1, 0.1, 0.1])
        grad = jax.grad(loss_fn)(p)
        self.assertTrue(jnp.all(jnp.isfinite(grad)))

    def test_retract(self):
        """Verify retraction moves point to surface."""
        verts = jnp.array([[0,0,0], [1,0,0], [0,1,0]], dtype=float)
        faces = jnp.array([[0,1,2]])
        mesh = TriangularMesh(verts, faces)
        
        x = jnp.array([0.5, 0.5, 0.0])
        delta = jnp.array([0.1, 0.1, 1.0])
        retracted = mesh.retract(x, delta)
        
        # Result should be on surface (z=0)
        np.testing.assert_allclose(retracted[2], 0.0, atol=1e-5)
        # Should match projected x+delta
        np.testing.assert_allclose(retracted, jnp.array([0.5, 0.5, 0.0]), atol=1e-5)

    def test_degenerate_triangle(self):
        """Verify stability with degenerate triangles."""
        # Triangle with three points on a line
        verts = jnp.array([[0,0,0], [1,0,0], [2,0,0]], dtype=float)
        faces = jnp.array([[0,1,2]])
        mesh = TriangularMesh(verts, faces)
        
        # Project should still work (fallback to edge)
        p = jnp.array([0.5, 1.0, 0.0])
        proj = mesh.project(p)
        np.testing.assert_allclose(proj, jnp.array([0.5, 0.0, 0.0]), atol=1e-5)
        
        # to_tangent should return zero vector (as guarded)
        v = jnp.array([1.0, 1.0, 1.0])
        v_tan = mesh.to_tangent(proj, v)
        np.testing.assert_allclose(v_tan, 0.0, atol=1e-5)

    def test_random_sample_jit_compatible(self):
        """Verify random_sample doesn't crash (even if not JIT'd itself, check n logic)."""
        verts = jnp.array([[0,0,0], [1,0,0], [0,1,0]], dtype=float)
        faces = jnp.array([[0,1,2]])
        mesh = TriangularMesh(verts, faces)
        
        # Should work with multi-dim shape
        samples = mesh.random_sample(self.key, (2, 3))
        self.assertEqual(samples.shape, (2, 3, 3))

if __name__ == '__main__':
    unittest.main()