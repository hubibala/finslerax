import jax
import jax.numpy as jnp
from ham.vis.isosurface import differentiable_marching_cubes, compute_analytical_normals

def test_marching_cubes_gradients():
    grid_extent = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    
    def loss_fn(volume):
        triangles, mask = differentiable_marching_cubes(volume, 0.5, grid_extent)
        # We penalize the sum of y-coordinates of the extracted surface
        return jnp.sum(triangles[..., 1] * mask[..., None])
        
    # Create a simple spherical distance field
    nx, ny, nz = 5, 5, 5
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    z = jnp.linspace(0, 1, nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # Distance from center
    volume = jnp.sqrt((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2)
    
    grad_volume = jax.grad(loss_fn)(volume)
    
    assert grad_volume.shape == volume.shape
    # Gradients should not be completely zero if the surface intersects the volume
    assert jnp.any(jnp.abs(grad_volume) > 1e-6)

def test_analytical_normals():
    grid_extent = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    nx, ny, nz = 5, 5, 5
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    z = jnp.linspace(0, 1, nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # Simple linear field in X: normals should be (1, 0, 0)
    volume = X
    
    triangles, mask = differentiable_marching_cubes(volume, 0.5, grid_extent)
    normals = compute_analytical_normals(triangles, volume, grid_extent)
    
    assert normals.shape == triangles.shape
    
    # Filter valid normals
    valid_normals = normals[mask]
    
    # They should all be close to (1, 0, 0)
    expected = jnp.array([1.0, 0.0, 0.0])
    diff = jnp.linalg.norm(valid_normals - expected, axis=-1)
    
    # Check max error is small
    if len(diff) > 0:
        assert jnp.max(diff) < 1e-5
