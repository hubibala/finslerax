"""
Differentiable Marching Cubes and Isosurface Extraction.

This module provides a fully JAX-native implementation of the Marching Cubes
algorithm designed specifically for gradient preservation. It reconstructs
isosurfaces from volumetric arrival time fields while ensuring that geometric
coordinates remain differentiable with respect to the input grid.
"""


import jax
import jax.numpy as jnp

from ham.vis.mc_table import TRIANGLE_TABLE


def _interpolate_edge(v1, v2, val1, val2, isovalue):
    """Linearly interpolate between v1 and v2 where the field equals isovalue."""
    # Ensure no division by zero
    diff = val2 - val1
    diff = jnp.where(jnp.abs(diff) < 1e-7, 1e-7, diff)
    t = (isovalue - val1) / diff
    return v1 + t * (v2 - v1)


def differentiable_marching_cubes(
    volume: jax.Array,
    isovalue: float,
    grid_extent: tuple[float, float, float, float, float, float],
):
    """
    Differentiable implementation of Marching Cubes over a 3D grid.

    Extracts a 3D isosurface from a continuous volumetric scalar field. By relying
    on exact linear interpolation along voxel edges, the spatial coordinates of
    extracted triangle vertices remain completely differentiable with respect to the
    scalar field values. This enables gradient flow from downstream surface-based
    penalties back into the grid values.

    Args:
        volume: Scalar field volume. Shape ``(nx, ny, nz)``.
        isovalue: The threshold scalar value defining the surface boundary.
        grid_extent: Bounding box limits ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

    Returns:
        A tuple ``(triangles, valid_mask)``:
        - **triangles**: Array of extracted triangle vertices. Shape ``(nx-1, ny-1, nz-1, 5, 3, 3)``.
        - **valid_mask**: Boolean mask indicating which of the 5 potential triangles per voxel are valid. Shape ``(nx-1, ny-1, nz-1, 5)``.
    """
    nx, ny, nz = volume.shape
    xmin, xmax, ymin, ymax, zmin, zmax = grid_extent

    x_coords = jnp.linspace(xmin, xmax, nx)
    y_coords = jnp.linspace(ymin, ymax, ny)
    z_coords = jnp.linspace(zmin, zmax, nz)
    X, Y, Z = jnp.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    # 8 corners of every voxel
    V0 = jnp.stack([X[:-1, :-1, :-1], Y[:-1, :-1, :-1], Z[:-1, :-1, :-1]], axis=-1)
    V1 = jnp.stack([X[1:, :-1, :-1], Y[1:, :-1, :-1], Z[1:, :-1, :-1]], axis=-1)
    V2 = jnp.stack([X[1:, 1:, :-1], Y[1:, 1:, :-1], Z[1:, 1:, :-1]], axis=-1)
    V3 = jnp.stack([X[:-1, 1:, :-1], Y[:-1, 1:, :-1], Z[:-1, 1:, :-1]], axis=-1)
    V4 = jnp.stack([X[:-1, :-1, 1:], Y[:-1, :-1, 1:], Z[:-1, :-1, 1:]], axis=-1)
    V5 = jnp.stack([X[1:, :-1, 1:], Y[1:, :-1, 1:], Z[1:, :-1, 1:]], axis=-1)
    V6 = jnp.stack([X[1:, 1:, 1:], Y[1:, 1:, 1:], Z[1:, 1:, 1:]], axis=-1)
    V7 = jnp.stack([X[:-1, 1:, 1:], Y[:-1, 1:, 1:], Z[:-1, 1:, 1:]], axis=-1)

    val0 = volume[:-1, :-1, :-1]
    val1 = volume[1:, :-1, :-1]
    val2 = volume[1:, 1:, :-1]
    val3 = volume[:-1, 1:, :-1]
    val4 = volume[:-1, :-1, 1:]
    val5 = volume[1:, :-1, 1:]
    val6 = volume[1:, 1:, 1:]
    val7 = volume[:-1, 1:, 1:]

    # Compute cubeindex (0-255)
    cubeindex = (
        (val0 <= isovalue).astype(jnp.int32)
        | ((val1 <= isovalue).astype(jnp.int32) << 1)
        | ((val2 <= isovalue).astype(jnp.int32) << 2)
        | ((val3 <= isovalue).astype(jnp.int32) << 3)
        | ((val4 <= isovalue).astype(jnp.int32) << 4)
        | ((val5 <= isovalue).astype(jnp.int32) << 5)
        | ((val6 <= isovalue).astype(jnp.int32) << 6)
        | ((val7 <= isovalue).astype(jnp.int32) << 7)
    )

    # Pre-calculate all 12 interpolated edge positions for all voxels
    # Edge 0: V0 to V1
    e0 = _interpolate_edge(V0, V1, val0[..., None], val1[..., None], isovalue)
    # Edge 1: V1 to V2
    e1 = _interpolate_edge(V1, V2, val1[..., None], val2[..., None], isovalue)
    # Edge 2: V2 to V3
    e2 = _interpolate_edge(V2, V3, val2[..., None], val3[..., None], isovalue)
    # Edge 3: V3 to V0
    e3 = _interpolate_edge(V3, V0, val3[..., None], val0[..., None], isovalue)
    # Edge 4: V4 to V5
    e4 = _interpolate_edge(V4, V5, val4[..., None], val5[..., None], isovalue)
    # Edge 5: V5 to V6
    e5 = _interpolate_edge(V5, V6, val5[..., None], val6[..., None], isovalue)
    # Edge 6: V6 to V7
    e6 = _interpolate_edge(V6, V7, val6[..., None], val7[..., None], isovalue)
    # Edge 7: V7 to V4
    e7 = _interpolate_edge(V7, V4, val7[..., None], val4[..., None], isovalue)
    # Edge 8: V0 to V4
    e8 = _interpolate_edge(V0, V4, val0[..., None], val4[..., None], isovalue)
    # Edge 9: V1 to V5
    e9 = _interpolate_edge(V1, V5, val1[..., None], val5[..., None], isovalue)
    # Edge 10: V2 to V6
    e10 = _interpolate_edge(V2, V6, val2[..., None], val6[..., None], isovalue)
    # Edge 11: V3 to V7
    e11 = _interpolate_edge(V3, V7, val3[..., None], val7[..., None], isovalue)

    # edges shape: (nx-1, ny-1, nz-1, 12, 3)
    edges = jnp.stack([e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11], axis=-2)

    # lookup shape: (nx-1, ny-1, nz-1, 16)
    tri_indices = TRIANGLE_TABLE[cubeindex]

    # Reshape to (nx-1, ny-1, nz-1, 5, 3) to represent up to 5 triangles (15 vertices)
    tri_indices = tri_indices[..., :15].reshape((nx - 1, ny - 1, nz - 1, 5, 3))

    # Gather edge vertices based on tri_indices
    # Valid mask: if an index is -1, it's not a valid triangle
    valid_mask = jnp.all(tri_indices != -1, axis=-1)

    # Replace -1 with 0 so we can safely index (we use valid_mask to filter later)
    safe_indices = jnp.where(tri_indices == -1, 0, tri_indices)

    # Map edge indices to actual edge coordinate arrays via a per-voxel gather.
    def gather_edges(edges_b, indices_b):
        return edges_b[indices_b]

    gather_edges_vmap = jax.vmap(jax.vmap(jax.vmap(gather_edges)))
    triangles = gather_edges_vmap(edges, safe_indices)

    # triangles shape: (nx-1, ny-1, nz-1, 5, 3, 3) where last dims are (num_verts, coords)

    return triangles, valid_mask


def compute_analytical_normals(
    triangles: jax.Array,
    volume: jax.Array,
    grid_extent: tuple[float, float, float, float, float, float],
):
    """
    Computes precise analytical normals for the extracted isosurface.

    Instead of computing discrete facet normals via cross products of the extracted
    mesh, this function evaluates the spatial gradient $\\nabla T(x)$ of the underlying
    scalar volume directly. The gradients are trilinearly interpolated to the precise
    sub-voxel coordinates of the isosurface vertices.

    This guarantees $C^1$ smooth, continuous surface normals suitable for high-fidelity
    rendering or downstream geometric modeling operations (like generating wind fields).

    Args:
        triangles: Extracted triangle vertices. Shape ``(..., 3, 3)``.
        volume: Scalar field volume representing distance or arrival time. Shape ``(nx, ny, nz)``.
        grid_extent: Bounding box limits ``(xmin, xmax, ymin, ymax, zmin, zmax)``.

    Returns:
        normals: Analytical $L_2$-normalized surface normals at each vertex. Shape matches ``triangles``.
    """
    nx, ny, nz = volume.shape
    xmin, xmax, ymin, ymax, zmin, zmax = grid_extent
    hx = (xmax - xmin) / (nx - 1)
    hy = (ymax - ymin) / (ny - 1)
    hz = (zmax - zmin) / (nz - 1)

    # Compute central differences for the entire volume
    grad_x = jnp.gradient(volume, hx, axis=0)
    grad_y = jnp.gradient(volume, hy, axis=1)
    grad_z = jnp.gradient(volume, hz, axis=2)
    grad_field = jnp.stack([grad_x, grad_y, grad_z], axis=-1)  # type: ignore[list-item]

    # Map triangle coordinates to grid indices (continuous)
    # triangles: (..., 5, 3, 3)
    grid_coords_x = (triangles[..., 0] - xmin) / hx
    grid_coords_y = (triangles[..., 1] - ymin) / hy
    grid_coords_z = (triangles[..., 2] - zmin) / hz

    # Trilinear interpolation of the gradient field is possible using jax.scipy.ndimage.map_coordinates
    # Since we need to interpolate the 3 channels separately:
    grid_coords = jnp.stack([grid_coords_x, grid_coords_y, grid_coords_z], axis=0)

    def interpolate_channel(chan):
        return jax.scipy.ndimage.map_coordinates(
            chan, grid_coords, order=1, mode="nearest"
        )

    normals_x = interpolate_channel(grad_field[..., 0])
    normals_y = interpolate_channel(grad_field[..., 1])
    normals_z = interpolate_channel(grad_field[..., 2])

    normals = jnp.stack([normals_x, normals_y, normals_z], axis=-1)

    # Normalize
    norm = jnp.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / jnp.maximum(norm, 1e-12)

    return normals
