"""Triangular mesh manifold for discrete surfaces in R^N.

Provides the TriangularMesh class, which implements the Manifold ABC
for piecewise-linear surfaces defined by vertex/face arrays. Used by
DiscreteRanders (zoo.py) for anisotropic mesh-based metrics.

See also: spec/ARCH_SPEC.md § 5 (Module Structure).
"""

from ham.utils.config import DEFAULT_JNP_DTYPE, DEFAULT_NP_DTYPE
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Tuple, List, Union
from .manifold import Manifold
from ham.utils.math import safe_norm, GRAD_EPS, NORM_EPS

__all__ = ["TriangularMesh"]


def _build_spatial_grid(
    vertices: jax.Array,
    faces: jax.Array,
    grid_size: int,
) -> Tuple[jax.Array, jax.Array, jax.Array, int]:
    """Build a 2D spatial hash grid mapping XY cells → face index lists.

    Runs entirely on the CPU with numpy at construction time (not under JIT).
    Each face is assigned to the grid cell containing its XY centroid.

    Args:
        vertices: (V, N) vertex positions.
        faces:    (F, 3) face index array.
        grid_size: Number of cells per axis (G).

    Returns:
        grid_indices: (G, G, max_M) int32 padded face index array.
            Unused slots contain -1.
        grid_origin:  (2,) float64 XY origin of the bounding box.
        grid_cell_size: (2,) float64 cell dimensions.
        max_m:        Maximum faces in any single cell (static).
    """
    # Work in numpy so the build is pure Python, not JAX-traced.
    verts_np = np.asarray(jax.device_get(vertices))
    faces_np = np.asarray(jax.device_get(faces), dtype=np.int32)
    tris_np = verts_np[faces_np]          # (F, 3, N)

    # XY bounding box
    xy = verts_np[:, :2]
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    extent = np.maximum(max_xy - min_xy, 1e-6)
    cell_size = extent / grid_size

    # Assign each face to ALL cells its XY bounding box overlaps (not just centroid).
    # This prevents misses when large triangles span multiple cells.
    grid: list = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
    for face_idx in range(len(faces_np)):
        tri_xy = tris_np[face_idx, :, :2]   # (3, 2)
        bb_min = tri_xy.min(axis=0)
        bb_max = tri_xy.max(axis=0)
        # Cell range the bounding box covers (inclusive on both ends)
        ci_lo = int(np.clip(np.floor((bb_min[0] - min_xy[0]) / cell_size[0]), 0, grid_size - 1))
        ci_hi = int(np.clip(np.floor((bb_max[0] - min_xy[0]) / cell_size[0]), 0, grid_size - 1))
        cj_lo = int(np.clip(np.floor((bb_min[1] - min_xy[1]) / cell_size[1]), 0, grid_size - 1))
        cj_hi = int(np.clip(np.floor((bb_max[1] - min_xy[1]) / cell_size[1]), 0, grid_size - 1))
        for ci in range(ci_lo, ci_hi + 1):
            for cj in range(cj_lo, cj_hi + 1):
                grid[ci][cj].append(face_idx)

    # Find maximum occupancy (must be ≥ 1 to avoid zero-size arrays)
    max_m = max(
        (len(grid[i][j]) for i in range(grid_size) for j in range(grid_size)),
        default=1,
    )
    max_m = max(max_m, 1)

    # Build padded index array (-1 = empty slot)
    grid_indices_np = np.full((grid_size, grid_size, max_m), -1, dtype=np.int32)
    for i in range(grid_size):
        for j in range(grid_size):
            cell_faces = grid[i][j]
            grid_indices_np[i, j, : len(cell_faces)] = cell_faces

    return (
        jnp.array(grid_indices_np),
        jnp.array(min_xy, dtype=DEFAULT_JNP_DTYPE),
        jnp.array(cell_size, dtype=DEFAULT_JNP_DTYPE),
        int(max_m),
    )


class TriangularMesh(Manifold):
    """A discrete 2-manifold defined by a triangular mesh in R^N.

    This class represents a surface embedded in N-dimensional space as a 
    collection of flat triangular faces. It provides differentiable projection, 
    tangent space operations, and area-weighted sampling.

    Attributes:
        vertices: Array of shape (V, N) containing vertex positions.
        faces: Integer array of shape (F, 3) indexing three vertices per face.
        triangles: Array of shape (F, 3, N) containing the coordinates of 
            each triangle vertex. Computed as ``vertices[faces]``.
    """
    vertices: jax.Array
    faces: jax.Array
    triangles: jax.Array
    # Spatial hash grid for O(max_M) nearest-face lookup instead of O(F).
    # Built once at construction time from the XY bounding box of the mesh.
    _grid_indices: jax.Array   # (G, G, max_M) int32 — padded face index grid
    _grid_origin: jax.Array    # (2,) float64 — XY origin of the grid
    _grid_cell_size: jax.Array # (2,) float64 — cell dimensions in metres
    _grid_size: int = eqx.field(static=True)   # G (number of cells per axis)
    _grid_max_m: int = eqx.field(static=True)  # max faces per cell (static for XLA)

    def __init__(self, vertices: jax.Array, faces: jax.Array, grid_size: int = 16):
        """Construct a TriangularMesh with a precomputed spatial hash grid.

        Args:
            vertices: Array of shape (V, N) — the V vertex positions in R^N.
            faces: Integer array of shape (F, 3) — each row indexes three
                vertices forming a triangle.
            grid_size: Number of grid cells per axis for the spatial hash.
                Larger values give faster ``project`` but more memory.
                Default: 16 (yields ~F/256 candidates per lookup).
        """
        self.vertices = vertices
        self.faces = faces
        self.triangles = vertices[faces]
        self._grid_size = int(grid_size)

        g_idx, g_origin, g_cell, max_m = _build_spatial_grid(
            vertices, faces, int(grid_size)
        )
        self._grid_indices = g_idx
        self._grid_origin = g_origin
        self._grid_cell_size = g_cell
        self._grid_max_m = int(max_m)

    @property
    def ambient_dim(self) -> int:
        """The dimension N of the ambient embedding space R^N."""
        return self.vertices.shape[-1]
    
    @property
    def intrinsic_dim(self) -> int:
        """The intrinsic dimension of the mesh (always 2 for a surface)."""
        return 2

    def _point_triangle_distance(
        self, p: jax.Array, tri: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute squared distance and closest point from p to triangle tri.

        Uses barycentric coordinates to test interior containment, then
        falls back to edge-segment projection for exterior points.

        Args:
            p: Query point in R^N.
            tri: Array of shape (3, N) containing the three triangle vertices.

        Returns:
            dist_sq: Squared Euclidean distance to the closest point.
            closest: The closest point on the triangle, shape (N,).
        """
        a, b, c = tri[0], tri[1], tri[2]
        ab, ac, ap = b - a, c - a, p - a
        
        # Edge-vector Gram matrix entries
        d1, d2 = jnp.dot(ab, ap), jnp.dot(ac, ap)
        d3, d4, d5 = jnp.dot(ab, ab), jnp.dot(ab, ac), jnp.dot(ac, ac)
        
        det_raw = d3 * d5 - d4 * d4
        det = jnp.maximum(det_raw, 1e-10)
        s = (d5 * d1 - d4 * d2) / det
        t = (d3 * d2 - d4 * d1) / det
        
        # Clamp barycentric coordinates to avoid overflow in near-degenerate triangles
        # during reverse-mode AD through the unused p_in branch.
        s_clamped = jnp.clip(s, -10.0, 10.0)
        t_clamped = jnp.clip(t, -10.0, 10.0)
        p_in = a + s_clamped * ab + t_clamped * ac
        
        # Edge projections for fallback
        def project_segment(u, v):
            uv = v - u
            up = p - u
            len_sq = jnp.dot(uv, uv)
            # Guard division for zero-length edges
            frac = jnp.clip(jnp.dot(up, uv) / jnp.maximum(len_sq, 1e-10), 0.0, 1.0)
            return u + frac * uv

        p_ab = project_segment(a, b)
        p_bc = project_segment(b, c)
        p_ca = project_segment(c, a)
        
        # Correctly identify interior only for non-degenerate triangles
        is_inside = (s >= 0) & (t >= 0) & (s + t <= 1) & (det_raw > 1e-8)
        
        def dist_sq_fn(x): return jnp.sum((x - p)**2)
        d_edge_vals = jnp.array([dist_sq_fn(p_ab), dist_sq_fn(p_bc), dist_sq_fn(p_ca)])
        
        # Select best edge if outside triangle
        best_edge_idx = jnp.argmin(d_edge_vals)
        p_edge = jnp.stack([p_ab, p_bc, p_ca])[best_edge_idx]
        
        closest = jnp.where(is_inside, p_in, p_edge)
        return dist_sq_fn(closest), closest

    @eqx.filter_jit
    def project(self, x: jax.Array) -> jax.Array:
        """Project a point from ambient space onto the nearest triangle face.

        Uses the precomputed spatial hash grid to restrict the search to a
        3×3 neighbourhood of grid cells (~9 × max_M candidate faces instead
        of all F faces), giving an O(max_M) cost per call under JIT/vmap.

        Dummy padding entries (index == -1) are assigned infinite distance
        and excluded from the argmin.

        Args:
            x: Query point in R^N.

        Returns:
            Closest point on the mesh surface, shape (N,).
        """
        all_candidate_indices = self._candidate_face_indices(x)
        # Replace dummy index -1 with 0 for safe gather (we mask those out below)
        safe_indices = jnp.maximum(all_candidate_indices, 0)
        local_triangles = self.triangles[safe_indices]   # (9*max_M, 3, N)

        dist_fn = lambda tri: self._point_triangle_distance(x, tri)
        dists_sq, points = jax.vmap(dist_fn)(local_triangles)

        # Mask out dummy slots with infinite distance
        is_valid = all_candidate_indices >= 0
        masked_dists = jnp.where(is_valid, dists_sq, jnp.inf)

        min_idx = jnp.argmin(masked_dists)
        return points[min_idx]

    def _candidate_face_indices(self, x: jax.Array) -> jax.Array:
        """Return the (9*max_M,) int32 array of candidate face indices for x.

        Used by project, get_face_index, and get_face_weights to restrict the
        search to a 3x3 neighbourhood of grid cells instead of all F faces.
        Dummy padding slots have value -1.
        """
        G = self._grid_size
        frac = (x[:2] - self._grid_origin) / self._grid_cell_size
        ci = jnp.clip(jnp.floor(frac[0]).astype(jnp.int32), 0, G - 1)
        cj = jnp.clip(jnp.floor(frac[1]).astype(jnp.int32), 0, G - 1)
        return jnp.concatenate([
            self._grid_indices[
                jnp.clip(ci + di, 0, G - 1),
                jnp.clip(cj + dj, 0, G - 1),
            ]
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
        ], axis=0)

    @eqx.filter_jit
    def get_face_index(self, x: jax.Array) -> jax.Array:
        """Return the index of the triangle face closest to the query point.

        Uses the spatial hash grid for O(max_M) cost instead of O(F).

        Args:
            x: Query point in R^N.

        Returns:
            Scalar integer index into self.faces / self.triangles.
        """
        all_indices = self._candidate_face_indices(x)
        safe_indices = jnp.maximum(all_indices, 0)
        local_triangles = self.triangles[safe_indices]
        dist_fn = lambda tri: self._point_triangle_distance(x, tri)[0]
        dists_sq = jax.vmap(dist_fn)(local_triangles)
        is_valid = all_indices >= 0
        masked_dists = jnp.where(is_valid, dists_sq, jnp.inf)
        local_min = jnp.argmin(masked_dists)
        return safe_indices[local_min]

    @eqx.filter_jit
    def get_face_weights(self, x: jax.Array, temperature: float = 100.0) -> jax.Array:
        """Compute differentiable face-proximity weights via softmax.

        Weights are computed as softmax(-d^2 * temperature) where d^2 is the
        squared distance from x to each face.  Higher temperature yields a
        sharper (more one-hot) distribution.

        This method intentionally scans all F faces (O(F)) because it is
        used in the differentiable ``metric_fn`` path (in DiscreteRanders and
        CovariateMeshRanders) and the softmax must normalise over the complete
        face set for correct gradient flow.  For non-differentiable
        nearest-face lookup use ``get_face_index`` (which uses the spatial
        hash grid and is O(max_M)).

        Args:
            x: Query point in R^N.
            temperature: Inverse-bandwidth parameter. Defaults to 100.0.

        Returns:
            Array of shape (F,) summing to 1, giving each face's weight.
        """
        dist_fn = lambda tri: self._point_triangle_distance(x, tri)[0]
        dists_sq = jax.vmap(dist_fn)(self.triangles)
        return jax.nn.softmax(-dists_sq * temperature)

    @eqx.filter_jit
    def to_tangent(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Project an ambient vector onto the tangent plane of the nearest face.

        Constructs an orthonormal basis {e1, e2} for the face closest to x 
        via Gram-Schmidt on edge vectors (B-A) and (C-A), then returns the 
        component of v in that subspace.

        Args:
            x: Base point in R^N (identifies the closest face).
            v: Ambient vector in R^N to project.

        Returns:
            Tangent-projected vector in R^N (lies in the face's plane).
        """
        idx = self.get_face_index(x)
        tri = self.triangles[idx]
        a, b, c = tri[0], tri[1], tri[2]
        
        # Gram-Schmidt for tangent basis
        u = b - a
        w = c - a
        
        # Use safe_norm to avoid NaN in basis construction for degenerate faces
        norm_u = safe_norm(u, eps=GRAD_EPS)
        e1 = u / norm_u
        
        w_perp = w - jnp.dot(w, e1) * e1
        norm_w_perp = safe_norm(w_perp, eps=GRAD_EPS)
        e2 = w_perp / norm_w_perp
        
        # If the triangle is degenerate (collinear edges), the basis is ill-defined.
        # safe_norm prevents NaN, but we should guard the final projection.
        d3, d4, d5 = jnp.dot(u, u), jnp.dot(u, w), jnp.dot(w, w)
        det = d3 * d5 - d4 * d4
        is_degenerate = det < NORM_EPS**2
        
        v_proj = jnp.dot(v, e1) * e1 + jnp.dot(v, e2) * e2
        return jnp.where(is_degenerate, jnp.zeros_like(v), v_proj)

    @eqx.filter_jit
    def retract(self, x: jax.Array, delta: jax.Array) -> jax.Array:
        """Retraction via projection: returns project(x + delta).

        Args:
            x: Base point on the mesh.
            delta: Ambient vector (approximate tangent vector).

        Returns:
            Point on the mesh surface closest to x + delta.
        """
        candidate = x + delta
        return self.project(candidate)

    def random_sample(self, key: jax.Array, shape: Tuple[int, ...]) -> jax.Array:
        """Sample points uniformly on the mesh surface.

        Triangles are selected with probability proportional to area. Points
        within each triangle are sampled uniformly via the standard fold-over 
        method.

        Args:
            key: JAX PRNG key.
            shape: Output batch shape; returned array has shape (*shape, N).

        Returns:
            Points on the mesh surface, shape (*shape, N).
        """
        A, B, C = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        u, w = B - A, C - A
        
        # Area = 0.5 * sqrt(|u|^2|w|^2 - (u.w)^2)
        u_sq, w_sq = jnp.sum(u**2, axis=1), jnp.sum(w**2, axis=1)
        uw_dot = jnp.sum(u * w, axis=1)
        areas = 0.5 * jnp.sqrt(jnp.maximum(u_sq * w_sq - uw_dot**2, 1e-10))
        
        k1, k2 = jax.random.split(key)
        
        # Fixed: calculate n using Python to avoid ConcretizationTypeError in JIT
        n = 1
        for s in shape:
            n *= s
            
        indices = jax.random.choice(
            k1, len(self.faces), shape=(n,), p=areas/jnp.sum(areas)
        )
        
        coords = jax.random.uniform(k2, (n, 2))
        r1, r2 = coords[:, 0], coords[:, 1]
        mask = r1 + r2 > 1
        r1 = jnp.where(mask, 1 - r1, r1)
        r2 = jnp.where(mask, 1 - r2, r2)
        
        tris = self.triangles[indices]
        pts = (
            (1 - r1[:, None] - r2[:, None]) * tris[:, 0] + 
            r1[:, None] * tris[:, 1] + 
            r2[:, None] * tris[:, 2]
        )
        return pts.reshape(shape + (self.ambient_dim,))