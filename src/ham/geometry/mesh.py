"""Triangular mesh manifold for discrete surfaces in R^N.

Provides the TriangularMesh class, which implements the Manifold ABC
for piecewise-linear surfaces defined by vertex/face arrays. Used by
DiscreteRanders (zoo.py) for anisotropic mesh-based metrics.

See also: spec/ARCH_SPEC.md § 5 (Module Structure).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, List, Union
from .manifold import Manifold
from ham.utils.math import safe_norm, GRAD_EPS, NORM_EPS

__all__ = ["TriangularMesh"]

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

    def __init__(self, vertices: jax.Array, faces: jax.Array):
        """Construct a TriangularMesh.

        Args:
            vertices: Array of shape (V, N) — the V vertex positions in R^N.
            faces: Integer array of shape (F, 3) — each row indexes three 
                vertices forming a triangle.
        """
        self.vertices = vertices
        self.faces = faces
        self.triangles = self.vertices[self.faces] 

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

        Performs a linear scan over all F faces and returns the closest point.

        Args:
            x: Query point in R^N.

        Returns:
            Closest point on the mesh surface, shape (N,).
        """
        dist_fn = lambda tri: self._point_triangle_distance(x, tri)
        dists_sq, points = jax.vmap(dist_fn)(self.triangles)
        min_idx = jnp.argmin(dists_sq)
        return points[min_idx]

    @eqx.filter_jit
    def get_face_index(self, x: jax.Array) -> jax.Array:
        """Return the index of the triangle face closest to the query point.

        Args:
            x: Query point in R^N.

        Returns:
            Scalar integer index into self.faces / self.triangles.
        """
        dist_fn = lambda tri: self._point_triangle_distance(x, tri)[0]
        dists_sq = jax.vmap(dist_fn)(self.triangles)
        return jnp.argmin(dists_sq)

    @eqx.filter_jit
    def get_face_weights(self, x: jax.Array, temperature: float = 100.0) -> jax.Array:
        """Compute differentiable face-proximity weights via softmax.

        Weights are computed as softmax(-d² * temperature) where d² is the 
        squared distance from x to each face. Higher temperature yields a 
        sharper (more one-hot) distribution.

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
        area_sq = d3, d4, d5 = jnp.dot(u, u), jnp.dot(u, w), jnp.dot(w, w)
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