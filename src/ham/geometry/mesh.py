import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial
from .manifold import Manifold

class TriangularMesh(Manifold):
    """
    A discrete manifold defined by a triangular mesh in R^N.
    """
    def __init__(self, vertices: jnp.ndarray, faces: jnp.ndarray):
        self.vertices = vertices
        self.faces = faces
        self.triangles = self.vertices[self.faces] 

    @property
    def ambient_dim(self) -> int:
        return self.vertices.shape[-1]
    
    @property
    def intrinsic_dim(self) -> int:
        return 2

    def _point_triangle_distance(self, p: jnp.ndarray, tri: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Computes distance and closest point. (Logic identical to previous version)"""
        a, b, c = tri[0], tri[1], tri[2]
        ab, ac, ap = b - a, c - a, p - a
        
        # Metric Tensor entries
        d1, d2 = jnp.dot(ab, ap), jnp.dot(ac, ap)
        d3, d4, d5 = jnp.dot(ab, ab), jnp.dot(ab, ac), jnp.dot(ac, ac)
        
        det = jnp.maximum(d3 * d5 - d4 * d4, 1e-10)
        s = (d5 * d1 - d4 * d2) / det
        t = (d3 * d2 - d4 * d1) / det
        
        p_in = a + s * ab + t * ac
        
        # Edge projections
        def project_segment(u, v):
            uv = v - u
            up = p - u
            len_sq = jnp.dot(uv, uv)
            frac = jnp.clip(jnp.dot(up, uv) / jnp.maximum(len_sq, 1e-10), 0.0, 1.0)
            return u + frac * uv

        p_ab, p_bc, p_ca = project_segment(a, b), project_segment(b, c), project_segment(c, a)
        
        is_inside = (s >= 0) & (t >= 0) & (s + t <= 1)
        
        def dist_sq(x): return jnp.sum((x - p)**2)
        d_edge_vals = jnp.array([dist_sq(p_ab), dist_sq(p_bc), dist_sq(p_ca)])
        best_edge_idx = jnp.argmin(d_edge_vals)
        p_edge = jnp.stack([p_ab, p_bc, p_ca])[best_edge_idx]
        
        closest = jnp.where(is_inside, p_in, p_edge)
        return dist_sq(closest), closest

    @partial(jax.jit, static_argnums=(0,))
    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        dist_fn = lambda tri: self._point_triangle_distance(x, tri)
        dists, points = jax.vmap(dist_fn)(self.triangles)
        min_idx = jnp.argmin(dists)
        return points[min_idx]

    @partial(jax.jit, static_argnums=(0,))
    def get_face_index(self, x: jnp.ndarray) -> jnp.ndarray:
        """Returns the index of the triangle closest to x."""
        dist_fn = lambda tri: self._point_triangle_distance(x, tri)[0]
        dists = jax.vmap(dist_fn)(self.triangles)
        return jnp.argmin(dists)

    @partial(jax.jit, static_argnums=(0,))
    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        idx = self.get_face_index(x)
        tri = self.triangles[idx]
        a, b, c = tri[0], tri[1], tri[2]
        
        # Gram-Schmidt for tangent basis
        u = b - a
        w = c - a
        e1 = u / (jnp.linalg.norm(u) + 1e-10)
        w_perp = w - jnp.dot(w, e1) * e1
        e2 = w_perp / (jnp.linalg.norm(w_perp) + 1e-10)
        
        return jnp.dot(v, e1) * e1 + jnp.dot(v, e2) * e2

    @partial(jax.jit, static_argnums=(0,))
    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        candidate = x + delta
        return self.project(candidate)

    def random_sample(self, key: jax.Array, shape: Tuple[int, ...]) -> jnp.ndarray:
        # (Same area-weighted logic as verified in tests)
        A, B, C = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        u, v = B - A, C - A
        u_sq, v_sq = jnp.sum(u**2, axis=1), jnp.sum(v**2, axis=1)
        uv_dot = jnp.sum(u * v, axis=1)
        areas = 0.5 * jnp.sqrt(jnp.maximum(u_sq * v_sq - uv_dot**2, 1e-10))
        
        k1, k2 = jax.random.split(key)
        n = jnp.prod(jnp.array(shape))
        indices = jax.random.choice(k1, len(self.faces), shape=(n,), p=areas/jnp.sum(areas))
        
        coords = jax.random.uniform(k2, (n, 2))
        r1, r2 = coords[:, 0], coords[:, 1]
        mask = r1 + r2 > 1
        r1 = jnp.where(mask, 1 - r1, r1)
        r2 = jnp.where(mask, 1 - r2, r2)
        
        tris = self.triangles[indices]
        pts = (1 - r1[:, None] - r2[:, None]) * tris[:, 0] + r1[:, None] * tris[:, 1] + r2[:, None] * tris[:, 2]
        return pts.reshape(shape + (self.ambient_dim,))