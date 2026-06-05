"""Mesh-based fully anisotropic Godunov Eikonal solver.

Extends the Eulerian Fast Sweeping Method to unstructured triangulations.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from ham.geometry.metric import AsymmetricMetric
from ham.geometry.mesh_adjacency import MeshAdjacency
from ham.solvers.eikonal import (
    compute_two_point_update,
    compute_one_point_update,
    sharp_min,
    steady_state_min
)

def _triangle_update(
    T_nbr1: jax.Array, T_nbr2: jax.Array, T0: jax.Array,
    g11: jax.Array, g12: jax.Array, g22: jax.Array,
    b1: jax.Array, b2: jax.Array,
    m1x: jax.Array, m1y: jax.Array,
    m2x: jax.Array, m2y: jax.Array
) -> jax.Array:
    """Computes the Godunov update for a single triangle."""
    t0_2pt = compute_two_point_update(T_nbr1, T_nbr2, g11, g12, g22, b1, b2, m1x, m1y, m2x, m2y)
    t0_1pt_m1 = compute_one_point_update(T_nbr1, g11, g12, g22, b1, b2, m1x, m1y)
    t0_1pt_m2 = compute_one_point_update(T_nbr2, g11, g12, g22, b1, b2, m2x, m2y)
    
    t0_1pt = sharp_min(t0_1pt_m1, t0_1pt_m2)
    t0_candidate = sharp_min(t0_2pt, t0_1pt)
    return steady_state_min(T0, t0_candidate)

def _vertex_update(
    v_idx: jax.Array,
    T: jax.Array,
    adj_data: jax.Array,
    vertices: jax.Array,
    G_faces: jax.Array,
    B_faces: jax.Array,
    source_mask: jax.Array
) -> jax.Array:
    """Updates a single vertex by checking all its adjacent triangles."""
    T = jnp.asarray(T)
    vertices = jnp.asarray(vertices)
    G_faces = jnp.asarray(G_faces)
    B_faces = jnp.asarray(B_faces)
    
    # adj_data: (max_adj, 4) -> (center, v1, v2, face_idx)
    T0 = T[v_idx]
    
    def process_face(T0_curr, face_data):
        v1_idx = face_data[1]
        v2_idx = face_data[2]
        face_idx = face_data[3]
        
        # Valid face mask
        is_valid = face_idx >= 0
        
        T1 = jnp.where(is_valid, T[v1_idx], 1e5)
        T2 = jnp.where(is_valid, T[v2_idx], 1e5)
        
        # Vectors from target (v_idx) to donors (v1, v2)
        target_pt = vertices[v_idx]
        m1 = jnp.where(is_valid, vertices[v1_idx] - target_pt, jnp.zeros_like(target_pt))
        m2 = jnp.where(is_valid, vertices[v2_idx] - target_pt, jnp.zeros_like(target_pt))
        
        # Construct local 2D orthonormal basis (u1, u2) for the triangle
        norm1_sq = jnp.sum(m1**2)
        norm1 = jnp.sqrt(jnp.maximum(norm1_sq, 1e-12))
        u1 = m1 / norm1
        
        proj = jnp.sum(m2 * u1)
        u2 = m2 - proj * u1
        norm2 = jnp.sqrt(jnp.maximum(jnp.sum(u2**2), 1e-12))
        u2 = u2 / norm2
        
        # 2D coordinates of the edges in the (u1, u2) basis
        m1x = norm1
        m1y = 0.0
        m2x = proj
        m2y = jnp.sum(m2 * u2)
        
        # Retrieve the N x N metric and N x 1 drift for this face
        G_face = jnp.where(is_valid, G_faces[face_idx], jnp.eye(G_faces.shape[-1]))
        B_face = jnp.where(is_valid, B_faces[face_idx], jnp.zeros(B_faces.shape[-1]))
        
        # Project metric G and drift B into the 2D tangent plane
        # g_local = U^T G U
        g11 = jnp.dot(u1, jnp.dot(G_face, u1))
        g12 = jnp.dot(u1, jnp.dot(G_face, u2))
        g22 = jnp.dot(u2, jnp.dot(G_face, u2))
        
        b1 = jnp.dot(u1, B_face)
        b2 = jnp.dot(u2, B_face)
        
        # If valid, compute update, else return 1e5
        T0_new = _triangle_update(T1, T2, T0_curr, g11, g12, g22, b1, b2, m1x, m1y, m2x, m2y)
        T0_new = jnp.where(is_valid, T0_new, T0_curr)
        return T0_new, None
        
    T0_final, _ = jax.lax.scan(process_face, T0, adj_data)
    
    # Enforce boundary condition: source nodes remain 0
    return jnp.where(source_mask[v_idx], 0.0, T0_final)

def sweep_mesh(
    T: jax.Array,
    orderings: jax.Array,
    adjacency: jax.Array,
    vertices: jax.Array,
    G_faces: jax.Array,
    B_faces: jax.Array,
    source_mask: jax.Array
) -> jax.Array:
    """Executes all topological sweeps over the mesh."""
    
    def sweep_pass(T_curr, ordering):
        def update_step(T_state, v_idx):
            # Update the vertex
            T_new_val = _vertex_update(
                v_idx, T_state, adjacency[v_idx], vertices, G_faces, B_faces, source_mask
            )
            # Write it back into the state array
            T_state = T_state.at[v_idx].set(T_new_val)
            return T_state, None
            
        T_out, _ = jax.lax.scan(update_step, T_curr, ordering)
        return T_out, None
        
    T_final, _ = jax.lax.scan(sweep_pass, T, orderings)
    return T_final

@jax.custom_vjp
def _fast_mesh_solve(G_faces, B_faces, source_mask, orderings, adjacency, vertices, max_iters, tol):
    def cond_fun(val):
        T, max_diff, iters = val
        return (max_diff > tol) & (iters < max_iters)
        
    def body_fun(val):
        T, _, iters = val
        T_new = sweep_mesh(T, orderings, adjacency, vertices, G_faces, B_faces, source_mask)
        diff = jnp.abs(T_new - T)
        max_diff = jnp.max(jnp.where(jnp.isnan(diff), 0.0, diff))
        return T_new, max_diff, iters + 1
        
    num_vertices = source_mask.shape[0]
    T_init = jnp.where(source_mask, 0.0, 1e5)
    final_val = jax.lax.while_loop(cond_fun, body_fun, (T_init, jnp.array(1e5), 0))
    return final_val[0]

def _solve_fwd(G_faces, B_faces, source_mask, orderings, adjacency, vertices, max_iters, tol):
    T_final = _fast_mesh_solve(G_faces, B_faces, source_mask, orderings, adjacency, vertices, max_iters, tol)
    return T_final, (T_final, G_faces, B_faces, source_mask, orderings, adjacency, vertices, max_iters)

def _solve_bwd(res, g_T):
    T_final, G_faces, B_faces, source_mask, orderings, adjacency, vertices, max_iters = res
    
    def single_sweep(T_in, G_in, B_in):
        return sweep_mesh(T_in, orderings, adjacency, vertices, G_in, B_in, source_mask)
        
    _, vjp_fn = jax.vjp(single_sweep, T_final, G_faces, B_faces)
    
    def cond_fun(val):
        x_curr, x_prev, iters = val
        return (jnp.max(jnp.abs(x_curr - x_prev)) > 1e-6) & (iters < max_iters)
        
    def body_fun(val):
        x_curr, _, iters = val
        dT, _, _ = vjp_fn(x_curr)
        x_next = g_T + dT
        return x_next, x_curr, iters + 1
        
    x_init = jnp.zeros_like(g_T)
    x_prev = jnp.full_like(g_T, 1e9)
    x_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, (x_init, x_prev, 0))
    
    _, dG, dB = vjp_fn(x_final)
    
    dG = jnp.where(jnp.isnan(dG), 0.0, dG)
    dB = jnp.where(jnp.isnan(dB), 0.0, dB)
    
    return dG, dB, None, None, None, None, None, None

_fast_mesh_solve.defvjp(_solve_fwd, _solve_bwd)

class MeshEikonalSolver(eqx.Module):
    """
    Mesh-based Eulerian Eikonal Solver for arbitrary triangulations.
    
    This solver extends the Fast Sweeping Method to unstructured manifolds (e.g. 
    triangle meshes). By utilizing a precomputed sequence of topologically-sorted 
    vertex sweeps, it propagates arrival times across the mesh in $O(N)$ time per 
    sweep, naturally supporting complex topologies like spheres or terrain models.
    
    The Godunov Hamiltonian evaluates the full N-dimensional ambient metric 
    projected strictly onto the 2D local tangent plane of each element.
    It integrates fully with ``jax.custom_vjp``, providing exact gradients of 
    the solution $T(x)$ with respect to the continuous metric tensors with 
    $O(1)$ memory consumption.
    """
    max_iters: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    
    def __init__(self, max_iters: int = 50, tol: float = 1e-4):
        self.max_iters = max_iters
        self.tol = tol

    def solve(self, metric: AsymmetricMetric, mesh_adj: MeshAdjacency, vertices: jax.Array, faces: jax.Array, source_coords: jax.Array) -> jax.Array:
        """
        Solves the Eikonal PDE on the provided mesh.
        
        Evaluates the metric tensor per-face, identifies the nearest vertices to 
        the source coordinates to seed the initial conditions, and executes the 
        fast sweeping while-loop to reach a steady-state global arrival time field.
        
        Args:
            metric: The AsymmetricMetric evaluated over the surface.
            mesh_adj: Precomputed topological adjacency containing the sorted sweep orderings.
            vertices: Coordinates of mesh vertices. Shape ``(V, D)`` where D is 2, 3, or N.
            faces: Triangular connectivity array. Shape ``(F, 3)``.
            source_coords: Initial fire ignition points. Shape ``(S, D)``.
            
        Returns:
            T: Global arrival times for each vertex. Shape ``(V,)``.
            
        Reference:
            Implementation utilizes characteristic ray clamping (Option A) for strict 
            causality preservation.
        """
        # 1. Map source points to closest vertices
        def dist_to_src(v):
            return jnp.min(jnp.sum((source_coords - v)**2, axis=-1))
            
        dists = jax.vmap(dist_to_src)(vertices)
        
        # Find the vertices closest to the sources
        # We assign source_mask = True for the absolute closest vertex to each source
        # To do this safely for multiple sources, we can iterate over sources
        def get_closest_v(src):
            return jnp.argmin(jnp.sum((vertices - src)**2, axis=-1))
            
        closest_vs = jax.vmap(get_closest_v)(source_coords)
        source_mask = jnp.zeros(len(vertices), dtype=bool)
        source_mask = source_mask.at[closest_vs].set(True)
        
        # 2. Evaluate Metric on Faces
        # The Godunov scheme assumes piecewise constant metric per face
        face_centroids = jnp.mean(vertices[faces], axis=1)
        
        def extract_GB(pt):
            H, W, lam = metric.zermelo_data(pt)
            B = - jnp.dot(H, W) / lam
            HW = jnp.dot(H, W)
            G = (H + jnp.outer(HW, HW) / lam) / lam
            return G, B
            
        G_faces, B_faces = jax.vmap(extract_GB)(face_centroids)
        
        # 3. Solve!
        T = _fast_mesh_solve(
            G_faces, B_faces, source_mask, 
            mesh_adj.sweep_orderings, mesh_adj.vertex_adjacency, 
            vertices, self.max_iters, self.tol
        )
        return T
