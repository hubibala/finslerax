"""Mesh adjacency and topological ordering for Eulerian PDE solvers.

This module provides the MeshAdjacency class to abstract the topological
connectivity of a TriangularMesh, enabling O(1) adjacency lookups required 
by the Godunov upwind Hamiltonian. It also computes multi-reference 
topological sweep orderings for mesh-based fast sweeping methods.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from typing import Tuple

class MeshAdjacency(eqx.Module):
    """Abstracts mesh adjacency information for Eulerian PDE solvers.
    
    This provides an intermediate representation to allow solver logic 
    to operate on abstract characteristic simplices, decoupling the solver
    from the raw face arrays and paving the way for future dynamic mesh 
    unfoldings.
    """
    num_vertices: int = eqx.field(static=True)
    num_faces: int = eqx.field(static=True)
    max_adj_faces: int = eqx.field(static=True)
    num_sweeps: int = eqx.field(static=True)
    
    # (num_vertices, max_adj_faces, 4)
    # The 4 values are: (center_vertex, adj_vertex_1, adj_vertex_2, face_idx)
    # Padding is indicated by face_idx = -1
    vertex_adjacency: jax.Array
    
    # (num_sweeps, num_vertices)
    # Array of vertex indices representing the order in which to traverse
    # the mesh during each sweep.
    sweep_orderings: jax.Array
    
    @classmethod
    def build(
        cls, 
        vertices: jax.Array, 
        faces: jax.Array, 
        num_ref_points: int = 4
    ) -> "MeshAdjacency":
        """Build the vertex adjacency and topological sweep orderings.
        
        Args:
            vertices: (V, 3) or (V, 2) vertex coordinate array.
            faces: (F, 3) integer array.
            num_ref_points: Number of reference points for multi-source sorting.
                Generates 2 * num_ref_points sweep orderings.
        """
        faces_np = np.asarray(faces)
        verts_np = np.asarray(vertices)
        num_vertices = len(verts_np)
        
        # 1. Build adjacency array
        bincount = np.bincount(faces_np.flatten(), minlength=num_vertices)
        max_adj = int(np.max(bincount))
        
        adj = np.full((num_vertices, max_adj, 4), -1, dtype=np.int32)
        counts = np.zeros(num_vertices, dtype=np.int32)
        
        perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        
        # We need an edge list for Dijkstra
        edges = []
        edge_weights = []
        
        for face_idx, face in enumerate(faces_np):
            for i, j, k in perms:
                center = face[i]
                v1 = face[j]
                v2 = face[k]
                c_idx = counts[center]
                adj[center, c_idx] = (center, v1, v2, face_idx)
                counts[center] += 1
                
                # Edges for graph traversal (undirected, so we add both ways)
                edges.append((center, v1))
                edges.append((center, v2))
                
        # 2. Build graph for Dijkstra using scipy.sparse
        import scipy.sparse as sp
        import scipy.sparse.csgraph as csgraph
        
        edges_np = np.array(edges)
        u = edges_np[:, 0]
        v = edges_np[:, 1]
        w = np.linalg.norm(verts_np[u] - verts_np[v], axis=1)
        
        graph = sp.csr_matrix((w, (u, v)), shape=(num_vertices, num_vertices))
        
        # 3. Generate reference points via Farthest Point Sampling (FPS)
        ref_points = [0]  # Start with vertex 0
        dist_matrix = csgraph.dijkstra(graph, indices=ref_points[0], directed=False)
        
        for _ in range(1, num_ref_points):
            farthest = int(np.argmax(dist_matrix))
            ref_points.append(farthest)
            new_dist = csgraph.dijkstra(graph, indices=farthest, directed=False)
            dist_matrix = np.minimum(dist_matrix, new_dist)
            
        # 4. Generate sweep orderings
        num_sweeps = num_ref_points * 2
        orderings = np.zeros((num_sweeps, num_vertices), dtype=np.int32)
        
        for i, rp in enumerate(ref_points):
            dists = csgraph.dijkstra(graph, indices=rp, directed=False)
            sorted_asc = np.argsort(dists)
            sorted_desc = sorted_asc[::-1]
            orderings[i * 2] = sorted_asc
            orderings[i * 2 + 1] = sorted_desc
            
        return cls(
            num_vertices=num_vertices,
            num_faces=len(faces_np),
            max_adj_faces=max_adj,
            num_sweeps=num_sweeps,
            vertex_adjacency=jnp.array(adj, dtype=jnp.int32),
            sweep_orderings=jnp.array(orderings, dtype=jnp.int32)
        )
