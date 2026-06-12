"""Graph coloring utilities for parallel Gauss-Seidel sweeps.

Provides vertex coloring algorithms that partition a graph into independent
sets. Vertices of the same color share no edges, so they can be updated
simultaneously in a parallel (colored) Gauss-Seidel sweep.

For a 1D path graph the chromatic number is 2 (even/odd), enabling a
straightforward 2-color parallel sweep. For general graphs (e.g. triangular
meshes), a greedy coloring is provided.

Reference:
    Giles, Diaz & Yuksel. *Augmented Vertex Block Descent.* SIGGRAPH 2025.
    The original VBD paper uses graph coloring to enable GPU-parallel
    Gauss-Seidel updates of independent vertex blocks.

See also: spec/ARCH_SPEC.md § 4.2.
"""


import jax.numpy as jnp
import numpy as np

__all__ = [
    "chain_coloring",
    "greedy_coloring",
    "mesh_vertex_coloring",
]


def chain_coloring(n_inner: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """2-color a 1D chain of inner path vertices.

    For a path graph with vertices at positions 1..n_inner in the full
    path array (0 and n_inner+1 are fixed boundaries), returns two arrays
    of vertex indices forming independent sets.

    Args:
        n_inner: Number of inner (free) vertices.

    Returns:
        Tuple of (color_0_indices, color_1_indices) where indices are
        1-based positions in the full path array. Vertices within each
        group share no edges and can be updated in parallel.

    Example:
        For n_inner=5 (full path has indices 0..6):
          color_0 = [1, 3, 5]  (odd positions)
          color_1 = [2, 4]     (even positions)
    """
    all_indices = jnp.arange(1, n_inner + 1)
    # Odd 1-based positions: 1, 3, 5, ...
    color_0 = all_indices[0::2]
    # Even 1-based positions: 2, 4, 6, ...
    color_1 = all_indices[1::2]
    return color_0, color_1


def greedy_coloring(
    adjacency: dict[int, set[int]],
    n_vertices: int,
) -> list[list[int]]:
    """Greedy graph coloring for general adjacency structures.

    Assigns colors to vertices such that no two adjacent vertices share
    a color. Uses a simple greedy algorithm (not optimal, but sufficient
    for the sparse graphs arising in mesh-based solvers).

    Args:
        adjacency: Mapping from vertex index to set of neighbor indices.
        n_vertices: Total number of vertices.

    Returns:
        List of lists, where ``result[c]`` contains the vertex indices
        assigned to color ``c``. The number of colors is at most
        ``max_degree + 1``.
    """
    colors = [-1] * n_vertices
    num_colors = 0

    for v in range(n_vertices):
        # Collect colors used by neighbors
        neighbor_colors: set[int] = set()
        for u in adjacency.get(v, set()):
            if colors[u] >= 0:
                neighbor_colors.add(colors[u])

        # Assign smallest available color
        c = 0
        while c in neighbor_colors:
            c += 1
        colors[v] = c
        num_colors = max(num_colors, c + 1)

    # Group vertices by color
    groups: list[list[int]] = [[] for _ in range(num_colors)]
    for v, c in enumerate(colors):
        groups[c].append(v)

    return groups


def mesh_vertex_coloring(
    faces: jnp.ndarray,
    n_vertices: int,
) -> list[jnp.ndarray]:
    """Color the vertices of a triangular mesh.

    Builds the vertex adjacency graph from the face array and applies
    greedy coloring. The result partitions vertices into independent sets
    suitable for parallel Gauss-Seidel sweeps on mesh-based solvers.

    Args:
        faces: Integer face array, shape ``(F, 3)``.
        n_vertices: Number of vertices in the mesh.

    Returns:
        List of JAX arrays, one per color, each containing the vertex
        indices assigned to that color.
    """
    # Build adjacency from faces (host-side; device arrays iterate slowly)
    adjacency: dict[int, set[int]] = {v: set() for v in range(n_vertices)}
    faces_np = np.asarray(faces)

    for face in faces_np:
        f = [int(face[0]), int(face[1]), int(face[2])]
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[f[i]].add(f[j])

    groups = greedy_coloring(adjacency, n_vertices)
    return [jnp.array(g, dtype=jnp.int32) for g in groups if len(g) > 0]
