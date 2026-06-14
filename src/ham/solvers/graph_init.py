"""Graph-based global initialiser for geodesic boundary-value problems.

A discrete geodesic energy under a stiff, data-driven metric is globally
non-convex: a cold straight-line guess between two distant latents dives into a
high-cost void, where every *local* solver (:class:`~ham.solvers.AVBDSolver` or
:class:`~ham.solvers.GaussNewtonGeodesic`) stalls, diverges, or — worse — finds a
spurious *tunnelling* minimum that hugs the data near each endpoint and leaps
across the void in between.  The cure is to seed the local solver inside the
correct basin (homotopy class).

This module provides that seed via the **eikonal ↔ geodesic duality**.  On a
point cloud of latent codes, the shortest path on a k-nearest-neighbour graph is
the discrete-eikonal (fast-marching) approximation of the global geodesic: it
picks the right way *around* the data manifold but is restricted to the samples
and is only piecewise-linear.  Handed to a continuous solver as ``init_path`` it
fixes the global topology cheaply (a Dijkstra solve is a small fraction of one
relaxation), after which the local solver recovers the true smooth, off-sample,
differentiable geodesic.  This is the standard global-planner + local-refiner
split, expressed with HAM's own geometry.

The graph build uses NumPy + :func:`scipy.sparse.csgraph.dijkstra` (SciPy is a
hard dependency); no extra packages are required.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ham.solvers.continuation import resample_path

__all__ = ["build_knn_graph", "geodesic_graph_init"]


def build_knn_graph(points: np.ndarray, k: int = 8) -> csr_matrix:
    """Build a symmetric k-nearest-neighbour graph with Euclidean edge weights.

    Args:
        points: Point cloud, shape ``(M, D)``.
        k: Number of neighbours per node.

    Returns:
        A symmetric sparse adjacency matrix (``M x M``) of edge distances.
    """
    points = np.asarray(points)
    m = points.shape[0]
    k = int(min(k, m - 1))
    # Pairwise squared distances; argpartition picks the k+1 smallest (incl self).
    sq = (
        np.sum(points**2, axis=1)[:, None]
        + np.sum(points**2, axis=1)[None, :]
        - 2.0 * points @ points.T
    )
    np.maximum(sq, 0.0, out=sq)
    nbr = np.argpartition(sq, k + 1, axis=1)[:, : k + 1]
    rows = np.repeat(np.arange(m), k + 1)
    cols = nbr.ravel()
    w = np.sqrt(sq[rows, cols])
    g = csr_matrix((w, (rows, cols)), shape=(m, m))
    # Symmetrise (mutual reachability): keep the larger of the two directed edges.
    return g.maximum(g.T)


def geodesic_graph_init(
    points: jax.Array,
    p_start: jax.Array,
    p_end: jax.Array,
    n_steps: int,
    *,
    k: int = 8,
    graph: Optional[csr_matrix] = None,
) -> jax.Array:
    """Global warm-start path for a geodesic BVP via a kNN-graph shortest path.

    Snaps ``p_start`` / ``p_end`` to their nearest points in the cloud, runs
    Dijkstra between them, prepends/appends the true endpoints, and resamples to
    ``n_steps + 1`` vertices.  Use the result as ``init_path`` for
    :meth:`AVBDSolver.solve` / :class:`GaussNewtonGeodesic` (typically straight
    into a *stiff* solve — annealing from a gentle metric would relax the path
    back toward the chord and discard the homotopy class).

    Args:
        points: Latent point cloud defining the data manifold, shape ``(M, D)``.
        p_start, p_end: Boundary points, shape ``(D,)``.
        n_steps: Number of segments; the returned path has ``n_steps + 1`` vertices.
        k: Neighbours per node for the graph (ignored if ``graph`` is given).
        graph: Optional precomputed graph from :func:`build_knn_graph` (reuse it
            across many endpoint pairs — the build is the only O(M²) cost).

    Returns:
        Warm-start path, shape ``(n_steps + 1, D)``, with exact endpoints.
    """
    pts = np.asarray(points)
    z0 = np.asarray(p_start)
    z1 = np.asarray(p_end)

    if graph is None:
        graph = build_knn_graph(pts, k=k)

    i0 = int(np.argmin(np.sum((pts - z0) ** 2, axis=1)))
    i1 = int(np.argmin(np.sum((pts - z1) ** 2, axis=1)))

    _, predecessors = dijkstra(graph, indices=i0, return_predecessors=True)

    # Reconstruct i0 -> i1 by walking predecessors back from i1.
    chain = [i1]
    while chain[-1] != i0:
        prev = predecessors[chain[-1]]
        if prev < 0:  # disconnected — fall back to the straight chord.
            waypoints = np.stack([z0, z1]).astype(pts.dtype)
            return resample_path(jnp.asarray(waypoints), n_steps)
        chain.append(prev)
    chain.reverse()

    waypoints = np.concatenate([z0[None], pts[chain], z1[None]], axis=0).astype(pts.dtype)
    return resample_path(jnp.asarray(waypoints), n_steps)
