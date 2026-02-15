import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

def project_to_poincare(x: jnp.ndarray) -> jnp.ndarray:
    """
    Projects points from Hyperboloid model (Minkowski) to Poincaré Ball.
    x: (..., D+1) where -x0^2 + x1^2 + ... = -1
    Returns: (..., D) in the unit ball.
    """
    # y = x_spatial / (1 + x0)
    x0 = x[..., 0:1]
    x_spatial = x[..., 1:]
    return x_spatial / (1.0 + x0)

def project_vector_to_poincare(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Pushforward of the projection map. Converts tangent vectors from
    Hyperboloid to Poincaré Ball.
    """
    x0 = x[..., 0:1]
    x_spatial = x[..., 1:]
    
    v0 = v[..., 0:1]
    v_spatial = v[..., 1:]
    
    # Derivation quotient rule:
    # w = (v_spatial * (1+x0) - x_spatial * v0) / (1+x0)^2
    denom = (1.0 + x0)**2
    num = v_spatial * (1.0 + x0) - x_spatial * v0
    return num / denom

def plot_poincare_disk(
    points: np.ndarray, 
    colors=None, 
    vectors: np.ndarray = None, 
    lineage_pairs: np.ndarray = None,
    title: str = "Hyperbolic Embedding (Poincaré Disk)",
    ax=None
):
    """
    Visualizes the embedding on the 2D Poincaré disk.
    
    Args:
        points: (N, 3) Hyperboloid coordinates (will be projected to 2D)
        colors: (N,) Array for coloring points (e.g. cell type or pseudotime)
        vectors: (N, 3) Optional vector field (e.g. the Wind) to plot arrows
        lineage_pairs: (M, 2) Indices of (Parent, Child) to draw edges
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        
    # 1. Project to Disk
    # Ensure inputs are numpy
    points_p = np.array(project_to_poincare(points))
    
    # 2. Draw Boundary
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--', alpha=0.5)
    ax.add_artist(circle)
    
    # 3. Draw Lineage Edges (Geodesics would be arcs, straight lines are approx)
    if lineage_pairs is not None:
        start_pts = points_p[lineage_pairs[:, 0]]
        end_pts = points_p[lineage_pairs[:, 1]]
        lines = np.stack([start_pts, end_pts], axis=1)
        lc = mc.LineCollection(lines, colors='gray', alpha=0.2, linewidths=0.5)
        ax.add_collection(lc)

    # 4. Draw Vectors (Wind)
    if vectors is not None:
        vectors_p = np.array(project_vector_to_poincare(points, vectors))
        # Subsample for clarity if too many
        if len(points) > 500:
            idx = np.random.choice(len(points), 500, replace=False)
            ax.quiver(points_p[idx, 0], points_p[idx, 1], 
                      vectors_p[idx, 0], vectors_p[idx, 1], 
                      color='red', alpha=0.6, scale=20, width=0.003)
        else:
            ax.quiver(points_p[:, 0], points_p[:, 1], 
                      vectors_p[:, 0], vectors_p[:, 1], 
                      color='red', alpha=0.6)

    # 5. Scatter Points
    sc = ax.scatter(points_p[:, 0], points_p[:, 1], c=colors, cmap='viridis', s=10, alpha=0.8)
    if colors is not None:
        plt.colorbar(sc, ax=ax, label="Cell Type / Time")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    
    return ax