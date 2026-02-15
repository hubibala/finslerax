import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ham.geometry.surfaces import Hyperboloid
from ham.vis.hyperbolic import plot_poincare_disk

def generate_synthetic_tree(n_branches=2, depth=5):
    """
    Generates a synthetic differentiation tree on the Hyperboloid.
    Root at origin, branching outwards using the Exponential Map.
    """
    manifold = Hyperboloid(intrinsic_dim=2)
    key = jax.random.PRNGKey(42)
    
    # 1. Root (Stem Cell) at Origin: (1, 0, 0)
    root = jnp.array([[1.0, 0.0, 0.0]])
    
    points = [root]
    colors = [0.0] # Pseudo-time
    lineage = []
    
    # Current layer indices
    current_layer_indices = [0]
    
    for d in range(depth):
        next_layer_indices = []
        for parent_idx in current_layer_indices:
            parent_pt = points[parent_idx]
            
            # Branch out
            key, subkey = jax.random.split(key)
            
            for b in range(n_branches):
                # Random direction + bias outwards
                raw_dir = jax.random.normal(subkey, (1, 3))
                tangent_dir = manifold.to_tangent(parent_pt, raw_dir)
                
                # Normalize and step
                norm = jnp.linalg.norm(tangent_dir)
                step_size = 0.6  # Geodesic distance
                tangent_step = (tangent_dir / (norm + 1e-6)) * step_size
                
                # Move via Exponential Map
                child_pt = manifold.exp_map(parent_pt, tangent_step)
                
                # Store
                child_idx = len(points)
                points.append(child_pt)
                colors.append(d + 1.0)
                lineage.append([parent_idx, child_idx])
                next_layer_indices.append(child_idx)
                
        current_layer_indices = next_layer_indices

    return jnp.concatenate(points, axis=0), np.array(colors), np.array(lineage)

def main():
    print("Generating synthetic differentiation tree on Hyperboloid...")
    # Generate tree
    points, colors, lineage_pairs = generate_synthetic_tree(n_branches=2, depth=4)
    points_np = np.array(points)
    
    # Create 'Wind' field (Vectors from Parent to Child)
    print("Computing wind field...")
    manifold = Hyperboloid(intrinsic_dim=2)
    vectors = np.zeros_like(points_np)
    
    for p_idx, c_idx in lineage_pairs:
        p_pt = jnp.array(points_np[p_idx])
        c_pt = jnp.array(points_np[c_idx])
        # v = Log_parent(child)
        v = manifold.log_map(p_pt, c_pt)
        vectors[p_idx] = v
        
    print("Plotting to Poincaré Disk...")
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_poincare_disk(
        points_np, 
        colors=colors, 
        vectors=vectors, 
        lineage_pairs=lineage_pairs,
        title="HAM: Differentiation Tree on Hyperboloid (Poincaré Projection)",
        ax=ax
    )
    
    output_file = "weinreb_hyperbolic_vis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()