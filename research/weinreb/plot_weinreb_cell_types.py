import anndata
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    print("Loading data...")
    data_path = 'data/weinreb_preprocessed.h5ad'
    if not os.path.exists(data_path):
        data_path = 'data/weinreb_raw.h5ad'

    adata = anndata.read_h5ad(data_path)
    
    # Coordinates
    x = adata.obs['SPRING-x'].values
    y = adata.obs['SPRING-y'].values
    
    # Cell types
    cell_types = adata.obs['Cell type annotation'].astype(str)
    
    # Handle NaNs or missing cell types
    cell_types.replace('nan', 'Unknown', inplace=True)
    
    unique_types = np.unique(cell_types)
    num_types = len(unique_types)
    
    # Get a colormap
    cmap = plt.get_cmap('tab20')
    if num_types > 20:
        # If there are more than 20 cell types, stack multiple colormaps
        # but 20 is usually enough for Weinreb
        colors = plt.cm.get_cmap('nipy_spectral', num_types)
        color_map = {t: colors(i) for i, t in enumerate(unique_types)}
    else:
        color_map = {t: cmap(i) for i, t in enumerate(unique_types)}
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cell type
    for ctype in unique_types:
        mask = (cell_types == ctype)
        ax.scatter(x[mask], y[mask], c=[color_map[ctype]], label=ctype,
                   s=2, alpha=0.7, edgecolors='none')
                   
    # Configure legend and axes
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              markerscale=5, title='Cell Type', fontsize='small')

    # --- ADDED CODE: Plot trajectories & RNA pseudo-velocities ---
    
    # 1. Project RNA Pseudo-Velocity from PCA to SPRING 2D space
    # The pseudo-velocity is stored in `velocity_pca` (Nx50). 
    # To plot it on the SPRING coordinates, we can approximate the 2D direction 
    # by taking the difference between the cell's SPRING coordinates and the 
    # average SPRING coordinates of its closest neighbors in PCA space in the 
    # direction of the velocity vector.
    v_pca = adata.obsm['velocity_pca'][:]
    
    # To keep it simple and fast specifically for visualization, we can just 
    # use a basic projection if we don't have an explicit SPRING velocity map.
    # We will compute a simple local finite-difference projection using kNN 
    # for a subset of cells that have non-zero velocity.
    print("Projecting RNA pseudo-velocity to 2D...")
    valid_vel = np.linalg.norm(v_pca, axis=1) > 0
    # Subsample indices for velocity arrows to avoid clutter
    np.random.seed(42)
    vel_idx = np.where(valid_vel)[0]
    n_vel_plot = min(1500, len(vel_idx))
    vel_idx_plot = np.random.choice(vel_idx, n_vel_plot, replace=False)
    
    from sklearn.neighbors import NearestNeighbors
    x_pca = adata.obsm['X_pca']
    spr = np.column_stack((x, y))
    
    nn = NearestNeighbors(n_neighbors=50).fit(x_pca)
    dists, idxs = nn.kneighbors(x_pca[vel_idx_plot])
    
    v_spr = np.zeros((len(vel_idx_plot), 2))
    
    for k, i in enumerate(vel_idx_plot):
        neighbors_pca = x_pca[idxs[k]]
        disp_pca = neighbors_pca - x_pca[i]
        disp_spr = spr[idxs[k]] - spr[i]
        
        # Norms
        disp_pca_norm = disp_pca / (np.linalg.norm(disp_pca, axis=1, keepdims=True) + 1e-8)
        
        # Projection of PCA displacement onto the velocity vector
        proj = np.dot(disp_pca_norm, v_pca[i])
        
        # Keep only neighbors in the direction of velocity
        valid = proj > 0
        if np.sum(valid) > 0:
            # Weighted average of SPRING displacement
            weights = np.exp(-dists[k]**2 / (np.mean(dists[k]**2) + 1e-8))
            v_spr_dir = np.average(disp_spr[valid], weights=weights[valid]*proj[valid], axis=0)
            
            # Scale length by original velocity norm
            # Here we just use a small constant scaling for visual representation
            v_spr[k] = v_spr_dir / (np.linalg.norm(v_spr_dir) + 1e-8) * 1.5
            
    # Plot pseudo-velocity arrows
    ax.quiver(spr[vel_idx_plot, 0], spr[vel_idx_plot, 1], v_spr[:, 0], v_spr[:, 1],
              scale=50, color='gray', alpha=0.6, width=0.002, 
              label='RNA Pseudo-Velocity')

    # 2. Plot Continuous Trajectories (Lineage pairs)
    triples_path = 'data/weinreb_lineage_triples.npy'
    if os.path.exists(triples_path):
        print(f"Loading lineage triples from {triples_path}...")
        triples = np.load(triples_path)
        
        # Subsample to avoid cluttering 
        n_plot = min(150, len(triples))
        idx_to_plot = np.random.choice(len(triples), n_plot, replace=False)
        triples_subset = triples[idx_to_plot]
        
        # Extract coordinates for start and end of each step
        i2 = triples_subset[:, 0]
        i4 = triples_subset[:, 1]
        i6 = triples_subset[:, 2]
        
        # Plot continuous lines: Day2 -> Day4 -> Day6
        for start, mid, end in zip(i2, i4, i6):
            px = [x[start], x[mid], x[end]]
            py = [y[start], y[mid], y[end]]
            ax.plot(px, py, '-', color='black', alpha=0.3, linewidth=1.0)
            
        # Draw explicit starting, middle, and end points
        ax.scatter(x[i2], y[i2], c='green', s=15, zorder=5, label='Day 2 (Start)', edgecolors='white', linewidths=0.5)
        ax.scatter(x[i4], y[i4], c='blue', s=15, zorder=5, label='Day 4 (Mid)', edgecolors='white', linewidths=0.5)
        ax.scatter(x[i6], y[i6], c='red', s=15, zorder=5, label='Day 6 (End)', edgecolors='white', linewidths=0.5)
                  
        # Re-create legend to include the trajectories and velocities
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        # Optional: Customize legend order or size here
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), 
                  loc='upper left', markerscale=2, title='Cell Type & Trajectories', 
                  fontsize='small')
    
    ax.set_title("Weinreb Dataset (SPRING Coordinates)")
    ax.set_xlabel("SPRING-x")
    ax.set_ylabel("SPRING-y")
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    output_file = "weinreb_cell_types_2d.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
if __name__ == "__main__":
    main()
