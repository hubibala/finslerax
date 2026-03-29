import anndata
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

def main():
    print("Loading data...")
    data_path = 'data/weinreb_preprocessed.h5ad'
    if not os.path.exists(data_path):
        data_path = 'data/weinreb_raw.h5ad'

    adata = anndata.read_h5ad(data_path)
    
    # Coordinates
    x = adata.obs['SPRING-x'].values
    y = adata.obs['SPRING-y'].values
    spr = np.column_stack((x, y))
    
    # Cell types
    cell_types = adata.obs['Cell type annotation'].astype(str)
    cell_types.replace('nan', 'Unknown', inplace=True)
    
    # Find cell colors
    unique_types = np.unique(cell_types)
    num_types = len(unique_types)
    cmap = plt.get_cmap('tab20')
    if num_types > 20:
        colors = plt.cm.get_cmap('nipy_spectral', num_types)
        color_map = {t: colors(i) for i, t in enumerate(unique_types)}
    else:
        color_map = {t: cmap(i) for i, t in enumerate(unique_types)}

    # Triples
    triples_path = 'data/weinreb_train_triples.npy'
    if not os.path.exists(triples_path):
        print(f"Error: {triples_path} not found.")
        return
        
    print(f"Loading lineage triples from {triples_path}...")
    triples = np.load(triples_path)
    
    # Identify target trajectories based on Day 6 cell type
    targets = ['Monocyte', 'Neutrophil', 'Erythroid']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PCA and Velocity for projection
    v_pca = adata.obsm['velocity_pca'][:]
    x_pca = adata.obsm['X_pca']
    
    print("Fitting NearestNeighbors for velocity projection...")
    nn = NearestNeighbors(n_neighbors=50).fit(x_pca)
    
    # Compute global bandwidth to avoid local density distortions
    print("Computing global bandwidth...")
    np.random.seed(42)
    sample_idx = np.random.choice(len(x_pca), min(10000, len(x_pca)), replace=False)
    global_dists, _ = nn.kneighbors(x_pca[sample_idx])
    global_sigma2 = np.median(global_dists**2)
    print(f"Global sigma^2: {global_sigma2:.4f}")
    
    for ax, target_type in zip(axes, targets):
        ax.set_title(f"{target_type} Destined Trajectories")
        
        # Plot background (all cells)
        ax.scatter(x, y, color='lightgray', s=1, alpha=0.3, edgecolors='none', zorder=1)
        
        # Filter triples by day 6 annotation
        target_triples = [t for t in triples if cell_types[t[2]] == target_type]
        target_triples = np.array(target_triples)
        
        if len(target_triples) == 0:
            print(f"No trajectories found destined for {target_type}")
            continue
            
        print(f"Found {len(target_triples)} trajectories destined for {target_type}")
        
        # Subsample to avoid completely blacking out the plot
        np.random.seed(42)
        n_plot = min(150, len(target_triples))
        idx_to_plot = np.random.choice(len(target_triples), n_plot, replace=False)
        triples_subset = target_triples[idx_to_plot]
        
        i2 = triples_subset[:, 0]
        i4 = triples_subset[:, 1]
        i6 = triples_subset[:, 2]
        
        line_color = color_map[target_type]
        
        # Plot continuous lines: Day2 -> Day4 -> Day6
        for start, mid, end in zip(i2, i4, i6):
            px = [x[start], x[mid], x[end]]
            py = [y[start], y[mid], y[end]]
            ax.plot(px, py, '-', color=line_color, alpha=0.5, linewidth=1.5, zorder=2)
            
        # Draw explicit starting, middle, and end points
        ax.scatter(x[i2], y[i2], c='green', s=20, zorder=5, label='Day 2 (Start)' if i2[0]==triples_subset[0,0] else "", edgecolors='white', linewidths=0.5)
        ax.scatter(x[i4], y[i4], c='blue', s=20, zorder=5, label='Day 4 (Mid)' if i2[0]==triples_subset[0,0] else "", edgecolors='white', linewidths=0.5)
        ax.scatter(x[i6], y[i6], c='red', s=20, zorder=5, label='Day 6 (End)' if i2[0]==triples_subset[0,0] else "", edgecolors='white', linewidths=0.5)
        
        # 1. Project RNA Pseudo-Velocity for ONLY the cells in these trajectories
        # To avoid clutter, we only plot velocity arrows originating from Day 2 and Day 4 cells 
        vel_idx_plot = np.unique(np.concatenate([i2, i4]))
        valid_vel = np.linalg.norm(v_pca[vel_idx_plot], axis=1) > 0
        vel_idx_plot = vel_idx_plot[valid_vel]
        
        if len(vel_idx_plot) > 0:
            dists, idxs = nn.kneighbors(x_pca[vel_idx_plot])
            v_spr = np.zeros((len(vel_idx_plot), 2))
            
            for k, i in enumerate(vel_idx_plot):
                neighbors_pca = x_pca[idxs[k]]
                disp_pca = neighbors_pca - x_pca[i]
                disp_spr = spr[idxs[k]] - spr[i]
                
                disp_pca_norm = disp_pca / (np.linalg.norm(disp_pca, axis=1, keepdims=True) + 1e-8)
                proj = np.dot(disp_pca_norm, v_pca[i])
                
                # BUG 1 FIX: use signed weights, do not discard proj <= 0
                weights = np.exp(-dists[k]**2 / (global_sigma2 + 1e-8))
                combined_weights = weights * proj
                
                v_spr_dir = np.sum(disp_spr * combined_weights[:, None], axis=0)
                v_spr[k] = v_spr_dir / (np.linalg.norm(v_spr_dir) + 1e-8) * 2.0
            
            ax.quiver(spr[vel_idx_plot, 0], spr[vel_idx_plot, 1], v_spr[:, 0], v_spr[:, 1],
                      scale=50, color='black', alpha=0.8, width=0.003, 
                      zorder=6)
                      
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    plt.tight_layout()
    output_file = "weinreb_destinations_trajectories.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    
if __name__ == "__main__":
    main()
