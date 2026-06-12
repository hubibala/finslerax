import numpy as np
import anndata
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

def main():
    print("Loading data...")
    adata = anndata.read_h5ad("data/weinreb_diffusion.h5ad")
    
    X_diff = adata.obsm['X_diffmap']
    X_umap = adata.obsm['X_umap']
    
    print("Loading trajectory...")
    try:
        traj_10d = np.load("results_weinreb_diffusion/trajectory_10d.npy")
    except FileNotFoundError:
        print("Error: Could not find results_weinreb_diffusion/trajectory_10d.npy")
        print("Please run experiment_h5_diffusion_finsler.py first.")
        return
        
    print(f"Trajectory shape: {traj_10d.shape}")
    
    # --------------------------------------------------------------------------
    # Projection Trick: Map 10D -> 2D UMAP using KNN
    # --------------------------------------------------------------------------
    print("Training KNN Regressor to map 10D Diffusion to 2D UMAP...")
    knn = KNeighborsRegressor(n_neighbors=5)
    # Fit on a subset to speed up if necessary, but 130k is fast enough for sklearn KNN usually
    # Subsampling for speed:
    sub_idx = np.random.choice(len(X_diff), size=20000, replace=False)
    knn.fit(X_diff[sub_idx], X_umap[sub_idx])
    
    print("Predicting UMAP coordinates for geodesic...")
    traj_umap = knn.predict(traj_10d)
    
    # --------------------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------------------
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    
    # Plot background
    subsample = 5
    plt.scatter(X_umap[::subsample, 0], X_umap[::subsample, 1], 
                s=2, c='lightgray', alpha=0.5, label='Weinreb Data')
                
    # Plot geodesic
    plt.plot(traj_umap[:, 0], traj_umap[:, 1], 
             color='red', linewidth=3, label='DPT-Finsler Geodesic')
             
    # Plot start and end points
    plt.scatter(traj_umap[0, 0], traj_umap[0, 1], 
                c='cyan', s=150, marker='o', edgecolors='black', zorder=5, label='Start (Stem)')
    plt.scatter(traj_umap[-1, 0], traj_umap[-1, 1], 
                c='magenta', s=150, marker='X', edgecolors='black', zorder=5, label='End (Mature)')
                
    plt.title("Weinreb DPT-Finsler Geodesic (Diffusion Space -> UMAP Projection)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    
    out_path = "results_weinreb_diffusion/weinreb_diffusion_geodesic_umap.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
