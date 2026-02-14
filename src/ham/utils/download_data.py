import scvelo as scv
import scanpy as sc

def download_benchmark_data():
    print("Downloading Pancreas data (approx 30MB)...")
    # This automatically saves 'pancreas.h5ad' to ./data/
    adata_pancreas = scv.datasets.pancreas()
    
    # Pre-calculate velocity vectors so we have a "Ground Truth" V to train on
    print("Preprocessing & estimating velocity...")
    scv.pp.filter_and_normalize(adata_pancreas)
    scv.pp.moments(adata_pancreas)
    scv.tl.velocity(adata_pancreas, mode='stochastic')
    scv.tl.velocity_graph(adata_pancreas)
    
    # Save the processed file
    filename = "data/pancreas_processed.h5ad"
    adata_pancreas.write(filename)
    print(f"Saved to {filename}")
    return filename

if __name__ == "__main__":
    download_benchmark_data()