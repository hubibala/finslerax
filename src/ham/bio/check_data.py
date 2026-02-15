import scanpy as sc
import os

data_path = "data/weinreb.h5ad"

if os.path.exists(data_path):
    print(f"Loading {data_path}...")
    try:
        adata = sc.read_h5ad(data_path)
        print("\n=== Observations (adata.obs) Columns ===")
        print(adata.obs.columns.tolist())
        
        print("\n=== Sample Rows ===")
        print(adata.obs.head())
        
        # Check specific keywords
        print("\n=== Diagnostics ===")
        has_clone = any(c in adata.obs.columns for c in ['clone_id', 'lineage', 'Lineage', 'clone'])
        has_time = any(c in adata.obs.columns for c in ['time_point', 'Time point', 'day', 'Day', 'time'])
        
        print(f"Has Clone/Lineage info? {has_clone}")
        print(f"Has Time info? {has_time}")
        
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"File not found: {data_path}")