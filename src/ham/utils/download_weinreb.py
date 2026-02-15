import os
import urllib.request
import gzip
import pandas as pd
import scipy.io
import scanpy as sc
import numpy as np

# Klein Lab Raw Data Repository
BASE_URL = "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/"
DATA_DIR = "data"
FINAL_FILE = os.path.join(DATA_DIR, "weinreb_raw.h5ad")

FILES = {
    "counts": "stateFate_inVitro_normed_counts.mtx.gz",
    "genes": "stateFate_inVitro_gene_names.txt.gz",
    "meta": "stateFate_inVitro_metadata.txt.gz",
    "clones": "stateFate_inVitro_clone_matrix.mtx.gz"
}

def download_file(filename):
    url = f"{BASE_URL}/{filename}"
    local_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None
    return local_path

def download_weinreb():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    # 1. Download all raw files
    print("Fetching raw data from Klein Lab...")
    for fname in FILES.values():
        download_file(fname)

def process_weinreb():
    download_weinreb()
    print("Assembling dataset...")
    
    # 2. Load Raw Components
    # Counts (Cells x Genes)
    X = scipy.io.mmread(os.path.join(DATA_DIR, FILES["counts"])).tocsr()
    
    # Gene Names
    with gzip.open(os.path.join(DATA_DIR, FILES["genes"]), 'rt') as f:
        genes = [l.strip() for l in f]
        
    # Metadata (Time points, Cell types)
    meta = pd.read_csv(os.path.join(DATA_DIR, FILES["meta"]), sep='\t')
    
    # Clone Matrix (Cells x Clones) - Binary matrix
    clones = scipy.io.mmread(os.path.join(DATA_DIR, FILES["clones"])).tocsr()
    
    # 3. Construct AnnData
    adata = sc.AnnData(X=X, obs=meta)
    adata.var_names = genes
    
    # 4. Extract Clone IDs
    # We convert the sparse clone matrix into a single column 'clone_id'
    # Cells with no clone get -1. Cells with >1 clone (doublets) get -1 for safety.
    print("Processing lineage barcodes...")
    rows, cols = clones.nonzero()
    from collections import Counter
    
    # Count how many clones each cell belongs to
    cell_counts = Counter(rows)
    # Only keep "Singlets" (cells belonging to exactly 1 clone)
    valid_cells = {r for r, c in cell_counts.items() if c == 1}
    
    # Map cell_idx -> clone_idx
    clone_map = {r: c for r, c in zip(rows, cols) if r in valid_cells}
    
    # Assign to observation
    clone_ids = np.full(adata.n_obs, -1)
    for r, c in clone_map.items():
        clone_ids[r] = c
        
    adata.obs['clone_id'] = clone_ids
    
    # 5. Standardize Time
    # The raw metadata usually has 'Time point'
    if 'Time point' in adata.obs:
        adata.obs['time_point'] = adata.obs['Time point']
    
    # Save
    adata.write(FINAL_FILE)
    print(f"Success! Saved authoritative dataset to {FINAL_FILE}")
    print(f"  - Total Cells: {adata.n_obs}")
    print(f"  - Clonal Cells: {len(valid_cells)}")
    print(f"  - Unique Clones: {len(set(clone_map.values()))}")

if __name__ == "__main__":
    process_weinreb()