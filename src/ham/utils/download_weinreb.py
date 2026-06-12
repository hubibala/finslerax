"""Utility script for downloading and processing the Weinreb et al. (2020) dataset.

Downloads raw counts, metadata, and clone tracking matrices from the Klein Lab
repository and assembles them into an authoritative AnnData object for lineage
tracing experiments.
"""

import os
import urllib.request
import gzip
import logging
from typing import Optional
from collections import Counter

import pandas as pd
import scipy.io
import scanpy as sc
import numpy as np

logger = logging.getLogger(__name__)

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

def download_file(filename: str) -> Optional[str]:
    """Download a single file from the Klein Lab repository.

    Args:
        filename: Name of the file to download.

    Returns:
        The local file path if successful, otherwise None.
    """
    url = f"{BASE_URL}/{filename}"
    local_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(local_path):
        logger.info(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, local_path)
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None
    return local_path

def download_weinreb() -> None:
    """Download all raw components of the Weinreb dataset."""
    if not os.path.exists(DATA_DIR): 
        os.makedirs(DATA_DIR)
    
    logger.info("Fetching raw data from Klein Lab...")
    for fname in FILES.values():
        download_file(fname)

def process_weinreb() -> None:
    """Download and assemble the Weinreb dataset into an AnnData object.
    
    Processes the raw count matrix, adds gene and cell metadata, and parses
    the clone matrix to assign a valid clone ID to each singlet cell. Doublets
    and unassigned cells are assigned a clone ID of -1.
    """
    download_weinreb()
    logger.info("Assembling dataset...")
    
    # 2. Load Raw Components
    # Counts (Cells x Genes)
    counts_path = os.path.join(DATA_DIR, FILES["counts"])
    X = scipy.io.mmread(counts_path).tocsr()
    
    # Gene Names
    genes_path = os.path.join(DATA_DIR, FILES["genes"])
    with gzip.open(genes_path, 'rt') as f:
        genes = [l.strip() for l in f]
        
    # Metadata (Time points, Cell types)
    meta_path = os.path.join(DATA_DIR, FILES["meta"])
    meta = pd.read_csv(meta_path, sep='\t')
    
    # Clone Matrix (Cells x Clones) - Binary matrix
    clones_path = os.path.join(DATA_DIR, FILES["clones"])
    clones = scipy.io.mmread(clones_path).tocsr()
    
    # 3. Construct AnnData
    adata = sc.AnnData(X=X, obs=meta)
    adata.var_names = genes
    
    # 4. Extract Clone IDs
    # We convert the sparse clone matrix into a single column 'clone_id'
    # Cells with no clone get -1. Cells with >1 clone (doublets) get -1 for safety.
    logger.info("Processing lineage barcodes...")
    rows, cols = clones.nonzero()
    
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
    logger.info(f"Success! Saved authoritative dataset to {FINAL_FILE}")
    logger.info(f"  - Total Cells: {adata.n_obs}")
    logger.info(f"  - Clonal Cells: {len(valid_cells)}")
    logger.info(f"  - Unique Clones: {len(set(clone_map.values()))}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_weinreb()