import jax.numpy as jnp
import numpy as np
import warnings

# Try importing single-cell libraries
try:
    import anndata
    import scanpy as sc
    import scvelo as scv  # Added scvelo support
    HAS_BIO = True
except ImportError:
    HAS_BIO = False
    
from typing import NamedTuple, Optional

class BioDataset(NamedTuple):
    X: jnp.ndarray  # Gene Expression (Cells, Genes)
    V: Optional[jnp.ndarray] # RNA Velocity (Cells, Genes)
    labels: Optional[jnp.ndarray] # Cell Types / Clusters

class DataLoader:
    """
    Universal Data Loader: Works for real .h5ad files OR synthetic simulation.
    """
    def __init__(self, path: Optional[str] = None, mode: str = 'real'):
        self.mode = mode
        if mode == 'real':
            if not HAS_BIO:
                raise ImportError("Please `pip install anndata scanpy scvelo` to load real files.")
            
            # If path is a keyword, fetch from web
            if path == 'pancreas':
                print("Downloading Pancreas dataset via scVelo...")
                self.adata = scv.datasets.pancreas()
            elif path == 'dentategyrus':
                print("Downloading Dentate Gyrus dataset via scVelo...")
                self.adata = scv.datasets.dentategyrus()
            else:
                self.adata = anndata.read_h5ad(path)
                
            print(f"Loaded AnnData: {self.adata.shape}")
        
        elif mode == 'synthetic':
            print("Initializing Synthetic Biological Organism...")
            self.adata = None

    def preprocess(self, n_top_genes: int = 2000, pca_components: int = 50):
        if self.mode == 'real':
            print("Preprocessing Real Data...")
            
            # 1. Standard Scanpy Flow
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes)
            self.adata = self.adata[:, self.adata.var.highly_variable]
            
            # 2. PCA
            sc.tl.pca(self.adata, n_comps=pca_components)
            print(f"Reduced to {pca_components} PCs.")
            
            # 3. Velocity Handling (Critical for HAM)
            # If we downloaded raw data, we might need to compute moments/velocity on the fly
            if 'velocity' not in self.adata.layers:
                print("Computing RNA Velocity moments...")
                try:
                    scv.pp.moments(self.adata)
                    scv.tl.velocity(self.adata, mode='stochastic')
                    print("Velocity estimated.")
                except Exception as e:
                    print(f"Warning: Could not compute velocity ({e}). V will be None.")
                    
        return self

    def get_jax_data(self, use_pca: bool = True) -> BioDataset:
        if self.mode == 'synthetic':
            # ... (Synthetic code from previous turn) ...
            return self.get_synthetic_data()
            
        # REAL DATA EXTRACTION
        if use_pca:
            X_np = self.adata.obsm['X_pca']
        else:
            X_np = self.adata.X
            if hasattr(X_np, "toarray"): X_np = X_np.toarray()
            
        # Extract Velocity (V)
        # We need V to be in the same space as X (e.g., PCA space)
        V_np = None
        if 'velocity' in self.adata.layers:
            # If we are in PCA mode, we need projected velocity
            if use_pca:
                # Project velocity into PCA space: V_pca = V_genes @ PCA_components.T
                # scVelo usually stores this in 'velocity_pca' if run, but we can compute it manually
                if 'velocity_pca' in self.adata.obsm:
                    V_np = self.adata.obsm['velocity_pca']
                else:
                    # Manual Projection
                    # V_genes (Cells, Genes)
                    V_genes = self.adata.layers['velocity']
                    if hasattr(V_genes, "toarray"): V_genes = V_genes.toarray()
                    # Only keep HVG columns if filtering happened... 
                    # Note: This is tricky. Let's assume 'velocity' layer is aligned with X.
                    
                    # PCA Components (PCs, Genes) -> (Genes, PCs)
                    pca_comps = self.adata.varm['PCs']
                    
                    # Check shapes
                    if V_genes.shape[1] == pca_comps.shape[0]:
                         V_np = V_genes @ pca_comps
                    else:
                        print("Warning: Velocity gene count mismatch. Returning None.")
            else:
                V_np = self.adata.layers['velocity']
                if hasattr(V_np, "toarray"): V_np = V_np.toarray()

        # Labels
        if 'clusters' in self.adata.obs:
            cats = self.adata.obs['clusters'].cat.codes.values
        elif 'leiden' in self.adata.obs:
            cats = self.adata.obs['leiden'].cat.codes.values
        elif 'celltype' in self.adata.obs: # Common in Pancreas
            cats = self.adata.obs['celltype'].cat.codes.values
        else:
            cats = np.zeros(X_np.shape[0])

        return BioDataset(
            X=jnp.array(X_np), 
            V=jnp.array(V_np) if V_np is not None else None,
            labels=jnp.array(cats)
        )
    
    # ... (Include get_synthetic_data from before) ...