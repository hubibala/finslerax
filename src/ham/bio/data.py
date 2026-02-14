import jax.numpy as jnp
import numpy as np
import os
from typing import NamedTuple, Optional

try:
    import anndata
    import scanpy as sc
    import scvelo as scv
    HAS_BIO = True
except ImportError:
    HAS_BIO = False

class BioDataset(NamedTuple):
    X: jnp.ndarray
    V: Optional[jnp.ndarray]
    labels: Optional[jnp.ndarray]
    lineage_pairs: Optional[jnp.ndarray] = None 

class DataLoader:
    def __init__(self, path: Optional[str] = None, mode: str = 'real'):
        self.mode = mode
        self.adata = None
        
        if mode == 'real':
            if not HAS_BIO:
                raise ImportError("Please `pip install anndata scanpy scvelo`.")
            if path == 'weinreb' and os.path.exists('data/weinreb.h5ad'):
                 self.adata = anndata.read_h5ad('data/weinreb.h5ad')
            elif path and os.path.exists(path):
                self.adata = anndata.read_h5ad(path)
            else:
                 # Fallbacks or errors
                 pass

    def preprocess(self, n_top_genes: int = 2000, pca_components: int = 50):
        if self.mode == 'real' and self.adata is not None:
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes)
            self.adata = self.adata[:, self.adata.var.highly_variable]
            sc.tl.pca(self.adata, n_comps=pca_components)
            if 'velocity' not in self.adata.layers:
                try:
                    scv.pp.moments(self.adata)
                    scv.tl.velocity(self.adata, mode='stochastic')
                except Exception: pass
        return self

    def extract_lineage_pairs(self) -> jnp.ndarray:
        if self.adata is None or 'clone_id' not in self.adata.obs:
            return None
        obs = self.adata.obs
        clone_counts = obs['clone_id'].value_counts()
        valid_clones = clone_counts[clone_counts > 1].index
        pairs = []
        for clone in valid_clones:
            indices = np.where(obs['clone_id'] == clone)[0]
            # Simple assumption: All pairs in a clone are related. 
            # Ideally filter by time (parent t < child t)
            if 'time_point' in obs:
                 times = obs['time_point'].iloc[indices].values
                 min_t = np.min(times)
                 parents = indices[times == min_t]
                 children = indices[times > min_t]
                 for p in parents:
                     for c in children:
                         pairs.append([p, c])
        return jnp.array(pairs) if pairs else None

    def get_jax_data(self, use_pca: bool = True) -> BioDataset:
        if self.mode != 'real' or self.adata is None:
             N = 100
             return BioDataset(jnp.zeros((N, 50)), jnp.zeros((N, 50)), jnp.zeros(N), None)

        X_np = self.adata.obsm['X_pca'] if use_pca else self.adata.X
        if hasattr(X_np, "toarray"): X_np = X_np.toarray()
        
        V_np = None
        if 'velocity' in self.adata.layers:
             if use_pca and 'velocity_pca' in self.adata.obsm:
                 V_np = self.adata.obsm['velocity_pca']
        
        cats = self.adata.obs['cell_type'].cat.codes.values if 'cell_type' in self.adata.obs else np.zeros(X_np.shape[0])
        pairs = self.extract_lineage_pairs()

        return BioDataset(
            X=jnp.array(X_np),
            V=jnp.array(V_np) if V_np is not None else jnp.zeros_like(X_np),
            labels=jnp.array(cats),
            lineage_pairs=pairs
        )