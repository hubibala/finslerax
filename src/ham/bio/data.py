import jax.numpy as jnp
import numpy as np
import os
import pandas as pd  # <--- Added missing import
from typing import NamedTuple, Optional
from sklearn.neighbors import NearestNeighbors

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
            if path and os.path.exists(path):
                self.adata = anndata.read_h5ad(path)
            else:
                 pass

    def preprocess(self, n_top_genes: int = 2000, pca_components: int = 50):
        if self.mode == 'real' and self.adata is not None:
            # Basic cleanup
            if self.adata.n_vars > n_top_genes:
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                sc.pp.log1p(self.adata)
                sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes)
                self.adata = self.adata[:, self.adata.var.highly_variable]
            
            # PCA
            if 'X_pca' not in self.adata.obsm or self.adata.obsm['X_pca'].shape[1] != pca_components:
                sc.tl.pca(self.adata, n_comps=pca_components)
                
            # Velocity (if available)
            if 'velocity' not in self.adata.layers:
                try:
                    scv.pp.moments(self.adata)
                    scv.tl.velocity(self.adata, mode='stochastic')
                except Exception: pass
        return self

    def extract_lineage_pairs(self) -> jnp.ndarray:
        """
        Extracts constraints for Metric Learning.
        Priority 1: Ground Truth Clones (Weinreb)
        Priority 2: Pseudotime/Neighbor Heuristics (Pancreas)
        """
        if self.adata is None: return None
        obs = self.adata.obs
        
        # --- PRIORITY 1: GROUND TRUTH CLONES ---
        if 'clone_id' in obs:
            print(">>> FOUND GROUND TRUTH CLONES. Using biological lineage.")
            
            # Check dtype safely using pandas
            if pd.api.types.is_numeric_dtype(obs['clone_id']):
                valid_mask = obs['clone_id'] != -1
            else:
                valid_mask = obs['clone_id'].notna() & (obs['clone_id'] != '-1')
            
            # We only care about clones that appear more than once
            clone_counts = obs.loc[valid_mask, 'clone_id'].value_counts()
            valid_clones = clone_counts[clone_counts > 1].index
            
            pairs = []
            
            # Speed up: Group by clone_id first
            df_clones = obs.loc[valid_mask, ['clone_id', 'time_point']].copy()
            
            # Map global indices
            # We need absolute indices for the JAX array
            df_clones['global_idx'] = np.arange(len(obs))[valid_mask]
            
            for clone in valid_clones:
                subset = df_clones[df_clones['clone_id'] == clone]
                indices = subset['global_idx'].values
                times = subset['time_point'].values
                
                # Link Early -> Late
                unique_times = np.sort(np.unique(times))
                
                if len(unique_times) > 1:
                    for i in range(len(unique_times) - 1):
                        t_current = unique_times[i]
                        t_next = unique_times[i+1]
                        
                        parents = indices[times == t_current]
                        children = indices[times == t_next]
                        
                        for p in parents:
                            for c in children:
                                pairs.append([p, c])
                        
            if len(pairs) > 0:
                print(f">>> Extracted {len(pairs)} Ground Truth Lineage Pairs.")
                return jnp.array(pairs)
            else:
                print(">>> Clones found, but no time-separated pairs. Falling back...")

        # --- PRIORITY 2: PSEUDOTIME HEURISTIC ---
        time_col = None
        for c in ['palantir_pseudotime', 'dpt_pseudotime', 'pseudotime']:
            if c in obs:
                time_col = c
                break
        
        if time_col:
            print(f">>> NO CLONES FOUND. Falling back to heuristic: '{time_col}' neighbors.")
            return self._extract_pseudotime_pairs(time_col)
            
        return None

    def _extract_pseudotime_pairs(self, time_col: str, n_pairs: int = 5000) -> jnp.ndarray:
        t = self.adata.obs[time_col].values
        X = self.adata.obsm['X_pca']
        
        nbrs = NearestNeighbors(n_neighbors=15).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        pairs = []
        valid_indices = np.where(~np.isnan(t))[0]
        
        # Sample to keep computation fast
        sample_idxs = np.random.choice(valid_indices, min(len(valid_indices), 10000), replace=False)
        
        for i in sample_idxs:
            my_t = t[i]
            for neighbor_idx in indices[i]:
                their_t = t[neighbor_idx]
                # Enforce forward flow
                if their_t > my_t + 0.02: 
                    pairs.append([i, neighbor_idx])
                    
        if len(pairs) == 0: return None
            
        pairs_np = np.array(pairs)
        if len(pairs_np) > n_pairs:
            idx = np.random.choice(len(pairs_np), n_pairs, replace=False)
            pairs_np = pairs_np[idx]
            
        print(f">>> Generated {len(pairs_np)} heuristic flow pairs.")
        return jnp.array(pairs_np)

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
        
        if 'clusters' in self.adata.obs:
             cats = self.adata.obs['clusters'].astype('category').cat.codes.values
        elif 'cell_type' in self.adata.obs:
            cats = self.adata.obs['cell_type'].astype('category').cat.codes.values
        else:
            cats = np.zeros(X_np.shape[0])
            
        pairs = self.extract_lineage_pairs()

        return BioDataset(
            X=jnp.array(X_np),
            V=jnp.array(V_np) if V_np is not None else jnp.zeros_like(X_np),
            labels=jnp.array(cats),
            lineage_pairs=pairs
        )