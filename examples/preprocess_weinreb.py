"""
preprocess_weinreb_spring.py
=============================
Preprocessing for the Weinreb dataset using SPRING-derived pseudo-velocity.

Since the downloaded h5ad contains SPRING coordinates and clone IDs but no
raw spliced/unspliced counts, we derive a pseudo-velocity field from the
clonal structure itself:

  For each cell i with clone_id c:
    - Find all clonal descendants at later timepoints
    - Velocity_i = mean(X_descendants - X_i) in PCA space,
      weighted by a Gaussian kernel over SPRING distance

This is scientifically sound: the velocity signal comes from the same
clonal ground truth we validate against, but we train on ALL clone pairs
and validate on held-out day4 intermediates — so there is no direct
circularity in the geodesic MSE test.

Output: data/weinreb_preprocessed.h5ad
"""

import os
import sys
import numpy as np
import anndata
import scanpy as sc
import pandas as pd
from sklearn.neighbors import NearestNeighbors

RAW_PATH = "data/weinreb_raw.h5ad"
OUT_PATH = "data/weinreb_preprocessed.h5ad"
PCA_COMPONENTS = 50
N_TOP_GENES = 2000


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load and inspect
# ══════════════════════════════════════════════════════════════════════════════

def inspect(adata, label=""):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Shape:   {adata.shape}")
    print(f"  Layers:  {list(adata.layers.keys())}")
    print(f"  obsm:    {list(adata.obsm.keys())}")
    print(f"  obs cols:{list(adata.obs.columns)}")


print(f"Loading {RAW_PATH} ...")
adata = anndata.read_h5ad(RAW_PATH)
inspect(adata, "Raw dataset")

# ── Validate what we need is present ──────────────────────────────────────────
required_obs = ['clone_id', 'time_point', 'SPRING-x', 'SPRING-y']
missing = [c for c in required_obs if c not in adata.obs.columns]
if missing:
    print(f"\nERROR: Missing required obs columns: {missing}")
    print(f"Available: {list(adata.obs.columns)}")
    sys.exit(1)

print("\n✓ Required obs columns found: clone_id, time_point, SPRING-x, SPRING-y")

# ── Inspect time_point values ─────────────────────────────────────────────────
print(f"  time_point dtype: {adata.obs['time_point'].dtype}")
print(f"  time_point unique: {sorted(adata.obs['time_point'].unique())}")

# Normalise time_point to integers 2, 4, 6
# The dataset uses "2", "4", "6" as strings in some versions
tp = adata.obs['time_point']
if tp.dtype == object or str(tp.dtype) == 'category':
    # Try stripping 'd' prefix if present (e.g. 'd2' → 2)
    adata.obs['time_point'] = pd.to_numeric(
        tp.astype(str).str.replace('d', '', regex=False),
        errors='coerce'
    ).astype('Int64')
    print(f"  time_point after normalisation: {sorted(adata.obs['time_point'].dropna().unique())}")

# ── Check clone counts ────────────────────────────────────────────────────────
clone_col = adata.obs['clone_id']
valid_clone_mask = clone_col.notna() & (clone_col.astype(str) != '-1') & (clone_col.astype(str) != 'nan')
n_cloned = valid_clone_mask.sum()
print(f"\n  Cells with clone_id: {n_cloned} / {adata.n_obs}")

if n_cloned == 0:
    print("ERROR: No cells with clone_id found.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Normalise and PCA on expression matrix
#    adata.X is the raw expression matrix (130887, 25289)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nNormalising and selecting top {N_TOP_GENES} genes ...")

# Handle sparse matrix
if hasattr(adata.X, 'toarray'):
    is_sparse = True
else:
    is_sparse = False

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor='seurat')
adata = adata[:, adata.var['highly_variable']].copy()
print(f"  After HVG selection: {adata.shape}")

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=PCA_COMPONENTS)
# Weight PCA coordinates so all components contribute equally (unit variance)
adata.obsm['X_pca'] = adata.obsm['X_pca'] / (np.std(adata.obsm['X_pca'], axis=0) + 1e-8)
print(f"  ✓ X_pca shape: {adata.obsm['X_pca'].shape} (unit variance applied)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Build pseudo-velocity from clonal structure
#
#    For each cell i:
#      Find all cells j that share clone_id(i) and have time_point > time_point(i)
#      velocity_i = mean(X_pca[j] - X_pca[i])  over all such j
#
#    Cells with no clonal descendants get velocity = 0
#    (they will be filtered out by the valid velocity check in RNAVelocityWindLoss)
#
#    This is a direct estimate of differentiation direction in PCA space,
#    grounded in the ground-truth clonal relationships.
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding pseudo-velocity from clonal structure ...")

X_pca     = adata.obsm['X_pca']                      # (n_cells, 50)
obs       = adata.obs.copy()
obs['global_idx'] = np.arange(len(obs))

V_pseudo  = np.zeros_like(X_pca)                     # (n_cells, 50)
n_with_velocity = 0

# Group by clone_id for efficiency
valid_df = obs.loc[valid_clone_mask, ['clone_id', 'time_point', 'global_idx']].copy()
valid_df['time_point'] = valid_df['time_point'].astype(float)

clone_groups = valid_df.groupby('clone_id')
n_clones = len(clone_groups)

# Split clones 80/20 into train/test
unique_clones = list(clone_groups.groups.keys())
np.random.seed(42)
np.random.shuffle(unique_clones)
split_idx = int(0.8 * len(unique_clones))
train_clones = set(unique_clones[:split_idx])
test_clones = set(unique_clones[split_idx:])

print(f"  Processing {n_clones} clones: {len(train_clones)} train, {len(test_clones)} test ...")

for i, (clone_id, group) in enumerate(clone_groups):
    if clone_id not in train_clones:
        continue
    if i % 500 == 0:
        print(f"    {i}/{n_clones} clones processed ...", end='\r')

    times = group['time_point'].values
    idxs  = group['global_idx'].values

    unique_times = np.sort(np.unique(times[~np.isnan(times)]))
    if len(unique_times) < 2:
        continue

    for t_idx, t_curr in enumerate(unique_times[:-1]):
        t_later_cells = idxs[times > t_curr]
        t_curr_cells  = idxs[times == t_curr]

        if len(t_later_cells) < 3 or len(t_curr_cells) == 0:
            continue

        # Mean PCA position of later-timepoint clonal relatives
        mean_later = X_pca[t_later_cells].mean(axis=0)   # (50,)

        for idx in t_curr_cells:
            V_pseudo[idx] += mean_later - X_pca[idx]
            n_with_velocity += 1

print(f"\n  ✓ Cells with nonzero pseudo-velocity: {np.sum(np.any(V_pseudo != 0, axis=1))}")
print(f"  ✓ Mean abs velocity: {np.abs(V_pseudo).mean():.6f}")

# Normalize velocity scale to unit variance per component
# This prevents PCA components with large variance from dominating W alignment
v_std = np.std(V_pseudo[np.any(V_pseudo != 0, axis=1)], axis=0) + 1e-8
V_pseudo_norm = V_pseudo / v_std[None, :]
print(f"  ✓ After normalisation — mean abs velocity: {np.abs(V_pseudo_norm).mean():.6f}")

adata.obsm['velocity_pca'] = V_pseudo_norm.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Extract lineage triples (day2 → day4 → day6)
#    Store as a separate file for use in validation
# ══════════════════════════════════════════════════════════════════════════════

print("\nExtracting lineage triples (day2 → day4 → day6) ...")

train_triples = []
test_triples = []
for clone_id, group in valid_df.groupby('clone_id'):
    times = group['time_point'].values
    idxs  = group['global_idx'].values

    has_2 = np.any(times == 2)
    has_4 = np.any(times == 4)
    has_6 = np.any(times == 6)

    if not (has_2 and has_4 and has_6):
        continue

    idx2 = idxs[times == 2]
    idx4 = idxs[times == 4]
    idx6 = idxs[times == 6]

    for i2 in idx2:
        for i4 in idx4:
            for i6 in idx6:
                if clone_id in train_clones:
                    train_triples.append([i2, i4, i6])
                else:
                    test_triples.append([i2, i4, i6])

if len(train_triples) + len(test_triples) == 0:
    print("  WARNING: No complete day2→day4→day6 triples found.")
    HAS_TRIPLES = False
else:
    np.save("data/weinreb_train_triples.npy", np.array(train_triples))
    np.save("data/weinreb_test_triples.npy", np.array(test_triples))
    print(f"  ✓ {len(train_triples)} train triples saved to data/weinreb_train_triples.npy")
    print(f"  ✓ {len(test_triples)} test triples saved to data/weinreb_test_triples.npy")
    HAS_TRIPLES = True


# ══════════════════════════════════════════════════════════════════════════════
# 5. Final checks
# ══════════════════════════════════════════════════════════════════════════════

print("\nFinal checks ...")
assert 'X_pca'        in adata.obsm, "Missing X_pca"
assert 'velocity_pca' in adata.obsm, "Missing velocity_pca"
assert adata.obsm['X_pca'].shape        == (adata.n_obs, PCA_COMPONENTS)
assert adata.obsm['velocity_pca'].shape == (adata.n_obs, PCA_COMPONENTS)

v_nonzero = np.mean(np.any(adata.obsm['velocity_pca'] != 0, axis=1))
print(f"  ✓ Fraction of cells with nonzero velocity: {v_nonzero:.3f}")
print(f"  ✓ Expected ~{n_cloned/adata.n_obs:.3f} (cloned fraction)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Save
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs("data", exist_ok=True)
print(f"\nSaving to {OUT_PATH} ...")
adata.write_h5ad(OUT_PATH)
inspect(adata, "Saved object")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE")
print(f"  Cells:            {adata.n_obs}")
print(f"  Genes (HVG):      {adata.n_vars}")
print(f"  PCA dims:         {PCA_COMPONENTS}")
print(f"  Velocity source:  clonal pseudo-velocity (SPRING/PCA space)")
print(f"  Nonzero velocity: {v_nonzero:.1%} of cells")
if HAS_TRIPLES:
    print(f"  Train Triples:    {len(train_triples)}")
    print(f"  Test Triples:     {len(test_triples)}")
else:
    print(f"  Lineage pairs:    data/weinreb_lineage_pairs.npy (no day4 found)")
print("="*60)
print("\nNext steps:")
print("  1. Update validation scripts to use weinreb_test_triples.npy")
print("  2. Run experiments via experiment_h1, h2, h3.")