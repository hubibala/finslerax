# Code Review: `examples/plot_weinreb_cell_types.py`

**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary

This is a standalone visualization script that plots Weinreb hematopoiesis data on SPRING coordinates with cell-type annotations, RNA pseudo-velocity arrows, and lineage trajectories. The script is functional but has several robustness issues: the data-loading fallback silently proceeds to an invalid path, critical `obsm` keys are accessed without existence checks, deprecated API calls will break in upcoming library versions, and lineage-triple indices may fail on float-typed `.npy` files. Import organization and global random-state mutation are minor style concerns.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `examples/plot_weinreb_cell_types.py:9-12` | Fallback data path (`weinreb_raw.h5ad`) is never validated. If neither file exists, `anndata.read_h5ad` on line 12 throws an opaque `FileNotFoundError` with no indication that both paths were tried. | Add an explicit check: `if not os.path.exists(data_path): raise FileNotFoundError(f"Neither weinreb_preprocessed.h5ad nor weinreb_raw.h5ad found in data/")` after the fallback assignment. |
| 2 | **RISK** | `examples/plot_weinreb_cell_types.py:59` | `adata.obsm['velocity_pca']` is accessed without checking whether the key exists. Throws `KeyError` with no actionable message if the dataset was not preprocessed with scVelo or similar. | Guard with `if 'velocity_pca' not in adata.obsm: ...` and either skip the velocity section or raise a clear error. |
| 3 | **RISK** | `examples/plot_weinreb_cell_types.py:72` | Same issue for `adata.obsm['X_pca']`. | Same guard pattern. |
| 4 | **RISK** | `examples/plot_weinreb_cell_types.py:119-121` | Lineage-triple indices (`i2`, `i4`, `i6`) are extracted from `np.load(triples_path)` and used directly to index the coordinate arrays `x` and `y` (lines 125-132). If the `.npy` file stores floats (default for many pipelines), NumPy ≥ 1.25 will raise `IndexError: only integers...`. | Cast explicitly: `triples = np.load(triples_path).astype(int)`. |
| 5 | **RISK** | `examples/plot_weinreb_cell_types.py:32` | `plt.cm.get_cmap('nipy_spectral', num_types)` is deprecated since matplotlib 3.7 and will be removed in a future release. | Replace with `plt.colormaps['nipy_spectral'].resampled(num_types)`. Also update `plt.get_cmap('tab20')` on line 28 to `plt.colormaps['tab20']`. |
| 6 | **RISK** | `examples/plot_weinreb_cell_types.py:22` | `cell_types.replace('nan', 'Unknown', inplace=True)` uses the `inplace` parameter, which is deprecated since pandas 2.1 and scheduled for removal. | Replace with `cell_types = cell_types.replace('nan', 'Unknown')`. |
| 7 | **STYLE** | `examples/plot_weinreb_cell_types.py:71` | `from sklearn.neighbors import NearestNeighbors` is imported inside `main()` instead of at the top of the file. Delays `ImportError` discovery and violates PEP 8 import organization. | Move to the top-level import block (lines 1-4). |
| 8 | **STYLE** | `examples/plot_weinreb_cell_types.py:67` | `np.random.seed(42)` mutates the global random state. If this module is ever imported or called alongside other code, it silently changes their random behavior. | Use a local RNG: `rng = np.random.default_rng(42)` and replace `np.random.choice(...)` with `rng.choice(...)` on lines 69 and 115. |
| 9 | **STYLE** | `examples/plot_weinreb_cell_types.py:59` | `adata.obsm['velocity_pca'][:]` — the trailing `[:]` creates an unnecessary copy. `obsm` access already returns an array. | Remove `[:]`: `v_pca = adata.obsm['velocity_pca']`. |

## Test Coverage Assessment

This is a standalone visualization script in `examples/`, not a library module. There are no corresponding tests in `tests/`, which is expected for plotting scripts. No public API is exported.

| Function | Tested? | Notes |
|----------|---------|-------|
| `main()` | No | Script-level entry point; no unit tests expected. |

## Positive Patterns

- **Graceful degradation for trajectories:** The lineage-triples section (line 109) is guarded by `os.path.exists(triples_path)`, so the script produces a useful plot even when trajectory data is unavailable.
- **Subsampling for visual clarity:** Both velocity arrows (line 68, capped at 1,500) and trajectory lines (line 114, capped at 150) are subsampled to prevent overplotting, which is good visualization practice.
- **Epsilon guards in normalization:** Division operations consistently add `1e-8` (lines 86, 95, 100) to prevent division by zero, following a sound numerical-stability pattern.
- **Legend deduplication:** The `dict(zip(labels, handles))` pattern on line 136 correctly deduplicates legend entries when trajectories are overlaid on the cell-type scatter.
