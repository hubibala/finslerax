# Code Review: `examples/plot_weinreb_destinations.py`

**Reviewer:** Code Reviewer Agent
**Date:** 2025-05-15
**Arch Spec Version:** 1.1.0

## Summary

This plotting script visualises lineage-traced trajectories destined for specific cell types in the Weinreb dataset. It is reasonably well-structured for a single-file visualisation script, but has several **RISK**-level issues around missing error handling, a silent data fallback that will likely crash downstream, and an $O(n)$ Python loop for triple filtering that can be trivially vectorised. No outright **BUG**s that guarantee wrong output, but the data-loading path is fragile.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | `examples/plot_weinreb_destinations.py:9-11` | If `weinreb_preprocessed.h5ad` does not exist, the script silently falls back to `weinreb_raw.h5ad` with no log message and no check that the fallback file exists. If neither file exists, `anndata.read_h5ad` raises an unhandled `FileNotFoundError`. | Add an explicit `os.path.exists` check for the fallback path and raise a clear error if neither file is found. Print a warning when falling back. |
| 2 | **RISK** | `examples/plot_weinreb_destinations.py:10-11` → `46-47` | The fallback file (`weinreb_raw.h5ad`) is unlikely to contain `adata.obsm['velocity_pca']` or `adata.obsm['X_pca']`, which are computed during preprocessing. The script will crash with an unhelpful `KeyError` on line 46. | Guard the fallback: either refuse to run on raw data or check for required keys before proceeding. |
| 3 | **RISK** | `examples/plot_weinreb_destinations.py:19-22` | No `KeyError` handling for `adata.obs['SPRING-x']`, `adata.obs['SPRING-y']`, or `adata.obs['Cell type annotation']`. If the loaded file lacks these columns, the traceback will not explain what data is missing. | Validate required columns upfront and emit a descriptive error message. |
| 4 | **RISK** | `examples/plot_weinreb_destinations.py:65` | Triple filtering uses a Python-level list comprehension (`[t for t in triples if ...]`), which is $O(n)$ in pure Python. For large triple arrays this is unnecessarily slow. | Replace with vectorised NumPy: `mask = cell_types.values[triples[:, 2]] == target_type; target_triples = triples[mask]`. |
| 5 | **STYLE** | `examples/plot_weinreb_destinations.py:28` | Uses `plt.cm.get_cmap('nipy_spectral', num_types)` which is deprecated since Matplotlib 3.7. Line 26 uses a different spelling (`plt.get_cmap`), creating an inconsistency. | Use `plt.colormaps['nipy_spectral'].resampled(num_types)` (or `plt.colormaps['tab20']` on line 26) for forward-compatibility. |
| 6 | **STYLE** | `examples/plot_weinreb_destinations.py:46-47` | Inconsistent array copy: `v_pca` is copied with `[:]` but `x_pca` is a bare reference. Either both should be copied or neither. | Pick one convention and apply uniformly. |
| 7 | **STYLE** | `examples/plot_weinreb_destinations.py:83-85` | Legend label condition `i2[0]==triples_subset[0,0]` is always `True` by construction (`i2 = triples_subset[:, 0]`), making the ternary dead code. | Remove the condition and set the label string directly; the `by_label` deduplication on line 105 already handles duplicates. |
| 8 | **STYLE** | `examples/plot_weinreb_destinations.py:55, 74` | Uses legacy `np.random.seed(42)` (global state mutation). Re-seeding inside the loop on line 74 means every target type gets the same random subsample, which may or may not be intentional. | Use `rng = np.random.default_rng(42)` and pass `rng` to `rng.choice(...)`. |
| 9 | **STYLE** | `examples/plot_weinreb_destinations.py:112` | Output file is saved to CWD with a hardcoded name. No option to specify an output directory. | Accept an optional `--output` CLI argument or save relative to the script's directory. |

## Test Coverage Assessment

This is a standalone plotting/example script (`examples/`), not a library module. There is no corresponding test file in `tests/`, which is acceptable for visualisation utilities. No public API surface requires unit testing.

| Public Function | Tested? | Notes |
|-----------------|---------|-------|
| `main()` | No | Example script entry-point; no tests expected. |

## Positive Patterns

- **Numerical stability guards** — All norm-division operations include `+ 1e-8` safeguards (lines 96, 100, 103).
- **Global bandwidth** — Computing `global_sigma2` from a sample to avoid local density distortion (lines 54-58) is a sound approach that avoids bias in velocity projection.
- **Legend deduplication** — The `by_label = dict(zip(labels, handles))` pattern (line 105) is a clean way to avoid duplicate legend entries.
- **Subsampling before plotting** — Capping drawn trajectories at 150 (line 75) prevents visual clutter and keeps rendering fast.
