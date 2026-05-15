# Code Review: `examples/preprocess_weinreb.py`

**Reviewer:** Code Reviewer Agent
**Date:** 2025-05-15
**Arch Spec Version:** 1.1.0

## Summary

A well-structured preprocessing script that derives clonal pseudo-velocity from the Weinreb hematopoiesis dataset and stores it in `.h5ad` format for downstream HAMTools training. The core pipeline logic is correct, but the script has one output-level bug (referencing a nonexistent file), several risks around silent data loss and documentation/implementation mismatch, and multiple unused symbols. The biggest structural weakness is the absence of a `__main__` guard, which makes the file unimportable and untestable.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `examples/preprocess_weinreb.py:261` | Final summary prints `data/weinreb_lineage_pairs.npy` in the `else` branch, but no code path ever creates this file. The file produces misleading terminal output when `HAS_TRIPLES` is `False`. | Change the message to accurately reflect the state, e.g. `"No day2→day4→day6 triples were found."`, or remove the phantom filename. |
| 2 | **RISK** | `examples/preprocess_weinreb.py:13` vs `:179` | Module docstring (line 13) states velocities are _"weighted by a Gaussian kernel over SPRING distance"_, but the implementation (line 179) uses an unweighted mean: `X_pca[t_later_cells].mean(axis=0)`. This discrepancy could mislead downstream consumers about the nature of the velocity signal. | Either implement the Gaussian kernel weighting described in the docstring, or update the docstring to say _"unweighted mean of PCA displacements to later-timepoint clonal relatives"_. |
| 3 | **RISK** | `examples/preprocess_weinreb.py:71-75` | `pd.to_numeric(..., errors='coerce')` silently converts unparseable `time_point` values to `NaN`. If input data uses an unexpected label format (e.g. `'day2'`), cells are silently dropped from all downstream clonal analyses with no warning or count of how many were lost. | After coercion, count NaN values and emit a warning: `n_nan = adata.obs['time_point'].isna().sum(); if n_nan: print(f"WARNING: {n_nan} time_point values could not be parsed")`. |
| 4 | **RISK** | `examples/preprocess_weinreb.py:50-80` | All pipeline logic runs at module level with no `if __name__ == "__main__":` guard. The file cannot be imported (e.g. for testing or reuse of the `inspect` helper) without executing the entire pipeline and triggering `sys.exit`. | Wrap lines 50–268 in a `def main(): ...` function and add `if __name__ == "__main__": main()` at the end. |
| 5 | **RISK** | `examples/preprocess_weinreb.py:143` | `np.random.seed(42)` mutates the global NumPy random state. Any other code that relies on NumPy's global RNG before or after this script in the same process will be affected. | Use a local `rng = np.random.default_rng(42)` and call `rng.shuffle(unique_clones)`. |
| 6 | **STYLE** | `examples/preprocess_weinreb.py:28` | `from sklearn.neighbors import NearestNeighbors` is imported but never used anywhere in the file. | Remove the unused import. |
| 7 | **STYLE** | `examples/preprocess_weinreb.py:93-96` | Variable `is_sparse` is assigned (`True`/`False`) but never referenced. | Remove the dead code block. |
| 8 | **STYLE** | `examples/preprocess_weinreb.py:117,183` | Counter `n_with_velocity` is incremented in the loop but never used in any output or return value. | Remove the variable, or use it in the summary print. |
| 9 | **STYLE** | `examples/preprocess_weinreb.py:2` | Docstring header says `preprocess_weinreb_spring.py` but the actual filename is `preprocess_weinreb.py`. | Update to `preprocess_weinreb.py`. |
| 10 | **STYLE** | `examples/preprocess_weinreb.py:249` | The "Expected" nonzero-velocity fraction is printed as `n_cloned / adata.n_obs` (all cloned cells), but only training-set clones receive nonzero velocity. The comparison is misleading. | Print `len(train_clones) / len(unique_clones) * n_cloned / adata.n_obs` or clarify the message says "all cloned" vs "train cloned". |
| 11 | **STYLE** | `examples/preprocess_weinreb.py:30-31,233-234` | File paths (`RAW_PATH`, `OUT_PATH`, `.npy` saves) are hardcoded relative strings. The script must be launched from the repo root to work. | Accept paths via `argparse` or at least document the required working directory. |

## Test Coverage Assessment

| Public Symbol | Tested? | Notes |
|---|---|---|
| `inspect()` | No | Only helper; low priority. |
| Script pipeline (overall) | No | No test file exists for `preprocess_weinreb.py`. There is no `tests/test_preprocess*.py`. |

**Gap analysis:** The script has zero automated test coverage. Because all logic is at module level (Issue #4), it is currently impossible to unit-test individual stages (PCA, velocity, triple extraction) in isolation. Wrapping the pipeline in functions would make it straightforward to add integration tests that run against a small synthetic `.h5ad` fixture.

## Positive Patterns

1. **Clear section headers** — the `═══` / numbered section layout makes the pipeline easy to follow.
2. **Validation at entry** — required obs columns are checked immediately after load with a clear error message and `sys.exit(1)`.
3. **Train/test split by clone, not by cell** — prevents data leakage across the split boundary.
4. **Final assertions** — shape and key-existence assertions before saving provide a lightweight correctness gate.
5. **Informative progress output** — each stage prints a summary, making it easy to diagnose preprocessing issues from logs.
