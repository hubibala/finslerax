# Code Review: `ham.bio.data`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`data.py` is a data-ingestion module that wraps AnnData/scanpy/scvelo for single-cell biology datasets and produces `BioDataset` NamedTuples for downstream JAX consumption. It does not contain JAX transforms itself but serves as the boundary between raw file I/O and the JAX-based training stack. The module has **no dedicated test file** — only `BioDataset` construction is exercised in `test_joint_training.py`. The `DataLoader` class is entirely untested. Several issues relate to security of file I/O, non-deterministic random state, silent error swallowing, and missing test coverage.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | [data.py](src/ham/bio/data.py#L140) | `np.random.choice` in `_extract_pseudotime_pairs` uses the global NumPy RNG with no seed. Calls to this function produce non-reproducible training data, violating JAX deterministic-pipeline expectations. The same issue appears at L152. | Accept an `rng_seed` or `np.random.Generator` parameter; or at minimum use `np.random.default_rng(seed)` with a fixed default. |
| 2 | **BUG** | [data.py](src/ham/bio/data.py#L86) | `df_clones['global_idx'] = np.arange(len(obs))[valid_mask]` — the boolean mask `valid_mask` is a pandas Series, but it is used to index a NumPy integer arange. If `obs.index` is non-default (e.g., after subsetting an AnnData), `valid_mask` alignment may silently produce wrong indices. | Convert to NumPy: `valid_mask_np = valid_mask.values`; then `np.arange(len(obs))[valid_mask_np]`. |
| 3 | **RISK** | [data.py](src/ham/bio/data.py#L50) | `except Exception: pass` silently swallows all scvelo velocity-computation errors, including `ValueError`, `LinAlgError`, and out-of-memory errors. A user will never know velocity estimation failed. | At minimum log the exception: `except Exception as e: warnings.warn(f"Velocity estimation failed: {e}")`. |
| 4 | **RISK** | [data.py](src/ham/bio/data.py#L29) | `anndata.read_h5ad(path)` reads an arbitrary HDF5 file from a user-supplied path with no validation. While h5ad files are a domain standard, HDF5 deserialization can trigger pickle-based code execution if the file contains malicious objects (via `uns` dict). This is a system-boundary I/O operation. | Validate file extension, add a warning in docstring about untrusted files, or pass `backed='r'` for initial inspection before full load. |
| 5 | **RISK** | [data.py](src/ham/bio/data.py#L162) | `get_jax_data` returns `jnp.zeros_like(X_np)` as velocity when velocity is unavailable (L175). Downstream code (e.g., `ZermeloAlignmentLoss`) may treat this as real velocity data, producing misleading alignment loss values rather than signaling the absence of velocity. | Return `None` for `V` and let downstream losses check, or set a flag in `BioDataset` indicating whether velocity is real vs. synthetic zeros. |
| 6 | **RISK** | [data.py](src/ham/bio/data.py#L165) | `if hasattr(X_np, "toarray"): X_np = X_np.toarray()` — for very large sparse matrices, calling `.toarray()` will materialize the full dense array and may OOM. No size check is performed. | Add a size guard or warn when the dense array would exceed a threshold (e.g., > 1 GB). |
| 7 | **RISK** | [data.py](src/ham/bio/data.py#L36) | `preprocess` mutates `self.adata` in-place (`self.adata = self.adata[:, self.adata.var.highly_variable]`) and applies normalization/log1p without checking if these steps have already been applied. Calling `preprocess()` twice will double-normalize the data. | Add a guard flag (`self._preprocessed`) or check `adata.uns` for prior normalization markers. |
| 8 | **STYLE** | [data.py](src/ham/bio/data.py#L56-L127) | `extract_lineage_pairs` and `_extract_pseudotime_pairs` use `print()` for status messages. The rest of the HAM codebase does not use `print` for logging. | Replace with `logging.info()` or `warnings.warn()` for consistency. |
| 9 | **STYLE** | [data.py](src/ham/bio/data.py#L1) | `import jax.numpy as jnp` is imported at the top level but only used at conversion boundaries (`jnp.array(...)`, `jnp.zeros`). The module is fundamentally a NumPy/pandas data-wrangling module. This is fine but the JAX import adds unnecessary startup cost when JAX is not yet needed. | Minor — acceptable for a JAX-native library. No action needed. |
| 10 | **STYLE** | [data.py](src/ham/bio/data.py#L30) | Extra indentation on `pass` inside the `else` block of `__init__` (two leading spaces instead of the project-standard four). | Fix indentation to match surrounding code. |

---

## Test Coverage Assessment

| Public API | Tested? | Notes |
|---|---|---|
| `BioDataset` (NamedTuple) | **Partial** | Constructed manually in `test_joint_training.py:73` with synthetic data. No test validates field types or `None` defaults. |
| `DataLoader.__init__` | **No** | Not tested at all. |
| `DataLoader.preprocess` | **No** | Not tested at all. |
| `DataLoader.extract_lineage_pairs` | **No** | Not tested — neither clone-based nor pseudotime paths. |
| `DataLoader._extract_pseudotime_pairs` | **No** | Not tested. |
| `DataLoader.get_jax_data` | **No** | Not tested. |

**Gap Analysis:** `DataLoader` has zero test coverage. This is the highest-priority testing gap in the bio module. A dedicated `tests/test_data.py` should be created covering:
1. Construction with a mock AnnData object (no file I/O needed).
2. `preprocess` idempotency.
3. `extract_lineage_pairs` for both clone-based and pseudotime paths.
4. `get_jax_data` output shapes and dtypes.
5. Edge case: empty AnnData, missing velocity layer, missing PCA.

---

## Positive Patterns

1. **Clean data boundary.** `BioDataset` as a `NamedTuple` with typed fields provides a clear, immutable contract between I/O and JAX computation. This aligns well with the Metric-First design.
2. **Graceful optional dependencies.** The `try/except ImportError` guard for `anndata`, `scanpy`, and `scvelo` is well-structured and provides a clear error message.
3. **Priority-based lineage extraction.** The two-tier strategy (ground-truth clones → pseudotime heuristic) is a sensible design that degrades gracefully when gold-standard labels are unavailable.
4. **Sparse matrix handling.** The `.toarray()` check for sparse `X` prevents downstream JAX failures from scipy sparse inputs.
