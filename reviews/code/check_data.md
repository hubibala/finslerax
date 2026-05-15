# Code Review: `check_data`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`src/ham/bio/check_data.py` is a standalone diagnostic script (not a module) that loads a Weinreb `.h5ad` file and prints column diagnostics. It is **not imported by any module**, **not exported from `ham.bio`**, and **has zero test coverage**. The script uses top-level imperative code with a hardcoded relative path, a bare `except Exception`, and mutable global state — all of which violate ARCH_SPEC conventions. Because it is purely a developer utility and not part of the library API, the findings are mostly STYLE-level, with one RISK for the path-traversal-adjacent hardcoded relative path.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | `src/ham/bio/check_data.py:4` | Hardcoded relative path `"data/weinreb.h5ad"` is fragile — result depends on the working directory at invocation time. If the script is run from a different directory, the `os.path.exists` check silently passes, printing "File not found" with no actionable information. | Accept the path as a CLI argument (`sys.argv[1]`) or use `pathlib.Path(__file__).parent / ...` for a project-relative default. |
| 2 | **STYLE** | `src/ham/bio/check_data.py:1-27` | The entire file is top-level imperative code with no function boundary. This makes it impossible to `import` the module without executing side effects, violating ARCH_SPEC §1 ("no mutable global state"). | Wrap the logic in a `def main():` function with an `if __name__ == "__main__":` guard. |
| 3 | **STYLE** | `src/ham/bio/check_data.py:8` | `adata` is assigned as a module-level global variable upon import. Any accidental `from ham.bio.check_data import ...` will trigger the entire I/O and print cascade. | Eliminate module-level side effects (see #2). |
| 4 | **STYLE** | `src/ham/bio/check_data.py:22` | Bare `except Exception as e` catches everything (including `KeyboardInterrupt` on some Python versions, `SystemExit` if nested). For file I/O, catch `(OSError, ValueError)` specifically. | Replace with `except (OSError, ValueError) as e:`. |
| 5 | **STYLE** | `src/ham/bio/check_data.py:1` | `scanpy` is imported unconditionally, but it is not listed in `pyproject.toml` core dependencies (it is a heavy optional dependency). No guarded import or informative error if `scanpy` is missing. | Wrap in `try: import scanpy as sc except ImportError: raise ImportError("...")`. |
| 6 | **STYLE** | `src/ham/bio/check_data.py` | File is not referenced in `ham.bio.__init__.__all__` and is not imported anywhere in the library. It appears to be a leftover development/debugging script committed to the package source tree. | Move to `examples/` or `scripts/`, or delete if superseded by `examples/weinreb_smoke_test.py:check_dataset()`. |

## Test Coverage Assessment

| Public Symbol | Tested? | Notes |
|---------------|---------|-------|
| *(none — script has no public API)* | N/A | No functions or classes are defined. The entire file is imperative. |

**Gap analysis:** The file defines no importable API, so there are no functions to test. If the diagnostic logic is worth keeping, it should be refactored into a testable function.

## Positive Patterns

- The diagnostic checks (`has_clone`, `has_time`) search across multiple plausible column name variants, which is pragmatic for exploratory data loading.
- The `os.path.exists` guard prevents a noisy traceback when the data file is absent.
