# Documentation Review: `src/ham/bio/check_data.py`

**Reviewer:** Doc Reviewer Agent
**Date:** 2025-05-15

## Summary

Overall documentation quality: **needs work**.

`check_data.py` is a 28-line standalone diagnostic script with **zero documentation** — no module docstring, no inline comments explaining its purpose, no usage instructions, and no type annotations. It is not part of the public API (not exported from `ham.bio.__init__.py`, not listed in `spec/ARCH_SPEC.md § 5`), but it ships inside the `src/ham/bio/` package rather than in `examples/` or a `scripts/` directory, which makes its status ambiguous.

The file's only reference in the project is as a review target in `.github/agents/review-orchestrator.agent.md:45`. A comparable function (`check_dataset`) already exists in `examples/weinreb_smoke_test.py:135`, raising the question of whether this file is redundant.

## Issue Tracker

| # | Severity | Symbol / Section | Issue | Suggested Text / Recommended Action |
|---|----------|-----------------|-------|--------------------------------------|
| 1 | **MISSING** | Module docstring | The file has no module-level docstring. Its purpose (quick diagnostic for the Weinreb `.h5ad` file) is undocumented. A reader encountering this file in `src/ham/bio/` cannot determine whether it is a library module, a test, or a standalone script without reading every line. | Add a module docstring: `"""Quick diagnostic script for inspecting the Weinreb hematopoiesis dataset. Not part of the public API — run directly: python -m ham.bio.check_data or python src/ham/bio/check_data.py. Checks that the AnnData object contains expected observation columns (clone/lineage info, time-point info)."""` |
| 2 | **MISSING** | Script guard | The script executes on import because it lacks an `if __name__ == "__main__":` guard. If accidentally imported (e.g., `from ham.bio import check_data`), it runs side-effects immediately. This is both a usability issue and a documentation gap — the absence of a guard obscures the intended usage. | Wrap the executable body in `if __name__ == "__main__":` and document that this file is meant to be run as a script. |
| 3 | **INACCURATE** | Placement in `src/ham/bio/` | The file resides in the library source tree (`src/ham/bio/`) but is not listed in the module structure in `spec/ARCH_SPEC.md § 5`, is not exported from `ham.bio.__init__.py`, and provides no importable API. Its location implies it is a library module, contradicting its actual role as a disposable diagnostic script. | Move to `examples/` or `scripts/`, or explicitly document in the module docstring that this is a developer utility not intended for library consumers. Update `spec/ARCH_SPEC.md § 5` if it is to remain in `src/ham/bio/`. |
| 4 | **MISSING** | Hardcoded path (`data_path`) | `data_path = "data/weinreb.h5ad"` at `src/ham/bio/check_data.py:4` is a hardcoded relative path with no documentation of what working directory is assumed or how users should supply an alternative path. | Add a comment or docstring specifying the expected working directory (project root) and consider accepting the path as a CLI argument. |
| 5 | **UNCLEAR** | Diagnostic keywords | The keyword lists used to detect clone/lineage info (`['clone_id', 'lineage', 'Lineage', 'clone']`) and time info (`['time_point', 'Time point', 'day', 'Day', 'time']`) at `src/ham/bio/check_data.py:19-20` are not documented. A user cannot tell whether these are the only valid column names, whether case sensitivity is intentional, or where these names originate (e.g., the Weinreb 2020 dataset conventions). | Add inline comments explaining the source of these column names and their relationship to the expected `BioDataset` contract in `src/ham/bio/data.py`. |
| 6 | **UNCLEAR** | Error handling | The bare `except Exception as e` at `src/ham/bio/check_data.py:25` catches all exceptions and prints a generic message. There is no indication of what errors are expected (corrupt file, missing scanpy, wrong format) or what corrective action the user should take. | Narrow the exception type and/or add a comment noting common failure modes. |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|--------------|-----------------|-------------------|---------------|---------|
| *(no public symbols — script-only file)* | N/A | N/A | N/A | N/A | N/A |

The file defines no functions, classes, or module-level constants that constitute a public API. All logic is at the top-level scope.

## Spec Alignment Notes

1. **`spec/ARCH_SPEC.md § 5` (Module Structure):** The `bio/` sub-package is documented as containing only `vae.py` and `data.py`. `check_data.py` is absent from the spec, making it an undocumented file within a documented package. Either the spec should be updated to list it (if it is intentional infrastructure) or the file should be relocated outside `src/ham/`.

2. **`spec/ARCH_SPEC.md § 6` (Implementation Status):** No mention of diagnostic tooling or data validation scripts. If `check_data.py` is intended to persist, the spec should acknowledge it under a "Developer Utilities" or "Scripts" heading.

3. **Relationship to `examples/weinreb_smoke_test.py:135`:** The `check_dataset()` function in the smoke test performs similar column-existence checks with more structure (it operates on a `BioDataset` instance, checks `.X`, `.obs`, and `.lineage_triples`). The overlap is undocumented; it is unclear whether `check_data.py` is a predecessor, a complement, or a duplicate.
