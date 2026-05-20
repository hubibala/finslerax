# Documentation Review: `examples/weinreb_smoke_test.py`

**Reviewer:** Doc Reviewer Agent
**Date:** 2026-05-15

## Summary

Overall documentation quality: **needs work**.

The module-level docstring is a strong starting point — it explains the test's purpose, synthetic data shape, and success criteria. However, it contains two factual inaccuracies (wrong filename in the run command, wrong latent dimension), none of the 12 helper functions carry a docstring, and dependency / setup requirements are undocumented.

## Issue Tracker

| # | Severity | Location | Issue | Recommended Action |
|---|----------|----------|-------|--------------------|
| 1 | **INACCURATE** | `examples/weinreb_smoke_test.py:17` | Run command says `python smoke_test_weinreb.py` but the actual filename is `weinreb_smoke_test.py`. | Change to `python weinreb_smoke_test.py`. |
| 2 | **INACCURATE** | `examples/weinreb_smoke_test.py:7` | Docstring states "6 latent dims" but `build_smoke_model` (line 96) and `smoke_train` (line 112) both use `latent_dim=4`. | Update docstring to "4 latent dims". |
| 3 | **MISSING** | `examples/weinreb_smoke_test.py:1–19` | No mention of required dependencies (JAX, Equinox, Optax, NumPy) or that `weinreb_experiment.py` must be importable (i.e., run from the `examples/` directory or add it to `PYTHONPATH`). | Add a "Prerequisites" note listing package dependencies and the working-directory requirement. |
| 4 | **MISSING** | `examples/weinreb_smoke_test.py:52` | `make_synthetic_dataset` has no docstring. As the function that constructs the fake input, documenting what it returns (shape, content of each field) would help readers understand the test's assumptions. | Add a one-line docstring, e.g. `"""Return a BioDataset with n_cells rows, data_dim PCA features, and synthetic velocity/labels."""` |
| 5 | **MISSING** | `examples/weinreb_smoke_test.py:80` | `make_lineage_triples` has no docstring. | Add a brief docstring explaining that it returns an `(n_triples, 3)` array of random cell-index triples. |
| 6 | **MISSING** | `examples/weinreb_smoke_test.py:91` | `build_smoke_model` has no docstring. | Add a docstring noting it builds a minimal `GeometricVAE` with a `PullbackRanders` metric for testing. |
| 7 | **MISSING** | `examples/weinreb_smoke_test.py:110` | `smoke_train` has no docstring. | Add a docstring describing the two-phase training (Phase 1 VAE + data-driven Randers attachment). |
| 8 | **MISSING** | `examples/weinreb_smoke_test.py:134–188` | None of the seven `check_*` functions have docstrings. For a smoke test, each check's pass/fail criteria should be stated. | Add a one-line docstring to each (e.g., `"""Assert encode_mean returns a finite latent vector of the correct shape."""`). |
| 9 | **UNCLEAR** | `examples/weinreb_smoke_test.py:7–8` | Docstring lists "Fake lineage triples (day2 → day4 → day6)" but the code does not assign day labels; triples are random index triplets. The day semantics may confuse readers into expecting temporal metadata. | Reword to "Random cell-index triples used as surrogate lineage data." |
| 10 | **UNCLEAR** | `examples/weinreb_smoke_test.py:9` | "2 epochs per phase" — but only one `TrainingPhase` is constructed; the second "phase" is a non-parametric metric swap, not a training phase. | Clarify: "2 training epochs (Phase 1 only) + non-parametric data-driven Randers attachment." |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:---:|:---:|:---:|:---:|:---:|
| Module docstring | ✅ | — | — | — | — |
| `make_synthetic_dataset` | ❌ | ❌ | ❌ | n/a | n/a |
| `make_lineage_triples` | ❌ | ❌ | ❌ | n/a | n/a |
| `build_smoke_model` | ❌ | ❌ | ❌ | n/a | n/a |
| `smoke_train` | ❌ | ❌ | ❌ | n/a | n/a |
| `check_dataset` | ❌ | ❌ | ❌ | n/a | n/a |
| `check_encode_mean` | ❌ | ❌ | ❌ | n/a | n/a |
| `check_project_control` | ❌ | ❌ | ❌ | n/a | n/a |
| `check_zermelo_data` | ❌ | ❌ | ❌ | n/a | n/a |
| `check_two_segment_energy` | ❌ | ❌ | ❌ | n/a | n/a |
| `check_riemannian_baseline` | ❌ | ❌ | ❌ | n/a | n/a |
| `check_full_validation` | ❌ | ❌ | ❌ | n/a | n/a |
| `main` | ❌ | ❌ | ❌ | n/a | n/a |

## Spec Alignment Notes

- The smoke test is not referenced in `spec/ARCH_SPEC.md` or `spec/MATH_SPEC.md`. This is acceptable — it is an internal test helper, not part of the public API.
- The script imports and exercises `HAMPipeline`, `TrainingPhase`, `PullbackRanders`, `AVBDSolver`, and `GeometricVAE`, all of which are defined in `spec/ARCH_SPEC.md`. No discrepancies found between the imported API surface and the spec.
- No `README.md` mention of this smoke test; consider adding it to a "Testing" or "Development" section if one exists.
