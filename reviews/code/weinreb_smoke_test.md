# Code Review: weinreb_smoke_test.py
**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary
The smoke test exercises major code paths of `weinreb_experiment.py` with synthetic data, which is valuable for pre-commit validation. However, several check functions contain vacuous assertions that silently pass regardless of actual results, undermining the test's purpose. One fallback assertion path would crash at runtime due to JAX array truth-value ambiguity. There is also an unused import and a misleading docstring.

## Issue Tracker

| # | Severity | Location (file:line) | Description | Suggested Fix |
|---|----------|----------------------|-------------|---------------|
| 1 | **BUG** | `examples/weinreb_smoke_test.py:195-199` | `check_full_validation` contains a vacuous loop: `for k in required_keys: if k in results: pass`. This never asserts anything — the smoke test passes even if `run_validation` returns an empty `results` dict, completely defeating the purpose of the check. | Replace with `assert k in results, f"Missing key '{k}' in validation results"` and add at least one value assertion (e.g. `assert results[k]['n'] > 0`). |
| 2 | **RISK** | `examples/weinreb_smoke_test.py:154-161` | `check_zermelo_data` prints `"✓ _get_zermelo_data OK"` even when the metric does *not* have `_get_zermelo_data` (the `hasattr` guard silently skips all assertions). For the smoke test to be meaningful, this should either assert the attribute exists or print a distinct skip message. | Add an `else` branch: `else: print("  ⚠ _get_zermelo_data not present — skipped")` or `assert hasattr(...)`. |
| 3 | **RISK** | `examples/weinreb_smoke_test.py:171-173` | Fallback assertion uses `jax.tree_util.tree_leaves(a) == jax.tree_util.tree_leaves(b)`. Python's list `==` calls element-wise `__eq__` on JAX arrays, which returns boolean *arrays*, not scalars. This would raise `ValueError: The truth value of an array with more than one element is ambiguous` if the identity check (`is`) ever fails. In practice the identity check succeeds (because `build_riemannian_baseline` shares the encoder by reference), so this is dead code — but it will crash if ever reached. | Remove the fallback branch or use `jnp.allclose` on flattened leaves: `all(jnp.allclose(a, b) for a, b in zip(leaves1, leaves2))`. |
| 4 | **STYLE** | `examples/weinreb_smoke_test.py:42` | `plot_results` is imported but never called. The docstring claims the test verifies "every code path," but visualization is skipped without comment. | Either remove the import or add a minimal `plot_results` check (e.g. to a temp file) with a note explaining the skip. |
| 5 | **STYLE** | `examples/weinreb_smoke_test.py:31` | `PullbackRiemannian` is imported but never used directly in this file — `build_riemannian_baseline` (imported from `weinreb_experiment`) handles Riemannian model construction internally. | Remove the unused import. |
| 6 | **STYLE** | `examples/weinreb_smoke_test.py:7` | Docstring says "6 latent dims" but the code uses `latent_dim=4` everywhere (lines 97, 119, 216). | Update docstring to say "4 latent dims". |
| 7 | **STYLE** | `examples/weinreb_smoke_test.py:2` | Docstring filename `smoke_test_weinreb.py` does not match the actual filename `weinreb_smoke_test.py`. | Change to `weinreb_smoke_test.py`. |

## Test Coverage Assessment
This file is itself a smoke test (not a unit test), so the standard "public function coverage" lens does not directly apply. Instead, the assessment is whether each imported function from `weinreb_experiment` is exercised:

| Imported Function | Exercised? | Notes |
|---|---|---|
| `get_filter_fn` | Yes | Used in `smoke_train` (line 127) |
| `encode_mean` | Yes | Multiple checks |
| `two_segment_energy` | Yes | `check_two_segment_energy` |
| `build_riemannian_baseline` | Yes | `check_riemannian_baseline` |
| `run_validation` | Yes | `check_full_validation` — but result is not actually asserted (Issue #1) |
| `plot_results` | **No** | Imported but never called (Issue #4) |
| `attach_datadriven_randers_metric` | Yes | Used in `smoke_train` |

**Gap:** `run_validation` output is exercised in form only — the vacuous loop means no actual validation of return values occurs.

## Positive Patterns
- Clean synthetic data factories (`make_synthetic_dataset`, `make_lineage_triples`) with deterministic seeds — good for reproducibility.
- Numbered progress output (`[1/7]`, `[2/7]`, …) gives clear visibility into which stage fails.
- Appropriately small hyperparameters (2 epochs, batch 32, 5 anchors) keep the smoke test fast.
- Proper use of `jax.random.split` for key management in `build_smoke_model`.
- The overall structure — build → train → validate — mirrors the real experiment pipeline faithfully.
