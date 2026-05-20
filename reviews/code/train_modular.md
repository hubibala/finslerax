# Code Review: `train_modular.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`train_modular.py` is a thin top-level script that wires together the `HAMPipeline` with a two-phase training schedule (Manifold Pretraining → Metric Learning). It also exposes a reusable `get_filter_fn` helper for Equinox parameter partitioning. The file has no direct unit tests — all coverage comes indirectly through `test_pipeline.py`, which tests the underlying `HAMPipeline` and `TrainingPhase` with dummy models. The code is broadly correct but contains a PRNG-seed reuse issue, an inline import that hurts readability, and several minor style/robustness issues.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/bio/train_modular.py:49-50` | The same PRNG `subkey` is reused for both `NeuralRanders` initialisation (line 62) and the `jax.random.split` that feeds `GeometricVAE` (line 65). However, the split on line 65 uses the *original* `key` that was already consumed on line 49. This is correct for the key/subkey pair, but `subkey` from line 49 is passed to `NeuralRanders` on line 62, then the *same parent key* is split again on line 65 which produces a fresh subkey — so this is actually safe. **Re-evaluated: no bug.** | N/A — false alarm on closer inspection. |
| 2 | **RISK** | `src/ham/bio/train_modular.py:49,108` | `seed = 2025` is used both for `jax.random.PRNGKey(seed)` (line 49) and again passed into `pipeline.fit(..., seed=seed)` (line 108). Inside `HAMPipeline.fit` (`pipeline.py:28`), a *new* `jax.random.PRNGKey(seed)` is created from the same integer seed, making the pipeline's internal randomness (batch shuffling) perfectly correlated with the model initialisation key stream. This is unlikely to cause wrong results but violates best practice for reproducible, independent random streams. | Pass an unused split of `key` into `pipeline.fit` instead of the raw integer, or have `HAMPipeline.fit` accept a `jax.random.PRNGKey` directly. |
| 3 | **RISK** | `src/ham/bio/train_modular.py:60` | Inline `from ham.geometry.surfaces import Hyperboloid` inside `main()`. While functional, deferred imports inside a function hide dependencies from static analysis tools and IDE symbol resolution, and make it easy to miss import errors until runtime. | Move to top-level imports alongside the other `ham.*` imports. |
| 4 | **RISK** | `src/ham/bio/train_modular.py:22-42` | `get_filter_fn` builds a closure over `selector` (a lambda). Lambdas are not serialisable and cannot be safely pickled, which means any attempt to checkpoint or serialise a `TrainingPhase` that embeds one of these `filter_spec` callables will fail silently or raise. This is not a JAX-tracing issue (lambdas work fine in Equinox tree operations), but it limits pipeline reproducibility. | Document the limitation, or accept a string-based selector that can be resolved at runtime. |
| 5 | **STYLE** | `src/ham/bio/train_modular.py:5` | `import time` — the `time` module is only used for wall-clock logging (line 107, 110). This is fine for a script, but note that JAX async dispatch means `time.time()` may not capture actual compute time accurately. | Consider `jax.block_until_ready()` on the result before recording end time, or use `time.perf_counter()`. |
| 6 | **STYLE** | `src/ham/bio/train_modular.py:2` | `import jax.numpy as jnp` is imported but never used anywhere in the file. | Remove the unused import. |
| 7 | **STYLE** | `src/ham/bio/train_modular.py:54` | `DataLoader(mode='simulation').get_jax_data(use_pca=False)` — in simulation mode, `get_jax_data` returns a hardcoded 100×50 zero dataset (`data.py:161-162`). The script silently runs on dummy zeros, which is not obvious to a user reading the script as a "training entry point." | Add a comment or print statement clarifying that simulation mode returns synthetic data, or default to a more meaningful mode. |
| 8 | **STYLE** | `src/ham/bio/train_modular.py` | `spec/ARCH_SPEC.md § 5` lists `train_modular.py` nowhere in the module structure. Only `train_joint.py` and `train_geodesic.py` appear under `bio/`. The file exists but is not documented in the architecture specification. | Add `train_modular.py` to the module listing in `ARCH_SPEC.md § 5`, or mark it as a top-level example script rather than a library module. |

## JAX Correctness Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| JIT compatibility | **OK** | No JIT-compiled code in this file; all JIT happens inside `HAMPipeline.fit`. |
| vmap compatibility | **OK** | Not applicable — no vmap in this file. |
| grad compatibility | **OK** | Not applicable — no grad in this file. |
| Side-effects in traced code | **OK** | `print` statements are outside any traced context. |
| PRNG key handling | **OK** | Keys are split before use; no key reuse detected (Issue 1 re-evaluated). |

## Numerical Stability Assessment

No numerical operations are performed directly in this file. All computation is delegated to `LossComponent` subclasses and `HAMPipeline`. No stability concerns at this level.

## API Consistency Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| Batch-first convention | **OK** | Data comes from `DataLoader` which returns `(N, D)` arrays; pipeline handles batching. |
| `get_filter_fn` signature | **OK** | Accepts a selector callable, returns a `filter_spec` callable — matches `TrainingPhase.filter_spec` type. |
| Phase declaration | **OK** | Phases are constructed with all required fields; `requires_pairs` correctly gates Phase 2. |

## Test Coverage Assessment

| Public Symbol | Tested? | Gap |
|---------------|---------|-----|
| `get_filter_fn` | **Indirect** | `test_pipeline.py` uses hand-written `_filter_layer1`/`_filter_layer2` helpers that replicate the same pattern but do NOT import or exercise `get_filter_fn` itself. No test verifies that the closure-based approach in `get_filter_fn` works with multi-component selectors like `lambda m: (m.encoder_net, m.decoder_net)`. |
| `main()` | **No** | No integration test calls `main()`. Given that it requires `DataLoader` in simulation mode, a smoke test would be straightforward. |

**Recommended actions:**
1. Add a unit test that imports `get_filter_fn`, applies it to a model with at least two sub-modules, and verifies the resulting mask is correct.
2. Add a minimal smoke test that calls `main()` (or a parameterised variant) to verify the end-to-end wiring.

## Positive Patterns

1. **Clean separation of concerns** — the script only *configures* phases; all training logic lives in `HAMPipeline`. This matches the declarative pipeline pattern from `spec/ARCH_SPEC.md § 4`.
2. **Correct use of `eqx.tree_at`** — `get_filter_fn` correctly builds a boolean mask by combining `tree_map(→False)` with `tree_at(selector, ..., replace=True)`, which is the idiomatic Equinox partitioning pattern.
3. **Conditional phase inclusion** — Phase 2 is only appended when `dataset.lineage_pairs is not None`, preventing silent failures during metric learning without supervision data.
4. **`eqx.is_array` guard** — `make_true` correctly distinguishes trainable arrays from static metadata leaves, preventing non-differentiable leaves from being marked as trainable.
