# Code Review: `training/pipeline.py`
**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

The `HAMPipeline` is a well-designed multi-phase training orchestrator that correctly uses Equinox's `partition`/`combine` pattern for per-phase parameter freezing and `eqx.filter_jit` + `eqx.filter_value_and_grad` for JIT-compiled gradient computation. The core training loop is sound and aligns with `spec/ARCH_SPEC.md § 5` (training pipeline). However, there are several software quality issues: a closure over the mutable loop variable `phase` inside a JIT-decorated function risks stale captures in alternative execution patterns; the `loss_fn` closure captures loss components that are `eqx.Module` instances without explicitly passing them as arguments; batch tail-drop silently discards data; the dataset protocol is implicit and fragile (attribute access without `hasattr` guards); and there is duplicated data extraction code. No critical correctness bugs were found — the pipeline produces correct training results as verified by existing tests.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | `src/ham/training/pipeline.py:42–46` | `loss_fn` closes over `phase.losses` — a list of `eqx.Module` instances — from the outer loop scope. Inside `train_step`, this closure is passed to `jax.vmap`. The loss module weights and any internal arrays are captured as implicit constants. If a `LossComponent` had trainable parameters that were meant to be updated (e.g., a learned weight schedule), those parameters would be frozen at the values captured at trace time and never receive gradients. Currently all `LossComponent` subclasses use static `weight` fields, so this is not a bug today, but the pattern is fragile. | Pass `phase.losses` explicitly through the function chain or document that `LossComponent.weight` must never be trainable. |
| 2 | **RISK** | `src/ham/training/pipeline.py:61` | `phase.optimizer.update(grads, state, diff)` captures `phase.optimizer` via closure inside `@eqx.filter_jit`. Optax optimizers are not JAX pytrees in general — `eqx.filter_jit` hashes them as static values. If the user passes the *same* optimizer object for two phases with different hyperparameters mutated in place (e.g., `opt.learning_rate = 0.01`), the JIT cache could return a stale compiled function. Since `train_step` is redefined each phase iteration (new function object → new cache entry), this is safe in the current code, but the pattern relies on the implementation detail that `def` inside a loop creates a new function identity each iteration. | Extract optimizer calls from the JIT boundary, or pass the optimizer as an explicit argument to `train_step` so the tracing dependency is visible. |
| 3 | **RISK** | `src/ham/training/pipeline.py:96` | `steps_per_epoch = max(1, num_items // batch_size)` uses integer division, silently dropping the tail `num_items % batch_size` samples each epoch. For small datasets (e.g., 100 samples with `batch_size=256`), `steps_per_epoch = 1` and only the first `batch_size` indices are sliced from `perm`. Since `perm` has fewer elements than `batch_size`, JAX's out-of-bounds slicing clips to array length, producing a smaller-than-requested batch. This smaller batch is valid but may cause one-time JIT recompilation if the first epoch of a different phase had a full-sized batch. | Use `math.ceil(num_items / batch_size)` for steps, and pad the last batch's index array to `batch_size` (with modular wrap-around), or document the tail-drop behavior. |
| 4 | **RISK** | `src/ham/training/pipeline.py:70` | `dataset.lineage_pairs` is accessed via attribute lookup without an `hasattr` guard. If a user passes a dataset object that does not define `lineage_pairs`, an `AttributeError` is raised with no informative message. The `dataset.labels` access (line 76) has a `None` check but also assumes the attribute exists. | Define a minimal dataset protocol (e.g., `typing.Protocol`) requiring `X`, `V`, `labels`, and `lineage_pairs` attributes, or add `getattr(dataset, 'lineage_pairs', None)` defensive access. |
| 5 | **RISK** | `src/ham/training/pipeline.py:53` | `batch_keys = jax.random.split(step_key, batch_data[0].shape[0])` extracts the batch size dynamically from `batch_data[0].shape[0]`. The `shape[0]` is a traced value under JIT — this is fine because `jax.random.split` accepts traced `num` arguments. However, if `batch_data` is an empty tuple (no data elements), this will crash with an uninformative `IndexError`. This can only happen if a `LossComponent` expects an empty batch tuple, which is unlikely but not guarded. | Add an assertion: `assert len(batch_data) > 0, "batch_data must have at least one element"` outside the JIT boundary when constructing `batch_data`. |
| 6 | **STYLE** | `src/ham/training/pipeline.py:67–68,75–78` | `data_x, data_v = dataset.X, dataset.V` and `num_samples = data_x.shape[0]` appear twice — first on lines 67–68 (before the `requires_pairs` early-exit check) and again on lines 75–78 (after the check, with `data_labels`). The first extraction is immediately discarded if the `continue` on line 72 is taken, and otherwise overwritten on line 75. | Remove lines 67–68 and move `num_samples` to line 78 alongside the second extraction block. |
| 7 | **STYLE** | `src/ham/training/pipeline.py:117` | Inconsistent indentation: `batch_data = (data_x[idx], data_v[idx], data_labels[idx])` has an extra leading space compared to the surrounding `if/else` block (5 spaces vs. expected 24 for the `else` body). This does not affect execution but violates PEP 8. | Align to the same indentation as the `if` branch body above. |
| 8 | **STYLE** | `src/ham/training/pipeline.py:19–22` | `HAMPipeline` is a plain Python class with mutable `self.model` state, while the rest of the codebase uses `eqx.Module` for all stateful containers per `spec/ARCH_SPEC.md § 2`. This is intentional (the pipeline manages mutable training state outside JIT), but it creates a protocol mismatch — `HAMPipeline` cannot be passed through JAX transforms, serialised with `eqx.tree_serialise_leaves`, or composed with other Equinox modules. | Document that `HAMPipeline` is intentionally a non-JAX orchestrator, or convert to an `eqx.Module` that returns a new pipeline instance (functional style) from `fit`. |
| 9 | **STYLE** | `src/ham/training/pipeline.py:9–17` | `TrainingPhase` uses `@dataclass` rather than `eqx.Module`. This is appropriate for a config-only object, but the `losses` field holds `eqx.Module` instances and `filter_spec` holds a callable — if anyone ever tries to pass a `TrainingPhase` through JAX transforms (e.g., for hyperparameter search), it will fail. | Acceptable as-is for a config object; add a docstring clarifying this is not a JAX-traceable container. |
| 10 | **STYLE** | `src/ham/training/pipeline.py:128–129` | `epoch_stats[k_stat] += v_stat` accumulates JAX device scalars into a Python dict each step. Each `+=` triggers an implicit device-to-host transfer (or at least keeps a reference to a device array). For phases with many steps per epoch, this creates many small unreduced device arrays. | Accumulate using `jnp.add` into pre-allocated arrays, or block on `float(v_stat)` explicitly at accumulation time to avoid device memory pressure. |

## Test Coverage Assessment

| Public Symbol | Tested? | Test Location | Notes |
|---------------|---------|---------------|-------|
| `HAMPipeline.__init__` | Yes | `tests/test_pipeline.py` (all tests) | Implicitly via construction |
| `HAMPipeline.fit` (single phase, freeze layer1) | Yes | `tests/test_pipeline.py::TestParameterFreezing::test_freeze_layer2_update_layer1` | Verifies layer1 updates, layer2 frozen |
| `HAMPipeline.fit` (single phase, freeze layer2) | Yes | `tests/test_pipeline.py::TestParameterFreezing::test_freeze_layer1_update_layer2` | Verifies layer2 updates, layer1 frozen |
| `HAMPipeline.fit` (single phase, unfreeze all) | Yes | `tests/test_pipeline.py::TestParameterFreezing::test_unfreeze_all` | Verifies both layers update |
| `HAMPipeline.fit` (two-phase sequential) | Yes | `tests/test_pipeline.py::TestMultiPhaseExecution::test_two_phases_sequential` | Verifies both layers update across two phases |
| `HAMPipeline.fit` (skip phase, no pairs) | Yes | `tests/test_pipeline.py::TestMultiPhaseExecution::test_skip_phase_when_no_pairs` | Verifies weights unchanged when phase is skipped |
| `HAMPipeline.fit` (loss decreases) | Yes | `tests/test_pipeline.py::TestMultiPhaseExecution::test_loss_decreases_over_epochs` | Sanity check on 50-epoch convergence |
| `TrainingPhase` | Implicit | `tests/test_pipeline.py` (all tests) | Used as config, not directly tested |
| `HAMPipeline.fit` (`requires_pairs=True` with data) | **No** | — | No test provides actual lineage pairs or triples |
| `HAMPipeline.fit` (lineage_triples path) | **No** | — | The triple-based batching logic (lines 103–108) is entirely untested |
| `HAMPipeline.fit` (Traj_long path) | **No** | — | The `hasattr(dataset, "Traj_long")` branch (line 113) is untested |
| `HAMPipeline.fit` (multiple loss components) | **No** | — | `TestLossComponents::test_multiple_losses_sum` tests loss arithmetic but not pipeline-level multi-loss |
| `HAMPipeline.fit` (gradient correctness) | **No** | — | No test uses `jax.test_util.check_grads` or verifies gradient values |
| `HAMPipeline.fit` (vmap/jit compatibility) | **No** | — | No test wraps `fit` or `train_step` in explicit transforms |
| `HAMPipeline.fit` (labels batching) | **No** | — | `DummyDataset.labels = None` always; the fallback `jnp.zeros(...)` path is tested, but real labels never are |

### Coverage Gaps

1. **Lineage triples/pairs batching is untested.** Lines 85–114 contain branching logic for `requires_pairs`, `lineage_triples`, and `Traj_long` — none of which is exercised by any test. This is the most complex data-handling code in the module.
2. **No gradient correctness test.** The pipeline's core value is differentiable training. No test verifies that gradients flow correctly through the full `loss_fn → vmap → filter_value_and_grad` chain beyond checking that weights change.
3. **No multi-loss pipeline test.** While `ConstantLoss` tests arithmetic, no test runs the pipeline with multiple heterogeneous losses to verify stats accumulation and gradient combination.
4. **No test for label-aware losses.** The `data_labels` fallback (line 76–77) is always triggered since test datasets have `labels=None`.

## Positive Patterns

1. **Correct `eqx.partition` / `eqx.combine` usage** (lines 33, 39, 136) — this is the idiomatic Equinox pattern for per-phase parameter freezing and produces correct gradient masking.
2. **`eqx.filter_value_and_grad` with `has_aux=True`** (line 60) — correctly extracts both loss value and per-component stats in a single backward pass.
3. **`jax.random.fold_in` for step-level randomness** (line 119) — avoids splitting a key per step (which would require tracking accumulated splits), instead deterministically deriving step keys from the epoch subkey.
4. **Clean phase skip logic** (lines 70–72) — early `continue` when lineage data is missing prevents silent failures downstream.
5. **Declarative phase API** — `TrainingPhase` as a dataclass config cleanly separates training schedule from execution logic, making multi-phase experiments reproducible.
