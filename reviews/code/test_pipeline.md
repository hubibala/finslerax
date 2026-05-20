# Code Review: tests/test_pipeline.py

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0

## Summary

The test file is well-structured and covers the core pipeline mechanics — parameter freezing, loss component contracts, multi-phase execution, and phase skipping. However, several important pipeline code paths are entirely untested (lineage triples, `requires_pairs=True` with actual pairs), one test claims to verify pipeline-level loss summation but only tests it manually outside the pipeline, and the gradient-flow claim in the module docstring is not backed by any test that inspects actual gradient values. No correctness bugs were found in the test code itself.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | `tests/test_pipeline.py:191-200` | `test_multiple_losses_sum` calls two `ConstantLoss` instances directly and adds results in Python. This does **not** exercise the pipeline's internal loss-summation loop (`src/ham/training/pipeline.py:46-49`). A pipeline bug that fails to sum losses would not be caught. | Add a test that creates a `TrainingPhase` with multiple losses and verifies the combined gradient or total loss equals the expected sum through `HAMPipeline.fit`. |
| 2 | **RISK** | `tests/test_pipeline.py` (missing) | The `lineage_triples` code path in `pipeline.py:98-103` is never tested. This branch unpacks triples into `(i2, i4, i6)` indices and constructs a 5-element `batch_data` tuple — any off-by-one or shape error would go undetected. | Add a test that provides a `DummyDataset` with `lineage_pairs` and passes `lineage_triples` to `HAMPipeline.fit`, then asserts the phase runs and weights update. |
| 3 | **RISK** | `tests/test_pipeline.py` (missing) | `test_skip_phase_when_no_pairs` tests only the skip path. The complementary path — `requires_pairs=True` with actual pair data (`pipeline.py:104-109`) — is never exercised. | Add a test with `DummyDataset(lineage_pairs=jnp.array([[0,1],[2,3]]))` and `requires_pairs=True`, asserting that the phase executes and weights change. |
| 4 | **RISK** | `tests/test_pipeline.py` (missing) | The module docstring (line 6) claims coverage of "Gradient flow through unfrozen parameters only," but no test inspects actual gradient values or uses `jax.grad` to verify that frozen parameters receive zero gradients. The existing tests only check whether weights changed, which is necessary but not sufficient. | Add a test that uses `eqx.filter_value_and_grad` on the pipeline's loss function and asserts frozen-parameter gradients are zero (or pytree-of-zeros). |
| 5 | **RISK** | `tests/test_pipeline.py:174-180` | `test_mse_returns_scalar` tests the loss with a single unbatched example (`jnp.ones(2)`). In production, the pipeline vmaps the loss over the batch axis (`pipeline.py:55-57`). A loss that works unbatched but breaks under `vmap` (e.g., due to implicit broadcasting) would not be caught. | Add a companion assertion that wraps the loss call in `jax.vmap` with a batch of inputs and verifies the output shape is `(B,)`. |
| 6 | **STYLE** | `tests/test_pipeline.py:165-168` | `test_unfreeze_all` omits descriptive assertion messages on both `assertFalse` calls, unlike the earlier tests in `TestParameterFreezing` which include them. | Add messages: `"layer1 should update"` and `"layer2 should update"`. |
| 7 | **STYLE** | `tests/test_pipeline.py:55-56` | `ConstantLoss.__init__` sets `self.value` before calling `super().__init__()`. While Equinox permits this, conventional Python style calls `super().__init__()` first to initialize the base class before the derived class. | Move `self.value = value` after `super().__init__(weight, name)`. |
| 8 | **STYLE** | `tests/test_pipeline.py:61-68` | `DummyDataset` is a plain class with a hard-coded `PRNGKey(42)`, making the data implicit. Consider parameterizing the key or documenting why the specific seed is required. | Accept an optional `key` parameter with `PRNGKey(42)` as default. |

## Test Coverage Assessment

### Public API: `src/ham/training/pipeline.py`

| Symbol | Tested? | Notes |
|--------|---------|-------|
| `TrainingPhase` (construction) | Yes | All test classes construct phases |
| `HAMPipeline.__init__` | Yes | All test classes |
| `HAMPipeline.fit` — basic training | Yes | `test_freeze_*`, `test_unfreeze_all`, `test_loss_decreases_over_epochs` |
| `HAMPipeline.fit` — multi-phase | Yes | `test_two_phases_sequential` |
| `HAMPipeline.fit` — `requires_pairs` skip | Yes | `test_skip_phase_when_no_pairs` |
| `HAMPipeline.fit` — `requires_pairs` with pairs | **No** | Pair-unpacking logic at `pipeline.py:104-109` untested |
| `HAMPipeline.fit` — `lineage_triples` path | **No** | Triple-unpacking logic at `pipeline.py:98-103` untested |
| `HAMPipeline.fit` — `Traj_long` attachment | **No** | `hasattr(dataset, "Traj_long")` branch at `pipeline.py:108` untested |
| `HAMPipeline.fit` — `seed` parameter | **No** | Always uses default `seed=2025` |
| `HAMPipeline.fit` — multi-loss summation inside pipeline | **No** | Only tested manually outside the pipeline (Issue #1) |

### Gap Analysis

The main gaps are in the data-loading branches of `fit`. The core training loop (partition → optimize → recombine) is well-tested, but the three distinct `batch_data` construction paths (plain, pairs, triples) have only one of three tested. This is the highest-priority gap.

## Positive Patterns

1. **Deterministic PRNG keys**: All tests use fixed `jax.random.PRNGKey` values, ensuring reproducibility across runs.
2. **Clear test-class separation**: `TestParameterFreezing`, `TestLossComponents`, and `TestMultiPhaseExecution` cleanly partition concerns.
3. **Behavioral freezing tests**: The symmetric pair `test_freeze_layer2_update_layer1` / `test_freeze_layer1_update_layer2` provides strong evidence that the partition logic works correctly in both directions.
4. **Descriptive assertion messages**: Most assertions include human-readable failure messages, aiding diagnosis.
5. **Lightweight fixtures**: `DummyModel`, `MSELoss`, and `DummyDataset` are minimal and avoid coupling tests to production model complexity.
