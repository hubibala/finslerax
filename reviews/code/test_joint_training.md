# Code Review: `tests/test_joint_training.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`test_joint_training.py` provides minimal smoke-test coverage of the `HAMPipeline` multi-phase training system. It has three tests — one per phase plus a full-pipeline run. The test structure is sound but contains a PRNG key reuse bug that makes `X` and `V` identical, weak assertions in 2 of 3 tests, and no verification that parameter freezing actually works. Coverage of edge cases (missing lineage pairs, empty batches, single-sample batches) is absent.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | [tests/test_joint_training.py](tests/test_joint_training.py#L71-L72) | `X` and `V` are generated with **the same PRNG key** (`self.key`), producing identical arrays. This means every test operates on data where the observation matrix equals the velocity matrix, which is biologically meaningless and may mask bugs in losses that depend on distinct `X` and `V` (e.g., `ReconstructionLoss`, `ZermeloAlignmentLoss`). | Split the key: `k1, k2 = jax.random.split(self.key); X = jax.random.normal(k1, ...); V = jax.random.normal(k2, ...)`. |
| 2 | **BUG** | [tests/test_joint_training.py](tests/test_joint_training.py#L42-L56) | `MockMetric` inherits from `(Riemannian, eqx.Module)` but never calls `Riemannian.__init__` (or `FinslerMetric.__init__`). Since `FinslerMetric.__init__` sets `self.manifold` from its argument and `Riemannian.__init__` sets `self.g_net`, the mock bypasses both by assigning fields directly. This works only because Equinox `Module.__init_subclass__` handles field registration at the class level — but it means `FinslerMetric.__init__`'s logic (if it ever validates inputs) is silently skipped. More concretely, `eqx.Module` is redundant in the bases since `Riemannian → FinslerMetric → eqx.Module` already provides it, and the double base may cause confusion. | Remove `eqx.Module` from the bases: `class MockMetric(Riemannian):`. Call `super().__init__(manifold, g_net=eqx.nn.Linear(1, 1, key=jax.random.PRNGKey(0)))` to properly initialize the parent. |
| 3 | **RISK** | [tests/test_joint_training.py](tests/test_joint_training.py#L113-L116) | `test_phase2_metric` only asserts `assertIsInstance(trained, GeometricVAE)`. This is trivially true — `HAMPipeline.fit()` returns `self.model` which is always a `GeometricVAE`. The test does not verify that metric parameters changed, that VAE parameters were frozen, or that the loss decreased. A silent no-op would pass. | Add assertions: (a) metric weights changed (`assertFalse(jnp.allclose(old_metric, new_metric))`), (b) encoder/decoder weights did NOT change (frozen), and (c) optionally check loss is finite. |
| 4 | **RISK** | [tests/test_joint_training.py](tests/test_joint_training.py#L139-L141) | `test_full_pipeline` also only asserts `assertIsInstance(trained, GeometricVAE)`. Same weakness as Issue #3 — the assertion is too weak to catch regressions. | Assert that both encoder/decoder and metric weights differ from initial values after the full pipeline. |
| 5 | **RISK** | [tests/test_joint_training.py](tests/test_joint_training.py#L95) | `old_dec = self.vae.decoder_net.layers[0].weight.copy()` accesses internal MLP structure (`.layers[0].weight`). If `eqx.nn.MLP` changes its attribute layout, this test breaks for reasons unrelated to training correctness. | Use `jax.tree_util.tree_leaves(self.vae.decoder_net)` to snapshot all decoder params generically, then compare after training. |
| 6 | **RISK** | [tests/test_joint_training.py](tests/test_joint_training.py#L73) | `labels = jnp.zeros(self.N)` — all labels are identical (class 0). Any loss or metric that depends on class diversity (e.g., a contrastive loss using labels to form positive/negative pairs) will see degenerate input. While current tests don't use label-dependent losses, this fixture silently masks label-related bugs if tests are extended. | Use `labels = jax.random.randint(self.key, (self.N,), 0, 3)` for non-trivial labels. |
| 7 | **RISK** | [tests/test_joint_training.py](tests/test_joint_training.py#L74) | `lineage_pairs = jnp.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])` — only 5 pairs for 50 samples. With `batch_size=5` in `test_phase2_metric` and `test_full_pipeline`, there is exactly 1 step per epoch (`5 // 5 = 1`), so each epoch sees the same 5 pairs without any shuffling benefit. This is not a bug, but the test would be more robust with more pairs to exercise the batching/shuffling logic. | Recommended Action: increase to ~15–20 pairs for better coverage of the data-loading path. |
| 8 | **RISK** | [tests/test_joint_training.py](tests/test_joint_training.py#L42-L56) | `MockMetric` does not override `metric_fn` (inherited from `Riemannian`). The inherited `metric_fn` calls `self.g_net(x)` which is an `eqx.nn.Linear(1, 1)` — input dimension 1 but `x` has dimension 2 (latent_dim). If any code path calls `metric_fn` on this mock (e.g., through `energy()` → `spray()`), it will crash with a shape error. The mock's `spray()` returns zeros which avoids this, but it means `MockMetric.metric_fn` is silently broken. | Override `metric_fn` to return `jnp.sqrt(jnp.sum(v**2) + 1e-12)` (Euclidean fallback), or change `g_net` to `eqx.nn.Linear(latent_dim, latent_dim**2, key=...)` and reshape. |
| 9 | **STYLE** | [tests/test_joint_training.py](tests/test_joint_training.py#L24-L40) | `get_filter_fn` is a 15-line helper that reimplements parameter filtering logic. It is not tested itself and could contain subtle bugs (e.g., if `selector` returns a nested structure). The same pattern is presumably tested via the pipeline integration tests, but if it breaks, all three tests fail with a confusing error. | Consider importing a shared test utility or at minimum adding a docstring explaining the invariants it maintains. |
| 10 | **STYLE** | [tests/test_joint_training.py](tests/test_joint_training.py#L10-L18) | Six loss classes are imported (`ReconstructionLoss`, `KLDivergenceLoss`, `ZermeloAlignmentLoss`, `ContrastiveAlignmentLoss`, `MetricAnchorLoss`, `MetricSmoothnessLoss`) but only four are used. `ZermeloAlignmentLoss` and `MetricSmoothnessLoss` are unused. | Remove unused imports. |
| 11 | **STYLE** | [tests/test_joint_training.py](tests/test_joint_training.py#L65) | The class is named `TestModularTraining` but tests `HAMPipeline` which is the "declarative pipeline" per ARCH_SPEC §6. The module `test_joint_training.py` originally targeted `train_joint.py` but now tests `pipeline.py`. The naming mismatch may confuse developers. | Rename to `TestHAMPipeline` and/or rename the file to `test_pipeline.py`. |

---

## Test Coverage Assessment

| Public API (`pipeline.py`) | Tested? | Notes |
|---|---|---|
| `TrainingPhase` construction | **Yes** | Constructed in all 3 tests. |
| `HAMPipeline.__init__` | **Yes** | Constructed in all 3 tests. |
| `HAMPipeline.fit` — single phase, no pairs | **Yes** | `test_phase1_manifold`. |
| `HAMPipeline.fit` — single phase, with pairs | **Yes** | `test_phase2_metric`. |
| `HAMPipeline.fit` — multi-phase sequential | **Yes** | `test_full_pipeline`. |
| `HAMPipeline.fit` — with `lineage_triples` | **No** | The `lineage_triples` kwarg path is untested. |
| `HAMPipeline.fit` — phase skip when pairs missing | **No** | The `requires_pairs=True` with `lineage_pairs=None` path that prints "Skipping phase" is not tested. |
| Parameter freezing correctness | **No** | No test verifies that frozen parameters remain unchanged. |
| Loss decrease over epochs | **No** | No test verifies that training actually reduces the loss. |
| Batch size > dataset size | **No** | Edge case not tested. |
| Single-sample batch | **No** | Edge case not tested. |

**Gap Analysis:** The three existing tests verify that `HAMPipeline.fit` runs without crashing and returns a `GeometricVAE`, but they do not verify *training correctness*. The most critical gap is the absence of parameter-freezing assertions — the defining feature of multi-phase training. The `lineage_triples` code path is completely untested.

---

## Positive Patterns

1. **Clean `setUp` fixture** — test data, model, and dataset are constructed once per test method, ensuring full test isolation via `unittest.TestCase`.
2. **Declarative `TrainingPhase` usage** — the tests correctly mirror the intended API from `ARCH_SPEC.md` §6, constructing phases with named losses and filter specs.
3. **`test_phase1_manifold` weight-change assertion** — one test correctly captures a weight snapshot before training and verifies it changed, which is the right pattern (should be extended to the other tests).
4. **Small, fast fixtures** — `N=50`, `epochs=2`, `batch_size=5/10` keep tests fast while still exercising the full training loop.
