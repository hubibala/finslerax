# Code Review: `ham.bio.train_geodesic`

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`train_geodesic.py` implements `GeodesicFlowTrainer`, a standalone Phase-2 training loop that learns a Randers metric by minimizing geodesic action via the AVBD solver. The module is **legacy/dead code**: it is not exported from `ham.bio.__init__`, not imported anywhere in the codebase, and has **zero test coverage**. The codebase has since migrated to the declarative `HAMPipeline` approach (see `src/ham/training/pipeline.py` and `tests/test_geodesic_learning.py`). Despite being unused, several genuine software-engineering issues exist that would produce wrong results or JAX failures if the class were ever called.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/bio/train_geodesic.py:11` | `GeodesicFlowTrainer` is a plain Python class but `train_step` is decorated with `@eqx.filter_jit`. Equinox's `filter_jit` expects `self` to be a pytree (i.e., an `eqx.Module`), but a plain class is not a registered JAX pytree. This will raise a JAX tracing error when `train_step` is called because `self.solver` and `self.optimizer` are not pytree-registered. | Make `GeodesicFlowTrainer` inherit from `eqx.Module`, or remove `@eqx.filter_jit` and instead JIT the inner `loss_fn` / `train_step` body explicitly. |
| 2 | **BUG** | `src/ham/bio/train_geodesic.py:13-14` | `self.optimizer` and `self.opt_state` are stored as mutable instance attributes. When `@eqx.filter_jit` traces the method, these are captured as constants. Later, `train_phase2` mutates `self.opt_state` on every batch — but the JIT-compiled function sees the original state. This means optimizer state updates are silently discarded, and Adam's momentum/variance accumulators never progress beyond the first batch. | Pass `opt_state` as an explicit argument (already partially done via the `opt_state` parameter of `train_step`), but `self.opt_state` is also referenced in `train_phase2`. Ensure the return value from `train_step` is consistently propagated — do not store optimizer state on `self`. |
| 3 | **BUG** | `src/ham/bio/train_geodesic.py:30-32` | `m.metric._get_zermelo_data(x)` accesses a private method, but the actual access assumes a specific Randers subclass. If `m.metric` is not a `Randers` or `NeuralRanders` (e.g., a `Riemannian` metric), this will crash with `AttributeError`. The function signature accepts any model with a `.metric` attribute but has no type guard. | Add an assertion or type check, or use a public API method (if one exists) for obtaining metric components. |
| 4 | **BUG** | `src/ham/bio/train_geodesic.py:31` | `M.shape[-1]` and the `jnp.eye(dim)` regularizer assume `M` (the Riemannian tensor $H$) is a square 2D matrix. If `_get_zermelo_data` ever returns a batched tensor `(B, D, D)`, `.shape[-1]` would still be correct, but the `jnp.eye(dim)` would not broadcast correctly against the `(B, D, D)` shape. Since `_get_zermelo_data` operates on a single point `x` (no batch dim), this works today but is fragile. | Use `jnp.eye(M.shape[-1])` with explicit broadcast: `jnp.broadcast_to(jnp.eye(dim), M.shape)`. |
| 5 | **RISK** | `src/ham/bio/train_geodesic.py:15` | `AVBDSolver(step_size=0.1, iterations=15)` is hardcoded and never configurable. The default `AVBDSolver.step_size` is `0.05` (see `src/ham/solvers/avbd.py:43`); doubling it here risks divergence for stiff metrics. Similarly, `n_steps=8` on line 22 is hardcoded. | Expose solver configuration as constructor parameters of `GeodesicFlowTrainer`. |
| 6 | **RISK** | `src/ham/bio/train_geodesic.py:55-56` | `z_all` is pre-encoded using a **fixed** `jax.random.PRNGKey(0)` for every sample, meaning the VAE's stochastic sampling uses the same random seed for every data point. For a VAE with a non-trivial posterior, this collapses the reparameterization noise to a single deterministic draw — biasing the latent embeddings. | Use unique keys per sample: `keys = jax.random.split(jax.random.PRNGKey(0), X.shape[0])` and `jax.vmap(self.model.encode)(X, keys)`. |
| 7 | **RISK** | `src/ham/bio/train_geodesic.py:60` | `np.random.permutation` uses NumPy's global RNG, which is not seeded here. Training is non-reproducible. Also mixes NumPy RNG with JAX's functional RNG model, which is a style violation per ARCH_SPEC principles. | Seed `np.random.RandomState` explicitly or use `jax.random.permutation` for consistency. |
| 8 | **RISK** | `src/ham/bio/train_geodesic.py:36` | `trajectories.xs[:, ::2, :]` subsamples every other point from the trajectory for regularization. If `n_steps=8`, the trajectory has 9 points, and subsampling yields indices `[0, 2, 4, 6, 8]` — 5 points. This is fine, but the comment "Sample a few points" is misleading for large `n_steps`. More importantly, the outer `jax.vmap(jax.vmap(...))` assumes fixed shapes — if batch sizes vary across the last batch (line 63: `idx = perm[i:i+batch_size]`), JAX will fail because the sliced batch may be smaller than `batch_size`. | Pad the last batch or skip incomplete batches. |
| 9 | **RISK** | `src/ham/bio/train_geodesic.py:41` | The loss function returns `action_loss + 1.0 * reg_loss` with a hardcoded weight of `1.0`. There is no mechanism to tune the regularization strength, and the comment `+ 1.0 *` is redundant. For practical use, the relative scale of action loss vs. regularization loss can vary by orders of magnitude depending on the metric initialization. | Make regularization weight a constructor parameter. |
| 10 | **STYLE** | `src/ham/bio/train_geodesic.py:1-7` | `numpy` is imported as `np` but only used for `np.random.permutation` (line 60). This mixes JAX and NumPy random subsystems. | Use `jax.random.permutation` or at minimum document the mixed usage. |
| 11 | **STYLE** | `src/ham/bio/train_geodesic.py:8-15` | `GeodesicFlowTrainer` is not exported from `ham.bio.__init__` (`__init__.py` only exports `GeometricVAE` and `BioDataset`). The class is unreachable from the public API. | Either export it or mark/remove it as deprecated dead code. |
| 12 | **STYLE** | `src/ham/bio/train_geodesic.py:17-43` | `train_step` mixes concerns: geodesic solving, action computation, regularization, and gradient update are all in one method. The `HAMPipeline` + modular `LossComponent` pattern (see `spec/ARCH_SPEC.md § 5` and `src/ham/training/pipeline.py`) is the project's canonical approach. This module duplicates and diverges from that architecture. | Refactor to use `HAMPipeline` with `TrainingPhase` and `LossComponent`, or delete if superseded. |

## Test Coverage Assessment

| Public API | Tested? | Notes |
|---|---|---|
| `GeodesicFlowTrainer.__init__` | **No** | No test file references `GeodesicFlowTrainer`. |
| `GeodesicFlowTrainer.train_step` | **No** | Cannot be tested as-is due to BUG #1 (non-pytree `self` under `filter_jit`). |
| `GeodesicFlowTrainer.train_phase2` | **No** | No integration test exercises the full training loop. |

**Gap analysis:** The module has **0% test coverage**. The equivalent functionality is tested in `tests/test_geodesic_learning.py`, but that file uses the `HAMPipeline`-based approach and never imports `train_geodesic.py`. There is no regression safety net for this module — any change (or deletion) would be undetected.

## Positive Patterns

- **Encoder freezing via `stop_gradient`** (line 57): `z_all = jax.lax.stop_gradient(z_all)` correctly prevents gradients from flowing back through the frozen encoder, matching the two-phase training intent.
- **Batch-level vmapping of the solver** (line 22): `jax.vmap(lambda s, e: self.solver.solve(...))` correctly vectorizes the BVP solver over the batch dimension, consistent with the Batch-First principle from `spec/ARCH_SPEC.md § 1`.
- **Meaningful loss decomposition** (lines 25–39): The separation of action loss and metric regularization, while hardcoded, reflects the correct physical intuition — preventing metric collapse while minimizing geodesic energy.

## Recommendation

This module is **dead code** superseded by `HAMPipeline`. It contains 3 confirmed bugs that would prevent execution (#1, #2, #3) and several risks. **Recommended action:** delete the file or, if retained for reference, mark it clearly as deprecated and do not export it.
