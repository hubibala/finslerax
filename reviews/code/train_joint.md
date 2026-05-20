# Code Review: `ham.bio.train_joint`

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`train_joint.py` implements a dual-phase training loop for `GeometricVAE`: Phase 1 trains the VAE encoder/decoder ("Manifold Learning"), Phase 2 trains the Randers metric wind field using lineage pairs ("Metric/Wind Learning"). The module is **largely superseded** by the declarative `HAMPipeline` in `src/ham/training/pipeline.py` (which the test file `test_joint_training.py` exclusively exercises), yet it remains in the codebase as a standalone alternative. It contains several JAX correctness issues — most notably a shared optimizer state across two phases with different parameter scopes, a PRNG key reuse bug in Phase 2, and the lack of parameter freezing in either phase. No function in this module is directly tested.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | [train_joint.py](src/ham/bio/train_joint.py#L107) | `GeometricTrainer.__init__` creates a **single** `opt_state` from the full model's arrays and reuses it for both `step_manifold` (VAE params) and `step_metric` (metric params). Optax Adam maintains per-parameter momentum buffers; applying updates computed for metric gradients using an `opt_state` shaped for the full model will silently produce wrong momentum/velocity updates, corrupting optimizer state. The modular `HAMPipeline` correctly creates a separate `opt_state` per phase. | Create separate optimizers and `opt_state` for each phase, or re-initialize `opt_state` at the Phase 2 boundary. |
| 2 | **BUG** | [train_joint.py](src/ham/bio/train_joint.py#L155-L162) | In Phase 2, `key_p` and `key_c` are derived from `subkey` via `jax.random.split(subkey, 2)`, but `subkey` is only split once per **epoch** (L153). Every batch within the same epoch uses the **same** `key_p` and `key_c` for encoding, producing identical stochastic samples from the VAE posterior regardless of input, defeating the purpose of the reparameterization trick. | Split the key per step: `key_p, key_c = jax.random.split(jax.random.fold_in(subkey, step), 2)`. |
| 3 | **BUG** | [train_joint.py](src/ham/bio/train_joint.py#L26) | `batch_loss_fn` returns `stats` as the raw vmapped output `(recon, kl, spray, align)` — each is shape `(B,)`. The caller destructures it as `(loss, (recon, kl, spray, align))` and then applies `jnp.mean` to each. However, `eqx.filter_value_and_grad` treats the second element of the tuple as auxiliary data and does **not** differentiate through it. The raw tuple `stats` is returned as-is from `jax.vmap`, so each element is `(B,)`. This is not wrong per se, but the caller ignores `spray` (returned by `loss_fn` at index 2) and does not log it, while `align` **is** logged. The mismatch between the `loss_fn` signature (which returns `spray_loss` and `align_loss`) and the caller silently drops a diagnostic. | Either log all four stats or explicitly name them. |
| 4 | **BUG** | [train_joint.py](src/ham/bio/train_joint.py#L13-L30) | `step_manifold` updates **all** model parameters (encoder, decoder, **and** metric) because no parameter filter is applied — `eqx.filter_value_and_grad` differentiates through every array leaf. Phase 1 should freeze metric parameters. Similarly, `step_metric` (L35-L88) updates **all** parameters including encoder/decoder, contradicting the stated intent of Phase 2 ("focus purely on the Metric field training", L160). The `stop_gradient` on `z_parents`/`z_children` only prevents gradient flow through the encoding path for that specific computation, but the full model is still passed to `contrastive_loss` and its gradients include encoder/decoder weights via `_get_zermelo_data` and `log_map`. | Use `eqx.partition` with a filter spec to freeze non-target parameters in each phase, as `HAMPipeline` does. |
| 5 | **RISK** | [train_joint.py](src/ham/bio/train_joint.py#L63-L64) | `step_metric` computes `jac_fn = jax.vmap(jax.jacfwd(get_w_single))` inside `contrastive_loss`, which is called inside `eqx.filter_value_and_grad`. This means JAX must differentiate through a full Jacobian computation (backward-over-forward mode). For latent dimensions > ~8 this becomes extremely expensive ($O(d^3)$ per sample) and may exceed memory. The `MetricSmoothnessLoss` in `losses.py` has the same pattern but is applied per-sample (no vmap inside grad), which is cheaper. | Consider using `jax.checkpoint` around the Jacobian computation, or replace with a finite-difference approximation for the smoothness penalty. |
| 6 | **RISK** | [train_joint.py](src/ham/bio/train_joint.py#L50) | `align_scores = -jax.vmap(m.manifold._minkowski_dot)(W_batch, v_tan)` — if `v_tan` is near-zero (parent ≈ child), the alignment signal is dominated by noise. No norm guard or minimum-distance filter is applied. | Filter out pairs with `\|v_{tan}\| < \epsilon` or add a safe normalization before computing alignment. |
| 7 | **RISK** | [train_joint.py](src/ham/bio/train_joint.py#L96-L107) | `GeometricTrainer` is a plain Python class holding mutable JAX arrays (`self.model`, `self.opt_state`, `self.key`). Mutating these inside a loop is fine for eager execution but incompatible with `jax.jit` tracing of the outer loop. If a user ever wraps `train()` in a jitted context, it will fail with tracer leaks. This is acceptable for a training script but violates the ARCH_SPEC principle of JAX-native functional patterns. | Document that `GeometricTrainer` is an imperative convenience wrapper, not suitable for JIT. Or refactor to a functional `train_loop` returning updated state. |
| 8 | **RISK** | [train_joint.py](src/ham/bio/train_joint.py#L131) | `for step in range(num_samples // batch_size)` drops the remainder samples when `num_samples` is not divisible by `batch_size`. For small datasets (common in biology) this can silently discard a significant fraction of data each epoch. The same pattern appears in Phase 2 (L155). | Use `math.ceil(num_samples / batch_size)` with appropriate index clamping, or pad the last batch. |
| 9 | **RISK** | [train_joint.py](src/ham/bio/train_joint.py#L5) | `import time` is imported but never used. | Remove unused import. |
| 10 | **STYLE** | [train_joint.py](src/ham/bio/train_joint.py#L125-L140) | Phase 1 and Phase 2 training loops use `print()` for progress logging. The `HAMPipeline` also uses `print()`, so this is consistent within the codebase, but both should migrate to `logging`. | Use `logging.info()` for consistency with general Python best practices. |
| 11 | **STYLE** | [train_joint.py](src/ham/bio/train_joint.py#L96) | `GeometricTrainer` is not exported from `ham.bio.__init__` (`__all__` only contains `GeometricVAE` and `BioDataset`). Combined with the fact that `HAMPipeline` fully replaces its functionality, this module appears to be dead code. | Either export `GeometricTrainer` and document it as a simpler alternative, or deprecate/remove the module in favor of `HAMPipeline`. |
| 12 | **STYLE** | [train_joint.py](src/ham/bio/train_joint.py#L35-L88) | `step_metric` accesses private API `m.metric._get_zermelo_data` and `m.manifold._minkowski_dot` directly. While the same pattern exists in `losses.py`, coupling to private methods makes the code fragile to internal refactors. | Promote `_get_zermelo_data` and `_minkowski_dot` to public API, or add public wrappers. |

---

## Test Coverage Assessment

| Public API | Tested? | Notes |
|---|---|---|
| `step_manifold()` | **No** | Not tested directly. `test_joint_training.py` tests `HAMPipeline` instead. |
| `step_metric()` | **No** | Not tested directly. |
| `GeometricTrainer.__init__` | **No** | Not tested. |
| `GeometricTrainer.train_step_manifold` | **No** | Not tested. |
| `GeometricTrainer.train_step_metric` | **No** | Not tested. |
| `GeometricTrainer.train` | **No** | Not tested. The equivalent two-phase workflow is tested via `HAMPipeline` in `test_joint_training.py:test_full_pipeline`. |

**Gap Analysis:** This module has **zero** direct test coverage. The test file `test_joint_training.py` exclusively tests the `HAMPipeline` code path. If `train_joint.py` is to remain in the codebase, it needs:
1. A smoke test for `step_manifold` with a mock model (verify loss decreases, shapes correct).
2. A smoke test for `step_metric` with synthetic parent-child pairs.
3. A test for `GeometricTrainer.train` end-to-end with a small dataset.
4. A test verifying that Phase 2 is skipped when `lineage_pairs is None`.

If the module is deprecated in favor of `HAMPipeline`, it should be marked as such or removed.

---

## Positive Patterns

1. **Correct use of `eqx.filter_jit` and `eqx.filter_value_and_grad`** — the JIT-compiled step functions properly separate static and dynamic arguments via Equinox filtering.
2. **`jax.lax.stop_gradient` for Phase 2 encoding** (L160) — correctly prevents gradient flow through the encoder when training the metric, which is the right design intent even though the full model gradient issue (Issue #4) undermines it.
3. **Principled loss decomposition** — the contrastive loss in `step_metric` cleanly separates alignment, anchor regularization, and smoothness terms with explicit weight coefficients and comments explaining each.
4. **PRNG key management in Phase 1** — `jax.random.fold_in(subkey, step)` correctly creates unique per-step keys without splitting overhead.
5. **Separation of pure functions from stateful wrapper** — `step_manifold` and `step_metric` are pure functions outside the class, making them independently testable and JIT-compatible.
