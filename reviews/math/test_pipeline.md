# Math Review: test_pipeline
**Reviewer:** Math Reviewer Agent
**Date:** 2026-05-15
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

Correct. The test file verifies training-pipeline infrastructure (parameter freezing, loss weighting, loss summation, convergence) rather than differential-geometric formulae. All mathematical assertions—MSE formula, weight-scaling linearity, additive loss composition, and the convergence sanity check—are analytically sound. Tolerances are appropriate for float32 JAX arithmetic. Two minor notes are recorded below.

## Formula-by-Formula Audit

### MSELoss — `jnp.mean((model(x) - y) ** 2) * self.weight`
- **Spec Reference:** Not geometry-specific; standard supervised loss.
- **Literature Reference:** Standard element-wise MSE.
- **Implementation:** [tests/test_pipeline.py](tests/test_pipeline.py#L38-L43)
  ```python
  def __call__(self, model, batch, key):
      x = batch[0]
      y = batch[1]
      return jnp.mean((model(x) - y) ** 2) * self.weight
  ```
- **Verdict:** CORRECT
- **Notes:** For an input $x \in \mathbb{R}^D$ and target $y$, the loss is
  $\mathcal{L} = w \cdot \frac{1}{D}\sum_{j=1}^{D}(f(x)_j - y_j)^2$,
  which is the standard MSE scaled by the component weight $w$. Multiplication by `self.weight` outside the `mean` preserves exact linearity, verified by `test_weight_scaling`.

### ConstantLoss — `jnp.float32(self.value) * self.weight`
- **Spec Reference:** N/A (test fixture only).
- **Implementation:** [tests/test_pipeline.py](tests/test_pipeline.py#L50-L55)
  ```python
  def __call__(self, model, batch, key):
      return jnp.float32(self.value) * self.weight
  ```
- **Verdict:** CORRECT
- **Notes:** Returns a constant independent of model parameters ($\nabla_\theta \mathcal{L} = 0$). Used exclusively in `test_multiple_losses_sum` to verify additive composition, not for actual training.

### test_weight_scaling — `scaled ≈ base × 3.0`
- **Spec Reference:** N/A.
- **Implementation:** [tests/test_pipeline.py](tests/test_pipeline.py#L183-L190)
  ```python
  base = MSELoss(weight=1.0)(model, (x, y), ...)
  scaled = MSELoss(weight=3.0)(model, (x, y), ...)
  self.assertAlmostEqual(float(scaled), float(base) * 3.0, places=5)
  ```
- **Verdict:** CORRECT
- **Notes:** Because the weight multiplies the entire expression, $\mathcal{L}(w{=}3) = 3\,\mathcal{L}(w{=}1)$ holds exactly in exact arithmetic. Tolerance `places=5` ($5 \times 10^{-6}$) gives ~40× slack over float32 machine epsilon ($\approx 1.2\times10^{-7}$) — appropriate.

### test_multiple_losses_sum — `C(2) + C(3) = 5`
- **Spec Reference:** Pipeline loss composition: `total_loss += val` in [src/ham/training/pipeline.py](src/ham/training/pipeline.py#L44).
- **Implementation:** [tests/test_pipeline.py](tests/test_pipeline.py#L192-L200)
  ```python
  total = l1(model, ...) + l2(model, ...)
  self.assertAlmostEqual(float(total), 5.0, places=5)
  ```
- **Verdict:** CORRECT
- **Notes:** The pipeline sums loss components (not averages). The test mirrors this by manually summing `ConstantLoss(2.0)` and `ConstantLoss(3.0)`, obtaining $2 + 3 = 5$. Matches the pipeline's `total_loss += val` semantics exactly.

### test_loss_decreases_over_epochs — convergence check
- **Spec Reference:** N/A (empirical sanity check).
- **Implementation:** [tests/test_pipeline.py](tests/test_pipeline.py#L234-L260)
  ```python
  initial_loss = jnp.mean(initial_pred ** 2)
  final_loss   = jnp.mean(final_pred ** 2)
  self.assertLess(float(final_loss), float(initial_loss))
  ```
- **Verdict:** CORRECT
- **Notes:** The `DummyDataset` sets targets $y = V = 0$, so the pipeline's `MSELoss` computes $\frac{1}{ND}\sum_{i,j}f(x_i)_j^2$. The test's evaluation metric, `jnp.mean(pred ** 2)`, computes the same quantity over the full dataset (which equals the training set since `n=10, batch_size=10`). After 50 epochs of Adam with $\eta = 10^{-2}$, the loss should decrease for this trivially-overfittable problem. The assertion is consistent with what the pipeline optimises.

### Parameter freezing — `eqx.partition` correctness
- **Spec Reference:** N/A (software correctness, not geometry).
- **Implementation:** [tests/test_pipeline.py](tests/test_pipeline.py#L96-L170)
- **Verdict:** CORRECT
- **Notes:** Frozen parameters are partitioned out by `eqx.partition` and never passed to the optimizer. They should be bitwise identical after training, making `jnp.allclose` with default tolerances ($\text{rtol}=10^{-5}$, $\text{atol}=10^{-8}$) more than adequate.

## Findings

### 1. NOTE — test_multiple_losses_sum tests summation outside the pipeline
- **File:** [tests/test_pipeline.py](tests/test_pipeline.py#L192-L200)
- **Observation:** `test_multiple_losses_sum` manually sums two `ConstantLoss` calls with Python `+`. This verifies that the `LossComponent.__call__` returns the correctly weighted scalar, but it does not exercise the pipeline's internal accumulation loop (`total_loss += val` in [src/ham/training/pipeline.py](src/ham/training/pipeline.py#L44)). A model-independent `ConstantLoss` has $\nabla_\theta = 0$, so running it through the pipeline would produce no parameter update, limiting what can be asserted. The current approach is pragmatic; this note is for completeness only.

### 2. NOTE — Convergence test uses full-dataset evaluation identical to training loss
- **File:** [tests/test_pipeline.py](tests/test_pipeline.py#L249-L255)
- **Observation:** The convergence test evaluates `jnp.mean(final_pred ** 2)` on the same 10-point dataset used for training (with `batch_size=10`). This is effectively the training loss, not a generalisation metric. For a mathematical sanity check this is fine (the assertion is $\mathcal{L}_{\text{final}} < \mathcal{L}_{\text{init}}$, which must hold for a convergent optimiser on a small, noise-free problem), but it would not detect optimiser bugs that only manifest with mini-batching or distribution shift.

## Open Questions

1. The pipeline's `train_step` applies `jax.vmap(loss_fn, in_axes=(None, None, 0, 0))` followed by `jnp.mean`, making the batch loss the arithmetic mean of per-sample losses. Some loss formulations (e.g., the real `GeodesicSprayLoss` in [src/ham/training/losses.py](src/ham/training/losses.py)) may already perform internal reductions. There is no test here that checks for double-averaging when a production loss component is plugged in. This is not a bug in `test_pipeline.py` itself but a gap that could merit an integration test.
