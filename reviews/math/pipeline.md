# Math Review: pipeline

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** `src/ham/training/pipeline.py`

## Summary

**Correct.** The pipeline file is an optimisation-infrastructure module. It contains no
novel geometric or physical formulae; its mathematical content is limited to standard
mini-batch stochastic gradient descent: additive loss aggregation, batched gradient
estimation via `vmap`/mean, and optax-based parameter updates. All operations are
analytically correct and consistent with `spec/MATH_SPEC.md`. No critical or
warning-level issues were found.

---

## Formula-by-Formula Audit

### 1. Additive Loss Aggregation

- **Spec Reference:** N/A (standard optimisation convention)
- **Implementation:** `src/ham/training/pipeline.py:44–50`

```python
total_loss = 0.0
for loss_comp in phase.losses:
    loss_key, step_key = jax.random.split(step_key)
    val = loss_comp(full_model, batch_data, loss_key)
    total_loss += val
```

Computes $\mathcal{L}_{\text{total}} = \sum_i \mathcal{L}_i(\theta; x)$, where each
$\mathcal{L}_i$ internally multiplies by its own weight $w_i$ (set in the
`LossComponent` subclass). Differentiation commutes with summation, so
$\nabla_\theta \mathcal{L}_{\text{total}} = \sum_i \nabla_\theta \mathcal{L}_i$.

- **Verdict:** OK
- **Notes:** None.

---

### 2. Mini-batch Gradient Estimation (vmap + mean)

- **Spec Reference:** N/A (standard SGD)
- **Implementation:** `src/ham/training/pipeline.py:55–60`

```python
def batch_loss(d_m):
    l, s = jax.vmap(loss_fn, in_axes=(None, None, 0, 0))(
        d_m, static, batch_data, batch_keys)
    return jnp.mean(l), jax.tree_util.tree_map(jnp.mean, s)

(loss, stats), grads = eqx.filter_value_and_grad(batch_loss, has_aux=True)(diff)
```

Implements the standard estimator:
$$\hat{g}_t = \nabla_\theta \left[\frac{1}{B}\sum_{b=1}^{B}
\mathcal{L}(\theta_t;\, x_b)\right]
= \frac{1}{B}\sum_{b=1}^{B} \nabla_\theta \mathcal{L}(\theta_t;\, x_b)$$

`in_axes=(None, None, 0, 0)` correctly broadcasts the model parameters over the batch
while mapping each sample to its own data slice and PRNG key.
`eqx.filter_value_and_grad` differentiates only w.r.t. the differentiable partition
`diff`, treating `static` (closed over) as a constant.

- **Verdict:** OK
- **Notes:** None.

---

### 3. Optimizer Update

- **Spec Reference:** N/A (optax standard)
- **Implementation:** `src/ham/training/pipeline.py:62–64`

```python
updates, new_state = phase.optimizer.update(grads, state, diff)
new_diff = eqx.apply_updates(diff, updates)
```

Standard parameter update: $\theta_{t+1} = \theta_t + \Delta\theta_t$, where
$\Delta\theta_t$ is produced by the optax `GradientTransformation`. No mathematical
concern.

- **Verdict:** OK
- **Notes:** None.

---

### 4. Parameter Partitioning

- **Spec Reference:** N/A (Equinox mechanism)
- **Implementation:** `src/ham/training/pipeline.py:36–37`

```python
trainable_mask = phase.filter_spec(self.model)
diff_model, static_model = eqx.partition(self.model, trainable_mask)
```

Splits the model pytree into differentiable and frozen subtrees.
Gradients are computed only for `diff_model`; `static_model` is constant.
After each phase the recombination (`eqx.combine`, line 112) preserves all learned
parameters for subsequent phases.

- **Verdict:** OK
- **Notes:** None.

---

### 5. Epoch Loss Reporting

- **Spec Reference:** N/A
- **Implementation:** `src/ham/training/pipeline.py:99–100, 106–108`

```python
epoch_loss += loss
...
epoch_loss / steps_per_epoch
```

Reports the arithmetic mean of per-step losses. Because
`steps_per_epoch = max(1, num_items // batch_size)` (line 93) and the loop runs
exactly `steps_per_epoch` iterations, every processed batch has exactly `batch_size`
samples. The mean-of-means therefore equals the true mean over all processed samples:

$$\bar{\mathcal{L}}_{\text{epoch}} =
\frac{1}{S}\sum_{s=1}^{S}\frac{1}{B}\sum_{b=1}^{B}\mathcal{L}_{s,b}
= \frac{1}{SB}\sum_{i=1}^{SB}\mathcal{L}_i$$

- **Verdict:** OK
- **Notes:** Samples beyond `(num_items // batch_size) * batch_size` are silently
  dropped each epoch. This is common practice and does not affect gradient correctness
  (the permutation ensures uniform coverage across epochs).

---

### 6. PRNG Key Management

- **Spec Reference:** N/A
- **Implementation:** `src/ham/training/pipeline.py:31, 76, 99, 101`

```python
key = jax.random.PRNGKey(seed)          # line 31
key, subkey = jax.random.split(key)     # line 76
step_key = jax.random.fold_in(subkey, step)  # line 101
```

Each epoch receives a unique `subkey` (via `split`). Within an epoch, each step
receives a deterministic key via `fold_in(subkey, step)`. Inside the JIT'd function,
the step key is further split into per-sample keys. This yields an injective mapping
from `(seed, epoch, step, sample)` to a PRNG key, ensuring no key reuse.

- **Verdict:** OK
- **Notes:** None.

---

## Open Questions

1. **Interaction with stochastic losses:** Several `LossComponent` subclasses (e.g.,
   `ReconstructionLoss`, `ContrastiveAlignmentLoss`) sample from a distribution inside
   the loss call. The pipeline passes a unique key per sample (via vmap), which
   is correct. However, the gradient estimator is the *single-sample* reparameterised
   gradient. Whether this has acceptable variance is an empirical question outside the
   scope of this review, but mathematically the estimator is unbiased.

2. **Phase-to-phase weight transfer:** The recombination at line 112
   (`self.model = eqx.combine(diff_model, static_model)`) assumes that `diff_model`
   and `static_model` have compatible pytree structures. This is guaranteed by
   `eqx.partition`/`eqx.combine` contract, but if a `filter_spec` ever returns a mask
   inconsistent with the model's current structure, it would fail at runtime.
   Mathematically this is not an issue; it is an API contract.
