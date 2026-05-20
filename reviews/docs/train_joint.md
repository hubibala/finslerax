# Documentation Review: `src/ham/bio/train_joint.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2025-05-15

## Summary

Overall documentation quality: **needs work**.

The module contains two public JIT-compiled step functions (`step_manifold`, `step_metric`) and one public class (`GeometricTrainer`). None of the three has a docstring beyond a single sentence for the class. No function-level argument/return documentation exists anywhere in the file. The module itself has no module-level docstring. Additionally, `train_joint.py` is not listed in `spec/ARCH_SPEC.md § 5` (Module Structure), is not exported from `ham.bio.__init__`, and is not referenced by any example script or test — making it effectively invisible to users.

## Issue Tracker

| # | Severity | Symbol / Section | Issue | Suggested Text |
|---|----------|-----------------|-------|----------------|
| 1 | **MISSING** | Module (top-level) | No module-level docstring. Users cannot discover the purpose of this file via `help()` or IDE tooltips. | `"""Dual-phase joint trainer for GeometricVAE.\n\nPhase 1 trains the VAE manifold (encoder/decoder + alignment via spray loss).\nPhase 2 trains the Finsler metric/wind field using lineage-pair contrastive alignment.\n\nSee also: ham.training.pipeline (HAMPipeline) for the declarative alternative.\n"""` |
| 2 | **MISSING** | `step_manifold()` | Entirely undocumented public function. No docstring for args, returns, or mathematical semantics. | See §Suggested Docstrings below. |
| 3 | **MISSING** | `step_metric()` | Entirely undocumented public function. The inline comments describe the algorithm but no formal docstring exists. Args/returns are unspecified. | See §Suggested Docstrings below. |
| 4 | **MISSING** | `GeometricTrainer` | Class docstring is a single sentence (`Dual-Phase Trainer for HAM.`). Constructor args, class attributes, and overall usage pattern are undocumented. | See §Suggested Docstrings below. |
| 5 | **MISSING** | `GeometricTrainer.train()` | No docstring. This is the primary entry point for users yet has zero documentation of its args, return value, or two-phase behavior. | See §Suggested Docstrings below. |
| 6 | **MISSING** | `GeometricTrainer.train_step_manifold()` | No docstring — thin wrapper whose delegation semantics are not explained. | Add: `"""Delegate to step_manifold with current model and optimizer state."""` |
| 7 | **MISSING** | `GeometricTrainer.train_step_metric()` | No docstring. | Add: `"""Delegate to step_metric with current model and optimizer state."""` |
| 8 | **INACCURATE** | `step_manifold()` return | Returns 6 values `(model, opt_state, loss, recon, kl, align)` but the unpacked `stats` tuple from `loss_fn` is `(recon, kl, spray, align)`. The function silently drops `spray` and returns `align` — this is undocumented and potentially misleading. `src/ham/bio/train_joint.py:29` | Document that `spray_loss` is not returned, or add it to the return tuple. |
| 9 | **UNCLEAR** | `step_metric()` | The contrastive loss uses `m.manifold._minkowski_dot` and `m.manifold._minkowski_norm` (private helpers on `Hyperboloid`). It is unclear to ML-engineer readers when this function is valid — it silently assumes a Minkowski-signature manifold. | Add a note: *"Assumes manifold has Minkowski inner product (e.g., Hyperboloid). Not valid for Sphere or EuclideanSpace."* |
| 10 | **UNCLEAR** | `step_metric()` | Hard-coded loss weights (`1.0 * loss_h_reg + 0.1 * loss_smooth`) are documented only in inline comments. No guidance on tuning or rationale beyond "Reg should be strong enough." | Recommend moving weight constants to function parameters or documenting the rationale in the docstring. |
| 11 | **INACCURATE** | `step_metric()` | Inline comment says *"Alignment is ~1.0"* — this is only true for unit-norm wind/tangent vectors on a specific manifold scale. The claim is not generally accurate and could mislead users tuning the loss. `src/ham/bio/train_joint.py:85` | Remove or qualify: *"Alignment magnitude depends on manifold scale and tangent-vector norms."* |
| 12 | **MISSING** | `step_metric()` | The mathematical connection to `spec/MATH_SPEC.md § 5` (Zermelo Parameterization) is not stated. The function trains $H$ and $W$ but never references the spec's Randers formula. | Add: *"Trains the Zermelo navigation data $(H, W)$ of a NeuralRanders metric. See spec/MATH_SPEC.md § 5."* |
| 13 | **INACCURATE** | `GeometricTrainer.__init__` | Type annotation says `learning_rate: float = 1e-3` but this is applied uniformly to both phases via a single `optax.adam`. The spec (`spec/ARCH_SPEC.md § 4`, item 4) describes per-phase optimizers in `HAMPipeline`. Users may expect per-phase LR here. | Document that a single optimizer/LR is shared across both phases, unlike `HAMPipeline`. |
| 14 | **MISSING** | Module | No `__all__` or export. The class is also absent from `ham.bio.__init__`, making it undiscoverable. `src/ham/bio/__init__.py:6` | Either add `GeometricTrainer` to `__all__` and `ham.bio.__init__`, or document this module as internal/legacy. |
| 15 | **MISSING** | Module | No example script or test references `train_joint.py` or `GeometricTrainer`. Without a usage example, the dual-phase workflow is opaque to new users. | Add a pointer to `train_modular.py` or create a minimal usage snippet in the class docstring. |
| 16 | **UNCLEAR** | `GeometricTrainer.train()`, Phase 2 | `stop_gradient` on latent encodings is critical design: *"We stop gradients on Z here to focus purely on the Metric field training."* This is only in a comment, not in the docstring. Mathematicians need to know that Phase 2 freezes the embedding. | Document in the method docstring that encoder parameters receive no gradients during Phase 2. |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|--------------|-----------------|-------------------|---------------|---------|
| `step_manifold()` | No | No | No | No | No |
| `step_metric()` | No (inline comments only) | No | No | No | No |
| `GeometricTrainer` | Partial (1 sentence) | No | N/A | No | No |
| `GeometricTrainer.__init__()` | No | No | N/A | No | No |
| `GeometricTrainer.train_step_manifold()` | No | No | No | No | No |
| `GeometricTrainer.train_step_metric()` | No | No | No | No | No |
| `GeometricTrainer.train()` | No | No | No | No | No |

## Spec Alignment Notes

1. **Not in ARCH_SPEC module tree.** `train_joint.py` does not appear in `spec/ARCH_SPEC.md § 5` (Module Structure). The spec lists `bio/vae.py` and `bio/data.py` only. Either the spec should be updated or this file should be marked internal/legacy.

2. **Relationship to `HAMPipeline` unclear.** `spec/ARCH_SPEC.md § 6`, item 4 describes `HAMPipeline` as the canonical multi-phase trainer with per-phase parameter freezing and modular losses. `GeometricTrainer` appears to be an earlier, monolithic alternative. No documentation clarifies the relationship or when to prefer one over the other.

3. **Zermelo parameterization.** `step_metric()` trains the Zermelo navigation data $(H, W)$ referenced in `spec/MATH_SPEC.md § 5`, but the docstring makes no mention of this. The constraint $\|W\|_h < 1$ from the spec is enforced elsewhere (inside `NeuralRanders`) but is not documented here as a pre/post-condition.

4. **`_minkowski_dot` / `_minkowski_norm` usage.** These are private methods on `Hyperboloid` (`src/ham/geometry/surfaces.py`). Using them in a nominally manifold-agnostic trainer creates an implicit coupling not documented in the spec.

## Suggested Docstrings

### `step_manifold`

```python
def step_manifold(model, opt_state, optimizer, x_batch, v_batch, key):
    """Single JIT-compiled gradient step for Phase 1 (manifold learning).

    Computes the batched GeometricVAE loss (reconstruction + KL + spray +
    alignment) and applies one optimizer update.

    Args:
        model: GeometricVAE instance.
        opt_state: Current optax optimizer state.
        optimizer: optax.GradientTransformation.
        x_batch: Gene-expression batch, shape ``(B, data_dim)``.
        v_batch: RNA-velocity batch, shape ``(B, data_dim)``.
        key: JAX PRNG key for sampling.

    Returns:
        Tuple of (new_model, new_opt_state, total_loss, recon_loss, kl_loss, align_loss).
        Note: spray_loss is computed internally but not returned.
    """
```

### `step_metric`

```python
def step_metric(model, opt_state, optimizer, z_parent, z_child):
    """Single JIT-compiled gradient step for Phase 2 (metric/wind learning).

    Trains the Zermelo navigation data (H, W) of a NeuralRanders metric
    by maximising alignment between the learned wind field W and the
    tangent direction from parent to child latent codes.

    The loss has three terms:
      - **Alignment:** ``-<W, log_map(parent, child)>`` (Minkowski inner product).
      - **H-anchor:** ``||H - I||^2``  — prevents H from collapsing.
      - **Smoothness:** ``||dW/dz||^2`` — Jacobian penalty on the wind field.

    Assumes the manifold supports ``_minkowski_dot`` (e.g., Hyperboloid).
    See spec/MATH_SPEC.md § 5 for the Zermelo parameterization.

    Args:
        model: GeometricVAE whose ``metric`` is a NeuralRanders instance.
        opt_state: Current optax optimizer state.
        optimizer: optax.GradientTransformation.
        z_parent: Parent latent codes, shape ``(B, ambient_dim)``.
        z_child: Child latent codes, shape ``(B, ambient_dim)``.

    Returns:
        Tuple of (new_model, new_opt_state, contrastive_loss).
    """
```

### `GeometricTrainer`

```python
class GeometricTrainer:
    """Dual-phase trainer coupling a GeometricVAE with Finsler metric learning.

    Phase 1 (manifold): trains encoder, decoder, and alignment using
    ``GeometricVAE.loss_fn`` (reconstruction + KL + spray + alignment).

    Phase 2 (metric): freezes the encoder (via ``stop_gradient``) and
    trains the NeuralRanders wind/metric field on lineage pairs.

    For a more flexible, declarative alternative see
    ``ham.training.pipeline.HAMPipeline``.

    Args:
        model: A ``GeometricVAE`` instance.
        learning_rate: Adam learning rate shared by both phases.
        seed: Random seed for data shuffling.

    Example:
        >>> trainer = GeometricTrainer(vae_model, learning_rate=1e-3)
        >>> trained_model = trainer.train(dataset, epochs_manifold=50, epochs_metric=25)
    """
```
