# Documentation Review: `src/ham/bio/train_modular.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** May 15, 2026

---

## Summary

Overall documentation quality: **needs work.**

`train_modular.py` is a runnable entry-point script that demonstrates the two-phase HAMPipeline workflow (manifold pretraining → metric learning). It exposes one public helper (`get_filter_fn`) and one entry-point (`main`). Neither the module nor `main()` carries a docstring; the existing docstring on `get_filter_fn` is incomplete. The file is also absent from `spec/ARCH_SPEC.md` § 5 (Module Structure), creating a discoverability gap for both audiences.

---

## Issue Tracker

| # | Severity | Symbol / Section | Issue | Suggested Text / Recommended Action |
|---|----------|-----------------|-------|--------------------------------------|
| 1 | **MISSING** | Module | No module-level docstring. The file purpose, relationship to `train_joint.py` / `train_geodesic.py`, and the two-phase pipeline concept are undocumented. | Add a module docstring: `"""Declarative two-phase training script for GeometricVAE + NeuralRanders via the HAMPipeline. Phase 1 pretrains the encoder/decoder (manifold pretraining); Phase 2 learns the metric from lineage supervision. See spec/ARCH_SPEC.md § 4 and § 6 for pipeline details. Usage: python -m ham.bio.train_modular"""` |
| 2 | **MISSING** | `main()` (`train_modular.py:46`) | The `main()` function has no docstring. All hyperparameter choices (seed, latent_dim, hidden_dim, batch_size, epochs, loss weights) are hard-coded with no explanation of rationale or tuning guidance. | Add a docstring listing the default configuration and noting that parameters should be adjusted for real data. |
| 3 | **UNCLEAR** | `get_filter_fn` (`train_modular.py:23`) | The docstring mentions "`eqx.partition`" and shows a lambda example, but does not document (a) the `selector` parameter type, (b) the return type (`Callable[[eqx.Module], PyTree[bool]]`), or (c) the semantic purpose: selecting which model parameters are trainable during a `TrainingPhase`. An ML engineer unfamiliar with Equinox would not understand how to write a valid selector. | Suggested replacement: `"""Returns a boolean pytree filter function for eqx.partition. Args: selector: A callable that, given the full model pytree, returns the sub-tree(s) whose leaves should be trainable. E.g., ``lambda m: (m.encoder_net, m.decoder_net)`` to train only the VAE components. Returns: A callable ``filter_spec(model) -> PyTree[bool]`` where ``True`` marks trainable leaves. This is passed to ``TrainingPhase.filter_spec``."""` |
| 4 | **INACCURATE** | `main()` (`train_modular.py:64`) | The code instantiates `Hyperboloid(intrinsic_dim=latent_dim)` as the base manifold, but `spec/ARCH_SPEC.md` § 6 ("Known Limitations", item 6) explicitly warns that Hyperboloid-based joint training is **numerically sensitive** and recommends `EuclideanSpace` as the default for biological applications. The script contains no warning or comment about this, so a user running it on real data will hit solver collapse without explanation. | Add a comment or docstring note: `# NOTE: Hyperboloid is used here for demonstration. For biological data, EuclideanSpace is recommended (see ARCH_SPEC § 6).` |
| 5 | **MISSING** | `main()` (`train_modular.py:55`) | `DataLoader(mode='simulation').get_jax_data(use_pca=False)` — the choice of `mode='simulation'` and `use_pca=False` is not explained. A bioinformatics user would expect `mode='real'` with their own `.h5ad` path. | Add an inline comment: `# 'simulation' mode generates synthetic data for testing; switch to 'real' and pass path='...' for actual datasets.` |
| 6 | **MISSING** | Spec alignment | `train_modular.py` is **not listed** in the module structure at `spec/ARCH_SPEC.md` § 5. The bio subdirectory only lists `vae.py` and `data.py`. Similarly the file is not exported from `src/ham/bio/__init__.py`. | Either add the file to the spec module tree (alongside `train_geodesic.py` and `train_joint.py`) or clarify in the module docstring that this is an example/script, not a library module. |
| 7 | **UNCLEAR** | Phase 1 definition (`train_modular.py:73–84`) | The loss weights (`ReconstructionLoss(1.0)`, `KLDivergenceLoss(1e-4)`, `ZermeloAlignmentLoss(0.1)`, `GeodesicSprayLoss(1e-3)`) are uncommented. A mathematician would want to know which loss terms correspond to which variational objective; an ML engineer would want to know the sensitivity to these weights. | Add a brief comment block above phase_1 explaining the loss decomposition. E.g., `# Reconstruction + KL form the ELBO; ZermeloAlignment encourages latent velocity ↔ Wind alignment; GeodesicSpray penalises acceleration to promote geodesic paths (see MATH_SPEC § 2).` |
| 8 | **UNCLEAR** | Phase 2 definition (`train_modular.py:88–97`) | `ContrastiveAlignmentLoss`, `MetricAnchorLoss`, and `MetricSmoothnessLoss` are listed without explanation of their roles or why they apply only when `lineage_pairs` are available. | Add a comment: `# Phase 2 uses biological lineage supervision: ContrastiveAlignment aligns Wind with parent→child direction; MetricAnchor regularises H→I; MetricSmoothness penalises Jacobian of W.` |
| 9 | **MISSING** | Cross-references | No pointer to related example scripts (`examples/experiment_h1_geometric.py` through `experiment_h4_simulation.py`) or to the more mature `train_joint.py` / `train_geodesic.py` modules that address similar workflows. | Add a "See Also" section in the module docstring referencing `train_joint.py`, `train_geodesic.py`, and relevant example scripts. |
| 10 | **TYPO** | `get_filter_fn` (`train_modular.py:25`) | Minor: the docstring says "e.g.," but the trailing example is a bare lambda with no closing context or rendered formatting hint. | Wrap the example in a `.. code-block::` or backtick block for clarity: `` `lambda m: (m.encoder_net, m.decoder_net)` ``. |
| 11 | **MISSING** | `get_filter_fn` (`train_modular.py:23`) | No `Raises` documentation. If `selector` returns a non-existent attribute, `eqx.tree_at` will raise a cryptic error. | Document: `Raises: AttributeError if the selector references model attributes that do not exist.` |
| 12 | **UNCLEAR** | `main()` (`train_modular.py:101`) | `pipeline.fit(dataset, phases, batch_size=256, seed=seed)` — the `seed` parameter is reused from the outer scope (line 48) but it is unclear whether this restarts the PRNG sequence or continues it, which matters for reproducibility. | Add a comment: `# seed is re-used to initialise the pipeline's own PRNG; the model init used a separate key derived from the same seed.` |

---

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:------------:|:---------------:|:-----------------:|:-------------:|:-------:|
| `get_filter_fn` | Yes (partial) | No | No | N/A | Inline lambda shown |
| `main` | No | N/A (no params) | N/A | N/A | Is itself the example |

---

## Spec Alignment Notes

1. **ARCH_SPEC § 5 — Module Structure:** `train_modular.py` is absent from the listed bio/ contents (`vae.py`, `data.py` only). The three training scripts (`train_geodesic.py`, `train_joint.py`, `train_modular.py`) should all be listed or explicitly scoped as runnable examples.

2. **ARCH_SPEC § 6, item 4 — Training Pipeline:** The spec states that `HAMPipeline` supports "per-phase parameter freezing, lineage-triple batching, and modular losses." `train_modular.py` demonstrates parameter freezing and modular losses but uses `lineage_pairs`, not lineage **triples**. The file does not surface the triple-batching capability documented in the spec and implemented in `pipeline.py` (which handles `lineage_triples` separately). This creates a perception gap: a user reading only this script would not know triples are supported.

3. **ARCH_SPEC § 6, item 6 — Hyperboloid VAE limitation:** The script uses `Hyperboloid` without acknowledging the known instability. See Issue #4 above.

4. **MATH_SPEC § 5 — Zermelo Parameterization:** The `ZermeloAlignmentLoss` and `GeodesicSprayLoss` in Phase 1 directly implement concepts from MATH_SPEC § 5 (Randers metric via Zermelo navigation) and § 2 (Geodesic Spray), but these cross-references are absent from the code. Adding them would bridge the gap for mathematician-readers.
