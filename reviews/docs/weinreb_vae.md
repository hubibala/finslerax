# Documentation Review: `examples/weinreb_vae.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2026-05-15

## Summary

Overall documentation quality: **adequate**.

The script has a detailed module-level docstring and extensive inline section comments that explain the rationale behind each component. Loss components are individually motivated with clear "why" commentary. However, several issues reduce accessibility: the module-level docstring references a wrong filename, hyperparameter defaults are undocumented, the relationship to the paper's H1–H4 hypotheses is never stated, and multiple helper functions lack docstrings entirely. The dual-audience goal (mathematicians / ML engineers) is partially met — the geometric motivation ($G = J^\top J$, pullback metric) is mentioned but never formally defined, and there is no pointer to the spec for readers wanting rigour.

## Issue Tracker

| # | Severity | Location | Issue | Recommended Action |
|---|----------|----------|-------|--------------------|
| 1 | **INACCURATE** | `examples/weinreb_vae.py:2` | Module-level docstring says `weinreb_vae_diagnostic.py` but the actual filename is `weinreb_vae.py`. | Change line 2 to `weinreb_vae.py` (or rename the file to match). |
| 2 | **INACCURATE** | `examples/weinreb_vae.py:36` | Docstring references `preprocess_weinreb_spring.py` as the prerequisite script, but the workspace file is `examples/preprocess_weinreb.py`. | Update to `preprocess_weinreb.py`. |
| 3 | **MISSING** | `examples/weinreb_vae.py` (module-level) | No reference to the paper's hypotheses H1–H4. `spec/RESEARCH_LOG.md` § 3.1 states this script supports experiments H1–H4, and `spec/ARCH_SPEC.md` § 6 describes a phased approach (this is Phase 1 — VAE geometry). The module docstring should state which hypothesis/phase this script addresses and point to the experiment scripts that build on its output. | Add a "Relationship to experiments" paragraph, e.g.: *"This script produces the Phase 1 VAE checkpoint consumed by `experiment_h1_geometric.py` through `experiment_h4_simulation.py`. It validates the geometric foundation (H1) by ensuring the pullback metric $G = J^\top J$ is smooth and non-degenerate."* |
| 4 | **MISSING** | `examples/weinreb_vae.py:69–73` | `make_tanh_mlp` has a one-line docstring but does not document its parameters (`in_dim`, `out_dim`, `hidden`, `depth`, `key`), return type, or why `tanh` is chosen (the section comment explains it, but the docstring itself does not). | Add Args/Returns to the docstring, or at minimum a "See section comment above" pointer. |
| 5 | **MISSING** | `examples/weinreb_vae.py:361–403` | `build_diagnostic_vae` has a docstring explaining the design but does not document its **return type** (`GeometricVAE`). The `n_cell_types` parameter is also undocumented. | Add `Returns: GeometricVAE with classifier_head attached.` and document `n_cell_types`. |
| 6 | **MISSING** | `examples/weinreb_vae.py:409–414` | `attach_pullback_metric` does not document Args or Returns. An ML engineer reading the script cannot tell what `key` is used for without reading `PullbackRiemannian` source. | Add a brief Args/Returns docstring. |
| 7 | **MISSING** | `examples/weinreb_vae.py:420–431` | `get_trainable_mask` has no docstring. Its purpose (freeze everything except encoder, decoder, classifier) is non-obvious. | Add docstring: *"Returns an Equinox filter mask marking encoder_net, decoder_net, and classifier_head as trainable."* |
| 8 | **MISSING** | `examples/weinreb_vae.py:434–455` | `train_vae` has no docstring. This is the primary public entry point of the training pipeline. Parameters `kl_cycle_len`, `kl_beta_max`, `triplet_weight`, `triplet_margin`, `coherence_weight`, `cls_weight`, `vel_weight` are undocumented — an ML engineer cannot understand their effect without reading the loss class implementations. | Add a full docstring with Args, Returns, and a brief description of the training loop (cyclic KL annealing, multi-loss, manual epoch loop). |
| 9 | **UNCLEAR** | `examples/weinreb_vae.py:224–227` | `cyclic_beta` docstring is in the section comment above it, not in the function itself. The function has no docstring at all. The ramp schedule (linear for 50%, flat for 50%) is only described in the comment, not in the function body. | Add a docstring to `cyclic_beta` explaining the saw-tooth schedule, and note the relationship to posterior-collapse prevention per `spec/ARCH_SPEC.md` § 6 (known limitations around Hyperboloid VAE instability). |
| 10 | **UNCLEAR** | `examples/weinreb_vae.py:151–167` | `KNNTripletLoss.__call__` accesses `batch[0]`, `batch[1]`, `batch[2]` but the class docstring says these are "anchor PCA coords", "positive PCA coords", "negative PCA coords". In practice, the batch contains **gene-expression** (post-PCA, post-scaling) arrays, not raw PCA coordinates. The word "PCA" is misleading since StandardScaler has been applied. | Clarify: *"anchor gene-expression vectors (PCA-projected, standardised)"* or similar. |
| 11 | **UNCLEAR** | `examples/weinreb_vae.py:185–207` | `TrajectoryCoherenceLoss` docstring says `batch[0] = x_day2, batch[3] = x_day4, batch[4] = x_day6` but does not explain what `batch[1]` and `batch[2]` contain (velocity and labels respectively, per the `batch_main` assembly at line 556). A reader must jump 350 lines forward to understand the batch layout. | Add a complete batch layout reference at the top of the file or in the `train_vae` docstring: `batch_main = (x, v, labels, x_day4, x_day6)`. |
| 12 | **UNCLEAR** | `examples/weinreb_vae.py:267–310` | `VelocityConsistencyLoss` docstring describes the intent well but does not explain what `model.project_control` does. A mathematician would not know that this is JVP-based pushforward of RNA velocity through the encoder. | Add one sentence: *"Uses `model.project_control` (JVP pushforward through the encoder) to map data-space velocities to latent-space velocities."* |
| 13 | **MISSING** | `examples/weinreb_vae.py:612–621` | `encode_all` docstring is minimal — does not explain the batched inference strategy or why `eqx.filter_vmap` with `in_axes=(None, 0)` is used. | Add brief explanation: *"Encodes in batches of 1024 to avoid OOM. Uses filter_vmap to broadcast the model across the batch dimension."* |
| 14 | **MISSING** | `examples/weinreb_vae.py:636–660` | `compute_pullback_det` has a docstring but does not document the return tuple elements. The return is `(GX, GY, log_det_grid, pca2)` — four items — but the docstring says "Returns (grid_x, grid_y, log_det_grid)" (three items). | Fix the Returns line to include all four returned values. |
| 15 | **MISSING** | `examples/weinreb_vae.py:663–676` | `knn_preservation_score` has a docstring but does not document Args. The parameter `X` is described as "PCA space" but it is actually the standardised PCA matrix. | Add Args section with types and shapes. |
| 16 | **UNCLEAR** | `examples/weinreb_vae.py:26–28` | The module docstring mentions `data/weinreb_lineage_triples.npy` as a prerequisite but does not explain what a "lineage triple" is (day-2/day-4/day-6 index triples tracking the same clone). This is critical context for an ML engineer unfamiliar with the Weinreb dataset. | Add a one-line gloss: *"Each triple (i₂, i₄, i₆) tracks a clonal lineage across days 2, 4, 6."* |
| 17 | **MISSING** | `examples/weinreb_vae.py` (global) | No explanation of default hyperparameter values. Why `latent_dim=8`? Why `batch_size=512`? Why `kl_beta_max=5e-4`? Why `epochs=120` with `kl_cycle_len=20` (i.e. 6 full KL cycles)? These choices are critical for reproducibility but entirely undocumented. | Add a "Hyperparameters" section to the module docstring or as a comment block before `train_vae`, explaining the reasoning or citing ablation results (e.g., `train_vae_ablation.py`). |
| 18 | **TYPO** | `examples/weinreb_vae.py:700` | `plot_diagnostics` docstring says "6-panel figure" but the list only names panels [0]–[5] without labelling them, and the text for panel [1] says "coloured by day (if day info available, else by label)" but the implementation (line 733–743) actually highlights target fates, not days. | Fix panel [1] description to: *"Latent space with target fates highlighted."* |
| 19 | **MISSING** | `examples/weinreb_vae.py` (module-level) | No spec cross-references. The docstring should cite `spec/ARCH_SPEC.md` § 5 (module structure — `bio/vae.py`, `models/learned.py`) and `spec/MATH_SPEC.md` § 5 (Zermelo parameterisation) to help mathematicians find the formal definitions. | Add a "References" subsection to the module docstring. |
| 20 | **UNCLEAR** | `examples/weinreb_vae.py:315–355` | `ReconstructionLossDeterministic` docstring explains the dual-path (stochastic + deterministic) approach but does not clarify why the default split is 50/50 (`stochastic_weight=0.5`, `deterministic_weight=0.5`). This weighting has a direct effect on ELBO tightness vs. geometric stability. | Add a sentence explaining the trade-off and the rationale for the default. |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example/Pointer |
|---|---|---|---|---|---|
| `make_tanh_mlp` | Minimal | No | No | No | No |
| `build_knn_triplet_indices` | Yes | Partial (missing `labels`) | Yes | No | No |
| `KNNTripletLoss` | Yes | No (`__init__` args) | No | No | No |
| `TrajectoryCoherenceLoss` | Yes | Partial | No | Partial ($z_4 \approx 0.5(z_2 + z_6)$) | No |
| `cyclic_beta` | No | No | No | No | No |
| `AnnealedKLLoss` | Minimal | No | No | No | No |
| `CellTypeClassificationLoss` | No (class-level only via section comment) | No | No | No | No |
| `VelocityConsistencyLoss` | Yes | No | No | No | No |
| `ReconstructionLossDeterministic` | Yes | Partial | No | Partial ($G = J^\top J$) | No |
| `build_diagnostic_vae` | Yes | Partial | No | No | No |
| `attach_pullback_metric` | Minimal | No | No | No | No |
| `get_trainable_mask` | No | No | No | No | No |
| `train_vae` | No | No | No | No | No |
| `encode_all` | Minimal | Partial | Yes | No | No |
| `compute_pullback_det` | Yes | Partial | Incomplete (3 of 4) | Yes ($\log\det G(z)$) | No |
| `knn_preservation_score` | Yes | No | Yes | No | No |
| `plot_diagnostics` | Yes | Partial | No | No | No |
| `main` | No | N/A | N/A | N/A | N/A |

## Spec Alignment Notes

1. **`spec/ARCH_SPEC.md` § 6 ↔ module docstring:** The spec describes the Weinreb application as using `DataDrivenPullbackRanders`, but the module docstring explicitly states *"No Randers metric — PullbackRiemannian only."* This is **consistent** — this script is Phase 1 (geometry foundation) and the Randers wind is added in Phase 2. However, the script does not state this phasing explicitly, which could confuse a reader checking against the spec.

2. **`spec/MATH_SPEC.md` § 5 (Zermelo) ↔ script:** The script correctly excludes the Zermelo/Randers parameterisation, consistent with its stated scope. No misalignment.

3. **`spec/ARCH_SPEC.md` § 2.2 ↔ `VelocityConsistencyLoss`:** The loss calls `model.project_control`, which uses JVP pushforward through the encoder (see `src/ham/bio/vae.py:103–120`). The spec does not explicitly document this pushforward operation — it only describes `project_control` in the context of Zermelo dynamics. The script's usage (encoder-only pushforward, no wind) is valid but undocumented in the spec.

4. **`spec/RESEARCH_LOG.md` § 3.1 ↔ script:** The research log lists H1–H4 as validated, with `examples/experiment_h{1,2,3,4}_*.py` as the experiment scripts. This VAE script is their shared prerequisite (produces the Phase 1 checkpoint at `data/weinreb_vae_phase1.eqx`), but this dependency chain is not documented anywhere in the script or the spec.

5. **Filename mismatch:** The module docstring header says `weinreb_vae_diagnostic.py` (line 2), the actual file is `weinreb_vae.py`. The prerequisite script is listed as `preprocess_weinreb_spring.py` (line 36) but the workspace has `preprocess_weinreb.py`. Both are factual inaccuracies.
