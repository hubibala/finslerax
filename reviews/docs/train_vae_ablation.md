# Documentation Review: `examples/train_vae_ablation.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2026-05-15

## Summary

Overall documentation quality: **adequate**.

The script has a concise module-level docstring that identifies the ablation variable (`vel_weight=0.0`) and the checkpoint it produces. However, several aspects important to an ablation study — the scientific rationale, how results are consumed downstream, hyperparameter provenance, and usage instructions — are missing or under-documented.

## Issue Tracker

| # | Severity | Symbol / Section | Location | Issue | Suggested Text |
|---|----------|-----------------|----------|-------|----------------|
| 1 | **MISSING** | Module docstring | `examples/train_vae_ablation.py:1-11` | No `Usage` or `Requires` section. The reader must guess the prerequisites and invocation command. `weinreb_vae.py` has both; this script should match. | Add:<br>`Usage`<br>`-----`<br>`python train_vae_ablation.py`<br><br>`Requires:`<br>`data/weinreb_preprocessed.h5ad`<br>`data/weinreb_lineage_triples.npy`<br>`(produced by preprocess_weinreb.py)` |
| 2 | **MISSING** | Module docstring | `examples/train_vae_ablation.py:1-11` | No scientific rationale explaining **why** zeroing `vel_weight` is a meaningful ablation. An ML engineer can see *what* changed but not *why* it matters. | Add a short paragraph, e.g.: `Setting vel_weight=0.0 removes the velocity-consistency loss, which enforces alignment between RNA velocity and the latent displacement vector. Without it, the decoder Jacobian is unconstrained by velocity information, isolating the contribution of the velocity signal to downstream Randers-metric quality (H2) and discriminative geodesic cost (H3).` |
| 3 | **MISSING** | Module docstring | `examples/train_vae_ablation.py:1-11` | The docstring references "H2/H3" experiments but does not name or link the scripts that consume the ablated checkpoint (`experiment_h2_directional.py`, `experiment_h3_discriminative.py`). Tracing the experimental pipeline requires grep. | Add: `Consumed by: experiment_h2_directional.py, experiment_h3_discriminative.py` |
| 4 | **UNCLEAR** | Module docstring | `examples/train_vae_ablation.py:8-9` | The comparison table lists only two conditions. It is not clear whether this is the *complete* ablation design or one arm of a larger study. No mention of how many ablation axes exist or whether other scripts produce other ablated checkpoints. | Clarify, e.g.: `This is the only ablation checkpoint; all other hyperparameters are held at Phase-1 defaults (see weinreb_vae.py).` |
| 5 | **MISSING** | `main()` | `examples/train_vae_ablation.py:30-99` | The `main()` function has no docstring. While the body is straightforward, a one-liner would help automated doc generators and readers scanning the file. | Add: `"""Train the ablated VAE (vel_weight=0) and save checkpoint."""` |
| 6 | **MISSING** | Hyperparameter block | `examples/train_vae_ablation.py:78-93` | Fourteen hyperparameters are passed to `train_vae()`. There is no comment or docstring explaining why these values were chosen, whether they match `weinreb_vae.py` Phase 1, or which are the "defaults" vs. the ablation variable. Only `vel_weight` has a comment. | Add a block comment before the call, e.g.: `# All hyperparameters match weinreb_vae.py Phase 1 defaults`<br>`# except vel_weight, which is zeroed for the ablation.` |
| 7 | **UNCLEAR** | `kl_cycle_len`, `kl_beta_max` | `examples/train_vae_ablation.py:86-87` | Cyclical KL annealing parameters (`kl_cycle_len=20`, `kl_beta_max=5e-4`) appear without explanation. A mathematician unfamiliar with VAE training would not know what these control. A brief inline comment would help the dual audience. | Add inline comments: `kl_cycle_len = 20,    # KL annealing: restart cycle length (epochs)`<br>`kl_beta_max  = 5e-4,  # KL annealing: max β weight` |
| 8 | **MISSING** | Random seed | `examples/train_vae_ablation.py:77` | `jax.random.PRNGKey(2026)` — the seed differs from `weinreb_vae.py` (which uses `PRNGKey(0)` or `PRNGKey(42)`) and the choice is undocumented. For reproducibility reporting, seed provenance should be noted. | Add comment: `# Seed 2026 chosen to differ from Phase 1 (seed 42) for independence.` or explain the rationale. |
| 9 | **TYPO** | Module docstring | `examples/train_vae_ablation.py:4` | States the script is "identical to weinreb_vae.py phase 1" but `weinreb_vae.py`'s own docstring calls itself `weinreb_vae_diagnostic.py` (line 2). The naming mismatch may confuse readers about which file is canonical. | Align naming: either say "identical to the Phase 1 training in `weinreb_vae.py`" or rename the docstring header in `weinreb_vae.py`. |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:------------:|:---------------:|:------------------:|:-------------:|:-------:|
| Module-level  | ✅ | N/A | N/A | ❌ | ❌ |
| `main()`      | ❌ | N/A | N/A | N/A | N/A |

## Spec Alignment Notes

- `spec/ARCH_SPEC.md § 5` (line 198) mentions "Experiments H1-H4 validate geometric topology, directional asymmetry, discriminative cost, and forward predictive simulation." The ablation script supports H2/H3 but the spec does not mention an ablation arm or `vel_weight` as a controlled variable. Either the spec should document the ablation design, or the script should note that the ablation is supplementary and not part of the core H1-H4 protocol.
- Neither `spec/ARCH_SPEC.md` nor `spec/MATH_SPEC.md` mentions `vel_weight` or the velocity-consistency loss by name. The script's ablation rationale cannot currently be cross-referenced against spec documentation.
