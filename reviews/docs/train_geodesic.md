# Documentation Review: `src/ham/bio/train_geodesic.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2025-05-15

## Summary

Overall documentation quality: **needs work**.

The module contains zero docstrings — no module-level docstring, no class docstring for `GeodesicFlowTrainer`, and no method docstrings for `__init__`, `train_step`, or `train_phase2`. All public API is entirely undocumented. While inline comments provide partial intuition, they are not a substitute for formal docstrings and do not follow the project's documentation conventions. Additionally, `GeodesicFlowTrainer` is not exported from `src/ham/bio/__init__.py` and is not listed in `spec/ARCH_SPEC.md § 5` (Module Structure), raising a spec-alignment concern.

---

## Issue Tracker

| # | Severity | Symbol / Section | Issue | Suggested Text |
|---|----------|-----------------|-------|----------------|
| 1 | **MISSING** | Module (file-level) | No module-level docstring. The file provides no summary of purpose, relationship to Phase 2 training, or connection to the AVBD solver. | `"""Phase 2 geodesic regression trainer.\n\nTrains a GeometricVAE's metric parameters by minimising the action (geodesic energy)\nalong parent→child lineage trajectories using the AVBDSolver. The encoder is frozen\n(stop-gradient); only metric parameters receive gradients.\n\nSee Also:\n    ham.solvers.avbd.AVBDSolver\n    ham.bio.vae.GeometricVAE\n"""` |
| 2 | **MISSING** | `GeodesicFlowTrainer` (class) | No class-level docstring. Users and maintainers cannot discover purpose, expected `model` type, or training protocol from introspection. | `"""Geodesic regression trainer for Phase 2 of the HAM training pipeline.\n\nGiven a model with a Zermelo-parameterised Randers metric, optimises metric\nparameters so that geodesics between parent–child pairs minimise the Finsler\naction energy, subject to a metric-identity regularisation anchor.\n\nArgs:\n    model: An Equinox module exposing ``model.metric._get_zermelo_data(x)``\n        and ``model.encode(x, key)`` (typically a ``GeometricVAE``).\n    learning_rate: Adam learning rate (default ``1e-3``).\n"""` |
| 3 | **MISSING** | `__init__` | No docstring. Constructor parameters `model` and `learning_rate` are undocumented. Expected type for `model` is unclear. | Add a brief docstring or rely on the class-level docstring (see issue 2). At minimum, type annotations should be added to the signature: `model: eqx.Module, learning_rate: float = 1e-3`. |
| 4 | **MISSING** | `train_step` | No docstring. This is a JIT-compiled method with non-obvious semantics (batched BVP solve, energy + regularisation loss). Args, returns, and mathematical description are all absent. | `"""Single gradient step of geodesic regression.\n\nSolves the BVP from z_parent → z_child via AVBD, computes the\nFinsler action loss $\\mathbb{E}[E(\\gamma)]$, adds a metric-identity\nregularisation term, and applies one Adam update.\n\nArgs:\n    model: Current model parameters (Equinox module).\n    opt_state: Current optimiser state.\n    z_parent: Latent codes of parent cells, shape ``(B, D)``.\n    z_child: Latent codes of child cells, shape ``(B, D)``.\n\nReturns:\n    Tuple of (updated_model, updated_opt_state, scalar_loss).\n"""` |
| 5 | **MISSING** | `train_phase2` | No docstring. This is the main entry point for Phase 2 training. Args (`dataset`, `epochs`, `batch_size`), expected `dataset` type (`BioDataset`), return value, and side-effects (prints to stdout) are all undocumented. | `"""Run Phase 2 geodesic regression over lineage pairs.\n\nPre-encodes the full dataset with a frozen encoder, then trains\nthe metric parameters by minimising geodesic action over\nparent→child lineage pairs.\n\nArgs:\n    dataset: A ``BioDataset`` with non-None ``lineage_pairs``.\n    epochs: Number of training epochs (default ``50``).\n    batch_size: Mini-batch size for lineage pairs (default ``64``).\n\nReturns:\n    The updated model with trained metric parameters.\n"""` |
| 6 | **UNCLEAR** | `train_step` lines 22–28 | Inline comment references "Randers: $F(v) = \|v\|_M - \langle W, v \rangle$". This uses a *minus* sign, but `spec/MATH_SPEC.md § 5` gives the full navigation formula with a different structure. The simplified inline formula may mislead mathematicians about which convention is in use. | Recommended action: Replace the inline comment with a reference to `spec/MATH_SPEC.md § 5` and the Zermelo navigation formula, or clarify that the minus sign is the "headwind increases cost" convention per the spec note. |
| 7 | **UNCLEAR** | `train_step` lines 33–36 | The regularisation anchor forces $M(x) \approx I$ and penalises $\|W\|^2$. The mathematical motivation (preventing metric collapse) is only hinted at in an inline comment. An ML engineer would not understand why this is necessary; a mathematician would want the precise penalty functional. | Recommended action: Document the regularisation term in the `train_step` docstring: "Regularisation: $\lambda_{\mathrm{reg}} \bigl( \|M(x) - I\|_F^2 + 0.1\|W(x)\|^2 \bigr)$ averaged over sub-sampled trajectory points, preventing metric degeneration." |
| 8 | **INACCURATE** | `train_step` line 34 | Comment says "H(x) approx Identity" but the code accesses `M, W, _` — the variable is `M`, not `H`. The comment should say "$M(x) \approx I$". | Replace `H(x) approx Identity` with `M(x) ≈ I` to match the code variable name and spec notation (`spec/MATH_SPEC.md § 5`). |
| 9 | **MISSING** | `train_phase2` line 62 | The encoder is frozen via `stop_gradient`, but there is no docstring or comment explaining *why* this is the recommended strategy. The code comment "In a real rigorous setting…" is informal and potentially confusing. | Recommended action: Add a note in the docstring: "The encoder is frozen (stop-gradient) for stability; fine-tuning it jointly is possible but risks Phase 1 collapse." |
| 10 | **MISSING** | `__init__` line 14 | The `AVBDSolver` is instantiated with hard-coded `step_size=0.1, iterations=15` annotated as "Fast settings for training". These are not configurable. Users cannot discover that training-time solver settings differ from default without reading source. | Recommended action: Document in the class docstring that AVBD solver settings are tuned for training speed, or expose them as constructor parameters. |

---

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:------------:|:---------------:|:------------------:|:-------------:|:-------:|
| `GeodesicFlowTrainer` (class) | ✗ | ✗ | N/A | ✗ | ✗ |
| `GeodesicFlowTrainer.__init__` | ✗ | ✗ | N/A | ✗ | ✗ |
| `GeodesicFlowTrainer.train_step` | ✗ | ✗ | ✗ | ✗ | ✗ |
| `GeodesicFlowTrainer.train_phase2` | ✗ | ✗ | ✗ | ✗ | ✗ |

---

## Spec Alignment Notes

1. **Not listed in `spec/ARCH_SPEC.md § 5`:** The module structure tree lists `bio/vae.py` and `bio/data.py` but does **not** list `bio/train_geodesic.py`. Either the spec is out-of-date or this module is considered internal/experimental. Recommended action: update `spec/ARCH_SPEC.md § 5` to include `train_geodesic.py`, or document this module as private.

2. **Not exported from `src/ham/bio/__init__.py`:** `GeodesicFlowTrainer` is absent from `__all__` in [src/ham/bio/__init__.py](src/ham/bio/__init__.py). If this is intentional (internal module), it should be documented with a leading underscore or an explicit "internal" note. If public, it should be exported.

3. **Randers formula convention (`spec/MATH_SPEC.md § 5`):** The inline comment on line 28 uses a simplified $F(v) = \|v\|_M - \langle W, v \rangle$ form. The spec gives the full Zermelo navigation formula $F(x,v) = \frac{\sqrt{\lambda\|v\|_h^2 + \langle W,v\rangle_h^2} - \langle W,v\rangle_h}{\lambda}$. The comment's simplification is not mathematically wrong as a conceptual gloss, but the notation $\|\cdot\|_M$ vs $\|\cdot\|_h$ is inconsistent with the spec (which uses $h_{ij}$ for the Riemannian "sea" metric). This should be reconciled.

4. **Training pipeline relationship (`spec/ARCH_SPEC.md § 6.4`):** The spec states that `HAMPipeline` supports "multi-phase training with per-phase parameter freezing." This module implements Phase 2 independently of `HAMPipeline`. Documentation should clarify whether this is a standalone alternative or is called by `HAMPipeline`.
