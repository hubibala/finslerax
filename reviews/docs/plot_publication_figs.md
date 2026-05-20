# Documentation Review: `examples/plot_publication_figs.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2026-05-15

## Summary

Overall documentation quality: **adequate**.

The script has a strong module-level docstring that clearly describes the 3-panel figure. Helper functions have concise docstrings. However, several inline documentation gaps make it harder for a reader to understand data transformations, the relationship between this script and the H1–H4 experiments, and the precise mathematical quantities being visualized. Notation in cost annotations is inconsistent with the spec.

---

## Issue Tracker

| #  | Severity     | Location (file:line)                                    | Issue | Suggested Text / Action |
|----|--------------|---------------------------------------------------------|-------|-------------------------|
| 1  | **MISSING**  | `examples/plot_publication_figs.py:1–16`                | Module docstring does not state which experiments (H1–H4) this figure supports, nor does it reference the spec. A reader cannot tell how this figure relates to the paper's claims. | Add a line such as: `"This figure accompanies Experiments H1 (geometric topology) and H2 (directional asymmetry), see spec/ARCH_SPEC.md § 5, item 5."` |
| 2  | **MISSING**  | `examples/plot_publication_figs.py:1–16`                | Module docstring does not mention required input files (`data/weinreb_vae_phase1.eqx`, `data/weinreb_preprocessed.h5ad`, `data/weinreb_test_triples.npy`) or how to produce them. A new user cannot run this script without hunting through other scripts. | Add a "Prerequisites" or "Usage" block listing required files and which scripts generate them (e.g., `preprocess_weinreb.py`, `weinreb_vae.py`). |
| 3  | **MISSING**  | `examples/plot_publication_figs.py:1–16`                | No indication of expected output filename or format. The output is `weinreb_publication_figure.png` at 220 dpi, but this is only visible deep inside `main()`. | Add to module docstring: `"Output: weinreb_publication_figure.png (220 dpi)."` |
| 4  | **UNCLEAR**  | `examples/plot_publication_figs.py:48–66`               | `compute_logdet_grid` docstring says *"Evaluate log det J^T J"* but does not clarify that $J$ is the decoder Jacobian, that $G(z) = J^T J$ is the pullback Riemannian metric tensor, or what `vae` model is expected. For a mathematician, the provenance of $J$ is ambiguous; for an ML engineer, the connection to the VAE decoder is implicit. | Expand to: `"Evaluate log det G(z) = log det(J^T J) where J = d(decoder)/dz is the decoder Jacobian. The pullback metric G(z) measures how the decoder stretches latent space. vae must expose a .decode() method."` |
| 5  | **MISSING**  | `examples/plot_publication_figs.py:48`                  | `compute_logdet_grid` — parameter `vae` has no type annotation or docstring description. Reader cannot tell what interface it requires. | Add `vae: GeometricVAE` type hint and document that it must have a `.decode()` method. |
| 6  | **MISSING**  | `examples/plot_publication_figs.py:68–96`               | `compute_wind_grid` docstring does not explain what "the wind vector $W(z)$" is or how it relates to the Randers/Zermelo formulation in `spec/MATH_SPEC.md § 5`. | Add: `"W(z) is the Zermelo wind field from the Randers metric (see spec/MATH_SPEC.md § 5). The wind encodes directional bias — transport with the wind is cheaper than against it."` |
| 7  | **MISSING**  | `examples/plot_publication_figs.py:68`                  | `compute_wind_grid` — parameter `vae` has no type annotation or docstring description. Same issue as #5. | Add type hint and note it must expose `vae.metric.w_net`. |
| 8  | **UNCLEAR**  | `examples/plot_publication_figs.py:98–101`              | `arc_length_segment` docstring says *"Finsler arc length of the straight line from z_start to z_end"* but does not clarify that this is an approximation — the straight line in latent space is not a geodesic, so this is not the true Finsler distance. The cost label $E$ on the figure may mislead readers into thinking this is an energy functional. | Clarify: `"Approximate Finsler arc length along the straight-line chord from z_start to z_end (not a geodesic). Uses trapezoidal discretization with midpoint evaluation."` |
| 9  | **INACCURATE** | `examples/plot_publication_figs.py:355–360`           | The annotation labels the per-path cost as `$E$` (energy), but `arc_length_segment` computes $\int F(z, \dot{z})\,dt$ which is **length**, not energy $\int F^2(z, \dot{z})\,dt$. The symbol $E$ typically denotes the energy functional in Finsler geometry (see `spec/MATH_SPEC.md § 3`). | Change the annotation label from `$E$` to `$L$` or `$\ell$` (length), or clarify in the title/legend that $E$ here means "cost" not energy. |
| 10 | **UNCLEAR**  | `examples/plot_publication_figs.py:345–350`            | `e_obs1` and `e_cf1` are computed as `arc_fn(vae_randers.metric, z2, z4)` — they are identical. The variable names suggest one is "observed" and one is "counterfactual", but the segment `z2→z4` is shared by both paths. No comment explains this intentional sharing. | Add inline comment: `# z2→z4 is the shared initial segment; both paths diverge after Day 4.` |
| 11 | **MISSING**  | `examples/plot_publication_figs.py:155–175`             | The data-loading block has no inline comments explaining what `X_pca`, `V_pca`, `ct_series`, or `time_point` columns represent, nor why `StandardScaler` normalization is applied. | Add brief comments, e.g., `# X_pca: 50-dim PCA of gene expression`, `# V_pca: RNA velocity projected into PCA space`, `# Normalize to zero-mean unit-variance for VAE input`. |
| 12 | **MISSING**  | `examples/plot_publication_figs.py:183–192`             | No comment explaining why a 20 000-cell subsample is taken, or how the random seed `0` was chosen. | Add: `# Subsample 20k cells for plotting speed; seed=0 for reproducibility.` |
| 13 | **UNCLEAR**  | `examples/plot_publication_figs.py:108`                 | `find_exemplar_triple` docstring says *"Return (z2, z4, z6_obs, z6_counter)"* but the return signature in the code is `(z2, z4, z6, cf_centroid)`. The names `z6_obs` and `z6_counter` in the docstring don't match the code. | Fix docstring to: `"Return (z2, z4, z6, cf_centroid) for a single exemplar trajectory, or None if not found."` |
| 14 | **MISSING**  | `examples/plot_publication_figs.py:108`                 | `find_exemplar_triple` parameters (`Z_all`, `labels_np`, `fate_names`, `test_triples`, `target_fate`, `fallback_fate`) are not documented. The function is non-trivial and domain-specific. | Add an Args block documenting each parameter. |
| 15 | **TYPO**     | `examples/plot_publication_figs.py:215`                 | Title string reads `log det  G(z)` (double space before `G`). | Change to `log det G(z)`. |
| 16 | **MISSING**  | `examples/plot_publication_figs.py:134–145`             | `main()` has no docstring. As the entry point and the longest function in the script, it should at minimum describe the pipeline: load data → load model → encode → build 3-panel figure → save. | Add a brief docstring: `"Load Weinreb data and trained VAE, encode cells, and generate the 3-panel publication figure."` |
| 17 | **UNCLEAR**  | `examples/plot_publication_figs.py:87–89`               | `compute_wind_grid` accesses `vae.metric.w_net(pts)` without any comment. A reader unfamiliar with the codebase does not know what `w_net` is. | Add inline comment: `# w_net: learned wind-field network W(z) from the Randers metric`. |

---

## Coverage Matrix

| Public Symbol             | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------------------|:------------:|:---------------:|:------------------:|:-------------:|:-------:|
| `compute_logdet_grid`     | ✅           | ❌ (`vae` missing) | ✅                 | Partial ($J^T J$ but no $G$) | ❌      |
| `compute_wind_grid`       | ✅           | ❌ (`vae` missing) | ✅                 | ❌ (no Zermelo ref) | ❌      |
| `arc_length_segment`      | ✅           | ✅               | ✅ (implicit)      | ❌ (confuses length/energy) | ❌      |
| `find_exemplar_triple`    | ✅           | ❌               | ⚠️ (names mismatch) | ❌            | ❌      |
| `main`                    | ❌           | N/A             | N/A                | N/A           | N/A     |

---

## Spec Alignment Notes

1. **Pullback metric notation** — The module docstring and Panel A title use $G(z) = J^T J$, which is the Riemannian part of the pullback metric. `spec/MATH_SPEC.md § 5` defines the full Randers metric $F(x,v)$ via the Zermelo parameterization. The figure correctly separates the Riemannian component (Panel A) from the wind (Panel B), but nowhere in the script is this decomposition explained.

2. **Energy vs. length** — Panel C labels the path cost as $E$. In `spec/MATH_SPEC.md § 3`, $E = \frac{1}{2} F^2$ is the energy, while `arc_length` in `src/ham/geometry/metric.py:69` computes $\int F\,dt$ (length). The label is inconsistent with the spec's convention. See Issue #9.

3. **Experiment mapping** — `spec/ARCH_SPEC.md § 5, item 5` references experiments H1–H4. Panel A corresponds to H1 (geometric topology), Panel B to H2 (directional asymmetry), Panel C to H3 (discriminative cost). This mapping is not stated anywhere in the script.

4. **`find_exemplar_triple` is unused** — The function is defined (line 108) but `main()` uses its own inline logic (lines 273–310) to find an exemplar triple. The dead function's docstring is therefore misleading — it claims to return `z6_counter` but `main()` computes counterfactual targets differently. Consider removing the unused function or refactoring `main()` to call it.
