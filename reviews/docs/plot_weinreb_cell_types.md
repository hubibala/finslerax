# Documentation Review: `examples/plot_weinreb_cell_types.py`

**Reviewer:** Doc Reviewer Agent
**Date:** 2026-05-15

## Summary

Overall documentation quality: **needs work**.

The script contains useful inline comments in the velocity-projection section but is missing a module-level docstring, a function docstring, biological context for the Weinreb dataset, and documentation of expected inputs/outputs. Several magic numbers are unexplained. A reader unfamiliar with the Weinreb 2020 hematopoiesis dataset would struggle to understand what this script produces or why.

## Issue Tracker

| # | Severity | Location | Issue | Suggested Text |
|---|----------|----------|-------|----------------|
| 1 | **MISSING** | `examples/plot_weinreb_cell_types.py:1` | No module-level docstring. The script's purpose, required input files, expected output, and biological context are entirely undocumented. | **Recommended Action:** Add a module-level docstring, e.g.: `"""Plot Weinreb hematopoiesis cell types on SPRING 2-D coordinates.\n\nVisualises the Weinreb et al. (2020) dataset: each cell is coloured by\nits annotated haematopoietic cell type, overlaid with RNA pseudo-velocity\narrows and clonal lineage trajectories (Day 2 → Day 4 → Day 6).\n\nInputs\n------\ndata/weinreb_preprocessed.h5ad (or data/weinreb_raw.h5ad)\n    AnnData object with obs columns 'SPRING-x', 'SPRING-y',\n    'Cell type annotation' and obsm keys 'velocity_pca', 'X_pca'.\ndata/weinreb_lineage_triples.npy (optional)\n    (N, 3) int array of (day2_idx, day4_idx, day6_idx) triples.\n\nOutput\n------\nweinreb_cell_types_2d.png (200 dpi)\n"""` |
| 2 | **MISSING** | `examples/plot_weinreb_cell_types.py:6` | `main()` has no docstring. | **Recommended Action:** Add a one-line docstring, e.g.: `"""Generate and save the Weinreb cell-type visualisation."""` |
| 3 | **UNCLEAR** | `examples/plot_weinreb_cell_types.py:14-16` | The SPRING coordinate columns (`SPRING-x`, `SPRING-y`) are used without any explanation of what SPRING embedding is. An ML engineer unfamiliar with single-cell biology would not know these are force-directed graph-layout coordinates. | **Recommended Action:** Add a comment: `# SPRING coordinates: 2-D force-directed layout of the kNN graph (Weinreb et al., 2018).` |
| 4 | **UNCLEAR** | `examples/plot_weinreb_cell_types.py:18-19` | `'Cell type annotation'` column is used but the annotation source and expected categories are not documented. | **Recommended Action:** Add a comment noting these are expert-curated lineage annotations from the original Weinreb 2020 paper. |
| 5 | **MISSING** | `examples/plot_weinreb_cell_types.py:8-10` | The fallback from `weinreb_preprocessed.h5ad` to `weinreb_raw.h5ad` is not explained. The reader cannot tell whether the two files have the same schema or whether the raw file will actually contain the required obsm keys (`velocity_pca`, `X_pca`). | **Recommended Action:** Add a comment explaining the fallback, e.g.: `# Prefer preprocessed file (contains velocity_pca, X_pca); fall back to raw if preprocessing has not been run.` |
| 6 | **UNCLEAR** | `examples/plot_weinreb_cell_types.py:53-64` | The velocity-projection algorithm is described at a high level ("basic projection", "local finite-difference projection using kNN") but the mathematical intent is not stated. A differential geometer would want to know this approximates the Jacobian of the SPRING embedding applied to the PCA velocity. | **Recommended Action:** Add a one-line mathematical note: `# Approximates v_SPRING ≈ J · v_PCA via a kNN finite-difference estimator of the Jacobian.` |
| 7 | **UNCLEAR** | `examples/plot_weinreb_cell_types.py:67` | `n_neighbors=50` is a magic number with no justification. | **Recommended Action:** Extract to a named constant or add a comment: `# 50 neighbors balances smoothness vs. locality for velocity projection.` |
| 8 | **UNCLEAR** | `examples/plot_weinreb_cell_types.py:62` | `n_vel_plot = min(1500, len(vel_idx))` — the choice of 1500 arrows is unexplained. | **Recommended Action:** Add a comment: `# Cap at 1500 arrows to keep the figure readable.` |
| 9 | **UNCLEAR** | `examples/plot_weinreb_cell_types.py:84` | The constant `1.5` used to scale the velocity arrows (`* 1.5`) is a magic number. Its units and rationale are unclear. | **Recommended Action:** Extract to a named constant (e.g., `ARROW_SCALE = 1.5`) or explain in a comment. |
| 10 | **UNCLEAR** | `examples/plot_weinreb_cell_types.py:87-88` | `scale=50` in `ax.quiver(...)` is another magic number that interacts with the `1.5` scaling above. The net visual effect is undocumented. | **Recommended Action:** Document the relationship: `# quiver scale=50 with per-arrow norm ≈ 1.5 gives arrows ~3 % of axis span.` |
| 11 | **UNCLEAR** | `examples/plot_weinreb_cell_types.py:93-94` | `n_plot = min(150, len(triples))` — the choice of 150 lineage triples is unexplained. | **Recommended Action:** Add a comment: `# Subsample to 150 triples to avoid over-plotting.` |
| 12 | **MISSING** | `examples/plot_weinreb_cell_types.py:91-92` | No documentation of the expected shape or semantics of `weinreb_lineage_triples.npy`. A reader must reverse-engineer that each row is `(day2_idx, day4_idx, day6_idx)`. | **Recommended Action:** Add a comment: `# Each row is (day2_cell_idx, day4_cell_idx, day6_cell_idx) indexing into adata.obs.` |
| 13 | **UNCLEAR** | `examples/plot_weinreb_cell_types.py:66` | `from sklearn.neighbors import NearestNeighbors` is imported inside the function body without explanation. This hides a dependency that is not listed in the module-level imports. | **Recommended Action:** Move to module-level imports, or add a comment: `# Deferred import: sklearn is only needed for velocity projection.` |
| 14 | **MISSING** | `examples/plot_weinreb_cell_types.py` (global) | No reference to the original publication. The Weinreb dataset should cite Weinreb et al. (2020) *Science* 367(6479) or the corresponding DOI. | **Recommended Action:** Add a citation in the module docstring: `Reference: Weinreb et al. (2020), doi:10.1126/science.aaw3381` |
| 15 | **MISSING** | `examples/plot_weinreb_cell_types.py` (global) | No connection to `spec/ARCH_SPEC.md § 5` ("Bio Application (Weinreb)") or to other example scripts in the Weinreb pipeline (`preprocess_weinreb.py`, `weinreb_experiment.py`). A reader doesn't know where this script fits in the overall workflow. | **Recommended Action:** Add a "See Also" note in the docstring pointing to `preprocess_weinreb.py` (data preparation) and `weinreb_experiment.py` (model training). |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|--------------|-----------------|-------------------|---------------|---------|
| (module) | No | N/A | N/A | N/A | N/A |
| `main()` | No | N/A | N/A | N/A | N/A |

## Spec Alignment Notes

- `spec/ARCH_SPEC.md` line 198 describes the Weinreb application ("GeometricVAE + DataDrivenPullbackRanders trained on Weinreb hematopoiesis data. Experiments H1-H4 validate geometric topology, directional asymmetry, discriminative cost, and forward predictive simulation."). This plotting script is not mentioned in the spec, and the script itself makes no reference back to the spec or the experiment pipeline it supports.
- `spec/ARCH_SPEC.md` line 162 references `data.py` for "BioDataset (AnnData integration, lineage triples)". The plotting script duplicates lineage-triple loading logic (lines 91-95) rather than using the `BioDataset` API; the documentation should at minimum note this is a standalone convenience script not tied to the `BioDataset` abstraction.
- `spec/RESEARCH_LOG.md` line 68 lists `examples/experiment_h{1,2,3,4}_*.py` as the Weinreb experiment scripts but does not mention `plot_weinreb_cell_types.py`. The script's role in the example suite is therefore ambiguous to any reader who starts from the spec.
