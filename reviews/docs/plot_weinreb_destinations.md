# Documentation Review: `examples/plot_weinreb_destinations.py`

**Reviewer:** Doc Reviewer Agent
**Date:** 2025-05-15

## Summary

Overall documentation quality: **needs work**.

The script is a standalone visualization tool that plots lineage-traced hematopoietic trajectories destined for three target cell types (Monocyte, Neutrophil, Erythroid) on SPRING 2D coordinates, with overlaid RNA pseudo-velocity arrows. It performs non-trivial kNN-based velocity projection from PCA space to SPRING coordinates. Despite being a biologically important example, the file lacks a module-level docstring entirely and has only sparse inline comments. A reader unfamiliar with the Weinreb 2020 dataset or the SPRING coordinate system will struggle to understand the script's purpose, data requirements, and algorithmic choices.

## Issue Tracker

| # | Severity | Location | Issue | Recommended Action |
|---|----------|----------|-------|--------------------|
| 1 | **MISSING** | `examples/plot_weinreb_destinations.py:1` | No module-level docstring. The script has no description of its purpose, expected inputs, outputs, or the biological context it visualises. Compare with the sibling `examples/preprocess_weinreb.py` which has a thorough module docstring. | Add a module-level docstring explaining: (a) the script visualises Weinreb hematopoiesis lineage-tracing data (Weinreb et al., 2020), (b) it plots trajectories destined for Monocyte / Neutrophil / Erythroid on SPRING 2D coordinates, (c) it requires `data/weinreb_preprocessed.h5ad` and `data/weinreb_train_triples.npy` as inputs, (d) it outputs `weinreb_destinations_trajectories.png`. |
| 2 | **MISSING** | `examples/plot_weinreb_destinations.py:7` | The `main()` function has no docstring. | Add a brief docstring summarising the function's workflow: load data → filter triples by Day 6 annotation → plot trajectories → overlay velocity arrows → save figure. |
| 3 | **UNCLEAR** | `examples/plot_weinreb_destinations.py:15-18` | The column names `'SPRING-x'`, `'SPRING-y'` are used without explanation. Readers unfamiliar with SPRING (Weinreb et al., 2018) will not know what coordinate system this is or why it is used. | Add a comment explaining that SPRING is a force-directed 2D embedding of single-cell gene expression, and cite Weinreb et al. (2018) _SPRING: a kinetic interface for visualizing high dimensional single-cell expression data_. |
| 4 | **UNCLEAR** | `examples/plot_weinreb_destinations.py:35-36` | `triples_path = 'data/weinreb_train_triples.npy'` is loaded without explaining the triple format. The reader cannot tell what the three columns represent (Day 2, Day 4, Day 6 cell indices) until lines 87–89, 40 lines later. | Add a comment at or near line 35 explaining the triple structure: each row is `(day2_idx, day4_idx, day6_idx)`, representing a clonally-related trajectory across three timepoints. |
| 5 | **UNCLEAR** | `examples/plot_weinreb_destinations.py:49-50` | `v_pca = adata.obsm['velocity_pca'][:]` and `x_pca = adata.obsm['X_pca']` are loaded without documenting what these fields contain or how they were produced. | Add a comment noting that `velocity_pca` is the clone-derived pseudo-velocity in PCA space (produced by `preprocess_weinreb.py`) and `X_pca` is the PCA embedding of gene expression. |
| 6 | **UNCLEAR** | `examples/plot_weinreb_destinations.py:55-60` | The global bandwidth computation (`global_sigma2`) is not explained. While there is a `print` statement, there is no comment or docstring explaining *why* a global bandwidth is preferred over local bandwidths. The inline comment `# Compute global bandwidth to avoid local density distortions` is present but too terse for the target audience. | Expand the comment to explain: a single global bandwidth avoids velocity arrows being dominated by local density variation; the median squared kNN distance is used as a robust scale estimator. |
| 7 | **UNCLEAR** | `examples/plot_weinreb_destinations.py:100-130` | The kNN velocity projection block is the most mathematically involved section of the script but has minimal documentation. The algorithm — project PCA-space velocity to SPRING coordinates via a weighted combination of neighbour displacements — is not described anywhere as a coherent procedure. | Add a block comment above line 100 describing the projection algorithm in plain language: for each cell, find kNN in PCA space, compute directional similarity between each neighbour displacement and the velocity vector, weight by a Gaussian kernel, and sum the corresponding SPRING-space displacements. |
| 8 | **MISSING** | `examples/plot_weinreb_destinations.py:42-43` | The list `targets = ['Monocyte', 'Neutrophil', 'Erythroid']` has no explanation of why these three cell types were chosen or what biological significance they have. | Add a brief comment: these are the three major myeloid/erythroid terminal fates in murine hematopoiesis that are well-represented in the Weinreb clonal-tracing dataset (see `spec/ARCH_SPEC.md § 6`). |
| 9 | **MISSING** | `examples/plot_weinreb_destinations.py` (file-level) | No usage instructions. Unlike `preprocess_weinreb.py`, the script does not document how to run it (`python examples/plot_weinreb_destinations.py`) or list prerequisites (the preprocessed data must exist). | Add usage instructions to the module docstring, including the prerequisite step: `python examples/preprocess_weinreb.py` must be run first. |
| 10 | **TYPO** | `examples/plot_weinreb_destinations.py:127` | The comment `# BUG 1 FIX: use signed weights, do not discard proj <= 0` references a bug fix but does not describe *what* BUG 1 was. A reader seeing this for the first time has no context. | Rewrite the comment: `# Use signed projection weights (not clipped to positive), so that neighbours in the opposite direction contribute negative weight, yielding a more accurate velocity estimate.` Optionally reference the sibling script `plot_weinreb_cell_types.py` where the unfixed version exists. |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:------------:|:---------------:|:------------------:|:-------------:|:-------:|
| (module)      | No           | N/A             | N/A                | N/A           | N/A     |
| `main()`      | No           | N/A             | N/A                | N/A           | N/A     |

## Spec Alignment Notes

- `spec/ARCH_SPEC.md § 6` identifies the Weinreb hematopoiesis application as a key validated use-case. This script is the primary visualisation of destination-stratified trajectories but is not mentioned in the spec's module listing (`§ 5`). The `examples/` directory is not catalogued in the spec at all.
- The kNN velocity-projection algorithm used in this script (lines 100–130) is not documented in `spec/MATH_SPEC.md`. It is a visualisation heuristic rather than a core geometric computation, but its relationship to the Finsler/Randers velocity framework should be clarified in a comment (i.e., this is an approximate projection for display purposes, not the principled Finsler exponential map).
