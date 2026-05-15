# Math Review: plot_weinreb_cell_types

**Reviewer:** Math Reviewer Agent
**Date:** 2026-05-15
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

This script is a visualization utility that plots Weinreb hematopoiesis data on SPRING 2D coordinates, colored by cell type, with overlaid RNA pseudo-velocity arrows and lineage trajectories. The only non-trivial mathematical operation is the kNN-based projection of PCA-space pseudo-velocities to 2D SPRING coordinates (lines 51–100). This projection contains **two known bugs** that were already fixed in the sibling script [examples/plot_weinreb_destinations.py](examples/plot_weinreb_destinations.py) (see its "BUG 1 FIX" comment at line 127) but were never back-ported here. No differential-geometric formulas from `spec/MATH_SPEC.md` are used directly. **Verdict: Minor Issues.**

## Formula-by-Formula Audit

### 1. kNN Velocity Projection — Directional Filtering

- **Spec Reference:** Not in `spec/MATH_SPEC.md`; this is a standard RNA-velocity embedding projection (La Manno et al., *Nature* 2018, doi:10.1038/s41586-018-0414-6; Bergen et al., *Nature Biotechnology* 2020, doi:10.1038/s41587-020-0591-3).
- **Literature Reference:** The standard transition-probability formulation computes the embedded velocity as $\tilde{v}_i = \sum_j \pi_{ij}\,(\tilde{x}_j - \tilde{x}_i)$ where $\pi_{ij}$ incorporates **signed** cosine correlations between the velocity direction and neighbor displacements. Neighbors with negative correlation (behind the cell) receive small but non-zero weight, acting as a soft directional filter.
- **Implementation:** [examples/plot_weinreb_cell_types.py](examples/plot_weinreb_cell_types.py#L89)
  ```python
  valid = proj > 0
  if np.sum(valid) > 0:
      weights = np.exp(-dists[k]**2 / (np.mean(dists[k]**2) + 1e-8))
      v_spr_dir = np.average(disp_spr[valid], weights=weights[valid]*proj[valid], axis=0)
  ```
  Hard-thresholding at `proj > 0` discards all backward neighbors, biasing the estimator. In low-density or boundary regions where few neighbors lie ahead of the velocity direction, this can produce severely distorted arrows or zero-vectors (the `else` branch defaults to zero).
- **Verdict:** WARNING
- **Notes:** The sibling script [examples/plot_weinreb_destinations.py](examples/plot_weinreb_destinations.py#L127) already fixes this with the comment `# BUG 1 FIX: use signed weights, do not discard proj <= 0`, using all neighbors with signed weights `combined_weights = weights * proj`. The same fix should be applied here.
- **Recommended Action:** Remove the `valid = proj > 0` filter. Use signed weights over all neighbors:
  ```python
  combined_weights = weights * proj
  v_spr_dir = np.sum(disp_spr * combined_weights[:, None], axis=0)
  ```

### 2. kNN Velocity Projection — Local vs. Global Bandwidth

- **Spec Reference:** Not in `spec/MATH_SPEC.md`.
- **Literature Reference:** Gaussian kernel density estimation with a global bandwidth is the standard practice for velocity embedding to avoid density-dependent distortion (see scVelo documentation, Bergen et al. 2020).
- **Implementation:** [examples/plot_weinreb_cell_types.py](examples/plot_weinreb_cell_types.py#L92)
  ```python
  weights = np.exp(-dists[k]**2 / (np.mean(dists[k]**2) + 1e-8))
  ```
  The bandwidth $\sigma^2_k = \mathrm{mean}(d_{k,j}^2)$ is recomputed **per cell**. In dense regions $\sigma^2$ is small, concentrating weight on the very nearest neighbors; in sparse regions $\sigma^2$ is large, spreading weight broadly. This creates a systematic density-dependent distortion of the velocity field.
- **Verdict:** WARNING
- **Notes:** The sibling script uses a **global** bandwidth: `global_sigma2 = np.median(global_dists**2)` (line 62), computed once from a random sample. This ensures all cells are smoothed at a consistent spatial scale.
- **Recommended Action:** Pre-compute a global bandwidth as in `plot_weinreb_destinations.py`:
  ```python
  sample_idx = np.random.choice(len(x_pca), min(10000, len(x_pca)), replace=False)
  global_dists, _ = nn.kneighbors(x_pca[sample_idx])
  global_sigma2 = np.median(global_dists**2)
  # then in the loop:
  weights = np.exp(-dists[k]**2 / (global_sigma2 + 1e-8))
  ```

### 3. Velocity Magnitude Normalization

- **Spec Reference:** N/A (visualization only).
- **Implementation:** [examples/plot_weinreb_cell_types.py](examples/plot_weinreb_cell_types.py#L97)
  ```python
  v_spr[k] = v_spr_dir / (np.linalg.norm(v_spr_dir) + 1e-8) * 1.5
  ```
  All arrows are normalized to length 1.5, discarding magnitude information from the PCA velocity.
- **Verdict:** NOTE
- **Notes:** For a qualitative visualization this is acceptable. However, the PCA velocity norm $\|v_i\|$ carries biologically meaningful rate-of-change information. If magnitude fidelity is ever needed, the scaling should incorporate $\|v_i^{PCA}\|$.

### 4. Gaussian Kernel — Missing Factor of 2

- **Spec Reference:** Standard Gaussian kernel: $K(d) = \exp\!\bigl(-d^2 / (2\sigma^2)\bigr)$.
- **Implementation:** [examples/plot_weinreb_cell_types.py](examples/plot_weinreb_cell_types.py#L92)
  ```python
  weights = np.exp(-dists[k]**2 / (np.mean(dists[k]**2) + 1e-8))
  ```
  This is $K(d) = \exp(-d^2 / h)$ with $h = \overline{d^2}$, which is equivalent to setting $\sigma^2 = h/2$. The missing factor of 2 is absorbed into the bandwidth definition.
- **Verdict:** NOTE
- **Notes:** Not a bug — this is a common simplification. The bandwidth is effectively halved compared to the canonical parameterisation. The sibling script uses the same convention. No action required.

### 5. Lineage Trajectory Plotting

- **Implementation:** [examples/plot_weinreb_cell_types.py](examples/plot_weinreb_cell_types.py#L108-L135)
  ```python
  i2 = triples_subset[:, 0]  # Day 2 index
  i4 = triples_subset[:, 1]  # Day 4 index
  i6 = triples_subset[:, 2]  # Day 6 index
  ```
  Lineage triples are loaded from `weinreb_lineage_triples.npy` and plotted as piecewise-linear paths Day 2 → Day 4 → Day 6 in SPRING coordinates. The indexing is straightforward array look-up with no mathematical transformation.
- **Verdict:** CORRECT
- **Notes:** No mathematical issue. The index-to-coordinate mapping (`x[i2]`, `y[i2]`) is a simple array dereference.

## Open Questions

1. **Was the `proj > 0` filtering intentional in this script?** The sibling script explicitly marks it as a bug and fixes it. If the cell_types script pre-dates the fix, it should be updated. If the filtering was intentional for aesthetic reasons (e.g., to avoid backward-pointing arrows near stem cells), this should be documented.

2. **Should `np.average` (normalized) or `np.sum` (unnormalized) be used for the weighted displacement?** The cell_types script uses `np.average` (line 93) while destinations uses `np.sum` (line 123). Since the result is normalized to unit length immediately after, the choice is immaterial — but the code should be consistent across sibling scripts for maintainability.
