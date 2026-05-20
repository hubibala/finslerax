# Math Review: `ham.bio.data`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

The module contains **no differential-geometry formulas**; its mathematical content is limited to standard single-cell preprocessing (library-size normalisation, log-transform, PCA) and combinatorial lineage-pair extraction. No formula contradicts the spec. However, two issues materially affect the geometric pipeline downstream: **(1)** the velocity preprocessing pipeline never projects RNA velocity from gene space to PCA space, silently falling back to a zero velocity field; **(2)** the normalisation/log-transform gate is conditioned on gene count rather than on whether the data is already processed, risking PCA on raw counts.

**Verdict: Minor Issues**

---

## Formula-by-Formula Audit

### 1. Library-Size Normalisation (`preprocess`, line 37)

- **Spec Reference:** Not in `MATH_SPEC.md` (preprocessing, not geometry).
- **Literature Reference:** Scanpy default; equivalent to CPM ($\times 10^4$): $\tilde{x}_{ij} = x_{ij} \cdot \frac{T}{\sum_j x_{ij}}$, $T = 10^4$.
- **Implementation:**
  ```python
  sc.pp.normalize_total(self.adata, target_sum=1e4)  # line 37
  ```
- **Verdict:** OK
- **Notes:** Standard counts-per-10k normalisation. Mathematically correct.

---

### 2. Log Transform (`preprocess`, line 38)

- **Spec Reference:** Not in `MATH_SPEC.md`.
- **Literature Reference:** Standard $\log(1+x)$ variance-stabilising transform (Lun et al., 2016).
- **Implementation:**
  ```python
  sc.pp.log1p(self.adata)  # line 38
  ```
- **Verdict:** OK
- **Notes:** Correct application of $x \mapsto \log(1 + x)$.

---

### 3. Conditional Gate on Normalisation (`preprocess`, lines 36–41)

- **Spec Reference:** N/A.
- **Implementation:**
  ```python
  if self.adata.n_vars > n_top_genes:          # line 36
      sc.pp.normalize_total(...)                # line 37
      sc.pp.log1p(...)                          # line 38
      sc.pp.highly_variable_genes(...)          # line 39
      self.adata = self.adata[:, ...]           # line 40
  ```
- **Verdict:** WARNING
- **Notes:** The entire normalisation–HVG pipeline is skipped when `n_vars ≤ n_top_genes`. If a dataset has fewer than 2000 genes but still contains **raw integer counts**, PCA will be applied directly to raw counts without library-size normalisation or log-transform. This produces a PCA embedding dominated by sequencing-depth variation rather than biological signal, yielding a coordinate space in which the learned Finsler metric $g_{ij}(x,v)$ would be geometrically meaningless.

  **Recommended Action:** Decouple the normalisation/log-transform from the HVG-selection guard. Always normalise and log-transform; only skip HVG selection when `n_vars ≤ n_top_genes`.

---

### 4. PCA (`preprocess`, lines 44–45)

- **Spec Reference:** Not in `MATH_SPEC.md`.
- **Literature Reference:** Standard truncated SVD; Scanpy's `sc.tl.pca` centres the data ($\bar{X} = 0$) but does **not** standardise per-gene variance (unlike the Weinreb preprocessing in [preprocess_weinreb.py](../../examples/preprocess_weinreb.py) which calls `sc.pp.scale()` before PCA).
- **Implementation:**
  ```python
  sc.tl.pca(self.adata, n_comps=pca_components)  # line 45
  ```
- **Verdict:** WARNING
- **Notes:** Without `sc.pp.scale()`, the PCA components carry decreasing variance ($\sigma_1^2 \gg \sigma_{50}^2$). When these coordinates are fed to the Finsler energy $E(x,v)$, Euclidean distances are dominated by the first few principal components. The Weinreb preprocessing ([preprocess_weinreb.py:118](../../examples/preprocess_weinreb.py#L118)) explicitly normalises PCA columns to unit variance:

  $$X_{\text{pca}}^{(k)} \;\leftarrow\; \frac{X_{\text{pca}}^{(k)}}{\sigma_k + \varepsilon}$$

  This creates a coordinate system where the Euclidean prior is isotropic — a more natural starting point for metric learning. `data.py` omits this step, producing an anisotropic ambient space that forces the learned metric to absorb the PCA variance spectrum, complicating convergence.

  **Recommended Action:** Apply per-component variance normalisation to `X_pca` after PCA, consistent with the Weinreb pipeline.

---

### 5. RNA Velocity Projection to PCA Space (`preprocess` + `get_jax_data`, lines 48–53, 167–170)

- **Spec Reference:** `MATH_SPEC.md § 5` — The Zermelo formulation requires a wind vector $W^i(x)$ that lives in the **same tangent space** as the position coordinates.
- **Literature Reference:** Bergen et al. (2020, Nature Biotechnology) — scVelo projects velocity to PCA space via $V_{\text{pca}} = V_{\text{gene}} \cdot W_{\text{PCA}}$, where $W_{\text{PCA}} \in \mathbb{R}^{G \times d}$ is the PCA loadings matrix. This is the standard first-order approximation:

  $$v_{\text{pca}}^i = \sum_{g=1}^{G} v_g \cdot w_{gi}$$

- **Implementation:**
  ```python
  # preprocess (line 48-53): computes velocity in gene space
  scv.pp.moments(self.adata)
  scv.tl.velocity(self.adata, mode='stochastic')

  # get_jax_data (lines 167-170): only loads if already projected
  if 'velocity' in self.adata.layers:
       if use_pca and 'velocity_pca' in self.adata.obsm:
           V_np = self.adata.obsm['velocity_pca']
  ```
- **Verdict:** CRITICAL
- **Notes:** The `preprocess()` method calls `scv.tl.velocity()`, which stores velocity in `adata.layers['velocity']` (gene space, $\mathbb{R}^G$). It does **not** call `scv.tl.velocity_embedding()` or manually project via the PCA loadings. Consequently, `velocity_pca` is never created. In `get_jax_data()`, the check `'velocity_pca' in self.adata.obsm` fails, `V_np` remains `None`, and velocity is silently replaced by:

  ```python
  V=jnp.array(V_np) if V_np is not None else jnp.zeros_like(X_np)  # line 182
  ```

  The downstream Randers/Zermelo pipeline receives $W^i = 0$ (no wind), which means **the directional (asymmetric) component of the Finsler metric is never informed by RNA velocity** when data is loaded through this path. This defeats the purpose of the Randers parameterisation (spec § 5).

  Furthermore, even when `use_pca=False`, velocity is never loaded because the inner condition requires `use_pca=True` — so gene-space velocity is unreachable from `get_jax_data()`.

  **Recommended Action:** After computing velocity, project it to PCA space:
  ```python
  pca_loadings = self.adata.varm['PCs']  # (n_genes, n_comps)
  V_gene = self.adata.layers['velocity']
  self.adata.obsm['velocity_pca'] = V_gene @ pca_loadings
  ```
  Also handle the `use_pca=False` branch by loading `adata.layers['velocity']` directly.

---

### 6. Pseudotime-Based Pair Extraction (`_extract_pseudotime_pairs`, lines 121–158)

- **Spec Reference:** Not in `MATH_SPEC.md` (data heuristic).
- **Literature Reference:** Standard k-NN + pseudotime ordering heuristic for constructing flow pairs.
- **Implementation:**
  ```python
  nbrs = NearestNeighbors(n_neighbors=15).fit(X)          # line 126
  # ...
  if their_t > my_t + 0.02:                                # line 140
      pairs.append([i, neighbor_idx])
  ```
- **Verdict:** OK
- **Notes:** The forward-flow threshold $\Delta t > 0.02$ is a heuristic guard against numerical ties in pseudotime. It is not a formula derived from the spec. The k-NN search uses Euclidean distance in PCA space, which is consistent with the flat ambient metric assumed before metric learning. Mathematically sound for its purpose.

---

### 7. Clone-Based Lineage Pair Extraction (`extract_lineage_pairs`, lines 57–117)

- **Spec Reference:** Not in `MATH_SPEC.md`.
- **Implementation:** Pairs every cell at time $t_i$ with every cell at time $t_{i+1}$ within the same clone. For a clone with $n_1$ cells at $t_1$ and $n_2$ at $t_2$, this produces $n_1 \times n_2$ directed pairs.
- **Verdict:** OK
- **Notes:** The combinatorial pairing is correct. It enumerates all parent–child combinations within each clone, ordered by time. This creates the supervision signal for metric learning (spec § 5, Zermelo navigation).

---

### 8. Zero-Velocity Fallback (`get_jax_data`, line 182)

- **Spec Reference:** `MATH_SPEC.md § 5` — Wind field $W^i(x)$.
- **Implementation:**
  ```python
  V=jnp.array(V_np) if V_np is not None else jnp.zeros_like(X_np)
  ```
- **Verdict:** WARNING
- **Notes:** Setting $V = 0$ means the Randers metric degenerates to a pure Riemannian metric ($F = \sqrt{v^T g v}$, $\beta = 0$). While this is mathematically valid (the Randers metric reduces gracefully), it does so **silently**. Any downstream analysis assuming directional asymmetry will produce symmetric results without any warning. This interacts with Finding 5 above — the zero fallback is almost always triggered due to the missing PCA projection.

  **Recommended Action:** Emit an explicit warning when velocity falls back to zero.

---

## Open Questions

1. **Is the lack of `sc.pp.scale()` before PCA intentional?** The Weinreb pipeline applies both scaling and post-PCA variance normalisation. If `data.py` is meant to be a general-purpose loader, it should document whether anisotropic PCA coordinates are expected.

2. **Should `preprocess_weinreb.py`'s per-component normalisation ($X_{\text{pca}}^{(k)} / \sigma_k$) be adopted in the general pipeline?** This affects the meaning of the Euclidean prior in the fundamental tensor $g_{ij}$ — isotropic coordinates yield a more interpretable baseline metric.

3. **The velocity PCA projection gap (Finding 5) — is this code path ever exercised?** If all real-world usage goes through `preprocess_weinreb.py` (which computes `velocity_pca` from clonal structure, bypassing scVelo entirely), then this bug may be latent. However, for any non-Weinreb dataset relying on scVelo, the wind field will be zero.
