# Experiment Plan: Lagrangian Wildfire Propagation on Real Terrain
**Architect:** Research Architect Agent  
**Date:** May 18, 2026  
**Source Document:** [reviews/science/comparative_gahtan2026.md](reviews/science/comparative_gahtan2026.md)  
**Supersedes (in scope):** Phases 4–5 of [reviews/science/experiment_plan_gahtan_lagrangian.md](reviews/science/experiment_plan_gahtan_lagrangian.md)  
**Reference Paper:** Gahtan, Shpund & Bronstein. *Wildfire Simulation with Differentiable Randers-Finsler Eikonal Solvers.* arXiv:2603.00035 [cs.CE], Feb 2026, Sections 4–6.  
**Reference Code:** [BarakGahtan/differentiable-eikonal-wildfire](https://github.com/BarakGahtan/differentiable-eikonal-wildfire) (MIT License)

---

## Literature Brief

### Baseline Paper Summary (Sections 4–6)

Gahtan et al. train a `FlexibleCovariateEncoder` (CNN + MLP) that maps per-pixel environmental covariates (topography, fuel, vegetation) and global weather scalars to per-pixel Randers parameters $(G(x), b(x))$. The forward model solves the Randers eikonal PDE with fast sweeping from an ignition point; the backward pass differentiates through the converged solution implicitly. Training is supervised by fire mask arrival times extracted from Landsat/MODIS-derived imagery.

**Key baseline numbers (per-scene training, Section 6.1):**

| Metric | Gahtan Value | Source |
|--------|-------------|--------|
| Pearson correlation (mean ± std) | 0.824 ± 0.044 | Figure 7 |
| IoU@50 (mean) | 0.609 | Figure 7 |
| Relative RMSE | reported per scene | Table 9 |
| Cross-scene correlation | 0.766 | Table 4 |
| Sim-to-real transfer correlation | 0.588 ± 0.172 | Table 8 |
| Training time per fire (GPU eikonal) | ~10 ms | Section 4.3 |
| Number of scenes | 19 | Section 6 |
| Number of fires | 32,000+ | README |

### Why the Neural Approach Works Here (Unlike Phase 1)

Phase 1 (synthetic isotropic recovery) asked a neural net to learn `G: R^2 → R` from coordinates alone, where `G` has a sharp piecewise-constant boundary. This is the pathological case for spectral bias — MLPs initialized with small weights prefer smooth, low-frequency functions.

This experiment uses a fundamentally different encoding: the input to the neural metric is not a spatial coordinate but a vector of physical covariates:

$$\mathbf{G}(x), \mathbf{b}(x) = f_\theta(\underbrace{\text{elev}(x), \text{slope}(x), \text{aspect}(x), \text{fuel}(x), \text{canopy}(x)}_{\mathbf{Y}(x) \in \mathbb{R}^5}, \underbrace{T_\text{air}, q, u, v}_{\mathbf{y} \in \mathbb{R}^4})$$

The function $f_\theta$ maps a 9-dimensional physics vector (the same at every point with the same environmental conditions) to 5 Randers parameters. The target function is smooth in covariate space: slope angle smoothly modulates fire speed, fuel type is a discrete but small categorical variable. There is no spectral bias problem because:
1. The network does not see bare coordinates — it sees domain-meaningful covariates.
2. The target $(\mathbf{G}, \mathbf{b})$ is a smooth function of the covariates.
3. The discrete categorical fuel variable (13 FBFM classes) is handled via an embedding layer, not directly as a real number.

This is confirmed by Gahtan's architecture: their `FlexibleCovariateEncoder` is a standard ResNet-style CNN + MLP with no Fourier features and achieves 0.82 correlation, demonstrating that neural nets work well for this covariate-to-metric mapping.

### Dataset

**Sim2Real-Fire** (Li et al., NeurIPS 2024): Publicly available via Google Drive. Structure per scene:
```
<scene_id>/
├── Topography_Map/   {Elevation.tif, Slope.tif, Aspect.tif}
├── Fuel_Map/         {FBFM13.tif, Canopy_Cover.tif, Canopy_Height.tif, ...}
├── Vegetation_Map/   {Existing_Vegetation_Type.tif, ...}
├── Satellite_Images_Mask/<event_id>/  {out1.jpg ... out72.jpg}
└── Weather_Data/     {<event_id>.txt}
```
- **19 scenes**, **32,000+ fires**, **hourly weather** (temperature, relative humidity, wind speed, wind direction)
- Fire masks: binary per-frame (72 frames ≈ 72 hours per fire)
- All raster data: 30m resolution, projected to UTM

**Download:** `python experiments/wildfire/download_data.py --output_dir <path>`  
**License:** Apache-2.0

**Access status:** Available via Google Drive (20 packages of ~500 GB total). For initial development, one package (package 1) covering scenes 1–19 is sufficient.

---

## Scope & Success Criteria

This plan validates HAMTools' **covariate-conditioned Lagrangian Randers approach** for wildfire propagation prediction on the Sim2Real-Fire dataset, then extends it to true 3D terrain geometry where the Lagrangian approach has a unique structural advantage over eikonal methods.

**Done** looks like:
1. **Phase W1 (flat grid):** Per-scene Pearson correlation ≥ 0.70 on ≥ 10 scenes, with honest runtime table. Randers (with wind) outperforms Riemannian (without) by a statistically significant margin.
2. **Phase W2 (terrain mesh):** ≥ 2% correlation improvement over flat-grid baseline on scenes with slope variability > 15°. 3D geodesic visualization on terrain surface. Demonstrates a capability that Gahtan Section 7 explicitly lists as future work.
3. **Geometric analysis:** Fire corridor geodesics and Jacobi field divergence maps produced for ≥ 5 fires.

---

## Phase W1: Covariate-Conditioned Randers on Sim2Real-Fire (Flat Grid)

### Objective
Apply HAMTools' Lagrangian solver to the Sim2Real-Fire wildfire dataset, enabling direct quantitative comparison with Gahtan's eikonal baseline (Section 6.1). Validate that the covariate-conditioned neural Randers metric, trained with AVBD arrival-time supervision, achieves competitive performance.

### Hypothesis
The AVBD solver with covariate-conditioned metric will achieve per-scene correlation ≥ 0.70 (vs. Gahtan's 0.824). The gap reflects:
1. **Solver speed** — AVBD solves ~200 BVPs per fire; fast sweeping solves the full PDE. Fewer observations per gradient step means slower convergence.
2. **Information density** — BVP arc lengths at K sampled points vs. full distance field across all pixels.

However, the Lagrangian approach can match or exceed Gahtan for fires with **complex boundary topology** (non-convex perimeters, ridge-following fire) because AVBD finds explicit paths that respect the terrain, whereas fast sweeping on a grid requires careful stencil design for non-convex obstacles.

### Method

**HAMTools Components:**
- Manifold: `EuclideanSpace(2)` from [src/ham/geometry/manifolds/](src/ham/geometry/manifolds/)
- Metric: New `CovariateConditionedRanders` (mini-spec below)
- Solver: `AVBDSolver` from [src/ham/solvers/avbd.py](src/ham/solvers/avbd.py)
- Loss: `ArrivalTimeLoss` from [src/ham/training/losses.py](src/ham/training/losses.py) (already implemented)

**New Components Needed:**

#### CovariateConditionedRanders

A Randers metric that evaluates G and b from environmental covariates rather than from spatial coordinates. This is the core modeling difference from the Phase 1 `GridMetricField`.

**Mathematical formulation:**

$$f_\theta: \mathbb{R}^{d_\text{spatial} + d_\text{global}} \to \mathbb{R}^5$$

$$(\tilde{g}_{11}, \tilde{g}_{12}, \tilde{g}_{22}, \tilde{b}_1, \tilde{b}_2) = f_\theta(\mathbf{Y}(x), \mathbf{y})$$

where:
- $\mathbf{Y}(x) = [\text{elev}(x), \text{slope}(x), \text{sin(aspect)}(x), \text{cos(aspect)}(x), \text{FBFM\_emb}(x), \text{canopy\_cover}(x)] \in \mathbb{R}^{5 + d_\text{emb}}$ — spatial covariates at position $x$, bilinearly interpolated from raster data
- $\mathbf{y} = [T_\text{air}, q, \sin(\phi_w), \cos(\phi_w)] \in \mathbb{R}^4$ — global weather scalars for the fire event
- $d_\text{emb} = 4$ — FBFM13 fuel code (13 classes) embedded to 4 dimensions

**Projection to feasible $(G, b)$:**

$$G = \text{project\_SPD}\left(\begin{bmatrix} \tilde{g}_{11} & \tilde{g}_{12} \\ \tilde{g}_{12} & \tilde{g}_{22} \end{bmatrix}, \epsilon_G, G_\text{max}\right), \quad b = \text{project\_Bnorm}(\tilde{b}, G, \alpha=0.9)$$

where $\text{project\_SPD}$ clamps eigenvalues to $[\epsilon_G, G_\text{max}]$ and $\text{project\_Bnorm}$ ensures $\|b\|_{G^{-1}} < 0.9$. These are differentiable eigenvalue projections following Gahtan's `project_G_eigenvalue` / `project_B_norm` (in `eikonal/nn/encoder.py`).

**Architecture:**
```
Global pathway: MLP(R^4 → R^64 → R^64 → R^5)   ← fire-wide weather baseline
Local pathway:  MLP(R^(5+d_emb) → R^64 → R^64 → R^5)  ← terrain/fuel residual
Output: G_final = project_SPD(G_baseline + G_residual)
         b_final = project_Bnorm(b_baseline + b_residual, G_final)
```

**Proposed API sketch:**
```python
class CovariateConditionedRanders(FinslerMetric):
    global_mlp: eqx.Module   # (4,) → (5,) — weather baseline
    local_mlp: eqx.Module    # (5+d_emb,) → (5,) — terrain residual
    fuel_embedding: jax.Array  # (13, d_emb) — trainable FBFM embeddings
    eps_G: float = 0.1
    max_G: float = 10.0
    max_b_norm: float = 0.9

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Randers cost F(x, v). Covariates looked up via bilinear interp."""
        G, b = self._get_params(x)
        # Zermelo formula: sqrt(v^T G v + (b·v)^2) - b·v  ... (spec/MATH_SPEC.md § 1.2)
        ...
    
    def bind_scene(self, topo_raster, fuel_raster, veg_raster, weather_vec):
        """Return a metric with scene covariates baked in (no dynamic lookup)."""
        ...
```

Extends `spec/ARCH_SPEC.md` § 3 (Metric Hierarchy) and `spec/MATH_SPEC.md` § 1.2 (Randers metric_fn).

> **Math Reviewer**: The Zermelo formula in metric_fn must match `zoo/randers.py` exactly — use the Zermelo navigation formulation: $F = \frac{\sqrt{\lambda \|v\|_H^2 + \langle W, v \rangle_H^2} - \langle W, v \rangle_H}{\lambda}$ where $\lambda = 1 - \|W\|_H^2$, not a simplified version. Cross-check against `spec/MATH_SPEC.md § 1.2` before implementation.

#### Arrival Time Extraction from Fire Masks

The Sim2Real-Fire masks give, for each fire event, a sequence of binary frames `masks[t] ∈ {0,1}^{H×W}`. The arrival time at pixel $(i,j)$ is:

$$T_{ij} = \min\{t : \text{masks}[t, i, j] = 1\} \times \Delta t$$

where $\Delta t$ is the inter-frame interval (nominally 1 hour). This gives a quantized arrival time field with integer values in $[0, 72]$ hours. Normalise to $[0, 1]$ per fire by dividing by max arrival time.

> **WARNING:** Frames are hourly but fire spread is continuous. The quantization error is $\pm \Delta t / 2 = \pm 30$ min. This is systematic noise in the target, not a bug — Gahtan trains on the same data. Use it but report it as a limitation.

#### Data Pipeline per Training Fire

1. **Load scene:** `Sim2RealFireLoader.load_scenario(scene_id, event_id)` → covariates + masks + weather
2. **Find ignition:** `find_ignition_point(masks)` → $(x_0, y_0)$ = centroid of `masks[0]`
3. **Extract arrival times:** `extract_arrival_times(masks)` → `T_arr ∈ R^{H×W}` (float, hours)
4. **Sample observation points:** Sample $K = \min(500, \lfloor\text{burned\_pixels} \times 0.3\rfloor)$ pixels from the burned region ($T_{ij} < \infty$), stratified by arrival time decile (ensure coverage of early, mid, and late arrivals)
5. **Covariate lookup:** For each sampled pixel $(x_i, y_i)$: bilinearly interpolate terrain + fuel rasters at that pixel; attach global weather vector
6. **Normalize:** Scale arrival times to $[0, 1]$; normalize spatial covariates per-scene using training-set statistics (mean/std from training fires only)
7. **BVP solve + loss:** `ArrivalTimeLoss(metric, source=ignition, x_obs=sampled_pixels, t_obs=normalized_arrival_times)`

#### Training Pipeline

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| K observations per fire | min(500, burned_pixels × 0.3) | Density-proportional; avoids bias toward large fires |
| Batch size | 16 fires per gradient step | Memory-bounded by AVBD BVP |
| AVBD n_steps | 50 | **F6 fix:** Phase 1 used a 1×1 domain at 20×20 pixels = 0.05m spacing. Sim2Real-Fire scene patches are ~500×500 pixels at 30m = 15,000m. At n_steps=15 the quadrature interval is 1,000m — far too coarse. Use n_steps=50 (300m intervals) for training; run a convergence study at n_steps ∈ {20, 50, 100} on a held-out fire before committing. |
| AVBD iterations | 50 | Same tradeoff |
| Optimizer | Adam, lr=1e-3 | Matches Gahtan (F3 fix; Gahtan uses lr=1e-3 in `ExperimentConfig.learning_rate`) |
| LR schedule | Cosine decay over 100 epochs | Matches Gahtan |
| Early stopping | patience=20, metric=val correlation | Matches Gahtan |
| Global MLP | hidden=128, depth=3 | |
| Local MLP | hidden=128, depth=3 | |
| FBFM embedding dim | 4 | Small: 13 classes |
| TV regularization | $\lambda_G = 0.005$, $\lambda_b = 0.005$ | Matches Gahtan `lambda_G`, `lambda_B` |
| Train/val/test split | 70/15/15 per scene, **random split seed=42** (F8 fix). Primary protocol matches Gahtan. Run one additional temporal-split experiment (train on earliest 70% of fires by timestamp, test on latest 15%) to quantify temporal autocorrelation. Report both. |
| Seeds | 3 per scene (0, 1, 2) | Gahtan uses 1 run; 3 seeds gives error bars but is underpowered (F4). Report mean ± std; use paired Wilcoxon signed-rank test across scenes for Randers vs. Riemannian comparison (n=19 scenes, α=0.05). |

> **CRITICAL implementation note:** `bind_scene` must freeze the raster interpolation into JAX constants at scene load time, so that `metric_fn` does not call Python during JIT. Covariates must be stored as `eqx.field(static=False)` JAX arrays inside `CovariateConditionedRanders`, interpolated at forward-pass time.

### Ablations

| Ablation | Description | Purpose |
|----------|-------------|---------|
| W=0 (Riemannian) | `use_wind=False`, $b \equiv 0$ | Core ablation: does Finsler drift help? |
| K=100 vs. K=500 | Fewer/more observations per fire | How many BVPs are needed? |
| Temporal stratification | Uniform vs. decile-stratified sampling | Sampling strategy effect |
| Global weather baseline | With vs. without global MLP | Does weather improve beyond covariates? |
| FBFM embedding | Learned embedding vs. one-hot | Embedding vs. categorical |
| Randers vs. Riemannian | Full metric vs. $b=0$ | Key HAMTools-specific ablation |
| Per-scene vs. cross-scene | Train on 1 scene vs. train on N, test on held-out | Generalisation |

### Baseline Comparison

| Metric | Gahtan (per-scene) | Our Target (per-scene) | Paper Source |
|--------|--------------------|-----------------------|--------------|
| Pearson correlation (mean) | 0.824 | ≥ 0.70 | Figure 7, Table 9 |
| Pearson correlation (std) | ± 0.044 | report | Figure 7 |
| IoU@50 | 0.609 | ≥ 0.40 | Figure 7 |
| Cross-scene correlation | 0.766 | ≥ 0.65 | Table 4 |
| Riemannian vs. Randers gap | ~5–10% (estimated) | Randers ≥ Riemannian | Section 5 (synthetic) |
| Runtime per fire (training) | ~10 ms | ~500 ms – 5 s | Section 4.3 |

> **NOTE:** The runtime gap is expected and must be reported honestly. It is not a failure — it reflects the fundamental trade: AVBD gives explicit paths, fast sweeping gives full distance fields. The gap is 50–500× in wall-clock time but O(K) vs. O(N²) in solved equations.

> **CRITICAL — Evaluation protocol (F1):** Gahtan computes Pearson r by running their solver on **all burned pixels** in the test set (full distance field). AVBD cannot do this implicitly. We must run a **dense evaluation pass**: for every fire in the test split, run AVBD from the ignition point to each burned-pixel endpoint (vmapped) and compute the predicted arrival time. This evaluation pass is separate from training (no gradient required; use `eqx.filter_jit` + `jax.vmap`). For a typical 500-pixel burned area, this requires 500 serial or vmapped AVBD solves per fire — feasible at eval time. `K_train` (observations used during training) and `K_eval` (evaluation pixels) are independent parameters. Set `K_eval = all burned pixels` in the test split.

> **CRITICAL — IoU@50 definition (F2):** IoU@50 threshold is `t_threshold = max(GT_arrival_times_for_fire) × 0.5` — half the total fire duration. It is **not** the 50th percentile of the GT arrival time distribution. Predicted perimeter at time 0.5 = `{x : T_pred(x) ≤ t_threshold}`. GT perimeter = `{x : T_GT(x) ≤ t_threshold}`. Match Gahtan's exact implementation in `per_scene.py`.

### Visualizations

| # | Figure Title | Type | X-axis | Y-axis / Content | Baseline Overlay | Output File |
|---|-------------|------|--------|-------------------|-----------------|-------------|
| W1.1 | Per-scene correlation bar chart | Grouped bar | Scene ID | Pearson r | Gahtan value per scene (Table 9) | `figs/phaseW1_correlation_comparison.{pdf,png}` |
| W1.2 | Qualitative fire predictions (3 fires) | 3×3 grid | Spatial x | Spatial y / fire mask | GT mask overlay | `figs/phaseW1_qualitative.{pdf,png}` |
| W1.3 | Learned metric field (best scene) | Indicatrix + wind overlay on terrain | Spatial x | Spatial y | — | `figs/phaseW1_learned_metric.{pdf,png}` |
| W1.4 | Geodesic fire corridors | 2D paths overlaid on terrain | Spatial x | Spatial y | Gahtan arrival time heatmap | `figs/phaseW1_fire_corridors.{pdf,png}` |
| W1.5 | Randers vs. Riemannian (ablation) | Bar chart by scene | Scene ID | Pearson r | — | `figs/phaseW1_randers_ablation.{pdf,png}` |
| W1.6 | Runtime vs. correlation scatter | Scatter | log(ms per step) | Pearson r | Gahtan point | `figs/phaseW1_runtime_tradeoff.{pdf,png}` |

### Reproducibility Spec

| Parameter | Value |
|-----------|-------|
| Random seed | 42 (data splits), 0/1/2 (training seeds) |
| Hardware | 1× NVIDIA A100 80GB or Apple M3 Ultra |
| Expected runtime | ~2–8 h per scene (K=500, 100 epochs) |
| JAX version | ≥ 0.4.30 |
| Python | ≥ 3.11 |
| Dataset version | Sim2Real-Fire v1 (F9 fix: pin with SHA-256 hash of each downloaded archive after download; record in `data/wildfire_checksums.txt`) |
| Raster resolution | 30 m (native) |
| Scenes (initial) | 5 scenes with highest fire count |
| Full evaluation | All 19 scenes (after initial validation) |
| Covariate normalization | Per-scene, computed on **training fires only** (F7 fix). Save normalization stats at train time; apply frozen stats to val/test fires. Includes spatial covariates (elevation, slope, etc.) **and** weather scalars ($T_\text{air}$, $q$, wind). |
| Metric | Pearson r between predicted and GT arrival times at **all burned test-set pixels** (F1 fix: dense evaluation pass). Secondary: IoU@50 using `t_threshold = max(GT_times) × 0.5` (F2 fix). |

### Definition of Done
- [ ] Per-scene mean Pearson correlation ≥ 0.70 on ≥ 10 scenes (3-seed mean)
- [ ] Randers (with wind) correlation ≥ Riemannian (no wind) by ≥ 1% (mean over scenes)
- [ ] Runtime per training step reported (vs. Gahtan)
- [ ] Qualitative fire prediction figures for best, median, and worst scenes
- [ ] Geodesic fire corridor visualization for ≥ 5 fires
- [ ] Science Auditor sign-off

---

## Phase W2: Terrain Mesh Extension (Novel HAMTools Contribution)

### Objective
Operate the fire propagation model on the **actual 3D terrain surface** rather than the flat 2D projection, directly addressing the geometric limitation Gahtan identifies as future work (Section 7: *"Extension to triangulated surfaces would replace the eight-stencil system with triangle-fan neighborhoods"*). This is the clearest HAMTools-exclusive capability for this domain.

### Hypothesis
Modeling fire on the terrain surface captures slope-dependent propagation distances more accurately. On flat terrain, projection distortion is small; on steep terrain (slope > 20°), 2D Euclidean distance underestimates true surface distance by $\approx 1/\cos(\alpha)$ where $\alpha$ is the slope angle (15–30% for 30–45° slopes). This systematic error affects both fire arrival time prediction and wind direction alignment. We predict:
- Comparable or worse performance on flat terrain (slope < 10°): mesh adds noise, no signal
- Measurable improvement (≥ 3% correlation gain) on steep terrain scenes (slope > 20°)

This is a falsifiable hypothesis: if the improvement is not observed, it informs the community that 30m raster projection distortion is negligible for the typical fire scenarios in Sim2Real-Fire.

### Method

**HAMTools Components:**
- Manifold: `TriangularMesh` from [src/ham/geometry/mesh.py](src/ham/geometry/mesh.py)
- Metric: `DiscreteRanders` from [src/ham/geometry/zoo/discrete.py](src/ham/geometry/zoo/discrete.py) (already supports `TriangularMesh`) — extended to accept per-vertex covariate conditioning
- Solver: `AVBDSolver` — already supports `TriangularMesh` via the `Manifold` ABC; AVBD operates in the 3D ambient space with mesh-projected geodesics

**New Components Needed:**

#### DEMToMesh

Convert a raster DEM (H×W pixel array of elevations) to a `TriangularMesh`. Each pixel $(i, j)$ at spacing $\Delta s = 30\text{m}$ and elevation $z_{ij}$ becomes a vertex at $\mathbf{v}_{ij} = (i \cdot \Delta s, j \cdot \Delta s, z_{ij}) \in \mathbb{R}^3$. Each square cell is split into 2 triangles by the diagonal:

$$\text{Faces}_{ij} = [(v_{ij}, v_{i+1,j}, v_{i,j+1}),\; (v_{i+1,j+1}, v_{i,j+1}, v_{i+1,j})]$$

This gives a $2(H-1)(W-1)$-face mesh. Covariate fields (fuel, slope, aspect) are defined at vertices by bilinear interpolation from the raster grid.

> **NOTE:** The slope and aspect of each face can be computed analytically from the face normal: $\mathbf{n} = (v_1 - v_0) \times (v_2 - v_0)$, slope $= \arccos(n_z / \|\mathbf{n}\|)$, aspect $= \text{atan2}(n_y, n_x)$. These are more accurate than raster-derived slope/aspect used in the flat-grid case.

**Proposed API:**
```python
def dem_to_mesh(elevation_raster: jnp.ndarray, pixel_spacing_m: float = 30.0) -> TriangularMesh:
    """Convert (H, W) elevation array to TriangularMesh with 2(H-1)(W-1) faces."""
    ...

def interpolate_covariates_to_vertices(
    mesh: TriangularMesh,
    raster_covariates: dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """Map raster covariate fields to mesh vertices, shape (V, d_cov)."""
    ...
```

#### CovariateMeshRanders

An extension of `DiscreteRanders` that replaces per-face wind vectors with covariate-conditioned G and b. Each face receives a per-face covariate vector (mean of vertex covariates + global weather); the encoder maps this to $(G_f, b_f)$.

**Mathematical formulation:**

On the mesh, the Randers metric at a point $x$ in face $f$ is:

$$F(x, v) = \frac{\sqrt{\lambda_f \|v\|^2 + \langle b_f, v \rangle^2} - \langle b_f, v \rangle}{\lambda_f}, \quad \lambda_f = 1 - \|b_f\|^2$$

where $v$ is a tangent vector in the ambient $\mathbb{R}^3$ (projected onto the face tangent plane by `TriangularMesh.project_to_tangent(x, v)`), and $b_f \in T_f M$ is the per-face wind/drift restricted to the face.

> **Math Reviewer:** The face tangent plane projection must be applied before evaluating the Randers cost. The Euclidean norm $\|v\|$ in the formula must be the norm in $\mathbb{R}^3$ restricted to the tangent plane of face $f$, not the full 3D norm. This is currently handled by `DiscreteRanders.metric_fn` via `get_face_weights` and ambient-space arithmetic — verify that the projection to face tangent space is applied before the Randers cost evaluation. Cross-check with `spec/MATH_SPEC.md § 1.1` (pullback metric on submanifolds).

**Training pipeline:** Same as Phase W1 except:
1. Source ignition converted from pixel coordinates to 3D mesh coordinates
2. Observation points projected to nearest mesh vertex
3. AVBD operates in 3D ambient space; path vertices are constrained to the mesh surface

### Ablations

| Ablation | Description | Purpose |
|----------|-------------|---------|
| 3D mesh vs. flat grid (control) | Same scene, same encoder, same K | Isolate terrain geometry effect |
| Full resolution vs. decimated mesh | 100% / 50% / 25% of vertices | Computational tradeoff |
| Face-level vs. vertex-level covariates | Per-face mean vs. per-vertex interp | Granularity effect |
| Slope-stratified evaluation | Flat / moderate / steep bins | Where does 3D help? |
| With/without aspect covariate | Include/exclude aspect | Aspect resolves slope direction ambiguity |

### Terrain Slope Stratification

Assign each fire to a **slope class** based on the mean slope of its burned area:
- **Flat:** slope std ≤ 3° (uniform terrain, projection distortion negligible)
- **Moderate:** 3° < slope std ≤ 8° (varied terrain)
- **Rugged:** slope std > 8° (high terrain variability)

> **F5 fix (FLAW):** The original plan used **mean slope** as the stratification variable. This is incorrect. Pearson r is **scale-invariant**: a spatially uniform $1/\cos\alpha$ distortion does not reduce r at all (it only rescales all arrival times by the same constant, leaving rank order unchanged). The 3D mesh can only improve r where **slope varies spatially** — i.e., where the distortion factor $1/\cos\alpha(x)$ differs across the fire domain. Use **slope standard deviation** within the burned area as the stratification variable. Also add RMSE as a secondary metric (not scale-invariant; will capture the absolute distortion effect).

Report correlation and RMSE separately per class. The hypothesis predicts the 3D mesh improvement will be concentrated in the Rugged class for RMSE, and may also appear in r for Rugged class (where distortion is non-uniform).

**Geometric rationale:** At slope $\alpha$, surface distance is $d_\text{surface} = d_\text{flat} / \cos(\alpha)$. For $\alpha = 30°$: $d_\text{surface} = 1.155 \times d_\text{flat}$ (+15.5%). The eikonal solver on a flat grid systematically underestimates this; the AVBD solver on the 3D mesh does not.

### Baseline Comparison

| Metric | Phase W1 (flat grid) | Phase W2 (3D mesh) | Expected Improvement |
|--------|---------------------|--------------------|---------------------|
| Correlation — all terrain | ≥ 0.70 | ≥ 0.71 | +1–2% |
| Correlation — moderate terrain | lower | higher | +2–4% |
| Correlation — steep terrain | lowest | higher | ≥ +5% |
| Runtime | baseline | ~2–4× slower | — |
| Gahtan can reproduce | yes | **no** | HAMTools-exclusive |

### Visualizations

| # | Figure Title | Type | X-axis | Y-axis / Content | Output File |
|---|-------------|------|--------|-------------------|-------------|
| W2.1 | 3D terrain with geodesic fire paths | 3D surface rendering | Surface x (m) | Surface y (m) / paths | `figs/phaseW2_3d_fire_paths.{pdf,png}` |
| W2.2 | Flat vs. 3D correlation by slope bin | Grouped bar chart | Slope class | Pearson r | `figs/phaseW2_slope_stratified.{pdf,png}` |
| W2.3 | Metric indicatrices on terrain surface | 3D ellipse field overlay | Surface x | Surface y | `figs/phaseW2_terrain_indicatrices.{pdf,png}` |
| W2.4 | Geodesic arc length: 3D vs 2D projected | Scatter | 2D predicted arrival time | 3D predicted arrival time | `figs/phaseW2_length_comparison.{pdf,png}` |

### Reproducibility Spec

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Hardware | 1× NVIDIA A100 80GB |
| Expected runtime | ~4–12 h per scene |
| Scenes | 3 highest-slope-variability scenes from Phase W1 |
| Mesh generation | Regular grid triangulation (2 triangles/pixel), 30m spacing |
| Vertex count | ~H×W ≈ 250k vertices for a 500×500 raster |
| Face count | ~2×499×499 ≈ 500k faces |

### Definition of Done
- [ ] 3D mesh achieves ≥ 2% correlation improvement over flat grid on ≥ 1 steep-terrain scene
- [ ] Slope-stratified analysis shows largest improvement in steep class
- [ ] 3D geodesic visualization on terrain surface produced
- [ ] Runtime comparison: flat grid vs. 3D mesh reported
- [ ] Explicit statement: "this capability is not reproducible with the eikonal approach"

---

## Phase W3: Geometric Analysis (HAMTools-Only)

### Objective
Generate geometric analyses impossible with the eikonal approach — fire corridor geodesics, Jacobi field divergence (fire spread stability), and curvature anomaly maps. These are secondary outputs that demonstrate the scientific value of explicit paths.

### 3.1 Fire Corridor Geodesics

For a given fire and trained metric, trace geodesics from the ignition point in $N_\theta = 64$ uniformly-spaced angular directions. Color by predicted arrival time. Overlay on terrain and fire mask to show which terrain features channel or block fire spread.

**Output:** 2D figure, ≥5 fires. This is the clearest visual demonstration of the Lagrangian advantage.

### 3.2 Jacobi Field Divergence (Fire Spread Stability)

For each geodesic $\gamma_\theta(t)$, compute the geodesic bundle $\{\gamma_{\theta+\delta}(t)\}$ for small $\delta$. The divergence rate $\partial d(\gamma_\theta(t), \gamma_{\theta+\delta}(t)) / \partial t$ indicates:
- **Converging geodesics:** Fire channeled (narrow valley, firebreak)
- **Diverging geodesics:** Fire spreading unstably (open slope, high risk)

This is computed via the finite-difference approximation (no Jacobi field ODE required):

$$\text{div}(t) = \frac{1}{\delta} \|\gamma_{\theta+\delta}(t) - \gamma_\theta(t)\|$$

### 3.3 Curvature Anomaly Map

Compute the Finsler flag curvature $\kappa(x, v)$ at sampled points $(x_i, v_i)$ using `ham.geometry.curvature`. High-curvature regions correspond to abrupt changes in fire behavior. Overlay on terrain to identify fuel type boundaries and wind shear zones.

### Definition of Done
- [ ] Fire corridor geodesics for ≥ 5 fires (both flat and 3D terrain)
- [ ] Jacobi divergence map for ≥ 3 fires showing identifiable channelling feature
- [ ] Curvature map for ≥ 1 scene

---

## Implementation Tasks

| # | Task | Agent | Input | Output | Acceptance Criterion |
|---|------|-------|-------|--------|---------------------|
| W1.1 | Implement `CovariateConditionedRanders` | Implementer | Mini-spec above; `spec/ARCH_SPEC.md § 3`, `spec/MATH_SPEC.md § 1.2` | `src/ham/models/wildfire.py` or `src/ham/models/learned.py` extension | Shape tests, feasibility tests (G SPD, b-norm < 1) |
| W1.2 | Implement `bind_scene` + raster bilinear interp | Implementer | `CovariateConditionedRanders` API sketch; `src/ham/geometry/mesh.py` | As method of W1.1 | JIT-compatible: no Python calls in forward pass |
| W1.3 | Implement HAMTools Sim2Real-Fire data loader | Implementer | `Sim2RealFireLoader` in `differentiable-eikonal-wildfire` as reference | `src/ham/data/wildfire.py` | Loads all 19 scenes; JAX arrays output |
| W1.4 | Implement arrival time extraction + stratified sampling | Implementer | W1.3 data loader; spec above | In W1.3 file | Arrival times match Gahtan's `extract_arrival_times`; sampling is stratified by decile |
| W1.5 | Phase W1 training script | Implementer | W1.1–W1.4; Phase W1 full spec | `examples/experiment_wildfire_flat.py` | Produces all W1 figures; per-scene correlation reported |
| W2.1 | Implement `dem_to_mesh` + `interpolate_covariates_to_vertices` | Implementer | Phase W2 spec; `src/ham/geometry/mesh.py` | `src/ham/utils/terrain.py` | Correct face count; vertices at correct 3D positions |
| W2.2 | Extend `DiscreteRanders` to covariate conditioning | Implementer | `CovariateMeshRanders` mini-spec; `src/ham/geometry/zoo/discrete.py` | Updated `discrete.py` or new `mesh_randers.py` | Randers feasibility on mesh; per-face G/b from covariates |
| W2.3 | Phase W2 training script | Implementer | W2.1–W2.2; Phase W2 full spec | `examples/experiment_wildfire_mesh.py` | Produces all W2 figures; slope-stratified table |
| W3.1 | Fire corridor + Jacobi divergence visualization | Implementer | Trained Phase W1/W2 metrics; Phase W3 spec | `examples/experiment_wildfire_geometric.py` | Figures for ≥5 fires |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| AVBD too slow for 100-epoch training (K=500, batch=16) | **High** | High | Reduce K to 200 for initial experiments; profile and JIT the inner AVBD loop; use `parallel=True` in AVBDSolver |
| Sim2Real-Fire download fails / dataset changes | Medium | High | Download one package first; cache locally; use Gahtan's loader for reference |
| Covariate bilinear interp inside JIT causes recompilation | High | Medium | Bake raster into JAX array constant at scene load; no Python indexing inside `metric_fn` |
| `TriangularMesh` AVBD convergence on steep terrain | Medium | Medium | Reduce step_size for steep meshes; add convergence monitoring; fall back to flat grid if diverges |
| FBFM fuel codes are integers — gradient through embedding | Low | Medium | Use `jnp.take(embedding, fuel_code)` — differentiable w.r.t. embedding table, not fuel code |
| Gahtan baseline numbers differ from paper (implementation detail) | Low | Medium | Run Gahtan's code on same scenes before comparing; use their reported numbers only as reference |
| Phase W2 mesh too large for AVBD (500k faces, 250k vertices) | Medium | High | Decimate to 50% resolution for initial experiments; use 3 scenes only |
| Slope-stratified improvement hypothesis is wrong (no improvement) | Medium | Low | Negative result is publishable; report honestly with geometric analysis of why |

---

## Sequential Dependency

```
W1.1 (CovariateConditionedRanders)
    └─→ W1.2 (bind_scene)
    └─→ W1.3 (data loader)
           └─→ W1.4 (arrival time extraction)
                  └─→ W1.5 (flat-grid training script)  ← Phase W1 done
                              └─→ W2.3 (mesh training script, shares encoder)
W2.1 (dem_to_mesh)
    └─→ W2.2 (CovariateMeshRanders)
           └─→ W2.3 (mesh training script)           ← Phase W2 done
W1.5 + W2.3 → W3.1 (geometric analysis)             ← Phase W3 done
```

Phases W1 and W2 share the same `CovariateConditionedRanders` encoder (W1.1). Phase W2 only adds the DEM-to-mesh pipeline (W2.1) and the metric adaptation to the mesh manifold (W2.2). The training loop (W2.3) is a light modification of W1.5.

---

## Science Auditor Checklist (Pre-Implementation)

The following questions must be resolved by the Science Auditor before W1.5 and W2.3 are implemented:

1. **Train/val/test contamination:** Is the 70/15/15 split performed at the fire level within each scene (temporal split by timestamp), or randomly? Gahtan uses random split. Temporal split would be more realistic but harder.
2. **Multi-source:** Gahtan demonstrates 45% error reduction with 5 sources vs. 1 (Table 3). Sim2Real-Fire fires typically have a single ignition point. Is the single-source setup fair?
3. **Arrival time normalization:** Should arrival times be normalized per fire (0–1) or per scene? Per-fire normalization removes absolute speed information; per-scene keeps it.
4. **Correlation metric:** Resolved (F1). Dense evaluation pass over all GT-burned test pixels. `K_train` and `K_eval` are independent. `K_eval = all burned pixels` in test split.
5. **IoU computation:** Resolved (F2). Threshold = `max(GT_arrival_times) × 0.5`. Match Gahtan's `iou_50` in `per_scene.py:L201`.
