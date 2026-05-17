# Experiment Plan: Lagrangian Randers Metric Recovery — Reproducing and Extending Gahtan et al. (2026)

**Architect:** Research Architect Agent  
**Date:** May 17, 2026  
**Source Document:** [reviews/science/comparative_gahtan2026.md](reviews/science/comparative_gahtan2026.md)  
**Reference Paper:** Gahtan, Shpund & Bronstein. *Wildfire Simulation with Differentiable Randers-Finsler Eikonal Solvers.* arXiv:2603.00035 [cs.CE], Feb 2026.  
**Reference Code:** [BarakGahtan/differentiable-eikonal-wildfire](https://github.com/BarakGahtan/differentiable-eikonal-wildfire) (MIT License)

---

## Literature Brief

### Core Claims of the Reference Paper

Gahtan et al. present a differentiable Randers-Finsler eikonal solver on 2D Cartesian grids, combining GPU-friendly column/row fast sweeping ($O(n)$ per iteration, 2–3 iterations to convergence) with implicit differentiation for exact gradients. Key quantitative results:

| Result | Value | Paper Reference |
|--------|-------|-----------------|
| Forward solver accuracy (isotropic) | Sub-1% relative $L_2$ error | Table 1 |
| Isotropic metric recovery (100% obs.) | 5.6% error | Section 5, Figure 27 |
| Isotropic metric recovery (7% obs.) | 21.2% error | Section 5, Figure 28 |
| Drift field recovery | <3% error ($b_1$: 2.8%, $b_2$: 2.0%) | Section 5 |
| Multi-source improvement (1→5 sources) | 18.6% → 10.1% (45% reduction) | Table 3 |
| Within-scene wildfire correlation | 0.824 ± 0.044 | Section 6.1, Figure 7 |
| Cross-scene wildfire correlation | 0.766 | Section 6.2, Table 4 |
| Within-scene wildfire IoU@50 | 0.609 | Section 6.1 |
| Cross-scene wildfire IoU@50 | 0.295 | Table 4 |
| Neural encoder capacity (single-fire RMSE) | 9.4% (median 6.3%) | Table 5 |
| Real wildfire correlation (sim-trained) | 0.588 ± 0.172 | Table 8 |

### Mathematical Equivalence (Eulerian ↔ Lagrangian)

The Randers eikonal equation $(\nabla T - \mathbf{b})^\top \mathbf{G}^{-1} (\nabla T - \mathbf{b}) = 1$ is the static Hamilton–Jacobi equation whose characteristic ODE is the Randers geodesic spray $\ddot{x}^i + 2G^i(x, \dot{x}) = 0$ (see `spec/MATH_SPEC.md` § 2.1). The arrival time $T(x) = d_F(\text{source}, x) = \inf_\gamma \int_0^1 F(\gamma, \dot\gamma)\,dt$, which is exactly what `FinslerMetric.arc_length()` computes on the AVBD-solved geodesic (see `spec/MATH_SPEC.md` § 1.2). This equivalence is exact in the continuum; discrete discrepancies arise from different approximation schemes (grid stencils vs. discrete path vertices).

**Math Reviewer validation (NOTE):** The equivalence holds away from the cut locus. At cut locus points, the eikonal solution may select a different geodesic than AVBD, but both compute the correct distance. AVBD may converge to a local minimum (longer geodesic) in rare cases — this is addressed by multi-start initialization (see Phase 1 Method).

### Dataset Availability

**Sim2Real-Fire:** Publicly available on Google Drive under Apache-2.0 license. 15 scenes, 26,752 fires total. Five covariate modalities: topography (30m), vegetation, fuel, weather (hourly), satellite fire masks. Dataset paper: Li et al. (2024) NeurIPS.

**Gahtan code:** MIT license, PyTorch + Numba/CUDA implementation. Contains encoder architecture, loss functions, training scripts, and data loaders. Can be used to reproduce their baselines and extract exact ground truth arrival time fields for our experiments.

### Related Work Providing Additional Context

- **Mirebeau (2014):** Efficient fast marching with Finsler metrics (arXiv:1406.1233). Alternative eikonal solver approach.
- **Qian, Zhang & Zhao (2007):** Fast sweeping on triangular meshes. Foundation for our Phase 5 mesh extension.
- **Chen et al. (2018):** Neural ODEs (arXiv:1806.07366). Adjoint ODE differentiation for memory-efficient geodesic backward pass.
- **Crane, Weischedel & Wardetzky (2017):** The heat method for distance computation. Alternative distance field approach.

---

## Scope & Success Criteria

This experiment series validates that HAMTools' **Lagrangian geodesic approach** (AVBD/ExponentialMap solvers) can solve the same Randers metric inverse problem as Gahtan's Eulerian eikonal solver, then extends it to domains where eikonal solvers cannot operate. Success requires:

1. **Replication:** Recover synthetic Randers metrics from arrival-time data with quantitative comparison to Gahtan's baselines. Primary metric: relative recovery error within 2× of Gahtan at matched observation density.
2. **Wildfire:** Achieve within-scene correlation ≥ 0.70 on Sim2Real-Fire (Gahtan: 0.824). Report honest runtime comparison.
3. **Novel extension:** Demonstrate wildfire propagation on triangulated DEM surfaces — a capability Gahtan explicitly lists as future work (Section 7) — and show measurable improvement on steep terrain.
4. **Geometric analysis:** Produce geodesic path visualizations, Jacobi field stability maps, and curvature anomaly maps that are impossible with the eikonal approach.

---

## Phase 0: Prerequisite Bug Fix

### EulerLagrangeResidualLoss Formula Correction

**Math Reviewer finding (CRITICAL):** The local Lagrangian `L_smooth` in `EulerLagrangeResidualLoss` at [src/ham/training/losses.py](src/ham/training/losses.py#L290-L300) uses a simplified Randers energy $F_\epsilon = \sqrt{v^\top H v + \epsilon^2} - \langle W, v \rangle_H$ that **does not match** the Zermelo formula in [src/ham/geometry/zoo/randers.py](src/ham/geometry/zoo/randers.py#L73-L90) or `spec/MATH_SPEC.md` § 5. The correct formula involves $\lambda = 1 - \|W\|_H^2$ and the quadratic term $\langle W, v \rangle_H^2$ under the square root.

**Fix:** Replace the hand-coded Lagrangian with a call to `model.metric.energy(z_pt, v_pt)`, which already implements the correct formula. This fix must be applied and tested before any E-L residual experiments.

**Responsible agent:** Implementer  
**Acceptance criterion:** `EulerLagrangeResidualLoss` produces near-zero residual on AVBD-solved geodesics of `NeuralRanders` metrics.

---

## Phase 1: Synthetic Isotropic Metric Recovery

### Objective
Validate that HAMTools' Lagrangian approach can recover a spatially varying isotropic metric from arrival-time observations, establishing a quantitative baseline against Gahtan Section 5.

### Hypothesis
The AVBD solver computes exact geodesic distances (up to discretization), so the inverse problem is mathematically identical to the eikonal formulation. At matched observation density, recovery error should be comparable. At low density, the Lagrangian approach may outperform because each geodesic provides path information (not just endpoint distance).

### Method

**HAMTools Components:**
- Manifold: `EuclideanSpace(2)` from [src/ham/geometry/manifolds/](src/ham/geometry/manifolds/)
- Metric: `NeuralRanders` from [src/ham/models/learned.py](src/ham/models/learned.py) with `use_wind=False` initially (Riemannian-only ablation), then with wind
- Solver: `AVBDSolver` from [src/ham/solvers/avbd.py](src/ham/solvers/avbd.py), `n_steps=20`, `iterations=100`
- Pipeline: `HAMPipeline` from [src/ham/training/pipeline.py](src/ham/training/pipeline.py)

**New Components Needed:**

#### 1. ArrivalTimeLoss

**Mathematical formulation:**
$$\mathcal{L}_{\text{arrival}} = \frac{1}{K} \sum_{i=1}^{K} \left( T_i^{\text{pred}} - T_i^{\text{obs}} \right)^2$$

where $T_i^{\text{pred}} = \text{arc\_length}(\text{AVBD.solve}(\text{metric}, x_{\text{source}}, x_i))$ and $T_i^{\text{obs}}$ is the ground-truth arrival time.

The arc length on a discrete path $\{x_0, \ldots, x_N\}$:
$$T^{\text{pred}} = \sum_{k=0}^{N-1} F\left(x_k, \frac{x_{k+1} - x_k}{\Delta t}\right) \cdot \Delta t$$

where $\Delta t = 1/(N-1)$ (using the midpoint quadrature). By 1-homogeneity of $F$, this simplifies to $\sum_k F(x_k, x_{k+1} - x_k)$.

**Proposed API:**
```python
class ArrivalTimeLoss(LossComponent):
    """MSE between predicted geodesic distance and observed arrival time."""
    solver_steps: int  # AVBD path discretization
    solver_iters: int  # AVBD optimization iterations
    
    def __call__(self, model, batch, key):
        # batch = (source, x_obs, T_obs)
        # Vmapped over observations
        ...
```

Extends `spec/ARCH_SPEC.md` § 4 (Solver Interface) and `spec/MATH_SPEC.md` § 1.2 (Energy Functional).

#### 2. SyntheticEikonalDataGenerator

Generates ground-truth arrival time fields for synthetic Randers metrics. Uses Gahtan's eikonal solver (via their public code) or a dense grid of HAMTools geodesics to compute $T_{\text{true}}(x)$ for known $(h, W)$.

**Training Pipeline:**
- **Loss:** `ArrivalTimeLoss(weight=1.0)` + `MetricSmoothnessLoss(weight=λ_TV)`
- **Optimizer:** Adam, lr=1e-3 with cosine decay
- **Batch size:** K observations per source, sampled uniformly from grid
- **Number of steps:** 500 optimizer steps

**Data:**
- 80×80 grid, piecewise constant metric: $g(x) = 1$ for $x < 40$, $g(x) = 2$ for $x \geq 40$
- Source at grid center $(40, 40)$
- Ground-truth arrival times from Gahtan's eikonal solver (or analytical: $T(x) = d_F(\text{source}, x)$ under the piecewise metric)
- Random seed: 42

**Observation density sweep (Science Auditor CRITICAL fix):**

| Configuration | K (obs. points) | Density | Gahtan Baseline |
|---------------|-----------------|---------|-----------------|
| Dense | 6400 (100%) | 100% | 5.6% |
| Medium | 3200 (50%) | 50% | ~6% (phase transition) |
| Sparse | 448 (7%) | 7% | 21.2% |
| Very sparse | 200 (3.1%) | 3.1% | N/A (underdetermined) |

### Ablations (Science Auditor CRITICAL fix: Riemannian control)

| Ablation | Description | Purpose |
|----------|-------------|---------|
| **W=0 (Riemannian)** | `NeuralRanders(use_wind=False)` | Isolate metric tensor recovery from wind recovery |
| **Observation density** | K ∈ {200, 448, 3200, 6400} | Match Gahtan's density conditions |
| **AVBD steps** | n_steps ∈ {10, 15, 20, 30} | Solver discretization sensitivity |
| **AVBD iterations** | iterations ∈ {50, 100, 200} | Solver convergence sensitivity |
| **Network depth** | depth ∈ {2, 3, 4} | Encoder capacity |
| **TV regularization** | λ_TV ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1} | Match Gahtan's regularization sweep |
| **Multi-start AVBD** | 1 vs. 3 random initializations | Robustness to AVBD local minima |

### Baseline Comparison

| Metric | Gahtan (100% obs.) | Gahtan (7% obs.) | Our Target (100%) | Our Target (7%) | Paper Source |
|--------|--------------------|--------------------|--------------------|--------------------|--------------|
| Metric recovery error | 5.6% | 21.2% | ≤ 12% | ≤ 30% | Section 5, Fig. 27-28 |
| Convergence iterations | 300 | 300 | 500 | 500 | Section 5 |

### Visualizations

| # | Figure Title | Type | X-axis | Y-axis / Content | Baseline Overlay | Output File |
|---|-------------|------|--------|-------------------|-----------------|-------------|
| 1.1 | Recovered vs. True Metric (100% obs.) | Heatmap trio | Grid x | Grid y / metric value | Gahtan Fig. 27 layout | `figs/phase1_metric_recovery_full.{pdf,png}` |
| 1.2 | Recovered vs. True Metric (7% obs.) | Heatmap trio | Grid x | Grid y / metric value | Gahtan Fig. 28 layout | `figs/phase1_metric_recovery_sparse.{pdf,png}` |
| 1.3 | Error vs. Observation Density | Line plot | Density (%) | Relative error (%) | Gahtan Fig. 32 overlay | `figs/phase1_density_sweep.{pdf,png}` |
| 1.4 | Error vs. TV Regularization | Line plot (U-curve) | λ_TV | Relative error (%) | Gahtan Fig. 31 overlay | `figs/phase1_regularization.{pdf,png}` |
| 1.5 | Loss Convergence | Semi-log | Iteration | Loss | — | `figs/phase1_convergence.{pdf,png}` |
| 1.6 | Geodesic Paths (HAMTools-only) | 2D overlay on metric field | x | y / geodesic paths colored by arrival time | N/A (eikonal can't produce paths) | `figs/phase1_geodesic_paths.{pdf,png}` |

### Reproducibility Spec

| Parameter | Value |
|-----------|-------|
| Random seed | 42 (data generation), 0–4 (5 training seeds) |
| Hardware | Single GPU (NVIDIA A100 or Apple M-series) |
| Expected runtime | ~30 min per seed per density level |
| JAX version | ≥ 0.4.30 |
| Grid size | 80×80 |
| AVBD n_steps | 20 |
| AVBD iterations | 100 |
| Network hidden_dim | 64 |
| Network depth | 3 |
| Optimizer | Adam, lr=1e-3, cosine decay over 500 steps |
| TV regularization | λ_TV = 1e-3 (default; swept in ablation) |

### Definition of Done
- [ ] Metric recovery error ≤ 12% at 100% observation density (mean over 5 seeds)
- [ ] Metric recovery error ≤ 30% at 7% observation density
- [ ] Observation density sweep produces a curve qualitatively matching Gahtan Fig. 32 (phase transition ~50%)
- [ ] TV regularization sweep produces a U-curve qualitatively matching Gahtan Fig. 31
- [ ] Riemannian ablation (W=0) shows comparable or better recovery (confirming wind is not needed for isotropic ground truth)
- [ ] Geodesic path visualization demonstrates HAMTools-only capability
- [ ] All results reported with mean ± std over 5 seeds

---

## Phase 2: Synthetic Drift Recovery

### Objective
Validate that the Lagrangian approach can recover the Randers drift field $\mathbf{b}$ (wind) when the Riemannian base metric is known. This isolates the Finsler-specific contribution.

### Hypothesis
Drift creates a strong observational signature through upwind/downwind asymmetry. The AVBD solver captures this asymmetry through the Randers metric_fn, so recovery should be effective. We expect performance comparable to Gahtan (<3% at full density).

### Method

**Setup:**
- 80×80 grid, $\mathbf{G} = \mathbf{I}$ (known, frozen), constant true drift $\mathbf{b} = (0.15, 0.08)$
- Source at center
- `NeuralRanders` with frozen `h_net = lambda x: jnp.eye(2)`, learnable `w_net`
- Same `ArrivalTimeLoss` + `MetricSmoothnessLoss` pipeline

**Observation density sweep:** Same as Phase 1: K ∈ {200, 448, 3200, 6400}

### Ablations

| Ablation | Description | Purpose |
|----------|-------------|---------|
| Known G vs. learned G | Freeze vs. learn the base metric | Test interaction between G and b recovery |
| Drift magnitude | $\|b\| \in \{0.05, 0.15, 0.30, 0.50\}$ | Sensitivity to drift strength |
| Multi-source | 1, 3, 5 sources | Replicate Gahtan Table 3 improvement |

### Baseline Comparison

| Metric | Gahtan | Our Target | Paper Source |
|--------|--------|------------|--------------|
| $b_1$ recovery error (100% obs.) | 2.8% | ≤ 6% | Section 5 |
| $b_2$ recovery error (100% obs.) | 2.0% | ≤ 4% | Section 5 |
| Multi-source improvement | 45% reduction | ≥ 25% reduction | Table 3 |

### Visualizations

| # | Figure Title | Type | X/Y | Output File |
|---|-------------|------|-----|-------------|
| 2.1 | Recovered vs. True Drift | Vector field overlay | Grid coords | `figs/phase2_drift_recovery.{pdf,png}` |
| 2.2 | Drift Error Map | Heatmap | Grid coords | `figs/phase2_drift_error.{pdf,png}` |
| 2.3 | Geodesics: With vs. Without Wind | 2D path comparison | x, y | `figs/phase2_wind_geodesics.{pdf,png}` |
| 2.4 | Multi-Source Error Reduction | Bar chart | # Sources / Error | `figs/phase2_multisource.{pdf,png}` |

### Reproducibility Spec

Same as Phase 1 except: frozen `h_net`, only `w_net` trainable, 300 optimizer steps.

### Definition of Done
- [ ] Drift recovery error ≤ 6% per component at 100% density (mean over 5 seeds)
- [ ] Multi-source training shows ≥ 25% error reduction (1 → 5 sources)
- [ ] Geodesic visualization shows asymmetric paths consistent with recovered drift
- [ ] Results reported with mean ± std over 5 seeds

---

## Phase 3: Combined Randers Recovery (Anisotropic G + Drift b)

### Objective
Recover the full Randers metric (spatially varying $\mathbf{G}(x)$ and $\mathbf{b}(x)$) from multi-source arrival-time data. This is the hardest synthetic test, matching the full Randers-Finsler validation.

### Hypothesis
Joint recovery is underdetermined from a single source (Gahtan achieves 42–52% error on anisotropic recovery). Multi-source data with diverse geometric coverage should significantly improve recovery. The Lagrangian approach may have an advantage here because each BVP solution provides *path* information, not just distance.

### Method

**Setup:**
- 80×80 grid, piecewise-constant anisotropic metric: $\mathbf{G} = \text{diag}(g_{11}(x), g_{22}(x))$ with different values in each quadrant
- Spatially varying drift: $\mathbf{b}(x) = 0.15 \cdot (\cos(\theta(x)), \sin(\theta(x)))$ where $\theta$ varies linearly
- 3 and 5 ignition sources at varied positions
- Full `NeuralRanders` (both `h_net` and `w_net` learnable)
- TV regularization on both $\mathbf{G}$ and $\mathbf{b}$

**Multi-phase training pipeline:**
1. **Phase 3a (warm-up):** Freeze wind, learn G only from isotropic initial guess (200 steps)
2. **Phase 3b (joint):** Unfreeze wind, learn both G and b jointly (500 steps)

### Ablations

| Ablation | Description | Purpose |
|----------|-------------|---------|
| Warm-up vs. cold-start | With/without Phase 3a | Curriculum learning benefit |
| Euler-Lagrange residual (after Phase 0 fix) | Add E-L residual loss as auxiliary | Test solver-free supervision signal |
| Path-based supervision | Add intermediate path point matching | Exploit Lagrangian advantage |

### Baseline Comparison

| Metric | Gahtan | Our Target | Notes |
|--------|--------|------------|-------|
| $g_{11}$ recovery | 42% | ≤ 50% | Anisotropic recovery is inherently harder |
| $g_{22}$ recovery | 52% | ≤ 60% | |
| Joint with multi-source | ~10% (isotropic proxy) | ≤ 15% | Table 3 trend |

### Visualizations

| # | Figure Title | Type | Output File |
|---|-------------|------|-------------|
| 3.1 | G tensor component recovery | 4-panel heatmap (true/recovered × $g_{11}$/$g_{22}$) | `figs/phase3_anisotropic_recovery.{pdf,png}` |
| 3.2 | Recovered indicatrices overlaid on metric field | Ellipse field | `figs/phase3_indicatrices.{pdf,png}` |
| 3.3 | Geodesic fan from each source | 2D multi-panel | `figs/phase3_geodesic_fans.{pdf,png}` |

### Reproducibility Spec

Same as Phase 1 except: 3–5 sources, 700 total optimizer steps, `hidden_dim=64`, `depth=3`.

### Definition of Done
- [ ] Convergence of joint loss (G + b) demonstrated over training
- [ ] Multi-source recovery significantly outperforms single-source
- [ ] Warm-up curriculum shows measurable benefit (ablation table)
- [ ] Indicatrix visualization shows qualitative agreement with ground truth
- [ ] All results with mean ± std over 5 seeds

---

## Phase 4: Sim2Real-Fire Wildfire (Head-to-Head)

### Objective
Apply the Lagrangian metric learning approach to real wildfire propagation data from the Sim2Real-Fire dataset, enabling direct quantitative comparison with Gahtan's eikonal-based results.

### Hypothesis
The Lagrangian approach will produce lower correlation/IoU than the eikonal solver (which amortizes computation over the full grid), but will remain in a useful range (≥ 0.70 correlation) and provide richer geometric outputs (actual fire spread geodesics).

### Method

**HAMTools Components:**
- Manifold: `EuclideanSpace(2)`
- Metric: Covariate-conditioned `NeuralRanders` (new component needed; see mini-spec below)
- Solver: `AVBDSolver` (Strategy A) and `ExponentialMap` (Strategy C)
- Losses: `ArrivalTimeLoss` + `MetricSmoothnessLoss` + `WindThermodynamicLoss`

**New Components Needed:**

#### CovariateConditionedRanders

A neural Randers metric that conditions on environmental covariates (terrain, fuel, weather), analogous to Gahtan's `FlexibleCovariateEncoder`. Unlike HAMTools' existing `DataDrivenPullbackRanders` (which conditions on observed data in latent space), this operates in the spatial domain on raw covariate fields.

**Architecture:**
$$(\mathbf{G}(x), \mathbf{b}(x)) = f_\theta(\mathbf{Y}(x), \mathbf{y})$$

where $\mathbf{Y}(x)$ is the local spatial covariate vector at position $x$ (elevation, slope, aspect, fuel type, vegetation cover) and $\mathbf{y}$ is the global weather vector (temperature, humidity, wind speed, wind direction).

**Proposed API:**
```python
class CovariateConditionedRanders(Randers):
    """Randers metric conditioned on environmental covariates."""
    spatial_encoder: eqx.Module  # MLP: R^d_spatial → R^(3+2) = (g11, g12, g22, b1, b2)
    global_encoder: eqx.Module   # MLP: R^d_global → R^(3+2) baseline parameters
    
    def _get_zermelo_data(self, x, covariates_spatial, covariates_global):
        # Global baseline
        baseline = self.global_encoder(covariates_global)
        # Local residual
        residual = self.spatial_encoder(covariates_spatial)
        # Combine and project to feasible (G, b)
        ...
```

Extends `spec/ARCH_SPEC.md` § 3 (Metric Hierarchy) with covariate conditioning.

**Strategy A: Sparse BVP**
For each training fire:
1. Source = ignition point(s)
2. Sample $K$ observation points with arrival times from the fire
3. For each observation (vmapped): solve BVP, compute arc length
4. Loss = MSE(predicted, observed arrival times) + regularization

**K selection (Science Auditor CRITICAL fix):** K must be proportional to the fire's burned area, not fixed. Use $K = \min(500, \text{burned\_pixels} \times 0.3)$ to ensure ≥30% density within the burned region.

**Strategy C: Geodesic Shooting Fan**
1. Shoot $N_\theta = 128$ geodesics from source in uniformly spaced directions
2. Each ray traces out a geodesic path; arrival time is the integration parameter
3. For each observation, find the nearest geodesic ray and interpolate arrival time
4. Loss = MSE + regularization

**Math Reviewer WARNING:** The nearest-ray assignment is non-differentiable (argmin). Use a soft assignment via distance-weighted average:
$$T_i^{\text{pred}} = \frac{\sum_\theta w_{i\theta} \cdot T_\theta(r_i)}{\sum_\theta w_{i\theta}}, \quad w_{i\theta} = \exp\left(-\frac{d(x_i, \gamma_\theta)^2}{2\sigma^2}\right)$$

**Training Pipeline:**
- Per-scene: same 70/15/15 split as Gahtan
- Optimizer: Adam, lr=5e-4, cosine decay
- Epochs: 100 with early stopping (patience=20, monitoring val correlation)
- Batch size: 16 fires per batch (vs. Gahtan's 64; limited by BVP cost)

**Data:**
- Sim2Real-Fire dataset, 15 scenes (26,752 fires)
- Covariates: topography (elevation, slope, aspect), vegetation, fuel, weather
- Preprocessing: normalize covariates per-scene; extract arrival times from fire masks

### Ablations (Science Auditor WARNING fix: Strategy comparison)

| Ablation | Description | Purpose |
|----------|-------------|---------|
| Strategy A vs. C | BVP vs. shooting fan | Compare Lagrangian solver strategies |
| K sensitivity (Strategy A) | K ∈ {100, 200, 500} | Observation sampling sensitivity |
| $N_\theta$ sensitivity (Strategy C) | $N_\theta$ ∈ {32, 64, 128, 256} | Angular resolution sensitivity |
| With vs. without wind | `use_wind=True/False` | Riemannian vs. Randers (core ablation) |
| AVBD parallelism | `parallel=True/False` | Computational speedup impact |
| Temporal band decomposition | Single-shot vs. band training | Match Gahtan Appendix D curriculum |

### Baseline Comparison

| Metric | Gahtan (Per-Scene) | Our Target (Per-Scene) | Paper Source |
|--------|--------------------|-----------------------|--------------|
| Pearson correlation | 0.824 ± 0.044 | ≥ 0.70 | Figure 7 |
| Relative RMSE | varies by scene | report | Table 9 |
| IoU@50 | 0.609 | ≥ 0.40 | Figure 7 |
| Runtime per fire (training) | ~10ms (GPU eikonal) | report (expect ~1–10s) | Section 4.3 |

### Visualizations

| # | Figure Title | Type | Output File |
|---|-------------|------|-------------|
| 4.1 | Per-scene correlation comparison | Bar chart (Gahtan vs. HAMTools) | `figs/phase4_correlation_comparison.{pdf,png}` |
| 4.2 | Qualitative fire predictions (best/typical/worst) | 3×3 grid (GT / predicted / error) | `figs/phase4_qualitative.{pdf,png}` |
| 4.3 | Geodesic fire corridors | 2D paths overlaid on terrain | `figs/phase4_fire_corridors.{pdf,png}` |
| 4.4 | Strategy A vs. C runtime/accuracy tradeoff | Scatter (runtime vs. correlation) | `figs/phase4_strategy_comparison.{pdf,png}` |
| 4.5 | Learned metric field visualization | Indicatrix + wind overlay on terrain | `figs/phase4_learned_metric.{pdf,png}` |

### Reproducibility Spec

| Parameter | Value |
|-----------|-------|
| Random seed | 42 (data), 0–2 (3 training seeds per scene) |
| Hardware | 1× NVIDIA A100 80GB or Apple M3 Ultra |
| Expected runtime | ~2–6 hours per scene (Strategy A, K=200) |
| JAX version | ≥ 0.4.30 |
| Scenes (per-scene) | All 15 |
| Train/val/test split | 70/15/15 (same as Gahtan) |
| AVBD n_steps | 15 |
| AVBD iterations | 50 |
| Network hidden_dim | 64 |
| Network depth | 3 |
| Optimizer | Adam, lr=5e-4, cosine decay |
| Early stopping patience | 20 epochs |

### Definition of Done
- [ ] Per-scene mean correlation ≥ 0.70 (mean over seeds)
- [ ] At least 3 scenes with correlation ≥ 0.80
- [ ] Runtime comparison table: HAMTools vs. Gahtan per training step
- [ ] Geodesic fire corridor visualization produced for ≥ 5 fires
- [ ] Strategy A vs. C comparison table
- [ ] Riemannian vs. Randers ablation shows Randers improves correlation
- [ ] All results reported with mean ± std over 3 seeds

---

## Phase 5: Triangulated DEM Surface (Novel Contribution)

### Objective
Extend the wildfire propagation model to operate on the actual 3D terrain surface rather than a flat 2D projection, directly addressing Gahtan's stated limitation (Section 7: *"Extension to triangulated surfaces would replace the eight-stencil system with triangle-fan neighborhoods"*).

### Hypothesis
Modeling fire on the actual terrain surface captures slope-dependent propagation more accurately than a flat grid, particularly on steep terrain where the 2D projection distorts distances. We predict measurable improvement (≥ 5% correlation gain) on scenes with high terrain variability.

### Method

**HAMTools Components:**
- Manifold: `TriangularMesh` from [src/ham/geometry/mesh.py](src/ham/geometry/mesh.py)
- Metric: `DiscreteRanders` from [src/ham/geometry/zoo/discrete.py](src/ham/geometry/zoo/discrete.py) or a mesh-adapted `CovariateConditionedRanders`
- Solver: `AVBDSolver` (already supports `TriangularMesh` via manifold ABC)

**New Components Needed:**

#### DEMToMesh pipeline
Convert raster DEM (Digital Elevation Model) from Sim2Real-Fire topography layers to a `TriangularMesh`. Each pixel $(i, j)$ at elevation $z_{ij}$ becomes a 3D vertex $(x_i, y_j, z_{ij})$, connected into a triangulation via Delaunay or a regular grid triangulation (each cell split into 2 triangles).

#### Covariate interpolation on mesh
Map raster covariates (fuel, vegetation, weather) to mesh vertices via barycentric interpolation.

**Training Pipeline:**
- Same covariate encoder as Phase 4, but operating on 3D mesh vertices
- AVBD solver operates in 3D ambient space with `TriangularMesh` constraints
- Loss: same `ArrivalTimeLoss` formulation

**Science Auditor WARNING fix: Slope stratification**
Evaluate performance separately on:
- Flat terrain (slope < 10°): expect comparable to flat-grid
- Moderate terrain (10° < slope < 30°): expect slight improvement
- Steep terrain (slope > 30°): expect significant improvement

### Ablations

| Ablation | Description | Purpose |
|----------|-------------|---------|
| 3D mesh vs. flat grid (control) | Same scene, same encoder, different domain | Isolate terrain geometry effect |
| Mesh resolution | Full-res vs. decimated (50%, 25%) | Computational tradeoff |
| Slope-stratified evaluation | Flat / moderate / steep bins | Where does 3D help? |

### Baseline Comparison

| Metric | Flat Grid (Phase 4) | 3D Mesh (Phase 5) | Expected Improvement |
|--------|--------------------|--------------------|---------------------|
| Correlation (all terrain) | ≥ 0.70 | ≥ 0.72 | +2–5% |
| Correlation (steep terrain only) | lower | higher | ≥ +5% |
| Runtime | baseline | ~2–3× slower (3D) | — |

### Visualizations

| # | Figure Title | Type | Output File |
|---|-------------|------|-------------|
| 5.1 | 3D terrain with geodesic fire paths | 3D surface rendering | `figs/phase5_3d_fire_paths.{pdf,png}` |
| 5.2 | Flat vs. 3D correlation by slope bin | Grouped bar chart | `figs/phase5_slope_stratified.{pdf,png}` |
| 5.3 | Mesh-based metric indicatrices on terrain | 3D ellipse overlay | `figs/phase5_terrain_indicatrices.{pdf,png}` |

### Reproducibility Spec

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Hardware | 1× NVIDIA A100 80GB |
| Expected runtime | ~4–12 hours per scene |
| Scenes | 3 scenes with highest terrain variability |
| Mesh generation | Regular grid triangulation (2 triangles per pixel) |

### Definition of Done
- [ ] 3D mesh achieves ≥ 2% correlation improvement over flat grid on at least 1 scene
- [ ] Slope-stratified analysis shows largest improvement on steep terrain
- [ ] 3D geodesic visualization on terrain surface produced
- [ ] Runtime comparison: flat grid vs. 3D mesh
- [ ] All results with mean ± std over 3 seeds

---

## Phase 6: Geometric Analysis Extensions (HAMTools-Only Capabilities)

### Objective
Demonstrate geometric analysis capabilities that are fundamentally impossible with the eikonal approach: geodesic path visualization, stability analysis via Jacobi fields, and curvature-based anomaly detection.

### 6.1 Geodesic Fire Corridors

**Method:** Using the Phase 4 trained model, trace geodesics from ignition points in multiple directions. Color-code by arrival time. These represent the "fastest burn corridors" — directly interpretable for firefighting resource allocation.

**Output:** Side-by-side comparison with Gahtan's arrival time heatmap. The geodesic paths provide strictly more information (path + time vs. time only).

### 6.2 Geodesic Stability via Jacobi Fields

**Method:** For each fire, compute geodesic bundles (families of nearby geodesics) using `ExponentialMap.trace()` with perturbed initial velocities. The divergence rate of nearby geodesics indicates fire spread stability:
- **Converging geodesics:** Fire is channeled (valley, firebreak)
- **Diverging geodesics:** Fire is spreading unstably (open terrain, spotting risk)

Use Berwald parallel transport along geodesics to compute the connection map and assess the rate of divergence.

**Math Reviewer NOTE:** Full Jacobi field integration is not yet implemented in HAMTools. The geodesic bundle approach (finite-difference approximation via nearby geodesics) is a valid first-order proxy.

### 6.3 Curvature Anomaly Detection

**Method:** Compute the Finsler curvature tensor (available via HAMTools' auto-differentiated spray) at sampled points in the learned metric field. Regions of high curvature correspond to abrupt changes in fire behavior — useful for identifying:
- Terrain features that dramatically alter spread dynamics
- Fuel type boundaries
- Wind shear zones

**New Component (mini-spec):**
```python
def curvature_magnitude_field(metric, grid_points, v_reference):
    """Compute scalar curvature magnitude at each grid point."""
    # Uses ham.geometry.curvature module
    # Returns: (N,) array of |R| values
```

### Visualizations

| # | Figure Title | Type | Output File |
|---|-------------|------|-------------|
| 6.1 | Geodesic corridors vs. eikonal wavefronts | Side-by-side 2D | `figs/phase6_corridors_vs_wavefronts.{pdf,png}` |
| 6.2 | Geodesic bundle divergence map | 2D heatmap | `figs/phase6_geodesic_stability.{pdf,png}` |
| 6.3 | Curvature anomaly map overlaid on terrain | 2D heatmap + contours | `figs/phase6_curvature_anomalies.{pdf,png}` |
| 6.4 | Parallel transport of "fire direction" along geodesic | Vector field along path | `figs/phase6_parallel_transport.{pdf,png}` |

### Definition of Done
- [ ] Geodesic corridor visualization for ≥ 5 fires
- [ ] Geodesic bundle divergence computed and visualized for ≥ 3 fires
- [ ] Curvature anomaly map identifies ≥ 1 terrain feature per fire
- [ ] Parallel transport visualization shows wind rotation along geodesic
- [ ] Narrative comparison explaining what these provide beyond eikonal arrival times

---

## Implementation Tasks

| # | Task | Agent | Input | Output | Acceptance Criterion |
|---|------|-------|-------|--------|---------------------|
| 0.1 | Fix `EulerLagrangeResidualLoss` formula | Implementer | Math Reviewer CRITICAL finding; [src/ham/training/losses.py](src/ham/training/losses.py#L290-L300) | Fixed loss module | E-L residual ≈ 0 on AVBD geodesics |
| 0.2 | Unit test for E-L residual fix | Implementer | Fixed loss + `NeuralRanders` + `AVBDSolver` | Test in `tests/` | Passes with residual < 1e-3 |
| 1.1 | Implement `ArrivalTimeLoss` | Implementer | Mini-spec above; [src/ham/training/losses.py](src/ham/training/losses.py) | New loss class | Correct gradients verified |
| 1.2 | Implement `SyntheticEikonalDataGenerator` | Implementer | Phase 1 spec; Gahtan code for GT | Data generator script in `examples/` | Generates 80×80 arrival time fields |
| 1.3 | Phase 1 experiment script | Implementer | Phase 1 full spec | `examples/experiment_gahtan_phase1.py` | Produces all Phase 1 figures |
| 2.1 | Phase 2 experiment script | Implementer | Phase 2 spec | `examples/experiment_gahtan_phase2.py` | Produces all Phase 2 figures |
| 3.1 | Phase 3 experiment script | Implementer | Phase 3 spec | `examples/experiment_gahtan_phase3.py` | Produces all Phase 3 figures |
| 4.1 | Implement `CovariateConditionedRanders` | Implementer | Mini-spec above | New metric class in `src/ham/models/` | Passes shape + feasibility tests |
| 4.2 | Implement Sim2Real-Fire data loader | Implementer | Dataset spec; Gahtan's loader as ref | `src/ham/data/wildfire.py` or `examples/` | Loads all 15 scenes |
| 4.3 | Implement soft geodesic fan assignment | Implementer | Math Reviewer WARNING fix | Function in `src/ham/solvers/` | Differentiable, matches hard assignment |
| 4.4 | Phase 4 experiment script | Implementer | Phase 4 full spec | `examples/experiment_gahtan_phase4.py` | Produces all Phase 4 figures |
| 5.1 | Implement DEM-to-mesh pipeline | Implementer | Phase 5 spec | Utility in `src/ham/utils/` or `examples/` | Converts raster DEM to `TriangularMesh` |
| 5.2 | Phase 5 experiment script | Implementer | Phase 5 spec | `examples/experiment_gahtan_phase5.py` | Produces all Phase 5 figures |
| 6.1 | Phase 6 analysis scripts | Implementer | Phase 6 spec | `examples/experiment_gahtan_phase6.py` | Produces all Phase 6 figures |
| 7.1 | Publication figure compilation | Implementer | All phase outputs | `examples/plot_gahtan_publication.py` | PDF figures with consistent styling |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| AVBD converges to local minimum (wrong geodesic) | Medium | High — incorrect distances | Multi-start initialization (3 random starts, take minimum energy). Also: compare with ExponentialMap shooting for validation. |
| AVBD too slow for Phase 4 (K=500 BVPs per fire, 16 fires/batch) | Medium | High — impractical training | Use graph-coloring parallel AVBD (`parallel=True`). Reduce K. Use Strategy C (shooting fan) as fallback. JAX `vmap` over observations. |
| Sim2Real-Fire data format incompatible with HAMTools | Low | Medium — delays Phase 4 | Gahtan's code provides PyTorch data loaders; adapt to NumPy/JAX. Validate on 1 scene before full experiment. |
| Metric recovery worse than Gahtan by >3× | Medium | Medium — weakens paper narrative | Frame honestly as "Lagrangian approaches trade speed for geometric richness." Focus narrative on Phases 5–6 extensions. |
| 3D mesh AVBD numerically unstable | Medium | Medium — blocks Phase 5 | HAMTools already tests `TriangularMesh` + `AVBDSolver` (see `tests/test_mesh_solver.py`). Start with low-res mesh. |
| EulerLagrangeResidualLoss fix breaks existing experiments | Low | Medium | Run existing test suite before and after fix. The formula change only affects Randers metrics with $\|W\| > 0$. |
| Caustics in shooting fan (Strategy C) create coverage gaps | Medium | Low — affects Phase 4 only | Increase $N_\theta$. Use soft assignment to handle shadow regions. Fall back to Strategy A. |
| Insufficient GPU memory for vmapped BVP solves | Medium | Medium | Use `eqx.filter_checkpoint` (already used for `_solve_and_integrate`). Reduce batch size. Use gradient accumulation. |

---

## Summary: What HAMTools Provides Beyond Replication

The core scientific contribution is demonstrating that the **Lagrangian geodesic formulation is a viable alternative to Eulerian eikonal solvers** for Randers metric learning, with clear tradeoffs:

| Dimension | Eikonal (Gahtan) | Lagrangian (HAMTools) |
|-----------|------------------|-----------------------|
| **Speed** | $O(n)$ for full distance field | $O(K \cdot T \cdot I)$ per observation |
| **Output** | Arrival times only | Arrival times + geodesic paths |
| **Domain** | Flat 2D Cartesian grids | Arbitrary manifolds, meshes, dimensions |
| **Backward** | Implicit differentiation ($O(n)$) | JAX autodiff through solver |
| **Geometry** | Distance only | Distance + curvature + transport + stability |
| **Extensions** | Limited to $\mathbb{R}^2$ | 3D DEM surfaces, high-dimensional latent spaces |

The **novel contributions** not achievable with the eikonal approach:
1. **Geodesic fire corridors** — actual physical paths of fastest fire spread
2. **Stability analysis** — Jacobi field / geodesic bundle divergence
3. **Curvature anomaly maps** — terrain features causing abrupt fire behavior changes
4. **3D terrain surface** — fire propagation on the actual DEM, not its flat projection
5. **Berwald parallel transport** — how fire direction rotates along the geodesic
