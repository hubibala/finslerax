# Science Audit: Comparative Analysis — HAMTools vs. Gahtan et al. (2026)
**Auditor:** Science Auditor Agent  
**Date:** May 16, 2026  
**Scope:** Framework capability comparison (excluding bio/Weinreb domain)

**Reference Paper:** Gahtan, Shpund & Bronstein. *Wildfire Simulation with Differentiable Randers-Finsler Eikonal Solvers.* arXiv:2603.00035 [cs.CE], Feb 2026.  
**Reference Code:** [BarakGahtan/differentiable-eikonal-wildfire](https://github.com/BarakGahtan/differentiable-eikonal-wildfire) (PyTorch + Numba/CUDA)

---

## Summary

Both HAMTools and Gahtan et al. operate in the Randers–Finsler geometric framework and solve the same core mathematical problem: learning spatially varying asymmetric metrics from data. However, they target **orthogonal computational regimes** and make fundamentally different solver design choices. HAMTools is a general-purpose *intrinsic* Finsler geometry library (geodesic spray, parallel transport, curvature); Gahtan et al. is a purpose-built *extrinsic* eikonal distance-field solver on Cartesian grids. The frameworks are **complementary rather than competing**, and HAMTools could adopt key ideas from the Gahtan solver to fill current gaps.

**Overall verdict:** HAMTools cannot currently replicate Gahtan et al.'s results (grid-scale distance fields via fast sweeping), but possesses strictly richer geometric machinery that Gahtan's framework lacks entirely. Neither subsumes the other.

---

## 1. Shared Mathematical Foundation

Both frameworks are built on the **Randers metric** $F(x,v) = \sqrt{v^\top M(x) v} + \beta(x) \cdot v$, combining a Riemannian base metric $M$ (or $\mathbf{G}$) with a drift/wind vector $\beta$ (or $\mathbf{b}$). Both enforce the feasibility constraint $\|\beta\|_{M^{-1}} < 1$ to guarantee positive-definiteness and well-posed propagation.

| Aspect | HAMTools | Gahtan et al. |
|--------|----------|---------------|
| Metric type | General Finsler (Randers as special case) | Randers only |
| Parameterization | Zermelo navigation $(h, W)$ | Direct $(G, b)$ with eigenvalue clamping |
| Feasibility enforcement | $\tanh$ gating on $\|W\|_h$ | Differentiable projection: eigenvalue clamping + norm rescaling |
| Dimensionality | Arbitrary $d$ (tested $d=2,3,50$) | $d = 2$ only (Cartesian grid) |

**STRONG:** Both use differentiable feasibility projections, which is essential for gradient-based learning. HAMTools's Zermelo parameterization is mathematically cleaner (it is the natural parameterization of the Randers family), while Gahtan's direct parameterization is more efficient for 2D grids.

---

## 2. Claims Audit

### Claim 1: "HAMTools can compute Randers geodesic distances"
- **Evidence provided:** AVBD boundary-value solver and RK4-based exponential map (IVP) in `src/ham/solvers/`. Validated on spheres, tori, hyperboloids, and triangular meshes.
- **Gahtan comparison:** Gahtan computes *distance fields* (arrival times at all grid points from a source) via the eikonal equation. This is a fundamentally different computational primitive — it solves for the full distance function $T(x)$ simultaneously, not individual geodesic paths.
- **Verdict:** **MISSING**
- **Recommendation:** HAMTools currently has no eikonal solver. It can compute individual geodesics (point-to-point) but cannot efficiently compute full distance fields. For applications requiring global distance information (e.g., Voronoi diagrams, level-set propagation, wavefront simulation), an eikonal solver component would be needed.

### Claim 2: "Gahtan's fast sweeping achieves $O(n)$ complexity per iteration"
- **Evidence provided:** Convergence in 2–3 iterations independent of grid size, confirmed empirically (Table 1, Table 10 in the paper). Sub-1% error across isotropic, anisotropic, and Randers configurations.
- **HAMTools context:** HAMTools's AVBD solver is $O(T \cdot K)$ per geodesic, where $T$ = path discretization and $K$ = iteration count. Computing the equivalent of a distance field would require $O(n)$ independent solves from a single source, yielding $O(n \cdot T \cdot K)$ total — orders of magnitude slower than eikonal fast sweeping.
- **Verdict:** **WEAKNESS** (for HAMTools, in the distance-field regime)
- **Recommendation:** If HAMTools ever needs distance fields on grids, integrating a JAX-native fast sweeping module would close this gap. The mathematical formulation is compatible.

### Claim 3: "HAMTools provides richer geometric structure"
- **Evidence provided:** Geodesic spray (auto-differentiated from energy), Berwald parallel transport, curvature tensor, holonomy computation — all absent from Gahtan.
- **Gahtan comparison:** Gahtan computes only arrival times and their gradients w.r.t. metric parameters. There is no notion of parallel transport, curvature, or intrinsic geometric structure beyond the distance function.
- **Verdict:** **STRONG** (for HAMTools)
- **Key advantage breakdown:**

| Geometric capability | HAMTools | Gahtan |
|----------------------|----------|--------|
| Geodesic spray $G^i(x,v)$ | ✓ (auto-diff from energy) | ✗ |
| Berwald parallel transport | ✓ | ✗ |
| Curvature tensor | ✓ | ✗ |
| Holonomy computation | ✓ | ✗ |
| Exponential/logarithmic map | ✓ | ✗ |
| Distance fields (eikonal) | ✗ | ✓ |
| Implicit differentiation of PDE | ✗ | ✓ |
| Triangulated mesh support | ✓ | ✗ (Cartesian grid only) |
| Arbitrary manifold topology | ✓ | ✗ (flat $\mathbb{R}^2$ only) |

### Claim 4: "Gahtan's implicit differentiation avoids unrolling"
- **Evidence provided:** Backward pass solves a sparse lower-triangular adjoint system via single reverse-time back-substitution (Algorithm 2 in the paper). This gives exact gradients at non-stencil-boundary points and avoids storing intermediate solver states.
- **HAMTools context:** HAMTools relies on JAX's standard automatic differentiation through its ODE integrator (RK4 steps in `ExponentialMap`). This is unrolled differentiation — memory scales linearly with the number of integration steps.
- **Verdict:** **WEAKNESS** (for HAMTools in scaling scenarios)
- **Recommendation:** For large-scale problems, HAMTools could benefit from adjoint-method ODE differentiation (e.g., `diffrax`'s adjoint methods) rather than naive unrolling. This is orthogonal to the eikonal question but addresses the same memory concern.

### Claim 5: "Gahtan recovers spatially varying metrics from sparse observations"
- **Evidence provided:** Isotropic metric recovery to 5.6% error, drift fields to <3% error, multi-source configurations reducing error by 45% (Section 5 of the paper). Validated on synthetic inverse problems with known ground truth.
- **HAMTools context:** HAMTools learns metrics end-to-end through trajectory data (lineage triples in the bio domain, point-pair geodesics in the geometry domain). The inverse problem formulation is different: HAMTools observes *paths*, Gahtan observes *arrival times*. Both are valid supervision signals for metric learning.
- **Verdict:** **NOTE** — Different inverse problem formulations; neither is strictly stronger.
- **Recommendation:** HAMTools could add an arrival-time loss component (eikonal residual minimization) alongside its existing trajectory-based losses, if an eikonal solver were integrated.

### Claim 6: "Gahtan's encoder architecture maps covariates to Randers parameters"
- **Evidence provided:** `FlexibleCovariateEncoder` combining a global MLP (weather) with a fully-convolutional network (spatial covariates), outputting $(g_{11}, g_{12}, g_{22}, b_1, b_2)$ per pixel.
- **HAMTools context:** HAMTools uses `PSDMatrixField` (for $h$) and `VectorField` (for $W$) networks in `NeuralRanders`. The `PullbackRanders` variant derives $h$ from a decoder's Jacobian (pullback metric) and learns only $W$. The `DataDrivenPullbackRanders` variant further conditions on observed data covariates.
- **Verdict:** **STRONG** (for HAMTools) — HAMTools's pullback approach is more geometrically principled: the Riemannian component is derived from the data manifold's intrinsic geometry rather than learned freely, which provides stronger inductive bias.

---

## 3. Key Architectural Differences

### 3.1 Domain Representation

| | HAMTools | Gahtan |
|---|---|---|
| **Domain** | Smooth manifolds ($S^2$, $\mathbb{H}^2$, torus, meshes, $\mathbb{R}^d$) | Regular Cartesian grids in $\mathbb{R}^2$ |
| **Discretization** | Continuous ODE + point samples | Fixed grid with $N \times N$ resolution |
| **Topology** | Arbitrary (handled by `Manifold` ABC) | Flat, simply connected |

**Implication:** HAMTools can model problems on curved spaces where eikonal solvers on flat grids would require parameterization and suffer from chart singularities. Gahtan's approach is far more efficient for problems naturally posed on flat 2D domains.

### 3.2 Solver Strategy

| | HAMTools | Gahtan |
|---|---|---|
| **Forward problem** | ODE integration (RK4) of geodesic spray | PDE solve (fast sweeping of eikonal equation) |
| **Output** | Single geodesic path $\gamma(t)$ | Full distance field $T(x)$ |
| **Backward pass** | JAX autodiff through ODE steps | Implicit differentiation of converged PDE |
| **Complexity (single geodesic)** | $O(T)$ steps | $O(n)$ for full field (amortized per query) |
| **Framework** | JAX + Equinox | PyTorch + Numba + CUDA |

### 3.3 Backward Pass

Gahtan's key innovation is the **implicit differentiation** of the converged eikonal solution. The Jacobian $\partial R / \partial T$ is sparse and lower-triangular, enabling $O(n)$ adjoint solve without storing the forward computation graph. HAMTools uses standard AD through its ODE solver, which is simpler but less memory-efficient for long integration horizons.

---

## 4. Can HAMTools Reproduce Gahtan's Results?

**Short answer: No, not currently. But the gap is narrow and bridgeable.**

### What would be needed:

1. **Eikonal solver module.** A JAX-native fast-sweeping or fast-marching solver for the Randers eikonal equation on Cartesian grids. The mathematical formulation is already present in HAMTools (the Randers metric), only the PDE solver is missing. This would be a new module under `src/ham/solvers/eikonal.py`.

2. **Implicit differentiation backend.** JAX supports custom VJP rules (`jax.custom_vjp`), which could implement the adjoint-based backward pass described in Gahtan Section 3.3 without requiring PyTorch or Numba.

3. **Grid-based metric representation.** HAMTools currently represents metrics as continuous functions (neural networks or analytical). A grid-based representation (per-pixel metric parameters) would be needed for direct comparison with Gahtan's inverse problem experiments.

### What HAMTools already has that Gahtan lacks:

1. **General manifold support.** Gahtan is limited to $\mathbb{R}^2$ grids. HAMTools can solve on spheres, hyperboloids, tori, and triangulated meshes.

2. **Full differential geometry stack.** Parallel transport, curvature, and holonomy enable analyses that distance fields alone cannot provide (e.g., stability of geodesic bundles, conjugate points, Jacobi fields).

3. **General Finsler support.** HAMTools is not limited to the Randers family. Any Finsler metric definable via an energy function $E(x,v)$ is supported. Gahtan is restricted to the Randers class $F = \sqrt{v^\top G v} + b \cdot v$.

4. **Higher-dimensional support.** HAMTools operates in arbitrary dimension $d$, while Gahtan's fast sweeping with 8 triangular stencils is specific to 2D.

5. **Mesh support.** `TriangularMesh` + `DiscreteRanders` enables metric learning on non-flat surfaces — precisely the limitation Gahtan acknowledges in their conclusion.

---

## 5. Advantages and Disadvantages

### HAMTools Advantages Over Gahtan

| Advantage | Significance |
|-----------|-------------|
| General Finsler metrics (not just Randers) | High — enables exploration of non-Randers asymmetric geometries |
| Arbitrary manifold topology | High — essential for applications on curved spaces |
| Full differential geometry (transport, curvature) | Medium — needed for stability analysis, not needed for distance computation |
| Mesh support | High — Gahtan explicitly lists this as future work |
| Higher dimensions ($d > 2$) | High — enables latent-space applications |
| Pullback metric from decoder | Medium — stronger geometric inductive bias for learning |
| Berwald connection (vs. none) | Medium — provides rigorous parallel transport |

### Gahtan Advantages Over HAMTools

| Advantage | Significance |
|-----------|-------------|
| $O(n)$ distance field computation (full grid) | **Critical** — enables wavefront and propagation applications |
| Implicit differentiation (memory-efficient backward) | High — avoids storing ODE integration history |
| GPU-batched solver (CUDA) | High — 2778× speedup for batch training |
| Proven inverse problem capability | High — 5.6% metric recovery, <3% drift recovery |
| Real-world application validation (wildfire) | High — demonstrates practical utility |
| Comprehensive computational benchmarking | Medium — 1400× speedup over Jacobi, 755× over finite differences |
| Temporal band decomposition training | Medium — elegant curriculum strategy for long-horizon PDE |
| Cross-scene generalization evidence | Medium — 0.766 correlation on unseen geographic regions |

### Shared Strengths

- Differentiable Randers feasibility enforcement
- End-to-end metric learning from data
- Neural network → metric parameter mapping

---

## 6. Opportunities for HAMTools

### 6.1 Adopt the Eikonal Solver (Priority: High)

Gahtan's core contribution (fast sweeping + implicit diff for Randers eikonal) could be implemented as a JAX module in HAMTools. Since JAX's `custom_vjp` mechanism is arguably cleaner than PyTorch's custom autograd functions, a JAX-native implementation might be more elegant. The 2D stencil geometry would need generalization for HAMTools's mesh and surface support.

### 6.2 Extend to Triangulated Surfaces (Gahtan's Stated Limitation)

Gahtan explicitly acknowledges (Section 7): *"Extension to triangulated surfaces would replace the eight-stencil system with triangle-fan neighborhoods."* HAMTools already has `TriangularMesh` with `DiscreteRanders`. Adding an eikonal solver for triangulated meshes would directly address Gahtan's stated limitation and represent a novel contribution beyond both current frameworks.

### 6.3 Adjoint ODE Differentiation

Independent of the eikonal solver, HAMTools's geodesic ODE solver would benefit from adjoint-method differentiation (as in Neural ODEs, Chen et al. 2018 — already cited by Gahtan). Libraries like `diffrax` provide this natively in JAX. This would match the memory efficiency of Gahtan's implicit differentiation for the geodesic case.

---

## 7. Literature Context

- **Mirebeau (2014):** Efficient fast marching with Finsler metrics (arXiv:1406.1233). The algorithmic ancestor of Gahtan's approach. HAMTools does not implement any Mirebeau-style algorithms.
- **Qian, Zhang & Zhao (2007):** Fast sweeping on triangular meshes. This is the natural extension point for HAMTools to bridge the gap.
- **Chen et al. (2018):** Neural ODEs (arXiv:1806.07366). The adjoint method used there would benefit HAMTools's geodesic solver.
- **Crane, Weischedel & Wardetzky (2017):** The heat method for distance computation. An alternative to eikonal solvers that could leverage HAMTools's existing Laplacian infrastructure.

---

---

## 8. Can HAMTools Solve the Same Problem via a Different Approach?

**Yes.** The underlying scientific problem is identical: *learn a spatially varying Randers metric $(h, W)$ from observed arrival-time data*. The eikonal equation and geodesic ODE are two views of the same mathematics — the eikonal is the Hamilton–Jacobi equation whose characteristics are exactly the Randers geodesics. The arrival time $T(x)$ at a point $x$ is the geodesic distance $d_F(\text{source}, x)$. Any machinery that computes Finsler distances and is differentiable can, in principle, solve the inverse problem.

HAMTools already has all the necessary components. Below is a concrete experimental design.

### 8.1 The Mathematical Equivalence

The Randers eikonal equation solved by Gahtan:

$$(\nabla T - b)^\top G^{-1} (\nabla T - b) = 1$$

is the **static Hamilton–Jacobi equation** whose characteristic ODE is exactly the geodesic spray equation that HAMTools integrates:

$$\ddot{x}^i + 2G^i(x, \dot{x}) = 0$$

The solutions are related by $T(x) = \inf_\gamma \int_0^1 F(\gamma, \dot\gamma)\, dt$ where the infimum is over paths from the source to $x$. This is precisely what `FinslerMetric.arc_length()` computes on a geodesic path, and what `AVBDSolver.solve()` finds by optimizing the discrete energy.

### 8.2 Proposed Approach: Lagrangian Metric Learning

Instead of solving the PDE on a grid (Eulerian), HAMTools can solve the inverse problem via a **Lagrangian** (particle-tracking) approach:

**Pipeline:**
1. **Encoder:** A covariate-conditioned `NeuralRanders` or `DataDrivenPullbackRanders` maps environmental covariates → Zermelo data $(h(x), W(x))$. HAMTools already has `PSDMatrixField` and `VectorField` for this.

2. **Forward model:** For each observation point $x_i$ with observed arrival time $T_i^{\text{obs}}$, compute the predicted Finsler distance from the source:
   - **Option A (BVP):** Use `AVBDSolver.solve(metric, source, x_i)` → `metric.arc_length(trajectory)` = $T_i^{\text{pred}}$
   - **Option B (IVP/Shooting):** Use `ExponentialMap.shoot(metric, source, v_0)` with optimized initial direction $v_0$
   - **Option C (Direct energy):** Use `FinslerActionMatchingLoss`-style evaluation: $T_i^{\text{pred}} \approx F(x_{\text{source}}, x_i - x_{\text{source}})$ for nearby points

3. **Loss:** MSE between predicted and observed arrival times:
   $$\mathcal{L} = \frac{1}{|\Omega_{\text{obs}}|} \sum_{i \in \Omega_{\text{obs}}} (T_i^{\text{pred}} - T_i^{\text{obs}})^2 + \lambda \mathcal{R}(h, W)$$

4. **Backward:** Standard JAX autodiff through the geodesic solver.

### 8.3 Components Already Available in HAMTools

| Required component | HAMTools equivalent | Status |
|---|---|---|
| Randers metric with feasibility | `Randers` in `geometry/zoo.py` (Zermelo parameterization) | ✓ Ready |
| Neural metric from covariates | `NeuralRanders`, `DataDrivenPullbackRanders` in `models/learned.py` | ✓ Ready |
| PSD matrix field network | `PSDMatrixField` in `nn/networks.py` | ✓ Ready |
| Wind vector field network | `VectorField` in `nn/networks.py` | ✓ Ready |
| Geodesic distance (BVP) | `AVBDSolver.solve()` → `metric.arc_length()` | ✓ Ready |
| Geodesic shooting (IVP) | `ExponentialMap.shoot()` | ✓ Ready |
| Euler–Lagrange residual loss | `EulerLagrangeResidualLoss` in `training/losses.py` | ✓ Ready |
| Direct Finsler energy loss | `FinslerActionMatchingLoss` in `training/losses.py` | ✓ Ready |
| Flow matching loss | `FinslerianFlowMatchingLoss` in `training/losses.py` | ✓ Ready |
| Multi-phase training pipeline | `HAMPipeline` in `training/pipeline.py` | ✓ Ready |
| Metric smoothness regularization | `MetricSmoothnessLoss` in `training/losses.py` | ✓ Ready |
| Metric anchor regularization | `MetricAnchorLoss` in `training/losses.py` | ✓ Ready |
| 2D flat domain manifold | `EuclideanSpace` in `geometry/surfaces.py` | ✓ Ready |
| Triangulated mesh domain | `TriangularMesh` in `geometry/mesh.py` | ✓ Ready |

**What's missing (minor, all implementable):**
- An `ArrivalTimeLoss` component wrapping the geodesic distance computation with MSE against observed times
- A covariate-conditioned encoder (like Gahtan's CNN+MLP) — HAMTools's `VectorField` and `PSDMatrixField` accept point coordinates but not auxiliary feature maps; a thin wrapper would be needed
- Total variation regularization on the metric field (HAMTools has `MetricSmoothnessLoss` via Jacobian penalty, which is similar but uses $L_2$ on the Jacobian rather than TV)

### 8.4 Three Concrete Attack Strategies

#### Strategy A: Sparse Geodesic BVP (most direct)

Sample $K$ observation points per fire, solve $K$ BVPs from source to each point, compute arc lengths, and compare to observed arrival times.

```
For each training fire:
    source = ignition point
    {x_i, T_i^obs} = sampled observation points with arrival times
    For each x_i (vmapped):
        traj = AVBDSolver.solve(metric, source, x_i, n_steps=15)
        T_i^pred = metric.arc_length(traj.xs)
    Loss = MSE(T^pred, T^obs) + λ_TV * MetricSmoothnessLoss
```

**Pros:** Most mathematically rigorous; directly solves the inverse geodesic problem. Fully differentiable through AVBD.
**Cons:** $O(K)$ BVP solves per fire. At $K=100$ with 15-step AVBD at 50 iterations, this is ~75,000 metric evaluations per fire per training step. Slower than Gahtan's $O(n)$ eikonal solve, but feasible with JAX `vmap` parallelism.

#### Strategy B: Euler–Lagrange Residual (solver-free)

HAMTools already has `EulerLagrangeResidualLoss`, which avoids solving any ODE. Instead, given a candidate path (e.g., the straight line from source to observation in the 2D domain), it evaluates the Euler–Lagrange residual $R = \frac{d}{dt}\frac{\partial L}{\partial v} - \frac{\partial L}{\partial x}$. If the metric is correct, the minimum-time path should have $R \approx 0$.

**Adaptation for wildfire:** Construct candidate "wavefront rays" from the source at many angles. For each ray, sample points and evaluate whether the Euler–Lagrange residual is zero (indicating the ray is a geodesic) and whether the integrated cost matches the observed arrival time.

**Pros:** No iterative solver in the forward pass — pure pointwise evaluation. Very fast per iteration. Already implemented.
**Cons:** Requires good candidate paths. The straight-line approximation deteriorates in strongly inhomogeneous media.

#### Strategy C: Geodesic Shooting Fan (Lagrangian wavefront)

Shoot geodesics from the source in $N_\theta$ uniformly spaced directions using `ExponentialMap`. This traces out the Lagrangian wavefront — the set of points reached by geodesics at each time. The arrival time at each grid point is determined by which geodesic reaches it first.

```
For n_theta directions:
    v_0 = (cos θ, sin θ) * speed
    trajectory = ExponentialMap.full_trajectory(metric, source, v_0)
    # Each point on the trajectory has arrival time t
For each observation (x_i, T_i^obs):
    T_i^pred = arrival time from nearest geodesic ray
Loss = MSE(T^pred, T^obs)
```

**Pros:** Amortizes computation — $N_\theta$ geodesic solves cover the whole domain. Produces actual geodesic paths (which the eikonal solver does not). Natural for JAX `vmap`.
**Cons:** Rays may miss parts of the domain (caustic/shadow regions). Interpolation between rays introduces approximation. Needs sufficient angular resolution.

### 8.5 Expected Performance Comparison

| Aspect | Gahtan (Eikonal) | HAMTools Strategy A (BVP) | HAMTools Strategy B (E-L Residual) | HAMTools Strategy C (Shooting Fan) |
|--------|------------------|--------------------------|-----------------------------------|------------------------------------|
| Forward cost per fire | $O(n)$ grid points | $O(K \cdot T \cdot I)$ per obs point | $O(K)$ pointwise evals | $O(N_\theta \cdot T)$ ray steps |
| Backward cost | $O(n)$ adjoint | JAX AD through solver | JAX AD (cheap, no solver) | JAX AD through ODE |
| Memory | $O(n)$ (implicit diff) | $O(K \cdot T)$ | $O(K)$ | $O(N_\theta \cdot T)$ |
| Accuracy ceiling | Sub-1% (verified) | Same (converges to true geodesic) | Depends on path quality | Depends on angular resolution |
| Mesh/surface support | ✗ | ✓ | ✓ | ✓ |
| Higher dimensions | ✗ ($d=2$ only) | ✓ | ✓ | ✓ (but ray count grows as $O(N_\theta^{d-1})$) |
| Produces geodesic paths | ✗ (only distances) | ✓ | ✗ | ✓ |

**Realistic assessment:** For the wildfire problem on 2D grids at Gahtan's scale (300×300), Strategy A with $K \sim 200$ sparse observations and JAX `vmap` is likely **10–50× slower** per training step than the eikonal solver, but entirely feasible on modern hardware (seconds vs. milliseconds per fire). Strategy B is competitive in speed but may underperform in accuracy for strongly heterogeneous media.

### 8.6 What This Buys Beyond Replication

Solving Gahtan's problem with HAMTools's Lagrangian approach would not just replicate — it would enable capabilities the eikonal framework fundamentally cannot provide:

1. **Geodesic path visualization.** HAMTools produces the actual fire spread paths (geodesics), not just arrival times. These trace the fastest-burn corridors and are directly interpretable for firefighting resource allocation.

2. **Stability analysis via Jacobi fields.** The Berwald parallel transport can assess whether nearby geodesics diverge (unstable spread) or converge (channeled spread). This maps to fire behavior concepts like "spotting risk" and "firebreak effectiveness."

3. **Curvature-based anomaly detection.** Regions of high Finslerian curvature correspond to abrupt changes in fire behavior — useful for identifying terrain features that dramatically alter spread dynamics.

4. **Mesh extension (Gahtan's stated limitation).** Running the same experiment on a DEM triangulated mesh rather than a flat grid would immediately address the limitation Gahtan identifies in their conclusion and would constitute a novel contribution.

5. **Higher-dimensional covariates.** The metric can be defined in a latent space that jointly encodes terrain, fuel, and weather — a pullback metric from a covariate encoder. This is more expressive than Gahtan's direct pixel-wise parameterization.

### 8.7 Recommended Experiment Plan

**Phase 1 — Synthetic replication (validates approach):**
- Reproduce Gahtan's Section 5 inverse problem: piecewise constant isotropic metric on 80×80 grid.
- Use `EuclideanSpace(2)` manifold + `NeuralRanders` metric.
- Training via Strategy A (sparse BVP) with $K=200$ observation points.
- Target: match Gahtan's 5.6% metric recovery error.
- Deliverable: direct quantitative comparison.

**Phase 2 — Wildfire on flat grid (head-to-head comparison):**
- Use Sim2Real-Fire dataset (same as Gahtan).
- Covariate encoder → `NeuralRanders` parameters.
- Training via Strategy A or C.
- Target: compare correlation/RMSE/IoU against Gahtan's 0.824 within-scene correlation.
- Report runtime overhead vs. eikonal approach.

**Phase 3 — Wildfire on triangulated DEM (novel contribution):**
- Convert terrain elevation grid to `TriangularMesh` + `DiscreteRanders`.
- Same covariate encoder, same loss, but on the actual terrain surface.
- This is something Gahtan **cannot do** and explicitly lists as future work.
- Hypothesis: modeling fire spread on the actual 3D terrain surface (rather than a flat 2D projection) improves accuracy, particularly on steep terrain.

---

## 9. Conclusion

**The same inverse problem is fully approachable with both frameworks.** The eikonal equation and geodesic ODE are dual formulations (Eulerian vs. Lagrangian) of the same Randers distance computation. HAMTools already possesses every mathematical component needed to solve Gahtan's wildfire metric recovery problem — `NeuralRanders` for the metric, `AVBDSolver`/`ExponentialMap` for distance computation, differentiable losses, and multi-phase training.

The key tradeoff is **speed vs. generality:**
- Gahtan's eikonal solver is purpose-built for 2D grids and will be faster per iteration.
- HAMTools's geodesic-based approach is slower but works on arbitrary manifolds, in arbitrary dimensions, and produces richer geometric output (paths, curvature, transport).

**As a baseline comparison, this is highly viable.** The recommended path is Phase 1 (synthetic inverse problem, match Gahtan's 5.6% error) followed by Phase 2 (Sim2Real-Fire head-to-head). Phase 3 (mesh extension) would be the novel contribution that no existing framework can deliver.
