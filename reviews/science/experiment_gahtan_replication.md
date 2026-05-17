# Science Audit: Experiment Plan — Gahtan et al. (2026) Replication & Extension via HAMTools Lagrangian Approach

**Auditor:** Science Auditor Agent  
**Date:** May 17, 2026  
**Scope:** Five-phase experiment plan for reproducing and extending arXiv:2603.00035 using HAMTools geodesic (Lagrangian) methods instead of eikonal (Eulerian) methods.

**Reference Paper:** Gahtan, Shpund & Bronstein. *Wildfire Simulation with Differentiable Randers-Finsler Eikonal Solvers.* arXiv:2603.00035 [cs.CE], Feb 2026.  
**Reference Code:** [BarakGahtan/differentiable-eikonal-wildfire](https://github.com/BarakGahtan/differentiable-eikonal-wildfire) (MIT License)  
**Dataset:** Sim2Real-Fire (Li et al., NeurIPS 2024) — [TJU-IDVLab/Sim2Real-Fire](https://github.com/TJU-IDVLab/Sim2Real-Fire)

---

## Summary

**Overall scientific rigor assessment: needs major revision.**

The experiment plan is scientifically well-motivated and addresses a genuine gap (Lagrangian vs. Eulerian Randers distance computation). However, it contains a **critical observation-density confound** in Phase 1 that would invalidate the primary synthetic comparison, **lacks statistical rigor** throughout (no seeds, no variance, no significance tests), **underspecifies key ablations**, and makes **overly optimistic performance targets** for the Lagrangian approach given the known density–accuracy phase transition in Gahtan's inverse problem. Phase 5 (mesh extension) is a strong novel contribution but requires a clearer experimental hypothesis and control design.

---

## Claims Audit

### Claim 1: "Phase 1 can match Gahtan's 5.6% metric recovery error with K=200 sparse BVP observations on an 80×80 grid"

- **Evidence provided:** Plan states K=200 observation points sampled from 80×80 = 6,400 grid, using AVBD arc length as the distance signal, with MSE loss against ground truth arrival times.
- **Literature context:** Gahtan Section 5 and Appendix I report the 5.6% figure for **100% pixel observation** (all 6,400 grid points). When observation density is reduced to 7% (≈449 points on 80×80), error jumps to **21.2%**. Figure 32 (Appendix I) shows a clear **phase transition near 50% observation density**: below ~50%, error plateaus at ~21%; above, it saturates near 6%. K=200 on an 80×80 grid is **3.1% observation density**, firmly in the underdetermined regime.
- **Verdict:** **CRITICAL**
- **Recommendation:**
  1. The target of ≤10% error at 3.1% density is inconsistent with Gahtan's own observation-density curve, which shows ~21% error even at 7% density. Either increase K to at least 3,200 (50%) or **explicitly acknowledge the density limitation** and compare against Gahtan's sparse-observation results (21.2% at 7%), not the full-observation result (5.6%).
  2. Run a density sweep: K ∈ {50, 100, 200, 500, 1000, 3200, 6400} to characterize the Lagrangian approach's own phase transition and compare it to Gahtan's Figure 32.
  3. At matched density (e.g., K=449 ≈ 7% for both approaches), the fair comparison is Lagrangian error vs. 21.2%, not vs. 5.6%.

---

### Claim 2: "Phase 2 drift recovery target of ≤5% is achievable (Gahtan achieves <3%)"

- **Evidence provided:** Plan proposes recovering constant drift $\mathbf{b} = (0.15, 0.08)$ with known metric $\mathbf{G} = \mathbf{I}$.
- **Literature context:** Gahtan achieves 2.8% error for $b_1$ and 2.0% for $b_2$ (Appendix I), again with full-grid observation. Drift creates a stronger observational signature (characteristic upwind/downwind asymmetry), so the density sensitivity is less severe than for metric recovery. The ≤5% target is plausible but has the same density caveat.
- **Verdict:** **WARNING**
- **Recommendation:** Specify observation density explicitly. If using K=200 at 3.1% density, the comparison must be against Gahtan at matched density, not full-grid. Drift recovery may be less density-sensitive, but this should be empirically verified, not assumed.

---

### Claim 3: "Phase 3 multi-source (3–5 ignition points) combined recovery demonstrates convergence, not necessarily matching Gahtan"

- **Evidence provided:** The plan appropriately tempers expectations by not targeting specific error numbers.
- **Literature context:** Gahtan Table 3 shows multi-source recovery reducing error from 18.6% (1 source) to 10.1% (5 sources) at 7% density per source. The improvement exceeds additive observation count gains, suggesting complementary geometric constraints.
- **Verdict:** **NOTE**
- **Recommendation:** Good that targets are not overpromised. However, define explicit convergence criteria: (a) metric field error decreasing monotonically with source count, (b) loss stabilizing below a threshold, (c) recovered metric field qualitatively matching the ground truth spatial structure. Without criteria, "demonstrates convergence" is unfalsifiable.

---

### Claim 4: "Phase 4 can achieve correlation ≥ 0.70 on Sim2Real-Fire (Gahtan: 0.824)"

- **Evidence provided:** Strategy A (K=200 BVP per fire) or Strategy C (N_θ=64 geodesic shooting fan). The plan does not distinguish per-scene (Section 6.1 in Gahtan) from cross-scene (Section 6.2) evaluation.
- **Literature context:**
  - Gahtan per-scene: correlation 0.824 ± 0.044, using **full-grid** distance fields with grids ranging from 132×190 to 574×770.
  - Gahtan cross-scene: correlation 0.766 on 4 held-out scenes.
  - The Sim2Real-Fire grid sizes far exceed 80×80. At 300×300 = 90,000 pixels, K=200 is 0.22% observation density. At 574×770 ≈ 442,000 pixels, it is 0.045%.
  - Strategy C with N_θ=64 shoots 64 geodesics. Even with 100 time steps each, this produces ~6,400 trajectory points — roughly 7% of a 300×300 grid, and coverage is non-uniform (dense near source, sparse far away).
- **Verdict:** **CRITICAL**
- **Recommendation:**
  1. K=200 at the Sim2Real-Fire scale is catastrophically sparse. The Lagrangian approach will not see enough of the domain to learn meaningful metric fields. Either substantially increase K (≥5,000 per fire) or use Strategy C with much higher N_θ (≥256) and address the interpolation/coverage problem.
  2. Clearly distinguish per-scene and cross-scene evaluation protocols. These are different experimental conditions with different baselines (0.824 vs. 0.766).
  3. The 0.70 target needs justification beyond "it's lower than Gahtan." Is there a theoretical bound on how much accuracy is lost by sparse Lagrangian sampling vs. full-grid eikonal? If not, the target is arbitrary.
  4. Specify evaluation metrics exactly: Pearson correlation on arrival times, relative RMSE, IoU@50 — Gahtan reports all three. Report all three for comparability.

---

### Claim 5: "Phase 5 mesh extension addresses Gahtan's stated limitation and improves accuracy on steep terrain"

- **Evidence provided:** Convert terrain DEM to `TriangularMesh` + `DiscreteRanders`, same covariate encoder and loss.
- **Literature context:** Gahtan Section 7 explicitly states: *"Extension to triangulated surfaces would replace the eight-stencil system with triangle-fan neighborhoods."* This is a genuine gap. Qian, Zhang & Zhao (2007) (arXiv reference [18] in Gahtan) provide fast sweeping on triangular meshes but without Randers metrics or differentiability.
- **Verdict:** **WARNING** (mixed: strong concept, weak experimental design)
- **Recommendation:**
  1. **STRONG** — The concept of addressing Gahtan's stated limitation is the strongest part of this plan and constitutes a genuine novel contribution.
  2. **MISSING** — The hypothesis "3D terrain surface improves accuracy on steep terrain" requires:
     - A precise definition of "steep terrain" (slope threshold in degrees or gradient magnitude).
     - A pre-specified subset of Sim2Real-Fire scenes/fires with steep terrain, identified *a priori* from the topographic covariates.
     - A null hypothesis: flat-grid Lagrangian performance = mesh Lagrangian performance on the steep-terrain subset.
     - A statistical test (paired t-test or Wilcoxon signed-rank on per-fire correlations).
  3. **MISSING** — No DEM-to-mesh pipeline is described. Mesh resolution, quality constraints (Delaunay/constrained Delaunay), and how terrain features are preserved in the triangulation must be specified.
  4. **MISSING** — Flat-grid Lagrangian control: Phase 5 should compare mesh-Lagrangian vs. flat-Lagrangian (Phase 4) on the same fires, not just mesh-Lagrangian vs. Gahtan-eikonal. Otherwise improvements could be attributed to the mesh rather than the Lagrangian approach.

---

### Claim 6: "HAMTools-only extensions (geodesic paths, Jacobi fields, curvature anomalies, latent-space metric learning) are scientifically meaningful"

- **Evidence provided:** Listed as qualitative extensions beyond Gahtan's capabilities.
- **Literature context:**
  - Geodesic path visualization: directly interpretable for fire corridor analysis. Gahtan could extract paths via gradient descent on $T(x)$, but does not.
  - Jacobi field stability: novel application to wildfire. No prior work linking Finslerian Jacobi field divergence to fire spread stability.
  - Curvature anomaly detection: plausible but speculative without validation.
  - Latent-space metric: requires separate justification (not wildfire-specific).
- **Verdict:** **NOTE**
- **Recommendation:** These are interesting directions but must be presented as **exploratory analyses**, not claims. Specifically:
  - For Jacobi fields: validate against known fire behavior (e.g., do divergent geodesic bundles correlate with empirically observed spotting or erratic fire behavior?).
  - For curvature anomalies: define what constitutes "anomalous" curvature (threshold, percentile) and test whether flagged regions correlate with features visible in the covariates (ridgelines, fuel breaks, water bodies).

---

## Methodology Audit

### Baseline Fairness

- **Verdict:** **CRITICAL**
- The Eulerian (eikonal) and Lagrangian (geodesic) approaches have fundamentally different computational and information-theoretic profiles. The eikonal solver sees the **entire grid simultaneously** — it solves for $T(x)$ at all $n$ grid points in one forward pass. The Lagrangian solver computes individual point-to-point distances. Comparing a method that observes 100% of the domain to one that observes 0.2–3% is not a fair baseline.
- **Recommended fix:** Frame the comparison around **information parity**:
  - **Parity by observation count:** Restrict Gahtan's observation set to the same K points used by HAMTools (evaluate eikonal MSE only at those K locations, even though the forward solve covers the full grid).
  - **Parity by compute budget:** Fix wall-clock time or FLOP budget and compare best-achievable accuracy.
  - **Parity by supervision signal:** Both methods are given the same $(x_i, T_i^{\text{obs}})$ pairs. The eikonal solver uses the full grid for the *forward model* but evaluates loss only at the K observation points. HAMTools only needs forward computation at K points. The comparison then measures solver fidelity, not information advantage.

### Runtime Comparison

- **Verdict:** **WARNING**
- The plan proposes a runtime comparison but does not control for:
  - **Hardware:** Gahtan uses PyTorch + Numba/CUDA; HAMTools uses JAX. JAX `vmap` has different parallelism characteristics than CUDA kernels.
  - **Amortization:** Gahtan's $O(n)$ eikonal solve produces distances to all grid points; HAMTools' $O(K \cdot T \cdot I)$ produces distances to K points only. Cost-per-distance-query is the relevant metric, not cost-per-forward-pass.
  - **Compilation:** JAX JIT compilation time should be excluded from benchmarks (report separately).
- **Recommendation:** Report (a) wall-clock per training step on matched hardware (single GPU type), (b) cost per usable distance query, (c) total training time to convergence.

---

## Reproducibility Checklist

- [ ] **Random seeds fixed** — **MISSING.** No mention of fixing JAX PRNG seeds for weight initialization, observation point sampling, or data shuffling. The plan does not specify a seed protocol.
- [ ] **Hyperparameters logged** — **MISSING.** No mention of learning rate schedules, optimizer choice, regularization weights $\lambda$, AVBD step counts, convergence tolerances, or early stopping criteria for any phase.
- [ ] **Data preprocessing deterministic and versioned** — **PARTIAL.** Sim2Real-Fire is publicly available (Google Drive, Apache-2.0). However:
  - The exact 15 scenes Gahtan uses are not listed in the plan (only in Gahtan's Appendix F, Table 9).
  - The 70/15/15 train/val/test split per scene is not specified in the plan.
  - The plan does not mention whether Gahtan's exact splits are recoverable (no seed/split specification in Gahtan's code).
- [ ] **Results include variance estimates** — **MISSING.** No mention of running experiments over multiple seeds and reporting mean ± std. Single-run results are scientifically insufficient for comparison.
- [ ] **Baselines are appropriate and fairly implemented** — **CRITICAL gap.** As detailed above, the observation-density mismatch makes the primary comparison unfair.

---

## Missing Ablations

### 1. Finsler vs. Riemannian ablation (Severity: **CRITICAL**)
The entire premise is that Finsler (Randers) metrics capture directional asymmetry that Riemannian metrics miss. **No phase in the plan tests this.** Phase 2 recovers drift, but there is no ablation where drift is set to zero ($W = 0$) to quantify the marginal benefit of the Finsler component for wildfire prediction. This is the single most important ablation for the paper's core claim.

**Recommendation:** In every phase, run a Riemannian control ($W = 0$, optimize only $h$). Report:
- Metric recovery error with and without drift.
- Wildfire correlation/RMSE/IoU with and without drift.
- $\|\hat{W}\|$ distribution to assess whether learned wind is meaningful or near-zero.

### 2. Strategy A vs. B vs. C comparison (Severity: **WARNING**)
Three forward-model strategies are proposed (BVP, Euler–Lagrange residual, shooting fan) but the plan selects "A or C" for Phase 4 without specifying selection criteria or comparing them.

**Recommendation:** Run all three strategies on Phase 1 (synthetic isotropic, cheapest) to determine:
- Accuracy at matched K.
- Wall-clock time per training step.
- Convergence speed (epochs to target error).
Use the winner for Phases 4–5, but report the comparison.

### 3. K sensitivity (Severity: **CRITICAL**)
K=200 is proposed without justification. This is the most important hyperparameter for the Lagrangian approach — it controls the information budget.

**Recommendation:** Sweep K ∈ {50, 100, 200, 500, 1000, 2000, 5000} on Phase 1 and report error vs. K. Determine the minimum K for convergence and the diminishing-returns threshold.

### 4. AVBD solver configuration (Severity: **WARNING**)
The plan specifies "15-step AVBD at 50 iterations" without justification. These are critical numerical parameters that affect both accuracy and runtime.

**Recommendation:** Ablate:
- AVBD steps (n_steps): {5, 10, 15, 20, 30}.
- AVBD iterations (max_iter): {20, 50, 100}.
Report geodesic arc-length error against known analytical solutions (Euclidean, Randers on flat space).

### 5. Regularization ablation (Severity: **WARNING**)
The plan mentions `MetricSmoothnessLoss` but does not specify the regularization type or strength. Gahtan demonstrates that TV regularization is critical (22.4% error without, 8.1% with optimal $\lambda$; Tikhonov fails entirely).

**Recommendation:** Compare HAMTools' Jacobian-based `MetricSmoothnessLoss` against a TV-style penalty. Report optimal $\lambda$ via validation sweep.

### 6. Multi-source vs. single-source (Severity: **NOTE**)
Phase 3 proposes 3–5 sources but does not ablate this. Gahtan shows a 45% error reduction from 1→5 sources (Table 3).

**Recommendation:** Run Phase 3 with 1, 2, 3, 5 sources and report the same analysis as Gahtan Table 3.

---

## Dataset Access and Reproducibility

### Sim2Real-Fire Availability
- **Verdict:** **STRONG**
- The dataset is publicly available from [TJU-IDVLab/Sim2Real-Fire](https://github.com/TJU-IDVLab/Sim2Real-Fire) on Google Drive (20 simulation data packages + real-world data). Licensed under Apache-2.0.
- Gahtan's code includes `download_data.py` for automated download.
- The specific 15 scenes used by Gahtan (scenes 0001–0025) fall within `simulation_data_01` (scenes 0001–0019) and likely `simulation_data_02` (scenes 0020–0046).

### Scene/Split Reproducibility
- **Verdict:** **WARNING**
- Gahtan lists the 15 scenes in Appendix F (Table 9) with fire counts, but does not publish the exact train/val/test fire IDs. The plan should either: (a) use Gahtan's code to reproduce the splits, or (b) specify its own splits and note the deviation.

### Gahtan Code Availability
- **Verdict:** **STRONG**
- Public GitHub repository under MIT license. All experiments have reproducibility commands in the README. This enables direct comparison and split recovery.

---

## Lagrangian Feasibility Assessment

### Can AVBD-based BVP realistically solve the wildfire inverse problem?

- **Verdict:** **WARNING**
- The mathematical equivalence (eikonal characteristics = geodesics) is rigorous: $T(x) = d_F(\text{source}, x) = \inf_\gamma \int_0^1 F(\gamma, \dot\gamma)\, dt$. This is correct.
- The computational challenge is severe. AVBD is an iterative boundary-value solver designed for finding geodesics between two points. For the wildfire problem:
  - Each observation point requires a separate BVP solve.
  - AVBD convergence is not guaranteed for arbitrary boundary conditions in strongly heterogeneous media.
  - The existing comparative analysis (Section 8.5 of `reviews/science/comparative_gahtan2026.md`) estimates **10–50× slower per training step** than eikonal — this may be optimistic for realistic grid sizes (300×300+).
- The plan does not address AVBD failure modes:
  - What if AVBD fails to converge for some $(source, x_i)$ pairs? (Heterogeneous metrics with barriers.)
  - What is the AVBD accuracy at 15 steps on Randers metrics? (Validated on spheres/tori, not on complex 2D spatial fields.)
  - How does AVBD scale with domain diameter? (Longer geodesics = more integration error.)
- **Recommendation:** Before Phase 4, run a **forward-only validation**: given Gahtan's ground-truth metric $(G, b)$ from a solved synthetic problem, compute AVBD geodesic distances at all grid points and compare to the eikonal arrival times. This measures pure solver accuracy independent of the learning problem.

---

## Suggested Experiments

The following experiments would substantially strengthen the paper:

1. **Observation density sweep** (Phase 1): K ∈ {50, 100, 200, 500, 1000, 3200, 6400} on the 80×80 isotropic problem. Plot error vs. K and overlay Gahtan's Figure 32 for direct comparison of density–accuracy tradeoffs.

2. **Forward solver validation** (Pre-Phase 4): Given known Randers metric parameters, compute AVBD geodesic distances at N=1000 grid points and compare to eikonal arrival times. Report absolute and relative error. This validates the solver before using it for learning.

3. **Randers vs. Riemannian ablation** (All phases): Run every experiment with $W=0$ (Riemannian only). This is the paper's most important control.

4. **Zermelo vs. direct parameterization** (Phase 1): Compare HAMTools' Zermelo parameterization $(h, W)$ against Gahtan's direct parameterization $(G, b)$ in the Lagrangian setting. Does the parameterization choice affect learning dynamics?

5. **Strategy comparison** (Phase 1): Run BVP (Strategy A), E-L residual (Strategy B), and shooting fan (Strategy C) on the same problem. Report accuracy, runtime, and convergence.

6. **Terrain slope stratification** (Phase 5): Partition fires by maximum terrain slope into bins (0–10°, 10–20°, 20–30°, 30°+). Report performance per bin for flat-grid vs. mesh approaches. This tests the specific hypothesis that mesh improves steep-terrain accuracy.

7. **Multi-seed stability** (All phases): Run each experiment with ≥5 random seeds. Report mean ± std for all metrics. Use Welch's t-test or Mann-Whitney U for comparing HAMTools vs. Gahtan at matched density.

8. **Convergence analysis** (All phases): Plot training loss, validation metric, and recovered metric error vs. epoch/iteration for each phase. Include Gahtan's convergence curves (extractable from their code) for comparison.

9. **Geodesic path qualitative analysis** (Phase 4): For representative fires, visualize the actual geodesic paths (fire corridors) alongside Gahtan's arrival-time contours. Assess whether geodesic paths reveal information not visible in distance fields (bottlenecks, bifurcations).

10. **Adjoint ODE sensitivity** (Phase 4): Compare JAX standard AD vs. `diffrax` adjoint-method differentiation for the geodesic ODE. Report memory usage and gradient accuracy for varying integration horizons.

---

## Per-Phase Summary

| Phase | Assessment | Key Issue |
|-------|-----------|-----------|
| 1. Synthetic Isotropic | **Needs major revision** | Observation density confound invalidates headline comparison |
| 2. Synthetic Drift | **Needs revision** | Same density caveat; target needs density-matched baseline |
| 3. Synthetic Full Randers | **Acceptable with caveats** | Convergence criteria undefined; missing Riemannian ablation |
| 4. Sim2Real-Fire | **Needs major revision** | K=200 catastrophically sparse for real grid sizes; per-scene vs. cross-scene unspecified; no variance |
| 5. Mesh Extension | **Strong concept, weak design** | Hypothesis untestable without slope stratification, flat-grid control, and DEM-to-mesh pipeline spec |

---

## Conclusion

The experiment plan identifies a scientifically valuable comparison (Lagrangian vs. Eulerian Randers metric learning) and a genuinely novel contribution (mesh extension). However, in its current form, the plan would produce results that are not directly comparable to Gahtan's due to the **observation-density mismatch** and would lack the statistical rigor required for publication. The three highest-priority fixes are:

1. **Match observation density** or explicitly design the comparison around information parity.
2. **Add Riemannian ($W=0$) controls** to every phase.
3. **Specify seeds, repetitions, and variance reporting** for all quantitative results.

With these revisions, the plan could produce a strong contribution demonstrating the complementary strengths of Lagrangian and Eulerian approaches to Randers metric learning, with the mesh extension as the headline novel result.
