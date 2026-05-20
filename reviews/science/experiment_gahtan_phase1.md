# Science Audit: `experiment_gahtan_phase1.py`

**Auditor:** Science Auditor Agent  
**Date:** May 17, 2026  
**Experiment file:** [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py)  
**Experiment plan:** [reviews/science/experiment_plan_gahtan_lagrangian.md](reviews/science/experiment_plan_gahtan_lagrangian.md)  
**Reference paper:** Gahtan, Shpund & Bronstein (2026). *Wildfire Simulation with Differentiable Randers-Finsler Eikonal Solvers.* arXiv:2603.00035, Section 5.

---

## Summary

**Overall scientific rigor assessment: NEEDS MAJOR REVISION**

The experiment is well-structured in intent — density sweep, regularization sweep, multi-seed error bars, and geodesic visualizations. However, the implementation contains one **CRITICAL** flaw in the ground-truth computation that invalidates quantitative comparison with Gahtan baselines, a missing ablation that was explicitly required by the experiment plan, and several parameter deviations from the plan that are undocumented. The experiment in its current form cannot support the claim that Lagrangian recovery is "comparable" to Eulerian eikonal baselines because the training targets themselves are systematically biased for a significant fraction of grid points.

---

## Claims Audit

### Claim 1: "Can boundary-value geodesic solvers learn a Finsler metric from arrival-time supervision, matching Eulerian eikonal baselines?"

- **Evidence provided:** Density sweep with 5 seeds × 4 densities, error bars, Gahtan baseline overlay.
- **Literature context:** Gahtan et al. Section 5 reports 5.6% (100% obs.) and 21.2% (7% obs.) relative recovery error for isotropic piecewise-constant metrics using an Eulerian eikonal solver. The ground-truth setup (80×80 grid, piecewise-constant $g \in \{1, 2\}$, central source) is designed to match.
- **Verdict:** **CRITICAL** — Ground-truth arrival times are incorrect (see Finding F1 below), invalidating the quantitative comparison.
- **Recommendation:** Fix `compute_true_arrival_times` to solve Snell's law exactly, or use Gahtan's eikonal solver (available under MIT license at `BarakGahtan/differentiable-eikonal-wildfire`) to generate ground-truth fields. Then re-run the entire experiment.

---

### Claim 2: "NeuralRanders with use_wind=False provides a Riemannian ablation"

- **Evidence provided:** The docstring at [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L17) states "use_wind=False for Riemannian ablation." The code comment at [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L260) says "Riemannian: use_wind implicitly False via Randers with zero wind."
- **Literature context:** The experiment plan ([reviews/science/experiment_plan_gahtan_lagrangian.md](reviews/science/experiment_plan_gahtan_lagrangian.md)) explicitly lists "W=0 (Riemannian)" as a required ablation.
- **Verdict:** **CRITICAL** — `NeuralRanders.__init__` at [src/ham/models/learned.py](src/ham/models/learned.py#L69) does **not** accept a `use_wind` parameter and does **not** pass one to the parent `Randers.__init__`, which defaults to `use_wind=True`. The experiment therefore trains a **full Randers metric with a learnable wind field**, even though the ground truth is isotropic with no wind. This wastes model capacity, may bias the metric tensor recovery, and the Riemannian control ablation claimed in the docstring is never actually performed.
- **Recommendation:** Either (a) add `use_wind` parameter to `NeuralRanders.__init__` and pass it to `super().__init__`, or (b) explicitly run a Riemannian ablation by constructing a `NeuralRanders` and freezing its `w_net` parameters. Report results for both `use_wind=True` and `use_wind=False` side by side.

---

### Claim 3: "TV regularization encourages spatial smoothness while allowing sharp transitions"

- **Evidence provided:** `tv_regularization()` at [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L209) computes `jnp.mean(jacs ** 2)`, the L2 squared Jacobian norm.
- **Literature context:** Total variation (TV) regularization is defined as $\text{TV}(f) = \int |\nabla f| \, dx$ (L1 norm of gradient), which is the standard choice for recovering piecewise-constant fields because it is edge-preserving. L2 squared gradient penalty ($\int |\nabla f|^2 \, dx$) is Tikhonov regularization, which penalizes sharp transitions and promotes smooth solutions. These are distinct regularizers with opposite inductive biases for the target function. Gahtan et al. Section 5 use standard TV regularization.
- **Verdict:** **WARNING** — The regularizer is mislabeled and has the wrong inductive bias for piecewise-constant recovery. An L2 penalty will smooth out the sharp metric boundary at $x_1 = 0.5$, systematically biasing the recovery toward gradual transitions. The regularization sweep results may still be informative (U-curve shape), but the comparison with Gahtan's TV sweep is not apples-to-apples.
- **Recommendation:** Implement true L1-norm TV: $\sum_i \|\nabla g(x_i)\|_1$ or use the Huber approximation $\sum_i \sqrt{|\nabla g(x_i)|^2 + \epsilon^2}$ for differentiability. Re-run the regularization sweep with the corrected regularizer and compare with Gahtan Fig. 31.

---

### Claim 4: "Geodesic path visualization demonstrates HAMTools-only capability"

- **Evidence provided:** `plot_geodesic_paths()` at [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L508) overlays geodesic paths colored by arrival time on the metric field.
- **Literature context:** Eikonal solvers produce arrival time fields $T(x)$ on the entire grid but do not explicitly compute geodesic paths. Paths can be recovered via backtracing $\nabla T$, but this is a post-hoc step not supported in Gahtan's codebase. The Lagrangian AVBD approach naturally produces explicit paths as its primary output.
- **Verdict:** **STRONG** — This is a well-motivated and genuinely differentiating visualization. The fan of 24 × 2 targets in multiple radii provides good coverage.
- **Recommendation:** Add a quantitative measure: compute the Snell's law refraction angle predicted by the learned metric at the boundary crossing and compare it to the theoretical value. This would further demonstrate the geometric fidelity of the Lagrangian approach.

---

### Claim 5: "Gahtan baselines are fairly compared"

- **Evidence provided:** `plot_density_sweep()` at [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L452) plots Gahtan baselines as two diamond markers (100% → 5.6%, 7% → 21.2%) with no error bars. HAMTools results are plotted with error bars from 5 seeds.
- **Literature context:** Gahtan Section 5 reports single-run results (no variance), but their training uses different random seeds for initialization. Their density sweep covers more data points than the two shown.
- **Verdict:** **WARNING** — The comparison is visually biased in HAMTools' favor (error bars on one series, none on the other). Only two Gahtan baseline points are shown, while their paper presents a full density curve. The Gahtan numbers are hardcoded at [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L789), making it impossible to verify they are transcribed correctly.
- **Recommendation:** (a) Reproduce Gahtan baselines using their public code to obtain error bars and a full density curve. (b) If reproduction is not feasible, clearly annotate the Gahtan points as "reported in paper, single run" and cite the exact figure/table number. (c) Match density levels: the experiment uses {100%, 50%, 7%, 3%} but only 100% and 7% have Gahtan counterparts. Add 50% and 3% Gahtan baselines if available.

---

## Detailed Findings

### F1: Ground-Truth Arrival Time Computation Is Systematically Biased (CRITICAL)

**Location:** [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L137-L183)

**Problem:** The function `compute_true_arrival_times` has two errors:

1. **Same-side paths assumed optimal (line 164):** When source and target are on the same side of the boundary, `dist_straight = sqrt(g_source) * ||target - source||` assumes the straight-line path is optimal. This is **wrong** when both points are in the high-cost region ($g = 2$). A path that detours through the low-cost region ($g = 1$) along the boundary can be cheaper.

   **Worked example:** Source at $(0.5, 0.5)$ (on boundary, classified as $g = 2$), target at $(0.51, 0.9)$ (also $g = 2$, $same\_side = \text{True}$):
   - Straight-line cost: $\sqrt{2} \cdot \sqrt{0.01^2 + 0.4^2} \approx 0.566$
   - Optimal path (along boundary in $g = 1$, then short crossing): $1.0 \cdot 0.4 + \sqrt{2} \cdot 0.01 \approx 0.414$
   - **Relative bias: +36.7%**

   This bias affects all $g = 2$ targets with large transverse offsets relative to the source. On the 80×80 grid, approximately 25-40% of points in the $g = 2$ half are significantly affected.

2. **Boundary-crossing paths use straight-line crossing, not Snell's law (line 175):** The crossing point is computed as the linear interpolation $y_{\text{cross}} = s_y + \frac{d_1}{d_1 + d_2}(t_y - s_y)$. The correct crossing point satisfies Snell's law: $\sqrt{g_1} \sin\theta_1 = \sqrt{g_2} \sin\theta_2$, which requires solving a quartic equation or using numerical optimization. However, since the source is on the boundary ($d_1 = 0$), this case degenerates and the approximation is coincidentally exact.

**Impact:** The training targets $T_{\text{obs}}$ are biased upward for a substantial fraction of grid points. This means the learned metric will be trained toward incorrect arrival times, and the reported recovery errors are measured against incorrect ground truth. Any comparison with Gahtan's eikonal-generated ground truth (which IS correct) is invalid.

**Recommended action:** Compute exact arrival times by either:
- Solving the eikonal equation on the grid (use Gahtan's public solver or the fast marching method)
- Solving Snell's law exactly at the boundary, accounting for the possibility of total-internal-reflection-like routing through the cheaper medium
- Using the heat method (Crane et al. 2017) as an independent verification

---

### F2: Undocumented Parameter Deviations From Experiment Plan (WARNING)

**Location:** [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L74-L93) vs [reviews/science/experiment_plan_gahtan_lagrangian.md](reviews/science/experiment_plan_gahtan_lagrangian.md)

| Parameter | Plan Value | Implementation Value | Impact |
|-----------|-----------|---------------------|--------|
| `n_train_steps` | 500 | 400 | 20% fewer optimization steps; may underfit |
| `solver_steps` (AVBD `n_steps`) | 20 | 15 | Coarser path discretization; larger quadrature error in arc length |
| `solver_iters` (AVBD `iterations`) | 100 | 80 | 20% fewer solver iterations; AVBD may not converge to minimum |
| `obs_per_train_step` (batch size) | Not specified | 32 | Reasonable but undocumented in plan |

**Verdict:** **WARNING** — None of these individually is likely to change qualitative conclusions, but the accumulated effect of fewer steps, fewer iterations, and coarser paths moves the operating point further from the "carefully controlled comparison" intended by the plan. At minimum, these deviations must be documented with justification.

**Recommended action:** Either update the implementation to match the plan, or update the plan with a justification for the reduced values (e.g., "reduced to keep wall-clock time under 2 hours on M-series hardware").

---

### F3: Missing Ablations Required by Experiment Plan (WARNING)

**Location:** The experiment plan specifies 7 ablations; the implementation performs 2.

| Planned Ablation | Implemented? | Status |
|-----------------|-------------|--------|
| W=0 (Riemannian) | **No** — see Claim 2 | **CRITICAL** |
| Observation density sweep | **Yes** | ✓ |
| AVBD steps (`n_steps` sweep) | **No** | MISSING |
| AVBD iterations sweep | **No** | MISSING |
| Network depth sweep | **No** | MISSING |
| TV regularization sweep | **Yes** | ✓ |
| Multi-start AVBD | **No** | MISSING |

**Verdict:** **WARNING** — The AVBD steps and iterations sweeps are important for establishing that results are not sensitive to solver discretization. Without them, a reviewer could reasonably argue that the results are an artifact of the particular discretization chosen.

**Recommended action:** At minimum, add the AVBD steps sweep (`n_steps ∈ {10, 15, 20, 30}`) and the Riemannian ablation. The depth and multi-start ablations are lower priority but should be included for publication.

---

### F4: No Held-Out Evaluation (NOTE)

**Location:** [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L338-L362)

`evaluate_recovery` computes the metric field error on the **entire grid**, including points used for training. The density sweep subsamples for training but evaluates on the full grid. This is acceptable for metric field recovery (the evaluation is on the *metric values*, not the arrival times, so it tests generalization of the metric field to unobserved locations). However, this should be explicitly stated.

**Verdict:** **NOTE** — Evaluation is on the metric field, not arrival times, so train/test overlap is not a concern for the primary metric. But an additional "held-out arrival time" evaluation would strengthen the claim.

---

### F5: Source Placement on Metric Boundary (NOTE)

**Location:** [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L670)

The source is at $(0.5, 0.5)$, exactly on the metric boundary at $x_1 = 0.5$. With the `x[0] < boundary` condition, the source is classified in the $g = 2$ region. This is a special case where boundary-crossing ground truth (Finding F1, part 2) is coincidentally exact. It does not test the general case and may mask the ground-truth bias for off-boundary sources.

**Verdict:** **NOTE** — Consistent with Gahtan's setup. But a robustness check with an off-center source (e.g., $(0.3, 0.5)$) would reveal the ground-truth bias from F1 more clearly.

---

### F6: `use_fourier=False` May Hurt Recovery at Boundaries (NOTE)

**Location:** [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L262)

`NeuralRanders` is initialized with `use_fourier=False`, disabling Random Fourier Features in the wind network. The experiment plan does not specify this setting. For recovering a piecewise-constant function with a sharp boundary, Fourier features can help capture high-frequency transitions. However, since the metric network (`PSDMatrixField`) does not use Fourier features regardless, this mainly affects the wind network — which should ideally be zero for this isotropic ground truth.

**Verdict:** **NOTE** — Acceptable for the Riemannian case; could affect the Randers comparison if wind recovery is later tested.

---

### F7: Arrival Time Comparison Uses Fixed Evaluation Key (STRONG)

**Location:** [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L592)

The arrival time comparison plot uses `key = jax.random.PRNGKey(99)` to sample evaluation points, separate from training keys. This ensures consistent evaluation across runs.

**Verdict:** **STRONG** — Good practice for reproducible evaluation.

---

### F8: Cosine Decay Schedule Well-Chosen (STRONG)

**Location:** [examples/experiment_gahtan_phase1.py](examples/experiment_gahtan_phase1.py#L274)

The optimizer uses `optax.cosine_decay_schedule` with the correct total step count. This avoids the common error of using a fixed learning rate with Adam, which can lead to oscillation in later training stages.

**Verdict:** **STRONG** — Appropriate choice for the training duration.

---

## Reproducibility Checklist

- [x] Random seeds fixed — `SEED = 42` for data, seeds `[0, 1, 2, 3, 4]` for training. `jax_enable_x64` enabled for deterministic numerics.
- [ ] Hyperparameters logged — Configuration dict is well-structured but deviates from the plan without documentation (see F2).
- [ ] Data preprocessing deterministic and versioned — Ground truth is computed analytically but **incorrectly** (see F1). No versioning of the ground-truth computation.
- [x] Results include variance estimates — Mean ± std over 5 seeds for both sweeps.
- [ ] Baselines are appropriate and fairly implemented — Gahtan baselines are hardcoded, not reproduced. Comparison is visually biased (see Claim 5). Ground-truth mismatch invalidates quantitative comparison (see F1).

---

## Suggested Experiments

1. **Fix ground truth (BLOCKER):** Recompute arrival times using an exact solver (eikonal or Snell's law), then re-run the full experiment. This is a prerequisite for all other improvements.

2. **Riemannian ablation (W=0):** Add explicit `use_wind=False` runs. For the isotropic ground truth, Riemannian recovery should match or outperform Randers (fewer parameters to fit). Failure to do so would indicate a problem with the optimization or loss landscape.

3. **Reproduce Gahtan baselines:** Run Gahtan's public code on the same grid and metric configuration to obtain comparable error numbers with variance. This eliminates transcription errors and ensures apples-to-apples comparison.

4. **Off-center source:** Repeat with source at $(0.3, 0.5)$ and $(0.7, 0.5)$ to test recovery when the source is far from the boundary. This is more challenging and representative.

5. **AVBD discretization sensitivity:** Sweep `solver_steps ∈ {10, 15, 20, 30}` to show that recovery quality is stable above a resolution threshold. Plot error vs. solver_steps.

6. **True TV regularization:** Replace the L2 Jacobian penalty with L1 (or Huber) total variation and re-run the regularization sweep. The U-curve shape and optimal $\lambda$ may change significantly.

7. **Multi-source experiment:** Use 3 and 5 sources at distinct locations to match Gahtan Table 3 and demonstrate multi-source improvement. This is planned for Phase 2 but a preliminary single-density test in Phase 1 would be valuable.

8. **Solver convergence verification:** Before training, verify that the AVBD solver converges to the correct geodesic for the known metric by solving several BVPs with the ground-truth metric and comparing arc lengths to analytical distances. This establishes the solver's baseline accuracy independent of learning.
