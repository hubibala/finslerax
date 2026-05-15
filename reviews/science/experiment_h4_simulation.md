# Science Audit: experiment_h4_simulation.py

**Auditor:** Science Auditor Agent  
**Date:** 2026-05-15  
**Source:** `examples/experiment_h4_simulation.py`  
**Spec Reference:** `spec/RESEARCH_LOG.md` §3.1 (H4)

---

## Summary

**Overall scientific rigor assessment: needs revision**

H4 tests whether Randers wind-guided geodesic shooting from day-2 progenitors lands closer to the correct day-6 fate centroid than a Riemannian (wind-free) baseline. The experimental concept is sound and the biological question is well-motivated. However, there are **several methodological weaknesses** that undermine the strength of any published claim: no variance estimation, a single hard-coded decision threshold, potential information leakage through shared fate centroids, and only two target fates evaluated.

---

## Claims Audit

### Claim 1: "Randers wind improves predictive shooting (delta > 3%)"

`experiment_h4_simulation.py:128–131`

> ```python
> if delta > 0.03:
>     print("\n✓ H4 SUPPORTED: Randers wind improves predictive shooting.")
> ```

- **Evidence provided:** Single-run accuracy difference (Randers vs Riemannian null) on N_EVAL=600 test triples. Decision criterion is `delta > 0.03`.
- **Literature context:** Predictive shooting experiments in trajectory inference (e.g., PRESCIENT, Tong et al. NeurIPS 2020; Waddington-OT, Schiebinger et al. Cell 2019) report results with confidence intervals across multiple random subsets or bootstrap replicates.
- **Verdict:** WARNING
- **Recommendation:**
  1. Report accuracy with 95% bootstrap confidence intervals (≥1000 resamples over the 600 test triples).
  2. The 3% threshold is arbitrary and not grounded in any power analysis. Replace with a formal one-sided McNemar's test or permutation test comparing paired per-sample correctness between Randers and Null.
  3. Run with multiple random seeds (at least for anchor subsampling and test-triple ordering) to confirm stability.

---

### Claim 2: "Deterministic geodesic shooting (N_TRAJ=1) is a valid evaluation"

`experiment_h4_simulation.py:39`

> ```python
> N_TRAJ = 1  # deterministic = 1 trajectory per start
> ```

- **Evidence provided:** The experiment fires a single deterministic geodesic per start cell and evaluates nearest-centroid assignment. No stochastic perturbation or ensemble.
- **Literature context:** Biological fate decisions are inherently stochastic; PRESCIENT (Yeo et al. 2021) simulates $K \geq 50$ stochastic trajectories per start cell and reports fate probability distributions. Single-trajectory deterministic shooting conflates metric geometry with integration noise and does not produce fate probability estimates.
- **Verdict:** WARNING
- **Recommendation:**
  1. Add stochastic perturbation of the initial velocity (e.g., Gaussian noise on $v_0$) and shoot $K \geq 20$ trajectories per start cell to estimate fate probabilities.
  2. Evaluate with a probabilistic metric (cross-entropy on predicted fate distribution vs. observed fate) rather than hard nearest-centroid assignment.
  3. At minimum, acknowledge the deterministic limitation and frame the result as a lower bound on discriminative capacity.

---

### Claim 3: "Wind field as initial velocity (v0 = 0.8 * W / ||W||) is the correct shooting protocol"

`experiment_h4_simulation.py:99–101`

> ```python
> w_norm = jnp.linalg.norm(w) + 1e-8
> v0 = w * (0.8 / w_norm)
> ```

- **Evidence provided:** The initial velocity is the normalised wind field scaled to magnitude 0.8. This is a design choice with no justification or ablation.
- **Literature context:** In Zermelo navigation, the wind $W$ modifies the geodesic ODE but is not the same as the "intended travel direction." Using $W$ as $v_0$ conflates two distinct roles: $W$ is an external drift, whereas $v_0$ should represent the cell's autonomous differentiation direction. The magnitude 0.8 is arbitrary.
- **Verdict:** CRITICAL
- **Recommendation:**
  1. Ablate the initial velocity magnitude (sweep 0.1–2.0) and report sensitivity.
  2. Justify why $v_0 \propto W$ rather than, e.g., using the RNA velocity push-forward directly or the direction toward the nearest fate centroid.
  3. For the null model (Riemannian), $v_0 = \mathbf{0}$ (line 103) means the null geodesic is a fixed point — this is not a fair comparison. The null model should receive the same initial velocity protocol (e.g., RNA velocity direction), otherwise the experiment is testing "shooting with velocity vs. staying still" rather than "Finsler vs. Riemannian geometry."

---

### Claim 4: "Riemannian (Null) baseline is a fair comparison"

`experiment_h4_simulation.py:96–104`

> ```python
> has_wind = hasattr(model.metric, 'w_net') and getattr(model.metric, 'use_wind', False)
> if has_wind:
>     w = model.metric.w_net(z)
> else:
>     w = jnp.zeros_like(z)
> v0 = w * (0.8 / w_norm)
> ```

- **Evidence provided:** When `use_wind=False` (Null model), $w = \mathbf{0}$ and therefore $v_0 = \mathbf{0}$. The geodesic does not move.
- **Literature context:** A valid null baseline must isolate the variable under test. Here, the Randers model receives a nonzero initial velocity from the wind field, while the Null model receives zero velocity. The comparison tests "wind as velocity vs. no velocity" rather than "Finsler geometry vs. Riemannian geometry."
- **Verdict:** CRITICAL
- **Recommendation:**
  1. Both models must receive identical initial velocities (e.g., from the RNA velocity push-forward, or from a shared direction field).
  2. The Randers model's advantage should come from its *geodesic curvature* (how the wind bends the trajectory), not from receiving a nonzero starting kick.
  3. This is the most important fix — without it, the entire H4 comparison is confounded.

---

### Claim 5: "Fate centroids from day-6 cells provide a valid evaluation target"

`experiment_h4_simulation.py:66–76`

> ```python
> day6_indices = np.unique(all_triples[:, 2])
> ...
> fate_centroids[fname] = jnp.array(Z_all[mask].mean(axis=0))
> ```

- **Evidence provided:** Fate centroids are computed from day-6 cells indexed by ALL test triples (not just the N_EVAL subset), encoded through the same VAE.
- **Literature context:** Using the same encoder for both trajectory simulation and target construction is standard (PRESCIENT, WOT do this), but the centroids include all day-6 cells from the test set. If a test cell's own day-6 endpoint contributes to its fate centroid, there is a mild optimistic bias (the centroid is shifted toward the true endpoint).
- **Verdict:** WARNING
- **Recommendation:**
  1. Use leave-one-out centroids: for each test triple, compute the fate centroid excluding the specific day-6 cell from that triple.
  2. Alternatively, compute centroids from the training set only (day-6 cells from training clones).

---

### Claim 6: "TARGET_FATES = ['Monocyte', 'Neutrophil'] is sufficient scope"

`examples/weinreb_vae.py:889`

- **Evidence provided:** Only 2 of the ~10+ annotated cell types in the Weinreb dataset are evaluated.
- **Literature context:** The Weinreb dataset (Weinreb et al. Cell 2020) tracks at least Monocyte, Neutrophil, Basophil, Erythrocyte, and other myeloid/lymphoid fates. Restricting to 2 fates limits generalizability and may cherry-pick the easiest separation.
- **Verdict:** WARNING
- **Recommendation:**
  1. Include all fates with ≥50 day-6 cells in the evaluation (or explain the exclusion criterion explicitly).
  2. Report per-fate accuracy to reveal whether the metric helps uniformly or only for the dominant lineages.
  3. At minimum, add a "Limitations" note acknowledging the restricted scope.

---

### Claim 7: "ExponentialMap(step_size=0.015, max_steps=120) is a valid integration setting"

`experiment_h4_simulation.py:85`

- **Evidence provided:** No convergence analysis is presented for the RK4 integrator. `step_size=0.015` with `max_steps=120` gives a total integration time of $T = 1.8$, which is specific to the latent space scale.
- **Literature context:** Standard practice for ODE-based trajectory inference (Neural ODEs, Chen et al. NeurIPS 2018) includes convergence checks by halving step size and verifying endpoint stability.
- **Verdict:** WARNING
- **Recommendation:**
  1. Run with `max_steps` ∈ {60, 120, 240} and verify that accuracy does not change by more than 1%.
  2. Report the effective integration time $T$ and justify it relative to the temporal span of the data (day 2 → day 6 = 4 biological days).

---

## Reproducibility Checklist

- [x] Random seeds fixed — `jax.random.PRNGKey(42)` at `experiment_h4_simulation.py:61`
- [ ] Hyperparameters logged — `step_size=0.015`, `max_steps=120`, `N_EVAL=600`, `n_anchors=2000`, `sigma=0.4`, `v0_scale=0.8` are hard-coded but not written to any output file or artifact.
- [x] Data preprocessing deterministic and versioned — `preprocess_weinreb.py` uses `np.random.seed(42)` for clone train/test split.
- [ ] Results include variance estimates — **No.** Single-run accuracy only; no bootstrap CI, no repeated seeds.
- [ ] Baselines are appropriate and fairly implemented — **No.** The Null (Riemannian) model receives $v_0 = \mathbf{0}$ while Randers receives $v_0 \propto W$. This is a confounded comparison (see Claim 4 above).

---

## Severity Summary

| # | Claim | Severity | Issue |
|---|-------|----------|-------|
| 3 | $v_0 = 0.8 \cdot W / \|W\|$ as shooting protocol | **CRITICAL** | Arbitrary magnitude, no ablation, no justification |
| 4 | Null baseline fairness | **CRITICAL** | Null receives $v_0 = \mathbf{0}$; comparison is confounded |
| 1 | Delta > 3% decision rule | WARNING | No variance estimate, no statistical test |
| 2 | N_TRAJ = 1 deterministic | WARNING | Ignores stochastic nature of fate; no fate probabilities |
| 5 | Shared fate centroids | WARNING | Mild optimistic bias from non-held-out centroids |
| 6 | Only 2 target fates | WARNING | Limited generalizability |
| 7 | No integrator convergence check | WARNING | RK4 accuracy unverified |

---

## Suggested Experiments

1. **Fix the null baseline (highest priority):** Give both Randers and Riemannian models the same initial velocity (RNA velocity push-forward or learned direction field). The Randers model should gain advantage through geodesic curvature, not initial condition.

2. **Bootstrap confidence intervals:** Resample the 600 test triples with replacement (1000×) and report accuracy ± 95% CI for both models. Apply McNemar's test for paired comparison.

3. **Initial velocity ablation:** Sweep $\|v_0\| \in \{0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0\}$ and report accuracy for both models. This reveals whether the result depends on the specific magnitude choice.

4. **Stochastic shooting:** Perturb $v_0$ with Gaussian noise ($\sigma \in \{0.01, 0.05, 0.1\}$), shoot $K=50$ trajectories per start cell, and compute fate probability via soft nearest-centroid assignment. Evaluate with cross-entropy.

5. **Extended fate panel:** Include all fates with ≥50 day-6 cells. Report per-fate accuracy and macro-averaged accuracy.

6. **Integrator convergence:** Run with `max_steps` ∈ {60, 120, 240, 480} and plot accuracy vs. steps to demonstrate convergence.

7. **Comparison against PRESCIENT/WOT:** Benchmark against established trajectory inference methods on the same Weinreb test set with the same evaluation metric.

8. **Leave-one-out centroids:** Recompute fate centroids excluding each test cell's own day-6 endpoint.
