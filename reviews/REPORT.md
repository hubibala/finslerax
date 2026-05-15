# HAMTools Full Codebase Review — Master Report

**Date:** 2026-05-15 (updated)  
**Scope:** 23 source files + 5 experiment scripts + 16 test files + 13 demo/example scripts  
**Dimensions:** Math (39+), Code (39+), Docs (36+), Science (5) — 161 individual reviews

---

## Executive Summary

HAMTools implements a novel differentiable Finsler geometry framework in JAX. The **core geometric engine** (metric tensor, spray coefficients, geodesic ODE, parallel transport) is **mathematically sound** — the Euler-Lagrange implicit spray solve, Berwald connection, and Zermelo-Randers formulas are all correctly implemented.

However, the review uncovered **9 CRITICAL findings**, **12 BUGs**, and **20+ WARNINGs** across four dimensions. The most severe issues are:

1. **Silent nullification of the Finsler advantage (C4):** RNA velocity is never projected to PCA space, so the wind field `W = 0` everywhere — the entire directional metric collapses to Riemannian.
2. **Wrong Lagrangian in E-L loss (C1):** `EulerLagrangeResidualLoss` hand-codes a formula that doesn't match the actual Randers metric.
3. **Unregularized VAE posterior (C3):** KL divergence is missing the `½‖μ‖²` term — the posterior mean is entirely unregularized.
4. **Systematic experimental methodology issues (C5–C9):** Data leakage, hardcoded conclusions, confounded baselines, and no multiple-comparison correction.

Documentation quality is uniformly poor (~80% of public symbols undocumented), and test coverage has critical gaps (`curvature.py`: 0%, 11/16 loss classes: 0%, no differentiability-through-solver tests).

---

## Critical Findings (P0 — Must Fix Before Release)

| ID | File | Dimension | Finding |
|----|------|-----------|---------|
| C1 | `training/losses.py:275` | Math | `EulerLagrangeResidualLoss` hand-codes the wrong Lagrangian — doesn't match the Zermelo-Randers metric formula. Fix: replace with `model.metric.energy()` |
| C2 | `geometry/curvature.py:93` | Math | `scalar_curvature()` uses Euclidean `jnp.dot` for Gram-Schmidt instead of the metric inner product `g_ij` — produces incorrect flag curvature for any non-Euclidean metric |
| C3 | `bio/vae.py:41–43` | Math | `kl_divergence_std_normal` omits the `½‖μ‖²` term — posterior mean is entirely unregularized |
| C4 | `bio/data.py` | Math / Science | Velocity is never projected from gene space to PCA space — `W = 0` everywhere, silently disabling the entire Finsler directional metric |
| C5 | `weinreb_experiment.py:636` | Science | Unknown provenance of validation file `weinreb_lineage_triples.npy`; likely contains training data. Partial circularity: velocity derived from same clonal descendants used for validation |
| C6 | `experiment_h1_geometric.py:87` | Science | Hardcoded string "successfully extracts meaningful topology" printed unconditionally regardless of metric values; no baselines; single run with no variance estimate |
| C7 | `experiment_h2_directional.py` | Science | 15 Wilcoxon tests with no FDR/Bonferroni correction; tautological design — `L_fwd < L_bwd` is expected by construction from the wind field |
| C8 | `experiment_h3_discriminative.py` | Science | Transductive leakage — wind field built from the full dataset, including training clones that neighbour test cells |
| C9 | `experiment_h4_simulation.py` | Science | Confounded baseline: Randers receives `v₀ ∝ W` while the null receives `v₀ = 0` — tests "velocity vs. no velocity" rather than "Finsler vs. Riemannian" |

Individual reviews: [math/losses.md](math/losses.md) · [math/curvature.md](math/curvature.md) · [math/vae.md](math/vae.md) · [math/data.md](math/data.md) · [science/weinreb_experiment.md](science/weinreb_experiment.md) · [science/experiment_h1_geometric.md](science/experiment_h1_geometric.md) · [science/experiment_h2_directional.md](science/experiment_h2_directional.md) · [science/experiment_h3_discriminative.md](science/experiment_h3_discriminative.md) · [science/experiment_h4_simulation.md](science/experiment_h4_simulation.md)

---

## Bug Findings (P1 — Fix Before Next Release)

| ID | File | Finding |
|----|------|---------|
| B1 | `geometry/transport.py` | `dt = 1/len(path_x)` is off-by-one — should be `1/(len(path_x) - 1)`. Causes ~5% systematic undershoot at N=20 |
| B2 | `geometry/transport.py` | `Connection` / `BerwaldConnection` are plain Python classes, not `eqx.Module` — breaks JAX transform composability |
| B3 | `geometry/manifold.py` | `_safe_norm_ratio_jvp` uses bare `jnp.linalg.norm` instead of `safe_norm` — NaN gradients at the origin |
| B4 | `geometry/curvature.py:92,95` | `jnp.linalg.norm` instead of `safe_norm` — NaN gradients at zero vectors. Module also has **zero test coverage** |
| B5 | `geometry/surfaces.py` | `Torus` and `Paraboloid` use positional indexing (`x[:2]`, `x[0]`) instead of `x[..., :2]` — breaks all batched inputs (5 instances) |
| B6 | `geometry/zoo.py` | Dead zero-vector guard: `safe_norm` returns ~1e-6 at `v=0`, exceeding the 1e-7 threshold — `F(x, 0) ≈ 3e-5` instead of `0` |
| B7 | `solvers/geodesic.py` | `step_size` constructor parameter is stored but never used — `dt` is always `t_max / max_steps` |
| B8 | `solvers/avbd.py:62` | `.astype(jnp.int32)` called on a traced value → `ConcretizationTypeError` under JIT |
| B9 | `training/losses.py` | Multiple loss classes return bare Python `0.0` instead of `jnp.float32(0.0)` — type mismatch under `jax.grad` |
| B10 | `models/learned.py` | `KernelWindField.__call__` has dead `if/else` branches that execute identical code |
| B11 | `bio/train_geodesic.py` | 3 blocking bugs: plain Python class decorated with `@eqx.filter_jit` (crashes at trace time), invisible optimizer state mutation, missing type guard on private API |
| B12 | `bio/train_joint.py` | 4 bugs: shared optimizer state across mismatched phases, PRNG key reuse per batch, no parameter freezing, dropped diagnostic stat |

Individual reviews: [code/transport.md](code/transport.md) · [code/manifold.md](code/manifold.md) · [code/curvature.md](code/curvature.md) · [code/surfaces.md](code/surfaces.md) · [code/zoo.md](code/zoo.md) · [code/geodesic.md](code/geodesic.md) · [code/avbd.md](code/avbd.md) · [code/losses.md](code/losses.md) · [code/learned.md](code/learned.md) · [code/train_geodesic.md](code/train_geodesic.md) · [code/train_joint.md](code/train_joint.md)

---

## Warning Findings (P2 — Quality Issues)

### Numerical Stability

- **W1** — Hardcoded epsilons (`1e-4`, `1e-9`, `1e-10`, `1e-12`) in 10+ files instead of the canonical `PSD_EPS` / `GRAD_EPS` from `utils/math.py`
- **W2** — `safe_norm`'s `maximum`-clamp breaks positive 1-homogeneity near `v = 0` and creates a kink in higher-order derivatives (affects Berwald connection)
- **W3** — `PSD_EPS` is defined in `utils/math.py` but never imported by any downstream file
- **W4** — `jnp.linalg.norm` used in gradient-carrying paths in `curvature.py`, `geodesic.py`, `zoo.py`, and `fields.py`

### Architecture

- **W5** — No batch-dimension support in `transport`, `curvature`, `mesh`, `fields` (ARCH_SPEC violation)
- **W6** — Augmented Lagrangian in `avbd.py` is a stub — dual variables are never updated; solver degenerates to a fixed quadratic penalty
- **W7** — Dead parameters: `tol`, `energy_tol`, `momentum` in `avbd.py`; `step_size` in `geodesic.py`
- **W8** — `train_geodesic.py` and `train_joint.py` appear to be dead legacy code superseded by `HAMPipeline`

### Mathematical

- **W9** — Spray loss (`‖G‖² → 0`) pushes the metric toward flatness rather than encouraging trajectories to be geodesic
- **W10** — Alignment losses use Euclidean / Minkowski norms instead of the Finsler / Riemannian norm
- **W11** — `scalar_curvature` samples only 1 sectional curvature — mathematically wrong for n > 2
- **W12** — RK4 integrator applies manifold projection only after full steps (not at intermediate stages), reducing convergence from O(Δt⁴) to O(Δt²) on curved submanifolds

### Spec Discrepancies

- **W13** — `MATH_SPEC` §2.1 is missing a factor of 2 on `∂E/∂xˡ`; §2.2 has a sign error (`2G` should be `-2G`). The code is correct in both cases.
- **W14** — `ARCH_SPEC` module tree does not match the actual directory structure
- **W15** — `ARCH_SPEC` §3 references a `LearnedFinsler` class that does not exist

### Documentation

- **W16** — ~80% of public symbols have no docstring at all
- **W17** — No module-level docstrings anywhere in the codebase
- **W18** — No `Args` / `Returns` documentation on any public function

---

## Test Coverage Gaps

| Module | Gap |
|--------|-----|
| `geometry/curvature.py` | **Zero test coverage** — the only geometry module with no tests |
| `training/losses.py` | 11 of 16 loss classes have zero dedicated tests |
| `models/learned.py` | Only 1 of 7 public symbols tested (`NeuralRanders`) |
| `solvers/geodesic.py` | No differentiability-through-solver test (critical: solver is used inside training loops) |
| `solvers/avbd.py` | No `jax.grad` through `solve` test |
| `geometry/manifold.py` | Default `log_map` with custom JVP has zero direct tests |
| `vis/` | 0 of 9 public functions tested |
| `bio/data.py` | Zero test coverage for `DataLoader` |
| All modules | No `jit` / `vmap` / `grad` compatibility tests |

---

## File-Level Verdict Matrix

| File | Math | Code | Docs | Science |
|------|:----:|:----:|:----:|:-------:|
| `geometry/metric.py` | ✅ 3W | ✅ 5R | ⚠️ | — |
| `geometry/manifold.py` | ✅ 2W | 🐛 | ⚠️ | — |
| `geometry/transport.py` | ✅ 2W | 🐛 🐛 | ⚠️ | — |
| `geometry/curvature.py` | 🔴 | 🐛 | ⚠️ | — |
| `geometry/surfaces.py` | ✅ 3W | 🐛 ×5 | ⚠️ | — |
| `geometry/zoo.py` | ✅ 3W | 🐛 | ⚠️ | — |
| `geometry/mesh.py` | ✅ 3W | ⚠️ | ⚠️ | — |
| `solvers/geodesic.py` | ✅ 4W | 🐛 | ⚠️ | — |
| `solvers/avbd.py` | ✅ 3W | 🐛 | ⚠️ | — |
| `models/learned.py` | ✅ 3W | 🐛 | ⚠️ | — |
| `nn/networks.py` | ✅ 1W | ✅ 3R | ⚠️ | — |
| `training/losses.py` | 🔴 | 🐛 🐛 | ⚠️ | — |
| `training/pipeline.py` | ✅ | ✅ 5R | ⚠️ | — |
| `bio/vae.py` | 🔴 🔴 | 🐛 | ⚠️ | — |
| `bio/data.py` | 🔴 | 🐛 🐛 | ⚠️ | — |
| `bio/train_geodesic.py` | ✅ 3W | 🐛 ×3 | ⚠️ | — |
| `bio/train_joint.py` | ⚠️ | 🐛 ×4 | ⚠️ | — |
| `bio/train_modular.py` | ✅ 3W | ⚠️ | ⚠️ | — |
| `bio/check_data.py` | — | ⚠️ | ⚠️ | — |
| `utils/math.py` | ✅ 3W | ✅ 1R | ⚠️ | — |
| `vis/vis.py` | — | ✅ 3R | ⚠️ | — |
| `vis/hyperbolic.py` | ✅ 2W | 🐛 | ⚠️ | — |
| `sim/fields.py` | ✅ 1W | ⚠️ | ⚠️ | — |
| `weinreb_experiment.py` | — | — | — | 🔴 🔴 |
| `experiment_h1_geometric.py` | — | — | — | 🔴 |
| `experiment_h2_directional.py` | — | — | — | 🔴 |
| `experiment_h3_discriminative.py` | — | — | — | 🔴 |
| `experiment_h4_simulation.py` | — | — | — | 🔴 |

**Legend:** 🔴 CRITICAL &nbsp;·&nbsp; 🐛 BUG &nbsp;·&nbsp; ⚠️ WARNING / Needs Work &nbsp;·&nbsp; ✅ OK &nbsp;·&nbsp; W = warnings &nbsp;·&nbsp; R = risks

---

## Recommended Fix Priority

### Immediate (P0)

1. **C4** — Add PCA velocity projection in `bio/data.py` (silently disables the entire Finsler framework)
2. **C1** — Replace hand-coded Lagrangian in `EulerLagrangeResidualLoss` with `model.metric.energy()`
3. **C3** — Add `½‖μ‖²` term to `kl_divergence_std_normal` in `bio/vae.py`
4. **C2** — Replace `jnp.dot` with metric inner product `g_ij` in `scalar_curvature()`

### Short-term (P1)

5. **B1** — Fix `dt` off-by-one in `geometry/transport.py`
6. **B8** — Fix traced `.astype(jnp.int32)` in `solvers/avbd.py`
7. **B9** — Replace bare `return 0.0` with `jnp.zeros(())` in all loss classes
8. **B3, B4** — Replace all `jnp.linalg.norm` with `safe_norm` in gradient-carrying paths
9. **B5** — Fix positional indexing in `Torus` / `Paraboloid` to support batching (`x[..., :2]`)
10. **W1 / W3** — Import and use canonical epsilon constants (`PSD_EPS`, `GRAD_EPS`) everywhere

### Medium-term (P2)

11. **C5–C9** — Redesign experiments: fix data splits, add external baselines (PRESCIENT, WOT), add multiple-comparison correction
12. **W8** — Deprecate `train_geodesic.py` and `train_joint.py` in favour of `HAMPipeline`
13. Add test coverage for `curvature.py`, all loss classes, and solver differentiability
14. Documentation pass — add module-level and function-level docstrings with `Args` / `Returns`
15. Update `MATH_SPEC` and `ARCH_SPEC` to match the actual implementation

---

## Individual Review Index

### Math Reviews

[metric.md](math/metric.md) · [manifold.md](math/manifold.md) · [transport.md](math/transport.md) · [curvature.md](math/curvature.md) · [surfaces.md](math/surfaces.md) · [zoo.md](math/zoo.md) · [mesh.md](math/mesh.md) · [geodesic.md](math/geodesic.md) · [avbd.md](math/avbd.md) · [learned.md](math/learned.md) · [networks.md](math/networks.md) · [losses.md](math/losses.md) · [pipeline.md](math/pipeline.md) · [vae.md](math/vae.md) · [data.md](math/data.md) · [train\_geodesic.md](math/train_geodesic.md) · [train\_joint.md](math/train_joint.md) · [train\_modular.md](math/train_modular.md) · [math.md](math/math.md) · [hyperbolic.md](math/hyperbolic.md) · [fields.md](math/fields.md)

### Code Reviews

[metric.md](code/metric.md) · [manifold.md](code/manifold.md) · [transport.md](code/transport.md) · [curvature.md](code/curvature.md) · [surfaces.md](code/surfaces.md) · [zoo.md](code/zoo.md) · [mesh.md](code/mesh.md) · [geodesic.md](code/geodesic.md) · [avbd.md](code/avbd.md) · [learned.md](code/learned.md) · [networks.md](code/networks.md) · [losses.md](code/losses.md) · [pipeline.md](code/pipeline.md) · [vae.md](code/vae.md) · [data.md](code/data.md) · [train\_geodesic.md](code/train_geodesic.md) · [train\_joint.md](code/train_joint.md) · [train\_modular.md](code/train_modular.md) · [check\_data.md](code/check_data.md) · [math.md](code/math.md) · [vis.md](code/vis.md) · [hyperbolic.md](code/hyperbolic.md) · [fields.md](code/fields.md)

### Documentation Reviews

[metric.md](docs/metric.md) · [manifold.md](docs/manifold.md) · [transport.md](docs/transport.md) · [curvature.md](docs/curvature.md) · [surfaces.md](docs/surfaces.md) · [zoo.md](docs/zoo.md) · [mesh.md](docs/mesh.md) · [geodesic.md](docs/geodesic.md) · [avbd.md](docs/avbd.md) · [learned.md](docs/learned.md) · [networks.md](docs/networks.md) · [losses.md](docs/losses.md) · [pipeline.md](docs/pipeline.md) · [vae.md](docs/vae.md) · [data.md](docs/data.md) · [train\_geodesic.md](docs/train_geodesic.md) · [train\_joint.md](docs/train_joint.md) · [train\_modular.md](docs/train_modular.md) · [check\_data.md](docs/check_data.md) · [math.md](docs/math.md) · [vis.md](docs/vis.md) · [hyperbolic.md](docs/hyperbolic.md) · [fields.md](docs/fields.md)

### Science Audits

[weinreb\_experiment.md](science/weinreb_experiment.md) · [experiment\_h1\_geometric.md](science/experiment_h1_geometric.md) · [experiment\_h2\_directional.md](science/experiment_h2_directional.md) · [experiment\_h3\_discriminative.md](science/experiment_h3_discriminative.md) · [experiment\_h4\_simulation.md](science/experiment_h4_simulation.md)

---

## Addendum: Test Files Review (16 files — Math + Code)

**Date:** 2026-05-15  
**Scope:** All 16 test files in `tests/`  
**Dimensions:** Math (mathematical correctness of assertions) + Code (test quality and coverage)

### Action Required (CRITICAL / BUG from Tests)

| ID | File | Dim | Finding |
|----|------|-----|---------|
| T-C1 | `test_hyperbolic_vae.py` | Math | KL divergence formula in source `bio/vae.py:42-44` is missing geodesic distance term $d(\mu,o)^2/2$ and log-det Jacobian correction. Test only checks finiteness — cannot detect this. |
| T-C2 | `test_learned_metric.py` | Math | No test for positive 1-homogeneity $F(x,\lambda v) = \lambda F(x,v)$ or positive definiteness of $g_{ij}$ — both defining axioms of Finsler metrics (§1.1). |
| T-B1 | `test_geodesic.py` | Code/Math | `Sphere(1.0)` passes `1.0` as `intrinsic_dim` (creating $S^1$) instead of `radius` — should be `Sphere(radius=1.0)`. Also in line 82. |
| T-B2 | `test_geodesic_learning.py` | Code | `test_loss_decreases` compares losses from two *different* model initialisations (`PRNGKey(0)` vs `PRNGKey(2025)`) — doesn't verify training reduces loss for the same model. |
| T-B3 | `test_geodesic_learning.py` | Code | `test_vortex_direction` computes `eval_pts` and `W_pred` but never uses them; evaluation runs on training data. |
| T-B4 | `test_hyperbolic_vae.py` | Code | Lines 112-113 and 140-141 reuse the same PRNG key for both `x` and `v_rna`, producing identical arrays. |
| T-B5 | `test_joint_training.py` | Code | PRNG key reuse makes `X ≡ V`; `MockMetric` has broken MRO/init chain. |
| T-B6 | `test_network.py` | Code | `test_fourier_features` has no assertion on output difference — it's a no-op test. |
| T-B7 | `test_surfaces.py` | Code | PRNGKey reused without splitting in `test_sphere_high_dim`. |
| T-B8 | `test_transport.py` | Math | Wrong holonomy formula: code uses $2\pi\cos\theta$ instead of $2\pi(1-\cos\theta)$. Test passes by accident due to cosine insensitivity to the complement. |
| T-B9 | `test_zoo.py` | Code | `test_randers_zero_wind` uses `places=2`, far too loose. |
| T-B10 | `test_joint_training.py` | Code | 2 of 3 tests only check `assertIsInstance` — no mathematical invariant verified. |

### Test File Summaries

| File | Math Verdict | Code Verdict | Key Issues |
|------|:---:|:---:|-----------|
| `test_fields.py` | ✅ 3W | ⚠️ 1B 3R | Untested fields (`rossby_haurwitz`, `harmonic_vortices`); no JAX transform tests |
| `test_geodesic.py` | ⚠️ 3W | 🐛 2B 3R | `Sphere(1.0)` positional arg bug; ad-hoc `Plane` class; no spray ODE test |
| `test_geodesic_learning.py` | ⚠️ 4W | 🐛 2B 2R | Loss comparison across different models; eval on training data; no actual geodesic test |
| `test_hyperbolic_vae.py` | 🔴 1C 4W | 🐛 1B 6R | KL divergence gap; no round-trip tests; PRNG key reuse |
| `test_hyperboloid.py` | ⚠️ 5W | ⚠️ 3R 4S | No `exp_map`/`log_map`/`parallel_transport` tests; 3 core methods untested |
| `test_joint_training.py` | ⚠️ 2W 2N | 🐛 2B 6R | PRNG reuse; broken MockMetric; weak assertions |
| `test_learned_metric.py` | 🔴 2C 4W | 🐛 1B 3R | Missing Finsler axiom tests; only 1 of 7 public symbols tested |
| `test_mesh.py` | ✅ 2W 1N | ⚠️ 5R 4S | No batch/vmap tests; `get_face_weights` untested |
| `test_mesh_solver.py` | ✅ 2W 2N | ⚠️ 10R 5S | Only 2 of ~8 scenarios covered; loose endpoint tolerance |
| `test_metric.py` | ✅ 5W | ⚠️ 7R 5S | Spec errors in MATH_SPEC (code correct); no PD test; `energy()` untested |
| `test_network.py` | ⚠️ 3W | 🐛 1B 4R | No-op Fourier test; PSD floor not validated |
| `test_pipeline.py` | ✅ 2N | ⚠️ 4R 3S | Only 1 of 3 `batch_data` branches exercised |
| `test_solver.py` | ✅ 4W | ⚠️ 8R 4S | No IVP test; no energy conservation; no JIT/grad tests |
| `test_surfaces.py` | ✅ 5W | 🐛 1B 6R | Hyperboloid untested; no exp/log edge cases; PRNGKey reuse |
| `test_transport.py` | ⚠️ 5W | 🐛 1B 4R | Wrong holonomy formula; zero-connection tests only; `Christoffel` untested |
| `test_zoo.py` | ✅ 3W 3N | 🐛 1B 5R | Overly loose tolerances; no JAX transform tests |

### Cross-Cutting Test Patterns

1. **No JAX transform tests anywhere**: Not a single test file verifies `jit`/`vmap`/`grad` compatibility, despite ARCH_SPEC mandating batch-first, JIT-compatible, differentiable operations.
2. **Systematic PRNG key reuse**: At least 5 test files reuse PRNG keys without splitting, producing correlated or identical test data.
3. **Weak assertion patterns**: Multiple tests use `assertIsInstance` or finiteness checks instead of verifying mathematical invariants.
4. **No negative/edge-case tests**: Zero-vector inputs, degenerate geometries, antipodal points, and near-singular metrics are systematically untested.
5. **`unittest` vs `pytest`**: All test files use `unittest.TestCase` instead of pytest idioms, limiting parametrization and fixture capabilities.
6. **Debug `print` statements**: Stray `print()` calls in at least 6 test files.
7. **Coverage gaps vs source**: `curvature.py` has zero coverage; `losses.py` has 11/16 classes untested; `learned.py` has 1/7 symbols tested.

---

## Addendum: Demo & Example Files Review (13 files — Doc + Code + Math)

**Date:** 2026-05-15  
**Scope:** All non-experiment example scripts in `examples/`  
**Dimensions:** Documentation, Code quality, Mathematical correctness

### Action Required (CRITICAL / BUG from Examples)

| ID | File | Dim | Finding |
|----|------|-----|---------|
| E-C1 | `demo_learned_wind.py` | Math | Energy loss $\mathcal{L}_E = \mathbb{E}[\frac{1}{2}F^2(x, v)]$ evaluated at observed wind has global minimum at $W_{\text{learned}} \equiv 0$, not $W_{\text{true}}$. The loss does not uniquely identify the ground-truth wind field. |
| E-C2 | `weinreb_vae.py` | Math | KL divergence missing $\frac{1}{2}\|\mu\|^2$ term (same as C3 above, inherited from `bio/vae.py`). Also: `batch[0]` feeds random cells (not lineage day-2 cells) into coherence loss, making midpoint constraint meaningless. |
| E-B1 | `demo_discrete_zermelo.py` | Code | Cross-metric energy evaluation: Riemannian path's energy computed under Randers metric, but label says "Energy Riemannian path". |
| E-B2 | `demo_trajectories.py` | Code | `isinstance(trajectory, (tuple, NamedTuple))` — `typing.NamedTuple` is not a valid runtime type; breaks on Python ≥ 3.11. |
| E-B3 | `demo_weinreb_vis.py` | Code | PRNG key not re-split per branch — all sibling children collapse to the same point. |
| E-B4 | `demo_zermelo.py` | Code | `Sphere(radius)` passes `1.0` as `intrinsic_dim` — creates $S^1$ instead of $S^2$. |
| E-B5 | `demo_learned_wind.py` | Code | Loss computed twice per step — should use `eqx.filter_value_and_grad`. |
| E-B6 | `preprocess_weinreb.py` | Code | Final summary references a file (`weinreb_lineage_pairs.npy`) that is never created. |
| E-B7 | `weinreb_smoke_test.py` | Code | `check_full_validation` has a vacuous `for k in required_keys: if k in results: pass` loop — test always passes. |
| E-B8 | `weinreb_vae.py` | Code | Dead `data_dim` assignment; division-by-near-zero in `VelocityConsistencyLoss`. |
| E-B9 | `plot_publication_figs.py` | Code | Dead variable `quiv_colors` computed but never used; redundant arc-length JIT call. |
| E-B10 | `plot_weinreb_cell_types.py` | Code | Unvalidated fallback data path produces opaque crash when neither `.h5ad` file exists. |

### Example File Summaries

| File | Math | Code | Docs | Key Issues |
|------|:---:|:---:|:---:|-----------|
| `demo_discrete_zermelo.py` | ⚠️ 4W | 🐛 1B 2R | ⚠️ 6M 6U | `Sphere()` positional arg; cross-metric energy; no module docstring |
| `demo_learned_wind.py` | 🔴 1C 2W | 🐛 1B 3R | ⚠️ 7M 4U | Energy loss has wrong minimum; double forward pass; no docs |
| `demo_trajectories.py` | ⚠️ 2W 2N | 🐛 1B 4R | ⚠️ 7M 4U | `NamedTuple` runtime check breaks; velocity scaling fragile |
| `demo_vortex.py` | ✅ 1W | ✅ 1R 4S | ⚠️ 7M 5U | Well-structured code; needs docs and `beta` disambiguation |
| `demo_weinreb_vis.py` | ⚠️ 1W 1N | 🐛 1B 2R | ⚠️ 8M 6U | PRNG bug; Euclidean norm for Minkowski tangent; misleading filename |
| `demo_zermelo.py` | ⚠️ 2W 2N | 🐛 1B 1R | ⚠️ 6M 2I | `Sphere(radius)` bug; stale wind strength comment |
| `plot_publication_figs.py` | ⚠️ 2W 4N | 🐛 1B 3R | ⚠️ 8M 5U | "$E$" label should be "$L$"; dead code; raw wind quiver |
| `plot_weinreb_cell_types.py` | ⚠️ 2W 2N | 🐛 1B 5R | ⚠️ 5M 10U | Backward-filter bias; per-cell bandwidth distortion |
| `plot_weinreb_destinations.py` | ⚠️ 2W | ⚠️ 4R 5S | ⚠️ 4M 5U | k-NN bandwidth self-contamination; no input validation |
| `preprocess_weinreb.py` | ⚠️ 2W 2N | 🐛 1B 5R | ⚠️ 2I 9M 6U | Docstring-code mismatch; ghost file reference; PCA whitening concern |
| `train_vae_ablation.py` | ✅ 1N (🔴 via vae.py) | ⚠️ 2R 7S | ⚠️ 5M 2U | Inherits KL bug; fragile paths; discarded history |
| `weinreb_smoke_test.py` | ⚠️ 3W | 🐛 1B 2R | ⚠️ 2I 6M 2U | No-op validation loop; wrong filename in docstring |
| `weinreb_vae.py` | 🔴 2C 2W | 🐛 1B 7R | ⚠️ 2I 9M 8U | Missing KL $\mu^2$ term; mismatched coherence data; 6 untested loss classes |

### Cross-Cutting Example Patterns

1. **No module-level docstrings**: Every single example/demo file lacks a proper module docstring explaining purpose, usage, prerequisites, and expected output.
2. **Recurring `Sphere(radius)` bug**: Appears in `demo_zermelo.py`, `demo_discrete_zermelo.py`, and `test_geodesic.py` — passing radius as positional arg to `intrinsic_dim`.
3. **Raw vs. squashed wind visualization**: Multiple demos (`demo_vortex.py`, `demo_learned_wind.py`, `plot_publication_figs.py`) plot the raw `w_net` output instead of the `tanh`-squashed wind actually used by the Randers metric, overstating anisotropy.
4. **KL divergence bug propagation**: The missing $\frac{1}{2}\|\mu\|^2$ term from `bio/vae.py` propagates through `weinreb_vae.py` and `train_vae_ablation.py`.
5. **Undocumented magic numbers**: Hyperparameters, tolerance values, and scaling constants are rarely justified across all example files.
6. **Missing `__main__` guards**: Most scripts run at import time, making them untestable and unsuitable for module-level imports.

---

## Updated Review Coverage

### Tests (16 files × 2 dimensions = 32 reviews)

| File | Math | Code |
|------|:---:|:---:|
| `test_fields.py` | [test\_fields.md](math/test_fields.md) | [test\_fields.md](code/test_fields.md) |
| `test_geodesic.py` | [test\_geodesic.md](math/test_geodesic.md) | [test\_geodesic.md](code/test_geodesic.md) |
| `test_geodesic_learning.py` | [test\_geodesic\_learning.md](math/test_geodesic_learning.md) | [test\_geodesic\_learning.md](code/test_geodesic_learning.md) |
| `test_hyperbolic_vae.py` | [test\_hyperbolic\_vae.md](math/test_hyperbolic_vae.md) | [test\_hyperbolic\_vae.md](code/test_hyperbolic_vae.md) |
| `test_hyperboloid.py` | [test\_hyperboloid.md](math/test_hyperboloid.md) | [test\_hyperboloid.md](code/test_hyperboloid.md) |
| `test_joint_training.py` | [test\_joint\_training.md](math/test_joint_training.md) | [test\_joint\_training.md](code/test_joint_training.md) |
| `test_learned_metric.py` | [test\_learned\_metric.md](math/test_learned_metric.md) | [test\_learned\_metric.md](code/test_learned_metric.md) |
| `test_mesh.py` | [test\_mesh.md](math/test_mesh.md) | [test\_mesh.md](code/test_mesh.md) |
| `test_mesh_solver.py` | [test\_mesh\_solver.md](math/test_mesh_solver.md) | [test\_mesh\_solver.md](code/test_mesh_solver.md) |
| `test_metric.py` | [test\_metric.md](math/test_metric.md) | [test\_metric.md](code/test_metric.md) |
| `test_network.py` | [test\_network.md](math/test_network.md) | [test\_network.md](code/test_network.md) |
| `test_pipeline.py` | [test\_pipeline.md](math/test_pipeline.md) | [test\_pipeline.md](code/test_pipeline.md) |
| `test_solver.py` | [test\_solver.md](math/test_solver.md) | [test\_solver.md](code/test_solver.md) |
| `test_surfaces.py` | [test\_surfaces.md](math/test_surfaces.md) | [test\_surfaces.md](code/test_surfaces.md) |
| `test_transport.py` | [test\_transport.md](math/test_transport.md) | [test\_transport.md](code/test_transport.md) |
| `test_zoo.py` | [test\_zoo.md](math/test_zoo.md) | [test\_zoo.md](code/test_zoo.md) |

### Demo/Example Scripts (13 files × 3 dimensions = 39 reviews)

| File | Math | Code | Docs |
|------|:---:|:---:|:---:|
| `demo_discrete_zermelo.py` | [demo\_discrete\_zermelo.md](math/demo_discrete_zermelo.md) | [demo\_discrete\_zermelo.md](code/demo_discrete_zermelo.md) | [demo\_discrete\_zermelo.md](docs/demo_discrete_zermelo.md) |
| `demo_learned_wind.py` | [demo\_learned\_wind.md](math/demo_learned_wind.md) | [demo\_learned\_wind.md](code/demo_learned_wind.md) | [demo\_learned\_wind.md](docs/demo_learned_wind.md) |
| `demo_trajectories.py` | [demo\_trajectories.md](math/demo_trajectories.md) | [demo\_trajectories.md](code/demo_trajectories.md) | [demo\_trajectories.md](docs/demo_trajectories.md) |
| `demo_vortex.py` | [demo\_vortex.md](math/demo_vortex.md) | [demo\_vortex.md](code/demo_vortex.md) | [demo\_vortex.md](docs/demo_vortex.md) |
| `demo_weinreb_vis.py` | [demo\_weinreb\_vis.md](math/demo_weinreb_vis.md) | [demo\_weinreb\_vis.md](code/demo_weinreb_vis.md) | [demo\_weinreb\_vis.md](docs/demo_weinreb_vis.md) |
| `demo_zermelo.py` | [demo\_zermelo.md](math/demo_zermelo.md) | [demo\_zermelo.md](code/demo_zermelo.md) | [demo\_zermelo.md](docs/demo_zermelo.md) |
| `plot_publication_figs.py` | [plot\_publication\_figs.md](math/plot_publication_figs.md) | [plot\_publication\_figs.md](code/plot_publication_figs.md) | [plot\_publication\_figs.md](docs/plot_publication_figs.md) |
| `plot_weinreb_cell_types.py` | [plot\_weinreb\_cell\_types.md](math/plot_weinreb_cell_types.md) | [plot\_weinreb\_cell\_types.md](code/plot_weinreb_cell_types.md) | [plot\_weinreb\_cell\_types.md](docs/plot_weinreb_cell_types.md) |
| `plot_weinreb_destinations.py` | [plot\_weinreb\_destinations.md](math/plot_weinreb_destinations.md) | [plot\_weinreb\_destinations.md](code/plot_weinreb_destinations.md) | [plot\_weinreb\_destinations.md](docs/plot_weinreb_destinations.md) |
| `preprocess_weinreb.py` | [preprocess\_weinreb.md](math/preprocess_weinreb.md) | [preprocess\_weinreb.md](code/preprocess_weinreb.md) | [preprocess\_weinreb.md](docs/preprocess_weinreb.md) |
| `train_vae_ablation.py` | [train\_vae\_ablation.md](math/train_vae_ablation.md) | [train\_vae\_ablation.md](code/train_vae_ablation.md) | [train\_vae\_ablation.md](docs/train_vae_ablation.md) |
| `weinreb_smoke_test.py` | [weinreb\_smoke\_test.md](math/weinreb_smoke_test.md) | [weinreb\_smoke\_test.md](code/weinreb_smoke_test.md) | [weinreb\_smoke\_test.md](docs/weinreb_smoke_test.md) |
| `weinreb_vae.py` | [weinreb\_vae.md](math/weinreb_vae.md) | [weinreb\_vae.md](code/weinreb_vae.md) | [weinreb\_vae.md](docs/weinreb_vae.md) |

---

*161 individual reviews written to `reviews/{math,code,docs,science}/`. Generated by HAMTools Review Orchestrator — 2026-05-15.*
