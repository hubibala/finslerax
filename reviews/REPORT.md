# HAMTools Full Codebase Review — Master Report

**Date:** 2026-05-15  
**Scope:** 23 source files + 5 experiment scripts  
**Dimensions:** Math (23), Code (23), Docs (23), Science (5) — 74 individual reviews

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

*74 individual reviews written to `reviews/{math,code,docs,science}/`. Generated by HAMTools Review Orchestrator — 2026-05-15.*
