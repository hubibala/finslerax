# Code Review: `examples/experiment_gahtan_phase1.py`
**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-17  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

The Phase 1 experiment script is well-organized, with clear separation of data generation, training, evaluation, and visualization. The training loop correctly uses `eqx.filter_jit`, `eqx.filter_value_and_grad`, and `optax` following HAMTools conventions. However, there are two significant performance issues (recompilation on every mini-batch due to dynamic shapes, and a redundant full-grid AVBD solve at evaluation time), one numerical concern in the ground-truth arrival time computation, and several JAX-correctness issues in the TV regularization and visualization code.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `examples/experiment_gahtan_phase1.py:299` | `train_step` is marked `@eqx.filter_jit` but receives `batch_x` and `batch_t` whose **shapes change** when `n_obs` is not divisible by `batch_size` (last batch may be smaller) or when `density` changes across calls. Each unique shape triggers a full XLA recompilation. In the current code, `batch_size` is fixed via `min(cfg['obs_per_train_step'], n_obs)`, so within a single `train_metric` call the shapes are constant. **However**, across different `density` values in the density sweep, the same `train_step` function object is reused with different `reg_pts` shapes (since `n_reg = min(64, n_obs)` varies). This causes recompilation for each density. | Pad `reg_pts` to a fixed shape (e.g., always 64 points, repeating if needed) to avoid recompilation. Alternatively, define `train_step` inside `train_metric` so each density gets its own jitted function. Currently `train_step` is already defined inside `train_metric`, so this is only a concern if the function is somehow cached — verify that Python scoping creates a new closure each call. |
| 2 | **RISK** | `examples/experiment_gahtan_phase1.py:222–228` | `tv_regularization` uses `jax.jacfwd(get_metric_value)` over `sample_points` vmapped. For a 2D metric ($H \in \mathbb{R}^{2 \times 2}$), the Jacobian is shape $(4, 2)$ per point, which is cheap. But `get_metric_value` contains `hasattr(metric, '_get_zermelo_data')` — a Python-time check inside a function that will be vmapped and potentially jitted. This is technically fine (traced once, then baked in), but if `tv_regularization` is ever called with a different metric type in the same Python session, the cached trace would use the wrong branch. Since `train_step` is `@eqx.filter_jit` and captures `tv_lambda` from the closure, the `if tv_lambda > 0` branch is also a trace-time decision. | This is acceptable for the experiment script's use case (single metric type per run). Document the assumption. |
| 3 | **RISK** | `examples/experiment_gahtan_phase1.py:157–176` | `compute_true_arrival_times` computes refracted distances using a straight-line approximation of the optimal crossing point. The docstring correctly notes this is an upper bound, not the true Snell's-law distance. For the central source at $(0.5, 0.5)$ — which lies exactly on the boundary — `d1 = |0.5 - 0.5| = 0.0`, making `frac = 0 / max(0 + d2, 1e-10)` and the refracted segment degenerates. This means **all points** will use `dist_straight` (since `same_side` is always true when $s_x = \text{boundary}$), but `g_source` will be `jnp.where(0.5 < 0.5, 1.0, 2.0) = 2.0`. So every target gets `sqrt(2.0) * ||t - s||` regardless of which side the target is on. This produces **incorrect** ground-truth arrival times for targets in the $g=1$ region. | Move the source off the boundary (e.g., `source = jnp.array([0.25, 0.5])`) or fix the `same_side` logic to handle `sx == boundary` correctly. For a strict comparison with Gahtan et al., the ground truth should use a proper Dijkstra/FMM solve on the grid. |
| 4 | **RISK** | `examples/experiment_gahtan_phase1.py:315` | `batch_idx = jax.random.choice(k_batch, n_obs, shape=(batch_size,), replace=False)` — when `batch_size == n_obs`, `replace=False` is fine. But when `n_obs < batch_size` (which can't happen due to the `min` on line 305), this would crash. More subtly, `jax.random.choice` with `replace=False` uses a shuffle internally which has $O(n)$ complexity in `n_obs`, not `batch_size`. For `n_obs = 6400` (80×80 grid at 100% density), this creates a 6400-element permutation on every step. Not a bug, but a performance note. | For large `n_obs`, consider pre-generating epoch-level permutations outside the jitted function. |
| 5 | **RISK** | `examples/experiment_gahtan_phase1.py:583–601` | `plot_arrival_time_comparison` calls `solver.solve` inside `single_dist`, then vmaps it over 200 grid points. This is done **outside** a `jit` context (pure Python loop in the plotting code). Each vmap call will trigger a full JIT compilation of the AVBD solver. Since this is a visualization step (not training), it's not performance-critical, but it could take several minutes on a CPU. | Wrap the vmap call in `jax.jit` explicitly, or note the expected runtime in the script output. |
| 6 | **RISK** | `examples/experiment_gahtan_phase1.py:247` | `obs_idx = jax.random.choice(k_init, n_total, shape=(n_obs,), replace=False)` followed by `obs_idx = jnp.sort(obs_idx)`. The sort is used "for reproducibility" per the comment, but sorting a random subset does not improve reproducibility — the PRNG key already determines the subset. However, the sort does ensure that `x_train` and `t_train` are in a consistent spatial order, which may affect AVBD solver convergence if the solver's initialization depends on the input ordering. This is harmless but misleading. | Remove the sort or clarify the comment (e.g., "for deterministic ordering of training data"). |
| 7 | **STYLE** | `examples/experiment_gahtan_phase1.py:60–61` | `from ham.geometry.manifolds import EuclideanSpace` — the ARCH_SPEC lists the module as `geometry/manifold.py` (singular), but the actual import uses `manifolds` (plural). This is correct for the codebase (the directory is `manifolds/`) but inconsistent with the spec. | Not actionable for this file — the spec should be updated. |
| 8 | **STYLE** | `examples/experiment_gahtan_phase1.py:280` | `schedule = optax.cosine_decay_schedule(cfg['lr'], cfg['n_train_steps'])` — `cosine_decay_schedule` has been renamed to `optax.schedules.cosine_decay_schedule` in newer Optax versions. The current import works but may trigger deprecation warnings. | Use `optax.schedules.cosine_decay_schedule` or pin the Optax version in `pyproject.toml`. |
| 9 | **STYLE** | `examples/experiment_gahtan_phase1.py:46` | `import numpy as np` alongside `jax.numpy as jnp`. The `np` import is used only for plotting (`np.array` conversions for matplotlib). This is fine but should be noted — all computation should stay in `jnp` until the final host-transfer for plotting. Currently this is done correctly. | No action needed. |

## Test Coverage Assessment

`experiment_gahtan_phase1.py` is an experiment script, not a library module, so it does not have a dedicated test file. Coverage is assessed via its integration usage of library components:

| Component Used | Tested Elsewhere? | Notes |
|---|---|---|
| `ArrivalTimeLoss` | **Yes** | `tests/test_arrival_time_loss.py` — see companion review |
| `AVBDSolver` | **Yes** | `tests/test_solver.py`, `tests/test_avbd.py` |
| `NeuralRanders` | **Yes** | `tests/test_learned_metric.py`, `tests/test_network.py` |
| `EuclideanSpace` | **Yes** | `tests/test_manifold.py` |
| `tv_regularization` | **No** | Defined inline in the experiment; no test coverage |
| `compute_true_arrival_times` | **No** | Ground-truth generator is untested; Issue #3 indicates a bug |
| `piecewise_metric_field` | **No** | Trivial function, but used as ground truth |
| `train_metric` | **No** | Integration test; would benefit from a `--quick` smoke test in CI |
| `evaluate_recovery` | **No** | Evaluation function untested |

### Gap Analysis

The script lacks a `--quick` CI smoke test. The `get_config(quick=True)` path exists but is not exercised by any automated test. The ground-truth arrival time computation (Issue #3) has a critical bug that would cause all training to target incorrect values.

## Positive Patterns

1. **`eqx.filter_jit` + `eqx.filter_value_and_grad`** — correctly used throughout; no manual `jax.jit`/`jax.grad` with Equinox modules.
2. **`--quick` flag** — enables rapid smoke testing with reduced grid/iterations.
3. **Multi-seed averaging** — density and regularization sweeps average over multiple seeds with error bars.
4. **Publication-quality visualization** — matplotlib styling, PDF output, and colorbar usage are clean.
5. **Clear docstrings** — every function has typed args, return docs, and references to the paper/spec.
6. **Spec references in header** — links to `MATH_SPEC.md` and `ARCH_SPEC.md` sections are explicit and correct.
