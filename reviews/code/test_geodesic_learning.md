# Code Review: tests/test_geodesic_learning.py
**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary

This integration test suite validates that `NeuralRanders` can recover synthetic wind fields from trajectory pair data across four manifold scenarios. The overall strategy is sound, but the file contains several actionable defects: a bug where `test_loss_decreases` compares losses from two different model initialisations, dead code in `test_vortex_direction`, PRNG key reuse in data generators, and unused imports. Test organisation uses `unittest.TestCase` rather than pytest idioms, and there are no slow-test markers despite multi-thousand-epoch training in some cases.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `tests/test_geodesic_learning.py:374-389` | `test_loss_decreases` creates its initial model with `PRNGKey(0)` but `train_wind_field` (line 389) internally creates a *separate* model with `PRNGKey(2025)`. The "initial" and "final" losses therefore belong to two different random initialisations, so the assertion is not verifying that training reduced loss — it is comparing loss across unrelated parameter sets. | Pass the pre-created `model` into `train_wind_field`, or replicate the exact same seed (`seed=2025`) when creating the initial model for the baseline measurement. |
| 2 | **BUG** | `tests/test_geodesic_learning.py:298-312` | `test_vortex_direction` computes `eval_pts` and `W_pred` (lines 298-305) but never uses them. The actual evaluation (lines 308-309) runs on the first 100 *training* points (`dataset.starts[:100]`, `dataset.ends[:100]`) instead of held-out data. This means the test measures memorisation, not generalisation, and is inconsistent with all other test methods. | Either use `eval_pts` for evaluation (as the other tests do), or remove the dead `eval_pts`/`W_pred` computation. Evaluate on independent points for consistency. |
| 3 | **RISK** | `tests/test_geodesic_learning.py:43-46` | In `generate_river_data`, the same `key = jax.random.PRNGKey(0)` is consumed by *both* `jax.random.uniform` (line 44) and `jax.random.normal` (line 46). Reusing a PRNG key produces correlated random streams and violates JAX's PRNG contract. | Split the key: `k1, k2 = jax.random.split(key)` and use `k1` for `starts`, `k2` for `noise`. |
| 4 | **RISK** | `tests/test_geodesic_learning.py:323-325` | `test_hyperboloid_vortex_direction` evaluates on points from `manifold.random_sample`, which samples uniformly across the hyperboloid. However, the training data (`generate_hyperboloid_vortex`) is deliberately concentrated near the tip via `exp_map` with scale 0.8 (lines 91-96). Evaluation on widely-spread points — far outside the training distribution — makes the cosine-similarity check fragile and potentially meaningless. | Use evaluation points drawn from the same near-tip distribution as the training data (e.g., via `exp_map` with a different PRNG key). |
| 5 | **RISK** | `tests/test_geodesic_learning.py:316` | `test_hyperboloid_vortex_direction` trains for 3 000 epochs, and `test_sphere_vortex_direction` for 1 000 epochs. Without a `@pytest.mark.slow` (or equivalent skip/flag mechanism) these will run in every CI invocation, significantly increasing wall-clock time. | Add `@unittest.skip` or a custom slow marker so routine CI runs can exclude them. |
| 6 | **STYLE** | `tests/test_geodesic_learning.py:4-5` | Unused imports: `partial` (from `functools`), `Callable` and `Tuple` (from `typing`), and `numpy as np`. These add noise and may trigger linter warnings. | Remove the four unused imports. |
| 7 | **STYLE** | `tests/test_geodesic_learning.py:207-210` | `_filter_all` uses `lambda leaf: True if eqx.is_array(leaf) else False`, which is a verbose identity wrapper around a boolean. | Simplify to `lambda leaf: eqx.is_array(leaf)` or pass `eqx.is_array` directly. |
| 8 | **STYLE** | `tests/test_geodesic_learning.py:270,278,286,310,312,329,331,345,347,368,390` | Debugging `print()` statements are scattered through every test. These are invisible in pytest's default output mode and clutter stdout otherwise. | Replace with `logging.debug(...)` or use pytest's `capfd`/`-s` flag pattern for optional verbosity. |
| 9 | **STYLE** | `tests/test_geodesic_learning.py` (file-level) | The suite uses `unittest.TestCase` while the rest of the test suite layout suggests pytest is the runner. `unittest.TestCase` prevents use of pytest fixtures, parametrisation, and `@pytest.mark`. | Migrate to plain pytest functions (or at minimum use `pytest.mark` decorators alongside `TestCase`). |
| 10 | **STYLE** | `tests/test_geodesic_learning.py:260-268,285-289,296-305,322-328,340-346` | The inner function `get_w(pt)` that calls `trained._get_zermelo_data(pt)` is copy-pasted verbatim in every test method. | Extract a module-level helper or fixture, e.g. `def wind_field(model, pts)`. |

## Test Coverage Assessment

| Public Function / Scenario | Tested? | Notes |
|---|---|---|
| `generate_river_data` | Yes (via `test_river_direction`, `test_loss_decreases`) | |
| `generate_vortex_data` | Yes (via `test_vortex_direction`) | Evaluation on training data (see #2) |
| `generate_hyperboloid_vortex` | Yes (via `test_hyperboloid_vortex_direction`) | Out-of-distribution eval (see #4) |
| `generate_sphere_vortex` | Yes (via `test_sphere_vortex_direction`) | |
| `train_wind_field` | Yes (all scenarios) | |
| `DirectWindAlignmentLoss` | Indirectly (via training) | No isolated unit test for gradient correctness |
| `WindRegularizationLoss` | Indirectly (via training) | No isolated unit test |
| `MetricIdentityLoss` | Indirectly (via training) | No isolated unit test |
| `MetricModel` / `PairDataset` | Indirectly (via pipeline) | No unit tests for adapter correctness |
| `cosine_similarity` helper | Indirectly | No unit test (e.g. identical vectors → 1.0, orthogonal → 0.0) |

### Gap Analysis
- **No edge-case tests**: zero-length displacement, antipodal points on $S^2$, or very large displacements on the hyperboloid.
- **No gradient tests**: none of the custom `LossComponent` subclasses (`DirectWindAlignmentLoss`, etc.) are tested for differentiability via `jax.grad` or `check_grads`.
- **No `jit` / `vmap` compatibility tests**: the custom losses and `MetricModel` adapter are never explicitly tested under `jax.jit` or `jax.vmap` in isolation.
- **No noise-robustness test**: `generate_sphere_vortex` and `generate_hyperboloid_vortex` accept a `noise` parameter, but all calls pass `noise=0.0`. A noisy variant would test robustness.

## Positive Patterns
- **Clear scenario-based structure**: each test corresponds to a well-defined synthetic scenario (river, vortex, curved manifolds) making failures interpretable.
- **`SyntheticDataset` as `NamedTuple`**: clean, immutable data container.
- **`cosine_similarity` helper with numerical guard** (`+ 1e-8`): avoids division by zero when comparing directions.
- **`MetricModel` adapter**: cleanly separates test concerns from the pipeline's interface requirements.
- **Descriptive docstrings on every test method**: makes intent immediately clear.
