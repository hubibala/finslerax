# Code Review: `zoo.py`
**Reviewer:** Code Reviewer Agent
**Date:** 2025-05-15
**Arch Spec Version:** 1.1.0

## Summary

`zoo.py` implements the four concrete `FinslerMetric` subclasses (`Euclidean`, `Riemannian`, `Randers`, `DiscreteRanders`). The overall design is sound: classes correctly inherit from `eqx.Module`-based `FinslerMetric`, static fields are properly marked, and the Zermelo causality squasher is well-engineered. However, there is one genuine bug in the `Randers` zero-vector guard (dead code due to `safe_norm` floor), several inconsistent epsilon constants that bypass the canonical `ham.utils.math` definitions, and a gradient-unsafe `jnp.linalg.norm` call in `DiscreteRanders`. Test coverage is adequate for `Euclidean`, `Riemannian`, and `Randers`, but `DiscreteRanders.metric_fn` has no direct unit test.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/geometry/zoo.py:110-112` | **Dead zero-vector guard in `Randers.metric_fn`.** `safe_norm(v)` with default `eps=1e-12` returns `sqrt(1e-12) ≈ 1e-6` for a zero vector. The subsequent check `is_zero = v_mag < 1e-7` therefore evaluates to `False`, so the guard `v_safe = jnp.where(is_zero, ...)` never activates and the final `jnp.where(is_zero, 0.0, cost)` never short-circuits. Result: `F(x, 0) ≈ 3.16e-5 / λ` instead of `0.0`, violating the Finsler axiom $F(x, 0) = 0$. | Either (a) use `jnp.sum(v**2)` (raw, no floor) for the zero check while keeping `safe_norm` elsewhere, or (b) raise the threshold to `1e-5` so it exceeds the `safe_norm` floor. |
| 2 | **RISK** | `src/ham/geometry/zoo.py:138` | **`jnp.linalg.norm` in `DiscreteRanders.metric_fn`** produces `NaN` gradients when `W_raw` is zero (all face winds zero or weights collapse). The rest of the codebase uses `safe_norm` for this reason. | Replace `jnp.linalg.norm(W_raw)` with `safe_norm(W_raw)`. |
| 3 | **RISK** | `src/ham/geometry/zoo.py:119` | **Biased sqrt stabiliser.** `jnp.sqrt(jnp.maximum(discriminant, 0.0) + 1e-9)` adds the epsilon *inside* the sqrt unconditionally, introducing a small positive bias $\approx \sqrt{\varepsilon}$ even for exact-zero discriminant. The standard pattern is `jnp.sqrt(jnp.maximum(discriminant, 1e-9))` which clamps *then* takes the root. | Use `jnp.sqrt(jnp.maximum(discriminant, 1e-9))` for a tighter lower bound without additive bias. |
| 4 | **RISK** | `src/ham/geometry/zoo.py:82-85` | **Hardcoded PSD regularisation constants** (`0.01`, `0.005`) in `_get_zermelo_data`. These magic numbers bypass the canonical `PSD_EPS = 1e-4` defined in `ham.utils.math` and are aggressive (minimum eigenvalue floor ≈ 0.015). For learned metrics, this can mask gradient signal indicating a degenerate metric. | Import `PSD_EPS` from `ham.utils.math` and use a single `H = H + PSD_EPS * jnp.eye(...)` after symmetrisation, removing the diagonal clamping. Alternatively, document why the higher floor is needed. |
| 5 | **RISK** | `src/ham/geometry/zoo.py:137-148` | **No zero-vector guard in `DiscreteRanders.metric_fn`.** Unlike `Randers`, there is no `is_zero` check. For `v = 0`: `v_sq = 0`, `W_dot_v = 0`, `discriminant = 0`, `cost = sqrt(1e-8) / lam ≈ 1e-4 / lam`. This is non-zero, violating $F(x,0) = 0$. | Add the same zero-vector guard pattern as `Randers.metric_fn` (once Issue #1 is fixed). |
| 6 | **STYLE** | `src/ham/geometry/zoo.py:36,93,97,119,141` | **Inconsistent epsilon values.** Five different hardcoded epsilons (`1e-12`, `1e-8`, `1e-8`, `1e-9`, `1e-8`) are used across the file for analogous stabilisation purposes. `ham.utils.math` defines canonical constants (`GRAD_EPS`, `NORM_EPS`, `PSD_EPS`) that should be used instead. Ref: `spec/ARCH_SPEC.md § 5` module structure. | Import and use the canonical constants from `ham.utils.math`. |
| 7 | **STYLE** | `src/ham/geometry/zoo.py:55-56` | **`epsilon` and `use_wind` fields typed as `Any`.** These fields are `float` and `bool` respectively, but annotated as `Any` via `eqx.field(static=True)`. While Equinox doesn't require specific types, explicit annotations improve readability. | Change to `epsilon: float = eqx.field(static=True)` and `use_wind: bool = eqx.field(static=True)`. |
| 8 | **STYLE** | `src/ham/geometry/zoo.py:124-125` | **Redundant `norm` method on `Randers`.** It delegates directly to `metric_fn` and is the only metric class to define it. `FinslerMetric` does not declare a `norm` method, so this creates an inconsistent API surface across the zoo. | Either add `norm` as an alias in `FinslerMetric` base class, or remove it from `Randers`. |

## Test Coverage Assessment

| Public Symbol | Test File | Tested? | Notes |
|---|---|---|---|
| `Euclidean.metric_fn` | `tests/test_zoo.py` | **Yes** | `test_euclidean_basic` — value check against known norm. Also used in `test_mesh_solver.py`. |
| `Riemannian.metric_fn` | `tests/test_zoo.py` | **Yes** | `test_riemannian_scaling`, `test_riemannian_anisotropy` — isotropic and anisotropic cases. |
| `Randers.metric_fn` | `tests/test_zoo.py` | **Yes** | `test_randers_analytical_match`, `test_randers_convexity_protection`, `test_randers_zero_wind` — directional asymmetry, NaN safety, Riemannian fallback. |
| `Randers._get_zermelo_data` | `tests/test_zoo.py` | **Indirect** | Exercised through `metric_fn` tests but never directly asserted. No test verifies the causality constraint $\|W\|_H < 1$ is actually enforced (i.e., inspecting `lambda_factor > 0`). |
| `Randers.norm` | — | **No** | No test calls `norm` directly. |
| `DiscreteRanders.__init__` | `tests/test_mesh_solver.py` | **Yes** | Constructed in `test_obstacle_avoidance`. |
| `DiscreteRanders.metric_fn` | — | **No** | Only tested indirectly via the AVBD solver in `test_mesh_solver.py`. No unit test calls `metric_fn` directly with known inputs/outputs. |

### Gap Analysis
1. **Missing:** Direct unit test for `DiscreteRanders.metric_fn` with analytically verifiable inputs (e.g., zero wind → Euclidean, single-face mesh with known wind).
2. **Missing:** Zero-vector test for all metrics (`F(x, 0) == 0`). This would have caught Issue #1.
3. **Missing:** `jit`/`vmap`/`grad` compatibility tests. No test verifies that `jax.jit(metric.metric_fn)`, `jax.vmap(metric.metric_fn)`, or `jax.grad(metric.energy)` succeed without error on any zoo metric.
4. **Missing:** Test that `_get_zermelo_data` returns `lambda_factor > 0` for adversarial wind inputs.
5. **Weak:** `test_randers_zero_wind` uses `places=2` tolerance, masking the epsilon bias from Issue #3.

## Positive Patterns

1. **Causality squasher** (`_get_zermelo_data:93-100`): The `tanh`-based wind norm clamping is elegant — it smoothly enforces $\|W\|_H < 1 - \varepsilon$ while remaining fully differentiable. The test `test_randers_convexity_protection` validates this against extreme inputs.
2. **Static field discipline**: `epsilon` and `use_wind` are correctly marked `eqx.field(static=True)`, ensuring the `if not self.use_wind` branch is traced away at JIT time without retracing overhead.
3. **Symmetrisation**: Both `Riemannian` and `Randers` explicitly symmetrise the metric tensor via `0.5 * (G + G.T)`, preventing silent asymmetry bugs from network outputs.
4. **Tangent projection of wind**: `_get_zermelo_data` projects `W_raw` through `self.manifold.to_tangent(z, W_raw)`, enforcing the geometric constraint that the wind field must lie in the tangent space.
5. **Use of `safe_norm`** in `Euclidean` — correctly delegates to the canonical gradient-safe norm utility.
