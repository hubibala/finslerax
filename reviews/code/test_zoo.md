# Code Review: tests/test_zoo.py
**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0

## Summary

`tests/test_zoo.py` covers basic functional paths for `Euclidean`, `Riemannian`, and `Randers` metrics but has significant gaps. The `DiscreteRanders` class exported from `zoo.py` is entirely untested. No tests exercise JAX transforms (`jit`, `vmap`, `grad`) on any metric, despite this being a core library contract. Numerical tolerances in `test_randers_zero_wind` are unusually loose (`places=2`), masking a potential implementation discrepancy. Several edge cases (zero vector, identity metric, batched inputs) are absent, and there are no tests for inherited `FinslerMetric` methods (`energy`, `spray`, `inner_product`).

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `tests/test_zoo.py:130–141` | `test_randers_zero_wind` uses `places=2` (tolerance ~0.01). With zero wind, `Randers.metric_fn` should match `Riemannian.metric_fn` to machine precision, since `_get_zermelo_data` returns `W=0` and `lambda=1`. A loose tolerance hides a real discrepancy; the `1e-9` in `jnp.sqrt(... + 1e-9)` inside `metric_fn` adds `~3e-5` systematic error per call, which would be caught at `places=5` but is silently passing at `places=2`. | Tighten to `places=5` (matching the tolerance used in `test_randers_analytical_match`). Investigate the `+1e-9` offset in `Randers.metric_fn` if this starts failing — the offset should be inside `jnp.maximum`, not added to the argument of `jnp.sqrt`. |
| 2 | **RISK** | `tests/test_zoo.py` (file-level) | No tests verify `jit`, `vmap`, or `grad` compatibility for any metric class. The ARCH_SPEC mandates JIT-safety and batch-first support. A Python-side side-effect or non-JAX control flow could be introduced without detection. | Add tests: `jax.jit(metric.metric_fn)(x, v)`, `jax.vmap(metric.metric_fn)(xs, vs)`, `jax.grad(lambda v: metric.metric_fn(x, v))(v)` for each metric class. |
| 3 | **RISK** | `tests/test_zoo.py` (file-level) | `DiscreteRanders` (exported from `src/ham/geometry/zoo.py:148`) is not imported, constructed, or tested. This is a public class with non-trivial logic (differentiable face weights, wind squashing). | Add a test class for `DiscreteRanders` using a small synthetic `TriangularMesh`. |
| 4 | **RISK** | `tests/test_zoo.py` (file-level) | No test sends a zero vector `v = jnp.zeros(2)` into any metric. `Randers.metric_fn` has an explicit zero-guard (`is_zero = v_mag < 1e-7`); `Riemannian.metric_fn` relies on `jnp.maximum(quad, 1e-12)`. These guards are untested. | Add `test_zero_vector` cases for all three metrics, asserting `cost == 0.0`. |
| 5 | **RISK** | `tests/test_zoo.py` (file-level) | No test verifies gradient flow through any metric (e.g., `jax.grad(metric.energy, argnums=1)(x, v)`). The library's central design — implicit dynamics via auto-differentiation — is unvalidated at the zoo level. | Add at least one gradient-correctness test per metric, e.g., using `jax.test_util.check_grads`. |
| 6 | **RISK** | `tests/test_zoo.py` (file-level) | Batch-first convention (`(B, ...)` leading dimension per `ARCH_SPEC.md § 1`) is not tested. All inputs are single unbatched vectors of shape `(2,)`. | Add tests that `vmap` across a batch `(B, D)` and verify output shape is `(B,)`. |
| 7 | **STYLE** | `tests/test_zoo.py:86–104` | `test_randers_analytical_match` documents expected analytic costs (2.0 and 0.666...) in the docstring but only asserts directional ordering (`assertGreater`) and homogeneity — never the numeric values themselves. The docstring is misleading about what is actually verified. | Either assert the exact analytic values (`assertAlmostEqual(cost_east, 2.0, ...)` and `assertAlmostEqual(cost_west, 2.0/3.0, ...)`), or update the docstring to state that only asymmetry and homogeneity are tested. |
| 8 | **STYLE** | `tests/test_zoo.py:28` | `setUp` creates a fresh `FlatPlane` and PRNG key per test, which is fine for isolation. However, the PRNG key is always `PRNGKey(0)` and is consumed only once (in `test_randers_zero_wind`). If a second test reuses `self.key` without splitting, it will silently share the same random draw. | Use `jax.random.split` in any test that needs randomness, or document that `self.key` is single-use. |
| 9 | **STYLE** | `tests/test_zoo.py:3` | `import numpy as np` is unused. | Remove the unused import. |
| 10 | **STYLE** | `tests/test_zoo.py:14–22` | `FlatPlane` is a test fixture duplicated across multiple test files (cf. `test_metric.py`, `test_geodesic.py`). This creates a maintenance burden if the `Manifold` ABC changes. | Extract `FlatPlane` into a shared `tests/conftest.py` or `tests/fixtures.py` module. |

## Test Coverage Assessment

| Public Symbol | Tested? | Notes |
|---|---|---|
| `Euclidean.metric_fn` | Yes | Basic norm check only. No edge cases (zero vector, large vector). |
| `Riemannian.metric_fn` | Yes | Scaling and anisotropy. No zero vector, no non-diagonal G test. |
| `Randers.metric_fn` | Yes | Asymmetry, convexity protection, zero-wind fallback. Missing exact analytic values. |
| `Randers._get_zermelo_data` | Indirect | Tested only through `metric_fn`. No direct test of wind squashing behavior. |
| `Randers.norm` | No | Trivial wrapper, low priority. |
| `DiscreteRanders` | **No** | Entire class untested. |
| `FinslerMetric.energy` | No | Inherited method, not tested at zoo level. |
| `FinslerMetric.spray` | No | Inherited method, not tested at zoo level. |
| `FinslerMetric.inner_product` | No | Inherited method, not tested at zoo level. |
| `FinslerMetric.arc_length` | No | Inherited method, not tested at zoo level. |
| JAX transform compatibility | **No** | No `jit`/`vmap`/`grad` tests for any metric. |

## Positive Patterns

1. **Convexity protection test** (`test_randers_convexity_protection`, line 106): Directly tests a safety-critical code path with intentionally illegal inputs — excellent defensive testing.
2. **Physics-grounded docstrings**: Test docstrings clearly state the physical setup and expected result, making failures easy to diagnose.
3. **64-bit precision enabled** (line 7): `jax_enable_x64` is set at module scope, ensuring stable numerical comparisons throughout.
4. **Clean test isolation**: Each test creates its own metric instance; no cross-test state leakage.
