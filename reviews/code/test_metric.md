# Code Review: tests/test_metric.py
**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary

The test file covers a subset of `FinslerMetric` behaviour (spray, inner product, acceleration) with physically motivated tests, but has significant coverage gaps. Public methods `energy()` and `arc_length()` are entirely untested. No tests exercise batch dimensions, JAX transform compatibility (`jit`/`vmap`/`grad`), or numerical edge cases (zero vector, near-zero norm). A dead fixture (`ScaledEuclideanMetric`) and duplicated local class definitions reduce maintainability.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | `tests/test_metric.py:28-30` | `ScaledEuclideanMetric` is defined but never referenced by any test. Dead code obscures what is actually tested and may mislead coverage tools. | Remove the class, or add a test that exercises it (e.g., verify spray is still zero under uniform scaling). |
| 2 | **RISK** | `tests/test_metric.py:66-72`, `tests/test_metric.py:101-106` | `CurvedMetric` is defined identically inside two separate test methods. Duplication means a future correction must be applied in two places. | Extract `CurvedMetric` to module level (next to `EuclideanMetric`) and reuse in both tests. |
| 3 | **RISK** | `tests/test_metric.py` (file-level) | No test for `energy()`. This is a public method on `FinslerMetric` and the foundation for spray/inner-product derivations. A bug in `energy` would propagate silently if only downstream quantities are checked. | Add `test_energy_euclidean` verifying $E = 0.5 \|v\|^2$ and `test_energy_curved` for the diagonal metric. |
| 4 | **RISK** | `tests/test_metric.py` (file-level) | No test for `arc_length()`. This public method is completely uncovered. | Add a test computing arc length of a straight-line path in Euclidean space (should equal the Euclidean distance) and a simple curved case. |
| 5 | **RISK** | `tests/test_metric.py` (file-level) | No test for zero velocity vector `v = [0, 0, 0]`. `jnp.linalg.norm` returns 0 and the Hessian of the energy becomes degenerate at $v = 0$, making `jnp.linalg.solve` in `spray()` likely to produce `NaN`. This is a critical edge case for downstream solvers. | Add `test_spray_zero_velocity` asserting finite output (or a documented, graceful failure mode). |
| 6 | **RISK** | `tests/test_metric.py` (file-level) | No `jax.jit` compatibility test. The ARCH_SPEC requires all core operations to be JIT-compilable. A Python-level side-effect or non-JAX control flow in a future `metric_fn` would only be caught at runtime. | Add `test_jit_spray` wrapping `jax.jit(metric.spray)` and comparing to eager output. |
| 7 | **RISK** | `tests/test_metric.py` (file-level) | No `jax.vmap` / batch-dimension test. ARCH_SPEC §1 mandates batch-first `(B, ...)` convention for all operations. There is no test verifying that `spray`, `inner_product`, or `energy` can be vmapped over a batch. | Add `test_vmap_spray` applying `jax.vmap(metric.spray)` over a batch of `(B, 3)` inputs. |
| 8 | **STYLE** | `tests/test_metric.py:50`, `tests/test_metric.py:51`, `tests/test_metric.py:82`, `tests/test_metric.py:131` | Inconsistent absolute tolerance across assertions: `atol=1e-5` in some, `atol=1e-6` in others, with no documented rationale. This makes it hard to distinguish intentional precision expectations from copy-paste variance. | Standardise on a single tolerance constant (e.g., `ATOL = 1e-5`) defined at module level, or document why specific tests need tighter bounds. |
| 9 | **STYLE** | `tests/test_metric.py:37` | `self.key = jax.random.PRNGKey(42)` is created in `setUp` but never used. | Remove it, or add randomised property-based tests that consume the key. |
| 10 | **STYLE** | `tests/test_metric.py:33` | All tests live in a single class `TestMetricPhysics`. Grouping spray tests, inner-product tests, and acceleration tests into separate classes would improve readability as coverage grows. | Consider splitting into `TestSpray`, `TestInnerProduct`, `TestArcLength`. |
| 11 | **STYLE** | `tests/test_metric.py:23-25` | `EuclideanMetric.metric_fn` uses `jnp.linalg.norm(v)` without specifying `axis`. This works only for 1-D `v` (no batch dim). If a batch-dimension test is added, this will break. | Use `jnp.linalg.norm(v, axis=-1)` to be batch-safe. |
| 12 | **RISK** | `tests/test_metric.py:123-130` | `test_acceleration_sign` only tests on Euclidean metric where both spray and acceleration are near-zero. The relationship $a = -2G$ is trivially satisfied by $0 \approx -2 \cdot 0$. This does not actually validate the sign or magnitude. | Re-run the same test with `CurvedMetric` where spray is non-trivial. |

## Test Coverage Assessment

| Public Method | Tested? | Notes |
|---------------|---------|-------|
| `metric_fn` | Indirectly | Tested only through downstream `spray` / `inner_product` |
| `energy` | **No** | Gap — no dedicated test |
| `inner_product` | Yes | Covered for Euclidean and curved cases |
| `spray` | Yes | Euclidean (zero) and curved (homogeneity) |
| `geod_acceleration` | Partially | Only on Euclidean (trivial zero case) |
| `arc_length` | **No** | Gap — completely untested |

**JIT/vmap/grad transform tests:** None.
**Batch-dimension tests:** None.
**Edge-case tests (zero vector, large norm, near-degenerate):** None.

## Positive Patterns

- **Physically motivated tests:** Spray homogeneity (2-homogeneous in $v$) and Euclidean-spray-is-zero are excellent structural invariant tests that will catch genuine regressions.
- **Analytic ground truth:** `test_inner_product_curved` computes the expected metric tensor by hand and compares — this is the gold standard for numerical tests.
- **Clear docstrings:** Every test method has a docstring explaining the physical property being checked, making failures immediately interpretable.
- **Proper use of `np.testing.assert_allclose`:** Correct use of NumPy's tolerance-aware comparisons rather than bare `assertEqual`.
