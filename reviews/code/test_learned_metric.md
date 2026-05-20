# Code Review: tests/test_learned_metric.py

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0

## Summary

The test file covers only two properties of `NeuralRanders` (convexity enforcement and gradient existence) with **2 total tests**. It leaves the remaining four public classes in `src/ham/models/learned.py` (`NeuralRiemannian`, `PullbackRanders`, `PullbackRiemannian`, `DataDrivenPullbackRanders`) and one standalone module (`KernelWindField`) entirely untested. The existing tests contain a fragile floating-point equality check, an unchecked network branch, and no verification of JAX transform compatibility (`jit`, `vmap`). The batch-first convention mandated by `spec/ARCH_SPEC.md § 1` is never exercised.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `tests/test_learned_metric.py:62` | Gradient check for `h_net` is missing entirely. `test_gradients_exist` only asserts that `w_net` receives gradients; `h_net` gradients are never inspected. If gradient flow to the Riemannian component silently breaks, this test will still pass. | Add `h_grad = grads.h_net.mlp.layers[0].weight` and assert it is non-zero, mirroring the existing `w_net` assertion. |
| 2 | **RISK** | `tests/test_learned_metric.py:63` | `jnp.all(w_grad == 0)` performs exact floating-point equality against zero. Gradients that are extremely small but non-zero (e.g. `1e-38`) pass this check despite indicating near-dead gradient flow. Conversely, if a future JAX version flushes denormals, legitimate gradients could compare as exactly zero. | Replace with a magnitude check: `self.assertGreater(jnp.abs(w_grad).max(), 1e-10, ...)`. |
| 3 | **RISK** | `tests/test_learned_metric.py:33–35` | The injected-weight test (`eqx.tree_at` into `self.metric.w_net.mlp.layers[-1].weight`) couples the test to the internal MLP layer structure of `VectorField`. Any refactoring of `VectorField` (e.g. switching to `eqx.nn.Sequential`) will silently break this test at the tree path, not at the assertion. | Guard with a try/except or use `eqx.tree_at` with a lambda that fetches the output layer generically, or inject the wind via a simple callable stub instead. |
| 4 | **RISK** | `tests/test_learned_metric.py:56` | `grad_x` shape is never validated. The assertion `jnp.isfinite(grad_x).all()` would also pass for an empty array or a scalar. | Add `self.assertEqual(grad_x.shape, (3,))`. |
| 5 | **STYLE** | `tests/test_learned_metric.py:42` | `print(f"\nClamped Wind Norm: {w_norm}")` is debug output that pollutes test runner output. | Remove or gate behind a `logging.debug()` call. |
| 6 | **STYLE** | `tests/test_learned_metric.py:1–67` | No test verifies JAX `jit` compatibility of `NeuralRanders.metric_fn` or `energy`. A Python-level side effect inside any called function would silently break JIT tracing but pass these tests. | Add a test: `jit_fn = jax.jit(self.metric.energy); result = jit_fn(x, v); self.assertTrue(jnp.isfinite(result))`. |
| 7 | **STYLE** | `tests/test_learned_metric.py:1–67` | No test verifies `vmap` compatibility, which is required by the batch-first convention (`spec/ARCH_SPEC.md § 1`). | Add a batched test: `xs = jnp.ones((4, 3)); vs = jnp.ones((4, 3)); result = jax.vmap(self.metric.energy)(xs, vs); self.assertEqual(result.shape, (4,))`. |
| 8 | **STYLE** | `tests/test_learned_metric.py:1–67` | No test checks the fundamental Finsler positivity property `F(x, v) ≥ 0` or the zero-vector edge case `F(x, 0) == 0`. These are trivial to add and guard against sign errors in `metric_fn`. | Add: `self.assertGreaterEqual(self.metric.metric_fn(x, v), 0.0)` and `self.assertAlmostEqual(float(self.metric.metric_fn(x, jnp.zeros(3))), 0.0, places=5)`. |

## Test Coverage Assessment

### Classes in `src/ham/models/learned.py`

| Public class / function | Tested? | Gap |
|---|---|---|
| `NeuralRanders.__init__` | Yes (via `setUp`) | — |
| `NeuralRanders._get_zermelo_data` | Partially (convexity only) | No test for output shapes, symmetry of H, or lambda range `(0, 1]` |
| `NeuralRanders.metric_fn` | **No** | Not called directly in any test |
| `NeuralRanders.energy` | Partially (via `grad`) | Never asserted for correctness or positivity |
| `NeuralRiemannian` | **No** | Entire class untested |
| `PullbackRanders` | **No** | Entire class untested |
| `PullbackRiemannian` | **No** | Entire class untested |
| `DataDrivenPullbackRanders` | **No** | Entire class untested |
| `KernelWindField` | **No** | Entire class untested |
| `PullbackGNet` | **No** | Entire class untested |

### Key missing edge-case tests

- Zero tangent vector `v = 0` (should return zero cost without NaN).
- Near-singular metric (H close to rank-deficient).
- High-dimensional input (dim > 3).
- Repeated `jit` calls (trace caching correctness).
- `grad` through `metric_fn` (not just `energy`), needed for spray computation.

## Positive Patterns

1. **Convexity enforcement test** (`test_zermelo_convexity_enforcement`): Directly mutating network weights via `eqx.tree_at` to stress-test the causality squasher is a creative and effective adversarial testing pattern.
2. **Dual gradient check**: Testing gradients w.r.t. both inputs (for the solver) and parameters (for training) in `test_gradients_exist` correctly validates the two differentiation pathways that matter in practice.
3. **Clean `MockManifold`**: The mock is minimal and correct, making test failures easy to attribute to the metric rather than the manifold.
