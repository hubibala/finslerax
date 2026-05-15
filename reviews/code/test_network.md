# Code Review: `tests/test_network.py`

**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary

The test file covers basic shape and PSD-property checks for `VectorField` and `PSDMatrixField`, but has significant gaps. One test (`test_fourier_features`) contains a missing assertion that renders it a no-op for its stated purpose. There are no JAX-transform compatibility tests (`jit`, `vmap`, `grad`), no batch-dimension tests (violating ARCH_SPEC § 1 Batch-First principle), and `RandomFourierFeatures` is never tested in isolation. The existing tests are deterministic and well-structured but cover only a single input point each.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `tests/test_network.py:47` | `test_fourier_features` never asserts that `out_base` and `out_four` differ. The comment on line 47 says "They shouldn't be identical" but there is no assertion — the test passes even if Fourier features have zero effect. | Add `self.assertFalse(jnp.allclose(out_base, out_four), "Fourier and non-Fourier outputs should differ")` or a similar assertion. |
| 2 | **RISK** | `tests/test_network.py:34` | `assertGreater(min_eig, 0.0)` accepts arbitrarily small positive eigenvalues (e.g. `1e-30`). Given `PSDMatrixField` adds `1e-4 * I` (see `src/ham/nn/networks.py:97`), the minimum eigenvalue should be at least `1e-4`. The current assertion does not validate the regularisation floor. | Use `self.assertGreater(float(min_eig), 1e-5, "Minimum eigenvalue below regularisation floor")`. |
| 3 | **RISK** | `tests/test_network.py:13-48` | No `jax.jit` compatibility test for any network module. If a network contains a Python side-effect or non-JAX control flow, it would silently break under JIT tracing. | Add a test that wraps each network call in `eqx.filter_jit` and verifies identical output. |
| 4 | **RISK** | `tests/test_network.py:13-48` | No `jax.vmap` / batch-dimension test. `ARCH_SPEC.md` § 1 mandates batch-first convention `(B, ...)`. There is no test confirming the networks can be vmapped over a batch of inputs. | Add a test: `batched = jax.vmap(vf)(xs)` where `xs.shape == (B, D)` and assert output shape `(B, D)`. |
| 5 | **RISK** | `tests/test_network.py:13-48` | No `jax.grad` compatibility test. These networks are used inside differentiable metrics; if gradients cannot flow through them, the training pipeline breaks silently. | Add a test computing `jax.grad(lambda x: jnp.sum(vf(x)))(x)` and asserting finite output. |
| 6 | **RISK** | `tests/test_network.py:19-35` | `test_psd_matrix_properties` only tests at `x = jnp.ones(dim)`. PSD property should hold for all inputs. A single input point does not catch input-dependent failures (e.g. numerical instability near zero). | Test at multiple inputs: `jnp.zeros(dim)`, `jnp.ones(dim) * 100`, and a random point. |
| 7 | **STYLE** | `tests/test_network.py:33` | `print(f"\nMin Eigenvalue: {min_eig}")` in test body. Test output should be assertion-driven; debug prints pollute test runner output. | Remove the print statement or gate it behind a verbosity flag. |
| 8 | **STYLE** | `tests/test_network.py:1-48` | `RandomFourierFeatures` is a public class in `src/ham/nn/networks.py` but has no dedicated test. It is only exercised indirectly through `VectorField(use_fourier=True)`. | Add a unit test for `RandomFourierFeatures` verifying output shape `(2 * mapping_size,)` and output range `[-1, 1]` (cos/sin bounds). |
| 9 | **STYLE** | `tests/test_network.py:1-48` | No test for `PSDMatrixField` with `use_fourier=True`. The Fourier code path in `PSDMatrixField` is entirely untested. | Add a test instantiating `PSDMatrixField(..., use_fourier=True)` and verifying shape and PSD property. |
| 10 | **STYLE** | `tests/test_network.py:14` | `test_vector_field_shape` only tests at `x = jnp.zeros(dim)`. While finiteness is checked, a zero input may exercise a trivial code path (RFF would produce `cos(0)=1, sin(0)=0` uniformly). | Add a non-zero test input. |

## Test Coverage Assessment

| Public Symbol | Tested? | Gap |
|---|---|---|
| `VectorField.__call__` | Partial | Shape and finiteness tested at one point; no `jit`/`vmap`/`grad` test. Only `use_fourier=False` path verified for correctness. |
| `PSDMatrixField.__call__` | Partial | Shape, symmetry, and PD tested at one point; no `jit`/`vmap`/`grad` test. `use_fourier=True` path untested. |
| `RandomFourierFeatures.__call__` | **No** | Exercised indirectly via `VectorField` only; no standalone test. |
| `VectorField(use_fourier=True)` path | **No** | Test exists but lacks an assertion on the distinguishing behavior (Issue #1). |
| Batch-dimension `(B, D)` input | **No** | ARCH_SPEC § 1 batch-first convention entirely untested. |
| Gradient flow through networks | **No** | Critical for learned-metric use case; never verified. |

## Positive Patterns

1. **Deterministic seeding** (`PRNGKey(42)` in `setUp`) ensures reproducible tests — good practice.
2. **Finiteness check** in `test_vector_field_shape` (`jnp.isfinite(v).all()`) catches NaN/Inf regressions early.
3. **Eigenvalue-based PD verification** in `test_psd_matrix_properties` is the mathematically correct approach (via `eigvalsh` for symmetric matrices).
4. **Clear docstrings** on each test method communicate intent.
