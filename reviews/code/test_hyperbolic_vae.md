# Code Review: tests/test_hyperbolic_vae.py

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0

## Summary

The test file provides basic coverage for the hyperbolic VAE integration (exponential map gradients, parallel transport gradients, wrapped-normal sampling, VAE forward pass, and VAE gradient flow). However, it contains one clear bug (reused PRNG key producing identical "independent" inputs), a signature mismatch in the mock metric, several missing edge-case tests, and no `jit`/`vmap` compatibility checks. The five existing tests all pass through the happy path but lack rigor around boundary conditions and batch-dimension conventions required by `spec/ARCH_SPEC.md § 1`.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `tests/test_hyperbolic_vae.py:112-113` | `x` and `v_rna` are sampled with the **same** PRNG key (`self.key`), so `jax.random.normal(self.key, (data_dim,))` returns identical arrays for both. The test therefore never validates behaviour when data and velocity differ. Same issue at lines 140-141. | Split the key: `k1, k2 = jax.random.split(self.key); x = jax.random.normal(k1, ...); v_rna = jax.random.normal(k2, ...)`. |
| 2 | **RISK** | `tests/test_hyperbolic_vae.py:24-30` | `MockMetric.inner_product` has signature `(self, x, u, v=None, w=None)` while the base `FinslerMetric.inner_product` (see `src/ham/geometry/metric.py:35`) has `(self, x, v, w1, w2)`. When called with 4 positional args the mock computes `sum(u * v)` (i.e. `<reference_dir, w1>`) and silently ignores `w` (`w2`). This gives wrong results whenever the inner product is invoked on non-zero arguments. Currently masked because `MockMetric.spray` returns zeros. | Align mock signature: `def inner_product(self, x, v, w1, w2): return jnp.sum(w1 * w2, axis=-1)`. |
| 3 | **RISK** | `tests/test_hyperbolic_vae.py:87-91` | `test_wrapped_normal_sampling` loops over 10 individual samples with scalar `assertAlmostEqual`. A single outlier is enough to fail, yet 10 samples is too few to catch distributional problems. The check is also not vectorized — an `assert jnp.allclose(...)` over the full batch is both faster and more idiomatic. | Replace the loop with: `norm_sq = jax.vmap(self.manifold._minkowski_dot)(z, z); np.testing.assert_allclose(norm_sq, -1.0, atol=1e-5); self.assertTrue(jnp.all(z[:, 0] > 0))`. |
| 4 | **RISK** | `tests/test_hyperbolic_vae.py:101-155` | No test verifies that the `GeometricVAE` or any loss component works under `jax.jit` or `jax.vmap`. Per `spec/ARCH_SPEC.md § 1` ("Batch-First") all operations must support a leading batch dimension. | Add a test that jit-compiles a loss call and a test that vmaps over a batch of inputs. |
| 5 | **RISK** | `tests/test_hyperbolic_vae.py:41-55` | `test_exp_map_gradients` does not test the zero-tangent-vector edge case (`v = jnp.zeros(3)`). `exp_map` has special Taylor-series branches for `norm_v < TAYLOR_EPS`; these are untested. | Add a sub-case with `v = jnp.zeros(3)` and verify the result equals `x`. |
| 6 | **RISK** | `tests/test_hyperbolic_vae.py:57-73` | `test_parallel_transport_gradients` only checks that gradients are finite but never verifies the transported vector is tangent at the destination (`_minkowski_dot(y, v_trans) ≈ 0`) or that norm is preserved. | Add: `self.assertAlmostEqual(float(self.manifold._minkowski_dot(y, v_trans)), 0.0, places=5)`. |
| 7 | **RISK** | `tests/test_hyperbolic_vae.py:119-128` | `test_vae_forward_pass` asserts `jnp.isfinite` on all three losses but never checks sign or magnitude. `ReconstructionLoss` and `KLDivergenceLoss` should be non-negative by construction; this is not validated. | Add `self.assertGreaterEqual(float(r), 0.0)` and `self.assertGreaterEqual(float(k), 0.0)`. |
| 8 | **STYLE** | `tests/test_hyperbolic_vae.py:101-155` | `data_dim`, `latent_dim`, and VAE construction are duplicated across `test_vae_forward_pass` and `test_vae_gradients`. These could be moved into `setUp` to reduce repetition and keep tests DRY. | Extract shared VAE construction to `setUp`. |
| 9 | **STYLE** | `tests/test_hyperbolic_vae.py:131-155` | `test_vae_gradients` only verifies gradient flow through `ReconstructionLoss`. `KLDivergenceLoss` and `ZermeloAlignmentLoss` are not gradient-tested, leaving potential `stop_gradient` or detach bugs in those paths undetected. | Add gradient-flow tests for the other two loss components. |
| 10 | **STYLE** | `tests/test_hyperbolic_vae.py:75-91` | `test_wrapped_normal_sampling` never tests `kl_divergence_std_normal()`. This public method on `WrappedNormal` (see `src/ham/bio/vae.py:46`) has no dedicated test. | Add a test asserting `dist.kl_divergence_std_normal()` is finite, non-negative, and ≈ 0 when `scale ≈ 1`. |

## Test Coverage Assessment

| Public API | Tested? | Notes |
|---|---|---|
| `Hyperboloid.exp_map` | **Partial** | Gradient check only; no value correctness or edge cases |
| `Hyperboloid.parallel_transport` | **Partial** | Gradient check only; no tangency or norm-preservation assertion |
| `WrappedNormal.sample` | **Yes** | Manifold constraint checked |
| `WrappedNormal.kl_divergence_std_normal` | **No** | Called inside `KLDivergenceLoss` but never directly tested |
| `GeometricVAE` forward | **Yes** | Finiteness checked |
| `GeometricVAE` gradients | **Partial** | Only `ReconstructionLoss` path; KL and alignment paths untested |
| `GeometricVAE` under `jit`/`vmap` | **No** | Not tested |
| Batch-dim `(B, ...)` convention | **No** | All tests use single-sample (unbatched) inputs |

### Gap Analysis

- **Edge cases**: Zero tangent vector, antipodal points (large geodesic distance), very large/small scale for `WrappedNormal`, degenerate encoder output.
- **JAX transforms**: No `jit`, `vmap`, or `grad(grad(...))` tests.
- **Batch dimension**: All inputs are single vectors; the batch-first convention from `spec/ARCH_SPEC.md § 1` is never exercised.

## Positive Patterns

1. **64-bit precision enabled** (`config.update("jax_enable_x64", True)` at line 6) — correct for robust gradient checks.
2. **`MockMetric` with zero spray** is a pragmatic approach for isolating VAE logic from geodesic numerics.
3. **Gradient propagation tests** (lines 41–73) verify end-to-end differentiability through manifold ops, which is critical for a JAX-based library.
4. **Modular loss testing** (lines 101–128) correctly tests the new modular loss API rather than the monolithic `loss_fn`.
