# Code Review: `nn/networks.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`networks.py` provides three clean `eqx.Module` classes (`RandomFourierFeatures`, `VectorField`, `PSDMatrixField`) that serve as the neural building blocks for learned metrics. The code is compact and largely correct for the common case. However, there are **no `jit`/`vmap`/`grad` compatibility tests**, the RFF frequency matrix `B` is frozen when it arguably should be, the PSD epsilon is hard-coded instead of using the canonical constant, and several edge-case numerical risks exist. No critical bugs were found.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | `src/ham/nn/networks.py:13` | `RandomFourierFeatures.B` is a plain `jnp.ndarray`, making it a **trainable leaf** in Equinox's filter model. For the canonical RFF formulation, `B` should be frozen (non-trainable). If a user calls `eqx.filter(model, eqx.is_array)` to get trainable params, `B` will be included and optimised, silently breaking the theoretical guarantee of the feature map. | Wrap `B` as a non-trainable buffer: `self.B = jax.lax.stop_gradient(jax.random.normal(...) * scale)` is insufficient — instead mark the field `B: jnp.ndarray = eqx.field()` and filter it out at the training call site, or document that `B` is intentionally trainable if that is the design choice. |
| 2 | **RISK** | `src/ham/nn/networks.py:96` | PSD regularisation epsilon `1e-4` is hard-coded. The canonical constant `PSD_EPS` is already defined in `src/ham/utils/math.py:9` and is used by `PullbackGNet` in `models/learned.py:118`. Using a different literal here is a maintenance hazard — if the canonical value changes, this file will drift. | Import and use `from ..utils.math import PSD_EPS`; `G = ... + PSD_EPS * jnp.eye(self.dim)`. |
| 3 | **RISK** | `src/ham/nn/networks.py:88–96` | `PSDMatrixField.__call__` reshapes a flat vector into `A` and computes `A @ A.T`. When the MLP outputs very large values (e.g., during early training with bad initialisation or high learning rates), `A @ A.T` can overflow or produce an ill-conditioned matrix. No clamping or normalisation is applied to `flat_A`. | Consider applying `tanh` or clipping to `flat_A` before reshape, or at minimum document the expected output scale. The MLP already uses `tanh` activation on hidden layers, but the final linear layer is unbounded. |
| 4 | **STYLE** | `src/ham/nn/networks.py:22–54` | `VectorField` does not document the batch convention. Per `ARCH_SPEC.md § 1`, all operations assume a leading batch dimension `(B, ...)`. The `__call__` signature accepts `(D,)` only; batch handling must be provided externally via `vmap`. This is consistent with other modules but should be documented explicitly. | Add a note in the docstring: "Operates on single points `(D,)`. Use `jax.vmap` for batched evaluation." |
| 5 | **STYLE** | `src/ham/nn/networks.py:57–96` | Same batch-convention documentation gap as #4 for `PSDMatrixField`. | Same fix — document single-point contract in docstring. |
| 6 | **RISK** | `src/ham/nn/networks.py:37` | `VectorField.__init__` computes `map_size = hidden_dim // 2`. If `hidden_dim` is odd (e.g., 33), the RFF output dimension is `2 * (33 // 2) = 32`, which silently differs from `hidden_dim`. This is unlikely to cause a crash (the MLP `in_size` is set correctly), but the effective embedding dimension is smaller than the user might expect. | Either assert `hidden_dim % 2 == 0` or document the rounding behaviour. |
| 7 | **STYLE** | `src/ham/nn/networks.py:6–20` | `RandomFourierFeatures` is a public class (exported in `__init__.py`) but has no dedicated test. It is only tested indirectly via `test_fourier_features` in `test_network.py`, which merely checks that the with/without-Fourier paths produce different shapes. Output distribution properties (e.g., the kernel approximation quality) are not verified. | Add a test that checks: (a) output shape is `(2 * mapping_size,)`, (b) output values are bounded in `[-1, 1]`, (c) the approximation approaches the exact RBF kernel as `mapping_size` grows. |
| 8 | **RISK** | `tests/test_network.py` | **No `jit`/`vmap`/`grad` compatibility tests.** The test file never wraps any network call in `jax.jit`, `jax.vmap`, or `jax.grad`. Since these modules are differentiated through during geodesic solving and training, a tracer-incompatible operation (e.g., data-dependent Python control flow) would only surface at runtime. The `if self.embedding is not None` branch at lines 51–52 and 88–89 is safe because `embedding` is `None` or a module (static structure), but this should be verified by tests. | Add tests: `jax.jit(vf)(x)`, `jax.vmap(vf)(xs)`, `jax.grad(lambda x: jnp.sum(vf(x)))(x)`, and equivalently for `PSDMatrixField`. |
| 9 | **STYLE** | `tests/test_network.py` | `test_fourier_features` creates two `VectorField` instances with the same key but different `use_fourier` flags. The comment says "They shouldn't be identical" but no assertion enforces this — the test passes even if they are identical. | Add `self.assertFalse(jnp.allclose(out_base, out_four))`. |
| 10 | **RISK** | `src/ham/nn/networks.py:57–96` | `PSDMatrixField` outputs $D^2$ parameters for a full $D \times D$ factor matrix, but symmetry means only $D(D+1)/2$ unique elements are needed (lower-triangular Cholesky factor). The current approach is correct (any $A A^\top$ is PSD) but wastes $D(D-1)/2$ parameters and makes the mapping non-unique ($A$ and $-A$ produce the same $G$). For high-dimensional metrics this redundancy increases optimisation difficulty. | Consider outputting a lower-triangular matrix (Cholesky parameterisation) instead. This is a design suggestion, not a bug. |

## Test Coverage Assessment

| Public Symbol | Test Exists | Notes |
|---|---|---|
| `RandomFourierFeatures` | Indirect only (`test_fourier_features`) | No standalone test for output shape, range, or kernel approximation quality. |
| `VectorField` | `test_vector_field_shape` | Shape and finiteness tested. No `jit`/`vmap`/`grad` test. No edge-case inputs (zero vector, large values). |
| `PSDMatrixField` | `test_psd_matrix_properties` | Symmetry and PD tested. No `jit`/`vmap`/`grad` test. No test for conditioning or numerical stability under extreme inputs. |

**Gaps:**
- No gradient-flow test for any network (the gradient test in `test_learned_metric.py` covers the full `NeuralRanders`, not the isolated networks).
- No `jax.jit` smoke test — a tracer-breaking change would not be caught.
- No `jax.vmap` batching test.
- No edge-case test: zero input, very large input, `dim=1`.

## Positive Patterns

1. **Clean Equinox usage.** All three classes are proper `eqx.Module` subclasses with correctly typed fields. The `dim: int = eqx.field(static=True)` annotation in `PSDMatrixField` correctly marks a non-array field as static for JIT tracing.
2. **Optional composition.** The `Optional[RandomFourierFeatures]` pattern with a `None` check is the idiomatic Equinox way to handle optional sub-modules and is JIT-safe (the branch is resolved at trace time).
3. **Activation choice.** Using `jax.nn.tanh` instead of ReLU is a well-motivated choice for coordinate-based networks that need smooth second derivatives (critical for Hessian-based metric computations).
4. **PSD construction.** The $A A^\top + \epsilon I$ pattern is a well-known, simple, and differentiable way to guarantee positive definiteness.
5. **PRNG key splitting.** Proper `jax.random.split` usage separates the embedding and MLP random streams, ensuring reproducibility.
