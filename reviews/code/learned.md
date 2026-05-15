# Code Review: `models/learned.py`
**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

The module defines five learnable metric classes (`NeuralRiemannian`, `NeuralRanders`, `PullbackRanders`, `PullbackRiemannian`, `DataDrivenPullbackRanders`) plus two supporting modules (`PullbackGNet`, `KernelWindField`). The overall design aligns with `spec/ARCH_SPEC.md § 3` and § 5. The MRO for Equinox/ABC multiple inheritance is correct and the `PullbackGNet` approach is sound. However, there are several issues: the `KernelWindField` contains dead code branches and a numerically fragile distance computation, `PullbackGNet` uses `jacfwd` which may be suboptimal for wide decoders, `PullbackRiemannian` accepts an unused `key` parameter, and test coverage is critically thin — only `NeuralRanders` has dedicated tests while four other public classes and both supporting modules are entirely untested.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/models/learned.py:89–96` | `KernelWindField.__call__`: The `if z.ndim > 0` / `else` branches execute **identical code** in both arms. The `else` branch computes `jnp.dot(self.anchors_z, z.T).T` when `z.ndim == 0`, but `.T` on a scalar is a no-op and `jnp.dot(anchors, z)` would produce a 1-D array while `z_sq` is a scalar with `keepdims=True` — the shapes of `z_sq`, `anchors_sq`, and `dots` will be inconsistent for a true scalar input. More importantly, this function will be called under `vmap` where `z` is always rank-1, so the `else` branch is dead code that was likely a debugging artefact. The duplicated `if`/`else` on lines 93–96 is also identical in both branches. | Remove the `if z.ndim > 0` branching entirely and keep only the rank-1 path. If scalar inputs must be supported, reshape `z` to `(D,)` at entry. |
| 2 | **RISK** | `src/ham/models/learned.py:85–88` | `KernelWindField.__call__`: The expanded distance formula `z_sq + anchors_sq - 2 * dots` can produce **small negative values** due to floating-point cancellation when `z ≈ anchor`. These negative `dists_sq` are then divided by `2 * sigma²` and passed to `softmax`. While `softmax` is shift-invariant and the final result is unaffected by a uniform negative offset, individual negative squared-distances are mathematically nonsensical and could cause issues if this intermediate is ever used elsewhere (e.g., for logging or thresholding). | Add `dists_sq = jnp.maximum(dists_sq, 0.0)` after computation, or document that negative values are benign under `softmax`. |
| 3 | **RISK** | `src/ham/models/learned.py:110–113` | `PullbackGNet.__call__`: Uses `jax.jacfwd` to compute the full Jacobian of the decoder. For a decoder mapping $\mathbb{R}^d \to \mathbb{R}^D$ where $D \gg d$ (typical in VAEs — e.g., $d=10$, $D=5000$), `jacfwd` computes $d$ forward-mode passes, which is correct and efficient. However, if the decoder has internal parameters that are very wide, `jacfwd` materialises the full $(D, d)$ Jacobian in memory. For very large decoders this may cause OOM. More critically, `jacfwd` does **not** support all Equinox modules out of the box — if the decoder contains `eqx.nn.BatchNorm` or other stateful layers, tracing will fail. | Document the decoder contract (must be a pure function of `z` with no internal state beyond parameters). Consider offering a `jacrev` alternative for cases where $d > D$. |
| 4 | **RISK** | `src/ham/models/learned.py:115` | `PullbackGNet.__call__`: The regularisation `H + 1e-4 * jnp.eye(self.dim)` uses a **hardcoded epsilon** rather than the canonical `PSD_EPS` from `utils/math.py`. The `PSDMatrixField` in `nn/networks.py:107` also hardcodes `1e-4`. These should reference the shared constant to avoid drift if the canonical value is ever changed. | Import and use `PSD_EPS` from `ham.utils.math`: `from ..utils.math import PSD_EPS`. |
| 5 | **RISK** | `src/ham/models/learned.py:99` | `KernelWindField.__call__`: `softmax(-dists_sq / (2 * self.sigma**2))` — when `sigma` is very small (e.g., `0.01`), the argument to `softmax` becomes very large in magnitude, causing the softmax to saturate to a one-hot vector. While mathematically correct (nearest-neighbour limit), the **gradients** through the saturated softmax will be essentially zero, making `sigma` unlearnable if it were ever treated as a trainable parameter. Currently `sigma` is not marked `static`, so it *is* part of the PyTree and could be optimised by Optax. | If `sigma` is intended to be fixed, mark it `static=True` via `sigma: float = eqx.field(static=True)`. If it should be learnable, use `log_sigma` parameterisation and `jnp.exp(log_sigma)` to ensure positivity and better gradient conditioning. |
| 6 | **RISK** | `src/ham/models/learned.py:56–58` | `PullbackRanders.__init__`: The `decoder` is stored as a regular (non-static) Equinox field, meaning it is part of the PyTree and its parameters will be included in `eqx.filter_grad`. If the intent is to freeze the decoder during Randers training (as the docstring says "frozen decoder"), the caller must manually use `eqx.partition` or `eqx.filter` to exclude it. There is no enforcement of the "frozen" contract at the class level. The same applies to `DataDrivenPullbackRanders` (line 105) and `PullbackRiemannian` (line 126). | Document that callers must freeze the decoder externally, or provide a `freeze_decoder()` method that wraps parameters in `jax.lax.stop_gradient`. |
| 7 | **STYLE** | `src/ham/models/learned.py:128` | `PullbackRiemannian.__init__` accepts `key: jax.Array = None` and `hidden_dim: int = 32`, but **neither parameter is used** in the body. The pullback metric is fully determined by the decoder — there are no learnable parameters to initialise with a key, and no network that uses `hidden_dim`. These are vestigial parameters from a copy-paste of `NeuralRiemannian`. | Remove `key` and `hidden_dim` from the signature, or document their purpose if planned for future use. |
| 8 | **STYLE** | `src/ham/models/learned.py:1–9` | `from typing import Any` is imported but `Any` is used only as a type annotation on Equinox fields. Since Python 3.10+, `typing.Any` is available as a built-in. More importantly, `safe_norm` is imported (line 9) but **never used** anywhere in the module — all norm operations are delegated to the parent `Randers`/`Riemannian` classes. | Remove the unused `from ..utils.math import safe_norm` import. |
| 9 | **STYLE** | `src/ham/models/learned.py:44` | Blank line with trailing whitespace after `NeuralRanders.__init__`. Minor formatting issue. | Remove trailing whitespace. |
| 10 | **STYLE** | `src/ham/models/learned.py:12–22` | `NeuralRiemannian` uses diamond inheritance: `(Riemannian, eqx.Module)`. Since `Riemannian` → `FinslerMetric` → `eqx.Module`, the explicit `eqx.Module` in the bases is redundant. The MRO resolves correctly regardless, but it is unnecessary noise. Same applies to `NeuralRanders` (line 24), `PullbackRiemannian` (line 119). | Remove redundant `eqx.Module` from bases where the other parent already inherits from it. |

## Test Coverage Assessment

| Public Symbol | Tested? | Test Location | Notes |
|---------------|---------|---------------|-------|
| `NeuralRanders` | **Yes** | `tests/test_learned_metric.py` | Zermelo convexity enforcement and gradient existence tested |
| `NeuralRiemannian` | **No** | — | No dedicated test |
| `PullbackRanders` | **No** | — | Used in `weinreb_vae.py` integration but no unit test |
| `PullbackRiemannian` | **No** | — | No test at all |
| `DataDrivenPullbackRanders` | **No** | — | No test at all |
| `KernelWindField` | **No** | — | No test — dead code branch (Issue #1) is never exercised |
| `PullbackGNet` | **No** | — | Implicitly used via `PullbackRanders` in integration, no unit test |

### Coverage Gaps
1. **5 of 7 public symbols are untested.** Only `NeuralRanders` has dedicated tests. Per `spec/ARCH_SPEC.md § 6`, these are all "Completed & Validated" components.
2. **No `jit` or `vmap` test** for any class. This would catch tracing issues in `KernelWindField` (Issue #1) and `PullbackGNet` (Issue #3).
3. **No gradient test for pullback metrics.** The pullback $J^T J$ construction is the most differentiation-intensive code path — `jax.jacfwd` composed with `energy` → `spray` creates 3rd-order derivatives. No test verifies these are finite.
4. **No edge-case tests:** zero-vector input, identity decoder, single-anchor `KernelWindField`, very small `sigma`.
5. **`NeuralRiemannian`** is trivial (3-line `__init__`) but should still have a smoke test for `metric_fn`, `energy`, and `spray` to confirm the MRO and PSD construction work end-to-end.
6. **No test verifies that decoder freezing works** in `PullbackRanders` / `PullbackRiemannian` — i.e., that `eqx.filter_grad` through the metric does not update decoder weights.

## Positive Patterns

1. **Clean Zermelo delegation.** `NeuralRanders` and `PullbackRanders` correctly delegate all Randers physics to the parent `zoo.Randers` class, avoiding code duplication. The metric construction (PSD enforcement, wind squashing, causality constraint) lives in one place.
2. **`PullbackGNet` is mathematically clean.** The $J^T J + \epsilon I$ construction is the textbook pullback metric and the regularisation prevents singularity at decoder critical points.
3. **Correct Equinox `static` field usage.** `dim`, `epsilon`, and `use_wind` are correctly marked `static=True`, ensuring they are not traced as dynamic values and do not interfere with `jit` compilation.
4. **`KernelWindField` optimised distance formula.** The comment on line 84 correctly identifies that the expanded $(a-b)^2 = a^2 + b^2 - 2ab$ form avoids materialising a `(B, N, D)` intermediate under `vmap`, which is a genuine memory optimisation for large anchor sets.
5. **Clean `__init__.py` re-exports.** All 7 public symbols are explicitly listed in `__all__`, matching the module contents exactly.
