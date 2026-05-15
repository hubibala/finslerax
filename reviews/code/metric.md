# Code Review: `ham.geometry.metric`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)  
**Source:** `src/ham/geometry/metric.py` (79 lines)  
**Tests:** `tests/test_metric.py`

## Summary

The `FinslerMetric` base class is compact and correctly implements the Metric-First design from `spec/ARCH_SPEC.md § 2.2`. The auto-differentiation of the geodesic spray via `jax.grad`, `jax.jvp`, and `jax.hessian` is idiomatic Equinox/JAX and composes correctly with `jit`, `vmap`, and `grad`. Two substantive risks were identified: (1) the unconditional Hessian regularisation in `spray` introduces systematic bias for well-conditioned metrics, and (2) `arc_length` evaluates the metric at ambient-space midpoints without manifold projection. Test coverage is adequate for core spray physics but has gaps in `arc_length`, gradient correctness, and JIT/vmap integration tests.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | `src/ham/geometry/metric.py:60` | **Unconditional Hessian regularisation.** `hess_v + 1e-4 * jnp.eye(x.shape[0])` is applied to *all* metrics, not just Randers. For well-conditioned Riemannian metrics where $\lambda_{\min}(H_v) \gg 10^{-4}$, this adds a deterministic $O(\epsilon / \lambda_{\min})$ bias to the geodesic spray. The hardcoded `1e-4` is also not sourced from the canonical constant `PSD_EPS` in `utils/math.py` (which happens to equal `1e-4`, but the duplication is a maintenance hazard). | Make the regularisation strength a class-level attribute (e.g. `spray_reg: float = eqx.field(static=True, default=1e-4)`) so subclasses can override. Import `PSD_EPS` from `utils/math.py` for the default value. |
| 2 | **RISK** | `src/ham/geometry/metric.py:60` | **Dimension inferred from `x.shape[0]` instead of `hess_v.shape[0]`.** If `x` lives in ambient space (dim $N$) while $v$ lives in a lower-dimensional tangent parameterisation (dim $D < N$), `jnp.eye(x.shape[0])` would produce an $(N \times N)$ matrix while `hess_v` is $(D \times D)$, causing a shape mismatch. Current manifold implementations keep $x$ and $v$ in the same ambient dimension, so this does not trigger today, but it is a latent coupling. | Use `hess_v.shape[0]` (or equivalently `v.shape[-1]`). |
| 3 | **RISK** | `src/ham/geometry/metric.py:69-76` | **`arc_length` midpoint is not projected onto the manifold.** `self.metric_fn(0.5 * (x1 + x2), v)` evaluates the metric at the ambient-space midpoint, which may not lie on a constrained manifold (e.g. the sphere $S^2$). The ARCH_SPEC (§ 2.1) requires a `project` method on every `Manifold` for exactly this situation. | Apply `self.manifold.project(0.5 * (x1 + x2))` before passing to `metric_fn`. |
| 4 | **RISK** | `src/ham/geometry/metric.py:69-76` | **No guard for degenerate paths in `arc_length`.** If `gamma` has fewer than 2 points (shape `(1, D)` or `(0, D)`), `gamma[:-1]` is empty and `jax.vmap(segment_length)` receives empty arrays. Behaviour under `jit` is defined (returns 0 via `jnp.sum` of empty), but the silent success may mask bugs upstream. | Add an assertion or return `0.0` explicitly for `gamma.shape[0] < 2`. |
| 5 | **RISK** | `src/ham/geometry/metric.py:35` | **`energy` has no NaN guard at $v = 0$.** `self.metric_fn(x, v)**2` propagates any NaN from subclasses that use `jnp.linalg.norm` (which has NaN gradients at the origin). Subclasses in `zoo.py` correctly use `safe_norm`, but the test mocks in `tests/test_metric.py` use bare `jnp.linalg.norm`, which means tests pass while masking a gradient-safety responsibility that the base class does not enforce or document. | Document in the `metric_fn` docstring that implementations must be gradient-safe at $v = 0$ (or use `safe_norm`). |
| 6 | **STYLE** | `src/ham/geometry/metric.py:38-42` | **`inner_product` recreates `jax.hessian(...)` on every call.** While this is functionally correct (JAX retraces under JIT), allocating a fresh higher-order function object per invocation adds overhead for eager-mode profiling and debugging. | Compute `g_fn` once and store via a helper or `functools.partial`. Alternatively, accept as intentional and add a brief comment. |
| 7 | **STYLE** | `src/ham/geometry/metric.py:1-79` | **Type annotations use deprecated `jnp.ndarray`.** `jnp.ndarray` is an alias that JAX plans to remove. The canonical type is `jax.Array` (or `jaxtyping.Float[Array, "D"]` for shape annotations). | Replace `jnp.ndarray` with `jax.Array` throughout. |
| 8 | **STYLE** | `src/ham/geometry/metric.py:45-64` | **Spray closure captures outer `v` with same name as JVP tangent.** In `d_dv_fixed_v`, the closed-over `v` is the *velocity*, while the `(v,)` in the JVP tangent tuple is the *perturbation direction* (which happens to equal the velocity per the EL equation). The reuse of the name is correct but non-obvious. | Add a one-line comment: `# tangent direction = velocity per Euler-Lagrange contraction`. |
| 9 | **STYLE** | `src/ham/geometry/metric.py:66-67` | **`geod_acceleration` lacks a docstring.** All other public methods are documented. | Add a brief docstring: `"""Returns geodesic acceleration ddot{x} = -2G(x, v)."""` |

---

## Test Coverage Assessment

| Public Method | Tested? | Notes |
|---------------|---------|-------|
| `metric_fn` (abstract) | Yes | Tested via `EuclideanMetric` and `CurvedMetric` mocks. |
| `energy` | Indirect | Not tested directly; exercised implicitly by `spray` and `inner_product`. |
| `spray` | Yes | Euclidean-zero test (`test_euclidean_spray_is_zero`), 2-homogeneity test (`test_spray_homogeneity`). |
| `geod_acceleration` | Yes | Sign-convention test (`test_acceleration_sign`). |
| `inner_product` | Yes | Euclidean identity test and curved diagonal-metric test. |
| `arc_length` | **No** | No test exists. Gap. |

### Gap Analysis

1. **`arc_length` is untested.** This is the only public method without any test. Recommended: add a test computing the length of a straight-line path under Euclidean metric and verify it equals the Euclidean distance.
2. **No gradient correctness tests.** None of the tests use `jax.test_util.check_grads` or equivalent finite-difference checks. The spray and inner product rely on higher-order autodiff; a gradient regression test would catch subtle tracer bugs.
3. **No JIT/vmap integration tests.** All tests call methods eagerly. Adding `jax.jit(metric.spray)(x, v)` and `jax.vmap(metric.spray)(xs, vs)` tests would verify transform compatibility.
4. **Edge-case tests missing.** No tests for: zero velocity vector, collinear directions, very large velocity magnitudes, or near-singular Hessians.
5. **Test mocks use unsafe `jnp.linalg.norm`.** `EuclideanMetric` in `tests/test_metric.py` uses `jnp.linalg.norm(v)` instead of `safe_norm(v)`, so gradient-safety of the base class pipeline is not validated.

---

## Positive Patterns

1. **Efficient mixed-partial computation.** Using `jax.jvp` for the $J_x(\nabla_v E) \cdot v$ term in `spray` (line 55-57) is more efficient than materialising the full Jacobian, and is the recommended JAX pattern.
2. **Correct Equinox module design.** Inheriting from `eqx.Module` with `manifold` as a PyTree field ensures all subclasses are valid JAX PyTrees without boilerplate.
3. **Direction-dependent inner product.** The `inner_product(x, v, w1, w2)` API correctly includes the reference direction $v$, which is essential for Finsler (non-Riemannian) metrics where $g_{ij}$ depends on direction.
4. **Clean separation of concerns.** The base class only defines physics; subclasses only define `metric_fn`. This matches the Metric-First principle in `spec/ARCH_SPEC.md § 1`.
5. **Spray homogeneity is tested.** The $G(x, \lambda v) = \lambda^2 G(x, v)$ property test is a strong structural invariant that catches sign and scaling errors.
