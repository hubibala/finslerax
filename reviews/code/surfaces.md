# Code Review: `surfaces.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)  
**Source:** `src/ham/geometry/surfaces.py`  
**Tests:** `tests/test_surfaces.py`, `tests/test_hyperboloid.py`

## Summary

The module provides five concrete `Manifold` subclasses and two custom-JVP helpers. `Sphere`, `Hyperboloid`, and `EuclideanSpace` are well-implemented with correct batch-dimension handling, proper numerical stability patterns, and sound custom JVPs. However, `Torus` and `Paraboloid` contain a **batch-dimension BUG**: they use raw positional indexing (`x[:2]`, `x[2]`) instead of `x[..., :2]`, violating the ARCH_SPEC § 1 batch-first convention and breaking under `vmap` over leading dimensions. `_safe_arccos` is dead code (defined with a custom JVP but never called). Test coverage has significant gaps — `exp_map`/`log_map`/`parallel_transport` are untested for Hyperboloid, and no JIT/vmap/grad compatibility tests exist for any class.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | [surfaces.py](src/ham/geometry/surfaces.py#L150) `Torus.project` | Uses `x[:2]`, `x[2]` — positional indexing without `...` ellipsis. Breaks when `x` has a leading batch dimension `(B, 3)`. All other manifolds (`Sphere`, `Hyperboloid`, `EuclideanSpace`) correctly use `x[..., :2]` etc. This violates ARCH_SPEC § 1 ("Batch-First"). | Replace `x[:2]` → `x[..., :2]`, `x[2]` → `x[..., 2]`. Replace `jnp.concatenate` of scalars with `jnp.stack` on `...`-indexed slices. Rewrite `jnp.array(...)` fallbacks as broadcastable `jnp.zeros_like`-based constructions. |
| 2 | **BUG** | [surfaces.py](src/ham/geometry/surfaces.py#L171) `Torus.to_tangent` | Same positional-indexing issue as `project`. `x[:2]`, `jnp.dot(n, v)` fail with batched inputs. | Same ellipsis fix. Replace `jnp.dot(n, v)` with `jnp.einsum('...i,...i->...', n, v)` or `jnp.sum(n * v, axis=-1)`. |
| 3 | **BUG** | [surfaces.py](src/ham/geometry/surfaces.py#L215) `Paraboloid.project` | Uses `x[0]`, `x[1]` — single-point only. No batch support. | Use `x[..., 0]`, `x[..., 1]`. Reconstruct output via `jnp.stack([x[..., 0], x[..., 1], z], axis=-1)`. |
| 4 | **BUG** | [surfaces.py](src/ham/geometry/surfaces.py#L219) `Paraboloid.to_tangent` | Same positional-indexing issue. Also `jnp.dot(n, v)` is unbatched. | Same ellipsis fix plus `jnp.sum(n * v, axis=-1, keepdims=True)`. |
| 5 | **BUG** | [surfaces.py](src/ham/geometry/surfaces.py#L228) `Paraboloid.retract` | `x[:2]`, `delta[:2]` — unbatched indexing. | Use `x[..., :2]`, `delta[..., :2]`. |
| 6 | **RISK** | [surfaces.py](src/ham/geometry/surfaces.py#L101-L109) `Sphere.parallel_transport` | For near-antipodal points ($\langle x, y \rangle \approx -r^2$), the denominator $r^2 + \langle x, y \rangle \to 0$. The `jnp.maximum(..., 1e-5)` clamp prevents NaN but produces a numerically meaningless large-magnitude result. Parallel transport is geometrically undefined for antipodal points; the function should either document this or return `v` unchanged as a fallback. | Add a `jnp.where(denom < threshold, v, transported)` guard or document the precondition that `x` and `y` must not be antipodal. |
| 7 | **RISK** | [surfaces.py](src/ham/geometry/surfaces.py#L272-L285) `_safe_arccos` | Defined with a custom JVP but **never called** anywhere in the codebase. The `Sphere.log_map` performs its own clip-then-`jnp.arccos` pattern instead of delegating to `_safe_arccos`. Dead code with a non-trivial custom JVP is a maintenance risk — it may drift out of correctness without any test exercising it. | Either use `_safe_arccos` in `Sphere.log_map` (replacing the manual clip+arccos), or remove it. |
| 8 | **RISK** | [surfaces.py](src/ham/geometry/surfaces.py#L391) `Hyperboloid.retract` | Hardcoded `max_norm = 10.0` magic number for clamping tangent vector magnitude before `exp_map`. Undocumented. Could silently distort large retractions. | Extract to a module-level constant (e.g., `RETRACT_MAX_NORM`) or make it a configurable field. Document its purpose: preventing `sinh`/`cosh` overflow. |
| 9 | **RISK** | [surfaces.py](src/ham/geometry/surfaces.py#L370-L380) `Hyperboloid.exp_map` | Does not re-project the result onto the hyperboloid. Floating-point drift can cause $\langle \text{result}, \text{result}\rangle_L$ to deviate from $-1$ after many chained `exp_map` calls (e.g., in ODE solvers). The `retract` method calls `project` after `exp_map`, but direct `exp_map` callers do not benefit. | Document that callers performing iterative integration should use `retract` or manually `project` the output. |
| 10 | **RISK** | [surfaces.py](src/ham/geometry/surfaces.py#L383-L400) `Hyperboloid.log_map` | Uses `jnp.arcsinh(norm_u)` which is mathematically equivalent to $\text{acosh}(-\langle x,y\rangle_L)$ but can lose precision when $\langle x,y\rangle_L$ is very close to $-1$ (nearby points). The `arcsinh` path is fine for moderate distances; for very small distances the Taylor branch activates, but the transition threshold `NORM_EPS = 1e-8` is fairly aggressive — norm values in `(1e-8, 1e-6)` use `arcsinh`/division which may have higher relative error than the Taylor approximation for those magnitudes. | Consider raising the Taylor threshold to `TAYLOR_EPS` (1e-6) for consistency with `exp_map`, or verify that `arcsinh(t)/t` has acceptable relative error down to `t = 1e-8`. |
| 11 | **RISK** | [surfaces.py](src/ham/geometry/surfaces.py#L403-L412) `Hyperboloid.parallel_transport` | Uses `jnp.maximum(1.0 - xy, 2.0)` as the denominator. For valid hyperboloid points, $1 - \langle x,y\rangle_L \geq 2$ so the clamp never activates, but the `jnp.maximum` creates a non-smooth gradient at the boundary ($x = y$). Under `jax.grad`, if `1-xy` is exactly 2.0, the gradient may flow through either branch depending on implementation, potentially zeroing out useful gradient signal for near-identity transports. | Use `jnp.maximum(1.0 - xy, 2.0 + GRAD_EPS)` to keep the smooth branch always active for valid inputs, or use a soft clamp. |
| 12 | **STYLE** | [surfaces.py](src/ham/geometry/surfaces.py#L118-L200) `Torus`, `Paraboloid` | These classes lack `log_map` and `parallel_transport` methods that `Sphere`, `Hyperboloid`, and `EuclideanSpace` all provide. This API inconsistency means generic code cannot depend on these methods being available across all manifolds. | Either add implementations (even approximate ones using projected retraction) or document that these manifolds only support the minimal `Manifold` interface. |
| 13 | **STYLE** | [surfaces.py](src/ham/geometry/surfaces.py#L414) `Hyperboloid.metric_tensor` | Returns a constant Minkowski matrix $\text{diag}(-1, 1, \ldots, 1)$ independent of $x$. This is the **ambient** metric tensor, not the **induced** metric on the hyperboloid. The method name suggests the latter. No other manifold class defines this method, creating an inconsistent interface. | Rename to `ambient_metric_tensor` or `minkowski_tensor`, or remove and let the `FinslerMetric` layer handle it per ARCH_SPEC § 2.2. |
| 14 | **STYLE** | [surfaces.py](src/ham/geometry/surfaces.py#L1-L12) Module docstring | The docstring is a changelog of review-driven fixes rather than a description of the module's purpose and contents. | Move the changelog to a `CHANGELOG` section at the bottom or into commit messages. Add a summary of the module's purpose. |

---

## Test Coverage Assessment

| Public Method | `test_surfaces.py` | `test_hyperboloid.py` | Gap |
|---|---|---|---|
| `Sphere.__init__`, `ambient_dim`, `intrinsic_dim` | OK | — | — |
| `Sphere.project` | OK | — | — |
| `Sphere.to_tangent` | OK | — | — |
| `Sphere.exp_map` / `retract` | OK | — | — |
| `Sphere.log_map` | OK (roundtrip) | — | No antipodal-point edge case |
| `Sphere.parallel_transport` | **MISSING** | — | No test at all |
| `Sphere.random_sample` | OK | — | — |
| `Torus.project` | OK (via vmap) | — | — |
| `Torus.to_tangent` | **MISSING** | — | No test |
| `Torus.exp_map` / `retract` | **MISSING** | — | No test |
| `Torus.random_sample` | OK (implicit via project consistency) | — | — |
| `Paraboloid.project` | OK | — | — |
| `Paraboloid.to_tangent` | **MISSING** | — | No test |
| `Paraboloid.exp_map` / `retract` | **MISSING** | — | No test |
| `Paraboloid.random_sample` | OK | — | — |
| `Hyperboloid.project` | — | OK (constraints + idempotence) | — |
| `Hyperboloid.to_tangent` | — | OK (orthogonality) | — |
| `Hyperboloid.exp_map` | — | **MISSING** | No test |
| `Hyperboloid.log_map` | — | **MISSING** | No test |
| `Hyperboloid.parallel_transport` | — | **MISSING** | No test |
| `Hyperboloid.retract` | — | OK (stays on manifold) | — |
| `Hyperboloid.random_sample` | — | OK | — |
| `Hyperboloid.metric_tensor` | — | OK (signature check) | — |
| `EuclideanSpace.project` | OK | — | — |
| `EuclideanSpace.exp_map` | OK | — | — |
| `EuclideanSpace.log_map` | OK | — | — |
| `EuclideanSpace.parallel_transport` | **MISSING** | — | No test |
| `EuclideanSpace.retract` | **MISSING** | — | No test |
| `_safe_minkowski_self_norm` | — | Implicit via project | No direct unit test; no JVP/grad test |
| `_safe_arccos` | **MISSING** | — | Dead code, zero tests |

**Cross-cutting gaps:**
- No `jax.jit` compatibility tests for any class.
- No `jax.vmap` compatibility tests (except `Torus.project` via `jax.vmap(torus.project)`).
- No `jax.grad` or `jax.jacfwd` tests through `exp_map`/`log_map`.
- No edge-case tests: zero tangent vectors, antipodal/coincident points, very large tangent vectors.
- No exp-log roundtrip consistency test for `Hyperboloid`.

---

## Positive Patterns

1. **Well-designed custom JVPs**: `_safe_minkowski_self_norm` correctly implements a gradient-safe Minkowski norm with an explicit `is_zero` guard that prevents NaN propagation. The JVP formula is mathematically sound.
2. **Consistent use of `safe_norm` from `utils.math`**: The module imports the canonical `safe_norm` rather than reimplementing it, following the P0-2 review fix.
3. **Taylor-expansion stability**: `Sphere.exp_map`, `Sphere.log_map`, `Hyperboloid.exp_map`, and `Hyperboloid.log_map` all use explicit Taylor-branch switching near $\theta \approx 0$ with `jnp.where`, avoiding `0/0` in `sin(θ)/θ` and `sinh(θ)/θ` patterns. The expansion coefficients are correct.
4. **Static fields on `eqx.Module`**: Geometric constants (`radius`, `R`, `r`, `_intrinsic_dim`) are correctly marked `static=True`, ensuring they are treated as compile-time constants by JAX's tracing and not traced as dynamic values.
5. **Sphere and Hyperboloid batch support**: These two classes consistently use `[..., idx]` indexing and `axis=-1` reductions, correctly supporting arbitrary leading batch dimensions without requiring explicit `vmap`.
6. **Sphere.project zero-vector guard**: Gracefully handles zero-length input by falling back to a canonical "north pole" direction, preventing NaN propagation.
