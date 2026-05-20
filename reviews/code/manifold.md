# Code Review: `ham.geometry.manifold`

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)  
**Source:** `src/ham/geometry/manifold.py` (161 lines)  
**Tests:** No dedicated `tests/test_manifold.py`; coverage via `tests/test_surfaces.py` and `tests/test_transport.py` (mock manifolds)

---

## Summary

The `Manifold` ABC is compact and correctly implements the topology abstraction from `spec/ARCH_SPEC.md § 2.1`. The `Manifold(eqx.Module, ABC)` inheritance is idiomatic Equinox. The default `log_map` implementation is carefully engineered with a custom JVP (`_safe_norm_ratio_jvp`) to avoid NaN gradients at coincident points, which is commendable. Two substantive issues were identified: (1) the `_safe_norm_ratio_jvp` helper uses bare `jnp.linalg.norm` instead of the canonical `safe_norm`, reintroducing a NaN-gradient hazard at the origin; and (2) hardcoded `1e-12` thresholds bypass the centralised constants in `utils/math.py`, creating a maintenance divergence. The file has no dedicated test module, though concrete manifold methods are exercised indirectly. Test gaps exist for the default `log_map`, JIT/vmap composition of abstract methods, and gradient correctness of `_safe_norm_ratio_jvp`.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/geometry/manifold.py:14-15` | **`_safe_norm_ratio_jvp` uses `jnp.linalg.norm` instead of `safe_norm`.** The forward pass computes `nx = jnp.linalg.norm(x, ...)` and `ny = jnp.linalg.norm(y, ...)`. `jnp.linalg.norm` has undefined gradients at the origin ($\nabla \|x\| = x / \|x\|$ → NaN when $x = 0$). While the primal value is guarded by the `jnp.where(ny < 1e-12, ...)` branch, JAX's `custom_jvp` evaluates *both* branches of `jnp.where` during differentiation unless the guard is structurally baked in. Any higher-order differentiation (e.g. Hessian) through `log_map → _safe_norm_ratio_jvp` when `x ≈ y` will trace through the unsafe `jnp.linalg.norm` branch and produce NaN tangents. The codebase has a canonical `safe_norm` in `utils/math.py` that uses `sqrt(max(sum(x²), eps))` — this should be used instead. | Replace `jnp.linalg.norm(x, ...)` and `jnp.linalg.norm(y, ...)` with `safe_norm(x, ...)` and `safe_norm(y, ...)` from `utils/math.py` in both the primal and JVP functions. |
| 2 | **RISK** | `src/ham/geometry/manifold.py:36` | **JVP of `dnx` uses `nx_safe` which is set to 1.0 when `is_zero` (i.e. when $\|y\| < 10^{-12}$), not when $\|x\| < 10^{-12}$.** The guard `jnp.where(nx < 1e-12, 0.0, ...)` protects the $x = 0$ case independently, but the denominator used is `nx_safe = jnp.where(is_zero, 1.0, nx)` — this replaces `nx` with 1.0 when *y* is near-zero, not when *x* is near-zero. If $\|x\| \approx 0$ but $\|y\| \gg 0$, `dnx` divides by `nx` (which is near-zero). The outer `jnp.where(nx < 1e-12, 0.0, ...)` guards this, but with bare `jnp.linalg.norm` the gradient through `nx` is already NaN before the select. | Introduce a separate `nx_safe_for_dnx = jnp.maximum(nx, 1e-12)` guard for the `dnx` computation, independent of the `is_zero` (y-based) guard. |
| 3 | **RISK** | `src/ham/geometry/manifold.py:14-16, 26, 36` | **Hardcoded `1e-12` thresholds.** The file uses `1e-12` in four places as the zero-norm threshold. `utils/math.py` defines `GRAD_EPS = 1e-12` for exactly this purpose. The duplication is a maintenance hazard: if the canonical constant is updated, this file remains stale. | Import and use `GRAD_EPS` from `utils/math.py`. Note: `safe_norm` is already imported but the associated constant is not. |
| 4 | **RISK** | `src/ham/geometry/manifold.py:128-145` | **Default `log_map` scales by `\|y - x\| / \|v\|` but `v` may be zero when `y - x` is purely normal.** If $y - x$ is entirely orthogonal to $T_x M$ (e.g. moving radially from a sphere), then `v = self.to_tangent(x, y - x)` is the zero vector, while `y - x` is non-zero. The `_safe_norm_ratio_jvp` returns 1.0 when $\|v\| < 10^{-12}$, yielding `v * 1.0 = 0`, which is geometrically correct (the tangent component is genuinely zero). However, for near-normal displacements where $\|v\|$ is non-zero but extremely small, the ratio $\|y - x\| / \|v\|$ can be astronomically large, producing a numerically unstable scaling. This matters for the AVBD solver (`src/ham/solvers/avbd.py:103-104`) which calls `log_map` at every vertex. | Clamp the scale factor: `scale = jnp.clip(_safe_norm_ratio_jvp(y - x, v), 0.0, max_scale)` or document the near-normal failure mode and require callers to ensure points are "horizontally close." |
| 5 | **RISK** | `src/ham/geometry/manifold.py:8-42` | **`_safe_norm_ratio_jvp` only defines a custom JVP, not a custom VJP.** `jax.custom_jvp` composes correctly with `jax.grad` (via VJP-from-JVP transpose), but JAX's transposition of custom JVPs can be fragile for complex control flow with `jnp.where`. If any downstream code calls `jax.grad` through `log_map` in a way that requires VJP transposition, subtle numerical errors may arise. The training losses (`src/ham/training/losses.py:129, 360`) differentiate through `log_map`, making this a live risk. | Add a focused gradient regression test: `jax.grad(lambda y: jnp.sum(manifold.log_map(x, y)))(y)` for coincident and near-coincident points, verifying no NaN or Inf. |
| 6 | **STYLE** | `src/ham/geometry/manifold.py:5` | **`from typing import Tuple` is deprecated.** Since Python 3.9+, `tuple[int, ...]` is the standard spelling. The ARCH_SPEC targets modern Python (JAX requires ≥ 3.10). | Replace `from typing import Tuple` with bare `tuple` in type hints. |
| 7 | **STYLE** | `src/ham/geometry/manifold.py:9` | **Misleading function name `_safe_norm_ratio_jvp`.** The `_jvp` suffix implies this is the JVP rule, but it's actually the *primal* function decorated with `@jax.custom_jvp`. The JVP rule is `_safe_norm_ratio_jvp_def`. | Rename to `_safe_norm_ratio` (primal) and `_safe_norm_ratio_jvp` (JVP rule) for clarity. |
| 8 | **STYLE** | `src/ham/geometry/manifold.py:1-161` | **Type annotations use deprecated `jnp.ndarray`.** The canonical JAX array type is `jax.Array`. All method signatures use `jnp.ndarray`. | Replace `jnp.ndarray` with `jax.Array` in type hints, consistent with `random_sample`'s `key: jax.Array` parameter which already uses the correct type. |
| 9 | **STYLE** | `src/ham/geometry/manifold.py:110-115` | **`exp_map` default is trivially `retract` without any comment on when this is inadequate.** All concrete manifolds (`Sphere`, `Hyperboloid`) override `exp_map` with closed-form formulas. The default passthrough is correct as a fallback, but could mislead implementors of new manifolds into thinking the retraction is always sufficient. | Add a one-line docstring note: "Subclasses with known geodesic formulas should override this with the exact exponential map." |

---

## Test Coverage Assessment

### Dedicated Test File

There is **no** `tests/test_manifold.py`. The `Manifold` ABC's concrete methods (`log_map`, `exp_map`) are tested only indirectly through concrete subclass tests in `tests/test_surfaces.py` and mock manifolds in `tests/test_transport.py`.

### Coverage Matrix

| Public Method | Tested? | Notes |
|---------------|---------|-------|
| `ambient_dim` (abstract) | Yes | All concrete manifolds tested in `test_surfaces.py`. |
| `intrinsic_dim` (abstract) | Yes | All concrete manifolds tested in `test_surfaces.py`. |
| `project` (abstract) | Yes | Sphere, Torus, Paraboloid, EuclideanSpace tested. |
| `to_tangent` (abstract) | Yes | Sphere tangent orthogonality tested (`test_surfaces.py:33`). |
| `retract` (abstract) | Yes | Sphere, EuclideanSpace tested. |
| `exp_map` (default) | Indirect | Tested via `Sphere.exp_map` (which overrides the default). The *default* passthrough (`return self.retract(x, v)`) is never directly tested. |
| `log_map` (default) | **Partial** | `EuclideanSpace.log_map` overrides the default and is tested (`test_surfaces.py:84`). `Sphere.log_map` also overrides. The *default* `Manifold.log_map` implementation (the one using `_safe_norm_ratio_jvp`) is only indirectly exercised by manifolds that do *not* override it (Torus, Paraboloid), but those have no `log_map` tests. |
| `random_sample` (abstract) | Yes | All concrete manifolds tested. |
| `_safe_norm_ratio_jvp` (private) | **No** | No test validates the custom JVP rule directly — no gradient correctness test, no coincident-point test, no near-normal displacement test. |

### Gap Analysis

1. **Default `log_map` is untested.** No test calls `Manifold.log_map` on a manifold that does *not* override it (e.g. `Torus`, `Paraboloid`). This is the implementation with the custom JVP scaling correction — the most complex code in the file — and it has zero direct test coverage.
2. **`_safe_norm_ratio_jvp` has no gradient correctness test.** The custom JVP is the most fragile code in this module. A test using `jax.test_util.check_grads` (or finite-difference comparison) should validate the JVP against numerical derivatives for normal, coincident, and near-normal cases.
3. **No JIT/vmap composition tests.** The abstract methods and `log_map` are called under `jax.vmap` in the AVBD solver and training losses. No test verifies `jax.jit(manifold.log_map)` or `jax.vmap(manifold.log_map)` work correctly.
4. **No edge-case tests.** Missing tests for: coincident points ($x = y$), antipodal points (Sphere), purely normal displacements, zero tangent vectors.
5. **Mock manifolds in `test_transport.py` use bare `jnp.linalg.norm`.** The `Sphere` mock (`test_transport.py:29`) uses `jnp.linalg.norm` in `project` and `to_tangent`, which masks gradient-safety issues.

---

## Positive Patterns

1. **Careful custom JVP for `log_map` scaling.** The `_safe_norm_ratio_jvp` function correctly identifies the numerical hazard of the $\|y - x\| / \|v\|$ ratio when points coincide and provides a clean JVP rule with fallback to 1.0. The engineering intent is excellent.
2. **Clean ABC/Equinox composition.** `Manifold(eqx.Module, ABC)` is the idiomatic pattern for Equinox-compatible abstract classes. Abstract properties (`ambient_dim`, `intrinsic_dim`) are correctly declared with `@property @abstractmethod`.
3. **Thorough docstrings.** Every public method has a clear docstring specifying mathematical semantics, arguments, and return values, including the retraction axioms for `retract`.
4. **Correct separation of concerns.** The manifold defines only topology (projection, tangent space), not geometry (metrics, geodesics), precisely matching `spec/ARCH_SPEC.md § 2.1`.
5. **`safe_norm` import.** The canonical `safe_norm` is imported from `utils/math.py`, establishing the right pattern — though it is not used in the `_safe_norm_ratio_jvp` helper where it is most needed (see Issue #1).
