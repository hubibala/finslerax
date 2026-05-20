# Code Review: `ham.geometry.curvature`

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)  
**Source:** `src/ham/geometry/curvature.py` (99 lines)  
**Tests:** **None.** No `tests/test_curvature.py` exists. The only mention of "curvature" in the test suite is a comment in `tests/test_transport.py:69`.

---

## Summary

The module implements Finsler curvature quantities (nonlinear connection, Riemann curvature tensor, sectional/flag curvature, scalar curvature) purely via autodiff of `FinslerMetric.spray`. The core tensor computation in `riemann_curvature_tensor` is structurally sound and the `einsum` index contractions match the documented formula. However, two bugs and several risks were identified: (1) `scalar_curvature` uses raw `jnp.linalg.norm` instead of the codebase's `safe_norm`, producing NaN gradients at zero tangent vectors; (2) `scalar_curvature` uses a hardcoded PRNG seed and Euclidean Gram-Schmidt, making it silently wrong for highly anisotropic metrics and non-reproducibly "random" in a misleading way; (3) the entire module lacks batch-dimension support mandated by `spec/ARCH_SPEC.md § 1`; and (4) no tests exist at all. The deep autodiff nesting (4th-order through `riemann_curvature_tensor`) is an inherent performance concern but is architecturally consistent with the Metric-First/Implicit Dynamics philosophy.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/geometry/curvature.py:92,95` | **`jnp.linalg.norm` used instead of `safe_norm`.** `scalar_curvature` normalises tangent vectors via `jnp.linalg.norm(t1)` and `jnp.linalg.norm(t2)`. If `manifold.to_tangent` returns a near-zero or exactly-zero vector (e.g. when the random sample lies along the normal), `jnp.linalg.norm` produces `0.0`, the division yields `inf`/`NaN`, and—more critically—the backward pass through `sqrt(0)` produces `NaN` gradients. The codebase already provides `ham.utils.math.safe_norm` (with a canonical `GRAD_EPS = 1e-12`) for exactly this purpose. | Replace with `safe_norm(t1)` and `safe_norm(t2)`, importing from `ham.utils.math`. |
| 2 | **BUG** | `src/ham/geometry/curvature.py:93` | **Euclidean `jnp.dot` used for Gram-Schmidt orthogonalisation.** The line `t2 = t2 - jnp.dot(t1, t2) * t1` orthogonalises using the ambient Euclidean inner product. For metrics with high condition number or strong anisotropy (e.g. a Randers metric with $\lVert W\rVert_h \to 1$), two vectors that are Euclidean-orthogonal can be nearly metric-parallel. This makes the sectional curvature denominator $g_{11}g_{22} - g_{12}^2$ degenerate even though the vectors appear well-conditioned, silently returning `0.0` via the `jnp.where` guard. The orthogonalisation should use `metric.inner_product` for correctness. | Use `t2 = t2 - metric.inner_product(x, t1, t1, t2) / metric.inner_product(x, t1, t1, t1) * t1`. |
| 3 | **RISK** | `src/ham/geometry/curvature.py:85` | **Hardcoded `jax.random.PRNGKey(42)` inside `scalar_curvature`.** This makes the function deterministic (always the same tangent plane), non-configurable, and misleading—the docstring says "random vectors" but they never change. A user calling this function twice at different points with different geometries gets curvature evaluated in the *same ambient directions*, which may not span the tangent space well on curved manifolds. Moreover, the function does not accept `key` as a parameter, breaking the JAX convention for PRNG-consuming functions. | Accept `key: jax.random.PRNGKey` as a parameter. Alternatively, if determinism is desired, document the function as evaluating a *fixed* canonical plane rather than claiming randomness. |
| 4 | **RISK** | `src/ham/geometry/curvature.py:5–99` | **No batch-dimension support.** All four public functions accept single-point inputs `(D,)`. `spec/ARCH_SPEC.md § 1` mandates batch-first `(B, ...)` semantics. Users must manually `jax.vmap` over these functions, and the deep autodiff nesting (see #6) makes vmap compilation extremely expensive. | Provide `vmap`-wrapped batch entry points, or at minimum document that these are single-point functions requiring explicit `vmap`. |
| 5 | **RISK** | `src/ham/geometry/curvature.py:67–68` | **`1e-12` epsilon for degenerate-plane guard is too tight for float32.** `jnp.maximum(denominator, 1e-12)` and `jnp.where(denominator < 1e-12, ...)` use $10^{-12}$ as the threshold. For float32 inputs (JAX default), the denominator involves products of inner products of unit-ish vectors, so machine-epsilon–scale denominators are around $10^{-7}$. A denominator of $10^{-10}$ would pass the guard but produce a curvature value amplified by $10^{10}$, swamping any meaningful signal. The codebase defines `NORM_EPS = 1e-8` in `ham.utils.math` for this purpose. | Use `NORM_EPS` (or at least `1e-8`) instead of `1e-12`. Also guard against *negative* denominators (which indicate a degenerate fundamental tensor). |
| 6 | **RISK** | `src/ham/geometry/curvature.py:11–47` | **4th-order autodiff nesting creates compilation and memory pressure.** `riemann_curvature_tensor` → `jacfwd` of `nonlinear_connection` → `jacfwd` of `spray` → which internally uses `jax.grad` + `jax.hessian` of `energy`. This is 4th-order differentiation. For `D`-dimensional inputs, the Jacobian tensors scale as $O(D^4)$ and XLA compilation can take minutes even for $D = 3$. This is architecturally consistent with the Implicit Dynamics principle but should be documented as a known performance characteristic. | Add a docstring note about compilation cost. Consider caching/memoising the compiled function, or providing a `jax.checkpoint`-wrapped variant to trade recomputation for memory. |
| 7 | **RISK** | `src/ham/geometry/curvature.py:16–17` | **`N_fn` lambda re-wraps `nonlinear_connection` inside `riemann_curvature_tensor`, causing redundant tracing.** `N_fn(x, v)` is called directly on line 29, while `jacfwd(N_fn, argnums=0)` and `jacfwd(N_fn, argnums=1)` each independently trace through `N_fn` (and through `spray` inside it). The explicit call on line 29 is a 3rd independent trace of the same computation. | Compute `N` from the already-traced Jacobian output (e.g. `N = dN_dv @ v` using Euler's theorem, or evaluate `N_fn` once and reuse via `jax.lax.stop_gradient` if appropriate). Alternatively, fuse all three computations into a single `jacfwd` call that returns both the primal and the Jacobian. |
| 8 | **STYLE** | `src/ham/geometry/curvature.py:5,11` | **`nonlinear_connection` and `riemann_curvature_tensor` are public (no underscore) but not exported in `__init__.py`.** The package `__init__.py` exports only `sectional_curvature` and `scalar_curvature`. The two tensor functions are importable via `from ham.geometry.curvature import ...` but not via the package namespace, creating an ambiguous API surface. | Either add them to `__all__` in `__init__.py` (if they are intended public API) or prefix with `_` to mark them as internal. |
| 9 | **STYLE** | `src/ham/geometry/curvature.py:70–99` | **`scalar_curvature` is misnamed.** In differential geometry, scalar curvature is the trace of the Ricci tensor (sum over all independent sectional curvatures). This function returns a single sectional curvature from one fixed tangent plane, which for manifolds with $\dim \geq 3$ is not scalar curvature. The docstring partially disclaims this ("Approximation for computational testing") but the function name will mislead users. | Rename to `sample_sectional_curvature` or `flag_curvature_sample`, and reserve `scalar_curvature` for a proper Ricci-trace implementation if needed. |

---

## Test Coverage Assessment

| Public Function | Exported | Tested | Notes |
|---|---|---|---|
| `nonlinear_connection` | No | **No** | No test exists. Should verify $N^i_j v^j = 2G^i$ (Euler's theorem for 2-homogeneous spray). |
| `riemann_curvature_tensor` | No | **No** | No test exists. Should verify $R = 0$ for Euclidean metric, known curvature for sphere. |
| `sectional_curvature` | Yes | **No** | No test exists. Should verify $K = 1/r^2$ on a sphere of radius $r$, $K = 0$ for Euclidean. |
| `scalar_curvature` | Yes | **No** | No test exists. Same tests as sectional for dim = 2 surfaces. |

**Gap analysis:** This is the only module in `src/ham/geometry/` with **zero** test coverage. All other geometry modules (`metric.py`, `manifold.py`, `transport.py`, `zoo.py`, `mesh.py`, `surfaces.py`) have corresponding test files. A minimal test suite should include:
1. **Euclidean zero-curvature test:** `sectional_curvature(Euclidean(...), x, v1, v2)` ≈ 0.
2. **Sphere constant-curvature test:** `sectional_curvature(Riemannian(Sphere(3), ...), x, v1, v2)` ≈ 1 for unit sphere.
3. **Gradient test:** `jax.grad(sectional_curvature, argnums=1)` does not produce NaN on well-conditioned inputs.
4. **JIT compilation test:** `jax.jit(sectional_curvature)` compiles without error (this validates the 4th-order autodiff pipeline end-to-end).

---

## Positive Patterns

1. **Correct `jacfwd` usage for tensor computation.** Using `jax.jacfwd` (forward-mode) for Jacobians of vector-valued functions is the right choice over `jacrev`, as the output dimension equals the input dimension.
2. **Clean index algebra.** The `einsum` expressions on lines 38–41 correctly implement the documented formula with explicit index labels and comments explaining each term.
3. **Safe division pattern in `sectional_curvature`.** The `jnp.maximum` + `jnp.where` guard (lines 67–68) follows the standard JAX-safe division pattern and avoids NaN in both forward and backward passes (modulo the too-tight epsilon noted in issue #5).
4. **Metric-First consistency.** All curvature computations derive from `metric.spray` via pure autodiff, fully consistent with `spec/ARCH_SPEC.md § 2` (Implicit Dynamics). No manual Christoffel symbol implementations.
5. **Correct `inner_product` usage in `sectional_curvature`.** The flag curvature numerator correctly passes `v1` as the reference direction for the fundamental tensor, consistent with the Finsler flag curvature convention.
