# Code Review: `ham.geometry.transport`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)  
**Source:** `src/ham/geometry/transport.py` (67 lines)  
**Tests:** `tests/test_transport.py`

## Summary

The Berwald parallel transport implementation is compact and structurally correct in its use of `jax.jacfwd` for Christoffel symbol computation and `jax.lax.scan` for ODE integration. However, two bugs were identified: (1) the Euler time step `dt = 1/N` is systematically wrong (should be `1/(N-1)` for N sample points), causing O(1/N) undershoot of the transport integral, and (2) the `Connection` / `BerwaldConnection` classes are plain Python objects rather than `eqx.Module`, breaking JAX transform composability when the connection object itself must participate in tracing. Additional risks include the use of first-order Euler integration (versus the RK4 used by the geodesic solver in `spec/ARCH_SPEC.md § 4.4`) and the absence of batch-dimension handling required by `spec/ARCH_SPEC.md § 1`.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/geometry/transport.py:44` | **Time step `dt = 1.0 / len(path_x)` is off by one.** N sample points define N−1 intervals. The `scan` runs N iterations but the output discards the last one (line 52), so N−1 Euler steps are effectively used. With `dt = 1/N`, the total integrated parameter span is $(N-1)/N$ instead of $1$. For `N=20` (as in several tests), this is a ~5% systematic undershoot of the transport. The bug is currently masked because (a) the Euclidean test has $\Gamma = 0$, making the result trivially exact regardless of `dt`, (b) the sphere norm-preservation test checks an approximate invariant that Euler already violates, and (c) the holonomy test uses `atol=1e-1`. | Change to `dt = 1.0 / (len(path_x) - 1)`. Alternatively, accept `dt` as an explicit parameter so users can match their path parameterisation. |
| 2 | **BUG** | `src/ham/geometry/transport.py:5–14` | **`Connection` and `BerwaldConnection` are plain Python classes, not `eqx.Module`.** `FinslerMetric` inherits from `eqx.Module` (see `src/ham/geometry/metric.py:8`), making it a valid JAX pytree. But `Connection` is a bare class storing `self.metric`. This means: (a) `jax.jit(conn.parallel_transport)` works only because `conn` is captured in a closure, not passed as an argument; (b) `jax.vmap` over a connection parameter is impossible; (c) `jax.grad` w.r.t. the metric parameters through a `BerwaldConnection` call requires careful closure-based tracing. The ARCH_SPEC mandates Equinox modules for all geometric objects. | Make `Connection` inherit from `eqx.Module` with `metric: FinslerMetric` as a typed field. |
| 3 | **RISK** | `src/ham/geometry/transport.py:36–46` | **Forward Euler (order 1) for transport ODE.** The geodesic IVP solver uses RK4 (`spec/ARCH_SPEC.md § 4.4`), but parallel transport uses bare forward Euler. For curves with non-trivial curvature, first-order integration accumulates O(dt) local error, giving O(1) global error for fixed total time unless N is large. The `test_sphere_holonomy` test requires `N=200` and still uses `atol=1e-1`, suggesting the integrator is the accuracy bottleneck. | Upgrade to RK4 or at minimum a symplectic Euler variant. Match the integration order used in `src/ham/solvers/geodesic.py`. |
| 4 | **RISK** | `src/ham/geometry/transport.py:25–29` | **Double `jacfwd` through `jnp.linalg.solve` in `christoffel_symbols`.** The spray (in `metric.py:45–62`) contains a `jnp.linalg.solve` with a regularised Hessian. `jacfwd(jacfwd(spray, 1), 1)` differentiates *through* `linalg.solve` twice, which is well-defined in JAX but numerically sensitive when the Hessian is near-singular (Randers metrics near the convexity boundary $|\beta|_h \to 1$). No additional regularisation or conditioning check is applied at this level. | Add a comment documenting the conditioning assumption. Consider clamping or monitoring the condition number of the Hessian before differentiation. |
| 5 | **RISK** | `src/ham/geometry/transport.py:30–55` | **No batch-dimension support.** `spec/ARCH_SPEC.md § 1` ("Batch-First") requires all operations to handle a leading `(B, ...)` dimension. `parallel_transport` accepts only single-instance arrays. Users must manually `vmap` over the batch axis. | Add a `vmap`-wrapped public entry point, or document explicitly that callers must `vmap`. |
| 6 | **RISK** | `src/ham/geometry/transport.py:48` | **`to_tangent` projection inside Euler step may accumulate bias.** Re-projecting `new_vec` onto the tangent space at `x` (the *current* point, not the *next* point) after each Euler step introduces a systematic bias on curved manifolds. The vector is projected tangent to $T_{x_i}\mathcal{M}$ but will be used at $x_{i+1}$. A midpoint or endpoint projection would be more consistent. | Project at the *next* path point: `self.metric.manifold.to_tangent(path_x_next, new_vec)`. This requires restructuring the scan to provide the next point. |
| 7 | **RISK** | `src/ham/geometry/transport.py:50–53` | **Wasted computation: scan processes `path_x[-1]` only to discard the result.** The scan iterates over all N inputs, computing Christoffel symbols at the last path point, but `transported_vecs[-1]` is sliced away on line 52. For expensive metrics (neural networks), this is a non-trivial waste. | Slice the scan input to `(path_x[:-1], path_v[:-1])` and run N−1 iterations. Adjust `dt` accordingly. |
| 8 | **STYLE** | `src/ham/geometry/transport.py:63–67` | **`berwald_transport` signature doesn't match `spec/ARCH_SPEC.md § 4.3`.** The spec shows `berwald_transport(metric, path, v0)` (single path array), but the implementation takes `(metric, path_x, path_v, vec_start)` (separate position and velocity arrays). The 4-argument form is arguably better (avoids recomputing velocity from position), but it deviates from the spec. | Update `spec/ARCH_SPEC.md § 4.3` to match the implementation, or add an overload that computes `path_v` via finite differences from a single `path` array. |
| 9 | **STYLE** | `src/ham/geometry/transport.py:1–67` | **Type annotations use deprecated `jnp.ndarray`.** Same issue as `metric.py` (see `reviews/code/metric.md` #7). The canonical type is `jax.Array`. | Replace `jnp.ndarray` with `jax.Array` throughout. |
| 10 | **STYLE** | `src/ham/geometry/transport.py:5` | **`Connection` base class has no `@abstractmethod` decorators.** `christoffel_symbols` and `parallel_transport` raise `NotImplementedError` manually. Using `abc.ABC` + `@abstractmethod` would surface missing overrides at class-definition time rather than at runtime. | Inherit from `ABC` and decorate both methods with `@abstractmethod`. |

---

## Test Coverage Assessment

| Public Function / Method | Tested? | Notes |
|--------------------------|---------|-------|
| `BerwaldConnection.christoffel_symbols` | Indirect | Only exercised internally by `parallel_transport`. No direct test validates the $(D, D, D)$ tensor output. |
| `BerwaldConnection.parallel_transport` | Yes (via `berwald_transport`) | 5 test cases cover Euclidean, Riemannian sphere (norm preservation), Randers (norm drift + velocity dependence), and sphere holonomy. |
| `berwald_transport` | Yes | Thin wrapper; tested through the above. |
| `Connection` (base class) | No | No test instantiates or exercises the base class. |

### Gap Analysis

1. **No direct Christoffel symbol test.** There is no test that computes `BerwaldConnection(metric).christoffel_symbols(x, v)` and checks the output against an analytically known $\Gamma^i_{jk}$. For the Euclidean metric, $\Gamma$ should be identically zero — a trivial but important regression test.
2. **No `jit` / `vmap` / `grad` compatibility tests.** No test wraps `berwald_transport` in `jax.jit`, `jax.vmap`, or `jax.grad`. Since the classes are not `eqx.Module` (Issue #2), `vmap` over the connection would silently fail.
3. **No zero-velocity edge case.** `christoffel_symbols` at $v = 0$ differentiates the spray, which contains `jnp.linalg.solve`. If the Hessian of the energy at $v=0$ is degenerate, the result may be NaN. No test covers this.
4. **No convergence-order test.** There is no test that verifies the integrator converges at the expected rate (order 1 for Euler) as N increases. Such a test would also expose the `dt` bug (Issue #1).
5. **Holonomy test tolerance is very loose.** `test_sphere_holonomy` uses `atol=1e-1`, which is ~6° of angular error. This is likely dominated by the first-order integrator (Issue #3) and the `dt` bug (Issue #1). With both fixed, tighter tolerance should be achievable.

---

## Positive Patterns

1. **`jax.lax.scan` for transport integration.** Using `scan` instead of a Python loop ensures the computation is JIT-friendly and avoids unrolling overhead. This is idiomatic JAX.
2. **Christoffel symbols via double `jacfwd`.** Computing $\Gamma^i_{jk} = \partial^2 G^i / \partial v^j \partial v^k$ via `jacfwd(jacfwd(spray, 1), 1)` is correct and leverages JAX's forward-mode AD efficiently for low-dimensional outputs. No manual tensor algebra is needed.
3. **Tangent-space re-projection.** Calling `manifold.to_tangent` after each step (line 48) prevents drift off the tangent bundle on constrained manifolds. This is a necessary stabilisation step.
4. **Clean wrapper function.** `berwald_transport` provides a flat functional API while keeping the OO structure internal. The public API is simple and discoverable.
5. **Test design covers Finsler-specific physics.** The `test_randers_norm_drift` and `test_randers_velocity_dependence` tests verify genuinely Finslerian behaviour (not just Riemannian), which is central to the library's purpose.
