# Code Review: `test_hyperboloid.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)  
**Source Under Test:** `src/ham/geometry/surfaces.py` (`Hyperboloid` class)  
**Test File:** `tests/test_hyperboloid.py`

## Summary

The test file provides basic coverage of `Hyperboloid`'s `project`, `to_tangent`, `random_sample`, `retract`, and `metric_tensor` methods. The existing tests are correct and clearly written. However, three core public methods — `exp_map`, `log_map`, and `parallel_transport` — have **zero test coverage**. There are no JAX transform compatibility tests (`jit`, `vmap`, `grad`), no edge-case tests, and no roundtrip consistency checks. The test helper `minkowski_dot` uses unbatched indexing, and several tests verify only a single hard-coded point. Overall the file validates happy-path behavior but leaves significant gaps that could mask regressions.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | [test_hyperboloid.py](tests/test_hyperboloid.py#L1-L120) | `Hyperboloid.exp_map` has **no test**. This is a core method used by `retract` and ODE solvers. Bugs in the Taylor-branch switching (`sinh(θ)/θ` near zero) or the `cosh`/`sinh` computation would go undetected. | Add tests: (a) `exp_map` from the origin with a known tangent vector and verify the result lies on the hyperboloid; (b) verify the zero-tangent-vector case returns the base point; (c) verify against a closed-form known geodesic endpoint. |
| 2 | **RISK** | [test_hyperboloid.py](tests/test_hyperboloid.py#L1-L120) | `Hyperboloid.log_map` has **no test**. This is the inverse of `exp_map`; without testing it, the `exp_map`/`log_map` roundtrip invariant ($\text{log}_x(\text{exp}_x(v)) = v$) is unverified. | Add tests: (a) `log_map(x, y)` produces a tangent vector at `x` (Minkowski-orthogonal to `x`); (b) `exp_map(x, log_map(x, y))` recovers `y`; (c) `log_map(x, x)` returns the zero vector. |
| 3 | **RISK** | [test_hyperboloid.py](tests/test_hyperboloid.py#L1-L120) | `Hyperboloid.parallel_transport` has **no test**. The source implementation (lines 403–412 of `surfaces.py`) has a potentially fragile denominator (`jnp.maximum(1.0 - xy, 2.0)`, flagged in `reviews/code/surfaces.md` issue #11). Without a test, regressions in the transport formula are invisible. | Add tests: (a) transporting a tangent vector from `x` to `y` yields a vector tangent to `y`; (b) transport preserves Minkowski norm; (c) transport from `x` to `x` (identity case) returns the original vector. |
| 4 | **RISK** | [test_hyperboloid.py](tests/test_hyperboloid.py#L1-L120) | No `exp_map`/`log_map` **roundtrip consistency** test. This is the primary correctness check for exponential/logarithmic map pairs on any manifold. | Add `test_exp_log_roundtrip`: pick several `(x, v)` pairs, compute `y = exp_map(x, v)`, then verify `log_map(x, y) ≈ v` and `exp_map(x, log_map(x, y)) ≈ y`. |
| 5 | **RISK** | [test_hyperboloid.py](tests/test_hyperboloid.py#L1-L120) | No **JAX transform compatibility** tests. ARCH_SPEC § 1 requires all operations to be `jit`/`vmap`/`grad`-compatible. The test file never wraps any call in `jax.jit`, `jax.vmap`, or `jax.grad`. A tracing bug (e.g., Python-side `if` over a traced value) could go undetected. | Add: (a) `test_jit_project`: `jax.jit(self.manifold.project)(x)` matches eager result; (b) `test_vmap_to_tangent`: `jax.vmap(self.manifold.to_tangent)(xs, vs)` works on batched inputs; (c) `test_grad_through_exp_map`: `jax.grad(lambda v: jnp.sum(self.manifold.exp_map(x, v)))(v)` does not raise. |
| 6 | **RISK** | [test_hyperboloid.py](tests/test_hyperboloid.py#L103-L111) | `test_retraction_stays_on_manifold` tests only **one base point** (origin `[1,0,0]`) and **one tangent vector** (`[0, 0.5, 0.5]`). Does not exercise the `max_norm=10.0` clamping branch in `retract` (source line 391), nor test a non-origin base point. | Add assertions for: (a) a non-origin base point (e.g., `[cosh(2), sinh(2), 0]`); (b) a large tangent vector that triggers the norm clamp; (c) a zero tangent vector (should return base point). |
| 7 | **RISK** | [test_hyperboloid.py](tests/test_hyperboloid.py#L67-L78) | `test_tangent_space_orthogonality` tests only **one point and one ambient vector**. Does not verify that `to_tangent` preserves already-tangent vectors (idempotence), nor that it works at the origin `[1,0,0]`. | Test multiple points. Add an idempotence check: `to_tangent(x, to_tangent(x, v)) ≈ to_tangent(x, v)`. |
| 8 | **RISK** | [test_hyperboloid.py](tests/test_hyperboloid.py#L1-L120) | No test for **higher intrinsic dimensions**. All tests use `intrinsic_dim=2`. The source code's `[..., 0]` / `[..., 1:]` indexing is dimension-agnostic, but this is never verified for $d > 2$. | Add a parameterized or separate test with `intrinsic_dim=5` (or similar) that repeats key checks (project, to_tangent, random_sample). |
| 9 | **STYLE** | [test_hyperboloid.py](tests/test_hyperboloid.py#L21-L23) | `minkowski_dot` helper uses **unbatched indexing** (`u[0]`, `u[1:]`) while the source's `_minkowski_dot` uses `u[..., 0]`, `u[..., 1:]`. The helper cannot verify batched outputs and is inconsistent with the source convention. | Use `u[..., 0]` and `u[..., 1:]` with `axis=-1` reduction, matching the source implementation. |
| 10 | **STYLE** | [test_hyperboloid.py](tests/test_hyperboloid.py#L44-L50), [test_hyperboloid.py](tests/test_hyperboloid.py#L87-L91) | `test_projection_constraints` and `test_random_sampling` use **Python `for` loops** to iterate over samples and call `assertAlmostEqual` per element. This is verbose, slow, and produces poor failure messages (only shows the failing index implicitly). | Use vectorized `np.testing.assert_allclose` on the entire batch: `norm_sq = vmap(self.minkowski_dot)(samples, samples)` then `np.testing.assert_allclose(norm_sq, -1.0, atol=1e-6)`. |
| 11 | **STYLE** | [test_hyperboloid.py](tests/test_hyperboloid.py#L44), [test_hyperboloid.py](tests/test_hyperboloid.py#L49), [test_hyperboloid.py](tests/test_hyperboloid.py#L78), [test_hyperboloid.py](tests/test_hyperboloid.py#L109) | `self.assertAlmostEqual` and `self.assertGreater` are called on **JAX array scalars**. These work via implicit Python conversion but are fragile: they bypass JAX's type system and would fail inside a traced (`jit`) context. `np.testing.assert_allclose` is the idiomatic pattern used elsewhere in the test suite (e.g., `test_projection_idempotence`). | Replace `self.assertAlmostEqual(jax_val, target, places=N)` with `np.testing.assert_allclose(jax_val, target, atol=tolerance)` consistently. Replace `self.assertGreater(p[0], 0.0)` with `assert float(p[0]) > 0.0` or `self.assertTrue(p[0] > 0.0, msg=...)`. |
| 12 | **STYLE** | [test_hyperboloid.py](tests/test_hyperboloid.py#L55-L62) | `test_projection_idempotence` only tests the **origin** (`[1,0,0]`). A single canonical point doesn't exercise the two branches in `Hyperboloid.project` (the `is_valid_candidate` path vs. the spatial-lift path). | Add at least one non-trivial point that enters each branch, e.g., `[cosh(1), sinh(1), 0]` (valid → scaling branch) and `[0.5, 0.3, 0.1]` (invalid → lift branch). |
| 13 | **STYLE** | [test_hyperboloid.py](tests/test_hyperboloid.py#L93-L100) | `test_metric_tensor_signature` does not verify that `metric_tensor` returns the **same value for different points**. Since the implementation is point-independent (returns a constant Minkowski matrix), a second assertion at a different point would catch accidental point-dependent behavior. | Add `g2 = self.manifold.metric_tensor(jnp.array([cosh(1), sinh(1), 0]))` and `np.testing.assert_allclose(g, g2)`. |

---

## Test Coverage Assessment

| Public Method | Tested? | Notes |
|---|---|---|
| `Hyperboloid.__init__` | Yes | Via `setUp` |
| `Hyperboloid.ambient_dim` | Yes | `test_dimensions` |
| `Hyperboloid.intrinsic_dim` | Yes | `test_dimensions` |
| `Hyperboloid.project` | Yes | Constraints + idempotence (origin only) |
| `Hyperboloid.to_tangent` | Yes | Single point/vector orthogonality check |
| `Hyperboloid.exp_map` | **MISSING** | Not tested at all |
| `Hyperboloid.log_map` | **MISSING** | Not tested at all |
| `Hyperboloid.parallel_transport` | **MISSING** | Not tested at all |
| `Hyperboloid.retract` | Yes | Single point, no edge cases |
| `Hyperboloid.random_sample` | Yes | 100 samples, constraint check |
| `Hyperboloid.metric_tensor` | Yes | Signature check at origin |
| `Hyperboloid._minkowski_dot` | Indirect | Via test helper (unbatched) |
| `Hyperboloid._minkowski_norm` | Indirect | Via `retract` → `exp_map` |

**Cross-cutting gaps:**
- No `jax.jit` compatibility test for any method.
- No `jax.vmap` compatibility test for any method.
- No `jax.grad` / `jax.jacfwd` test through any differentiable method.
- No edge-case tests: zero tangent vector, very large tangent vector, coincident points.
- No exp/log roundtrip consistency test.
- No higher-dimensional test (`intrinsic_dim > 2`).

---

## Positive Patterns

1. **64-bit precision enabled globally**: `config.update("jax_enable_x64", True)` ensures geometric identity checks (e.g., $\langle p, p \rangle_L = -1$) are accurate enough for `places=6` assertions.
2. **Clean test naming and docstrings**: Every test method has a descriptive name and a docstring explaining the invariant being checked. This makes the test file readable and self-documenting.
3. **Correct invariant selection**: The tests check the right mathematical properties — hyperboloid equation, upper-sheet constraint, Minkowski orthogonality, projection idempotence. These are the fundamental geometric invariants.
4. **Independent test helper**: The `minkowski_dot` helper is defined on the test class rather than importing the private `_minkowski_dot` from the source. This ensures the test uses an independent reference implementation, which is good test practice.
5. **Deterministic randomness**: `jax.random.PRNGKey(1337)` in `setUp` ensures reproducible test runs.
