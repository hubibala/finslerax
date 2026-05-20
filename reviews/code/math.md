# Code Review: `ham.utils.math`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)  
**Source:** `src/ham/utils/math.py` (24 lines)

## Summary

The module is small, well-focused, and correctly implements the gradient-safe norm pattern that the architecture spec mandates (`spec/ARCH_SPEC.md § 5`). The `safe_norm` function is JIT/vmap/grad-compatible by construction. No bugs were found. The main findings are (a) the codebase has significant adoption gaps — many modules use ad-hoc `jnp.linalg.norm + 1e-10` or inline `jnp.sqrt(jnp.maximum(...))` instead of the canonical `safe_norm`, and (b) there is no dedicated test file for this utility module; coverage is only indirect.

## Issue Tracker

| # | Severity | Location (file:line) | Description | Suggested Fix |
|---|----------|----------------------|-------------|---------------|
| 1 | **OK** | `src/ham/utils/math.py:16-24` | `safe_norm` implementation is correct. Uses `jnp.maximum(sq, eps)` which keeps the argument to `jnp.sqrt` strictly positive, producing finite gradients at $x = 0$. Fully compatible with `jit`, `vmap`, and `grad` — no Python side-effects, no data-dependent control flow. | — |
| 2 | **OK** | `src/ham/utils/math.py:7-10` | Canonical epsilon constants are well-documented and cover distinct stability regimes (`GRAD_EPS` for derivatives, `NORM_EPS` for comparisons, `PSD_EPS` for regularisation, `TAYLOR_EPS` for series switching). | — |
| 3 | **OK** | `src/ham/utils/__init__.py:1` | All public symbols (`safe_norm`, four constants) are re-exported through the package `__init__.py` and listed in `__all__`. | — |
| 4 | **WARNING** | `src/ham/utils/math.py:1` | `import jax` is imported but never used directly in this module. Only `jax.numpy` is used. | Remove `import jax` or keep it if future additions are planned (low priority). |
| 5 | **RISK** | `src/ham/utils/math.py:16` | `safe_norm` forward value at $x = 0$ returns $\sqrt{\varepsilon} \approx 10^{-6}$, not $0$. Callers that rely on the norm being exactly zero (e.g., zero-vector detection) will get a small positive number. This is the correct trade-off for gradient safety, but it should be documented explicitly so downstream code uses `NORM_EPS`-based threshold checks (as `surfaces.py:45` does) rather than equality to zero. | Add a note in the docstring: "Returns $\sqrt{\varepsilon}$ (not 0) when $x = 0$." |
| 6 | **RISK** | Multiple files | The codebase has **at least 10 call sites** that bypass the canonical `safe_norm` and use ad-hoc patterns with hardcoded epsilons, violating the consolidation goal stated in `src/ham/utils/math.py:5`. Examples: `sim/fields.py:35` (`jnp.linalg.norm(axis + 1e-10)`), `sim/fields.py:123`, `geometry/mesh.py:93,95`, `solvers/geodesic.py:32,40`, `geometry/curvature.py:97,102`, `geometry/zoo.py:36,97,130,150,158`, `geometry/surfaces.py:387`. Several of these use `jnp.linalg.norm` which has NaN gradients at zero. | Migrate all ad-hoc norm computations to `safe_norm` (or document exceptions where a different epsilon is intentionally needed). This is a latent gradient-correctness risk, particularly in `solvers/geodesic.py` and `geometry/curvature.py` which are differentiated through. |
| 7 | **RISK** | `src/ham/utils/math.py:16` | No `__all__` in the module itself. If someone does `from ham.utils.math import *`, all top-level names (including `jax`, `jnp`) leak into the caller's namespace. | Add `__all__ = ["safe_norm", "GRAD_EPS", "NORM_EPS", "PSD_EPS", "TAYLOR_EPS"]`. |
| 8 | **INFO** | `src/ham/utils/math.py:16` | `safe_norm` does not provide a `dtype` parameter. In mixed-precision contexts (float16 inputs), the squaring could overflow before the `maximum` clamp. Not a concern at present (the codebase appears to use float32/float64 exclusively). | Consider adding a `jnp.astype(jnp.float32)` guard if float16 support is ever needed. |

## Test Coverage Assessment

| Public Symbol | Directly Tested? | Indirect Usage in Tests | Gap |
|---------------|-------------------|-------------------------|-----|
| `safe_norm` | **No** — no dedicated unit test | Used in `test_geodesic.py:99`, `test_geodesic_learning.py:217-218`, `test_solver.py:34` as a helper, but these tests do not verify `safe_norm` behavior itself. | **No test for the zero-input case** — the most important property (finite gradients at $x=0$) is never explicitly checked. No test verifies the forward value at $x=0$ equals $\sqrt{\varepsilon}$. |
| `GRAD_EPS` | No | Implicitly tested via `safe_norm` default parameter | — |
| `NORM_EPS` | No | Used in `surfaces.py` but those tests don't target the constant | — |
| `PSD_EPS` | No | Not found in any test file | Unused in tests; unclear if any source module actually imports it. |
| `TAYLOR_EPS` | No | Used in `surfaces.py` but those tests don't target the constant | — |

**Recommended test additions:**
1. `test_safe_norm_forward`: verify `safe_norm(jnp.zeros(3))` ≈ `sqrt(GRAD_EPS)`.
2. `test_safe_norm_grad_at_zero`: verify `jax.grad(lambda x: safe_norm(x))(jnp.zeros(3))` is finite (no NaN).
3. `test_safe_norm_matches_linalg`: for non-zero inputs, verify `safe_norm(x)` ≈ `jnp.linalg.norm(x)`.
4. `test_safe_norm_vmap`: verify `jax.vmap(safe_norm)(batch)` works for a batch of vectors including a zero vector.
5. `test_safe_norm_jit`: verify `jax.jit(safe_norm)(x)` matches eager result.

## Positive Patterns

1. **Single canonical implementation** — the design intent to have one `safe_norm` used everywhere is architecturally sound and matches `spec/ARCH_SPEC.md § 5`.
2. **Tiered epsilon constants** — separating `GRAD_EPS`, `NORM_EPS`, `PSD_EPS`, and `TAYLOR_EPS` prevents accidental reuse of a single epsilon for different stability purposes. This is a well-thought-out practice.
3. **Pure functional style** — the module has no mutable state, no classes, no side effects. Fully compatible with JAX transformations.
4. **Clear docstring** — the `safe_norm` docstring explains both the pattern and its purpose, and explicitly claims canonical status.
