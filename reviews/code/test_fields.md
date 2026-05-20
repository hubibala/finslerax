# Code Review: `tests/test_fields.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`test_fields.py` provides basic correctness tests for four of the six public functions in `ham.sim.fields`. The tests verify physically meaningful properties (far-field decay, solid-body rotation, tangentiality) and the assertions are numerically correct. However, the file has several structural weaknesses: two public functions are imported but never tested, the assertion style (`self.assertTrue(jnp.allclose(...))`) produces useless failure messages, `jax_enable_x64` is not enabled (unlike peer test files), and there are no JAX-transform compatibility tests (`jit`, `vmap`, `grad`). These gaps reduce confidence that the fields module will survive refactoring or edge-case inputs.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `tests/test_fields.py:6–7` | `rossby_haurwitz` and `harmonic_vortices` are imported but **never tested**. This is not merely a coverage gap — it masks potential import-time or runtime errors in those two functions. If either function were broken, the test suite would still pass green. | Add at least smoke tests: call each with default params, assert output is finite and tangent to the sphere. |
| 2 | **RISK** | `tests/test_fields.py:24` | `self.assertTrue(jnp.allclose(v, jnp.array([0.0, 1.0, 0.0])))` — on failure this prints only `AssertionError: False` with no indication of actual vs expected values. This makes debugging test failures extremely difficult. The same anti-pattern appears on line 30. | Replace with `np.testing.assert_allclose(v, expected, atol=1e-6)`, which is used consistently in peer test files (`test_surfaces.py`, `test_transport.py`). |
| 3 | **RISK** | `tests/test_fields.py:1–60` | `jax_enable_x64` is **not** enabled. Peer test files (`test_transport.py`, `test_surfaces.py`) explicitly set `config.update("jax_enable_x64", True)` for geometric precision. Under f32 the stream-function gradient (`jax.grad(psi)`) and cross-product can accumulate ~$10^{-7}$ errors, which may cause `jnp.allclose` (default `atol=1e-8`) to intermittently fail depending on platform. | Add `from jax import config; config.update("jax_enable_x64", True)` at module level, consistent with the rest of the test suite. |
| 4 | **RISK** | `tests/test_fields.py:24,30` | `jnp.allclose` uses default tolerances (`rtol=1e-5`, `atol=1e-8`). These are implicitly coupling the test to a precision level that is neither documented nor consistent with the `places=5` / `places=1` tolerances used elsewhere in the same file. A single tolerance convention should be chosen. | Use explicit `atol`/`rtol` kwargs in all comparisons, or standardize on `np.testing.assert_allclose` with stated tolerances throughout. |
| 5 | **RISK** | `tests/test_fields.py:14–25` | `test_stream_function_flow` hardcodes a single test point `[1, 0, 0]` and a single stream function `ψ = z`. This exercises only one code path of `get_stream_function_flow`. A non-trivial `ψ` (e.g., `ψ = x·y`) or an off-axis test point would catch more bugs (e.g., incorrect cross-product argument order). | Add a second sub-case with a different `psi` or a point not aligned with a coordinate axis. |
| 6 | **RISK** | `tests/test_fields.py:27–31` | `test_tilted_rotation` only tests the degenerate case `alpha_deg=0.0`, which reduces `tilted_rotation` to pure z-axis rotation — identical to the stream-function test above. The normalization on `fields.py:32` and the full tilt logic are never exercised. | Add a parametrized case with `alpha_deg=90.0` (rotation around x-axis) where the expected output is analytically known: at `[0, 0, 1]`, `v = [0, -1, 0]`. |
| 7 | **STYLE** | `tests/test_fields.py:1` | Uses `unittest.TestCase` — consistent with the rest of the HAM test suite, so this is fine. However, `numpy` is not imported even though `np.testing.assert_allclose` is the preferred assertion in sibling files. | Add `import numpy as np` and switch to `np.testing.assert_allclose`. |
| 8 | **STYLE** | `tests/test_fields.py:14–60` | No `setUp` method is used. While the current tests are stateless, adding a `setUp` with a fixed `jax.random.PRNGKey` and reusable test points (equator, pole, arbitrary) would reduce duplication and enable future tests to share fixtures. Peer file `test_surfaces.py` demonstrates this pattern. | Add `def setUp(self): self.key = jax.random.PRNGKey(42)` and shared test points. |
| 9 | **STYLE** | `tests/test_fields.py:14–60` | No tests verify JAX-transform compatibility (`jit`, `vmap`, `grad`) for any field function. Since fields are closures returned by factory functions, verifying that `jax.jit(flow_fn)(x)` matches the unjitted call is a basic correctness check that guards against accidental Python side-effects in the closures. | Add at least one `jit` round-trip test: `assert_allclose(jax.jit(flow_fn)(x), flow_fn(x))`. |
| 10 | **STYLE** | `tests/test_fields.py:33–60` | The 3D-extension branches of `lamb_oseen_vortex` and `rankine_vortex` (`if x.shape[0] > 2` in `fields.py:108,132`) are never tested. All test inputs are 2D. | Add a test case with a 3D input `[10.0, 0.0, 5.0]` and verify the z-component of the output is zero. |

---

## Test Coverage Assessment

| Public Function | Tested? | Test Method | Gap |
|---|---|---|---|
| `get_stream_function_flow` | **Yes** | `test_stream_function_flow` | Single test point, single ψ |
| `tilted_rotation` | **Partial** | `test_tilted_rotation` | Only degenerate `alpha=0°`; non-trivial tilt untested |
| `rossby_haurwitz` | **No** | — | Imported but entirely untested |
| `harmonic_vortices` | **No** | — | Imported but entirely untested |
| `lamb_oseen_vortex` | **Yes** | `test_lamb_oseen_vortex_2d` | Far-field + near-origin; 3D branch untested |
| `rankine_vortex` | **Yes** | `test_rankine_vortex_2d` | Inside/outside core; 3D branch untested |

**Cross-cutting gaps:**

- No JAX-transform tests (`jit`, `vmap`, `grad`) for any function.
- No batch-dimension tests (consistent with source not supporting batch, but a `vmap`-wrapped test would verify compatibility per `spec/ARCH_SPEC.md` § 1).
- No edge-case tests at sphere poles (`[0, 0, ±1]`) for the 3D stream-function-based fields, where `cos(lat) → 0` and the complex-number azimuth trick encounters a near-zero denominator.

---

## Positive Patterns

1. **Physically grounded assertions**: Tests verify analytically known properties (far-field point-vortex decay, solid-body rotation inside Rankine core, zero velocity at Lamb-Oseen center) rather than snapshot-comparing opaque arrays. This makes the tests meaningful and self-documenting.
2. **Boundary region coverage for Rankine vortex**: Testing both inside (`r < r_c`) and outside (`r > r_c`) the core exercises the `jnp.where` branch in the implementation, which is the most error-prone part of that function.
3. **Correct expected values**: All hardcoded expected values (`0.5`, `0.1`, `[0, 1, 0]`) are analytically correct and match the formulas in the source file docstrings.
