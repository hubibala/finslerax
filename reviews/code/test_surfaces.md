# Code Review: tests/test_surfaces.py

**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary

The test file covers basic happy-path behaviour for `Sphere`, `Torus`, `Paraboloid`, and `EuclideanSpace`, but coverage is thin. Many public methods on every surface go untested, batch-dimension semantics (an ARCH_SPEC core principle) are never verified, and JAX transform compatibility (`jit`/`vmap`/`grad`) is not exercised. A PRNGKey-reuse bug produces correlated random samples, and assertion style is inconsistent throughout.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `tests/test_surfaces.py:52` | `self.key` is reused without splitting: the same PRNGKey seeds both `random_sample` (line 49) and `jax.random.normal` (line 53). JAX PRNGKeys must be split before reuse; reusing the same key produces correlated samples, undermining the independence of `x` and `v`. | Split the key: `k1, k2 = jax.random.split(self.key)` and use `k1` for `random_sample` and `k2` for `jax.random.normal`. |
| 2 | **RISK** | `tests/test_surfaces.py:15–93` | No tests verify behaviour under JAX transforms (`jax.jit`, `jax.vmap`, `jax.grad`). A surface method that accidentally captures a Python side-effect would pass these tests but break in production. | Add at least one test per surface that wraps a round-trip operation (e.g., `project`, `exp_map ∘ log_map`) in `jax.jit` and confirms the result matches the un-jitted version. |
| 3 | **RISK** | `tests/test_surfaces.py:15–93` | No test exercises a leading batch dimension. ARCH_SPEC §1 mandates batch-first `(B, ...)` semantics, yet every test passes single unbatched points. `Torus.project` (line 65) wraps with `jax.vmap` to handle a batch — this hints that calling `torus.project(pts)` without vmap may fail, but that inconsistency is never caught. | Add a test that passes batched inputs `(B, D)` directly to each method and confirms correct output shape and values. |
| 4 | **RISK** | `tests/test_surfaces.py:40–41` | The `log_map` round-trip test (`log(x, exp(x, v)) ≈ v`) only uses a single tangent vector. This does not probe edge cases: zero tangent vector (should return origin), near-antipodal points (large `v`), or co-located points (`y ≈ x`). Any of these can trigger division-by-zero in the Taylor branches of `Sphere.log_map`. | Add parametric sub-tests with `v = 0`, `‖v‖ ≈ π·r` (near-antipodal), and `‖v‖ ≈ 1e-10` (near-zero). |
| 5 | **RISK** | `tests/test_surfaces.py:62–66` | `test_torus_operations` only tests idempotence of `project` on sampled points. It does not test `to_tangent`, `exp_map`, `retract`, or `random_sample` distribution shape. A regression in any of those methods would not be detected. | Add sub-tests for `to_tangent` (verify orthogonality to the surface normal), `exp_map` round-trip, and output shape of `random_sample`. |
| 6 | **RISK** | `tests/test_surfaces.py:68–73` | `test_paraboloid_operations` tests only `project` and `random_sample`. `to_tangent`, `exp_map`, and `retract` are untested. | Add a tangent-projection test (normal component of `to_tangent` output should be zero) and an `exp_map`/`retract` round-trip test. |
| 7 | **RISK** | `tests/test_surfaces.py:75–82` | `test_euclidean_space` omits `to_tangent` (should be identity), `retract` (should equal `exp_map`), `parallel_transport` (should be identity), and `random_sample`. | Add assertions for the missing methods. |
| 8 | **STYLE** | `tests/test_surfaces.py:27,33,36` | Mixed assertion APIs: `self.assertAlmostEqual(..., places=5)` on lines 27/33/36, `self.assertAlmostEqual(..., delta=1e-5)` on lines 54/57, and `np.testing.assert_allclose(..., atol=1e-5)` on lines 23/41/51. The `places` and `delta` semantics differ (`places=5` means `|a-b| < 5e-6`, while `delta=1e-5` means `|a-b| < 1e-5`). | Standardise on `np.testing.assert_allclose` with an explicit `atol` everywhere for consistency and clearer error messages on failure. |
| 9 | **STYLE** | `tests/test_surfaces.py:16–41` | `test_sphere_operations` is a monolithic test that checks five distinct properties (dimensions, sampling, projection, retraction, log_map). A failure in the projection check hides whether retraction also fails. | Split into focused test methods: `test_sphere_dimensions`, `test_sphere_random_sample`, `test_sphere_project`, `test_sphere_exp_log_roundtrip`. |
| 10 | **STYLE** | `tests/test_surfaces.py:15–82` | All surface tests live in a single `TestSurfaces` class. Splitting into `TestSphere`, `TestTorus`, `TestParaboloid`, `TestEuclideanSpace` would improve discoverability and allow per-surface `setUp`. | Refactor into one `TestCase` subclass per surface. |
| 11 | **STYLE** | `tests/test_surfaces.py:27,33,36,54,57` | `self.assertAlmostEqual` is called on JAX `DeviceArray` scalars without `float()` wrapping (lines 27/33/36). Lines 54/57 do call `float()`. This is inconsistent and may produce confusing assertion messages with non-standard types. | Wrap all JAX scalars in `float()` or switch to `np.testing.assert_allclose`. |
| 12 | **STYLE** | `tests/test_surfaces.py:62–66` | `test_torus_operations` manually wraps `torus.project` in `jax.vmap`. If the source module's `project` is supposed to handle batches natively (per ARCH_SPEC batch-first), this vmap is a workaround rather than a real test of the API contract. | Test the raw `torus.project(pts)` call without vmap; if it fails, file a bug against the source, not a workaround in the test. |

## Test Coverage Assessment

| Public API (surfaces.py) | Sphere | Torus | Paraboloid | EuclideanSpace |
|---------------------------|--------|-------|------------|----------------|
| `ambient_dim` / `intrinsic_dim` | ✅ | ❌ | ❌ | ❌ |
| `project` | ✅ | ✅ (idempotence only) | ✅ | ✅ |
| `to_tangent` | ✅ | ❌ | ❌ | ❌ |
| `exp_map` | ✅ (via `retract`) | ❌ | ❌ | ✅ |
| `retract` | ✅ | ❌ | ❌ | ❌ |
| `log_map` | ✅ (single case) | ❌ | ❌ | ✅ |
| `parallel_transport` | ❌ | N/A | N/A | ❌ |
| `random_sample` | ✅ | ✅ (implicit) | ✅ | ❌ |
| Edge cases (zero vec, antipodal) | ❌ | ❌ | ❌ | ❌ |
| Batch `(B, D)` inputs | ❌ | ❌ | ❌ | ❌ |
| JIT compatibility | ❌ | ❌ | ❌ | ❌ |

**Gap analysis:** Coverage is concentrated on `Sphere`; the other three surfaces have minimal testing. No edge cases, no batch tests, and no JAX-transform tests exist for any surface.

## Positive Patterns

- **x64 mode enabled** (line 6): `jax_enable_x64` is set at module scope, ensuring numerical precision is consistent across environments.
- **Round-trip test for Sphere** (lines 36–41): Testing `log(x, exp(x, v)) ≈ v` is the correct pattern for verifying exp/log map consistency.
- **High-dimensional sphere test** (lines 43–57): Testing `S^6` in addition to `S^2` is good practice and catches dimension-dependent bugs.
- **`np.testing.assert_allclose`** used in several places provides informative mismatch diagnostics on failure.
