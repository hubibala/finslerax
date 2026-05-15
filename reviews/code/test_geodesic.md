# Code Review: tests/test_geodesic.py

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0

## Summary

The file provides four tests covering the `ExponentialMap` IVP solver: Euclidean ballistic motion, sphere great-circle shooting, Randers energy conservation, and manifold adherence. The test suite has **one clear bug** (wrong `Sphere` constructor argument that silently creates a circle instead of a 2-sphere), relies on `unittest.TestCase` rather than pytest idioms, contains debug `print` statements, and is missing tests for several important scenarios—batch-dimension handling, JIT/vmap compatibility, zero-velocity edge cases, and gradient flow through the solver.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `tests/test_geodesic.py:60` | `Sphere(1.0)` passes `1.0` as `intrinsic_dim` (first positional arg), creating S¹ (circle in ℝ²) instead of S² (sphere in ℝ³). The test then uses 3-D points and a 3×3 identity Riemannian tensor, so the manifold's `ambient_dim`/`intrinsic_dim` properties are wrong. The test currently passes only because `project`/`to_tangent` operate on array shapes rather than inspecting the stored dimension, masking the mismatch. | Change to `Sphere(radius=1.0)` or `Sphere(intrinsic_dim=2, radius=1.0)`. |
| 2 | **BUG** | `tests/test_geodesic.py:82` | Same issue: `Sphere(1.0)` in `test_manifold_adherence` creates S¹, not S². 3-D vectors are then used with a 2-D manifold. | Change to `Sphere(radius=1.0)`. |
| 3 | **RISK** | `tests/test_geodesic.py:28–31` | `Plane` is defined as an ad-hoc class with no base class. It satisfies the duck-typed `Manifold` interface (`project`, `to_tangent`) but doesn't inherit from `Manifold`, which means it lacks `ambient_dim`, `intrinsic_dim`, and `random_sample`. If the solver or metric ever calls those properties, the test will crash with `AttributeError`. | Use `EuclideanSpace(dim=2)` from `ham.geometry.surfaces` instead. |
| 4 | **RISK** | `tests/test_geodesic.py:72` | Energy-conservation threshold (`1e-4`) is absolute, not relative. For a different initial energy scale (e.g. very large `v0`), this threshold could be too tight or too loose. The comment "likely < 1e-5" suggests guesswork, not a principled bound. | Assert `max_deviation / e_start < rtol` using a relative tolerance, e.g. `rtol=1e-4`. |
| 5 | **RISK** | `tests/test_geodesic.py` (global) | No test verifies the solver under `jax.jit` or `jax.vmap`. The ARCH_SPEC mandates JIT and batch-first compatibility; the solver uses `jax.lax.fori_loop` and `jax.lax.scan`, which should be JIT-safe, but this is never tested. A regression in, e.g., the `dynamics` closure capturing a Python-side mutable could break JIT silently. | Add a test wrapping `solver.shoot` with `jax.jit` and verifying identical output. Add a `vmap`-over-batch test. |
| 6 | **RISK** | `tests/test_geodesic.py` (global) | No edge-case tests: zero initial velocity (`v0 = 0`), antipodal points on the sphere, or very large velocities that trigger the speed clamp (`curr_v_norm > 10.0`) in `_step_rk4`. These are the most likely failure modes. | Add parametrized edge-case tests for zero velocity (should return start point), antipodal shooting, and large-`v0` clamping. |
| 7 | **STYLE** | `tests/test_geodesic.py:1` | Uses `unittest.TestCase` instead of plain pytest functions/classes. The project's CI uses pytest (implied by the `pyproject.toml` and other test files), so mixing `unittest` adds friction: no native `@pytest.mark.parametrize`, `@pytest.fixture`, or `pytest.approx`. | Convert to pytest-style: plain functions with `assert`, `@pytest.fixture` for the solver, `@pytest.mark.parametrize` for metric/manifold combinations. |
| 8 | **STYLE** | `tests/test_geodesic.py:47,48,67,68,87` | Debug `print` statements in test bodies. These pollute test output and are not consumed by any assertion. | Remove `print` calls or replace them with `logging.debug`. |
| 9 | **STYLE** | `tests/test_geodesic.py` (global) | No batch-dimension test. ARCH_SPEC §1 mandates batch-first `(B, ...)` convention. The solver's `shoot` and `trace` are tested only for single `(D,)` inputs. | Add a test that `vmap`s `solver.shoot` over a batch of initial conditions and checks shape/consistency. |
| 10 | **STYLE** | `tests/test_geodesic.py:36` | `np.testing.assert_allclose` used alongside `self.assertLess`; mixing NumPy and unittest assertion styles. Consistency would improve readability. | Standardize on one style (preferably `np.testing.assert_allclose` for numerical checks, or `pytest.approx` if converting to pytest). |

## Test Coverage Assessment

| Public API | Tested? | Notes |
|---|---|---|
| `ExponentialMap.__init__` | ✅ | Via `setUp` |
| `ExponentialMap.shoot` | ✅ | `test_euclidean_ballistic`, `test_sphere_great_circle` |
| `ExponentialMap.trace` | ✅ | `test_energy_conservation_randers`, `test_manifold_adherence` |
| `ExponentialMap._step_rk4` | ⚠️ Indirect | Never unit-tested in isolation; only through `shoot`/`trace`. |

### Gaps

- **No gradient test:** The solver should be differentiable (required by the AVBD and training pipeline). No test calls `jax.grad` through `shoot` or `trace` to verify gradients flow.
- **No JIT test:** JIT compilation of the solver is untested.
- **No vmap / batch test:** Batch-first convention is untested.
- **No zero-velocity edge case:** `v0 = 0` should return the start point; untested.
- **No high-curvature / antipodal test:** Shooting to the antipode on a sphere is the hardest case for RK4 with projection; untested.
- **No velocity-clamp test:** The `_step_rk4` method hard-clips velocity at norm 10 and acceleration at norm 100. No test verifies behavior near or above these thresholds.
- **No `t_max != 1.0` test:** Both `shoot` and `trace` accept `t_max` but it is always called with the default `1.0`.

## Positive Patterns

1. **x64 precision enabled** (`config.update("jax_enable_x64", True)`) — essential for sensitive ODE comparisons.
2. **Energy-conservation test** (`test_energy_conservation_randers`) — correctly checks physics invariants, not just endpoint accuracy.
3. **Manifold-adherence test** — verifies the projection step keeps points on the constraint surface.
4. **Solver parameterization** in `setUp` uses 1000 steps with small `dt`, appropriate for tight ODE tolerances.
