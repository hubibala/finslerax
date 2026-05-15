# Code Review: tests/test_solver.py
**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary

`tests/test_solver.py` provides three integration tests for `AVBDSolver` across Torus, Sphere (Randers), and Paraboloid manifolds. The tests verify topological correctness rather than numerical precision, which is appropriate for a BVP solver. However, the file has significant coverage gaps: no boundary-condition verification, no JIT/vmap/grad compatibility tests (critical per `spec/ARCH_SPEC.md` § 1 Batch-First and § 4.2 differentiability guarantees), no `train_mode=False` path, and no edge-case tests. Several assertions lack diagnostic messages, and `print` statements are used in lieu of structured output.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | `tests/test_solver.py:17` | Solver is initialised with `iterations=200`. This is extremely high for a unit test and makes the suite slow. The default in `src/ham/solvers/avbd.py` is 20 iterations. | Reduce to the minimum iterations required for convergence (e.g. 30–50) or add a `@unittest.skipIf` guard for CI speed. |
| 2 | **RISK** | `tests/test_solver.py:35` | `self.assertLess(max_err, 0.1)` has no diagnostic message. When this fails, the only output is `AssertionError: 0.12 not less than 0.1`, making triage difficult. | Add a message: `self.assertLess(max_err, 0.1, f"Torus constraint violation {max_err:.2e} exceeds 0.1")`. |
| 3 | **RISK** | `tests/test_solver.py:64` | `self.assertLess(traj_down.energy, traj_up.energy)` does not verify that both energies are finite and positive. A NaN or negative energy would still pass if both are NaN (unittest treats `NaN < NaN` as `False`, but `NaN > NaN` is also `False`—the test would fail, but with a misleading message). | Add guards: `self.assertTrue(jnp.isfinite(traj_down.energy))` and `self.assertGreater(traj_down.energy, 0)` before the comparison. |
| 4 | **RISK** | `tests/test_solver.py:76` | `mid = traj.xs[10]` hardcodes the midpoint index. This implicitly depends on `n_steps=20` yielding exactly 21 path points. If `n_steps` is changed, this silently samples the wrong point. | Use `mid_idx = traj.xs.shape[0] // 2` or `n_steps // 2`. |
| 5 | **STYLE** | `tests/test_solver.py:33,34,62,63,77` | `print()` statements are used for diagnostics. These pollute stdout in CI and are not captured by test runners in verbose mode. | Remove `print` calls or use `self.subTest()` context managers / `logging.debug()`. |
| 6 | **STYLE** | `tests/test_solver.py:48–49` | `h_net` and `w_net` are plain Python lambdas (`lambda x: jnp.eye(3)`, `lambda x: jnp.array([0.5, 0.0, 0.0])`). While these work with JAX tracing, they are not `eqx.Module` instances and cannot be serialised or filtered by `eqx.partition`. If the solver or Randers internals ever call Equinox tree utilities, this will break. | Wrap in trivial `eqx.Module` subclasses or use `eqx.nn.Lambda`. |
| 7 | **RISK** | `tests/test_solver.py` (file-level) | No test verifies boundary conditions: that `traj.xs[0]` equals `p_start` and `traj.xs[-1]` equals `p_end`. The AVBD solver concatenates boundaries in its output (`src/ham/solvers/avbd.py:198–199`), but a regression could silently break this. | Add `jnp.allclose(traj.xs[0], start)` and `jnp.allclose(traj.xs[-1], end)` assertions to at least one test. |
| 8 | **RISK** | `tests/test_solver.py` (file-level) | No test for `train_mode=False` (the `while_loop` path in `src/ham/solvers/avbd.py:193`). This code path uses `jax.lax.while_loop` instead of `jax.lax.scan` and has different tracing behaviour. | Add a test that calls `solver.solve(..., train_mode=False)` and verifies the output matches or converges. |
| 9 | **RISK** | `tests/test_solver.py` (file-level) | No JIT compatibility test. `spec/ARCH_SPEC.md` § 4.2 states the solver is "fully differentiable." A `jax.jit(solver.solve)(...)` call would catch Python-side-effect bugs inside the traced computation. | Add `test_jit_compatibility` wrapping `solve` in `jax.jit` (or `eqx.filter_jit`). |
| 10 | **RISK** | `tests/test_solver.py` (file-level) | No gradient test. The AVBD solver is advertised as differentiable w.r.t. metric parameters (`spec/ARCH_SPEC.md` § 4.2). No test verifies that `jax.grad` through `solve` produces finite gradients. | Add a test that differentiates the solver output energy w.r.t. a learnable metric parameter and checks `jnp.isfinite(grad)`. |
| 11 | **STYLE** | `tests/test_solver.py:17` | `tol=1e-6` is set but the solver's convergence check based on `tol` is not actually exercised (the `while_loop` path uses only `step < iterations`; see `src/ham/solvers/avbd.py:193`). This parameter is misleading in the test setup. | Either remove `tol` from `setUp` to rely on defaults, or note in a comment that `tol` is not used in `train_mode=True`. |
| 12 | **STYLE** | `tests/test_solver.py:13` | Single test class `TestSolver` mixes integration tests for three unrelated manifold/metric combinations. | Consider splitting into `TestTorusSolver`, `TestSphereSolver`, `TestParaboloidSolver` for clearer failure attribution in CI. |

## Test Coverage Assessment

| Public API | Tested? | Notes |
|---|---|---|
| `AVBDSolver.solve(metric, start, end, n_steps)` | **Yes** | Tested on 3 manifold/metric combos |
| `AVBDSolver.solve(..., constraints=[...])` | **Partial** | Only `test_paraboloid_implicit` passes a constraint function |
| `AVBDSolver.solve(..., train_mode=False)` | **No** | `while_loop` inference path untested |
| `AVBDSolver.solve(..., key=...)` | **No** | Custom PRNG key path untested |
| `Trajectory` output fields (`xs`, `vs`, `energy`, `constraint_violation`) | **Partial** | `xs` and `energy` checked; `vs` and `constraint_violation` never asserted |
| JIT compatibility (`jax.jit(solve)`) | **No** | — |
| Differentiability (`jax.grad` through `solve`) | **No** | Critical gap given ARCH_SPEC § 4.2 |
| Boundary condition preservation | **No** | `traj.xs[0] == start`, `traj.xs[-1] == end` never checked |
| Coincident endpoints (`start == end`) | **No** | Edge case that triggers zero-velocity initialisation |
| `vmap` over batch of endpoint pairs | **No** | Batch-First principle (`spec/ARCH_SPEC.md` § 1) not verified |

## Positive Patterns

1. **Topological assertions** (issues #1 in `test_torus_topology`): checking both constraint satisfaction *and* qualitative path shape (max-Z > 0.5) is a strong pattern that catches shortcuts through the interior.
2. **Directional asymmetry test** (`test_sphere_zermelo`): verifying downwind < upwind energy is a clean, physics-meaningful invariant that doesn't depend on exact numerical values.
3. **Constraint pass-through** (`test_paraboloid_implicit`): explicitly passing an equality constraint function exercises the augmented Lagrangian branch of the solver.
4. **x64 mode enabled** (line 5): `jax_enable_x64` is correctly set at the top of the file, preventing silent float32 truncation in geodesic computations.
