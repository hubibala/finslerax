# Code Review: `solvers/geodesic.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

The IVP geodesic solver is compact (~85 lines) and correctly structured around `jax.lax.fori_loop` / `jax.lax.scan`, making it JIT-compatible. However, it contains one clear **BUG** (dead `step_size` parameter that misleads every caller in the codebase), two **RISK** items (hardcoded velocity/acceleration clamps that silently corrupt solutions, and off-manifold RK4 intermediate evaluations), and several **STYLE** issues. Test coverage is adequate for basic correctness but lacks gradient-through-solver and edge-case tests.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/solvers/geodesic.py:23-25`, `:63-64`, `:73-74` | **`step_size` is a dead parameter.** It is stored in `__init__` but never read. Both `shoot()` and `trace()` compute `dt = t_max / self.max_steps`, ignoring `step_size` entirely. Every caller in the codebase passes `step_size` (README, tests, examples) under the false assumption it controls resolution. The docstring even documents it as *"Timestep dt for the RK4 numerical integration"*. | Either (a) use `step_size` to derive `n_steps = int(t_max / self.step_size)` inside `shoot`/`trace`, or (b) remove the parameter and document that resolution is controlled solely by `max_steps`. Option (a) is preferred — it makes the API match the documented semantics. |
| 2 | **RISK** | `src/ham/solvers/geodesic.py:34-36` | **Hardcoded velocity clamp at 10.0 silently corrupts geodesics.** `safe_v = jnp.where(curr_v_norm > 10.0, ...)` caps velocity magnitude. For learned metrics or high-curvature manifolds, the natural geodesic velocity can legitimately exceed 10.0. When this clamp activates, the solver integrates a different ODE without warning, producing wrong trajectories. | Make the velocity cap a configurable parameter with a large default (e.g., `max_velocity=1e6`). Alternatively, remove the cap entirely and rely on adaptive step-size or the manifold projection to keep trajectories bounded. |
| 3 | **RISK** | `src/ham/solvers/geodesic.py:40-42` | **Hardcoded acceleration clamp at 100.0.** Same issue as #2. `safe_dv = jnp.where(dv_norm > 100.0, ...)` silently clips geodesic acceleration, altering the ODE. | Same fix as #2 — make configurable or remove. |
| 4 | **RISK** | `src/ham/solvers/geodesic.py:46-50` | **RK4 intermediate stages evaluate at off-manifold points.** The intermediate points `y0 + 0.5*dt*k1`, `y0 + 0.5*dt*k2`, `y0 + dt*k3` are not projected onto the manifold before calling `dynamics()`. For high-curvature manifolds (small Sphere, Hyperboloid), `metric.geod_acceleration` is evaluated at points not on $\mathcal{M}$, introducing systematic integration error. Only the final result (lines 55-56) is projected. | Project the position component at each RK4 stage: extract `x` from the intermediate `y`, project via `manifold.project`, then reassemble before calling `dynamics`. This is standard for Lie-group / manifold ODE integrators. |
| 5 | **RISK** | `src/ham/solvers/geodesic.py:33-34` | **Uses `jnp.linalg.norm` instead of `safe_norm`.** The codebase provides `ham.utils.math.safe_norm` (gradient-safe via `sqrt(max(sum(x²), eps))`), but the dynamics function uses `jnp.linalg.norm(curr_v) + 1e-12`. While adding `1e-12` avoids division by zero in the forward pass, `jnp.linalg.norm` still produces NaN gradients at zero. If a geodesic velocity passes through zero (e.g., geodesic turning point), reverse-mode AD will propagate NaN. | Replace with `safe_norm(curr_v)` from `ham.utils.math`. |
| 6 | **STYLE** | `src/ham/solvers/geodesic.py:11` | **`ExponentialMap` is a plain Python class, not an `eqx.Module`.** All other core abstractions (`FinslerMetric`, manifolds, models) inherit from `eqx.Module`. A plain class cannot participate in pytree operations and forces JAX to recompile when a different solver instance is used. | Make it `class ExponentialMap(eqx.Module):` with `step_size` and `max_steps` as static fields. |
| 7 | **STYLE** | `src/ham/solvers/geodesic.py:8` | **`GeodesicState.t` is dead code.** The `t` field is incremented at line 57 but never read anywhere — not in the solver, not in tests, not in any caller. It adds a traced scalar to the loop carry for no purpose. | Remove the `t` field, or use it for adaptive time-stepping / early termination. |
| 8 | **STYLE** | `src/ham/solvers/geodesic.py:60-61` | **`shoot` docstring is incomplete.** It says *"Computes the endpoint Exp_x0(t_max * v0)"* but does not document the input shapes, the single-sample convention (user must `vmap`), or the return shape. Per `spec/ARCH_SPEC.md` § 3 (Batch-First), the batch convention should be stated. | Add parameter/return docstring specifying shapes `(D,)` and the `vmap` pattern. |
| 9 | **STYLE** | `src/ham/solvers/geodesic.py:34,41` | **Magic epsilon `1e-12` not sourced from `ham.utils.math.GRAD_EPS`.** The codebase defines `GRAD_EPS = 1e-12` as the canonical epsilon for gradient-safe operations, but this file uses a hardcoded literal. | Import and use `GRAD_EPS`. |

## Test Coverage Assessment

| Public Function | Tested? | Details | Gaps |
|-----------------|---------|---------|------|
| `ExponentialMap.shoot()` | Yes | `test_euclidean_ballistic`, `test_sphere_great_circle` | No gradient-through-shoot test. No test with `t_max != 1.0`. |
| `ExponentialMap.trace()` | Yes | `test_energy_conservation_randers`, `test_manifold_adherence` | No test with non-unit `t_max`. |
| `ExponentialMap.__init__` | Yes (implicitly) | Constructed in `setUp` | No test verifying `step_size` actually controls resolution (because it doesn't — see Bug #1). |
| `GeodesicState` | Yes (implicitly) | Used internally | — |

### Critical Test Gaps

1. **No differentiability test.** `ExponentialMap` is used inside differentiable training (`losses.py:422`), but no test verifies `jax.grad(solver.shoot, ...)` produces correct gradients w.r.t. metric parameters. Recommended: add a test that differentiates through `shoot` on `Euclidean` and checks the gradient is finite and non-zero.
2. **No test with velocity exceeding the 10.0 clamp.** If the clamp is triggered, the solver silently changes behavior. A test should verify what happens when initial velocity magnitude exceeds 10.0.
3. **No test for `step_size` semantics.** Since `step_size` is documented as controlling `dt`, there should be a test that varying `step_size` changes integration accuracy. (This test would currently fail, exposing Bug #1.)
4. **No edge-case test for zero initial velocity.** `v0 = zeros` is a valid geodesic (stationary point). The solver should return `x0` unchanged.

## Positive Patterns

1. **Correct use of `jax.lax.fori_loop` and `jax.lax.scan`.** The solver avoids Python-level loops, making it fully JIT-compatible and efficient. `fori_loop` for `shoot` (no trajectory needed) and `scan` for `trace` (trajectory collected) is the idiomatic split.
2. **Manifold projection after each step.** Lines 55-56 call `manifold.project` and `to_tangent` to correct numerical drift, which is essential for constrained manifolds like the Sphere.
3. **`NamedTuple` state.** `GeodesicState` is a clean, immutable state container that is a valid JAX pytree.
4. **Separation of concerns.** The solver takes `metric` as an argument rather than storing it, keeping the solver reusable across different metrics without reconstruction.
5. **Compact, readable implementation.** The entire module is ~85 lines with clear structure: state definition → single-step → shoot → trace.
