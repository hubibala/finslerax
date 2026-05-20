# Code Review: `solvers/avbd.py`
**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

The AVBD solver is a well-structured, fully-differentiable BVP geodesic solver with correct use of `jax.lax.scan` for unrolled training and `jax.lax.while_loop` for inference. The overall architecture aligns with `spec/ARCH_SPEC.md § 4.2`. However, there are several issues: a data-dependent default PRNG key derivation that breaks `jit` tracing with non-concrete integer casts, a Python-level `if` on the dynamic value `num_constraints` inside a traced function, stochastic block descent that actually runs sequentially (defeating the purpose of randomisation under `scan`), and missing dual-variable updates for the augmented Lagrangian. Test coverage exists but is thin on edge cases and gradient correctness.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/solvers/avbd.py:62` | `data_hash = jnp.sum(p_start * 1e4 + p_end * 1e2).astype(jnp.int32)` — The `.astype(jnp.int32)` on a traced JAX value creates a **non-concrete** tracer inside `jax.random.fold_in`, which requires a concrete integer. This will fail under `jit` or `vmap` with a `ConcretizationTypeError`. The comment says it's meant to help `vmap`, but this pattern is precisely what breaks under `vmap`. | Use `jax.random.fold_in(key, jax.lax.bitcast_convert_type(jnp.sum(p_start + p_end), jnp.int32))` or, better, **require the caller to pass a key** and remove the default-key fallback entirely. Under `vmap`, the caller should split keys externally. |
| 2 | **BUG** | `src/ham/solvers/avbd.py:115–121` | `if num_constraints > 0:` is a **Python-level conditional** on a value that is fixed at trace time. This is safe *only* if `constraints` is always a compile-time constant (same list length across calls). However, `constraints` is accepted as `Optional[List[Callable]]`, meaning callers could pass lists of different lengths to the same `jit`-compiled function, causing silent re-tracing or stale traced code. More critically, the `jnp.stack([c(current_x) for c in constraints])` list comprehension is a Python loop that captures closures — this works under tracing but is fragile and forces one trace per distinct constraint count. | Document that `num_constraints` must be a compile-time constant, or wrap the constraint evaluation in `jax.lax.cond` / `jax.lax.switch`. Consider making `constraints` a `Tuple` to signal immutability. |
| 3 | **RISK** | `src/ham/solvers/avbd.py:155–170` | The stochastic block descent uses `jax.random.permutation` to randomise the update order, then feeds this permuted order into `jax.lax.scan`. But `scan` processes elements **sequentially** — each vertex update uses the *already-updated* path from the previous step in the scan. This means vertex `order[k]` sees updates from `order[0..k-1]`, creating order-dependent Gauss-Seidel-style updates. While mathematically valid (and sometimes faster converging), the randomisation only shuffles the Gauss-Seidel order rather than enabling true parallelism. If true Jacobi-style parallel updates were intended (as "Block Descent" suggests), all vertices should read the *same* `full_path` snapshot. | If Gauss-Seidel is intended, document this explicitly. If Jacobi updates are intended, replace `scan` with `vmap` over the vertex indices reading from a frozen path snapshot: `new_nodes = jax.vmap(lambda idx: update_vertex(full_path, idx - 1, s))(full_order)`. |
| 4 | **RISK** | `src/ham/solvers/avbd.py:173–174` | Dual variable updates (Lagrange multiplier `lambdas` and stiffness `stiffness`) are **never updated** — the comment says "Simplified: omitted." The `SolverState` propagates the initial zeros for `lambdas` and ones for `stiffness` throughout the entire solve. This means the augmented Lagrangian penalty in `loss_fn` (line 117–120) degenerates to a pure quadratic penalty `0.5 * 1.0 * c_val**2` with no dual ascent, which will not enforce constraints to zero — only minimise them. | Implement the standard ALM dual update: `new_lambdas = s.lambdas + s.stiffness * c_vals` and optionally `new_stiffness = jnp.minimum(s.stiffness * beta_growth, max_stiffness)` after each sweep. |
| 5 | **RISK** | `src/ham/solvers/avbd.py:96–100` | `local_action` computes `metric.energy(x_prev, v_in) + metric.energy(x, v_out)` where `v_in = log_map(x_prev, x)` and `v_out = log_map(x, x_next)`. The gradient of `local_action` w.r.t. `current_x` (line 110) differentiates through `log_map(x_prev, current_x)` as the second argument and `log_map(current_x, x_next)` as the first argument. The correctness of these gradients depends entirely on `log_map` being smoothly differentiable w.r.t. both arguments. The default `Manifold.log_map` uses a `_safe_norm_ratio_jvp` custom JVP for the scaling — if this custom JVP has any edge-case issues (e.g., when `x_prev == current_x`), the gradients will be wrong. | Add a test that explicitly checks `jax.grad` through `local_action` at coincident points (`x_prev == x`). |
| 6 | **RISK** | `src/ham/solvers/avbd.py:129` | Hard-coded `clip_value = 10.0` for gradient clipping is not exposed as a solver parameter. For metrics with very different scales (e.g., a learned metric whose energy is O(1000)), this clip threshold may be too aggressive or too permissive. | Expose `grad_clip` as an `AVBDSolver` attribute with default `10.0`. |
| 7 | **RISK** | `src/ham/solvers/avbd.py:188–189` | In non-train mode, `while_loop` uses `cond = lambda s: s.step < self.iterations`. This never checks for actual convergence (energy change < `energy_tol` or gradient norm < `tol`). The `tol` and `energy_tol` attributes are declared but **never referenced** anywhere in the code. The while loop is functionally identical to the `scan` version. | Implement proper convergence checking: `cond = lambda s: (s.step < self.iterations) & (jnp.abs(s.prev_energy - s.curr_energy) > self.energy_tol)`. |
| 8 | **STYLE** | `src/ham/solvers/avbd.py:3` | `List` and `Tuple` imported from `typing` but `Tuple` is unused. `Callable` and `Optional` are used. With Python 3.9+ (JAX requires ≥ 3.9), prefer `list[...]`, `tuple[...]` built-in generics. | Minor — clean up unused imports or modernise type hints. |
| 9 | **STYLE** | `src/ham/solvers/avbd.py:67` | `if constraints is None: constraints = []` — single-line compound statement. PEP 8 recommends separate lines for `if` body. | Split to two lines for readability. |
| 10 | **STYLE** | `src/ham/solvers/avbd.py:14–23` | `Trajectory` and `SolverState` are defined as `NamedTuple`. ARCH_SPEC uses `eqx.Module` for all stateful containers. `NamedTuple` is fine for pure data, but `SolverState` holds mutable-semantics fields like `step` and `prev_energy` that evolve across iterations — using `eqx.Module` would be more consistent with the codebase and automatically register as a PyTree. | Consider converting `SolverState` to `eqx.Module` for consistency, though `NamedTuple` is already a valid JAX PyTree. |

## Test Coverage Assessment

| Public Symbol | Tested? | Test Location | Notes |
|---------------|---------|---------------|-------|
| `AVBDSolver.solve` (Euclidean/Torus) | Yes | `tests/test_solver.py::test_torus_topology` | Checks surface constraint adherence and topology |
| `AVBDSolver.solve` (Randers/Sphere) | Yes | `tests/test_solver.py::test_sphere_zermelo` | Checks wind asymmetry |
| `AVBDSolver.solve` (Paraboloid + constraint) | Yes | `tests/test_solver.py::test_paraboloid_implicit` | Checks explicit constraint function |
| `AVBDSolver.solve` (Mesh/Pyramid) | Yes | `tests/test_mesh_solver.py::test_pyramid_surface_constraint` | Checks mesh surface adherence |
| `AVBDSolver.solve` (DiscreteRanders/Obstacle) | Yes | `tests/test_mesh_solver.py::test_obstacle_avoidance` | Checks anisotropic cost avoidance |
| `Trajectory` (output struct) | Implicit | All tests | Used but fields not individually validated |
| `SolverState` (internal) | No | — | Internal, but convergence fields untested |

### Coverage Gaps
1. **No gradient/differentiability test.** No test verifies that `jax.grad` through `AVBDSolver.solve` w.r.t. metric parameters produces finite, correct gradients. This is the solver's primary value proposition per `spec/ARCH_SPEC.md § 4.2`.
2. **No `jit` compilation test.** No test wraps `solve` in `jax.jit` to verify traceability — this would catch Issue #1.
3. **No `vmap` test.** No test verifies batched solving, which would also catch Issue #1.
4. **No convergence / energy monotonicity test.** No test checks that energy decreases across iterations.
5. **No edge case: coincident endpoints** (`p_start == p_end`). This exercises the noise injection (line 77) and zero-velocity edge case.
6. **No test for `train_mode=False`** (the `while_loop` path).
7. **`tol` and `energy_tol`** are set in test setUp but never exercised since the code never checks them (Issue #7).

## Positive Patterns

1. **Correct `jax.lax.scan` for differentiable unrolling** (line 186) — this is the right pattern for making the solver end-to-end differentiable through the training loop.
2. **Manifold-aware retraction** (line 134) — updates are properly retracted back to the manifold after each gradient step, preventing drift.
3. **`safe_norm` usage for gradient clipping** (line 127) — uses the canonical `safe_norm` from `utils.math` to avoid NaN gradients at zero.
4. **Noise injection for degenerate initialisation** (line 77) — adding small noise to the linear interpolation prevents zero-velocity segments that would cause gradient singularities.
5. **Clean separation of train/inference modes** (lines 185–189) — `scan` for differentiable training, `while_loop` for inference, is the standard JAX pattern.
6. **NamedTuple for `Trajectory`** — lightweight, immutable, and automatically a JAX PyTree.
