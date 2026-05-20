# Code Review: `tests/test_transport.py`
**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary

The test file covers the key `berwald_transport` function across Euclidean, Riemannian, and Randers metrics with five well-structured tests. The physical reasoning in docstrings is excellent and the test scenarios are intelligently chosen (flat invariance, isometry, norm-drift, velocity-dependence, holonomy). However, there are several issues: the tests only exercise the `berwald_transport` convenience wrapper and never test `BerwaldConnection` or `christoffel_symbols` directly; a `print()` call is embedded in test logic; mock manifolds duplicate definitions found elsewhere in the test suite; the `dt` hardcoding in the source creates an implicit coupling with test path resolution; and the holonomy test uses a very loose tolerance without justification.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `tests/test_transport.py:59` | **Off-by-one in assertion due to source `result` construction.** The source (`transport.py:59`) builds `result = concat([vec_start[None,:], transported_vecs[:-1]])`, meaning `result[0]` is always the unmodified input and `result[-1]` is the *second-to-last* transported vector — the final integration step is discarded. The Euclidean test asserts every row equals `[1,0]` so this bug is masked (all rows are identical in flat space), but the Riemannian sphere isometry test (`test_riemannian_sphere_isometry`) checks norms of `vecs` which includes the unstransported `vec_start` at index 0 and omits the truly final transported vector. This may hide norm-drift at the endpoint. | The test should explicitly document which semantics it expects (initial-aligned vs. final-aligned). Add an assertion `assert vecs.shape[0] == path_x.shape[0]` and verify that `vecs[-1]` corresponds to the transport at the last path point, not the penultimate one. |
| 2 | **RISK** | `tests/test_transport.py:121–125` | **Lambda functions `h_net` and `w_net` capture no closure state but are non-trivially traced by JAX.** `w_net = lambda x: jnp.array([0.5 * x[1], 0.0])` constructs a new `jnp.array` from a Python list every call. Under `jit`, this works, but `jnp.array([..., 0.0])` forces a concrete Python float `0.0` into the trace — which is fine here but fragile if the lambda is later modified to include dynamic values. | Use `jnp.stack([0.5 * x[1], jnp.zeros_like(x[0])])` or similar fully-traced construction for robustness. |
| 3 | **RISK** | `tests/test_transport.py:148,181` | **`print()` calls in test methods.** `print(f"Randers Norm Drift: ...")` and `print(f"Velocity Dependence Diff: ...")` produce console output on every test run. These are debugging artefacts that add noise to CI output and are not assertions. | Remove print statements or replace with `logging.debug(...)` behind a verbosity flag. |
| 4 | **RISK** | `tests/test_transport.py:228` | **Holonomy test uses `atol=1e-1` — very loose tolerance.** A 10% error margin on a cosine comparison could mask real numerical regression. The expected holonomy angle $2\pi\cos(\theta)$ is a well-known exact result. With 200 path steps and `float64`, error below `1e-2` is achievable for a first-order Euler integrator. | Tighten to `atol=5e-2` and add a comment justifying the tolerance from the discretization order and step count. Alternatively, increase path resolution to 500 points and tighten to `atol=1e-2`. |
| 5 | **RISK** | `tests/test_transport.py:14–37` | **Mock manifolds `FlatPlane` and `Sphere` duplicate definitions likely present in other test files.** If these mocks diverge from the canonical manifolds in `surfaces.py` or other tests, behavioral differences may mask bugs. | Extract shared test manifold fixtures into a `tests/conftest.py` or `tests/helpers.py` module, or use the real `EuclideanSpace` and `Sphere` from `ham.geometry.surfaces`. |
| 6 | **STYLE** | `tests/test_transport.py:1–12` | **Imports `numpy` as `np` alongside `jax.numpy` as `jnp`.** Both are used (`np.testing.assert_allclose` and `jnp` for array construction). This is acceptable but mixing the two makes it non-obvious which arrays are JAX-traced and which are plain NumPy. | Consistent and intentional — no change needed, but a brief top-of-file comment explaining the convention would help future readers. |
| 7 | **STYLE** | `tests/test_transport.py:40–41` | **`setUp` creates `self.sphere` but only two of five tests use it.** The `self.plane` fixture is used by three tests, `self.sphere` by two. Both are cheap to construct, so this is not a performance issue, but it slightly reduces test readability by suggesting all tests share the same fixtures. | Acceptable as-is. Mention for awareness only. |

## Test Coverage Assessment

| Public Symbol | Tested? | Notes |
|---|---|---|
| `berwald_transport()` | Yes | Covered by all 5 tests via the convenience wrapper. |
| `BerwaldConnection` class | **No (direct)** | Never instantiated directly in tests. Only tested indirectly through `berwald_transport`. |
| `BerwaldConnection.christoffel_symbols()` | **No** | No test verifies the Christoffel tensor output shape, symmetry, or values against known analytical results. |
| `BerwaldConnection.parallel_transport()` | **No (direct)** | Only exercised indirectly. |
| `Connection` base class | **No** | Abstract base — acceptable not to test, but a test verifying `NotImplementedError` on the abstract methods would improve coverage. |

### Gap Analysis
- **Missing: Christoffel symbol unit test.** For Euclidean space, $\Gamma^i_{jk} = 0$ everywhere. A test asserting `christoffel_symbols(x, v) == zeros((D,D,D))` would validate the Hessian differentiation pipeline independently of the integrator.
- **Missing: Zero-velocity edge case.** No test passes `path_v = zeros(...)` to check that transport degenerates gracefully (no division by zero in the Christoffel computation or ODE step).
- **Missing: Single-step path.** No test checks `berwald_transport` with a path of length 1 (single point), which would exercise the boundary condition in the `scan`/`concat` logic.
- **Missing: `jit` compatibility test.** No test wraps `berwald_transport` in `jax.jit(...)` to verify it traces without side effects. The source uses `len(path_x)` inside the scanned function which is a Python-level operation — this works because `scan` receives the full array, but an explicit `jit` test would document this contract.
- **Missing: `grad` compatibility test.** No test verifies that `berwald_transport` is differentiable w.r.t. `vec_start` or metric parameters. This is important for any downstream training loop that backpropagates through transport.
- **Missing: Batch dimension test.** ARCH_SPEC mandates batch-first `(B, ...)` convention. No test verifies whether `berwald_transport` handles batched paths or if `vmap` over it works correctly.

## Positive Patterns

1. **Excellent physical reasoning in docstrings.** Each test method explains the geometric scenario, the expected physics, and *why* the assertion should hold. This makes the tests self-documenting.
2. **Good progression of complexity.** Tests go from trivial (Euclidean flat) → non-trivial (Riemannian isometry) → Finsler-specific (Randers drift and velocity-dependence) → advanced (holonomy). This makes failure diagnosis easier.
3. **`jax_enable_x64` is correctly enabled.** Geometric computations with Christoffel symbols require double precision; this is set at file scope.
4. **`assertNotAlmostEqual` for the Randers drift test.** Correctly asserts that the norm *changes*, rather than converging — a subtle but important Finsler property.
5. **Holonomy test checks `cos(angle)` instead of raw angle.** Avoids $\pm 2\pi$ wrapping issues.
