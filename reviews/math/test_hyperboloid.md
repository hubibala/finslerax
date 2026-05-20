# Math Review: test_hyperboloid

**Reviewer:** Math Reviewer Agent
**Date:** 2026-05-15
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

The existing tests are **mathematically correct** in what they verify, but **coverage is critically incomplete**. The file validates only structural properties (constraint satisfaction, tangent orthogonality, idempotence) but contains **no tests for the exponential map, logarithmic map, geodesic distance, or parallel transport**—the core geometric operations. This means the most error-prone code paths (involving $\cosh$/$\sinh$ formulas and Taylor branches) have zero direct test coverage.

## Formula-by-Formula Audit

### 1. `minkowski_dot` helper (line 20)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Hyperboloid row — Minkowskian $\sqrt{\langle v, v\rangle_L}$
- **Literature Reference:** Standard Lorentzian inner product, Bao-Chern-Shen Ch. 12
- **Implementation:**
  ```python
  return -u[0] * v[0] + jnp.sum(u[1:] * v[1:])
  ```
  Computes $\langle u, v\rangle_L = -u_0 v_0 + \sum_{i=1}^n u_i v_i$.
- **Verdict:** CORRECT
- **Notes:** Matches [surfaces.py:314](src/ham/geometry/surfaces.py#L314) `_minkowski_dot`. However, the helper uses scalar indexing (`u[0]`) rather than `u[..., 0]`, so it only works for single vectors—adequate for a test helper.

### 2. `test_dimensions` (line 24)

- **Spec Reference:** Standard: $\mathbb{H}^n \subset \mathbb{R}^{n,1}$, ambient dim = intrinsic dim + 1
- **Implementation:** Asserts `intrinsic_dim == 2`, `ambient_dim == 3`.
- **Verdict:** CORRECT

### 3. `test_projection_constraints` (line 28)

- **Spec Reference:** Hyperboloid constraint $\langle x, x\rangle_L = -1$, upper sheet $x_0 > 0$.
- **Implementation:** Projects 10 random ambient points and checks both constraints.
- **Verdict:** CORRECT
- **Notes:** Tolerance `places=6` ($\sim 10^{-6}$) is appropriate given that projection involves a square root.

### 4. `test_projection_idempotence` (line 49)

- **Spec Reference:** A projection operator must satisfy $\pi \circ \pi = \pi$.
- **Implementation:** Uses $x = (1, 0, 0)$, the hyperboloid origin. Verifies $\langle x, x\rangle_L = -1$ trivially. Checks single and double projection at `atol=1e-8`.
- **Verdict:** CORRECT
- **Notes:** Only tests one base point (the origin). A stronger test would include a non-trivial point like $(\cosh(1), \sinh(1), 0)$.

### 5. `test_tangent_space_orthogonality` (line 57)

- **Spec Reference:** The tangent space $T_x\mathbb{H}^n = \{v : \langle x, v\rangle_L = 0\}$.
- **Implementation:**
  Uses $x = (\cosh 1, \sinh 1, 0)$ which satisfies $-\cosh^2 1 + \sinh^2 1 = -1$ ✓. Projects $v_{\text{amb}} = (1, 2, 3)$ via `to_tangent` and checks $\langle x, v_T\rangle_L = 0$.
- **Verdict:** CORRECT
- **Notes:** The source formula $v_T = v + \langle x, v\rangle_L \cdot x$ is verified by:
  $\langle x, v_T\rangle_L = \langle x, v\rangle_L + \langle x, v\rangle_L \langle x, x\rangle_L = \langle x, v\rangle_L(1 - 1) = 0$. ✓

### 6. `test_random_sampling` (line 73)

- **Spec Reference:** Hyperboloid constraint $\langle p, p\rangle_L = -1$, $p_0 > 0$.
- **Implementation:** 100 samples, `places=5` ($\sim 10^{-5}$).
- **Verdict:** CORRECT
- **Notes:** The source implementation ([surfaces.py:389](src/ham/geometry/surfaces.py#L389)) maps spatial $v \in \mathbb{R}^n$ via $x = (\cosh\|v\|, \frac{\sinh\|v\|}{\|v\|} v)$, which satisfies $-\cosh^2\|v\| + \sinh^2\|v\| = -1$ analytically. The looser tolerance is acceptable.

### 7. `test_metric_tensor_signature` (line 82)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 — Hyperboloid uses Minkowskian metric.
- **Implementation:** Asserts $g = \text{diag}(-1, 1, 1)$.
- **Verdict:** NOTE
- **Notes:** The `metric_tensor` method ([surfaces.py:402](src/ham/geometry/surfaces.py#L402)) returns the ambient Minkowski metric $\eta_{\mu\nu}$, not the induced Riemannian metric on $\mathbb{H}^n$. This is consistent with the implementation (all operations use the ambient inner product), but the docstring "The metric tensor returned should be Minkowski diag(-1, 1, 1)" could mislead readers into thinking the hyperboloid has an indefinite metric. The induced metric on the tangent space $T_x\mathbb{H}^n$ (a spacelike subspace) is positive-definite. The test is *internally consistent* but the naming is mathematically imprecise.

### 8. `test_retraction_stays_on_manifold` (line 92)

- **Spec Reference:** $\exp_x(v) = \cosh(\|v\|_L)\, x + \frac{\sinh(\|v\|_L)}{\|v\|_L}\, v$ (standard hyperboloid exp map).
- **Implementation:**
  Uses $x = (1, 0, 0)$, $v = (0, 0.5, 0.5)$. Tangent check: $\langle x, v\rangle_L = 0$ ✓. Verifies result is on manifold.
  The source `retract` ([surfaces.py:384](src/ham/geometry/surfaces.py#L384)) delegates to `project(exp_map(x, safe_delta))`, so this implicitly tests `exp_map`.
- **Verdict:** CORRECT
- **Notes:** This only checks that the *constraint* holds, not that the *value* of $\exp_x(v)$ is correct. For instance, $\exp_{(1,0,0)}((0, 0.5, 0.5))$ should equal $(\cosh\sqrt{0.5},\; \frac{\sinh\sqrt{0.5}}{\sqrt{0.5}} \cdot 0.5,\; \frac{\sinh\sqrt{0.5}}{\sqrt{0.5}} \cdot 0.5)$; the test does not verify this.

## Missing Coverage

### M-1. No test for `exp_map` correctness

- **Severity:** WARNING
- **File:** [tests/test_hyperboloid.py](tests/test_hyperboloid.py)
- **Details:** The exponential map $\exp_x(v) = \cosh(\|v\|_L)\, x + \frac{\sinh(\|v\|_L)}{\|v\|_L}\, v$ is implemented in [surfaces.py:345–355](src/ham/geometry/surfaces.py#L345-L355) with a Taylor branch for small $\|v\|_L$. No test verifies the *value* of the output against the analytical formula, nor exercises the Taylor branch ($\|v\|_L < \varepsilon$).
- **Recommended Action:** Add a test that computes $\exp_x(v)$ for known inputs and compares against $\cosh/\sinh$ values at `atol=1e-10`. Include a near-zero tangent vector ($\|v\|_L \sim 10^{-15}$) to exercise the Taylor branch.

### M-2. No test for `log_map`

- **Severity:** WARNING
- **File:** [tests/test_hyperboloid.py](tests/test_hyperboloid.py)
- **Details:** The log map ([surfaces.py:357–371](src/ham/geometry/surfaces.py#L357-L371)) uses $\text{arcsinh}$ (rather than $\text{arcosh}$) for numerical stability—a good choice—but has no tests at all.
- **Recommended Action:** Add a test verifying $\log_x(y)$ for known pairs. Minimum: check that for $y = (\cosh t, \sinh t, 0)$, $\log_{(1,0,0)}(y) = (0, t, 0)$.

### M-3. No roundtrip test for $\exp \circ \log = \text{id}$ and $\log \circ \exp = \text{id}$

- **Severity:** WARNING
- **File:** [tests/test_hyperboloid.py](tests/test_hyperboloid.py)
- **Details:** The inverse relationship between exp and log is the single most important consistency check for manifold implementations. Its absence means sign errors or wrong Taylor coefficients in either function could go undetected.
- **Recommended Action:** For random tangent vectors $v$ at several base points $x$:
  1. Assert `log_map(x, exp_map(x, v)) ≈ v` at `atol=1e-10`.
  2. For random pairs $(x, y)$ on the manifold: assert `exp_map(x, log_map(x, y)) ≈ y` at `atol=1e-10`.

### M-4. No test for geodesic distance

- **Severity:** WARNING
- **File:** [tests/test_hyperboloid.py](tests/test_hyperboloid.py)
- **Details:** The geodesic distance $d(x, y) = \text{arcosh}(-\langle x, y\rangle_L)$ is a fundamental quantity. While no explicit `distance` method may exist on `Hyperboloid`, the distance is implicitly encoded in `log_map` ($\|\log_x(y)\|_L = d(x, y)$). A test should verify $\|\log_x(y)\|_L = \text{arcosh}(-\langle x, y\rangle_L)$.
- **Recommended Action:** Add a distance consistency test comparing $\|\log_x(y)\|_L$ against $\text{arcosh}(-\langle x, y\rangle_L)$.

### M-5. No test for `parallel_transport`

- **Severity:** WARNING
- **File:** [tests/test_hyperboloid.py](tests/test_hyperboloid.py)
- **Details:** The parallel transport ([surfaces.py:373–379](src/ham/geometry/surfaces.py#L373-L379)) uses the closed-form formula $P_{x \to y}(v) = v + \frac{\langle y, v\rangle_L}{1 - \langle x, y\rangle_L}(x + y)$. I verified this is analytically correct (the result is tangent at $y$ and preserves the Minkowski norm). However, with zero test coverage, implementation bugs would be invisible.
- **Recommended Action:** Test three properties: (1) result is tangent at $y$: $\langle y, P(v)\rangle_L = 0$; (2) norm preservation: $\langle P(v), P(v)\rangle_L = \langle v, v\rangle_L$; (3) known value for a specific triple $(x, y, v)$.

### M-6. No edge-case tests

- **Severity:** NOTE
- **File:** [tests/test_hyperboloid.py](tests/test_hyperboloid.py)
- **Details:** Missing edge cases include:
  - **Zero tangent vector:** $\exp_x(0) = x$ and $\log_x(x) = 0$.
  - **Large distance:** points with $d(x,y) \gg 1$ (e.g., $d \approx 20$), where $\cosh$ and $\sinh$ overflow in float32. The test uses float64 which mitigates this, but verifying the large-distance regime is valuable.
  - **Projection of degenerate inputs:** all-zero vector, vectors with $x_0 \leq 0$.
- **Recommended Action:** Add targeted edge-case tests for at least the zero-vector and identity cases.

## Positive Observations

### S-1. Float64 precision enabled

- **Severity:** STRONG
- **File:** [tests/test_hyperboloid.py](tests/test_hyperboloid.py#L7)
- **Details:** `config.update("jax_enable_x64", True)` ensures geometric identity checks (like $\langle x, x\rangle_L = -1$) are not degraded by float32 rounding. This is correct practice for geometry tests.

### S-2. Independent `minkowski_dot` helper

- **Severity:** STRONG
- **File:** [tests/test_hyperboloid.py](tests/test_hyperboloid.py#L20)
- **Details:** The test defines its own Minkowski inner product rather than reusing the implementation's `_minkowski_dot`. This avoids circular validation—a correct testing pattern.

### S-3. Constraint verification approach

- **Severity:** STRONG
- **File:** [tests/test_hyperboloid.py](tests/test_hyperboloid.py)
- **Details:** Tests consistently verify both parts of the hyperboloid constraint ($\langle x, x\rangle_L = -1$ and $x_0 > 0$), not just one.

## Open Questions

1. **metric_tensor semantics:** Should `Hyperboloid.metric_tensor(x)` return the ambient Minkowski metric $\eta$ or the induced Riemannian metric on $T_x\mathbb{H}^n$ (an $n \times n$ positive-definite matrix)? The current implementation returns $\eta$, which is position-independent—correct for ambient computations, but may confuse downstream consumers expecting a Riemannian metric.

2. **Taylor branch thresholds:** The `exp_map` and `log_map` implementations switch to Taylor approximations below `TAYLOR_EPS` and `NORM_EPS` respectively. Are these thresholds validated for float64? The test file does not exercise these branches, so any threshold mistuning would go undetected.

3. **`parallel_transport` denominator clamp:** The implementation uses `jnp.maximum(1.0 - xy, 2.0)`. Since $1 - \langle x, y\rangle_L \geq 2$ on the manifold, this clamp is a no-op for valid inputs. However, it would silently mask off-manifold points (where $\langle x, y\rangle_L > -1$). Should this be an assertion or warning instead?
