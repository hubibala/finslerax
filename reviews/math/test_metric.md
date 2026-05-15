# Math Review: test_metric

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

The test file is **structurally sound** for the properties it covers: Euclidean spray vanishing, spray 2-homogeneity, inner-product consistency, and the geodesic-acceleration sign convention. All tested assertions are mathematically correct. However, the test suite has **significant coverage gaps**: no Finslerian (non-Riemannian) metrics are tested, several public API methods are untested, and important properties like positive definiteness, $v = 0$ behaviour, and regularization impact are absent. One defined test fixture (`ScaledEuclideanMetric`) is unused.

## Formula-by-Formula Audit

### 1. `test_euclidean_spray_is_zero` (lines 38–51)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Geometric Hierarchy table — Euclidean row: Spray $G^i = 0$.
- **Implementation:**
  ```python
  spray = self.euc.spray(x, v)          # expects 0
  acc   = self.euc.geod_acceleration(x, v)  # expects 0
  np.testing.assert_allclose(spray, jnp.zeros_like(spray), atol=1e-5)
  np.testing.assert_allclose(acc,   jnp.zeros_like(acc),   atol=1e-5)
  ```
- **Verification:** For $F(x,v) = \|v\|$, $E = \tfrac12\|v\|^2$, $\nabla_x E = 0$, $\text{Jac}_x(\nabla_v E) = 0$, so rhs $= 0$ and $G = 0$ regardless of Hessian regularization. Assertion is exact to machine precision.
- **Verdict:** CORRECT
- **Notes:** Tolerance `atol=1e-5` is conservative — the result is analytically exactly zero. This is fine.

---

### 2. `test_spray_homogeneity` (lines 53–76)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 — $G^i$ are 2-homogeneous in $v$ (follows from 1-homogeneity of $F$ and the structure of the spray formula).
- **Literature Reference:** Bao–Chern–Shen, *An Introduction to Riemann–Finsler Geometry*, Springer, 2000, Proposition 5.2.1.
- **Implementation:**
  ```python
  G_lambda_v = curved.spray(x, lambda_val * v)
  expected   = (lambda_val**2) * G_v
  np.testing.assert_allclose(G_lambda_v, expected, rtol=1e-4, atol=1e-6)
  ```
- **Verification:** The `CurvedMetric` defines $F(x,v) = \sqrt{\sum_i (1+x_i^2)v_i^2}$, a Riemannian metric with $g(x) = \text{diag}(1+x^2)$. The Hessian $\text{Hess}_v(E) = g(x)$ is $v$-independent, and the rhs is quadratic in $v$, so the regularization term $\epsilon I$ in [metric.py:60](src/ham/geometry/metric.py#L60) does not break 2-homogeneity for this metric class. The test is mathematically valid.
- **Verdict:** CORRECT

---

### 3. `test_inner_product_consistency` (lines 78–91)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 — $g_{ij}(x,v) = \frac{1}{2}\frac{\partial^2 F^2}{\partial v^i \partial v^j}$.
- **Implementation:**
  ```python
  val      = self.euc.inner_product(x, v, w1, w2)
  expected = jnp.dot(w1, w2)
  np.testing.assert_allclose(val, expected, atol=1e-5)
  ```
- **Verification:** For Euclidean $E = \tfrac12 \|v\|^2$, $g_{ij} = \delta_{ij}$, so $\langle w_1, w_2 \rangle_g = w_1 \cdot w_2$. The assertion is correct.
- **Verdict:** CORRECT

---

### 4. `test_inner_product_curved` (lines 93–115)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 — fundamental tensor definition.
- **Implementation:**
  ```python
  expected_g   = jnp.diag(1.0 + x**2)   # diag([2, 5, 10])
  expected_val = jnp.dot(w1, jnp.dot(expected_g, w2))
  np.testing.assert_allclose(val, expected_val, atol=1e-5)
  ```
- **Verification:** $E = \tfrac12 \sum_i (1+x_i^2)v_i^2$, $g_{ij} = (1+x_i^2)\delta_{ij}$. For $x=[1,2,3]$: $g = \text{diag}(2,5,10)$. With $w_1=[1,0,0.5]$, $w_2=[0,1,2]$: $w_1^T g\, w_2 = 0 + 0 + 10 = 10$. Correct.
- **Verdict:** CORRECT

---

### 5. `test_acceleration_sign` (lines 117–127)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 — geodesic equation $\ddot{x}^i + 2G^i = 0$, hence $\ddot{x}^i = -2G^i$.
- **Implementation:**
  ```python
  np.testing.assert_allclose(acc, -2.0 * spray, atol=1e-6)
  ```
  Source code at [metric.py:63](src/ham/geometry/metric.py#L63): `return -2.0 * self.spray(x, v)`.
- **Verification:** The relation $\ddot{x} = -2G$ is the defining identity. The test verifies the implementation is consistent. Correct.
- **Verdict:** CORRECT

---

## Coverage Gaps

### 6. No test for positive definiteness of $g_{ij}$

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1, condition 3 — "The Fundamental Tensor $g_{ij}$ is positive definite."
- **Severity:** WARNING
- **Notes:** No test checks that `jax.hessian(energy, argnums=1)(x, v)` has all positive eigenvalues. This is a core Finsler axiom. Without it, the linear solve in `spray()` could silently return non-physical results.
- **Recommended Action:** Add a test that computes the eigenvalues of $g_{ij}$ and asserts they are all positive, for at least the Euclidean, curved-Riemannian, and a Randers metric.

### 7. No Finslerian (non-Riemannian) test metric

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 — Randers metric $F(x,v) = \sqrt{v^T M v} + \beta \cdot v$.
- **Severity:** WARNING
- **Notes:** Every test metric in the file is Riemannian ($g_{ij}$ independent of $v$). The codebase's central claim is Finsler generality, yet no test exercises a metric where the fundamental tensor genuinely depends on the reference direction $v$. Spray homogeneity, in particular, is more fragile for Finsler metrics because the Hessian $\text{Hess}_v(E)$ then varies with $v$, and the regularization $\epsilon I$ in [metric.py:60](src/ham/geometry/metric.py#L60) **breaks exact 2-homogeneity** in that case.
- **Recommended Action:** Add a `RandersMetric` test fixture and verify spray homogeneity and inner-product $v$-dependence.

### 8. No test for $v = 0$ singularity

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1 — $F$ is only $C^\infty$ on $TM \setminus \{0\}$; $v = 0$ is a known singularity.
- **Severity:** WARNING
- **Notes:** The spray involves dividing by $g_{ij}$, which becomes singular or degenerate at $v = 0$ for general Finsler metrics. The regularization $\epsilon I$ is meant to handle this, but no test asserts that the output is finite (no NaN/Inf) when $v \to 0$.
- **Recommended Action:** Add a test with $v = [0, 0, 0]$ (or near-zero) asserting `jnp.all(jnp.isfinite(spray))`.

### 9. Regularization impact untested

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.2 (implicit solve).
- **Severity:** WARNING
- **Notes:** [metric.py:60](src/ham/geometry/metric.py#L60) adds `1e-4 * jnp.eye(...)` to the Hessian before solving. For metrics where the minimum eigenvalue of $g_{ij}$ is comparable to $10^{-4}$, this perturbation significantly alters the computed spray. No test quantifies this error or verifies it is bounded.
- **Recommended Action:** Add a test computing the spray with and without regularization for a well-conditioned metric, and assert the relative error is below a documented threshold.

### 10. `energy` method untested

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 — $E(x,v) = \tfrac12 F^2(x,v)$.
- **Severity:** NOTE
- **Notes:** The `energy()` method at [metric.py:33](src/ham/geometry/metric.py#L33) is the root of the computational graph. Although it is exercised indirectly through `spray()` and `inner_product()`, a direct unit test would catch regressions early.
- **Recommended Action:** Add `assert_allclose(metric.energy(x, v), 0.5 * metric.metric_fn(x, v)**2)`.

### 11. `arc_length` method untested

- **Spec Reference:** N/A (numerical integration, not a spec formula).
- **Severity:** WARNING
- **Notes:** [metric.py:66–74](src/ham/geometry/metric.py#L66-L74) implements arc-length integration via midpoint evaluation. No test covers this method.
- **Recommended Action:** Add a test on a known geodesic (e.g., a straight line in Euclidean space) and assert the computed length equals the Euclidean distance.

### 12. `ScaledEuclideanMetric` defined but unused

- **Spec Reference:** N/A.
- **Severity:** NOTE
- **Notes:** [test_metric.py:20–23](tests/test_metric.py#L20-L23) defines `ScaledEuclideanMetric` with $F(x,v) = 0.5\|v\|$. No test references it. This fixture could usefully verify that the spray and geodesics are invariant under constant rescaling of $F$.
- **Recommended Action:** Either add tests exercising `ScaledEuclideanMetric`, or remove the dead code.

---

## Spec Discrepancy (informational, not a test-file defect)

### 13. MATH_SPEC § 2.1 explicit formula — missing factor of 2

- **Spec states:**
  $$G^i(x, v) = \frac{1}{4} g^{il} \left( 2 \frac{\partial^2 E}{\partial v^l \partial x^k} v^k - \frac{\partial E}{\partial x^l} \right)$$
- **Correct formula** (Bao–Chern–Shen, converting $F^2 = 2E$):
  $$G^i(x, v) = \frac{1}{4} g^{il} \left( 2 \frac{\partial^2 E}{\partial v^l \partial x^k} v^k - 2\frac{\partial E}{\partial x^l} \right) = \frac{1}{2} g^{il} \left( \frac{\partial^2 E}{\partial v^l \partial x^k} v^k - \frac{\partial E}{\partial x^l} \right)$$
- **Impact:** The spec's $\frac{\partial E}{\partial x^l}$ term is missing a factor of 2. The **code is correct** — it computes $\text{rhs} = \nabla_x E - \text{Jac}_x(\nabla_v E) \cdot v$ and returns $-\frac{1}{2}g^{-1}\text{rhs}$, which matches the standard formula.
- **Severity:** CRITICAL (spec defect; not a code or test defect)
- **Recommended Action:** Fix `spec/MATH_SPEC.md` § 2.1 to read $2\frac{\partial E}{\partial x^l}$ or factor the expression as $\frac{1}{2}g^{il}(\ldots)$.

### 14. MATH_SPEC § 2.2 implicit formula — sign error

- **Spec states:**
  $$\text{Hess}_v(E) \cdot (2G) = \nabla_x E - \text{Jac}_x(\nabla_v E) \cdot v$$
- **Correct formula** (from E-L: $g\,\ddot{x} = \text{rhs}$, $\ddot{x} = -2G$):
  $$\text{Hess}_v(E) \cdot (-2G) = \nabla_x E - \text{Jac}_x(\nabla_v E) \cdot v$$
- **Impact:** The spec has $+2G$ where it should be $-2G$. The code docstring at [metric.py:46](src/ham/geometry/metric.py#L46) correctly states $-2G$, and the implementation is correct.
- **Severity:** CRITICAL (spec defect; not a code or test defect)
- **Recommended Action:** Fix `spec/MATH_SPEC.md` § 2.2 to use $-2G$ or equivalently flip the sign of the rhs.

---

## Open Questions

1. **Is the `1e-4` Hessian regularization in `spray()` documented or justified anywhere?** It is described as mitigating "Randers ill-conditioning near boundary" but the magnitude is not derived from any stability analysis. Should this be a configurable parameter?
2. **Should the test suite include a Randers metric fixture?** The Zermelo parameterisation is specified in `spec/MATH_SPEC.md` § 5, and Randers is the primary non-Riemannian use case. Testing only Riemannian metrics leaves the Finsler-specific code paths unvalidated.
3. **Float64 vs Float32:** No test enables `jax_enable_x64`. All tolerances assume float32 AD noise (~$10^{-5}$). Is this deliberate policy, or should there be a parallel float64 test suite with tighter tolerances?
