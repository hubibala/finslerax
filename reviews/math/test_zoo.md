# Math Review: test_zoo

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source File:** `tests/test_zoo.py`  
**Cross-Referenced:** `src/ham/geometry/zoo.py`, `src/ham/utils/math.py`, `spec/MATH_SPEC.md` § 4–5

---

## Summary

The test suite covers the three main metric classes in `zoo.py` — `Euclidean`, `Riemannian`, and `Randers` — with basic sanity checks. All mathematical assertions are **directionally correct**: no sign errors, no wrong expected values, and no wrong formulas. However, two **WARNING**-level issues reduce the test's diagnostic power: (1) `test_randers_zero_wind` uses an unnecessarily loose tolerance (`places=2`) that could mask $O(10^{-2})$ regressions in the Zermelo formula, and (2) `test_randers_analytical_match` performs only a qualitative check (direction of inequality) while its docstring quotes exact analytical values that the code **cannot** reproduce due to the `tanh` causality squasher. No quantitative test of the Randers formula against `spec/MATH_SPEC.md` § 5 exists.

**Verdict:** Minor Issues — all assertions are mathematically valid; test power is reduced by loose tolerances and missing quantitative coverage of the Zermelo formula.

---

## Formula-by-Formula Audit

### 1. `test_euclidean_basic` (lines 34–39)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table: $F(x,v) = \sqrt{v^T v}$
- **Implementation:**
  ```python
  v = jnp.array([3.0, 4.0])
  cost = metric.metric_fn(x, v)
  self.assertAlmostEqual(cost, 5.0, places=7)
  ```
- **Verdict:** CORRECT
- **Notes:** $\sqrt{3^2 + 4^2} = 5$. Expected value is exact. Tolerance of $10^{-7}$ is appropriate: the only source of deviation is the $\epsilon$-floor in `safe_norm` ($\epsilon = 10^{-12}$), which does not activate since $\|v\|^2 = 25 \gg 10^{-12}$.

---

### 2. `test_riemannian_scaling` (lines 42–52)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table: $F(x,v) = \sqrt{v^T g(x)\, v}$
- **Implementation:**
  ```python
  g_net = lambda x: 4.0 * jnp.eye(2)
  v = jnp.array([1.0, 0.0])
  cost = metric.metric_fn(x, v)
  self.assertAlmostEqual(cost, 2.0, places=7)
  ```
- **Verdict:** CORRECT
- **Notes:** $g = 4I$ implies $F = \sqrt{v^T (4I) v} = 2\|v\| = 2$. The symmetrisation step $(G + G^T)/2$ is a no-op on $4I$. The `jnp.maximum(quad, 1e-12)` guard does not activate since $v^T (4I) v = 4 > 0$.

---

### 3. `test_riemannian_anisotropy` (lines 54–67)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table: $F(x,v) = \sqrt{v^T g(x)\, v}$
- **Implementation:**
  ```python
  g_net = lambda x: jnp.diag(jnp.array([1.0, 4.0]))
  cost_x = metric.metric_fn(x, jnp.array([1.0, 0.0]))  # sqrt(1) = 1
  cost_y = metric.metric_fn(x, jnp.array([0.0, 1.0]))  # sqrt(4) = 2
  ```
- **Verdict:** CORRECT
- **Notes:** With $g = \mathrm{diag}(1, 4)$: $F(x, e_1) = \sqrt{1 \cdot 1} = 1$ and $F(x, e_2) = \sqrt{4 \cdot 1} = 2$. Both inline comments in the source are accurate. The test correctly verifies that anisotropic scaling is handled. Tolerance $10^{-7}$ is appropriate.

---

### 4. `test_randers_analytical_match` — Qualitative Ordering (lines 70–99)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo formula)
- **Literature Reference:** Bao–Robles–Shen, "Zermelo navigation on Riemannian manifolds," *J. Diff. Geom.* 66 (2004), 377–435.
- **Implementation:**
  ```python
  h_net = lambda x: jnp.eye(2)
  w_net = lambda x: jnp.array([-0.5, 0.0])
  # ...
  self.assertGreater(cost_east, cost_west)
  ```
- **Verdict:** WARNING
- **Notes:**

  The assertion `assertGreater(cost_east, cost_west)` is **mathematically correct** — the Zermelo formula guarantees that headwind increases cost. However, the docstring (lines 75–83) quotes exact analytical values:

  > East (Against Wind): Cost = 2.0  
  > West (With Wind): Cost = 0.666…

  These values are correct only for the **unsquashed** wind $W = [-0.5, 0]$ with $h = I$:

  $$\lambda = 1 - \|W\|_h^2 = 0.75$$
  $$F(x, e_1) = \frac{\sqrt{0.75 \cdot 1 + 0.25} + 0.5}{0.75} = \frac{1.5}{0.75} = 2.0 \quad\checkmark$$
  $$F(x, -e_1) = \frac{\sqrt{0.75 + 0.25} - 0.5}{0.75} = \frac{0.5}{0.75} = 0.\overline{6} \quad\checkmark$$

  However, the implementation (`zoo.py:88–99`) applies a `tanh` causality squasher to $W$ before evaluating $F$. For $\|W\|_h = 0.5$, the squashed norm is $(1 - \epsilon)\tanh(0.5) \approx 0.462$, yielding $\lambda \approx 0.787$ and actual costs $\approx 1.86$ (east) and $\approx 0.68$ (west). The docstring values **never appear** in the code's output.

  **Recommended Action:** Either (a) replace the docstring values with the post-squash analytical values, or (b) add explicit `assertAlmostEqual` checks against the post-squash values to make the test quantitative.

---

### 5. `test_randers_analytical_match` — Homogeneity Check (lines 101–102)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 (positive homogeneity: $F(x, \lambda v) = \lambda F(x, v)$ for $\lambda > 0$)
- **Implementation:**
  ```python
  cost_east_2x = metric.metric_fn(x, 2.0 * v_east)
  self.assertAlmostEqual(cost_east_2x, 2.0 * cost_east, places=5)
  ```
- **Verdict:** CORRECT
- **Notes:** This tests the defining property $F(x, 2v) = 2F(x,v)$. The only source of deviation is the additive $10^{-9}$ inside `jnp.sqrt(discriminant + 1e-9)` in `zoo.py:120`. For $\|v\| \sim 1$, the discriminant is $O(1)$, so the homogeneity error is $\sim 10^{-9}/(2\sqrt{1}) \approx 5 \times 10^{-10}$, well within the tolerance of $10^{-5}$. Tolerance is appropriate.

---

### 6. `test_randers_convexity_protection` (lines 104–120)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5: "$\|W\|_h < 1$"
- **Implementation:**
  ```python
  w_net_illegal = lambda x: jnp.array([100.0, 100.0])
  cost = metric.metric_fn(x, v)
  self.assertFalse(jnp.isnan(cost))
  self.assertTrue(jnp.isfinite(cost))
  ```
- **Verdict:** CORRECT
- **Notes:** This is a **safety test**, not a formula test. It verifies that the `tanh` squasher in `_get_zermelo_data` prevents the discriminant $\lambda \|v\|_h^2 + \langle W, v\rangle_h^2$ from becoming negative (which would produce `NaN` under `sqrt`). The squasher bounds $\|W_{safe}\|_h < 1 - \epsilon$, guaranteeing $\lambda > 0$. The test correctly exercises this mechanism with a grossly invalid wind field.

---

### 7. `test_randers_zero_wind` (lines 122–138)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5: Setting $W = 0$ in the Zermelo formula gives $\lambda = 1$ and $F = \sqrt{\|v\|_h^2} = \|v\|_h$, i.e. the Riemannian norm.
- **Implementation:**
  ```python
  w_net_zero = lambda x: jnp.zeros(2)
  c1 = randers.metric_fn(x, v)
  c2 = riem.metric_fn(x, v)
  self.assertAlmostEqual(c1, c2, places=2)
  ```
- **Verdict:** WARNING
- **Notes:**

  The mathematical property is correct: when $W = 0$, the Randers metric degenerates to Riemannian. The sole sources of numerical difference between the two code paths are:

  | Source | Randers path (`zoo.py:117–121`) | Riemannian path (`zoo.py:33–35`) |
  |---|---|---|
  | sqrt guard | $\sqrt{d + 10^{-9}}$ | $\sqrt{\max(q,\, 10^{-12})}$ |
  | zero-velocity guard | adds $10^{-7}$ when $\|v\| < 10^{-7}$ | none |

  For a generic random $v$ with $\|v\| \sim 1$, the discrepancy is $\sim 10^{-9} / (2\sqrt{1}) \approx 5 \times 10^{-10}$.

  A tolerance of `places=2` ($10^{-2}$) is **7 orders of magnitude looser** than the actual error. This means a bug that introduces an $O(10^{-3})$ bias in the Randers formula — large enough to affect geodesic integration — would pass this test silently.

  **Recommended Action:** Tighten to `places=6` or, at minimum, `places=5`. The test should catch regressions at the $10^{-5}$ scale.

---

## Missing Coverage (Mathematical Properties)

### M1. No quantitative Zermelo formula test

- **Severity:** WARNING
- **Notes:** No test computes $F(x, v)$ for known $(h, W, v)$ and compares against the closed-form Zermelo result from `spec/MATH_SPEC.md` § 5. All Randers assertions are either qualitative (`assertGreater`) or structural (`isfinite`). A single test with a hand-computed expected value — accounting for the `tanh` squash — would catch coefficient-level errors in the discriminant assembly at `zoo.py:118–120`.
- **Recommended Action:** Add a test that computes $F$ for a known setup (e.g., $h = I$, $W = [0.3, 0]$, $v = [1, 0]$), manually evaluates the post-squash Zermelo formula, and asserts equality to `places=6`.

### M2. No Riemannian/Euclidean symmetry test

- **Severity:** NOTE
- **Notes:** The Euclidean and Riemannian metrics are **reversible** ($F(x, v) = F(x, -v)$). This is a defining property distinguishing them from Randers metrics. No test verifies this, though it follows trivially from the quadratic form $v^T G v$.

### M3. No Randers asymmetry quantification

- **Severity:** NOTE
- **Notes:** The Randers metric is **irreversible** ($F(x, v) \neq F(x, -v)$ when $W \neq 0$). The `test_randers_analytical_match` implicitly tests this via the headwind/tailwind ordering, but does not check the **ratio** $F(x, v) / F(x, -v)$, which is the defining asymmetry measure.

### M4. No positive-definiteness test

- **Severity:** NOTE
- **Notes:** `spec/MATH_SPEC.md` § 1.1 requires $F(x, v) > 0$ for all $v \neq 0$. No test checks this property directly (e.g., sampling random $v$ and asserting `cost > 0`). The existing tests use specific vectors that happen to have positive cost.

---

## Open Questions

1. **Squash-aware analytical test:** Should the test suite include expected values that account for the `tanh` causality squasher, or should a bypass mode (e.g., a flag to disable squashing) be added for testing the raw Zermelo formula?

2. **`DiscreteRanders` coverage:** The `DiscreteRanders` class in `zoo.py` (lines 127–145) has **no tests** in this file. Is it tested elsewhere, or is this a coverage gap?
