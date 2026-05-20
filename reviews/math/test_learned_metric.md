# Math Review: test_learned_metric

**Reviewer:** Math Reviewer Agent  
**Date:** 2026-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Major Issues.** The test file covers only two of many required mathematical properties for a learned Randers metric. While the two tests present (`test_zermelo_convexity_enforcement`, `test_gradients_exist`) are mathematically sound in what they verify, the file is critically incomplete: it omits tests for positive homogeneity $F(x,\lambda v) = \lambda F(x,v)$, positive definiteness of $g_{ij}$, the Randers metric formula itself, energy $E = \frac{1}{2}F^2$, spray correctness, and triangle inequality. Two specific mathematical issues are flagged below.

## Formula-by-Formula Audit

### 1. Zermelo convexity constraint $\|W\|_h < 1$

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, Zermelo Parameterization
- **Implementation:** [tests/test_learned_metric.py](tests/test_learned_metric.py#L30-L46)
  ```python
  H, W, lam = broken_metric._get_zermelo_data(x)
  w_norm = jnp.sqrt(jnp.dot(W, jnp.dot(H, W)))
  self.assertLess(w_norm, 1.0, "Wind vector violated convexity constraint!")
  self.assertGreater(lam, 0.0, "Lambda became non-positive!")
  ```
- **Verdict:** WARNING
- **Notes:** The test correctly verifies the causality constraint $\|W\|_h < 1$ and $\lambda > 0$ as required by the Zermelo navigation formulation. However, the implementation in `_get_zermelo_data` ([src/ham/geometry/zoo.py](src/ham/geometry/zoo.py#L95)) squashes to `max_speed = 1 - ε` with $\varepsilon = 10^{-5}$, then applies `tanh`, guaranteeing the bound $\|W\|_h < 1 - \varepsilon$. The test asserts `w_norm < 1.0` but should assert `w_norm < 1.0 - epsilon` to tightly verify the safety margin. As written, the test would pass even if the squasher was broken and produced $\|W\|_h = 0.999999$, which is dangerously close to the singularity at $\lambda = 0$.

  **Recommended Action:** Tighten the assertion to `self.assertLess(w_norm, 1.0 - self.metric.epsilon)` or at least `self.assertLess(w_norm, 0.96)` to verify the tanh squashing margin actually works.

### 2. Gradient existence checks

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (spray depends on $\nabla_x E$), § 1.2 (energy functional)
- **Implementation:** [tests/test_learned_metric.py](tests/test_learned_metric.py#L48-L64)
  ```python
  grad_x = jax.grad(self.metric.energy, argnums=0)(x, v)
  self.assertTrue(jnp.isfinite(grad_x).all())
  ```
- **Verdict:** WARNING
- **Notes:** Checking `jnp.isfinite` is necessary but not sufficient. For a valid Finsler metric the gradient $\nabla_v E = g_{ij} v^j$ should not be zero when $v \neq 0$, since positive definiteness of $g_{ij}$ implies $\nabla_v E \neq 0$ for non-zero $v$. The test does not check $\nabla_v E$ at all, which is the momentum variable critical for the spray solver ([src/ham/geometry/metric.py](src/ham/geometry/metric.py#L42-L43)). Furthermore, the test only checks gradient w.r.t. the wind network parameters (`w_net`) but not the Riemannian component (`h_net`).

  **Recommended Action:** Add assertions for: (a) $\nabla_v E \neq 0$ when $v \neq 0$, (b) gradient flow to `h_net` parameters as well.

## Missing Tests (Mathematical Gaps)

### 3. Positive 1-homogeneity: $F(x, \lambda v) = \lambda F(x,v)$

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 (Axiom 2), § 6.2
- **Verdict:** CRITICAL
- **Notes:** Positive homogeneity is the defining axiom of a Finsler metric and is required for the spray coefficients to be well-defined (`spec/MATH_SPEC.md` § 6.2). The `NeuralRanders` class inherits from `Randers`, whose `metric_fn` implements the Zermelo formula which is analytically 1-homogeneous. However, this property must be verified empirically for the learned metric since floating-point issues, epsilon regularisation, and the `is_zero` branch at small $\|v\|$ can break it.

  **Recommended Action:** Add a test:
  ```python
  for lam in [0.1, 2.0, 10.0]:
      F_v = self.metric.metric_fn(x, v)
      F_lv = self.metric.metric_fn(x, lam * v)
      np.testing.assert_allclose(F_lv, lam * F_v, rtol=1e-4)
  ```

### 4. Positive definiteness of fundamental tensor $g_{ij}$

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 (Axiom 3), definition $g_{ij} = \frac{1}{2}\partial^2 F^2 / \partial v^i \partial v^j$
- **Verdict:** CRITICAL
- **Notes:** Strong convexity of $F^2$ in $v$ is a necessary condition for the geodesic ODE to be well-posed (the Hessian $g_{ij}$ appears in the denominator of the spray linear solve, [src/ham/geometry/metric.py](src/ham/geometry/metric.py#L52)). No test verifies that eigenvalues of $g_{ij}$ are positive for the learned metric.

  **Recommended Action:** Add a test computing `jax.hessian(self.metric.energy, argnums=1)(x, v)` and asserting all eigenvalues are positive.

### 5. Energy identity $E = \frac{1}{2}F^2$

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2
- **Verdict:** WARNING
- **Notes:** The base class `FinslerMetric.energy` implements this as `0.5 * self.metric_fn(x, v)**2`, which is correct by construction. However, a basic sanity test asserting `E(x,v) == 0.5 * F(x,v)**2` for the `NeuralRanders` instance would catch any future override bugs.

  **Recommended Action:** Add a one-line assertion: `assert_allclose(metric.energy(x,v), 0.5 * metric.metric_fn(x,v)**2)`.

### 6. Symmetry of fundamental tensor $g_{ij} = g_{ji}$

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 (definition of $g_{ij}$)
- **Verdict:** WARNING
- **Notes:** By definition $g_{ij} = \frac{1}{2}\partial^2 F^2 / \partial v^i \partial v^j$ is symmetric (equality of mixed partials), but numerical auto-differentiation can introduce slight asymmetries. No test checks this.

  **Recommended Action:** Compute the Hessian of energy w.r.t. $v$ and assert `assert_allclose(g, g.T, atol=1e-6)`.

### 7. Spray consistency / geodesic equation

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1, § 2.2
- **Verdict:** WARNING
- **Notes:** The spray $G^i$ is the core dynamical object. No test in this file verifies that `self.metric.spray(x, v)` returns finite values, or that the geodesic acceleration $\ddot{x}^i = -2G^i$ produces a consistent trajectory for the learned metric.

  **Recommended Action:** Add at minimum a finiteness check: `assert jnp.isfinite(self.metric.spray(x, v)).all()`.

### 8. Non-negativity: $F(x, v) \geq 0$, $F(x,v) = 0 \iff v = 0$

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1
- **Verdict:** WARNING
- **Notes:** The Randers `metric_fn` returns `0.0` for the `is_zero` branch but should be tested to ensure it never returns negative values for non-zero $v$, especially near the wind-velocity alignment boundary where the discriminant could be marginal.

  **Recommended Action:** Test with several random $v$ vectors and assert `F(x,v) > 0` for all $v \neq 0$.

### 9. MockManifold is trivially correct but untested for tangent space

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (wind must lie in tangent space)
- **Verdict:** NOTE
- **Notes:** `MockManifold.to_tangent` is the identity, so the wind projection in `_get_zermelo_data` line `W_raw = self.manifold.to_tangent(z, W_raw)` is a no-op. This is fine for $\mathbb{R}^3$ but means the test does not exercise the tangent-projection path critical for curved manifolds.

## Open Questions

1. **Tolerance adequacy:** The convexity test uses `assertLess(w_norm, 1.0)` — should the safety margin ($\varepsilon = 10^{-5}$) be part of the assertion, or is the loose bound intentional to avoid brittleness on different platforms?

2. **Edge case coverage:** Should the test include $v = 0$ (degenerate tangent vector) and $x$ at the boundary of the manifold domain? The `is_zero` branch in `Randers.metric_fn` ([src/ham/geometry/zoo.py](src/ham/geometry/zoo.py#L108)) is untested.

3. **Parameter gradient completeness:** The test only checks `w_net` gradients. Is it intentional to skip `h_net` gradient verification, or was this an oversight?

4. **Berwald connection test:** Given that `NeuralRanders` will be used with the Berwald parallel transport machinery (§ 3), should there be a test verifying that ${}^B\Gamma^i_{jk} = \partial^2 G^i / \partial v^j \partial v^k$ is finite and well-conditioned for the learned metric?
