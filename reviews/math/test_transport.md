# Math Review: `test_transport.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** [tests/test_transport.py](tests/test_transport.py)  
**Depends on:** [reviews/math/transport.md](reviews/math/transport.md) (source-level review of `transport.py`)

## Summary

The test file covers the three tiers of the geometric hierarchy (Euclidean, Riemannian, Randers) and checks qualitatively correct properties: trivial flat transport, norm preservation, non-metricity, velocity dependence, and holonomy. However, **two tests are mathematically degenerate** (the Berwald connection is identically zero in the chosen setups, so the tests exercise only the `to_tangent` projection and not the $\Gamma^i_{jk}$ computation), **one test contains a wrong holonomy formula** that is masked by a trigonometric identity in the comparison, and **one test conflates the velocity-dependence of $\Gamma$ with the velocity factor already present in the transport ODE**. No test exercises the Berwald connection in a regime where $\Gamma \neq 0$ and where an analytical reference value is available.

**Overall Verdict: Minor Issues (no incorrect pass/fail, but several tests are weaker than claimed).**

---

## Formula-by-Formula Audit

### 1. `test_euclidean_flat_invariance` (lines 46–62)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table Row 1 — Euclidean: $G^i = 0$, $\Gamma^i_{jk} = 0$.
- **Implementation:**
  ```python
  metric = Euclidean(self.plane)           # F(x,v) = |v|
  vecs = berwald_transport(metric, path_x, path_v, vec_start)
  np.testing.assert_allclose(vecs, expected, atol=1e-5)
  ```
- **Verdict:** CORRECT
- **Notes:**

  For $F(x,v) = \|v\|$, the energy is $E = \frac{1}{2}\|v\|^2$, which is independent of $x$. Hence $\nabla_x E = 0$ and the spray $G^i = 0$. The Berwald connection $\Gamma^i_{jk} = \partial^2 G^i / \partial v^j \partial v^k = 0$. The transport ODE reduces to $\dot{X}^i = 0$, so $X(t) = X(0) = (1, 0)$ for all $t$. The assertion is mathematically exact (up to floating-point).

  The tolerance `atol=1e-5` is appropriate: the Euler integrator's $dx = -\Gamma \cdot v \cdot X = 0$ introduces no discretization error, so the only error source is floating-point arithmetic in the double `jacfwd` (which should return exact zeros for a quadratic energy). With float64 enabled, `1e-5` is conservative. ✓

---

### 2. `test_riemannian_sphere_isometry` (lines 64–94)

- **Spec Reference:** `spec/MATH_SPEC.md` § 3.1 — "Spray-Induced: If the Spray is quadratic in $v$ (Riemannian case), $\Gamma$ depends only on $x$ (it becomes the Levi-Civita connection)."
- **Spec Reference:** `spec/MATH_SPEC.md` § 3.2 — Berwald parallel transport equation.
- **Literature Reference:** do Carmo, *Riemannian Geometry* (1992), Proposition 2.6 (Gauss equation: $\nabla^M_X Y = \text{proj}_{TM}\!\left(\bar\nabla_X Y\right)$).
- **Implementation:** [test_transport.py:72–74](tests/test_transport.py#L72-L74)
  ```python
  def sphere_metric(x):
      return jnp.eye(3)
  metric = Riemannian(self.sphere, sphere_metric)
  ```
- **Verdict:** WARNING
- **Notes:**

  **Issue A — Zero connection.** The metric tensor $G(x) = I_3$ is position-independent, making the energy $E(x,v) = \frac{1}{2}\|v\|^2$ identical to the Euclidean case. The spray is $G^i = 0$ and the Berwald connection is $\Gamma^i_{jk} = 0$ everywhere. The test therefore exercises **zero** non-trivial connection coefficients. Any "transport" that occurs is entirely due to the `to_tangent` projection in [transport.py:51](src/ham/geometry/transport.py#L51), not the Berwald ODE.

  By the Gauss equation, the `to_tangent` projection of the flat ambient connection does approximate the Levi-Civita connection of the induced metric in the limit $\Delta t \to 0$. So the test's norm-preservation check is not mathematically wrong — but it is testing the projection heuristic, not the connection computation.

  **Issue B — Degenerate test vector.** The initial vector is $X_0 = (0, 1, 0)$ and the path lies in the $xz$-plane. Since $X_0$ is orthogonal to the path plane, it is tangent to $S^2$ at every point along this path: $\langle X_0, \gamma(t)\rangle = 0$ for all $t$, because $\gamma^y(t) = 0$. The `to_tangent` projection never changes the vector. Both the norm and tangency assertions hold trivially — the vector is identically $(0, 1, 0)$ at every step.

  **Recommended Action:** To meaningfully test Riemannian transport, either:
  (a) Use an intrinsic coordinate chart where $G(x)$ is position-dependent and $\Gamma \neq 0$, or
  (b) Choose a non-degenerate initial vector (e.g., $(1, 0, 0)$) and a path that is not a great circle in a symmetry plane — this forces the projected vector to rotate in the tangent plane, exercising the projection mechanism.

---

### 3. `test_randers_norm_drift` (lines 96–148)

- **Spec Reference:** `spec/MATH_SPEC.md` § 3.1 — "Non-Metric: In general Finsler spaces, the Berwald connection does *not* preserve the Finsler norm ($Dg \neq 0$)."
- **Literature Reference:** Bao–Chern–Shen (2000), §10.4 (non-metricity of the Berwald connection for non-Berwaldian Finsler spaces).
- **Implementation:** [test_transport.py:127–130](tests/test_transport.py#L127-L130)
  ```python
  norms_randers = jax.vmap(metric.metric_fn)(path_x, vecs_randers)
  ...
  self.assertNotAlmostEqual(initial_norm, final_norm, places=3, ...)
  ```
- **Verdict:** CORRECT
- **Notes:**

  The assertion that Berwald transport does not preserve the Finsler norm in a Randers space is correct. For a Randers metric with position-dependent wind $W(x) = (0.5\,x_2, 0)$, the spray is non-quadratic in $v$ (per `spec/MATH_SPEC.md` § 4 table), so the Berwald connection is velocity-dependent and non-metric.

  The `places=3` tolerance (checks that norms differ by more than $5 \times 10^{-4}$) is reasonable for a wind gradient of $0.5$ over a path of unit length.

  **Minor observation:** The norm drift has two contributing effects: (1) the transported vector coordinates $X^i$ change via the Berwald ODE, and (2) the metric function $F(x, v)$ itself changes because the wind varies along the path. Even if $X^i$ were held constant (zero connection), the Finsler norm $F(\gamma(t), X)$ would still change because $W(\gamma(t))$ changes. The test conflates these two effects. A stronger test would compare the Berwald-transported vector against coordinate-constant transport and verify that they differ.

---

### 4. `test_randers_velocity_dependence` (lines 150–183)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table — Randers: $\Gamma^i_{jk}(x, v)$ depends on $v$.
- **Implementation:** [test_transport.py:170–176](tests/test_transport.py#L170-L176)
  ```python
  path_v2 = path_v1 * 2.0
  vecs_1 = berwald_transport(metric, path_x1, path_v1, vec_start)
  vecs_2 = berwald_transport(metric, path_x1, path_v2, vec_start)
  diff = jnp.linalg.norm(vecs_1[-1] - vecs_2[-1])
  self.assertGreater(diff, 1e-4, ...)
  ```
- **Verdict:** WARNING
- **Notes:**

  **Issue — Confounded comparison.** The parallel transport ODE is:

  $$\frac{dX^i}{dt} = -\Gamma^i_{jk}(\gamma, \dot\gamma)\;\dot\gamma^j\;X^k$$

  When $v \to 2v$ with the *same* discrete path and *same* `dt = 1/N`, the update at each step becomes:

  $$\Delta X^i = -\Gamma^i_{jk}(x, 2v)\;(2v^j)\;X^k\;\Delta t$$

  This differs from the original even if $\Gamma$ is velocity-**independent** (Riemannian case with $\Gamma \neq 0$), because the factor $2v^j$ in the contraction doubles the update magnitude while `dt` stays fixed. In correct continuous parameterization, doubling the speed halves the traversal time, and these effects cancel for velocity-independent $\Gamma$. But the discrete implementation uses a fixed `dt = 1/N` regardless of speed, breaking this cancellation.

  The test passes in the Randers case for two reasons: (1) $\Gamma(x, 2v) \neq \Gamma(x, v)$ (genuine velocity dependence), AND (2) the $2v^j$ factor with unchanged `dt` (an artifact). The test **cannot distinguish** between these.

  The test does NOT produce a false positive for the trivial case $\Gamma = 0$ (Euclidean or flat Riemannian), because then both transports give $X(t) = X(0)$ regardless of $v$. But it would produce a false positive for any curved Riemannian metric where $\Gamma(x) \neq 0$ — the test would claim "velocity dependence" when the difference is actually due to the `dt` artifact.

  **Recommended Action:** To isolate the velocity-dependence of $\Gamma$, compare Berwald transport against a reference Riemannian transport *on the same curved path*. If the Riemannian result is speed-invariant (after correcting `dt`) but the Randers result is not, the difference is attributable to $\Gamma$'s velocity dependence.

---

### 5. `test_sphere_holonomy` (lines 185–228)

- **Spec Reference:** Implicit — holonomy is a consequence of curvature, not directly stated in `spec/MATH_SPEC.md`.
- **Literature Reference:** do Carmo, *Riemannian Geometry* (1992), Chapter 4, §4; Berry phase on $S^2$. The holonomy angle for parallel transport around a latitude circle at colatitude $\theta$ is:
  $$\Omega = \iint_D K\;dA = 2\pi(1 - \cos\theta)$$
  where $K = 1$ is the Gaussian curvature and $D$ is the spherical cap enclosed by the latitude.
- **Implementation:** [test_transport.py:218](tests/test_transport.py#L218)
  ```python
  expected_angle = 2 * jnp.pi * jnp.cos(theta)
  ```
- **Verdict:** WARNING
- **Notes:**

  **Issue A — Wrong holonomy formula.** The code sets `expected_angle = 2π cos θ`. The correct holonomy angle is $\Omega = 2\pi(1 - \cos\theta)$. For $\theta = \pi/4$:

  | Formula | Value (rad) |
  |---------|-------------|
  | $2\pi\cos(\pi/4)$ (code) | $\approx 4.443$ |
  | $2\pi(1 - \cos(\pi/4))$ (correct) | $\approx 1.840$ |

  The test passes despite this error because the comparison uses `cos(angle) ≈ cos(expected\_angle)`, and these two formulas are complementary modulo $2\pi$:

  $$2\pi\cos\theta + 2\pi(1 - \cos\theta) = 2\pi$$

  $$\Rightarrow \cos\bigl(2\pi\cos\theta\bigr) = \cos\bigl(2\pi - 2\pi(1-\cos\theta)\bigr) = \cos\bigl(2\pi(1-\cos\theta)\bigr)$$

  The cosine comparison is therefore insensitive to this error. However, any future test that uses `expected_angle` directly (e.g., for sign-aware rotation checks) would get the wrong reference value.

  **Issue B — Self-contradictory docstring.** The docstring (lines 189–192) first mentions "$2\pi(1 - \cos\theta)$" and then says "the angle shift in the tangent plane should be $2\pi\cos\theta$." These quantities are complements, not equals.

  **Issue C — Same zero-connection problem as Test 2.** The metric is $G(x) = I_3$ (ambient Euclidean), giving $\Gamma^i_{jk} = 0$. The holonomy arises entirely from the `to_tangent` projection, not from non-zero connection coefficients. The test therefore verifies the projection-based transport mechanism, not the Berwald connection.

  **Issue D — Loose tolerance.** The `atol=1e-1` tolerance corresponds to $\pm 0.1$ on a cosine value (range $[-1, 1]$). This masks:
  - The `dt = 1/N` vs. `1/(N-1)` systematic error (see [reviews/math/transport.md](reviews/math/transport.md), Finding #3).
  - The wrong-base-point projection error (ibid., Finding #4).
  - First-order Euler drift.

  **Recommended Action:**
  1. Correct the formula to `expected_angle = 2 * jnp.pi * (1 - jnp.cos(theta))`.
  2. Fix the docstring to state the correct formula.
  3. Compare the angle directly (`np.testing.assert_allclose(angle, expected_angle, atol=...)`) rather than through `cos()`, once the formula is correct. This eliminates the sign/complement ambiguity.
  4. Tighten the tolerance by increasing $N$ and/or using a higher-order integrator.

---

## Missing Test Coverage

### 6. No test with analytically non-zero $\Gamma^i_{jk}$

- **Verdict:** WARNING
- **Notes:**

  All tests that involve curved manifolds (tests 2 and 5) use the ambient Euclidean metric $G(x) = I_3$, which produces $\Gamma^i_{jk} = 0$. The Berwald connection computation (`christoffel_symbols()`) is never exercised with non-trivial output in any test.

  **Recommended Action:** Add a test using an intrinsic-coordinate Riemannian metric where $\Gamma^i_{jk}(x)$ is analytically known (e.g., the Poincaré half-plane metric $ds^2 = (dx^2 + dy^2)/y^2$, where the connection coefficients are $\Gamma^1_{12} = \Gamma^1_{21} = -1/y$, $\Gamma^2_{11} = 1/y$, $\Gamma^2_{22} = -1/y$). Verify that `christoffel_symbols(x, v)` matches these values.

### 7. No convergence test

- **Verdict:** NOTE
- **Notes:**

  None of the tests verify that the transport error decreases as $N$ increases. For an Euler integrator, the global error should be $O(1/N)$. A convergence test (e.g., transport on the sphere with $N = 50, 100, 200, 400$ and check that the holonomy error halves each time) would validate both the mathematical formulation and the integrator order.

### 8. No test for $v = 0$ edge case

- **Verdict:** NOTE
- **Notes:**

  `spec/MATH_SPEC.md` § 6.1 notes that the Berwald connection involves 3rd derivatives of the energy and is "highly sensitive to the singularity at $v = 0$." No test checks the behavior of `berwald_transport` when $v$ passes through or near zero. This is relevant for paths that slow down (e.g., approaching a fixed point in the learned vector field).

### 9. No test for torsion-freeness ($\Gamma^i_{jk} = \Gamma^i_{kj}$)

- **Verdict:** NOTE
- **Notes:**

  The spec states that the Berwald connection is torsion-free (symmetric in $j, k$). No test verifies that `christoffel_symbols(x, v)` produces a tensor with `gamma[i,j,k] == gamma[i,k,j]`. This is a structural property that should hold by construction (Schwarz's theorem on the double `jacfwd`), but an explicit test would guard against future refactoring errors.

---

## Summary of Findings

| # | Severity | Location | Issue |
|---|---|---|---|
| 1 | CORRECT | [test_transport.py:46–62](tests/test_transport.py#L46-L62) | Euclidean flat transport — mathematically exact, appropriate tolerance |
| 2 | **WARNING** | [test_transport.py:64–94](tests/test_transport.py#L64-L94) | Sphere isometry test uses zero connection ($G(x)=I_3$) and a degenerate vector — doesn't exercise $\Gamma$ computation |
| 3 | CORRECT | [test_transport.py:96–148](tests/test_transport.py#L96-L148) | Randers norm drift — correct assertion of non-metricity |
| 4 | **WARNING** | [test_transport.py:150–183](tests/test_transport.py#L150-L183) | Velocity-dependence test confounded by fixed `dt` — cannot isolate $\Gamma(x,v)$ dependence on $v$ from the $v^j$ factor in the ODE |
| 5 | **WARNING** | [test_transport.py:218](tests/test_transport.py#L218) | Holonomy formula $2\pi\cos\theta$ is wrong; correct is $2\pi(1-\cos\theta)$. Test passes due to $\cos(2\pi\cos\theta) = \cos(2\pi(1-\cos\theta))$ |
| 6 | **WARNING** | [test_transport.py:189–192](tests/test_transport.py#L189-L192) | Docstring contradicts itself on the holonomy formula |
| 7 | **WARNING** | (coverage gap) | No test with analytically non-zero $\Gamma^i_{jk}$ — the core connection computation is untested |
| 8 | NOTE | (coverage gap) | No convergence test to verify integrator order |
| 9 | NOTE | (coverage gap) | No $v = 0$ edge-case test |
| 10 | NOTE | (coverage gap) | No explicit symmetry ($\Gamma^i_{jk} = \Gamma^i_{kj}$) test |

---

## Open Questions

1. **Are the sphere tests intended to test the Berwald connection or the projection mechanism?** If the latter, the tests should be documented as such. If the former, they need a non-trivial metric where $\Gamma \neq 0$.

2. **Should the velocity-dependence test be a differential test?** That is, compare Randers transport vs. Riemannian transport on the same curved geometry, rather than comparing Randers transport at two different speeds.

3. **What tolerance is acceptable for the holonomy test?** The current `atol=1e-1` on cosine values corresponds to roughly $\pm 6°$ angular error. Is this adequate for downstream use (e.g., stability analysis in the VAE pipeline)?
