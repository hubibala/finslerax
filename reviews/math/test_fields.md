# Math Review: test_fields

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source File:** `tests/test_fields.py`  
**Reviewed Against:** `src/ham/sim/fields.py`, `reviews/math/fields.md`

## Summary

**Verdict: Minor Issues.** All mathematical expected values and formulas used in the existing test assertions are analytically correct. The four tested functions (`get_stream_function_flow`, `tilted_rotation`, `lamb_oseen_vortex`, `rankine_vortex`) have valid expected values derived from the standard definitions. However, two of six imported functions (`rossby_haurwitz`, `harmonic_vortices`) are never tested, one tolerance is unjustifiably loose, and several important mathematical edge cases and geometric identities are absent from the test suite.

---

## Formula-by-Formula Audit

### 1. `test_stream_function_flow` (lines 16–25)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5; `reviews/math/fields.md` § 1.
- **Implementation:**
  ```python
  def psi(x): return x[2]        # ψ = z
  x = jnp.array([1.0, 0.0, 0.0])
  # Expected: v = [0, 0, 1] × [1, 0, 0] = [0, 1, 0]
  self.assertTrue(jnp.allclose(v, jnp.array([0.0, 1.0, 0.0])))
  ```
- **Verdict:** CORRECT
- **Notes:**
  $\nabla\psi = [0, 0, 1]$. The cross product $\nabla\psi \times \mathbf{x} = [0, 0, 1] \times [1, 0, 0] = [0 \cdot 0 - 1 \cdot 0,\; 1 \cdot 1 - 0 \cdot 0,\; 0 \cdot 0 - 0 \cdot 1] = [0, 1, 0]$. The expected value is exact. The default `jnp.allclose` tolerances ($\text{rtol}=10^{-5}$, $\text{atol}=10^{-8}$) are appropriate for a computation that should be exact to machine precision through JAX autodiff.

---

### 2. `test_tilted_rotation` (lines 27–31)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5; `reviews/math/fields.md` § 2.
- **Implementation:**
  ```python
  flow_fn = tilted_rotation(alpha_deg=0.0)  # axis = [0, 0, 1]
  x = jnp.array([1.0, 0.0, 0.0])
  # Expected: same as stream function test → [0, 1, 0]
  self.assertTrue(jnp.allclose(v, jnp.array([0.0, 1.0, 0.0])))
  ```
- **Verdict:** CORRECT
- **Notes:**
  At `alpha_deg=0.0`, the rotation axis is $[\sin 0, 0, \cos 0] = [0, 0, 1]$, so $\psi(\mathbf{x}) = \hat{a} \cdot \mathbf{x} = z$, reducing to the stream function test. The expected value $[0, 1, 0]$ is correct.

  Note: As flagged in `reviews/math/fields.md` § 2, the source code normalises via `jnp.linalg.norm(axis + 1e-10)` (epsilon added to components, not the norm). This introduces an $O(10^{-10})$ error in the axis direction. The default `jnp.allclose` tolerance is wide enough that this test does **not** detect the bug. This is not a test error per se, but the test provides no protection against the incorrect safe-norm pattern.

---

### 3. `test_lamb_oseen_vortex_2d` (lines 33–46)

- **Spec Reference:** `reviews/math/fields.md` § 5.
- **Literature Reference:** Lamb (1932), *Hydrodynamics*; $v_\theta(r) = \frac{\Gamma}{2\pi r}\left(1 - e^{-r^2/r_c^2}\right)$.
- **Implementation:**
  ```python
  # Parameters: center=[0,0], r_c=1.0, Γ=2π
  # Far field (r=10): v_θ = (2π)/(2π·10) · (1 − e^{−100}) ≈ 0.1
  x_far = jnp.array([10.0, 0.0])
  self.assertAlmostEqual(v_far[1], 0.1, places=1)   # ← NOTE
  self.assertAlmostEqual(v_far[0], 0.0, places=5)
  # Near center (r=1e-5): |v| → 0
  self.assertTrue(jnp.linalg.norm(v_near) < 1e-3)
  ```
- **Verdict:** CORRECT (expected values); NOTE (tolerance)
- **Notes:**
  1. **Far-field expected value:** At $r = 10$, $v_\theta = \frac{2\pi}{2\pi \cdot 10}(1 - e^{-100}) = 0.1 \cdot (1 - 3.7 \times 10^{-44}) \approx 0.1$ to machine precision. The expected value $0.1$ is correct. ✓
  2. **Near-center expected value:** At $r = 10^{-5}$, including the $10^{-10}$ regularisation on $r^2$: $r_{\text{sq}} \approx 2 \times 10^{-10}$, so $1 - e^{-r_{\text{sq}}/r_c^2} \approx 2 \times 10^{-10}$, giving $v_\theta \approx 1.4 \times 10^{-5} \ll 10^{-3}$. Correct. ✓
  3. **Tolerance issue:** See NOTE finding below.

---

### 4. `test_rankine_vortex_2d` (lines 48–62)

- **Spec Reference:** `reviews/math/fields.md` § 6.
- **Literature Reference:** $v_\theta = \Gamma r / (2\pi r_c^2)$ for $r \le r_c$; $v_\theta = \Gamma / (2\pi r)$ for $r > r_c$.
- **Implementation:**
  ```python
  # Parameters: center=[0,0], r_c=2.0, Γ=4π
  # Inside core (r=1): v_θ = 4π·1/(2π·4) = 0.5
  self.assertAlmostEqual(v_in[1], 0.5, places=5)
  # Outside core (r=4): v_θ = 4π/(2π·4) = 0.5
  self.assertAlmostEqual(v_out[1], 0.5, places=5)
  ```
- **Verdict:** CORRECT
- **Notes:**
  1. **Inside core:** $v_\theta = \frac{\Gamma r}{2\pi r_c^2} = \frac{4\pi \cdot 1}{2\pi \cdot 4} = 0.5$. At $\mathbf{x} = [1, 0]$: $v_x = -v_\theta \cdot 0 / 1 = 0$, $v_y = v_\theta \cdot 1 / 1 = 0.5$. ✓
  2. **Outside core:** $v_\theta = \frac{\Gamma}{2\pi r} = \frac{4\pi}{2\pi \cdot 4} = 0.5$. At $\mathbf{x} = [4, 0]$: $v_x = 0$, $v_y = 0.5$. ✓
  3. The `places=5` tolerance ($|x - x_0| < 5 \times 10^{-6}$) is appropriate: the $10^{-10}$ regularisation on $r$ introduces only $O(10^{-10})$ perturbation.

---

## Findings

### NOTE — Overly loose tolerance in Lamb-Oseen far-field test
`tests/test_fields.py:41`

The assertion `self.assertAlmostEqual(v_far[1], 0.1, places=1)` requires only $|v_y - 0.1| < 0.05$. At $r = 10$ with $\Gamma = 2\pi$ and $r_c = 1$, the exact value is $v_y = 0.1 \cdot (1 - e^{-100}) = 0.1$ to machine precision (the $10^{-10}$ regularisation on $r^2$ has negligible effect). Using `places=1` is ~4 orders of magnitude looser than necessary and could mask a sign error or factor-of-2 bug that still lands within $\pm 0.05$.

**Recommended Action:** Tighten to `places=5` to match the precision used in the Rankine vortex tests.

---

### WARNING — Imported functions `rossby_haurwitz` and `harmonic_vortices` are never tested
`tests/test_fields.py:9–10`

Both `rossby_haurwitz` and `harmonic_vortices` are imported (lines 9–10) but have no corresponding test methods. These are non-trivial stream functions involving multi-term expressions ($\cos^R\!\phi\,\sin\phi\,\cos(R\lambda)$ for Rossby–Haurwitz; $\cos^m\!\phi\,\sin(l\pi\sin\phi)\,\cos(m\lambda)$ for harmonic vortices). Without tests, regressions in the De Moivre phase computation (`xy_unit ** R`) or the latitudinal polynomials would go undetected.

**Recommended Action:** Add at least one test per function. For `rossby_haurwitz`, verify that at the equator ($z=0$) with $R=4$, the $\omega$-term vanishes and the wave term reduces to $K \cos(R\lambda)$. For `harmonic_vortices`, verify symmetry properties (e.g., vanishing at poles where $\cos^m\!\phi = 0$).

---

### WARNING — No test for tangentiality of spherical flows
`tests/test_fields.py` (global)

The defining geometric property of the stream function construction is that the resulting flow is tangent to the unit sphere: $\mathbf{v} \cdot \mathbf{x} = 0$ for $\|\mathbf{x}\| = 1$. This holds because $\mathbf{v} = \nabla\psi \times \mathbf{x}$ is perpendicular to $\mathbf{x}$ by the cross product. No test asserts this identity. A bug that accidentally adds a radial component (e.g., using `+` instead of `cross`) would not be caught by the existing single-point value checks.

**Recommended Action:** For each spherical flow test, add `self.assertAlmostEqual(float(jnp.dot(v, x)), 0.0, places=10)` at a selection of test points including non-axis points (e.g., $[1/\sqrt{3}, 1/\sqrt{3}, 1/\sqrt{3}]$).

---

### WARNING — Missing edge case: Rankine vortex boundary continuity at $r = r_c$
`tests/test_fields.py:48–62`

The Rankine vortex is piecewise-defined with a junction at $r = r_c$. Mathematical continuity requires both branches to agree: $\frac{\Gamma r_c}{2\pi r_c^2} = \frac{\Gamma}{2\pi r_c}$. The test only checks one interior point ($r = 1$) and one exterior point ($r = 4$) but does not test the boundary itself. A sign error or missing factor in one branch could produce correct values far from $r_c$ while breaking continuity at the junction.

**Recommended Action:** Add a test at $r = r_c$ (e.g., $\mathbf{x} = [2.0, 0.0]$) and verify $v_\theta = \Gamma / (2\pi r_c) = 4\pi / (2\pi \cdot 2) = 1.0$.

---

### NOTE — No test for vortex center ($r = 0$) behaviour
`tests/test_fields.py` (global)

Neither `test_lamb_oseen_vortex_2d` nor `test_rankine_vortex_2d` tests the exact center $\mathbf{x} = \text{center}$. For the Lamb-Oseen vortex, the near-center test uses $r = 10^{-5}$ and checks $\|\mathbf{v}\| < 10^{-3}$, which is a loose bound. At $\mathbf{x} = \text{center}$ itself, the $10^{-10}$ regularisation means $r = \sqrt{10^{-10}} \approx 3.16 \times 10^{-6}$, and the velocity should still be near zero. For the Rankine vortex (rigid-body core), $v_\theta \propto r \to 0$ at center. Testing $\mathbf{x} = [0, 0]$ exactly exercises the regularisation path.

**Recommended Action:** Add explicit center tests: `v_center = flow_fn(jnp.array([0.0, 0.0]))` and assert `jnp.allclose(v_center, jnp.zeros(2), atol=1e-3)`.

---

### NOTE — No test for divergence-free property of spherical flows
`tests/test_fields.py` (global)

The stream-function construction guarantees divergence-free flow on $S^2$. While this is an analytical guarantee rather than a numerical invariant, verifying $\nabla \cdot \mathbf{v} \approx 0$ (via finite differences or JAX divergence) at test points would provide a regression guard against implementation errors that break the cross-product structure.

**Recommended Action:** Low priority. Consider adding a numerical divergence check at 2–3 points for at least one spherical flow.

---

### STRONG — Analytical expected values for Rankine vortex
`tests/test_fields.py:48–62`

The test comments explicitly derive the expected values from the formula ($v_\theta = \Gamma r / (2\pi r_c^2) = 0.5r$ inside, $v_\theta = \Gamma / (2\pi r) = 2/r$ outside), making the test self-documenting and easy to audit. The choice of parameters ($\Gamma = 4\pi$, $r_c = 2$) produces clean rational expected values. This is exemplary practice.

---

### STRONG — Lamb-Oseen near-center test verifies regularisation
`tests/test_fields.py:44–46`

Testing near $r = 0$ verifies that the implementation correctly regularises the $1/r$ singularity (unlike a naïve point vortex which would blow up). This exercises the $r^2 + 10^{-10}$ epsilon path in the source code.

---

## Open Questions

1. **Should the tilted-rotation test detect the `norm(axis + eps)` bug?** At `alpha_deg=0`, the perturbation is $O(10^{-10})$ and invisible to `jnp.allclose`. A test at a non-trivial angle (e.g., `alpha_deg=45`) with a tighter tolerance, or a direct assertion that `jnp.linalg.norm(flow_fn(x))` matches the expected angular velocity magnitude, would provide better coverage. However, since the source-code bug itself is benign (the axis always has unit norm), this is low priority.

2. **Test isolation for `get_stream_function_flow`:** The only direct test of `get_stream_function_flow` uses a linear $\psi$, which produces a constant $\nabla\psi$ (no second derivatives involved). A test with a nonlinear $\psi$ (e.g., $\psi = x^2 + y^2$) would exercise JAX's autodiff through a more complex computational graph.

3. **Differentiability as a wind field:** As noted in `reviews/math/fields.md` § Open Question 3, the Rankine vortex has a discontinuous first derivative at $r = r_c$. If these fields feed into Finsler energy functions requiring 3rd-order differentiation (Berwald connection, `spec/MATH_SPEC.md` § 3.1), the test suite should include a smoke test for differentiability (e.g., `jax.grad` of the flow at a point near $r_c$).
