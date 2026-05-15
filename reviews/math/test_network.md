# Math Review: test_network.py

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**File:** `tests/test_network.py`

## Summary

**Verdict: Minor Issues.**  
The test file exercises the two principal network modules (`VectorField`, `PSDMatrixField`) for correct output shapes and the mathematically critical SPD property. The positive-definiteness test is structurally sound but uses a boundary tolerance that can spuriously fail. Two additional mathematical properties that the spec demands are untested: the lower bound $\lambda_{\min}(G) \geq \epsilon = 10^{-4}$ and the finiteness of `PSDMatrixField` output. The Fourier-features test lacks a meaningful assertion.

---

## Formula-by-Formula Audit

### 1. `test_vector_field_shape` — Shape and finiteness of $W(x)$

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 — the wind field $W^i(x) \in \mathbb{R}^D$.
- **Implementation** (`tests/test_network.py:13–18`):
  ```python
  v = vf(x)
  self.assertEqual(v.shape, (self.dim,))
  self.assertTrue(jnp.isfinite(v).all())
  ```
- **Verdict:** CORRECT
- **Notes:**
  Shape assertion correctly verifies $W: \mathbb{R}^D \to \mathbb{R}^D$. The `isfinite` check guards against NaN/Inf from the Fourier embedding or MLP initialisation. Both are appropriate for a smoke test.

---

### 2. `test_psd_matrix_properties` — Symmetry and positive-definiteness of $G(x)$

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1, condition 3 — the fundamental tensor $g_{ij}$ must be symmetric and positive definite.
- **Source Reference:** `src/ham/nn/networks.py:95–98` — constructs $G = AA^T + 10^{-4}I$.

#### 2a. Shape check

- **Implementation** (`tests/test_network.py:26`):
  ```python
  self.assertEqual(G.shape, (self.dim, self.dim))
  ```
- **Verdict:** CORRECT

#### 2b. Symmetry check

- **Implementation** (`tests/test_network.py:29–30`):
  ```python
  diff_sym = jnp.max(jnp.abs(G - G.T))
  self.assertLess(diff_sym, 1e-6, "Matrix is not symmetric")
  ```
- **Verdict:** CORRECT
- **Notes:**
  The construction $G = AA^T + \epsilon I$ is exactly symmetric analytically. A tolerance of $10^{-6}$ is appropriate to absorb floating-point rounding (32-bit matmul accumulation errors are typically $\mathcal{O}(D \cdot \varepsilon_{\text{f32}}) \approx 3 \times 10^{-7}$ for $D = 3$).

#### 2c. Positive-definiteness check

- **Implementation** (`tests/test_network.py:33–35`):
  ```python
  eigs = jnp.linalg.eigvalsh(G)
  min_eig = jnp.min(eigs)
  self.assertGreater(min_eig, 0.0, "Matrix is not positive definite")
  ```
- **Verdict:** WARNING
- **Notes:**
  The test asserts $\lambda_{\min}(G) > 0$, but the construction guarantees the stronger bound $\lambda_{\min}(G) \geq \epsilon = 10^{-4}$:
  $$
  v^T G\, v = \|A^T v\|^2 + \epsilon\,\|v\|^2 \;\geq\; \epsilon\,\|v\|^2 > 0
  $$
  Testing against the hard zero boundary is fragile: in float32, computed eigenvalues of a near-singular $AA^T$ could land at $\sim 10^{-4} - \mathcal{O}(\varepsilon_{\text{f32}})$, and rounding could place $\lambda_{\min}$ just below the true analytic value. While this is unlikely to cause a spurious failure for $\epsilon = 10^{-4}$ and $D = 3$, the test should verify the **architectural guarantee** by asserting the tighter bound:
  ```python
  self.assertGreater(min_eig, 1e-4 / 2, "Min eigenvalue below epsilon floor")
  ```
  This would also catch any accidental removal of the $\epsilon I$ regulariser during refactoring.

  **Recommended Action:** Replace `assertGreater(min_eig, 0.0)` with `assertGreater(min_eig, 5e-5)` to test the actual $\epsilon$-floor guarantee from the source (`src/ham/nn/networks.py:98`).

#### 2d. Missing finiteness check

- **Verdict:** WARNING
- **Notes:**
  Unlike `test_vector_field_shape`, this test does not assert `jnp.isfinite(G).all()`. If the MLP produces NaN (e.g., due to extreme initialisation), $G$ would contain NaN entries that pass the symmetry check (NaN comparisons evaluate to False) and may pass the eigenvalue check depending on backend behaviour. Adding a finiteness guard before the eigenvalue decomposition would make the test more robust.

  **Recommended Action:** Add `self.assertTrue(jnp.isfinite(G).all(), "G contains non-finite entries")` at `tests/test_network.py:27` (after the shape check, before symmetry).

---

### 3. `test_fourier_features` — Fourier embedding functional test

- **Spec Reference:** Not directly in `MATH_SPEC.md`; tests the RFF building block from `src/ham/nn/networks.py:6–19`.
- **Literature Reference:** Rahimi & Recht, NeurIPS 2007 — $z(x) = [\cos(Bx),\, \sin(Bx)]$.
- **Implementation** (`tests/test_network.py:38–48`):
  ```python
  vf_base = VectorField(3, 16, 2, self.key, use_fourier=False)
  vf_four = VectorField(3, 16, 2, self.key, use_fourier=True)
  ...
  self.assertEqual(out_base.shape, out_four.shape)
  # They shouldn't be identical (random weights differ, but functional path differs too)
  ```
- **Verdict:** WARNING
- **Notes:**
  The comment states the outputs "shouldn't be identical," but no assertion enforces this. The test only checks shape equality, which is trivially satisfied since both networks have `out_size=dim`. The Fourier path changes the MLP's `in_size` (from $D$ to $2 \lfloor H/2 \rfloor$) and adds a non-linear embedding, but the test never verifies that the embedding actually transforms the input or that the two paths produce meaningfully different outputs.

  Without an assertion, this test cannot detect a regression where the Fourier embedding is silently bypassed (e.g., if a refactoring accidentally sets `self.embedding = None` regardless of `use_fourier`).

  **Recommended Action:** Add an explicit assertion:
  ```python
  self.assertFalse(jnp.allclose(out_base, out_four),
                   "Fourier and non-Fourier paths produce identical output")
  ```

---

### 4. Missing test: multi-point consistency of PSD field

- **Verdict:** NOTE
- **Notes:**
  The PSD test evaluates $G(x)$ at a single point $x = \mathbf{1}$. The mathematical requirement (`spec/MATH_SPEC.md` § 1.1, condition 3) is that $g_{ij}(x, v) \succ 0$ for **all** $x$. A stronger test would evaluate PSD-ness at multiple points (including the origin and large-norm inputs) to catch input-dependent degeneracies. This is a coverage concern, not a correctness bug.

---

### 5. Missing test: interaction with auto-differentiation

- **Verdict:** NOTE
- **Notes:**
  The Berwald connection requires 3rd-order derivatives through the energy, which passes through $G(x)$. The `tanh` activation ensures $C^\infty$ smoothness, but no test verifies that `jax.grad` or `jax.hessian` through `PSDMatrixField` actually returns finite values. A test of the form:
  ```python
  grad_G = jax.jacobian(psd_net)(x)
  self.assertTrue(jnp.isfinite(grad_G).all())
  ```
  would guard against AD failures (e.g., shape mismatches, tracing errors) that could silently break downstream geodesic computation.

---

## Open Questions

1. **Tolerance choice for symmetry ($10^{-6}$):** Is the test ever run in float64 mode (via `jax.config.update("jax_enable_x64", True)`)? If so, the tolerance could be tightened to $10^{-14}$; if not, $10^{-6}$ is appropriate for float32.
2. **Cholesky vs. Gram parametrisation:** The source uses $G = AA^T + \epsilon I$ (reviewed in `reviews/math/networks.md` § 3c as WARNING). The test does not exercise the alternative Cholesky path. If a Cholesky variant is added, tests should verify both parametrisations.
3. **Norm constraint for Randers validity:** The `VectorField` test does not check $\|W(x)\|_h < 1$, which is the causality constraint from `spec/MATH_SPEC.md` § 5. This is correctly enforced downstream in the metric class, but a test verifying the end-to-end constraint (network → Randers metric → valid $F > 0$) would strengthen coverage.
