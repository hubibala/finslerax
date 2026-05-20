# Math Review: zoo

**Reviewer:** Math Reviewer Agent
**Date:** 2025-05-15
**Spec Version:** 1.1.0 (Berwald Revision)
**Source File:** `src/ham/geometry/zoo.py`

---

## Summary

The metric zoo implements four Finsler metric classes — `Euclidean`, `Riemannian`, `Randers`, and `DiscreteRanders`. The core Zermelo–Randers formula is **correct** and matches both `spec/MATH_SPEC.md` § 5 and the standard Bao–Robles–Shen derivation. The Euclidean and Riemannian specialisations are standard. Two **WARNING**-level issues are identified: (1) a positive-definiteness enforcement gap in the Randers class that does not guarantee $H \succ 0$, and (2) a minor homogeneity violation from the additive $\epsilon$ inside the discriminant square root. One **CRITICAL** issue is absent; all formulas are analytically correct at the symbolic level.

---

## Formula-by-Formula Audit

### 1. `Euclidean.metric_fn` (line 17)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table row "Euclidean": $F(x,v) = \sqrt{v^T v}$
- **Literature Reference:** Standard definition; any Finsler geometry textbook.
- **Implementation:**
  ```python
  def metric_fn(self, x, v):
      return safe_norm(v)          # sqrt(max(sum(v²), 1e-12))
  ```
- **Verdict:** OK
- **Notes:** `safe_norm` computes $\sqrt{\max(\|v\|^2,\,\epsilon)}$ with $\epsilon = 10^{-12}$ (from `ham/utils/math.py`). This matches the Euclidean column of the spec table. The $\epsilon$-floor is consistent with `spec/MATH_SPEC.md` § 6.1 and breaks 1-homogeneity only for $\|v\| < 10^{-6}$, which is acceptable for gradient safety.

---

### 2. `Riemannian.metric_fn` (lines 31–35)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table row "Riemannian": $F(x,v) = \sqrt{v^T g(x)\, v}$
- **Literature Reference:** Standard Riemannian norm; e.g., do Carmo, *Riemannian Geometry*, Ch. 1.
- **Implementation:**
  ```python
  G_x = self.g_net(x)
  G_x = 0.5 * (G_x + G_x.T)          # symmetrize
  quad = jnp.dot(v, jnp.dot(G_x, v))  # v^T G v
  return jnp.sqrt(jnp.maximum(quad, 1e-12))
  ```
- **Verdict:** OK
- **Notes:**
  - Symmetrisation $G \leftarrow \frac{1}{2}(G + G^T)$ is correct and necessary since the antisymmetric part is annihilated by the quadratic form anyway.
  - The `jnp.maximum(quad, 1e-12)` clamp prevents `sqrt` of negative values but silently masks non-positive-definite $G$. If `g_net` returns a matrix with a negative eigenvalue, $v^T G v < 0$ is possible for some $v$; the clamp hides the failure rather than signalling it. This is a design choice, not a formula error.
  - No explicit positive-definiteness enforcement exists. Unlike the `Randers` class (which clamps diagonals), `Riemannian` relies entirely on `g_net` producing PD output. Acceptable if `g_net` is constructed to be PD by architecture (e.g., $L L^T + \epsilon I$).

---

### 3. `Randers._get_zermelo_data` — Wind Squasher (lines 82–106)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5: "Wind field $W^i(x)$ with constraint $\|W\|_h < 1$."
- **Literature Reference:** Bao–Robles–Shen, "Zermelo navigation on Riemannian manifolds," *J. Diff. Geom.* 66 (2004), 377–435.
- **Implementation (squash logic):**
  ```python
  w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))      # ||W||_h^2
  w_norm    = jnp.sqrt(jnp.maximum(w_norm_sq, 1e-8))   # ||W||_h
  max_speed = 1.0 - self.epsilon                        # 1 - ε
  squash_factor = (max_speed * jnp.tanh(w_norm)) / (w_norm + 1e-8)
  W_safe = W_raw * squash_factor
  ```
- **Verdict:** OK
- **Notes:**
  - After squashing, $\|W_{\text{safe}}\|_h = (1-\epsilon)\tanh(\|W_{\text{raw}}\|_h)$, which is strictly less than $1 - \epsilon < 1$ for all inputs. This guarantees the Zermelo causality condition $\|W\|_h < 1$ is never violated. ✓
  - **Monotonicity:** The map $r \mapsto (1-\epsilon)\tanh(r)$ is monotonically increasing, so the squasher preserves the relative ordering of wind strengths. ✓
  - **Faithfulness for small winds:** For $\|W\|_h \ll 1$, $\tanh(r) \approx r$, so $\|W_{\text{safe}}\|_h \approx (1-\epsilon)\|W_{\text{raw}}\|_h \approx \|W_{\text{raw}}\|_h$. The distortion is negligible for well-behaved inputs. ✓

---

### 3a. `Randers._get_zermelo_data` — PD Enforcement of $H$ (lines 84–89)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, implicit: $h_{ij}(x)$ must be a Riemannian metric (positive definite).
- **Implementation:**
  ```python
  H = 0.5 * (H + H.T)                              # symmetrize
  diag = jnp.diag(H)
  diag_safe = jnp.maximum(diag, 0.01)
  H = jnp.where(jnp.eye(..., dtype=bool), jnp.diag(diag_safe), H)
  H = H + 0.005 * jnp.eye(H.shape[-1])
  ```
- **Verdict:** WARNING
- **Notes:**
  - Clamping diagonal entries to $\geq 0.01$ and adding $0.005\,I$ ensures positive diagonal entries but does **not** guarantee positive definiteness. A matrix with diagonal 0.015 and off-diagonal magnitude $\gg 0.015$ can have negative eigenvalues. Example in $\mathbb{R}^2$:
    $$H = \begin{pmatrix} 0.015 & 1 \\ 1 & 0.015 \end{pmatrix}, \quad \det(H) = 0.015^2 - 1 < 0.$$
  - In practice, neural network initialisations and training dynamics make this scenario unlikely, but the guarantee is not mathematically watertight.
  - **Recommended Action:** Replace diagonal clamping with a Cholesky-based construction ($H = LL^T + \epsilon I$ where $L$ is lower-triangular output of the network), or add a spectral clamp $H \leftarrow H + \max(0,\, \epsilon - \lambda_{\min}(H))\,I$.

---

### 4. `Randers.metric_fn` — Zermelo–Randers Formula (lines 108–128)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5:
  $$F(x, v) = \frac{\sqrt{\lambda \|v\|_h^2 + \langle W, v \rangle_h^2} - \langle W, v \rangle_h}{\lambda}$$
  where $\lambda = 1 - \|W\|_h^2$.
- **Literature Reference:** Bao–Robles–Shen (2004), Theorem 1.1. Also: Chern–Shen, *Riemann–Finsler Geometry*, § 12.6.
- **Implementation:**
  ```python
  Hv = jnp.matmul(H, v_safe)                        # H·v
  HW = jnp.matmul(H, W)                              # H·W
  v_sq_h   = jnp.sum(v_safe * Hv, axis=-1)           # v^T H v = ||v||_h^2
  W_dot_v  = jnp.sum(v_safe * HW, axis=-1)           # v^T H W = <W, v>_h
  discriminant = lam * v_sq_h + W_dot_v**2            # λ||v||² + <W,v>²
  cost = (jnp.sqrt(jnp.maximum(discriminant, 0.0) + 1e-9) - W_dot_v) / lam
  ```
- **Verdict:** OK
- **Notes:**
  - **Term-by-term verification:**
    - $\|v\|_h^2 = v^T H v$: computed as `v_safe · (H v_safe)` via elementwise product and sum. ✓
    - $\langle W, v\rangle_h = W^T H v = v^T H W$ (since $H$ is symmetric): computed as `v_safe · (H W)`. ✓
    - Discriminant $= \lambda\|v\|_h^2 + \langle W,v\rangle_h^2$: matches spec. ✓
    - Final formula $(\sqrt{\text{disc}} - \langle W,v\rangle_h) / \lambda$: matches spec. ✓
  - **Sign convention:** With wind $W$ pointing west and velocity $v$ pointing east, $\langle W,v\rangle_h < 0$, making the numerator $\sqrt{\cdot} - (\text{negative}) = \sqrt{\cdot} + |\langle W,v\rangle_h|$. Headwind increases cost. Matches spec § 5 note. ✓
  - **1-Homogeneity:** Under $v \to kv$ ($k > 0$): discriminant scales as $k^2$, $\langle W,v\rangle_h$ scales as $k$, so $F(x, kv) = kF(x,v)$. ✓ (analytically; see WARNING below for numerical caveat).
  - **Non-negativity of $F$:** Since $\sqrt{a + b^2} \geq |b|$, the numerator $\sqrt{\text{disc}} - \langle W,v\rangle_h \geq 0$ always. ✓

---

### 4a. `Randers.metric_fn` — Numerical $\epsilon$ in Square Root (line 125)

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1 (epsilon regularisation).
- **Implementation:**
  ```python
  cost = (jnp.sqrt(jnp.maximum(discriminant, 0.0) + 1e-9) - W_dot_v) / lam
  ```
- **Verdict:** WARNING
- **Notes:**
  - The additive constant $10^{-9}$ inside the square root changes the formula to:
    $$F_\epsilon(x,v) = \frac{\sqrt{\lambda\|v\|_h^2 + \langle W,v\rangle_h^2 + 10^{-9}} - \langle W,v\rangle_h}{\lambda}$$
    This **breaks 1-homogeneity**: $F_\epsilon(x, kv) \neq k\,F_\epsilon(x, v)$ because the $10^{-9}$ term does not scale with $k^2$.
  - The violation is negligible for $\|v\|_h \gg 10^{-4.5} \approx 3 \times 10^{-5}$, which covers all practical use. It is consistent in spirit with the $\epsilon$-regularisation strategy of § 6.1.
  - A more principled fix would follow the spec's pattern and regularise $F$ itself: $F_\epsilon = \sqrt{F^2 + \epsilon^2}$, which preserves approximate homogeneity more uniformly.

---

### 5. `Randers.metric_fn` — Zero-Velocity Guard (lines 109–111)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1: $F(x, 0) = 0$ by positive homogeneity.
- **Implementation:**
  ```python
  v_mag = safe_norm(v, axis=-1)
  is_zero = v_mag < 1e-7
  v_safe = jnp.where(is_zero[..., None], v + 1e-7, v)
  ...
  return jnp.where(is_zero, 0.0, cost)
  ```
- **Verdict:** OK
- **Notes:**
  - When $\|v\| < 10^{-7}$, the code substitutes $v_{\text{safe}} = v + 10^{-7}\mathbf{1}$ (element-wise) and masks the output to $0$. The intermediate computation on $v_{\text{safe}}$ only exists to avoid NaN in the forward/backward pass; the final mask enforces $F(x, 0) = 0$ correctly. ✓
  - The threshold $10^{-7}$ is compatible with single-precision float32 ($\approx 10^{-7.2}$ ULP). ✓

---

### 6. `Randers.norm` Override (line 130)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1: $F(x,v)$ is the Finsler norm by definition.
- **Implementation:**
  ```python
  def norm(self, x, v):
      return self.metric_fn(x, v)
  ```
- **Verdict:** OK
- **Notes:** Trivially correct — delegates to `metric_fn`, which computes $F(x,v)$. This method is not present on the base class `FinslerMetric`; adding it here is harmless but creates an API inconsistency with other metric classes.

---

### 7. `DiscreteRanders.metric_fn` — Zermelo on Flat Sea (lines 147–158)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 with $h_{ij} = \delta_{ij}$ (Euclidean sea).
- **Literature Reference:** Same Zermelo formula specialised to $h = I$.
- **Implementation:**
  ```python
  W_raw  = jnp.dot(weights, self.face_winds)
  w_norm = jnp.linalg.norm(W_raw)
  scale  = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
  W      = W_raw * scale
  lam    = 1.0 - (w_norm * scale)**2

  v_sq    = jnp.dot(v, v)                  # ||v||^2 (Euclidean)
  W_dot_v = jnp.dot(W, v)                  # W · v   (Euclidean)
  discriminant = lam * v_sq + W_dot_v**2
  cost = (jnp.sqrt(jnp.maximum(discriminant, 1e-8)) - W_dot_v) / lam
  ```
- **Verdict:** OK
- **Notes:**
  - With $h = I$: $\|v\|_h^2 = v \cdot v$, $\langle W,v\rangle_h = W \cdot v$, $\lambda = 1 - \|W\|^2$. All inner products reduce to Euclidean dot products. Formula matches spec § 5. ✓
  - **Lambda computation:** `lam = 1 - (w_norm * scale)^2`. Since $\|W\| = \|W_{\text{raw}}\| \cdot |\text{scale}| = w\_norm \cdot \text{scale}$ (scale > 0), this is $1 - \|W\|^2$. ✓
  - **Squasher:** Identical mathematical structure to the continuous `Randers` squasher; ensures $\|W\| < 1$. ✓

---

### 7a. `DiscreteRanders.metric_fn` — Missing Zero-Velocity Guard

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1: $F(x, 0) = 0$.
- **Verdict:** WARNING
- **Notes:**
  - Unlike `Randers.metric_fn`, `DiscreteRanders` does **not** guard against $v = 0$. When $v = 0$:
    - `v_sq = 0`, `W_dot_v = 0`, `discriminant = 0`.
    - `cost = (sqrt(max(0, 1e-8)) - 0) / lam = sqrt(1e-8) / lam ≈ 1e-4 / lam`.
    This returns a small positive value instead of the correct $F(x,0) = 0$.
  - The error magnitude ($\sim 10^{-4}$) is small but non-zero, and could accumulate in geodesic integration near stationary points.
  - **Recommended Action:** Add the same `is_zero` guard used in `Randers.metric_fn`.

---

### 8. Inherited Spray Correctness (via `FinslerMetric.spray` in `metric.py`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1–2.2.
- **Verdict:** OK (for the code; spec has a sign discrepancy — see below)
- **Notes:**
  - All four zoo classes inherit `FinslerMetric.spray()` from `src/ham/geometry/metric.py`. The spray solves:
    $$g_{ij}\,\ddot{x}^j = \nabla_x^i E - \left[\text{Jac}_x(\nabla_v E)\right]_{ik} v^k$$
    with $\ddot{x}^i = -2G^i$, giving $G^i = \frac{1}{2}g^{il}\bigl(\text{Jac}_x(\nabla_v E)_{lk}\,v^k - \nabla_x^l E\bigr)$.
  - The code computes `rhs = grad_x - mixed_term` then `G = -0.5 * solve(g, rhs)`, which is algebraically equivalent to the correct sign. ✓
  - **Spec discrepancy (NOTE):** `spec/MATH_SPEC.md` § 2.2 writes $\text{Hess}_v(E) \cdot (2G) = \nabla_x E - \text{Jac}_x(\nabla_v E) \cdot v$. From the Euler–Lagrange equations, the correct identity is $g \cdot (2G) = \text{Jac}_x(\nabla_v E) \cdot v - \nabla_x E$ (opposite sign on the RHS). The **code is correct**; the spec formula has a sign error that is compensated by the negation in `return -0.5 * acc`.

---

### 9. `Randers.use_wind=False` Fallback (line 104)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table row "Riemannian".
- **Implementation:**
  ```python
  if not self.use_wind:
      return H, jnp.zeros_like(W_safe), 1.0
  ```
- **Verdict:** OK
- **Notes:** With $W = 0$ and $\lambda = 1$, the Randers formula reduces to:
  $$F(x,v) = \frac{\sqrt{1 \cdot \|v\|_h^2 + 0} - 0}{1} = \|v\|_h$$
  which is the Riemannian metric induced by $H$. ✓

---

## Open Questions

1. **PD guarantee for `h_net` outputs (§ 3a):** Is there an architectural guarantee (e.g., Cholesky parameterisation) in the neural network modules that supply `h_net`? If so, the diagonal-clamping WARNING is mitigated by construction. Requires inspection of `src/ham/models/learned.py` or the training pipeline.

2. **Spec sign error (§ 8):** The sign discrepancy in `spec/MATH_SPEC.md` § 2.2 does not affect the code (which is correct), but a future implementer reading the spec literally could introduce a bug. Should the spec be corrected?

3. **Regularisation strategy (§ 4a):** The additive $10^{-9}$ inside the square root is pragmatic but breaks homogeneity. Would the spec-prescribed $F_\epsilon = \sqrt{F^2 + \epsilon^2}$ approach (§ 6.1) be more appropriate here, or does it introduce other numerical issues (e.g., non-zero cost at $v = 0$)?
