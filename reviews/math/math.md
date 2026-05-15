# Math Review: `ham.utils.math`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source File:** [src/ham/utils/math.py](src/ham/utils/math.py)

## Summary

The module is small (one function, four constants) and serves as the numerical stability backbone for the entire library. The `safe_norm` implementation is **correct for its stated purpose** (gradient-safe L2 norm) but carries two mathematical subtleties with downstream consequences for higher-order autodifferentiation through the Finsler energy functional: (1) the `jnp.maximum` clamping creates a non-smooth kink that annihilates the fundamental tensor $g_{ij}$ near $v = 0$, and (2) the pattern breaks positive 1-homogeneity of $F$ for small velocities. Both are mitigated in practice by the spray solver's Hessian regularisation and the non-generic nature of the $v = 0$ singularity, but the alternative additive regularisation $\sqrt{\|v\|^2 + \varepsilon}$ from `spec/MATH_SPEC.md` § 6.1 would be analytically preferable. Epsilon constant values are well-chosen for `float32` arithmetic. **Verdict: Minor Issues.**

---

## Formula-by-Formula Audit

### 1. `GRAD_EPS = 1e-12`

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1 (Epsilon Regularisation)
- **Implementation:** `src/ham/utils/math.py:7`
  ```python
  GRAD_EPS = 1e-12
  ```
- **Verdict:** OK
- **Notes:**
  Default epsilon for `safe_norm`. In `float32`, `sqrt(1e-12) = 1e-6`, which is above the `float32` denormal range ($\sim 1.4 \times 10^{-45}$) and within normal precision. For vectors of dimension $D$ with component magnitudes $\sim 10^{-7}$, $\sum v_i^2 \sim D \times 10^{-14}$, which correctly triggers the guard for $D < 100$. The value is appropriate for gradient-safety without being so large as to introduce visible forward-pass bias ($\sqrt{\varepsilon} = 10^{-6}$).

---

### 2. `NORM_EPS = 1e-8`

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1
- **Implementation:** `src/ham/utils/math.py:8`
  ```python
  NORM_EPS = 1e-8
  ```
- **Verdict:** OK
- **Notes:**
  Used as a threshold for "is this vector effectively zero?" comparisons (e.g., `src/ham/geometry/surfaces.py:45`: `is_zero = norm < NORM_EPS`). In `float32`, $\sqrt{\varepsilon_{\text{machine}}} \approx 3.5 \times 10^{-4}$, so `1e-8` is conservative—it only triggers for vectors that are numerically negligible. Sound choice.

---

### 3. `PSD_EPS = 1e-4`

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1 (epsilon regularisation); implicit in the spray solver formulation § 2.2
- **Implementation:** `src/ham/utils/math.py:9`
  ```python
  PSD_EPS = 1e-4
  ```
- **Verdict:** OK
- **Notes:**
  Defines the canonical eigenvalue floor for positive-definite regularisation ($G \mapsto G + \varepsilon I$). Adding $10^{-4} I$ to the fundamental tensor $g_{ij}$ biases the spray coefficients $G^i$ by $O(\varepsilon / \lambda_{\min}(g))$. For well-conditioned metrics ($\lambda_{\min} \sim O(1)$), this is a $\sim 0.01\%$ perturbation—negligible for `float32`. For ill-conditioned Randers metrics near the causality boundary ($\lambda \to 1$), the regularisation becomes protective rather than distortive. Value is appropriate.

---

### 4. `TAYLOR_EPS = 1e-6`

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1 (Surface formulations)
- **Literature Reference:** Standard practice for Taylor switching in geodesic computations on the sphere; see e.g., Absil, Mahony & Sepulchre, *Optimization Algorithms on Matrix Manifolds* (2008), § 8.1.
- **Implementation:** `src/ham/utils/math.py:10`
  ```python
  TAYLOR_EPS = 1e-6
  ```
- **Verdict:** OK
- **Notes:**
  Used in `src/ham/geometry/surfaces.py:62–69` to switch between exact `sin(θ)/θ` and Taylor approximation $1 - \theta^2/6$ for the sphere exponential/logarithmic maps. At the switching threshold $\theta = 10^{-6}$, the next Taylor term is $\theta^4/120 \approx 8.3 \times 10^{-27}$, which is far below `float32` precision ($\sim 10^{-7}$). This ensures the transition is invisible to the solver. Correct.

---

### 5. `safe_norm(x, axis=-1, keepdims=False, eps=GRAD_EPS)`

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 (Finsler norm $F$); § 6.1 (Epsilon Regularisation)
- **Literature Reference:** The `max`-clamping pattern is standard in deep learning frameworks (e.g., PyTorch `torch.nn.functional.normalize`). The *additive* alternative $\sqrt{\|x\|^2 + \varepsilon}$ is used in the Finsler regularisation literature and is explicitly specified in § 6.1.
- **Implementation:** `src/ham/utils/math.py:16–24`
  ```python
  def safe_norm(x, axis=-1, keepdims=False, eps=GRAD_EPS):
      sq = jnp.sum(x ** 2, axis=axis, keepdims=keepdims)
      return jnp.sqrt(jnp.maximum(sq, eps))
  ```

#### 5a. Forward-Pass Correctness

- **Verdict:** OK
- **Notes:**
  For $\|x\|^2 \geq \varepsilon$, returns exact $\|x\|_2$. For $\|x\|^2 < \varepsilon$, returns $\sqrt{\varepsilon} \approx 10^{-6}$ (a small positive constant). The forward value is correct within the intended tolerance.

#### 5b. Gradient Safety (1st-Order AD)

- **Verdict:** OK
- **Notes:**
  The derivative of $\|x\| = \sqrt{\sum x_i^2}$ is $x_i / \|x\|$, which diverges as $x \to 0$. With the `maximum` guard:
  - When $\sum x_i^2 \geq \varepsilon$: $\partial / \partial x_i = x_i / \sqrt{\sum x_j^2}$ — exact and finite.
  - When $\sum x_i^2 < \varepsilon$: `jnp.maximum` clamps to the constant $\varepsilon$, so $\partial / \partial x_i = 0$ — safe.

  No NaN or Inf in the gradient. Correct.

#### 5c. Positive Homogeneity Violation

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1, property 2: $F(x, \lambda v) = \lambda F(x, v)$ for $\lambda > 0$.
- **Verdict:** WARNING
- **Notes:**
  When `safe_norm` is used as the Finsler function $F$ (e.g., `src/ham/geometry/zoo.py:19`: `return safe_norm(v)` in the Euclidean metric), the clamping violates 1-homogeneity for $\|v\|^2 < \varepsilon$:

  $$\text{safe\_norm}(\lambda v) = \sqrt{\max(\lambda^2 \|v\|^2, \varepsilon)} \neq \lambda \sqrt{\max(\|v\|^2, \varepsilon)} = \lambda \cdot \text{safe\_norm}(v)$$

  when $\|\lambda v\|^2$ crosses the $\varepsilon$ threshold. This means the Euler theorem $g_{ij} v^j = E_{v^i}$ (a consequence of 2-homogeneity of $E$) fails near $v = 0$, which can perturb the spray derivation. However, $v = 0$ is a non-generic fixed point on geodesics, so this affects only degenerate configurations. The additive pattern $\sqrt{\|v\|^2 + \varepsilon}$ from § 6.1 also breaks homogeneity but does so smoothly, which is preferable for higher-order AD.

  **Recommended Action:** For the Euclidean `metric_fn`, consider whether the additive form $\sqrt{\|v\|^2 + \varepsilon^2}$ (as specified in `MATH_SPEC.md` § 6.1) would be more appropriate, since it preserves $C^\infty$ smoothness needed by the Berwald connection computation.

#### 5d. Fundamental Tensor Annihilation Near $v = 0$

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1, property 3: $g_{ij} = \frac{1}{2}\partial^2 F^2 / \partial v^i \partial v^j$ must be positive definite.
- **Verdict:** WARNING
- **Notes:**
  When `safe_norm` feeds into the energy $E = \frac{1}{2}F^2$, for $\|v\|^2 < \varepsilon$ the energy becomes:

  $$E(x,v) = \tfrac{1}{2}\max(\|v\|^2, \varepsilon) = \tfrac{1}{2}\varepsilon \quad \text{(constant)}$$

  All velocity derivatives vanish: $\partial E / \partial v^i = 0$, $\partial^2 E / \partial v^i \partial v^j = 0$. This means $g_{ij} = 0$ in this region, violating strong convexity (§ 1.1, property 3). The spray solver in `src/ham/geometry/metric.py:60` adds `1e-4 * jnp.eye(...)` to the Hessian, which rescues the linear solve, but the spray direction in this regime is determined entirely by regularisation rather than the metric geometry.

  By contrast, the additive form $E = \frac{1}{2}(\|v\|^2 + \varepsilon)$ has Hessian $\partial^2 E / \partial v^i \partial v^j = \delta_{ij}$ everywhere, preserving the Euclidean fundamental tensor structure even at $v = 0$.

  In practice, geodesic integration with adaptive step-size control rarely encounters exact $v = 0$, so this is not a runtime failure. However, during training with stochastic initialisation, transient near-zero velocities are possible.

  **Recommended Action:** Consider providing an alternative `safe_norm_additive` using $\sqrt{\|v\|^2 + \varepsilon}$ for use in `metric_fn` implementations where higher-order AD smoothness matters (spray, Berwald connection). The current `safe_norm` with `maximum` remains appropriate for contexts where only 1st-order gradients are needed (loss functions, normalisation).

#### 5e. Higher-Order AD Smoothness

- **Spec Reference:** `spec/MATH_SPEC.md` § 3.1 (Berwald connection requires $\partial^2 G^i / \partial v^j \partial v^k$, i.e., 4th derivatives of $E$)
- **Verdict:** WARNING
- **Notes:**
  `jnp.maximum(sq, eps)` is $C^0$ but not $C^1$ at $\text{sq} = \varepsilon$. In JAX, higher-order derivatives of `jnp.maximum` at the kink produce zero (subgradient convention), but the transition from nonzero to zero second derivatives is discontinuous. For the Berwald connection, which requires:

  $${}^B\Gamma^i_{jk} = \frac{\partial^2 G^i}{\partial v^j \partial v^k}$$

  this involves 4th-order differentiation of $E$ through the spray. The kink at $\|v\|^2 = \varepsilon$ in `safe_norm` creates a discontinuity in the 2nd derivative of $E$, which propagates to a distributional (delta-function-like) singularity in 3rd and 4th derivatives. While JAX silently returns finite values (using the subgradient convention), these values are **not mathematically meaningful** derivatives at the kink.

  This is unlikely to be triggered during normal geodesic integration (where $\|v\|$ is typically $O(1)$), but could arise in learned metric training when the velocity field passes through near-zero regions.

---

## Open Questions

1. **Spec Alignment:** `MATH_SPEC.md` § 6.1 specifies the additive regularisation $F_\varepsilon = \sqrt{F^2 + \varepsilon^2}$, while `safe_norm` implements the `maximum`-clamping pattern. Should the codebase provide both patterns — `safe_norm` (with `maximum`) for loss/normalisation use and an additive variant for `metric_fn` implementations — to align with the spec's intended regularisation strategy?

2. **Epsilon Hierarchy Documentation:** The four constants define a clear hierarchy ($10^{-12} < 10^{-8} < 10^{-6} < 10^{-4}$), but the rationale for each specific value is documented only in inline comments. Should the spec (§ 6) formally define the epsilon hierarchy and the contexts where each applies?

3. **`float64` Considerations:** All epsilon values are calibrated for `float32`. If `jax.config.update("jax_enable_x64", True)` is used, `GRAD_EPS = 1e-12` may be too aggressive (would clamp vectors with components $\sim 10^{-7}$, which are well-representable in `float64`). Should the constants adapt to `jax.numpy.finfo(jax.numpy.float32).eps` dynamically?
