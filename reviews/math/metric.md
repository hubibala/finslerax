# Math Review: `metric.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** [src/ham/geometry/metric.py](src/ham/geometry/metric.py)

## Summary

The implementation in `metric.py` is **mathematically correct**. The core spray computation uses the implicit Euler-Lagrange approach and correctly solves the linear system. The energy definition, fundamental tensor, inner product, geodesic acceleration, and arc-length discretization are all sound. Two issues deserve attention: (1) a hardcoded Tikhonov regularization constant `1e-4` in the spray solve introduces systematic bias, and (2) the MATH_SPEC.md itself contains two formula errors in §2.1 and §2.2 that **do not** affect the code (which derives things correctly from first principles).

---

## Formula-by-Formula Audit

### 1. Energy Functional — `energy()`

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2
- **Literature Reference:** Standard; see Bao–Chern–Shen, *Introduction to Riemann–Finsler Geometry* (2000), §1.2.
- **Implementation:** [metric.py](src/ham/geometry/metric.py#L33-L34)
  ```python
  def energy(self, x, v):
      return 0.5 * self.metric_fn(x, v)**2
  ```
- **Verdict:** OK
- **Notes:** Directly implements $E(x,v) = \tfrac{1}{2}F^2(x,v)$. No issues.

---

### 2. Fundamental Tensor / Inner Product — `inner_product()`

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1, equation for $g_{ij}$
- **Literature Reference:** $g_{ij}(x,v) = \frac{1}{2}\frac{\partial^2 F^2}{\partial v^i \partial v^j} = \frac{\partial^2 E}{\partial v^i \partial v^j}$
- **Implementation:** [metric.py](src/ham/geometry/metric.py#L36-L40)
  ```python
  g_fn = jax.hessian(self.energy, argnums=1)
  g_x_v = g_fn(x, v)
  return jnp.dot(w1, jnp.dot(g_x_v, w2))
  ```
- **Verdict:** OK
- **Notes:** `jax.hessian(E, argnums=1)` computes $\frac{\partial^2 E}{\partial v^i \partial v^j} = g_{ij}$. The inner product $g_{ij} w_1^i w_2^j$ is correctly computed as $w_1^T g \, w_2$. The tensor is evaluated at the reference direction $v$, consistent with the Finsler (direction-dependent) definition.

---

### 3. Geodesic Spray — `spray()`

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1, § 2.2
- **Literature Reference:** Bao–Chern–Shen (2000), Proposition 5.2.1; Chern–Shen, *Riemann–Finsler Geometry* (2005), §3.3.
- **Implementation:** [metric.py](src/ham/geometry/metric.py#L42-L66)
- **Verdict:** OK (code correct) / **WARNING** (spec inconsistency)

#### Detailed derivation check

The Euler-Lagrange equation $\frac{d}{dt}\frac{\partial E}{\partial v^i} - \frac{\partial E}{\partial x^i} = 0$ expands to:

$$\sum_j \frac{\partial^2 E}{\partial x^j \partial v^i} v^j + \sum_j \frac{\partial^2 E}{\partial v^j \partial v^i}\ddot{x}^j = \frac{\partial E}{\partial x^i}$$

Substituting $\ddot{x}^j = -2G^j$:

$$g_{ij}(-2G^j) = \frac{\partial E}{\partial x^i} - \frac{\partial^2 E}{\partial x^k \partial v^i} v^k$$

In matrix form:

$$\boxed{\operatorname{Hess}_v(E)\cdot(-2G) = \nabla_x E - \operatorname{Jac}_x(\nabla_v E)\cdot v}$$

**Code trace:**

| Code element | Mathematical meaning |
|---|---|
| `grad_x` | $\nabla_x E$ |
| `mixed_term` via `jax.jvp(d_dv_fixed_v, (x,), (v,))` | $\operatorname{Jac}_x(\nabla_v E)\cdot v = \sum_k \frac{\partial^2 E}{\partial x^k \partial v^i} v^k$ |
| `rhs = grad_x - mixed_term` | $\nabla_x E - \operatorname{Jac}_x(\nabla_v E)\cdot v$ |
| `acc = solve(hess_v, rhs)` | $\text{acc} = -2G$ |
| `return -0.5 * acc` | $G$ |

The JVP closure correctly captures `v` as a constant w.r.t. the $x$-differentiation, ensuring only the spatial Jacobian of $\nabla_v E$ is computed. **The implementation is correct.**

#### Spec inconsistency (§ 2.1) — WARNING

The spec writes:

$$G^i = \frac{1}{4}g^{il}\!\left(2\frac{\partial^2 E}{\partial v^l \partial x^k}v^k - \frac{\partial E}{\partial x^l}\right)$$

The standard formula (Bao–Chern–Shen, eq. 5.6) uses $F^2$, not $E$:

$$G^i = \frac{1}{4}g^{il}\!\left(\frac{\partial^2 [F^2]}{\partial x^k \partial v^l}v^k - \frac{\partial [F^2]}{\partial x^l}\right)$$

Since $F^2 = 2E$, both derivative terms acquire a factor of 2. The spec correctly doubles the mixed-derivative term ($2\frac{\partial^2 E}{\partial v^l\partial x^k}$) but **omits** the factor of 2 on the gradient term. The correct $E$-based formula is:

$$G^i = \frac{1}{2}g^{il}\!\left(\frac{\partial^2 E}{\partial x^k \partial v^l}v^k - \frac{\partial E}{\partial x^l}\right)$$

**Recommended Action:** Fix the explicit formula in `spec/MATH_SPEC.md` § 2.1 to read $\frac{1}{2}g^{il}\!\bigl(\frac{\partial^2 E}{\partial x^k \partial v^l}v^k - \frac{\partial E}{\partial x^l}\bigr)$.

#### Spec inconsistency (§ 2.2) — WARNING

The spec writes:

$$\operatorname{Hess}_v(E)\cdot(2G) = \nabla_x E - \operatorname{Jac}_x(\nabla_v E)\cdot v$$

The correct sign (derived above and consistent with the code docstring) is:

$$\operatorname{Hess}_v(E)\cdot\mathbf{(-2G)} = \nabla_x E - \operatorname{Jac}_x(\nabla_v E)\cdot v$$

The code docstring at [metric.py](src/ham/geometry/metric.py#L46) correctly states the $(-2G)$ convention and the implementation matches.

**Recommended Action:** Fix the sign in `spec/MATH_SPEC.md` § 2.2 from $(2G)$ to $(-2G)$.

---

### 4. Spray Hessian Regularization

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1 (general regularization philosophy)
- **Implementation:** [metric.py](src/ham/geometry/metric.py#L62)
  ```python
  acc = jnp.linalg.solve(hess_v + 1e-4 * jnp.eye(x.shape[0]), rhs)
  ```
- **Verdict:** WARNING
- **Notes:** Tikhonov regularization $g_{ij} + \epsilon I$ with $\epsilon = 10^{-4}$ is applied to prevent singular linear solves. This introduces a systematic bias:

  $$G_\epsilon \;=\; -\tfrac{1}{2}(g + \epsilon I)^{-1}\,\text{rhs} \;\neq\; -\tfrac{1}{2}\,g^{-1}\,\text{rhs} \;=\; G_{\text{exact}}$$

  The error is $O(\epsilon / \sigma_{\min}^2)$ where $\sigma_{\min}$ is the smallest eigenvalue of $g$. For well-conditioned Riemannian metrics this is negligible, but for Randers metrics near the causality boundary (where $g$ can become ill-conditioned) or for learned neural metrics during early training, $10^{-4}$ may be non-trivial. The constant is also hardcoded rather than using the canonical `PSD_EPS` from [src/ham/utils/math.py](src/ham/utils/math.py#L9).

  **Recommended Action:** Consider using `PSD_EPS` from `ham.utils.math` for consistency, or making the regularization strength a configurable parameter of `FinslerMetric`.

---

### 5. Geodesic Acceleration — `geod_acceleration()`

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1, equation $\ddot{x}^i + 2G^i = 0$
- **Implementation:** [metric.py](src/ham/geometry/metric.py#L68-L69)
  ```python
  def geod_acceleration(self, x, v):
      return -2.0 * self.spray(x, v)
  ```
- **Verdict:** OK
- **Notes:** The geodesic ODE is $\ddot{x}^i = -2G^i(x,\dot{x})$. The code returns exactly $-2G$. This is consumed by the RK4 integrator in [solvers/geodesic.py](src/ham/solvers/geodesic.py#L37) which sets `dv = metric.geod_acceleration(x, v)`. Correct.

---

### 6. Arc Length — `arc_length()`

- **Spec Reference:** Implicit from $\ell[\gamma] = \int_0^T F(\gamma(t), \dot\gamma(t))\,dt$ (standard Finsler length)
- **Implementation:** [metric.py](src/ham/geometry/metric.py#L71-L82)
  ```python
  def segment_length(x1, x2):
      v = x2 - x1
      return self.metric_fn(0.5 * (x1 + x2), v)
  return jnp.sum(jax.vmap(segment_length)(gamma[:-1], gamma[1:]))
  ```
- **Verdict:** OK
- **Notes:** The continuous integral is discretized via midpoint quadrature:

  $$\ell \approx \sum_i F\!\left(\tfrac{x_i + x_{i+1}}{2},\; x_{i+1} - x_i\right)$$

  By 1-homogeneity of $F$, if the parametrization has time-step $\Delta t$, then $F(x, \Delta x) = F(x, \dot{x}\,\Delta t) = \Delta t\, F(x, \dot{x})$, so the sum correctly accumulates $\sum \Delta t \cdot F$. This is a standard and valid approach. Second-order accurate for smooth curves.

---

## Cross-File Consistency Checks

### Berwald Connection (transport.py)

[transport.py](src/ham/geometry/transport.py#L28-L31) computes:
```python
hessian_v = jax.jacfwd(jax.jacfwd(self.metric.spray, argnums=1), argnums=1)
```
This yields ${}^B\Gamma^i_{jk} = \frac{\partial^2 G^i}{\partial v^j \partial v^k}$, consistent with `spec/MATH_SPEC.md` § 3.1. The spray being differentiated is the one from `metric.py`, so correctness of the Berwald coefficients depends on correctness of the spray — which is verified above.

### Geodesic Solver (geodesic.py)

[geodesic.py](src/ham/solvers/geodesic.py#L37) uses `metric.geod_acceleration(x, v)` as the velocity derivative `dv` in the phase-space ODE $(dx, dv) = (v, -2G)$. This is consistent with $\ddot{x}^i + 2G^i = 0$.

---

## Numerical Stability Assessment

| Concern | Location | Status |
|---|---|---|
| Hessian singularity at $v=0$ | [metric.py:62](src/ham/geometry/metric.py#L62) | Mitigated by `1e-4` regularization (see Finding #4) |
| Square-root gradient at $v=0$ | Delegated to `safe_norm` in subclasses | Handled by `sqrt(max(‖v‖², ε))` pattern in [utils/math.py](src/ham/utils/math.py#L27) |
| Randers causality ($\lambda \to 0$) | [zoo.py:108-114](src/ham/geometry/zoo.py#L108-L114) | Enforced via `tanh`-squash to $\|W\|_h < 1 - \epsilon$; $\lambda > 2\epsilon - \epsilon^2 > 0$ |
| `x.shape[0]` for identity matrix | [metric.py:62](src/ham/geometry/metric.py#L62) | Assumes 1D (non-batched) input; consistent with per-point API design |

---

## Open Questions

1. **Regularization magnitude:** Is $\epsilon = 10^{-4}$ empirically validated for the Randers metrics used in the bio/VAE pipeline? If the fundamental tensor eigenvalues are $O(1)$, this is fine; if they drop to $O(10^{-3})$ near the wind boundary, the bias could be $O(10\%)$.

2. **Spec corrections needed:** The two formula errors in `spec/MATH_SPEC.md` (§ 2.1: missing factor of 2 on $\nabla_x E$; § 2.2: wrong sign on $2G$) should be corrected to avoid confusion for future contributors. The code and docstring are authoritative.

3. **Submanifold spray computation:** The spray is computed in ambient coordinates and the geodesic solver projects post-hoc. For high-codimension submanifolds, the ambient Hessian may be rank-deficient along normal directions. The `1e-4` regularization covers this, but a constrained formulation (projecting the RHS and Hessian to the tangent space before solving) could be more precise.
