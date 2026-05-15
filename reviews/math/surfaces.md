# Math Review: surfaces.py

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** `src/ham/geometry/surfaces.py`

## Summary

**Verdict: Minor Issues.** All closed-form exponential maps, logarithmic maps, and parallel transports for the Sphere and Hyperboloid are analytically correct and match standard differential geometry references. The Torus and Paraboloid use projected retractions (honestly documented) but lack true exponential/logarithmic maps, which may affect geodesic accuracy in downstream solvers. No CRITICAL formula errors were found.

---

## Formula-by-Formula Audit

### 1. Sphere — `project` (lines 41–47)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 (Surface Formulations)
- **Literature Reference:** Standard radial projection onto $S^n(r)$: $\pi(x) = r \cdot x / \|x\|$
- **Implementation:**
  ```python
  direction = jnp.where(is_zero, pole, safe_x)
  return self.radius * direction
  ```
- **Verdict:** OK
- **Notes:** Correctly normalises to radius $r$, with a safe fallback to the north pole $(0, \ldots, 0, 1)$ for zero-length inputs.

---

### 2. Sphere — `to_tangent` (lines 49–51)

- **Spec Reference:** Tangent space of $S^n(r)$: $T_x S^n = \{v : \langle x, v \rangle = 0\}$
- **Literature Reference:** Orthogonal projection $v_\perp = v - \frac{\langle x, v \rangle}{\|x\|^2} x = v - \frac{\langle x, v \rangle}{r^2} x$
- **Implementation:**
  ```python
  proj = jnp.einsum('...i,...i->...', x, v)[..., None] / (self.radius ** 2)
  return v - proj * x
  ```
- **Verdict:** OK
- **Notes:** Correct for all $r > 0$. Uses $\|x\|^2 = r^2$ on the manifold.

---

### 3. Sphere — `exp_map` (lines 53–70)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1 (Surface Formulations)
- **Literature Reference:** do Carmo, *Riemannian Geometry*, §3. For $S^n(r)$, the geodesic from $x$ with velocity $v \in T_x S^n(r)$ is:
  $$\gamma(1) = \cos\!\left(\frac{\|v\|}{r}\right) x + \frac{r \sin\!\left(\frac{\|v\|}{r}\right)}{\|v\|} v = \cos\theta \, x + \frac{\sin\theta}{\theta}\, v, \quad \theta = \frac{\|v\|}{r}$$
- **Implementation:**
  ```python
  theta = norm_v / self.radius
  ...
  return cos_theta * x + sin_theta_over_theta * v
  ```
- **Verdict:** OK
- **Notes:** The coefficient of $v$ is $\frac{\sin\theta}{\theta} = \frac{r \sin(\|v\|/r)}{\|v\|}$, which is the correct scaling. Taylor switching at $\theta < \texttt{TAYLOR\_EPS}$ uses $\sin\theta/\theta \approx 1 - \theta^2/6$ and $\cos\theta \approx 1 - \theta^2/2$, both correct to $O(\theta^4)$.

  **Verified analytically:** $\|\exp_x(v)\|^2 = r^2(\cos^2\theta + \sin^2\theta) = r^2$ (uses $\langle x, v \rangle = 0$), so the result lies on $S^n(r)$.

---

### 4. Sphere — `retract` (line 72–73)

- **Implementation:** Delegates to `exp_map`.
- **Verdict:** WARNING
- **Notes:** Unlike the Hyperboloid's `retract` (which calls `project(exp_map(...))`), the Sphere's retract does not apply a final re-projection. Over many iterative steps (e.g., ODE integration), floating-point drift may cause points to leave $S^n(r)$. This is a minor robustness gap.
- **Recommended Action:** Consider adding a final `self.project(...)` call, matching the Hyperboloid pattern at `src/ham/geometry/surfaces.py:395`.

---

### 5. Sphere — `log_map` (lines 75–93)

- **Spec Reference:** Inverse of §3 above.
- **Literature Reference:** $\log_x(y) = \frac{\theta}{\sin\theta}(y - \cos\theta \cdot x)$, where $\cos\theta = \langle x, y \rangle / r^2$ and $\theta = \arccos(\cos\theta)$.
- **Implementation:**
  ```python
  u = jnp.sum(x * y, axis=-1, keepdims=True) / (self.radius ** 2)
  ...
  scale = jnp.where(dist < TAYLOR_EPS, 1.0 + (dist**2)/6.0, dist / safe_sin)
  diff = y - u_clipped * x
  return scale * diff
  ```
- **Verdict:** OK
- **Notes:**
  - $u = \cos\theta$, `diff` $= y - \cos\theta \cdot x$ is the component of $y$ tangent to $T_x S^n$. Scale $= \theta/\sin\theta$.
  - Taylor expansion $\theta / \sin\theta \approx 1 + \theta^2/6$ is correct.
  - Roundtrip verified: $\log_x(\exp_x(v)) = \frac{\theta}{\sin\theta} \cdot \frac{\sin\theta}{\theta} v = v$.
  - Clipping $\cos\theta$ away from $\pm 1$ introduces $O(\texttt{GRAD\_EPS})$ error in the tangent direction near identity and at the cut locus (antipodal points). This is a necessary numerical regularisation and does not affect correctness for non-degenerate inputs.

---

### 6. Sphere — `parallel_transport` (lines 95–102)

- **Spec Reference:** `spec/MATH_SPEC.md` § 3.2 (Parallel Transport)
- **Literature Reference:** Absil, Mahony & Sepulchre, *Optimization Algorithms on Matrix Manifolds*, §8.1. The closed-form Levi-Civita parallel transport on $S^n(r)$ from $x$ to $y$ is:
  $$P_{x \to y}(v) = v - \frac{\langle y, v \rangle}{r^2 + \langle x, y \rangle}(x + y)$$
- **Implementation:**
  ```python
  xy = jnp.sum(x * y, axis=-1)
  yv = jnp.sum(y * v, axis=-1)
  denom = jnp.maximum(self.radius**2 + xy, 1e-5)
  scale = yv / denom
  return v - scale[..., None] * (x + y)
  ```
- **Verdict:** OK
- **Notes:**
  - **Derivation verified analytically.** Using the general parallel transport formula via log maps:
    $P(v) = v - \frac{\langle v, \log_x(y)\rangle}{\theta^2}(\log_x(y) + \log_y(x))$, and simplifying with $\sin^2\theta = (\cos\theta - 1)(\cos\theta + 1) \cdot (-1)$ and $\langle v, x \rangle = 0$, one obtains the formula above.
  - **Tangency verified:** $\langle y, P(v)\rangle = \langle y,v\rangle(1 - \frac{\langle y,x\rangle + r^2}{r^2 + \langle x,y\rangle}) = 0$.
  - **Norm preservation verified:** $\|P(v)\|^2 = \|v\|^2$ (using $\|x+y\|^2 = 2(r^2 + \langle x,y\rangle)$).
  - Singular at antipodal points ($r^2 + \langle x,y \rangle \to 0$), clamped by `maximum(..., 1e-5)`. This is correct — parallel transport along a geodesic to the antipode is geometrically ill-defined (cut locus).

---

### 7. Torus — `project`, `to_tangent` (lines 121–156)

- **Literature Reference:** Torus $T^2(R, r)$ embedded in $\mathbb{R}^3$: surface at distance $r$ from the center circle of radius $R$ in the $xy$-plane.
- **Implementation:** Projects by computing the normal from the nearest center-circle point $R \cdot \hat{d}_{xy}$ to the input point, then moving to distance $r$ along that normal.
- **Verdict:** OK
- **Notes:** Geometrically correct. The outward normal is $\hat{n} = \frac{(\hat{d}_{xy}(\rho - R),\; z)}{\|(\hat{d}_{xy}(\rho - R),\; z)\|}$. Tangent projection correctly removes the normal component.

---

### 8. Torus — `exp_map` (lines 158–160)

- **Literature Reference:** The true Riemannian exponential map on a torus embedded in $\mathbb{R}^3$ follows geodesics that are helical curves (Clairaut's relation). No closed-form is available for the embedded torus.
- **Implementation:**
  ```python
  return self.project(x + delta)
  ```
- **Verdict:** WARNING
- **Notes:** This is a first-order projected retraction, not the Riemannian exponential map. The docstring honestly states "Approximate exp map via projected retraction." For small $\|\delta\|$ the approximation is first-order accurate, but for large tangent vectors the geodesic can wrap around the torus, which this retraction will not capture. Any solver relying on `exp_map` being exact (e.g., geodesic shooting) will accumulate $O(\|\delta\|^2)$ error per step.
- **Recommended Action:** Document the approximation order in the docstring; consider an ODE-based fallback for higher accuracy.

---

### 9. Torus — `log_map` (missing, falls back to base class)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1 (Surface Formulations)
- **Implementation (base class, `manifold.py:136–150`):**
  ```python
  v = self.to_tangent(x, y - x)
  scale = _safe_norm_ratio_jvp(y - x, v)
  return v * scale
  ```
- **Verdict:** WARNING
- **Notes:** The base-class `log_map` projects the ambient secant $y - x$ onto $T_x M$ and rescales. For a torus, this fails qualitatively when $x$ and $y$ are on opposite sides of the hole (the secant passes through the interior, while the true geodesic wraps around the surface). The rescaling heuristic preserves the ambient distance but not the intrinsic distance.
- **Recommended Action:** Implement a torus-specific `log_map` using the angular parameterisation, or at minimum document this limitation.

---

### 10. Paraboloid — `to_tangent` (lines 210–213)

- **Literature Reference:** For the surface $z = x^2 + y^2$, the outward normal is $\hat{n} = \frac{(-2x, -2y, 1)}{\sqrt{4x^2 + 4y^2 + 1}}$.
- **Implementation:**
  ```python
  n = jnp.array([-2 * x[0], -2 * x[1], 1.0])
  n = n / safe_norm(n)
  return v - jnp.dot(n, v) * n
  ```
- **Verdict:** OK
- **Notes:** Correct normal and orthogonal projection.

---

### 11. Paraboloid — `exp_map` / `retract` (lines 215–221)

- **Implementation:**
  ```python
  xy_new = x[:2] + delta[:2]
  z_new = jnp.sum(xy_new ** 2)
  ```
- **Verdict:** WARNING
- **Notes:** Same situation as the Torus: projected retraction, not true exponential map. The retraction ignores `delta[2]`, which is acceptable if `delta` is a true tangent vector (where $\delta_z = 2x\delta_x + 2y\delta_y$ is determined). However, if `delta` is an arbitrary ambient vector, the $z$-component is silently discarded.
- **Recommended Action:** No action required for math correctness; consider adding an assertion or note that `delta` should be tangent.

---

### 12. `_safe_minkowski_self_norm` + JVP (lines 248–274)

- **Literature Reference:** Minkowski norm: $\|x\|_L = \sqrt{-x_0^2 + \sum_{i \geq 1} x_i^2}$, defined for spacelike vectors.
- **Implementation (JVP):**
  ```python
  inner_dot = -x[..., 0]*x_dot[..., 0] + jnp.sum(x[..., 1:]*x_dot[..., 1:], axis=-1)
  tangent_out = inner_dot / safe_norm_val
  ```
- **Verdict:** OK
- **Notes:** The JVP is $\frac{d}{dt}\sqrt{\langle x, x\rangle_L} = \frac{\langle x, \dot{x}\rangle_L}{\sqrt{\langle x, x\rangle_L}}$, which is correct. Safely handles the zero-norm case by clamping the denominator and zeroing the tangent.

---

### 13. `_safe_arccos` + JVP (lines 277–290)

- **Literature Reference:** $\frac{d}{dx}\arccos(x) = -\frac{1}{\sqrt{1 - x^2}}$
- **Implementation:**
  ```python
  denom = jnp.sqrt(jnp.maximum(1.0 - x_clipped**2, GRAD_EPS))
  tangent_out = -x_dot / denom
  ```
- **Verdict:** OK
- **Notes:** Correct derivative. Clamping $1 - x^2 \geq \texttt{GRAD\_EPS}$ prevents infinite gradients at $|x| = 1$.

---

### 14. Hyperboloid — `_minkowski_dot` (line 318)

- **Literature Reference:** Minkowski inner product with signature $(-,+,\ldots,+)$: $\langle u, v \rangle_L = -u_0 v_0 + \sum_{i\geq 1} u_i v_i$
- **Implementation:**
  ```python
  return -u[..., 0]*v[..., 0] + jnp.sum(u[..., 1:]*v[..., 1:], axis=-1)
  ```
- **Verdict:** OK

---

### 15. Hyperboloid — `project` (lines 324–340)

- **Literature Reference:** Two canonical projection strategies: (a) Minkowski-normalise ($x / \sqrt{-\langle x,x\rangle_L}$) if $x$ is already timelike with $x_0 > 0$; (b) otherwise lift spatial components via $x_0 = \sqrt{1 + \|x_{1:}\|^2}$.
- **Verdict:** OK
- **Notes:** Both branches are correct. Branch (a) preserves the direction in Minkowski space; branch (b) ignores the original $x_0$ and reconstructs it.

---

### 16. Hyperboloid — `to_tangent` (lines 342–344)

- **Literature Reference:** $T_x H^n = \{v : \langle x, v\rangle_L = 0\}$. Projection: $v_\perp = v - \frac{\langle x, v\rangle_L}{\langle x, x\rangle_L} x = v + \langle x, v\rangle_L \, x$.
- **Implementation:**
  ```python
  return v + inner[..., None] * x
  ```
- **Verdict:** OK
- **Notes:** The $+$ sign is correct because $\langle x, x\rangle_L = -1$.

---

### 17. Hyperboloid — `exp_map` (lines 346–357)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1
- **Literature Reference:** Cannon, Floyd, Kenyon & Parry, *Hyperbolic Geometry* (2007). For $H^n$:
  $$\exp_x(v) = \cosh(\|v\|_L) \, x + \frac{\sinh(\|v\|_L)}{\|v\|_L}\, v$$
- **Implementation:**
  ```python
  return jnp.cosh(norm_v)[..., None] * x + sinh_over_norm[..., None] * v
  ```
- **Verdict:** OK
- **Notes:**
  - Taylor expansion $\sinh(t)/t \approx 1 + t^2/6$ is correct.
  - **Verified analytically:** $\langle \exp_x(v), \exp_x(v)\rangle_L = -\cosh^2 + \sinh^2 = -1$ (using $\langle x,v\rangle_L = 0$).

---

### 18. Hyperboloid — `log_map` (lines 359–375)

- **Literature Reference:** The inverse of §17. Given $y = \cosh(d) x + \sinh(d)\hat{v}$, the tangent projection is $u = y + \langle x, y\rangle_L x = y - \cosh(d) x = \sinh(d)\hat{v}$. Then $\|u\|_L = \sinh(d)$ and $d = \text{arcsinh}(\|u\|_L)$.
- **Implementation:**
  ```python
  u = y + xy[..., None] * x
  norm_u = self._minkowski_norm(u)
  dist = jnp.arcsinh(norm_u)
  scale = dist / safe_norm_u
  return scale[..., None] * u
  ```
- **Verdict:** OK
- **Notes:**
  - Using $\text{arcsinh}$ instead of $\text{arccosh}(-\langle x, y\rangle_L)$ is numerically superior: $\text{arcsinh}$ is defined and smooth on $[0, \infty)$, avoiding domain issues.
  - Roundtrip verified: $\log_x(\exp_x(v)) = \frac{d}{\sinh d} \cdot \sinh(d) \hat{v} = d\hat{v} = v$.
  - Taylor expansion $\text{arcsinh}(t)/t \approx 1 - t^2/6$ is correct.

---

### 19. Hyperboloid — `parallel_transport` (lines 377–384)

- **Spec Reference:** `spec/MATH_SPEC.md` § 3.2
- **Literature Reference:** Derived by analogy to the sphere formula (see §6 above), substituting the Minkowski inner product $\langle \cdot, \cdot \rangle_L$ and $\langle x, x\rangle_L = -1$:
  $$P_{x \to y}(v) = v + \frac{\langle y, v\rangle_L}{1 - \langle x, y\rangle_L}(x + y)$$
  where $1 - \langle x, y\rangle_L = 1 + \cosh(d) \geq 2$.
- **Implementation:**
  ```python
  denom = jnp.maximum(1.0 - xy, 2.0)
  scale = yv / denom
  return v + scale[..., None] * (x + y)
  ```
- **Verdict:** OK
- **Notes:**
  - **Sign is correct.** The unified formula for both sphere and hyperboloid is $P(v) = v - \frac{\langle y, v\rangle}{\langle x,x\rangle + \langle x,y\rangle}(x + y)$. For the hyperboloid, $\langle x,x\rangle_L = -1$, yielding the $+$ sign and the denominator $1 - \langle x,y\rangle_L$.
  - **Tangency verified:** $\langle y, P(v)\rangle_L = \langle y,v\rangle_L + \frac{\langle y,v\rangle_L}{1-\langle x,y\rangle_L}(\langle x,y\rangle_L - 1) = 0$.
  - **Norm preservation verified:** $\langle P(v), P(v)\rangle_L = \langle v,v\rangle_L$ (using $\langle x+y, x+y\rangle_L = -2(1 - \langle x,y\rangle_L)$).
  - The `maximum(..., 2.0)` clamp is a no-op for valid inputs ($1 + \cosh d \geq 2$ always), serving only as a numerical safety net.

---

### 20. Hyperboloid — `retract` (lines 386–395)

- **Implementation:**
  ```python
  scale = jnp.where(norm_delta > max_norm, max_norm / safe_nd, 1.0)
  safe_delta = delta * scale[..., None]
  return self.project(self.exp_map(x, safe_delta))
  ```
- **Verdict:** OK
- **Notes:** Clips $\|\delta\|_L \leq 10$ before applying `exp_map`, then re-projects. This prevents $\cosh/\sinh$ overflow ($\cosh(10) \approx 11013$, well within float32 range). The final `project` call ensures the result is exactly on $H^n$, unlike the Sphere's retract (see §4).

---

### 21. Hyperboloid — `metric_tensor` (lines 408–411)

- **Implementation:**
  ```python
  m = jnp.eye(self.ambient_dim)
  m = m.at[0, 0].set(-1.0)
  return m
  ```
- **Verdict:** INFO
- **Notes:** Returns the ambient Minkowski metric $\eta = \text{diag}(-1, 1, \ldots, 1)$, not the induced (pullback) Riemannian metric on $H^n$ in intrinsic coordinates. This is correct if downstream consumers interpret it as the ambient bilinear form, but could be misleading if they expect an intrinsic $n \times n$ metric tensor. The naming `metric_tensor` is ambiguous.

---

### 22. Torus — `random_sample` (lines 165–173)

- **Literature Reference:** The Riemannian volume element on $T^2(R, r)$ is $dA = r(R + r\cos v)\, du\, dv$. Uniform sampling in $(u, v)$ oversamples the inner rim relative to the outer.
- **Verdict:** INFO
- **Notes:** Sampling is uniform in angle space, not area-uniform on the surface. For geometry exploration and initialisation, this is acceptable; for Monte Carlo integration against the Riemannian measure, rejection sampling or importance weighting with density $\propto (R + r\cos v)$ would be needed.

---

### 23. EuclideanSpace (lines 418–457)

- **Verdict:** OK
- **Notes:** All operations are trivially correct for flat $\mathbb{R}^n$: `exp_map = x + v`, `log_map = y - x`, `parallel_transport = id`. Matches `spec/MATH_SPEC.md` § 4 hierarchy table (Spray $= 0$, $\Gamma = 0$).

---

## Open Questions

1. **Sphere retract drift:** The Sphere `retract` (line 72) does not re-project onto $S^n(r)$, unlike the Hyperboloid `retract` (line 395). Is this intentional, or an oversight? Over long ODE integrations, floating-point drift could move points off the manifold.

2. **Torus/Paraboloid exp/log limitations:** The projected retraction used for `exp_map` and the base-class secant used for `log_map` are first-order approximations. Are any downstream solvers (e.g., geodesic shooting, Berwald transport ODE) sensitive to the $O(\|\delta\|^2)$ approximation error? If so, these manifolds may need ODE-based exponential/log maps.

3. **`metric_tensor` semantics:** The Hyperboloid's `metric_tensor` returns the ambient Minkowski tensor $\eta \in \mathbb{R}^{(n+1)\times(n+1)}$ rather than the induced metric in intrinsic coordinates. Is there a convention in the codebase for which one `metric_tensor` should return? No other surface class defines this method.

4. **Paraboloid retract discards $\delta_z$:** The Paraboloid retract (line 219) only uses `delta[:2]` and recomputes $z$. If a caller passes an un-projected ambient vector, the $z$-component is silently ignored. Should there be a precondition check or should `to_tangent` be called internally?
