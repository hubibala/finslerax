# Math Review: demo_weinreb_vis

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

Minor Issues. The core geometric operations (exponential map, logarithmic map, Poincaré projection, pushforward) are mathematically correct and consistent with `spec/MATH_SPEC.md § 4`. However, the demo script uses the **Euclidean norm** to normalize tangent vectors before calling `exp_map`, instead of the **Minkowski norm** required by the hyperboloid geometry. This causes geodesic step lengths to deviate from the intended value at non-origin points. Additionally, a degenerate PRNG reuse causes all branches from a single parent to be identical, producing a path rather than a tree.

## Formula-by-Formula Audit

### 1. Hyperboloid origin point
- **Spec Reference:** `spec/MATH_SPEC.md § 4`, Table row "Hyperboloid"
- **Literature Reference:** Upper sheet of $H^n$: $-x_0^2 + x_1^2 + \cdots + x_n^2 = -1,\; x_0 > 0$
- **Implementation:** `examples/demo_weinreb_vis.py:19` — `root = jnp.array([[1.0, 0.0, 0.0]])`
- **Verdict:** CORRECT
- **Notes:** Check: $-1^2 + 0 + 0 = -1$. Valid point on $H^2$ with $x_0 > 0$.

### 2. Tangent projection (`to_tangent`)
- **Spec Reference:** `spec/MATH_SPEC.md § 4`, Minkowski inner product convention
- **Literature Reference:** Tangent space $T_x H^n = \{v : \langle x, v \rangle_L = 0\}$; projection $\pi_x(v) = v + \langle x, v \rangle_L x$ (since $\langle x, x \rangle_L = -1$)
- **Implementation:** `src/ham/geometry/surfaces.py:340` — `return v + inner[..., None] * x` where `inner = self._minkowski_dot(x, v)`
- **Verdict:** CORRECT
- **Notes:** Standard Minkowski-orthogonal projection. The sign is correct: $\langle x, \pi_x(v) \rangle_L = \langle x, v \rangle_L + \langle x, v \rangle_L \langle x, x \rangle_L = \langle x, v \rangle_L - \langle x, v \rangle_L = 0$.

### 3. Tangent vector normalization (Euclidean vs Minkowski norm)
- **Spec Reference:** `spec/MATH_SPEC.md § 4.1` — "exact $\cosh/\sinh$ exponential and logarithmic maps"
- **Literature Reference:** The Riemannian metric on $H^n$ is the restriction of $\eta_L$ to $T_x H^n$. The geodesic distance of $\exp_x(v)$ from $x$ is $\|v\|_L = \sqrt{\langle v, v \rangle_L}$.
- **Implementation:** `examples/demo_weinreb_vis.py:38–40`
  ```python
  norm = jnp.linalg.norm(tangent_dir)        # Euclidean norm!
  step_size = 0.6  # Geodesic distance
  tangent_step = (tangent_dir / (norm + 1e-6)) * step_size
  ```
- **Verdict:** WARNING
- **Notes:** The code normalizes by $\|v\|_E$ (Euclidean) instead of $\|v\|_L$ (Minkowski). Since `exp_map` internally computes $\|v\|_L$ to determine the geodesic distance, the actual step is $\|v'\|_L \neq 0.6$ in general. For a tangent vector $v$ at $x$, the relation is $\|v\|_E^2 = \|v\|_L^2 + 2 v_0^2$, so $\|v\|_E \geq \|v\|_L$, meaning the Euclidean-normalized vector is **shorter** in hyperbolic distance than intended. At the root $(1,0,0)$ the tangent condition forces $v_0 = 0$, so both norms coincide there. For points away from the origin, the discrepancy grows.

  **Recommended Action:** Replace `jnp.linalg.norm(tangent_dir)` with `manifold._minkowski_norm(tangent_dir.squeeze())` (adjusting shapes accordingly), or equivalently use `jnp.sqrt(manifold._minkowski_dot(tangent_dir, tangent_dir))`.

### 4. Exponential map (`exp_map`)
- **Spec Reference:** `spec/MATH_SPEC.md § 4.1`
- **Literature Reference:** $\exp_x(v) = \cosh(\|v\|_L)\, x + \frac{\sinh(\|v\|_L)}{\|v\|_L}\, v$ (Bao–Chern–Shen, Ch. 3; Cannon et al., "Hyperbolic Geometry")
- **Implementation:** `src/ham/geometry/surfaces.py:342–351`
  ```python
  return jnp.cosh(norm_v)[..., None] * x + sinh_over_norm[..., None] * v
  ```
- **Verdict:** CORRECT
- **Notes:** Taylor expansion for $\sinh(t)/t \approx 1 + t^2/6$ near $t = 0$ is also correct.

### 5. Logarithmic map (`log_map`)
- **Spec Reference:** `spec/MATH_SPEC.md § 4.1`
- **Literature Reference:** $\log_x(y) = \frac{d(x,y)}{\|u\|_L}\, u$ where $u = y + \langle x, y \rangle_L x$ and $d(x,y) = \operatorname{arcsinh}(\|u\|_L)$
- **Implementation:** `src/ham/geometry/surfaces.py:353–366`
  ```python
  xy = self._minkowski_dot(x, y)
  u = y + xy[..., None] * x
  norm_u = self._minkowski_norm(u)
  dist = jnp.arcsinh(norm_u)
  ```
- **Verdict:** CORRECT
- **Notes:** The identity $d = \operatorname{arccosh}(-\langle x, y \rangle_L) = \operatorname{arcsinh}(\|u\|_L)$ follows from $\|u\|_L^2 = \langle x, y \rangle_L^2 - 1 = \cosh^2(d) - 1 = \sinh^2(d)$. The Taylor approximation $\operatorname{arcsinh}(t)/t \approx 1 - t^2/6$ is correct.

### 6. Poincaré disk projection
- **Spec Reference:** Not in `MATH_SPEC.md` (visualization utility)
- **Literature Reference:** Standard stereographic projection $H^n \to \mathbb{B}^n$: $\phi(x) = \frac{x_{\text{spatial}}}{1 + x_0}$ (Cannon et al., "Hyperbolic Geometry", §4)
- **Implementation:** `src/ham/vis/hyperbolic.py:12–14`
  ```python
  return x_spatial / (1.0 + x0)
  ```
- **Verdict:** CORRECT
- **Notes:** This maps $H^n$ onto the open unit ball. The image is guaranteed inside the ball since for $x \in H^n$, $x_0 = \sqrt{1 + \|x_{\text{spatial}}\|^2} \geq 1$, so $\frac{\|x_{\text{spatial}}\|}{1 + x_0} < 1$.

### 7. Pushforward of Poincaré projection
- **Spec Reference:** Not in `MATH_SPEC.md` (visualization utility)
- **Literature Reference:** Differential of $\phi(x) = x_{\text{spatial}} / (1 + x_0)$ via the quotient rule
- **Implementation:** `src/ham/vis/hyperbolic.py:23–30`
  ```python
  num = v_spatial * (1.0 + x0) - x_spatial * v0
  return num / denom   # denom = (1 + x0)^2
  ```
- **Verdict:** CORRECT
- **Notes:** $d\phi_x(v) = \frac{v_{\text{spatial}}(1 + x_0) - x_{\text{spatial}} v_0}{(1 + x_0)^2}$ follows from differentiation of a vector-over-scalar quotient.

### 8. Wind vector overwriting for multi-child parents
- **Spec Reference:** N/A
- **Implementation:** `examples/demo_weinreb_vis.py:63–68`
  ```python
  for p_idx, c_idx in lineage_pairs:
      ...
      vectors[p_idx] = v
  ```
- **Verdict:** NOTE
- **Notes:** When a parent has $k > 1$ children, each child's $\log_{\text{parent}}(\text{child})$ overwrites the previous value. The final wind vector at each parent reflects only the direction to its **last** child in iteration order. Each individual log map is correct, but the resulting wind field is not an average or sum over all children. This is unlikely to be intentional but does not constitute a mathematical error — only a loss of information.

### 9. PRNG reuse across branches (degenerate tree)
- **Spec Reference:** N/A
- **Implementation:** `examples/demo_weinreb_vis.py:33–36`
  ```python
  key, subkey = jax.random.split(key)
  for b in range(n_branches):
      raw_dir = jax.random.normal(subkey, (1, 3))  # same subkey for all b
  ```
- **Verdict:** NOTE
- **Notes:** The same `subkey` generates the same random direction for all $n_{\text{branches}}$ children of a given parent. Combined with deterministic normalization and step size, all branches are identical, producing a degenerate tree (a chain with duplicated nodes at each level). Not a formula error, but the mathematical object produced is not a tree as documented.

  **Recommended Action:** Split `subkey` inside the inner loop:
  ```python
  for b in range(n_branches):
      key, branch_key = jax.random.split(key)
      raw_dir = jax.random.normal(branch_key, (1, 3))
  ```

## Open Questions

1. Is the use of Euclidean norm for step-size normalization intentional (e.g., for visual aesthetics), or should it be corrected to the Minkowski norm to ensure the `step_size` parameter truly represents geodesic distance?
2. Should the wind field aggregate (e.g., average) log-map vectors across all children of a multi-child parent, rather than keeping only the last?
