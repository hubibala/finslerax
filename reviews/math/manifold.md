# Math Review: `manifold.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2026-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** [src/ham/geometry/manifold.py](src/ham/geometry/manifold.py)

## Summary

The implementation in `manifold.py` is **mathematically correct** with **minor issues**. The file defines the abstract `Manifold` base class and a default logarithmic map approximation with a custom JVP helper for numerical stability. The custom JVP for the norm-ratio computation is analytically correct. The default `log_map` implements a well-motivated first-order heuristic that is sound for small separations but has known accuracy degradation for large geodesic distances—this is acknowledged in the docstring and overridden by concrete subclasses. One WARNING-level issue exists: a variable naming/conditioning mismatch in the JVP that, while functionally harmless, could mask a latent NaN-propagation risk under certain JAX compilation paths.

---

## Formula-by-Formula Audit

### 1. Norm Ratio — `_safe_norm_ratio_jvp`

- **Spec Reference:** Not in `spec/MATH_SPEC.md` (utility function).
- **Literature Reference:** Standard quotient rule for derivatives; $\frac{d}{dt}\frac{\|x\|}{\|y\|} = \frac{\|y\|\,\dot{\|x\|} - \|x\|\,\dot{\|y\|}}{\|y\|^2}$ where $\dot{\|x\|} = \frac{x \cdot \dot{x}}{\|x\|}$.
- **Implementation:** [manifold.py](src/ham/geometry/manifold.py#L8-L39)

#### 1a. Primal computation

```python
nx = jnp.linalg.norm(x, axis=-1, keepdims=True)
ny = jnp.linalg.norm(y, axis=-1, keepdims=True)
return jnp.where(ny < 1e-12, 1.0, nx / jnp.maximum(ny, 1e-12))
```

- **Verdict:** OK
- **Notes:** Computes $\|x\| / \|y\|$ with a safe fallback of $1.0$ when $\|y\| \approx 0$. The fallback value $1.0$ is motivated by the use-site in `log_map` where $x = y - p$ and $y = P_p(y - p)$: as the two points coincide, $v \to y - p$ and the ratio approaches $1$ by the smoothness of the tangent projection. Mathematically sound.

#### 1b. JVP computation

```python
dnx = jnp.where(nx < 1e-12, 0.0,
    jnp.sum(x * x_dot, axis=-1, keepdims=True) / nx_safe)
dny = jnp.where(is_zero, 0.0,
    jnp.sum(y * y_dot, axis=-1, keepdims=True) / ny_safe)
tangent_out = jnp.where(is_zero, 0.0,
    (ny_safe * dnx - nx * dny) / (ny_safe**2))
```

- **Verdict:** OK (analytically correct) / **WARNING** (naming and conditioning)
- **Notes:**

  The JVP implements the quotient rule:

  $$\frac{d}{dt}\frac{\|x\|}{\|y\|} = \frac{\|y\| \cdot \frac{x \cdot \dot{x}}{\|x\|} - \|x\| \cdot \frac{y \cdot \dot{y}}{\|y\|}}{\|y\|^2}$$

  **Verified analytically:** expanding the code:

  $$\texttt{tangent\_out} = \frac{n_y \cdot (x \cdot \dot{x} / n_x) - n_x \cdot (y \cdot \dot{y} / n_y)}{n_y^2} = \frac{x \cdot \dot{x}}{n_x \, n_y} - \frac{n_x (y \cdot \dot{y})}{n_y^3}$$

  This matches the standard formula. ✓

  **WARNING (variable conditioning mismatch at [manifold.py:28-29](src/ham/geometry/manifold.py#L28-L29)):** The variables `nx_safe` and `ny_safe` are both conditioned on `is_zero = (ny < 1e-12)`:
  ```python
  nx_safe = jnp.where(is_zero, 1.0, nx)   # guarded by ny, not nx
  ny_safe = jnp.where(is_zero, 1.0, ny)
  ```
  When `ny \ge 10^{-12}` but `nx < 10^{-12}`, `nx_safe = nx` (possibly $\sim 10^{-15}$), and the expression `x \cdot \dot{x} / nx\_safe` can produce very large intermediate values before the outer `jnp.where(nx < 1e\text{-}12, 0.0, \ldots)` discards them. While this is functionally correct (the outer guard selects $0$), some XLA backends can propagate `inf`/`NaN` from unevaluated `jnp.where` branches into downstream gradients if `_safe_norm_ratio_jvp` is itself further differentiated (e.g., via `jax.jacfwd` in Berwald transport).

  **Recommended Action:** Replace the `nx_safe` safeguard with a direct maximum:
  ```python
  dnx = jnp.where(nx < 1e-12, 0.0,
      jnp.sum(x * x_dot, ...) / jnp.maximum(nx, 1e-12))
  ```
  This eliminates the dependency on `ny` for safeguarding `nx` and avoids large intermediates.

---

### 2. Default Exponential Map — `exp_map`

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1 (hierarchy table; Sphere/Hyperboloid have exact maps)
- **Literature Reference:** Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix Manifolds* (2008), Definition 4.1.1.
- **Implementation:** [manifold.py](src/ham/geometry/manifold.py#L102-L108)
  ```python
  def exp_map(self, x, v):
      return self.retract(x, v)
  ```
- **Verdict:** OK
- **Notes:** A retraction $R_x: T_xM \to M$ satisfies $R_x(0) = x$ and $D R_x(0) = \text{Id}$, making it a first-order approximation of the Riemannian exponential map. Delegating `exp_map` to `retract` as a default is standard practice (Absil et al., §4.1). Concrete subclasses (`Sphere`, `Hyperboloid`) correctly override this with exact closed-form exponential maps (verified in [surfaces.py](src/ham/geometry/surfaces.py#L62-L77) and [surfaces.py](src/ham/geometry/surfaces.py#L339-L350)).

---

### 3. Default Logarithmic Map — `log_map`

- **Spec Reference:** Not explicitly in `spec/MATH_SPEC.md`.
- **Literature Reference:** Standard first-order log map approximation for embedded submanifolds; see Absil et al. (2008), §8.1.
- **Implementation:** [manifold.py](src/ham/geometry/manifold.py#L110-L130)
  ```python
  def log_map(self, x, y):
      v = self.to_tangent(x, y - x)
      scale = _safe_norm_ratio_jvp(y - x, v)
      return v * scale
  ```
- **Verdict:** WARNING
- **Notes:**

  **Mathematical derivation:** For a smooth submanifold $M \subset \mathbb{R}^N$, let $P_x$ be the orthogonal projection onto $T_xM$. The first-order approximation is:

  $$\log_x(y) \approx P_x(y - x)$$

  The code enhances this with a scaling correction:

  $$\log_x(y) \approx \frac{\|y - x\|}{\|P_x(y - x)\|} \cdot P_x(y - x)$$

  This preserves the direction of the tangent projection but rescales the magnitude to the chord length $\|y - x\|$ rather than the projected length $\|P_x(y-x)\|$.

  **Accuracy analysis (sphere $S^n(r)$ at geodesic angle $\theta$):**

  | Quantity | Formula | $\theta = \pi/6$ | $\theta = \pi/2$ | $\theta \to \pi$ |
  |---|---|---|---|---|
  | Exact $\|\log_x(y)\|$ | $r\theta$ | $0.524r$ | $1.571r$ | $3.14r$ |
  | Unscaled $\|P_x(y-x)\|$ | $r\sin\theta$ | $0.500r$ | $1.000r$ | $0$ |
  | Scaled $\frac{\|y-x\|}{\|P_x(y-x)\|}\|P_x(y-x)\|$ | $2r\sin(\theta/2)$ | $0.518r$ | $1.414r$ | $2.0r$ |
  | Scaled relative error | $\|2\sin(\theta/2) - \theta\| / \theta$ | $1.1\%$ | $10\%$ | $36\%$ |

  The scaling strictly improves accuracy over the unscaled version for all $\theta \in (0, \pi)$, but the default log map is still only a first-order approximation. This is acceptable since all concrete subclasses with non-trivial geometry override `log_map` with exact formulas.

  **Degenerate cases:**
  - $y = x$: Both $\|y - x\| = 0$ and $\|v\| = 0$. The `_safe_norm_ratio_jvp` returns $1.0$, so $\log_x(x) = 0$. ✓
  - Purely normal secant ($P_x(y - x) = 0$, $y - x \ne 0$): This can occur on a Torus when $y$ is displaced along the surface normal. Here $\|v\| = 0$, the ratio returns $1.0$, and $\log_x(y) = 0$. This is geometrically incorrect (the geodesic distance is nonzero), but the code comment at [manifold.py:119-122](src/ham/geometry/manifold.py#L119-L122) correctly identifies this as a deliberate trade-off to prevent topological shortcuts through the manifold interior.

  **Recommended Action:** No code change required. The docstring should note that this default is only first-order accurate and that subclasses should override for exact formulas. This is already implied by the docstring text "mathematically rigorous first-order approximation."

---

### 4. Retraction Contract — `retract`

- **Spec Reference:** Not in `spec/MATH_SPEC.md`.
- **Literature Reference:** Absil et al. (2008), Definition 4.1.1: A retraction on $M$ is a smooth map $R: TM \to M$ such that $R_x(0_x) = x$ and $D R_x(0_x) = \text{Id}_{T_xM}$.
- **Implementation:** [manifold.py](src/ham/geometry/manifold.py#L91-L100) (abstract)
  ```
  retract(x, 0) = x
  D(retract)(x, 0)[·] = Id
  ```
- **Verdict:** OK
- **Notes:** The docstring correctly states both retraction axioms. Verified that concrete implementations satisfy them:
  - `Sphere.retract` → delegates to `exp_map` (exact geodesic). ✓
  - `Torus.retract` → `project(x + delta)`. At $\delta = 0$: `project(x) = x` since $x \in M$. ✓ First-order: $D(\text{project} \circ (+))(x, 0)[\delta] = D\text{project}(x)[\delta] = \delta$ since projection is the identity on $T_xM$. ✓
  - `EuclideanSpace.retract` → `x + delta`. Trivially satisfies both axioms. ✓
  - `Hyperboloid.retract` → `project(exp_map(x, safe_delta))`. At $\delta = 0$: `project(exp_map(x, 0)) = project(x) = x`. ✓ First-order: `exp_map` is exact, projection is identity on the hyperboloid. ✓

---

### 5. Tangent Space Projection Contract — `to_tangent`

- **Spec Reference:** Implicit; orthogonal complement of normal space.
- **Literature Reference:** For an embedded submanifold $M \subset \mathbb{R}^N$ with induced metric, the tangent projection is $P_x(v) = v - (v \cdot \hat{n})\hat{n}$ for codimension-1 surfaces, or more generally $P_x(v) = v - \sum_\alpha (v \cdot n_\alpha) n_\alpha$ for higher codimension.
- **Implementation:** [manifold.py](src/ham/geometry/manifold.py#L78-L88) (abstract)
- **Verdict:** OK
- **Notes:** Abstract method with correct contract. Verified concrete implementations:
  - `Sphere.to_tangent(x, v)`: $v - \frac{\langle x, v \rangle}{r^2} x$. This is $v - (v \cdot \hat{n})\hat{n}$ where $\hat{n} = x/r$ is the outward unit normal. ✓
  - `Torus.to_tangent(x, v)`: $v - (v \cdot \hat{n})\hat{n}$ with $\hat{n}$ the surface normal. ✓
  - `Hyperboloid.to_tangent(x, v)`: $v + \langle x, v \rangle_L \, x$ where $\langle \cdot, \cdot \rangle_L$ is the Minkowski inner product. Since $\langle x, x \rangle_L = -1$ on the hyperboloid, $T_xH = \{v : \langle x, v \rangle_L = 0\}$, and the projection is $v - \frac{\langle x, v \rangle_L}{\langle x, x \rangle_L} x = v + \langle x, v \rangle_L \, x$. ✓
  - `EuclideanSpace.to_tangent(x, v)`: Identity. ✓

---

## Numerical Stability Assessment

| Concern | Location | Status |
|---|---|---|
| Division by $\|y\| = 0$ in norm ratio | [manifold.py:16](src/ham/geometry/manifold.py#L16) | Handled: `jnp.where(ny < 1e-12, 1.0, ...)` and `jnp.maximum(ny, 1e-12)` |
| Division by $\|x\| = 0$ in JVP | [manifold.py:35](src/ham/geometry/manifold.py#L35) | Handled via `jnp.where(nx < 1e-12, 0.0, ...)` but uses `nx_safe` conditioned on `ny` (see WARNING in §1b) |
| JVP branch NaN propagation | [manifold.py:28-37](src/ham/geometry/manifold.py#L28-L37) | **WARNING**: Large intermediates possible in discarded `jnp.where` branch; risk if further differentiated |
| Purely normal secant ($v = 0$, $y \ne x$) | [manifold.py:126-130](src/ham/geometry/manifold.py#L126-L130) | Returns $0$ by design; documented trade-off |
| Antipodal points on sphere | Default `log_map` | Direction ill-defined; concrete `Sphere.log_map` handles via arccos clipping |
| `@jax.custom_jvp` composability | [manifold.py:8](src/ham/geometry/manifold.py#L8) | Custom JVPs compose correctly under `jax.jit` and `jax.vmap`; higher-order derivatives (for Berwald) would require `custom_vjp` or explicit 2nd-order JVP — **not needed** since concrete subclasses override `log_map` |

---

## Cross-Reference Check: Concrete Subclass Overrides

The default `log_map` and `exp_map` are overridden by all concrete manifolds:

| Manifold | `exp_map` Override | `log_map` Override | Uses Default `log_map`? |
|---|---|---|---|
| `Sphere` | Exact (arccos/sin) at [surfaces.py:62-77](src/ham/geometry/surfaces.py#L62-L77) | Exact (arccos) at [surfaces.py:82-99](src/ham/geometry/surfaces.py#L82-L99) | No |
| `Hyperboloid` | Exact (cosh/sinh) at [surfaces.py:339-350](src/ham/geometry/surfaces.py#L339-L350) | Exact (arcsinh) at [surfaces.py:352-367](src/ham/geometry/surfaces.py#L352-L367) | No |
| `Torus` | Projected retraction at [surfaces.py:186-188](src/ham/geometry/surfaces.py#L186-L188) | **Inherits default** | **Yes** |
| `Paraboloid` | Projected retraction at [surfaces.py:225-227](src/ham/geometry/surfaces.py#L225-L227) | **Inherits default** | **Yes** |
| `EuclideanSpace` | Exact ($x + v$) at [surfaces.py:437](src/ham/geometry/surfaces.py#L437) | Exact ($y - x$) at [surfaces.py:441](src/ham/geometry/surfaces.py#L441) | No |

The Torus and Paraboloid use the default `log_map`, which is a first-order approximation. This is adequate for their role as test/visualization surfaces (not used in the bio/VAE pipeline).

---

## Open Questions

1. **Higher-order differentiation of `_safe_norm_ratio_jvp`:** If the default `log_map` is ever used inside a `jax.jacfwd(jax.jacfwd(...))` chain (e.g., for computing Berwald coefficients of a Torus metric), the `@jax.custom_jvp` would need a well-defined second-order JVP. Currently, JAX will attempt to differentiate through the JVP definition itself, which involves `jnp.where` branches that may produce NaN for degenerate inputs. Verify that no downstream code path computes 2nd+ derivatives through the default `log_map`.

2. **Torus log_map accuracy:** For the Torus, the default `log_map` cannot capture the topological winding structure (shortest path may wrap around the hole). If the Torus is ever used for geodesic learning, an exact or chart-based `log_map` should be implemented. Currently the Torus appears to be used only for visualization, so this is low priority.

3. **Consistency of `retract` vs `exp_map`:** The docstring for `retract` lists several possible implementations (projected retraction, Euler, Cayley, closed-form). Some concrete classes have `retract` delegate to `exp_map` (Sphere) while others have `exp_map` delegate to `retract` (Torus). This circular delegation is safe because at least one side provides a concrete implementation, but could be confusing.
