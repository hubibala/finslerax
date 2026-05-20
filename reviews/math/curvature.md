# Math Review: `curvature.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** [src/ham/geometry/curvature.py](src/ham/geometry/curvature.py)

## Summary

The core curvature computations — nonlinear connection $N^i_j$, Riemann curvature tensor $R^i_{\ jk}$, and flag (sectional) curvature — are **mathematically correct** and internally consistent. The Riemann curvature tensor follows the sign convention $R^i_{\ jk} = \delta_k N^i_j - \delta_j N^i_k$, and the flag curvature formula correctly contracts it with the flagpole and transverse edge using the direction-dependent fundamental tensor. However, the `scalar_curvature` function has **major mathematical issues**: it uses Euclidean orthogonalization instead of metric-based, computes only a single sectional curvature (insufficient for $n > 2$), and ignores the direction-dependence of Finsler scalar curvature. No test coverage exists for this module.

**Verdict: Minor Issues** (core formulas correct; `scalar_curvature` unreliable)

---

## Formula-by-Formula Audit

### 1. Nonlinear Connection — `nonlinear_connection()`

- **Spec Reference:** Implicit from `spec/MATH_SPEC.md` § 3.1; the Berwald coefficients ${}^B\Gamma^i_{jk} = \partial^2 G^i / \partial v^j \partial v^k$ require $N^i_j$ as an intermediate.
- **Literature Reference:** Bao–Chern–Shen, *An Introduction to Riemann–Finsler Geometry* (Springer GTM 200, 2000), §2.3; Shen, *Lectures on Finsler Geometry* (World Scientific, 2001), §5.1.
- **Implementation:** [curvature.py](src/ham/geometry/curvature.py#L5-L9)
  ```python
  def nonlinear_connection(metric, x, v):
      return jax.jacfwd(metric.spray, argnums=1)(x, v)
  ```
- **Verdict:** OK
- **Notes:** Computes $N^i_j = \frac{\partial G^i}{\partial v^j}$ by forward-mode Jacobian of the spray w.r.t. velocity. The standard definition of the nonlinear connection coefficients of the geodesic spray is exactly this. The output shape is $(D, D)$ where $N[i, j] = N^i_{\ j}$.

---

### 2. Riemann Curvature Tensor — `riemann_curvature_tensor()`

- **Spec Reference:** Not directly in `spec/MATH_SPEC.md` (the spec covers spray and Berwald connection but does not define the Riemann curvature tensor of the nonlinear connection).
- **Literature Reference:** Bao–Chern–Shen (2000), Chapter 6, §6.2; Bucataru–Miron, *Finsler-Lagrange Geometry* (Editura Academiei Române, 2007), §2.5. The curvature of the nonlinear connection is defined as:

  $$R^i_{\ jk} = \delta_k N^i_j - \delta_j N^i_k$$

  where $\delta_k = \frac{\partial}{\partial x^k} - N^l_k \frac{\partial}{\partial y^l}$ is the horizontal derivative induced by the nonlinear connection.

- **Implementation:** [curvature.py](src/ham/geometry/curvature.py#L11-L48)
- **Verdict:** OK
- **Notes:**

  Expanding the horizontal derivatives:

  $$R^i_{\ jk} = \frac{\partial N^i_j}{\partial x^k} - N^l_k \frac{\partial N^i_j}{\partial y^l} - \frac{\partial N^i_k}{\partial x^j} + N^l_j \frac{\partial N^i_k}{\partial y^l}$$

  Rearranging:

  $$R^i_{\ jk} = \frac{\partial N^i_j}{\partial x^k} - \frac{\partial N^i_k}{\partial x^j} + N^l_j \frac{\partial N^i_k}{\partial y^l} - N^l_k \frac{\partial N^i_j}{\partial y^l}$$

  This matches the docstring and the code term-by-term:

  | Code term | Expression | Matches |
  |---|---|---|
  | `term1 = dN_dx` | $+\frac{\partial N^i_j}{\partial x^k}$, shape $(i,j,k)$ | ✓ |
  | `term2 = -transpose(dN_dx, (0,2,1))` | $-\frac{\partial N^i_k}{\partial x^j}$ (swap $j \leftrightarrow k$) | ✓ |
  | `term3 = einsum('lj,ikl->ijk', N, dN_dv)` | $+N^l_j \frac{\partial N^i_k}{\partial v^l}$ | ✓ |
  | `term4 = -einsum('lk,ijl->ijk', N, dN_dv)` | $-N^l_k \frac{\partial N^i_j}{\partial v^l}$ | ✓ |

  **Verification by Riemannian reduction:** In the Riemannian case $N^i_j = \Gamma^i_{jk} y^k$, so $\partial N^i_j / \partial y^k = \Gamma^i_{jk}$, and the Finsler $R^i_{\ jk}$ reduces to $\hat{R}^i_{\ ljk} y^l$ where $\hat{R}^i_{\ ljk}$ is the standard Riemann curvature tensor. This gives positive flag curvature $K = +1$ for the unit sphere, consistent with the formula in §3 below.

  **Sign convention note:** The opposite convention $R^i_{\ jk} = \delta_j N^i_k - \delta_k N^i_j$ is also common (e.g., some editions of Bucataru–Miron). The code uses $\delta_k N^i_j - \delta_j N^i_k$, which gives the correct sign for positive curvature on the sphere. This should be documented explicitly.

---

### 3. Flag (Sectional) Curvature — `sectional_curvature()`

- **Spec Reference:** Not in `spec/MATH_SPEC.md`.
- **Literature Reference:** Bao–Chern–Shen (2000), §6.2, Definition 6.2.1; Shen (2001), §6.1. The flag curvature with flagpole $y$ and transverse edge $V$ is:

  $$K(y, V) = \frac{g_{im}(x,y)\, R^i_{\ k}(x,y)\, V^k\, V^m}{g_y(y,y)\, g_y(V,V) - g_y(y,V)^2}$$

  where $R^i_{\ k} = R^i_{\ jk} y^j$ is the Jacobi endomorphism.

- **Implementation:** [curvature.py](src/ham/geometry/curvature.py#L50-L73)
  ```python
  R_i = jnp.einsum('ijk,j,k->i', R_tensor, v1, v2)
  numerator = metric.inner_product(x, v1, R_i, v2)
  ```
- **Verdict:** OK
- **Notes:**

  **Step 1 — Contraction:** `R_i` $= R^i_{\ jk} v_1^j v_2^k = R^i_{\ k} v_2^k$ where $R^i_{\ k} = R^i_{\ jk} v_1^j$ is the Jacobi endomorphism evaluated at the flagpole $y = v_1$. ✓

  **Step 2 — Numerator:** `metric.inner_product(x, v1, R_i, v2)` computes $g_{im}(x, v_1)\, \tilde{R}^i\, v_2^m$ where $\tilde{R}^i = R^i_{\ k} v_2^k$. This gives $g_{im}\, R^i_{\ k}\, v_2^k\, v_2^m = g(R \cdot V, V)$, matching the literature. ✓

  **Step 3 — Denominator:** All three terms (`g_11`, `g_22`, `g_12`) use `metric.inner_product(x, v1, ...)`, correctly evaluating the fundamental tensor at the flagpole direction $v_1$. The expression $g_{v_1}(v_1, v_1) \cdot g_{v_1}(v_2, v_2) - g_{v_1}(v_1, v_2)^2$ is the area squared of the parallelogram spanned by $v_1, v_2$ in the $g_{v_1}$-metric. ✓

  **Step 4 — Sphere check:** For the unit sphere ($K = +1$), with the code's sign convention for $R^i_{\ jk}$, the Riemannian reduction gives $R_{mljk} v_1^l v_1^j v_2^k v_2^m = |v_1|^2|v_2|^2 - \langle v_1, v_2\rangle^2$ (verified algebraically using $R_{mljk} = g_{mk}g_{lj} - g_{mj}g_{lk}$). This yields $K = +1$. ✓

---

### 3a. Degenerate-Plane Guard in `sectional_curvature()`

- **Implementation:** [curvature.py](src/ham/geometry/curvature.py#L70-L73)
  ```python
  safe_denom = jnp.maximum(denominator, 1e-12)
  return jnp.where(denominator < 1e-12, 0.0, numerator / safe_denom)
  ```
- **Verdict:** WARNING
- **Notes:** When $v_1$ and $v_2$ are linearly dependent the sectional curvature is **undefined** (the flag plane degenerates). Returning $0.0$ is an arbitrary convention that silently produces a plausible-looking but meaningless number. For downstream code that checks curvature signs (e.g., curvature-based regularization), this could mask degenerate inputs.

  Additionally, the guard only protects against *small positive* denominators. If the fundamental tensor were indefinite (e.g., a bug in the metric), the denominator could be **negative**, and `jnp.maximum(denominator, 1e-12)` would clamp it to $10^{-12}$, producing a wildly incorrect curvature value. For the current codebase (positive-definite Finsler metrics only), this is a non-issue, but it is fragile.

  **Recommended Action:** Consider returning `jnp.nan` or `jnp.inf` for degenerate planes so downstream code can detect and handle the case, or at minimum log a warning.

---

### 4. Scalar Curvature — `scalar_curvature()`

- **Spec Reference:** Not in `spec/MATH_SPEC.md`.
- **Literature Reference:** In Riemannian geometry, the scalar curvature is $S(x) = \sum_{i<j} K(e_i, e_j)$ (summed over an orthonormal basis $\{e_i\}$ w.r.t. $g$). In Finsler geometry, all curvature quantities are direction-dependent: $S(x, y)$, not $S(x)$. See Shen (2001), §6.2.
- **Implementation:** [curvature.py](src/ham/geometry/curvature.py#L75-L100)
- **Verdict:** CRITICAL (mathematical correctness) / WARNING (scope — function is marked as "approximation")

#### Issue 4a — Euclidean Gram–Schmidt (CRITICAL)

  [curvature.py](src/ham/geometry/curvature.py#L93-L97):
  ```python
  t2 = t2 - jnp.dot(t1, t2) * t1
  t2 = t2 / jnp.maximum(jnp.linalg.norm(t2), 1e-8)
  ```

  The orthogonalization uses the **Euclidean** inner product (`jnp.dot`) and **Euclidean** norm (`jnp.linalg.norm`) instead of the Finsler fundamental tensor $g_{ij}(x, v)$. The resulting basis $\{t_1, t_2\}$ is *not* orthonormal w.r.t. the metric. On a Riemannian manifold where the metric tensor $G(x)$ differs significantly from the identity (e.g., strongly anisotropic learned metrics), the sectional curvature of a Euclidean-orthogonal pair does not equal the Gaussian curvature.

  **Correct orthogonalization:** $t_2 \leftarrow t_2 - \frac{g(t_1, t_2)}{g(t_1, t_1)} t_1$, then normalize $t_2 \leftarrow t_2 / \sqrt{g(t_2, t_2)}$.

  **Recommended Action:** Replace `jnp.dot(t1, t2)` with `metric.inner_product(x, t1, t1, t2) / metric.inner_product(x, t1, t1, t1)` and the Euclidean norm with the metric norm.

#### Issue 4b — Single Pair of Vectors (WARNING)

  For manifolds of dimension $n > 2$, the Ricci scalar curvature requires summing $\binom{n}{2}$ sectional curvatures over an orthonormal basis (or an equivalent trace). The function computes only **one** sectional curvature, which is the full Gaussian curvature only for 2-dimensional manifolds.

  **Recommended Action:** Either restrict the function to 2D surfaces (assert `dim == 2` or `intrinsic_dim == 2`), or implement the proper Ricci scalar as $S = \sum_{i \neq j} K(e_i, e_j)$.

#### Issue 4c — Direction-Dependence Ignored (WARNING)

  The function signature `scalar_curvature(metric, x) -> scalar` takes no direction argument, but in Finsler geometry the scalar curvature $S(x, y)$ depends on both position and direction. The function implicitly uses the random vector `t1` as the reference direction for the flag curvature and the fundamental tensor, but this is undocumented and uncontrollable.

  **Recommended Action:** Either (a) add a `v` parameter for the reference direction, or (b) clearly document that this function is only valid for Riemannian metrics (where all curvatures are direction-independent) and assert `isinstance(metric, Riemannian)`.

#### Issue 4d — Fixed Random Seed (NOTE)

  [curvature.py](src/ham/geometry/curvature.py#L86):
  ```python
  key = jax.random.PRNGKey(42)
  ```

  The hardcoded seed makes the function deterministic but the result depends on an arbitrary, fixed choice of tangent plane. This is a testing convenience, not a geometric invariant.

---

### 5. Numerical Stability of the Curvature Pipeline

- **Verdict:** WARNING
- **Notes:**

  The curvature tensor involves **third-order derivatives** of the energy $E$:
  - $G^i$ is computed from $E$ via first and second derivatives and a linear solve (in `spray()`).
  - $N^i_j = \partial G^i / \partial v^j$ adds one more derivative.
  - $\partial N^i_j / \partial x^k$ and $\partial N^i_j / \partial v^l$ add yet another.

  Each differentiation amplifies numerical noise. The spray's Tikhonov regularization ($\epsilon = 10^{-4}$ in `metric.py:62`) means the base $G^i$ is already biased, and differentiating a biased $G^i$ twice can compound the error. No regularization or conditioning check is applied in the curvature functions themselves.

  For Randers metrics near the causality boundary or for learned neural metrics in early training, curvature values may be unreliable. This is acknowledged implicitly by the architecture (the curvature module is not used in the training loop), but should be documented.

  **Recommended Action:** Add a note to the module docstring warning about the numerical sensitivity, and consider adding an optional `eps` parameter for regularizing the spray before curvature differentiation.

---

### 6. Relationship to the Berwald Connection

- **Verdict:** NOTE
- **Notes:**

  The Berwald connection coefficients ${}^B\Gamma^i_{jk} = \frac{\partial^2 G^i}{\partial v^j \partial v^k}$ (from `spec/MATH_SPEC.md` § 3.1 and [transport.py](src/ham/geometry/transport.py#L28-L30)) are the velocity-derivative of the nonlinear connection: ${}^B\Gamma^i_{jk} = \frac{\partial N^i_j}{\partial v^k}$.

  The curvature module computes $\frac{\partial N^i_j}{\partial v^k}$ as `dN_dv` ([curvature.py](src/ham/geometry/curvature.py#L29)). This is mathematically the same tensor as `BerwaldConnection.christoffel_symbols()` in [transport.py](src/ham/geometry/transport.py#L28-L30). The duplication is benign (both use JAX autodiff and will agree), but factoring out a shared `berwald_coefficients(metric, x, v)` function would ensure consistency and reduce recomputation.

---

## Open Questions

1. **Sign convention documentation.** The curvature tensor uses $R^i_{\ jk} = \delta_k N^i_j - \delta_j N^i_k$. Is this the convention intended throughout the library (e.g., for future Jacobi field or stability computations)? This should be recorded in `spec/MATH_SPEC.md`.

2. **Curvature in training loops.** Is the curvature module intended for analytical verification only, or will it be used in loss functions (e.g., curvature regularization)? If the latter, the numerical stability issues in §5 become critical and require mitigation.

3. **Test coverage.** No tests exist for `curvature.py`. At minimum, the following analytical checks are tractable and should be implemented:
   - $K = 0$ for Euclidean space.
   - $K = +1$ for the unit sphere with the round metric.
   - $R^i_{\ jk} = 0$ for flat (Euclidean) metrics.
   - Symmetry: $R^i_{\ jk} = -R^i_{\ kj}$ (antisymmetry in $j, k$).

4. **Scalar curvature scope.** Should `scalar_curvature` be corrected (metric-aware orthogonalization, proper summation) or deprecated in favor of explicit `sectional_curvature` calls? The current implementation is misleading for users who expect the Ricci scalar.
