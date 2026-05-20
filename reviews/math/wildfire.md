# Math Review: wildfire.py
**Reviewer:** Math Reviewer Agent
**Date:** 2026-05-18
**Spec Version:** 1.1.0 (MATH_SPEC.md, Berwald Edition)

## Summary

The implementation is **mathematically correct**. All three formulae — the 2×2 eigendecomposition in `project_spd`, the G⁻¹-norm in `project_b_norm`, and the Zermelo-Randers cost in `metric_fn` — match the theoretical definitions in `spec/MATH_SPEC.md § 5` and standard Finsler geometry references. One WARNING is raised about a non-obvious co-vector convention that differs from the `zoo/randers.py` notation but is self-consistent.

---

## Formula-by-Formula Audit

### `project_spd` — Eigenvalue formula

- **Spec Reference:** Not explicitly in MATH_SPEC.md; standard linear algebra.
- **Literature Reference:** Any 2×2 symmetric eigendecomposition reference (e.g., Wikipedia: Eigenvalue algorithm).
- **Implementation (wildfire.py:64–80):**
  ```
  trace = g11 + g22
  disc  = sqrt((g11-g22)^2 + 4*g12^2 + 1e-12)
  lam_max = (trace + disc) / 2
  lam_min = (trace - disc) / 2
  theta   = 0.5 * arctan2(2*g12, g11-g22+1e-12)
  G_new = lam_max_c * u1 u1^T + lam_min_c * u2 u2^T
  ```
- **Verification:**
  - Characteristic polynomial of $G$: $\lambda^2 - \text{tr}(G)\lambda + \det(G) = 0$. Discriminant $= \text{tr}^2 - 4\det = (g_{11}-g_{22})^2 + 4g_{12}^2$. ✓
  - First principal direction: $\theta = \frac{1}{2}\arctan2(2g_{12},\ g_{11}-g_{22})$. ✓
  - Reconstruction: $G = \lambda_{\max}[c,s]^T[c,s] + \lambda_{\min}[-s,c]^T[-s,c]$, giving the entries $(g_{11}^{new},g_{12}^{new},g_{22}^{new})$ in the implementation. ✓
  - The $+1\text{e-12}$ guard in `disc` prevents NaN gradients when $(g_{11}-g_{22})^2 + 4g_{12}^2 = 0$ (scalar matrix). ✓
  - The $+1\text{e-12}$ guard in `arctan2` prevents `arctan2(0, 0)` when $g_{12}=0, g_{11}=g_{22}$; convention is arbitrary but deterministic. ✓
- **Verdict:** CORRECT

### `project_b_norm` — G⁻¹-norm formula

- **Spec Reference:** spec/MATH_SPEC.md § 5 (Zermelo causality $\|W\|_h < 1$).
- **Implementation (wildfire.py:101–106):**
  ```
  det_G = g11*g22 - g12^2
  norm_sq = (b1^2*g22 - 2*b1*b2*g12 + b2^2*g11) / det_G
  ```
- **Verification:**
  - $G^{-1} = \frac{1}{\det G}\begin{pmatrix}g_{22} & -g_{12}\\-g_{12} & g_{11}\end{pmatrix}$.
  - $b^TG^{-1}b = \frac{b_1^2 g_{22} - 2b_1b_2 g_{12} + b_2^2 g_{11}}{\det G}$. ✓
- **Verdict:** CORRECT

### `metric_fn` — Zermelo navigation formula

- **Spec Reference:** spec/MATH_SPEC.md § 5.
- **Implementation (wildfire.py:329–351):**
  ```
  det_G  = G[0,0]*G[1,1] - G[0,1]^2
  b_Ginv_b = (b0^2*G11 - 2*b0*b1*G01 + b1^2*G00) / det_G
  lam    = max(1 - b_Ginv_b, 1e-6)
  v_sq_h = v^T G v
  bdotv  = b^T v
  disc   = lam * v_sq_h + bdotv^2
  F      = (sqrt(disc) - bdotv) / lam
  ```
- **Parameterisation convention:** This implementation uses the **co-vector parameterisation** $b = GW$, where $W$ is the wind in spec notation. Under this substitution:
  - $\|W\|_h^2 = W^TGW = (G^{-1}b)^TG(G^{-1}b) = b^TG^{-1}b$. ✓
  - $\langle W, v\rangle_h = W^TGv = (G^{-1}b)^TGv = b^Tv$. ✓
  - $\lambda = 1 - \|W\|_h^2 = 1 - b^TG^{-1}b$. ✓
  
  Substituting into spec/MATH_SPEC.md § 5:
  $$F = \frac{\sqrt{\lambda \|v\|_h^2 + (b^Tv)^2} - b^Tv}{\lambda}$$
  which matches the implementation exactly. ✓

- **Verdict:** CORRECT

- **WARNING (convention):** The co-vector convention ($b = GW$, drift in cotangent space) is internally consistent but differs from the vector convention ($W$, drift in tangent space) used in `zoo/randers.py`. Neither convention is wrong, but the difference means `b` in this file is **not** directly comparable to `W` in `Randers` — a reader may confuse them. A clarifying comment in the docstring of `_get_params` is recommended.

### `metric_fn` — 1-homogeneity

- $F(x, \lambda v) = \lambda F(x, v)$ for $\lambda > 0$ follows from linearity of $\langle b, \cdot \rangle$ and bilinearity of $\langle \cdot, G \cdot \rangle$. The `v_safe` guard perturbs the zero vector, but since the result is masked by `jnp.where(is_zero, 0.0, cost)`, homogeneity is preserved for $\lambda \neq 0$. ✓

### `metric_fn` — Zero-vector guard

- `v_safe = jnp.where(is_zero, v + sqrt(GRAD_EPS), v)` ensures no NaN at $v=0$. The result is masked to $0$. ✓
- `lam = jnp.maximum(1 - b_Ginv_b, 1e-6)`: since `project_b_norm` guarantees $b^TG^{-1}b < \text{max\_norm}^2 = 0.81$, `lam >= 0.19` in practice; the clamp is a defensive guard only. ✓

---

## Open Questions

1. **Fuel embedding gradient**: the embedding lookup `fuel_embedding[fuel_code]` with integer `fuel_code` is a sparse gather. During training, gradients will only flow to the embedding row corresponding to the fuel code at each position. This is standard behaviour (as in word embeddings), but the reviewer notes that if `fuel_embedding` is initialised to zeros and only one fuel type appears in training data, most rows will remain at zero indefinitely. This is a training dynamics concern, not a mathematical error.

2. **Homogeneity of aspect features**: the input features to `local_mlp` include $\sin(\text{aspect})$ and $\cos(\text{aspect})$, which is the standard periodic encoding. The elevation and canopy features are unbounded; normalisation is left to the user.
