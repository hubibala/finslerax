# Math Review: wildfire_cnn_encoder (`CovariateConditionedRanders` + `LocalTerrainCNN`)

**Reviewer:** Math Reviewer Agent  
**Date:** 2026-05-18  
**Spec Version:** 1.1.0 (Berwald Revision, December 2, 2025)  
**Source file:** [src/ham/models/wildfire.py](../../src/ham/models/wildfire.py)  
**Spec reference:** [spec/MATH_SPEC.md](../../spec/MATH_SPEC.md)

---

## Summary

The architecture is **mathematically sound** in its essential claims: the Zermelo formula is correctly implemented, all three Finsler axioms (positivity, 1-homogeneity, causality) are enforced by construction, and the IFT adjoint chain correctly propagates $\partial L / \partial (\text{metric\_field})$ back to CNN weights through the outer JAX autograd. Two **WARNINGs** are filed for issues that are mathematically incorrect in degenerate cases: (1) `jnp.clip` silently zeroes the spatial gradient $\partial F/\partial x$ when a path vertex exits the raster boundary — the Euler-Lagrange force is therefore dead at the boundary, preventing the solver from recovering a path that has strayed; (2) the `project_spd` gradient diverges (becomes $O(10^5)$) when the raw combined matrix is nearly isotropic, which can occur with probability $O(1)$ at early training when global and local raw outputs cancel. One additional **WARNING** targets a subtle sign-neutrality failure in the zero-velocity fallback that introduces a spurious anisotropic bias in $\nabla_v F$ at $v = 0$. All other findings are **NOTEs**.

---

## Formula-by-Formula Audit

---

### 1. Zermelo Navigation Formula — `metric_fn`

**Spec Reference:** spec/MATH_SPEC.md § 5, equation  
$$F(x,v) = \frac{\sqrt{\lambda \|v\|_h^2 + \langle W,v\rangle_h^2} - \langle W,v\rangle_h}{\lambda}, \quad \lambda = 1 - \|W\|_h^2$$

**Literature Reference:** Bao, Chern, Shen, *An Introduction to Riemann-Finsler Geometry* (Springer, 2000), Chapter 11 (Zermelo navigation). Shen, *Finsler Metrics with $\mathbf{K}=0$ and $\mathbf{S}=0$*, Canad. J. Math. 55 (2003), §2.

**Implementation** ([src/ham/models/wildfire.py:621–643](../../src/ham/models/wildfire.py)):
```python
disc = lam * v_sq_h + bdotv ** 2
cost = (jnp.sqrt(jnp.maximum(disc, GRAD_EPS)) - bdotv) / lam
```
where `lam = 1 - b_Ginv_b`, `v_sq_h = v^T G v`, `bdotv = b^T v`.

**Verdict:** CORRECT

**Notes:**  
The spec formula uses $\langle W, v\rangle_h$ (h-inner product of the wind vector $W$ with $v$). The code parameterises $b$ as a **covector** (1-form): $b_i$ is the index-lowered form of the Zermelo wind via $b_i = h_{ij}W^j$. Under this identification $\langle W,v\rangle_h = b_i v^i = b^\top v$, and $\|W\|_h^2 = h^{ij}b_i b_j = b^\top G^{-1} b$. Both agree exactly with the code. The minus-sign convention ("headwind increases cost") is correctly applied: for $b^\top v > 0$ (tailwind), cost decreases; for $b^\top v < 0$ (headwind), cost increases. ✓

---

### 2. Causality Invariant $\lambda > 0$ — `metric_fn` + `project_b_norm`

**Spec Reference:** spec/MATH_SPEC.md § 5 ("Zermelo causality, $\|W\|_h < 1$").

**Implementation** ([src/ham/models/wildfire.py:92–113](../../src/ham/models/wildfire.py)):
```python
norm_sq = (b[0] ** 2 * g22 - 2.0 * b[0] * b[1] * g12 + b[1] ** 2 * g11) / det_G
norm = jnp.sqrt(jnp.maximum(norm_sq, GRAD_EPS))
scale = jnp.minimum(1.0, max_norm / (norm + NORM_EPS))
return b * scale
```

**Verdict:** CORRECT

**Notes:**  
$\|b\|_{G^{-1}}^2 = b^\top G^{-1} b$. For a 2×2 symmetric $G$ with $\det G = g_{11}g_{22} - g_{12}^2$, Cramer's rule gives $G^{-1} = \frac{1}{\det G}\begin{bmatrix}g_{22} & -g_{12} \\ -g_{12} & g_{11}\end{bmatrix}$, so $b^\top G^{-1} b = (b_1^2 g_{22} - 2b_1 b_2 g_{12} + b_2^2 g_{11})/\det G$. This matches the code exactly. ✓  
`project_b_norm` is called with the **already-projected** $G$ (i.e., after `project_spd`), so the dual norm used for enforcement is consistent with the metric used in `metric_fn`. ✓  
With `max_norm = 0.9`, we have $\|b\|_{G^{-1}} \le 0.9$, giving $\lambda = 1 - \|b\|_{G^{-1}}^2 \ge 1 - 0.81 = 0.19 > 0$. The `jnp.maximum(1-b_Ginv_b, 1e-6)` guard in `metric_fn` will never trigger in finite arithmetic when the projection has been applied. ✓

---

### 3. 1-Homogeneity — `metric_fn`

**Spec Reference:** spec/MATH_SPEC.md § 1.1, axiom 2: $F(x, \lambda v) = \lambda F(x,v)$ for $\lambda > 0$.

**Implementation** ([src/ham/models/wildfire.py:621–643](../../src/ham/models/wildfire.py)): Zermelo formula applied to $\lambda v$.

**Verdict:** CORRECT

**Notes:**  
Scaling $v \mapsto \mu v$ (using $\mu > 0$ to avoid confusion with the symbol $\lambda$ in the Zermelo formula):
$$F(x, \mu v) = \frac{\sqrt{\lambda \cdot (\mu v)^\top G(\mu v) + (b^\top (\mu v))^2} - b^\top(\mu v)}{\lambda}
= \frac{\sqrt{\mu^2(\lambda\, v^\top G v + (b^\top v)^2)} - \mu b^\top v}{\lambda} = \mu\, F(x,v). \quad \checkmark$$
The parameters $G$, $b$, $\lambda$ depend only on $x$ (not on $v$), so homogeneity holds exactly. ✓

---

### 4. Positivity $F(x,v) > 0$ for $v \ne 0$

**Spec Reference:** spec/MATH_SPEC.md § 1.1, axiom 3 (strong convexity implies positivity).

**Verdict:** CORRECT

**Notes:**  
For any covector $b$ with $\|b\|_{G^{-1}} < 1$ and any SPD $G$, the Randers-Zermelo metric is positive on $TM\setminus\{0\}$. Proof sketch: let $\alpha = \sqrt{\lambda\, v^\top G v + (b^\top v)^2}$ and $\beta = b^\top v$. By Cauchy-Schwarz on the $G^{-1}$-inner product, $|b^\top v|^2 \le \|b\|_{G^{-1}}^2 \cdot v^\top G v$. Then $\alpha^2 = \lambda v^\top G v + \beta^2 > (1 - \|b\|_{G^{-1}}^2) v^\top G v + \|b\|_{G^{-1}}^2 v^\top G v = v^\top G v \ge \alpha_0^2$ (where $\alpha_0 = \sqrt{v^\top G v}$). Also $|\beta| \le \|b\|_{G^{-1}} \alpha_0 < \alpha_0 \le \alpha$, so $\alpha - \beta > 0$. ✓

---

### 5. SPD Projection — `project_spd`

**Spec Reference:** spec/MATH_SPEC.md § 5 (Randers causality); spec/MATH_SPEC.md § 1.1, axiom 3 (fundamental tensor must be positive definite).

**Implementation** ([src/ham/models/wildfire.py:48–83](../../src/ham/models/wildfire.py)):
```python
disc = jnp.sqrt((g11 - g22) ** 2 + 4.0 * g12 ** 2 + 1e-12)
lam_max = (trace + disc) * 0.5
lam_min = (trace - disc) * 0.5
theta = 0.5 * jnp.arctan2(2.0 * g12, g11 - g22 + 1e-12)
c, s = jnp.cos(theta), jnp.sin(theta)
g11_new = lam_max_c * c**2 + lam_min_c * s**2
g12_new = (lam_max_c - lam_min_c) * c * s
g22_new = lam_max_c * s**2 + lam_min_c * c**2
```

**Verdict:** CORRECT (formula) / WARNING (gradient stability)

**Notes:**  
The eigendecomposition formula is standard for 2×2 symmetric matrices: $\theta = \frac{1}{2}\arctan\!\frac{2g_{12}}{g_{11}-g_{22}}$ is the rotation angle of the larger eigenvector, and the reconstruction $G_{\text{new}} = Q\,\mathrm{diag}(\lambda_{\max}^c, \lambda_{\min}^c)\,Q^\top$ is correct. ✓

**WARNING (gradient stability near isotropic inputs):** The regulariser `+ 1e-12` inside `jnp.sqrt` prevents NaN at `disc = 0`, but the gradient $\frac{\partial \text{disc}}{\partial g_{11}} = (g_{11}-g_{22}) / \text{disc}$ blows up as $O(\text{disc}^{-1/2}) \approx O(10^6 / \sqrt{1\text{e-12} \cdot 10^{12}}) = O(1/\sqrt{10^{-12}}) = O(10^6)$ near the isotropic point $g_{11} = g_{22},\, g_{12} = 0$. At early training, when both `raw_global` and `raw_local` are near zero, their sum can sit near this singularity with $O(1)$ probability. This can produce gradient spikes up to $O(10^5)$–$O(10^6)$ that bypass the AVBD `grad_clip` (which clips path-vertex gradients, not metric-parameter gradients).  
**Recommended Action:** Increase the regulariser inside `disc` to `max(1e-6, ...)`, matching the `GRAD_EPS` convention already used elsewhere, or add a gradient-clip on `project_spd` outputs.

---

### 6. Bilinear Interpolation Spatial Gradient — `_bilinear_interp_field`

**Spec Reference:** spec/MATH_SPEC.md § 2.1 (Euler-Lagrange), requires $\partial E/\partial x^i$ to be non-zero wherever the metric varies spatially.

**Implementation** ([src/ham/models/wildfire.py:528–558](../../src/ham/models/wildfire.py)):
```python
px = (x_world[0] - origin[0]) / spacing
py = (x_world[1] - origin[1]) / spacing
px = jnp.clip(px, 0.0, W - 1.001)
py = jnp.clip(py, 0.0, H - 1.001)
x0 = jnp.floor(px).astype(jnp.int32)
...
fx = px - x0
fy = py - y0
return (field[y0, x0] * (1.0 - fx) * (1.0 - fy) + ...)
```

**Verdict:** CORRECT (interior) / WARNING (boundary)

**Notes:**  
Inside the raster domain, the gradient of the interpolated value w.r.t. $x_{\text{world}}$ is:
$$\frac{\partial \hat{f}}{\partial x_{\text{world},0}} = \frac{1}{s}\bigl[(f_{y_0,x_1} - f_{y_0,x_0})(1-f_y) + (f_{y_1,x_1} - f_{y_1,x_0})f_y\bigr]$$
where $s$ is `pixel_spacing_m`. JAX treats `x0 = jnp.floor(px).astype(jnp.int32)` as a stop-gradient constant (integer arrays are not differentiated), so $\partial f_x/\partial p_x = 1$ correctly. The formula yields the standard piecewise-bilinear spatial gradient needed by the Euler-Lagrange equations. ✓

**WARNING (boundary dead-gradient):** `jnp.clip` is applied to `(px, py)` before `fx, fy` are computed. When a path vertex $x_k$ lies outside the raster extent — i.e., $p_x < 0$ or $p_x > W-1.001$ — `jnp.clip` returns a constant, and $\partial \hat{f}/\partial x_{\text{world}} = 0$ exactly. The Euler-Lagrange force term $\partial E/\partial x^i$ at such a vertex is therefore zero, making that vertex a spurious fixed point. The geodesic solver cannot push an out-of-bounds vertex back into the domain. This is not merely a numerical approximation: the gradient is identically zero, not small.  
**Recommended Action:** Apply a soft boundary extension (e.g., linear extrapolation beyond the boundary using edge-pixel gradients) or enforce a domain-bounding constraint on path vertices at the AVBD level, so that the solver never evaluates outside the raster extent.

---

### 7. Additive Raw-Parameter Combination — `_get_params`

**Spec Reference:** spec/MATH_SPEC.md § 5 (Randers parameterisation). No spec precedent for additive decomposition.

**Implementation** ([src/ham/models/wildfire.py:575–581](../../src/ham/models/wildfire.py)):
```python
raw_local  = self._bilinear_interp_field(self.metric_field, x_world)  # (5,)
raw_global = self.global_mlp(self.weather_vec)                         # (5,)
raw        = raw_global + raw_local                                    # (5,)
G = project_spd(raw[:3_sym], self.eps_G, self.max_G)
b = project_b_norm(raw[3:5], G, self.max_b_norm)
```

**Verdict:** CORRECT (projection applied to the sum)

**Notes:**  
Both `raw_global` and `raw_local` are unbounded reals (the global MLP output layer and the CNN head have no final activation). Their sum `raw` is therefore also unbounded. The projections `project_spd` and `project_b_norm` are applied to the **combined** vector, not to each component separately. Because these projections map any real input to valid Randers data (SPD $G$ with bounded eigenvalues; $\|b\|_{G^{-1}} \le 0.9$), the sum can never produce invalid Randers geometry regardless of magnitude. ✓

**WARNING (projection compression at large magnitudes):** When `raw_global` and `raw_local` are large in magnitude but of opposing sign, the sum may land near zero — the centre of the raw-parameter space — where `project_spd` clips both eigenvalues to `eps_G`. This compresses the effective output range to the ball $\{G : \lambda_{\min}(G) = \lambda_{\max}(G) = \text{eps\_G}\}$, eliminating the anisotropy that the two-pathway architecture was designed to express. This is not a mathematical error (the Randers axioms still hold) but represents a failure mode where gradient signal from both pathways simultaneously drives the metric toward isotropy.  
No recommended change — flagging for experimental monitoring.

---

### 8. IFT Adjoint Gradient Flow: `dL/d(metric_field)` → `dL/d(CNN weights)`

**Spec Reference:** spec/MATH_SPEC.md § 2.2 (implicit-solve formulation); spec/ARCH_SPEC.md § 4.2.

**Literature Reference:** Implicit Function Theorem gradient. Krantz & Parks, *The Implicit Function Theorem* (2002), Ch. 1. For the discrete path-energy context see also: Giles, Diaz & Yuksel, *Augmented Vertex Block Descent*, SIGGRAPH 2025.

**Implementation** ([src/ham/solvers/avbd.py:140–200](../../src/ham/solvers/avbd.py)):
```python
# Forward: metric = model_with_field (metric_field is a concrete (H,W,5) leaf)
# Backward IFT step:
dG_dx = jax.jacobian(lambda p: _el_residual(p, metric, p_start, p_end))(inner)
lam, _, _, _ = jnp.linalg.lstsq(dG_dx.T, g_inner.ravel(), rcond=1e-4)
_, vjp_fn = jax.vjp(el_wrt_arr, m_arr)
grad_arr = vjp_fn(-lam)[0]
```

**Verdict:** CORRECT

**Notes:**  
The IFT identity $\nabla_\theta L = -\lambda^\top \frac{\partial G}{\partial \theta}$ (where $G = \nabla_x \mathcal{E}$, $\lambda = (G_{xx}^*)^{-\top} \nabla_{\text{inner}} L$) is implemented correctly via the `lstsq` solve and a single VJP call. ✓

**Chain rule decomposition is correct.** `get_differentiable_mask` does **not** exclude `metric_field` (the field name contains none of the excluded keywords: `['raster', 'pixel_spacing', 'origin', 'manifold', 'covariates', 'weather']`). Therefore `metric_field` is part of `m_arr` (differentiable leaves). The VJP `vjp_fn(-lam)` computes $\partial G / \partial (\text{metric\_field})$ via the chain:
$$\text{EL residual} \xrightarrow{\text{vmap energy}} \text{metric\_fn} \xrightarrow{} \text{\_bilinear\_interp\_field}(\text{metric\_field}, x) \xrightarrow{} \text{metric\_field}_{[y_0, x_0], [y_0, x_1], \ldots}$$
so `grad_arr.metric_field` correctly receives $\partial L / \partial (\text{metric\_field})$ as a dense `(H, W, 5)` array. ✓

The **outer** JAX autograd then propagates `grad_arr.metric_field` through `precompute_metric_field()`:
$$\text{local\_cnn}(\text{raster\_stack}, \text{fuel\_field}) \xrightarrow{\text{tree\_at}} \text{model\_with\_field.metric\_field}$$
yielding $\partial L / \partial (\text{CNN weights})$ and $\partial L / \partial (\text{fuel\_embedding})$ via the CNN's own backward pass. ✓  
The `local_cnn` weights are also in `m_arr`, but $\partial G / \partial (\text{conv weights}) = 0$ from the IFT path (CNN weights do not appear in the EL residual — only `metric_field` does). The total gradient for CNN weights therefore comes entirely from the outer autograd path, which is mathematically correct.

**NOTE (memory cost):** The `(H, W, 5)` gradient tensor for `metric_field` is materialised as a dense array even though at most $4 \times N_{\text{seg}}$ pixels are touched (the bilinear neighbourhood of each path vertex). For a $1000 \times 1000$ raster with $N_{\text{seg}} = 50$ path segments, only $\le 200$ out of $10^6$ pixel-entries are non-zero, but the full `(H, W, 5)` = 5 MB float64 array is allocated. For large rasters this may become a memory bottleneck. Consider sparse-gradient accumulation if rasters exceed $2048 \times 2048$.

---

### 9. Zero-Velocity Fallback Direction — `metric_fn`

**Spec Reference:** spec/MATH_SPEC.md § 6.1 (epsilon regularisation at $v = 0$).

**Implementation** ([src/ham/models/wildfire.py:601–610](../../src/ham/models/wildfire.py)):
```python
is_zero = v_sq_raw < GRAD_EPS
v_zero_safe = jnp.array([jnp.sqrt(GRAD_EPS / 2.0), jnp.sqrt(GRAD_EPS / 2.0)])
v_safe = jnp.where(is_zero, v_zero_safe, v)
...
return jnp.where(is_zero, 0.0, cost)
```

**Verdict:** WARNING

**Notes:**  
`jnp.where` in JAX does not short-circuit differentiation: both branches contribute to the VJP. When `is_zero = True`, the output is pinned to `0.0`, but the gradient $\partial(\text{output})/\partial v$ is computed by differentiating through the `cost` branch at `v_safe = (\sqrt{\epsilon/2}, \sqrt{\epsilon/2})$. This gradient points strictly in the 45° direction regardless of the local metric anisotropy. For an anisotropic Randers metric ($G \ne I$), the true directional derivative of $F$ as $v \to 0$ is metric-dependent; fixing the fallback direction to 45° introduces a spurious anisotropic bias in the gradient at near-zero velocities.

In practice the AVBD solver initialises paths by linear interpolation and never generates exactly-zero velocity segments for non-coincident endpoints, so this code path is rarely hit during training. However, if it is triggered (e.g., by a degenerate initialisation where two consecutive vertices collapse), the 45° gradient bias can steer subsequent path updates in a geometrically incorrect direction.

**Recommended Action (low priority):** Replace the fixed diagonal fallback with an isotropic radial fallback $v_{\text{safe}} = \sqrt{\epsilon}\, e_1$ where $e_1$ is the first canonical basis vector, plus a separate path-vertex degeneracy guard upstream.

---

## Open Questions

1. **Boundary handling policy:** Is it a design decision that path vertices outside the raster domain receive zero metric gradient (boundary dead-gradient, §6 WARNING)? If so, should the AVBD solver enforce a domain constraint to prevent vertices from exiting? Or is the raster assumed large enough that no path reaches its boundary?

2. **IFT rcond threshold:** The `lstsq` call uses `rcond=1e-4` (gradient amplification capped at $10^4$). For a trained metric where the path energy Hessian $\nabla^2_x \mathcal{E}$ has a near-zero eigenvalue (e.g., a very flat metric in some direction), this truncation introduces a biased gradient estimate. Is there a principled choice for `rcond` based on the metric eigenvalue bounds `eps_G` and `max_G`?

3. **Additive vs. multiplicative global–local combination:** The additive form `raw = raw_global + raw_local` makes the weather contribution a position-independent translation in raw-parameter space. This cannot model interactions such as "wind effect scaled by terrain roughness." Is this a deliberate architectural choice, or was a more expressive (e.g., FiLM-style) combination considered?

4. **`project_spd` gradient stability:** Is there a systematic check (e.g., gradient norm monitoring) that would catch the near-isotropic gradient spike described in §5? This failure mode would manifest as a sudden spike in parameter gradient norms, not as a NaN.
