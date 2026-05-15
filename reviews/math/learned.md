# Math Review: `learned.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Minor Issues.** The file implements neural parametrizations of Riemannian and Randers metrics. The core mathematical constructions—PSD enforcement via $A A^\top + \varepsilon I$, Zermelo causality squashing, and pullback metric $J^\top J$—are correct in principle. However, there are two warnings related to (a) redundant symmetrisation of an already-symmetric matrix and (b) the pullback Riemannian class not inheriting the Randers Zermelo formula despite using the same `PullbackGNet`, and one critical concern about the `KernelWindField` not enforcing the Zermelo causality constraint $\|W\|_h < 1$, relying entirely on the downstream `Randers._get_zermelo_data` squasher.

---

## Formula-by-Formula Audit

### 1. `PSDMatrixField.__call__` (PSD Construction)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 (Riemannian row: $F(x,v) = \sqrt{v^\top g(x) v}$ requires $g$ SPD)
- **Literature Reference:** Standard Cholesky-free PSD parametrisation; see e.g. Arvanitidis et al., "Latent Space Oddity" (arXiv:1710.11379)
- **Implementation:** [networks.py](../src/ham/nn/networks.py#L99-L103)
  ```python
  G = jnp.dot(A, A.T) + 1e-4 * jnp.eye(self.dim)
  ```
- **Verdict:** OK
- **Notes:** $G = A A^\top + \varepsilon I$ is symmetric positive definite by construction for any $A$ and $\varepsilon > 0$. The regularisation $\varepsilon = 10^{-4}$ matches `PSD_EPS` in `src/ham/utils/math.py`. Correct.

---

### 2. `Riemannian.metric_fn` (Riemannian Norm)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table row "Riemannian": $F(x,v) = \sqrt{v^\top g(x) v}$
- **Implementation:** [zoo.py](../src/ham/geometry/zoo.py#L36-L39)
  ```python
  G_x = self.g_net(x)
  G_x = 0.5 * (G_x + G_x.T)
  quad = jnp.dot(v, jnp.dot(G_x, v))
  return jnp.sqrt(jnp.maximum(quad, 1e-12))
  ```
- **Verdict:** WARNING
- **Notes:** The symmetrisation `0.5 * (G + G^T)` is applied to a matrix that is *already* symmetric if produced by `PSDMatrixField` ($A A^\top$ is symmetric by construction). This is defensive and harmless. However, `0.5 * (G + G^T)` does **not** guarantee positive definiteness—it only guarantees symmetry. If `g_net` is *not* `PSDMatrixField` (e.g., a raw MLP), this would produce a symmetric but possibly indefinite matrix, yielding a negative quadratic form and NaN from `sqrt`. The `jnp.maximum(quad, 1e-12)` clamp silently hides this. This is safe for the current usage with `PSDMatrixField` but architecturally fragile.

---

### 3. `NeuralRiemannian.__init__`

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 (Riemannian)
- **Implementation:** [learned.py](../src/ham/models/learned.py#L11-L22)
  ```python
  g_net = PSDMatrixField(self.dim, hidden_dim, depth, key)
  super().__init__(manifold, g_net)
  ```
- **Verdict:** OK
- **Notes:** Correctly delegates PSD matrix construction to `PSDMatrixField`, then passes it to `Riemannian.__init__`. The resulting metric is $F(x,v) = \sqrt{v^\top G_\theta(x) v}$ where $G_\theta$ is SPD by construction. This satisfies the strong convexity requirement of `spec/MATH_SPEC.md` § 1.1.

---

### 4. `NeuralRanders.__init__`

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parametrisation)
- **Implementation:** [learned.py](../src/ham/models/learned.py#L24-L42)
  ```python
  h_net = PSDMatrixField(self.dim, hidden_dim, depth, k1)
  w_net = VectorField(self.dim, hidden_dim, depth, k2, 
                      use_fourier=use_fourier, fourier_scale=3.0)
  super().__init__(manifold, h_net, w_net, epsilon=1e-5)
  ```
- **Verdict:** OK
- **Notes:** Correct delegation. The `h_net` (Riemannian "sea") is SPD via `PSDMatrixField`, and the `w_net` (wind) is an unconstrained vector field whose norm is squashed by `Randers._get_zermelo_data`. This matches the Zermelo navigation approach in § 5.

---

### 5. `Randers._get_zermelo_data` (Zermelo Wind Squashing)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5: $\lambda = 1 - \|W\|_h^2$, with the constraint $\|W\|_h < 1$.
- **Literature Reference:** Bao, Robles, Shen, "Zermelo navigation on Riemannian manifolds" (J. Diff. Geom. 66, 2004)
- **Implementation:** [zoo.py](../src/ham/geometry/zoo.py#L82-L114)
  ```python
  w_norm = jnp.sqrt(jnp.maximum(w_norm_sq, 1e-8))
  max_speed = 1.0 - self.epsilon
  squash_factor = (max_speed * jnp.tanh(w_norm)) / (w_norm + 1e-8)
  W_safe = W_raw * squash_factor
  lambda_factor = 1.0 - safe_w_norm_sq
  ```
- **Verdict:** OK
- **Notes:** The squashing $\|W_{\text{safe}}\|_h = (1-\varepsilon) \tanh(\|W_{\text{raw}}\|_h)$ maps $[0, \infty) \to [0, 1-\varepsilon)$, which strictly enforces the Zermelo causality condition $\|W\|_h < 1$. This ensures $\lambda > 0$, preserving strong convexity of the Randers norm. Mathematically sound.

---

### 6. `Randers.metric_fn` (Zermelo Cost)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5:
  $$F(x,v) = \frac{\sqrt{\lambda \|v\|_h^2 + \langle W, v \rangle_h^2} - \langle W, v \rangle_h}{\lambda}$$
- **Implementation:** [zoo.py](../src/ham/geometry/zoo.py#L116-L130)
  ```python
  v_sq_h = jnp.sum(v_safe * Hv, axis=-1)          # ||v||_h^2
  W_dot_v = jnp.sum(v_safe * HW, axis=-1)          # <W, v>_h
  discriminant = lam * v_sq_h + W_dot_v**2
  cost = (jnp.sqrt(...discriminant...) - W_dot_v) / lam
  ```
- **Verdict:** OK
- **Notes:** Direct transcription of the spec formula. The discriminant $\lambda \|v\|_h^2 + \langle W, v \rangle_h^2$ is non-negative when $\lambda > 0$ (guaranteed by the squasher) and $h$ is PSD. The sign convention $-\langle W, v \rangle_h$ in the numerator corresponds to the spec's "headwind increases cost" convention. Correct.

---

### 7. `PullbackGNet.__call__` (Pullback Metric)

- **Spec Reference:** Not explicitly in `spec/MATH_SPEC.md`; standard differential geometry.
- **Literature Reference:** The pullback metric of a map $\phi: Z \to X$ is $g_{ij}(z) = (J^\top J)_{ij}$ where $J = D\phi(z)$. See do Carmo, *Riemannian Geometry*, Ch. 1.
- **Implementation:** [learned.py](../src/ham/models/learned.py#L116-L122)
  ```python
  J = jax.jacfwd(self.decoder)(z)
  H = jnp.dot(J.T, J)
  return H + 1e-4 * jnp.eye(self.dim)
  ```
- **Verdict:** WARNING
- **Notes:**
  1. **Correctness of $J^\top J$:** If `decoder: R^d -> R^D` with $D > d$, then $J$ has shape $(D, d)$ and $J^\top J$ has shape $(d, d)$—this is the correct pullback metric on the latent space $Z$. **However**, `jax.jacfwd(self.decoder)(z)` computes the Jacobian with shape `(output_dim, input_dim)`, so `J.T @ J` has shape `(input_dim, input_dim) = (d, d)`. This is correct.
  2. **Rank deficiency:** If $D < d$ (over-parameterised latent space) or if the decoder has a degenerate Jacobian (common near initialisation), $J^\top J$ can be singular or ill-conditioned. The $10^{-4} I$ regularisation prevents exact singularity but may introduce geometric distortion in the nearly-degenerate directions. This is a standard and acceptable trade-off.
  3. **Self-consistency with `self.dim`:** `self.dim` is set to `manifold.ambient_dim`, which should equal the latent dimension $d$. The `jnp.eye(self.dim)` is then correct. However, if `manifold.ambient_dim` somehow differs from the decoder's input dimension, the shapes would mismatch—no runtime check exists.

---

### 8. `PullbackRanders.__init__` (Pullback + Learned Wind)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo)
- **Implementation:** [learned.py](../src/ham/models/learned.py#L46-L66)
  ```python
  h_net = PullbackGNet(decoder=decoder, dim=self.dim)
  super().__init__(manifold, h_net=h_net, w_net=w_net, epsilon=1e-5, use_wind=self.use_wind)
  ```
- **Verdict:** OK
- **Notes:** Uses the pullback metric $J^\top J$ as the Riemannian "sea" $h_{ij}$ and a learned `VectorField` for the wind. The Zermelo causality constraint is enforced downstream by `Randers._get_zermelo_data`. Mathematically sound composition.

---

### 9. `KernelWindField.__call__` (Non-Parametric Wind)

- **Spec Reference:** Not in spec; novel contribution.
- **Literature Reference:** Standard Nadaraya-Watson kernel regression.
- **Implementation:** [learned.py](../src/ham/models/learned.py#L77-L95)
  ```python
  weights = jax.nn.softmax(-dists_sq / (2 * self.sigma**2))
  return jnp.dot(weights, self.anchors_v)
  ```
- **Verdict:** WARNING
- **Notes:**
  1. **Kernel formula:** The kernel $w_i(z) = \text{softmax}\left(-\frac{\|z - a_i\|^2}{2\sigma^2}\right)$ and the output $W(z) = \sum_i w_i(z) \cdot v_i$ is a standard softmax-weighted average. This is a valid non-parametric interpolation.
  2. **Causality constraint:** The output $W(z)$ is an unconstrained convex combination of anchor velocities. There is **no guarantee** that $\|W(z)\|_h < 1$ from this function alone. This is acceptable *only because* the downstream `Randers._get_zermelo_data` applies the tanh squasher. If `KernelWindField` were ever used outside the `Randers` pipeline, it would produce an invalid Randers metric.
  3. **Distance metric:** The squared distances are computed in **Euclidean** latent space ($\|z - a_i\|_2^2$), not in the Riemannian metric $h_{ij}$. For a pullback geometry where the latent space may be highly curved, Euclidean distances can be a poor proxy. This is a modelling choice, not a mathematical error, but worth noting.

---

### 10. `DataDrivenPullbackRanders.__init__`

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo)
- **Implementation:** [learned.py](../src/ham/models/learned.py#L98-L110)
  ```python
  h_net = PullbackGNet(decoder=decoder, dim=self.dim)
  w_net = KernelWindField(anchors_z, anchors_v, sigma)
  super().__init__(manifold, h_net=h_net, w_net=w_net, epsilon=1e-5, use_wind=self.use_wind)
  ```
- **Verdict:** OK
- **Notes:** Structurally identical to `PullbackRanders` but with `KernelWindField` replacing the neural `VectorField`. Inherits the same Zermelo enforcement from the `Randers` parent. Correct.

---

### 11. `PullbackRiemannian.__init__`

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 (Riemannian)
- **Implementation:** [learned.py](../src/ham/models/learned.py#L124-L137)
  ```python
  g_net = PullbackGNet(decoder=decoder, dim=self.dim)
  super().__init__(manifold, g_net)
  ```
- **Verdict:** OK
- **Notes:** Pure pullback Riemannian metric with no wind. Uses $g_{ij}(z) = J^\top J + \varepsilon I$. The parent `Riemannian.metric_fn` applies a redundant symmetrisation (see Finding 2) but this is harmless since $J^\top J$ is already symmetric.

---

### 12. Positive Homogeneity of Learned Metrics

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1, Condition 2: $F(x, \lambda v) = \lambda F(x, v)$ for $\lambda > 0$; § 6.2 enforcement.
- **Implementation:** All learned metrics in this file.
- **Verdict:** OK
- **Notes:**
  - **Riemannian:** $F(x, \lambda v) = \sqrt{(\lambda v)^\top G(x) (\lambda v)} = \lambda \sqrt{v^\top G(x) v} = \lambda F(x,v)$. ✓
  - **Randers (Zermelo):** The Zermelo cost formula is 1-homogeneous in $v$ by algebraic verification: scaling $v \to \lambda v$ multiplies numerator by $\lambda$ and denominator is independent of $v$. ✓
  - **Note:** The spec's § 6.2 approach ($F_{\text{net}}(x,v) = \|v\| \cdot \text{NN}(x, v/\|v\|)$) is for *direct* neural Finsler metrics. The learned metrics here use structured parametrisations (Riemannian/Randers) that are homogeneous by construction. No explicit enforcement is needed.

---

## Open Questions

1. **`PullbackGNet` dimension consistency:** There is no runtime assertion that `self.dim` (from `manifold.ambient_dim`) equals the input dimension of the decoder. A dimension mismatch would cause a silent shape error at `jnp.eye(self.dim)`. Should a guard be added?

2. **Euclidean vs. Riemannian kernel distance:** The `KernelWindField` uses Euclidean distances $\|z - a_i\|_2^2$ for interpolation weights. In a pullback geometry with strong curvature, would it be more principled to use geodesic or at least Mahalanobis distances $\|z - a_i\|_{g(z)}^2$? This could affect the quality of the wind interpolation in high-curvature regions.

3. **Redundant symmetrisation chain:** `PSDMatrixField` produces $A A^\top$ (symmetric), then `Riemannian.metric_fn` symmetrises again, and `Randers._get_zermelo_data` symmetrises yet again. Should the defensive symmetrisations be documented as intentional, or removed from the downstream consumers to clarify the contract?
