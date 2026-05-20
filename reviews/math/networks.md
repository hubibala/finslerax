# Math Review: networks.py

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**File:** `src/ham/nn/networks.py`

## Summary

**Verdict: Correct — Minor Issues.**  
All three modules (`RandomFourierFeatures`, `VectorField`, `PSDMatrixField`) are mathematically sound. The PSD construction $G = AA^T + \epsilon I$ correctly guarantees symmetric positive-definite output, which is the critical requirement for downstream Riemannian/Randers energy computation. Two warnings are raised: (1) the $AA^T$ parametrisation is over-parameterised compared to a Cholesky factor, which may slow optimisation but does not affect correctness; (2) the missing $\sqrt{2/D}$ scaling in RFF deviates from the standard kernel-approximation result, though it is absorbed by learned layers and therefore benign.

---

## Formula-by-Formula Audit

### 1. RandomFourierFeatures — Fourier embedding

- **Spec Reference:** Not explicitly in `MATH_SPEC.md`. This is a neural-network building block, not a geometric object.
- **Literature Reference:** Rahimi & Recht, "Random Features for Large-Scale Kernel Machines," NeurIPS 2007. The canonical map is $z(x) = \sqrt{2/D}\,[\cos(\omega_1^T x + b_1), \ldots]$ with $\omega_i \sim \mathcal{N}(0, \sigma^{-2} I)$, $b_i \sim \mathrm{Uniform}(0, 2\pi)$.
- **Implementation** (`src/ham/nn/networks.py:14–19`):
  ```python
  self.B = jax.random.normal(key, (mapping_size, in_dim)) * scale
  ...
  projected = jnp.dot(self.B, x)
  return jnp.concatenate([jnp.cos(projected), jnp.sin(projected)], axis=0)
  ```
- **Verdict:** OK
- **Notes:**
  The $[\cos(Bx),\, \sin(Bx)]$ variant is a well-known equivalent of the $\cos(Bx + b)$ form that eliminates the random bias term. This is mathematically correct.

  **NOTE (severity: NOTE):** The standard RFF approximation to a shift-invariant kernel requires a $\sqrt{2/D}$ scaling factor for an unbiased kernel estimate. The implementation omits this factor. In a fully learned pipeline where subsequent linear/MLP layers absorb arbitrary scaling, this has no mathematical consequence — the kernel interpretation is lost, but the function-approximation capacity is preserved.

---

### 2. VectorField — Learnable vector field $W: \mathbb{R}^D \to \mathbb{R}^D$

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 — the wind field $W^i(x)$ in the Zermelo parameterisation.
- **Implementation** (`src/ham/nn/networks.py:22–54`):
  ```python
  self.mlp = eqx.nn.MLP(
      in_size=in_size, out_size=dim,
      width_size=hidden_dim, depth=depth,
      activation=jax.nn.tanh, key=k_mlp
  )
  ```
- **Verdict:** OK
- **Notes:**
  1. **Output dimension:** Correctly matches the ambient dimension $D$, consistent with $W^i(x) \in T_x M \cong \mathbb{R}^D$.
  2. **Activation choice:** `tanh` is $C^\infty$, which is essential because the Finsler spray computation requires at least second-order differentiation through the metric, and the Berwald connection requires third-order. A non-smooth activation (e.g., ReLU) would produce ill-defined higher derivatives. The choice is mathematically sound.
  3. **No norm constraint here:** The causality constraint $\|W\|_h < 1$ is enforced downstream in `Randers._get_zermelo_data()` (`src/ham/geometry/zoo.py:97–103`), not in the network itself. This is the correct architectural separation — the network learns an unconstrained field and the metric class squashes it.

---

### 3. PSDMatrixField — Learnable SPD matrix field $G: \mathbb{R}^D \to \mathrm{Sym}^{++}(D)$

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1, Definition 3 (strong convexity / positive-definiteness of $g_{ij}$); § 4 (Riemannian row: $F(x,v) = \sqrt{v^T g(x) v}$, requiring $g(x) \succ 0$).
- **Literature Reference:** Standard Gram-matrix parametrisation for SPD learning, e.g., Chen et al. "Metrics for Deep Generative Models," AISTATS 2018.
- **Implementation** (`src/ham/nn/networks.py:87–98`):
  ```python
  flat_A = self.mlp(x)
  A = flat_A.reshape(self.dim, self.dim)
  G = jnp.dot(A, A.T) + 1e-4 * jnp.eye(self.dim)
  ```

#### 3a. Symmetry

- **Claim:** $G = AA^T + \epsilon I$ is symmetric.
- **Proof:** $(AA^T)^T = A^{TT}A^T = AA^T$, and $I^T = I$. Therefore $G^T = G$. ✓
- **Verdict:** OK

#### 3b. Positive-Definiteness

- **Claim:** $G \succ 0$ for all $A$ and $\epsilon > 0$.
- **Proof:** For any $v \neq 0$:
  $$v^T G v = v^T AA^T v + \epsilon\, v^T v = \|A^T v\|^2 + \epsilon \|v\|^2 \geq \epsilon \|v\|^2 > 0$$
  The minimum eigenvalue satisfies $\lambda_{\min}(G) \geq \epsilon = 10^{-4}$.
- **Verdict:** OK
- **Notes:** The construction correctly guarantees the strong-convexity requirement (MATH_SPEC § 1.1, condition 3) for the fundamental tensor $g_{ij}(x)$.

#### 3c. Over-parametrisation of the factor $A$

- **Verdict:** WARNING
- **Notes:**
  The MLP outputs $D^2$ values for a full $D \times D$ matrix $A$. However, $\mathrm{Sym}^{++}(D)$ has dimension $D(D+1)/2$. A Cholesky parametrisation—outputting a lower-triangular $L$ with $\mathrm{softplus}$-enforced positive diagonal and constructing $G = LL^T$—would:
  1. Reduce the parameter count from $D^2$ to $D(D+1)/2$.
  2. Avoid the rank-ambiguity inherent in $A \mapsto AA^T$ (the map has a $O(D)$ symmetry group: $A$ and $AQ$ yield the same $G$ for any orthogonal $Q$).
  3. Eliminate the need for the $\epsilon I$ regulariser (positive diagonal of $L$ directly guarantees $G \succ 0$).

  The current parametrisation is **not mathematically wrong** — it still maps surjectively onto $\mathrm{Sym}^{++}(D)$ and produces valid metrics. The over-parametrisation is an optimisation-landscape concern (redundant degrees of freedom, flat directions in loss surface) rather than a correctness issue.

  **Recommended Action:** Consider replacing with Cholesky parametrisation for tighter optimisation, especially for $D > 3$.

#### 3d. Interaction with downstream symmetrisation

- **Verdict:** NOTE
- **Notes:**
  The `Riemannian.metric_fn()` in `src/ham/geometry/zoo.py:36` applies an additional symmetrisation step:
  ```python
  G_x = 0.5 * (G_x + G_x.T)
  ```
  This is redundant since $AA^T$ is already exactly symmetric (both analytically and in IEEE 754 arithmetic, because elements $(i,j)$ and $(j,i)$ involve identical multiply-accumulate sequences). The redundancy is harmless and acts as a defensive measure.

#### 3e. Epsilon magnitude

- **Verdict:** NOTE
- **Notes:**
  The regularisation $\epsilon = 10^{-4}$ in PSDMatrixField matches the canonical constant `PSD_EPS = 1e-4` defined in `src/ham/utils/math.py:9`. The spray solver in `FinslerMetric.spray()` (`src/ham/geometry/metric.py:59`) adds its own $10^{-4} I$ to the velocity Hessian before solving. For Riemannian metrics, the velocity Hessian **is** $G(x)$ itself, so the effective minimum eigenvalue at solve time is $2 \times 10^{-4}$, which is adequate for `float32` precision.

---

### 4. Compatibility with MATH_SPEC § 6.2 — Homogeneity Enforcement

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.2.
- **Implementation:** Not in `networks.py`.
- **Verdict:** OK
- **Notes:**
  The spec requires positive 1-homogeneity $F(x, \lambda v) = \lambda F(x,v)$. For the Riemannian case, the metric $F(x,v) = \sqrt{v^T G(x) v}$ is automatically 1-homogeneous in $v$ because $G(x)$ is independent of $v$. The `PSDMatrixField` outputs $G(x)$ — a function of position only — so no explicit homogeneity enforcement is needed at the network level. The spec's § 6.2 formula $F_\text{net}(x,v) = \|v\| \cdot \text{NN}(x, v/\|v\|)$ applies only to networks that directly predict $F(x,v)$, which is not the architecture used here.

---

### 5. Smoothness for higher-order auto-differentiation

- **Spec Reference:** `spec/MATH_SPEC.md` § 2 (spray requires 2nd derivatives of $E$), § 3 (Berwald connection requires 3rd derivatives of $E$).
- **Verdict:** OK
- **Notes:**
  All three modules use `tanh` activations, which are $C^\infty$. The spray computation requires:
  - $\nabla_v E$, $\text{Hess}_v(E)$: for Riemannian $E = \frac{1}{2} v^T G(x) v$, these are $G(x)v$ and $G(x)$, both trivially smooth.
  - $\nabla_x E$: requires differentiating through the neural network w.r.t. its input $x$. With `tanh` activations, this is $C^\infty$.
  
  The Berwald connection ${}^B\Gamma^i_{jk} = \partial^2 G^i / \partial v^j \partial v^k$ requires third derivatives of $E$ w.r.t. $v$, which vanish identically for Riemannian metrics (since $E$ is quadratic in $v$). This confirms that the Berwald connection reduces to the Levi-Civita connection as stated in MATH_SPEC § 4.

---

## Open Questions

1. **Cholesky vs. Gram parametrisation trade-off:** The $AA^T + \epsilon I$ construction is correct but over-parameterised. Has any empirical comparison been run against a Cholesky ($LL^T$ with softplus diagonal) variant? The optimisation landscape may differ significantly for $D \geq 5$.

2. **Conditioning under learning:** If the learned $A$ has very large singular values, the condition number $\kappa(G) = \lambda_{\max}/\lambda_{\min}$ can grow unboundedly (only $\lambda_{\min} \geq 10^{-4}$ is guaranteed). In the spray solve $G \cdot (2G^i) = \text{rhs}$, this could cause numerical instability. Should an upper bound on eigenvalues (e.g., spectral normalisation of $A$) be considered?

3. **RFF scale parameter:** The default `fourier_scale=10.0` in `VectorField` versus `1.0` in `PSDMatrixField` reflects the heuristic that wind fields are "less smooth" than metric fields. Is there empirical or theoretical justification for these specific values, or are they tuning artefacts?
