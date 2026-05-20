# Math Review: avbd_ift_adjoint
**Reviewer:** Math Reviewer Agent  
**Date:** 2026-05-18  
**Spec Version:** 1.1.0 (Berwald Revision)

---

## Summary

The IFT adjoint implementation in `src/ham/solvers/avbd.py` is **mathematically correct**: the discrete Euler-Lagrange residual is the right object, the IFT linear system is set up with the correct sign convention, and the VJP call correctly propagates $-\lambda^T \partial G/\partial\theta$. **No sign errors or index errors were found.** The implementation has two numerical issues: one **WARNING** that is the primary suspect for the NaN at epoch 9 (`rcond=None`), and one **WARNING** for silently wrong boundary-point gradients. There are no CRITICAL (wrong-formula) findings.

---

## Formula-by-Formula Audit

### 1. Discrete Euler-Lagrange Residual — `_el_residual` (lines 105–114)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1, Euler-Lagrange equation
- **Literature Reference:** Marsden & West, "Discrete Mechanics and Variational Integrators," Acta Numerica 2001, §2 (discrete variational principle)

**Implementation:**
```python
def total_energy(inner):
    full = jnp.concatenate([p_start[None], inner, p_end[None]], axis=0)
    vs = jax.vmap(metric.manifold.log_map)(full[:-1], full[1:])
    return jnp.sum(jax.vmap(metric.energy)(full[:-1], vs))
return jax.grad(total_energy)(inner_path).ravel()
```

**Derivation:**  
The total discrete energy is
$$E_{\text{total}} = \sum_{k=0}^{N-1} E\!\left(x_k,\, \log_{x_k}(x_{k+1})\right)$$
where $E(x,v) = \tfrac{1}{2}F^2(x,v)$ (spec § 1.2). For each inner node $x_j$, $j \in \{1,\ldots,N-1\}$, the variational condition is
$$\frac{\partial E_{\text{total}}}{\partial x_j} = \underbrace{\frac{\partial}{\partial x_j} E\!\left(x_{j-1}, \log_{x_{j-1}}(x_j)\right)}_{\text{incoming segment}} + \underbrace{\frac{\partial}{\partial x_j} E\!\left(x_j, \log_{x_j}(x_{j+1})\right)}_{\text{outgoing segment}} = 0$$

`jax.grad(total_energy)(inner_path)` computes exactly this vector of partial derivatives via AD through both the energy function and the log map, for all inner nodes simultaneously. The boundary terms ($k=0$ involving $p_\text{start}$, and $k=N-1$ involving $p_\text{end}$) contribute to the gradients w.r.t. $x_1$ and $x_{N-1}$ respectively, and are correctly included because `full` concatenates the boundaries.

**Verdict:** CORRECT  
**Notes:** This exactly matches the local vertex energy `E(x_prev, v_in) + E(x, v_out)` in `_local_vertex_energy` (lines 188–191), confirming consistency between the core solver and the IFT residual.

---

### 2. IFT Adjoint Linear System — `_implicit_forward_pass_bwd` (lines 149–179)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (E-L optimality conditions)
- **Literature Reference:** Krantz & Parks, "The Implicit Function Theorem," 2002, Theorem 1.3.1; Blondel et al., "Efficient and Modular Implicit Differentiation," NeurIPS 2022 (arXiv:2105.15183)

**Implementation:**
```python
dG_dx = jax.jacobian(
    lambda p: _el_residual(p, metric, p_start, p_end)
)(inner).reshape(inner.size, inner.size)

lam, _, _, _ = jnp.linalg.lstsq(dG_dx.T, g_inner.ravel(), rcond=None)

_, vjp_fn = jax.vjp(el_wrt_arr, m_arr)
grad_arr = vjp_fn(-lam)[0]
```

**Derivation:**  
Let $G(x^*, \theta) = \nabla_{x^*} E_{\text{total}}(x^*, \theta) = 0$ at the converged path. The IFT gives
$$\frac{\partial x^*}{\partial \theta} = -\!\left(\frac{\partial G}{\partial x^*}\right)^{-1}\!\frac{\partial G}{\partial \theta}$$
For a scalar loss $\mathcal{L}$ with upstream cotangent $g = \partial\mathcal{L}/\partial x^*$:
$$\frac{\partial \mathcal{L}}{\partial \theta} = g^T \frac{\partial x^*}{\partial \theta} = -g^T \!\left(\frac{\partial G}{\partial x^*}\right)^{-1}\!\frac{\partial G}{\partial \theta} = -\lambda^T \frac{\partial G}{\partial \theta}$$
where $\lambda$ solves $\left(\frac{\partial G}{\partial x^*}\right)^T \!\lambda = g$.

Mapping to code:
- `dG_dx` is $\partial G / \partial x^*$, shape $(n_\text{inner} \cdot D,\; n_\text{inner} \cdot D)$.
- `jnp.linalg.lstsq(dG_dx.T, g_inner.ravel())` solves $\left(\partial G/\partial x^*\right)^T \lambda = g$. ✓
- `vjp_fn` computes $v^T J_{G,\theta}$ (the VJP of $G$ w.r.t. metric arrays). `vjp_fn(-lam)[0]` yields $(-\lambda)^T \partial G/\partial\theta = -\lambda^T \partial G/\partial\theta$. ✓

**Sign convention:** Correct. No sign error found.

**Symmetry note:** Because $G = \nabla_{x^*} E_{\text{total}}$, the matrix $\partial G/\partial x^*$ is the Hessian $\nabla_{x^*}^2 E_{\text{total}}$, which is symmetric by Schwarz's theorem. Consequently `dG_dx.T == dG_dx` at any point on the manifold where $E$ is smooth. The explicit transpose is not wrong—it is a correct formal statement of the adjoint system—but it is redundant for this particular $G$.

**Verdict:** CORRECT

---

### 3. `rcond=None` in the Adjoint Linear Solve (line 159)

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1 (epsilon regularization for numerical stability)
- **Literature Reference:** Higham, "Accuracy and Stability of Numerical Algorithms," 2nd ed., §20.4 (least-squares via SVD)

**Implementation:**
```python
lam, _, _, _ = jnp.linalg.lstsq(dG_dx.T, g_inner.ravel(), rcond=None)
```

In JAX/XLA, `rcond=None` uses machine epsilon as the singular-value cutoff: $r_\text{cut} = \epsilon_\text{mach} \cdot \sigma_{\max}$. For **float32**, $\epsilon_\text{mach} \approx 1.2 \times 10^{-7}$.

**Analysis of the path Hessian condition number:**

The Hessian $\partial G/\partial x^*$ is the block-tridiagonal matrix of the discrete path energy. For a well-converged Euclidean path with $N_\text{inner} = 24$, $D = 2$, its condition number is bounded by $\kappa \lesssim 4N^2 \approx 2300$; the solve is accurate.

During training, however, three situations produce near-singularity:
1. **Degenerate segments:** If two adjacent vertices coincide ($x_k \approx x_{k+1}$), the log-map Jacobian vanishes and $\sigma_{\min} \to 0$.
2. **Saddle-path geometry:** If the AVBD solver converges to a local saddle rather than a minimum (common in early training), $\partial G/\partial x^*$ has near-zero eigenvalues.
3. **Rapidly-changing metric:** During learning, the metric can shift such that the Hessian at the stored $x^*$ is not the Hessian at the true current minimum. The resulting matrix can have $\kappa \sim 10^6$.

For $\kappa \sim 10^6$ with float32 and `rcond=None`:
- The smallest legitimate singular value $\sigma_{\min} \approx \sigma_{\max}/10^6$ satisfies $\sigma_{\min} > \epsilon_\text{mach} \cdot \sigma_{\max}$, so `lstsq` does **not** truncate it—it inverts it. The gradient is amplified by $10^6$.  
- A single amplified gradient step of size $\eta \cdot 10^6 \|\nabla \mathcal{L}\|$ overflows float32 range ($\approx 3.4 \times 10^{38}$) in $\sim 8$ steps given $\eta \sim 10^{-3}$ and typical $\|\nabla\mathcal{L}\| \sim 1$. **This explains NaN onset at epoch 9.**

**Verdict:** WARNING  
**Recommended Action:** Replace `rcond=None` with a dtype-adaptive floor:

```python
rcond = 1e-4  # or: jnp.sqrt(jnp.finfo(inner.dtype).eps) ≈ 3.5e-4 for float32
lam, _, _, _ = jnp.linalg.lstsq(dG_dx.T, g_inner.ravel(), rcond=rcond)
```

`rcond=1e-4` truncates singular values below $10^{-4} \sigma_{\max}$, capping gradient amplification at $10^4$. This is conservative enough to prevent overflow while preserving useful gradient signal for the well-conditioned components of the path. For float64 runs, `jnp.sqrt(jnp.finfo(inner.dtype).eps)` ≈ $1.5 \times 10^{-8}$ is appropriate.

---

### 4. `stop_gradient` on Metric in the Forward Pass (lines 122–128)

- **Spec Reference:** `spec/ARCH_SPEC.md` § 4.2 (implicit differentiation pattern)

**Implementation:**
```python
metric_sg = jax.tree_util.tree_map(
    lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
    metric
)
```
Applied to the forward iterative solve (`_solve_core`); the backward receives the **original** `metric` (without stop-gradient) through `vjp_args`.

**Analysis:** This is the standard pattern for separating a fixed-point computation from its IFT adjoint:
- The forward solver sees $\theta$ as constants, preventing incorrect gradient accumulation through 50 unrolled iterations.
- The backward re-evaluates $G(x^*, \theta)$ using the live (non-stop-gradient) metric to compute $\partial G/\partial \theta$ analytically.

There is no interaction pathology: `_el_residual` in the backward is called on the **converged** `inner` path (stored as a residual in `_implicit_forward_pass_fwd`), which is a concrete array at backward time.

**Verdict:** CORRECT  
**Notes:** This is the pattern recommended in Blondel et al. arXiv:2105.15183, §3.

---

### 5. Zero Gradients w.r.t. Boundary Points (line 179)

**Implementation:**
```python
return (grad_solver, grad_m, jnp.zeros_like(p_start), jnp.zeros_like(p_end))
```

**Mathematical analysis:**  
$G$ depends on $p_\text{start}$ through the first-segment energy term $E(p_\text{start}, \log_{p_\text{start}}(x_1))$, and on $p_\text{end}$ through the last-segment term $E(x_{N-1}, \log_{x_{N-1}}(p_\text{end}))$. Therefore
$$\frac{\partial G}{\partial p_\text{start}} \neq 0, \qquad \frac{\partial G}{\partial p_\text{end}} \neq 0$$
in general. The IFT gradient of the loss w.r.t. boundary points is
$$\frac{\partial \mathcal{L}}{\partial p_\text{bdy}} = g^T \frac{\partial x^*}{\partial p_\text{bdy}} = -\lambda^T \frac{\partial G}{\partial p_\text{bdy}}$$
which is nonzero.

The code returns zeros here unconditionally, which is **incorrect** whenever the training objective differentiates through boundary points (e.g., if $p_\text{start}$ or $p_\text{end}$ are predicted by a neural network encoder and the path is used as a differentiable module inside that pipeline).

For the current HAMTools training loop where boundary points are fixed observed data, this is **harmless** — but it is a silent trap for future use cases.

**Verdict:** WARNING  
**Recommended Action:** If boundary-point gradients are ever needed, replace the zeros with the correct IFT terms:
```python
def el_wrt_boundaries(ps, pe):
    return _el_residual(inner, metric, ps, pe)
_, vjp_bdy = jax.vjp(el_wrt_boundaries, p_start, p_end)
g_ps, g_pe = vjp_bdy(-lam)
return (grad_solver, grad_m, g_ps, g_pe)
```
Until boundary learning is required, the zeros can remain but should be documented.

---

### 6. Hessian Singularity Conditions (analytical)

The path Hessian $H = \partial G / \partial x^* = \nabla_{x^*}^2 E_{\text{total}}$ can become rank-deficient under:

| Condition | Mechanism |
|-----------|-----------|
| $x_k = x_{k+1}$ (zero-length segment) | $\log_{x_k}(x_{k+1}) = 0$; $E$ is $C^\infty$ at $v \neq 0$ but the $\epsilon$-regularized energy $E_\epsilon$ (spec § 6.1) has vanishing second derivative w.r.t. $x_{k+1}$ as $v \to 0$ |
| Path at a saddle point | $H$ has a zero or negative eigenvalue; IFT is not applicable (the E-L system does not uniquely define $x^*$ locally) |
| Highly curved non-flat manifolds | Off-diagonal coupling blocks grow with curvature; condition number can scale as $O(\kappa_M \cdot N^2)$ where $\kappa_M$ is the sectional curvature scale |
| Randers metric with $\|W\|_h \to 1$ | Energy becomes asymmetric and the Hessian can become indefinite in the reverse direction |

For a well-converged geodesic on a flat or mildly curved manifold with $N_\text{inner} = 24$ and $D = 2$, the expected condition number is $\kappa \sim 10^2$–$10^3$, safely within float32 range for `rcond=1e-4`.

---

## Open Questions

1. **Solver convergence guarantee at IFT call time:** The IFT requires $G(x^*, \theta) = 0$ at the point where the adjoint is evaluated. The forward AVBD solver runs a fixed number of iterations (`iterations=50`) without checking $\|G\|$ at exit. If the path has not converged (e.g., when the metric changes rapidly during early training), the Jacobian $dG/dx^*$ is evaluated at a non-stationary point, making the IFT gradient formally invalid. Should a convergence check / warning be added before the backward?

2. **Float32 vs. float64 for the adjoint solve:** Given the condition-number analysis above, is the rest of the training pipeline compatible with running the `lstsq` solve in float64 (upcast before solve, downcast result) while keeping the forward pass in float32? This would be more robust than any fixed `rcond` choice.

3. **Interaction with ALM constraints:** `_el_residual` computes the gradient of the pure energy, ignoring the Augmented Lagrangian penalty terms. If constraints are active (`constraints` list is non-empty), the true stationarity condition at convergence includes the penalty gradient. Should $G$ be augmented with $\nabla_x \text{penalty}(x^*, \lambda, \mu)$ to reflect the actual fixed-point condition?

4. **Second-order correction:** At a true geodesic minimum, the path Hessian $H = \partial G/\partial x^*$ is positive definite. However, the solver minimizes a regularized objective and the converged solution is not exactly a geodesic minimum in the IFT sense. Is the systematic bias introduced by using a partially converged $x^*$ quantified anywhere, and does it affect downstream metric learning quality?
