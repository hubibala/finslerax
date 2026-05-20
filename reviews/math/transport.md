# Math Review: `transport.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** [src/ham/geometry/transport.py](src/ham/geometry/transport.py)

## Summary

The core mathematical formulas — the Berwald connection ${}^B\Gamma^i_{jk} = \partial^2 G^i / \partial v^j \partial v^k$ and the parallel transport ODE $\dot{X}^i + \Gamma^i_{jk} \dot\gamma^j X^k = 0$ — are **correctly implemented**. The auto-differentiation strategy (double `jacfwd` on the spray) and the einsum contraction both produce the correct index structure. However, two **WARNING**-level issues affect numerical accuracy for embedded submanifolds and for small path discretizations: (1) a systematic off-by-one in the Euler time step, and (2) tangent-space projection at the wrong base point.

**Overall Verdict: Minor Issues.**

---

## Formula-by-Formula Audit

### 1. Berwald Connection Coefficients — `christoffel_symbols()`

- **Spec Reference:** `spec/MATH_SPEC.md` § 3.1
- **Literature Reference:** Bao–Chern–Shen, *Introduction to Riemann–Finsler Geometry* (2000), Definition 10.2.1; Szilasi–Lovas–Kertész, *Connections, Sprays and Finsler Structures* (2014), §7.3.
- **Implementation:** [transport.py:27–31](src/ham/geometry/transport.py#L27-L31)
  ```python
  jacobian_v = jax.jacfwd(self.metric.spray, argnums=1)
  hessian_v = jax.jacfwd(jacobian_v, argnums=1)
  return hessian_v(x, v)
  ```
- **Verdict:** OK
- **Notes:**

  The spec defines ${}^B\Gamma^i_{jk} = \frac{\partial^2 G^i}{\partial v^j \partial v^k}$.

  **JAX index analysis.** `jacfwd(f, argnums=1)` for `f: \mathbb{R}^D \to \mathbb{R}^D` produces a matrix of shape `(D, D)` where `result[i, j]` = $\partial f^i / \partial v^j$. Applying `jacfwd` again:

  | Expression | Shape | Index meaning |
  |---|---|---|
  | `spray(x, v)` | `(D,)` | $G^i$ |
  | `jacobian_v(x, v)` | `(D, D)` | $\partial G^i / \partial v^j$ at `[i, j]` |
  | `hessian_v(x, v)` | `(D, D, D)` | $\partial^2 G^i / (\partial v^j \partial v^k)$ at `[i, j, k]` |

  The last axis appended by `jacfwd` is the new derivative index, so `hessian_v[i, j, k]` = $\frac{\partial}{\partial v^k}\!\left(\frac{\partial G^i}{\partial v^j}\right) = \frac{\partial^2 G^i}{\partial v^j \partial v^k}$.

  This is exactly $\Gamma^i_{jk}$ as defined in the spec. By Schwarz's theorem the tensor is symmetric in $(j,k)$, consistent with the torsion-freeness property stated in `spec/MATH_SPEC.md` § 3.1. ✓

---

### 2. Parallel Transport ODE — `parallel_transport()`

- **Spec Reference:** `spec/MATH_SPEC.md` § 3.2
- **Literature Reference:** Bao–Chern–Shen (2000), §10.3 (Berwald parallel displacement); Chern–Shen, *Riemann–Finsler Geometry* (2005), §3.4.
- **Implementation:** [transport.py:43–46](src/ham/geometry/transport.py#L43-L46)
  ```python
  gamma = self.christoffel_symbols(x, v)
  dx = -jnp.einsum('ijk,j,k->i', gamma, v, carry_vec)
  ```
- **Verdict:** OK
- **Notes:**

  The spec states the Berwald parallel transport equation:

  $$\frac{dX^i}{dt} + {}^B\Gamma^i_{jk}(\gamma, \dot\gamma)\,\dot\gamma^j\,X^k = 0$$

  Rearranging: $\dot{X}^i = -\Gamma^i_{jk}\,v^j\,X^k$ where $v = \dot\gamma$.

  The einsum `'ijk,j,k->i'` computes $\sum_{j,k} \Gamma^i_{jk}\,v^j\,X^k$, which is contracted against the correct indices. The minus sign is applied via the prefix. ✓

  The connection is evaluated at $(\gamma(t), \dot\gamma(t))$ by passing `(x, v)` from the path arrays, correctly reflecting the velocity-dependence of the Finsler connection. ✓

---

### 3. Euler Discretization Time Step

- **Spec Reference:** No explicit spec for the integrator.
- **Literature Reference:** Standard forward Euler: $X_{k+1} = X_k + \Delta t \cdot f(t_k, X_k)$.
- **Implementation:** [transport.py:48–49](src/ham/geometry/transport.py#L48-L49)
  ```python
  dt = 1.0 / len(path_x)
  new_vec = carry_vec + dx * dt
  ```
- **Verdict:** WARNING
- **Notes:**

  For a path with $N$ equally-spaced sample points from $t=0$ to $t=1$, the inter-point spacing is $\Delta t = 1/(N-1)$, not $1/N$. The scan is iterated $N$ times but the result discards the last output (line 57: `transported_vecs[:-1]`), effectively integrating $N-1$ steps. Each step is scaled by $1/N$ instead of the correct $1/(N-1)$, giving a total integrated time of $(N-1)/N$ instead of $1$.

  **Error magnitude:** The multiplicative error factor is $\frac{N-1}{N} = 1 - \frac{1}{N}$. For the test suite's $N=20$ this is a 5% systematic underestimation of the transport; for $N=50$ it is 2%. This compounds with path curvature.

  **Impact on tests:** The sphere holonomy test (`test_sphere_holonomy`, $N=200$) uses `atol=1e-1` which masks the 0.5% dt error. The Euclidean test passes trivially since $\Gamma=0$.

  **Recommended Action:** Change to `dt = 1.0 / (len(path_x) - 1)` or, better, accept `dt` (or the path parameter values) as an input.

---

### 4. Tangent-Space Projection Base Point

- **Spec Reference:** `spec/MATH_SPEC.md` § 3.2 — the transported vector $X(t_{k+1})$ must lie in $T_{\gamma(t_{k+1})}M$.
- **Implementation:** [transport.py:51](src/ham/geometry/transport.py#L51)
  ```python
  new_vec = self.metric.manifold.to_tangent(x, new_vec)
  ```
- **Verdict:** WARNING
- **Notes:**

  After the Euler step, `new_vec` approximates $X(t_{k+1})$, which should be tangent at the *next* path point $\gamma(t_{k+1})$. However, `to_tangent(x, ...)` projects onto $T_xM = T_{\gamma(t_k)}M$ — the tangent space at the *current* point.

  For flat manifolds ($\mathbb{R}^n$) the tangent spaces are all identical, so this is harmless. For embedded submanifolds (sphere, hyperboloid), the tangent spaces at $\gamma(t_k)$ and $\gamma(t_{k+1})$ differ by $O(\Delta t \cdot \kappa)$ where $\kappa$ is the extrinsic curvature. This introduces an additional $O(\Delta t^2)$ error per step on top of Euler's $O(\Delta t^2)$, so the overall order of the integrator is unchanged, but the constant factor worsens.

  The fundamental problem is that the `jax.lax.scan` body receives `path_x[k]` but not `path_x[k+1]`, making it impossible to project at the correct point without restructuring the iteration.

  **Recommended Action:** Restructure the scan inputs so that the body receives pairs $(x_k, x_{k+1}, v_k)$, and project `new_vec` with `to_tangent(x_{k+1}, new_vec)`. Alternatively, use `path_x[1:]` as the projection points.

---

### 5. Return Value Assembly

- **Implementation:** [transport.py:56–57](src/ham/geometry/transport.py#L56-L57)
  ```python
  result = jnp.concatenate([vec_start[None, :], transported_vecs[:-1]], axis=0)
  ```
- **Verdict:** OK
- **Notes:**

  The scan produces $N$ outputs: `transported_vecs[k]` = $X_{k+1}$ (the vector after the $(k+1)$-th Euler step). The code constructs:

  $$\text{result} = [X_0,\; X_1,\; \ldots,\; X_{N-1}]$$

  where $X_0$ = `vec_start` is the initial vector at $\gamma_0$ and $X_k$ is the transported vector at $\gamma_k$. This gives one vector per path point, which is the expected output shape `(N, D)`. ✓

  The final scan output `transported_vecs[-1]` = $X_N$ (one step beyond the path) is correctly discarded.

---

### 6. Integrator Order and Geometric Fidelity

- **Spec Reference:** `spec/MATH_SPEC.md` § 6 (numerical stability considerations)
- **Literature Reference:** Hairer–Lubich–Wanner, *Geometric Numerical Integration* (2006), Chapter IV.
- **Verdict:** NOTE
- **Notes:**

  The implementation uses **forward Euler** (1st-order). For the parallel transport ODE on a Riemannian manifold, the Levi-Civita connection preserves the inner product $g(\gamma(t))(X(t), Y(t)) = \text{const}$. Forward Euler does not preserve this invariant; the transported vector norm drifts at rate $O(\Delta t)$ per step, accumulating to $O(1)$ over a unit-length path unless $N$ is large.

  For the Berwald connection on a non-Riemannian Finsler space, norm drift is *expected* (this is the correct physics, as noted in `spec/MATH_SPEC.md` § 3.1: "Non-Metric"). But for the Riemannian subcase, where the Berwald connection reduces to Levi-Civita, the integrator's numerical norm drift could be confused with physical drift.

  The test `test_riemannian_sphere_isometry` checks norm preservation with `atol=1e-5`, which succeeds only because $N=20$ and the path is short. A longer path or coarser discretization would fail.

  **Recommended Action (low priority):** Consider a midpoint or RK4 integrator for improved accuracy. For the Riemannian subcase, a norm-preserving integrator (e.g., Cayley retraction on orthogonal frames) would be ideal but may be over-engineering for the current use case.

---

### 7. Interaction: Spray Regularization Propagation

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1
- **Implementation:** The spray in [metric.py:62](src/ham/geometry/metric.py#L62) uses Tikhonov regularization $g + \epsilon I$ with $\epsilon = 10^{-4}$.
- **Verdict:** NOTE
- **Notes:**

  The Berwald connection $\Gamma^i_{jk} = \partial^2 G^i / \partial v^j \partial v^k$ inherits the regularization because it differentiates *through* the regularized spray. Since the regularization constant is independent of $v$, its second $v$-derivative is zero — so the Tikhonov term does not contribute additive bias directly to $\Gamma$. However, the regularization modifies $G^i$ itself (shifting the spray solution), and the Hessian of this shift w.r.t. $v$ is generally nonzero.

  For well-conditioned metrics the effect is $O(\epsilon)$. For ill-conditioned metrics near $v=0$, the spray is heavily regularized and the resulting connection may deviate significantly from the true Berwald connection.

  No action needed in `transport.py`; this is documented in the `metric.py` review ([reviews/math/metric.md](reviews/math/metric.md), Finding #4).

---

## Correctness of the `berwald_transport` Wrapper

- **Implementation:** [transport.py:62–64](src/ham/geometry/transport.py#L62-L64)
  ```python
  def berwald_transport(metric, path_x, path_v, vec_start):
      return BerwaldConnection(metric).parallel_transport(path_x, path_v, vec_start)
  ```
- **Verdict:** OK
- **Notes:** Thin wrapper, no mathematical content. Correctly delegates.

---

## Cross-File Consistency

| Dependency | File | Status |
|---|---|---|
| `self.metric.spray(x, v)` | [metric.py:42–66](src/ham/geometry/metric.py#L42-L66) | Verified correct in [reviews/math/metric.md](reviews/math/metric.md) |
| `self.metric.manifold.to_tangent(x, v)` | [manifold.py:85–95](src/ham/geometry/manifold.py#L85-L95) | Abstract; subclass correctness assumed |
| Berwald definition $\Gamma = \partial^2_v G$ | `spec/MATH_SPEC.md` § 3.1 | Exact match ✓ |
| Transport ODE $\dot{X}^i + \Gamma^i_{jk} v^j X^k = 0$ | `spec/MATH_SPEC.md` § 3.2 | Exact match ✓ |

---

## Summary of Findings

| # | Severity | Location | Issue |
|---|---|---|---|
| 1 | OK | [transport.py:27–31](src/ham/geometry/transport.py#L27-L31) | Berwald connection $\Gamma^i_{jk} = \partial^2_v G^i$ correctly computed via double `jacfwd` |
| 2 | OK | [transport.py:43–46](src/ham/geometry/transport.py#L43-L46) | Transport ODE $\dot{X}^i = -\Gamma^i_{jk} v^j X^k$ correctly contracted |
| 3 | **WARNING** | [transport.py:48](src/ham/geometry/transport.py#L48) | Time step `dt = 1/N` should be `1/(N-1)` — systematic $(1-1/N)$ underestimation |
| 4 | **WARNING** | [transport.py:51](src/ham/geometry/transport.py#L51) | `to_tangent` projects at $\gamma_k$ instead of $\gamma_{k+1}$ — wrong tangent space for submanifolds |
| 5 | NOTE | [transport.py:48–49](src/ham/geometry/transport.py#L48-L49) | Forward Euler integrator is only 1st-order; norm drift in Riemannian subcase |
| 6 | NOTE | (cross-file) | Spray regularization from `metric.py` propagates into connection coefficients |

---

## Open Questions

1. **Time-step convention:** Is the path assumed to be parameterized on $[0, 1]$ or on $[0, T]$ for some other $T$? If the path comes from a geodesic solver with its own parameterization, the `dt` here must be consistent with that solver's time discretization. Currently there is no contract enforcing this.

2. **Submanifold transport accuracy:** For the sphere and hyperboloid manifolds, what is the actual accumulated error of the current integrator (Euler + wrong-point projection) for typical path lengths used in the VAE pipeline? The holonomy test uses `atol=1e-1`, which is very loose.

3. **Batched transport:** The current API processes a single vector along a single path. If the VAE pipeline needs to transport Jacobian frames (multiple vectors), the double `jacfwd` computation is duplicated for each vector. A batched API computing $\Gamma$ once and transporting multiple vectors would be more efficient.
