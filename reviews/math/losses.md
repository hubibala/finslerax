# Math Review: losses

**Reviewer:** Math Reviewer Agent
**Date:** 2025-05-15
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Major Issues.** The file contains 16 loss components spanning VAE losses, geometric alignment losses, geodesic-based losses, and physically-informed E-L residual losses. Most standard losses (reconstruction, KL, regularisation) are mathematically straightforward and correct. However, one **CRITICAL** error exists: the `EulerLagrangeResidualLoss` re-implements a Lagrangian that does **not** match the actual Zermelo-Randers metric used by the solver (`Randers.metric_fn` in `zoo.py`). This means the E-L residual enforces geodesic equations for a different Finsler metric than the one the library actually uses. Several additional **WARNING**-level issues concern geometric consistency of inner products, misleading docstrings, and irreducible loss residuals.

---

## Formula-by-Formula Audit

### 1. `ReconstructionLoss` (line 20)
- **Spec Reference:** N/A (standard VAE reconstruction).
- **Implementation:** `jnp.mean((x - x_rec)**2)` — Mean squared error.
- **Verdict:** OK
- **Notes:** No Finsler math involved. Standard and correct.

---

### 2. `KLDivergenceLoss` (line 32)
- **Spec Reference:** N/A (standard variational inference).
- **Implementation:** Delegates to `dist.kl_divergence_std_normal()`.
- **Verdict:** OK
- **Notes:** Correctness depends on the distribution implementation, which is outside this file's scope.

---

### 3. `ZermeloAlignmentLoss` (line 42)
- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization).
- **Implementation:** `src/ham/training/losses.py:55–61`. Normalises $W$ and $u_{\text{lat}}$ using `_minkowski_norm`, then computes $-\langle \hat{W}, \hat{u} \rangle_L$ (Minkowski dot product).
- **Verdict:** WARNING
- **Notes:** The wind vector $W$ lives in the Randers metric structure where its natural norm is $\|W\|_H = \sqrt{W^T H W}$, not the Minkowski norm $\|W\|_L$. Using `_minkowski_norm` for normalisation and `_minkowski_dot` for alignment is only correct when the underlying manifold is a hyperboloid and $H$ coincides with the induced metric — but in general, this conflates two different inner product structures. The alignment is a reasonable heuristic but is not geometrically rigorous under a general learned Randers metric.

---

### 4. `GeodesicSprayLoss` (line 64)
- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (Spray Coefficients), § 2.2 (JAX Implementation).
- **Implementation:** `src/ham/training/losses.py:73–82`. Computes $\dot{z} = u_{\text{lat}} + W$, then $G = \text{spray}(z, \dot{z})$, then penalises $\|G\|^2_{g(z,\dot{z})} = g_{ij}(z,\dot{z}) G^i G^j$.
- **Verdict:** WARNING
- **Notes:** The docstring says *"Penalizes acceleration (spray vector norm) to encourage geodesic trajectories."* This is misleading. The spray $G^i(x, v)$ is a property of the geometry at the point $(x, v) \in TM$, not a property of a trajectory. Along a geodesic $\gamma$, we have $\ddot{\gamma}^i = -2G^i(\gamma, \dot\gamma)$ — the spray is generically non-zero. Setting $\|G\|^2 \to 0$ pushes the **geometry** toward flatness (Euclidean), not the **trajectory** toward a geodesic. A correct geodesic-deviation loss would penalise $\|\ddot{z} + 2G(z, \dot{z})\|^2$. The inner product computation via `model.metric.inner_product(z, \dot{z}, G, G)` correctly uses the fundamental tensor $g_{ij}(z, \dot{z})$, which is geometrically consistent.

---

### 5. `VelocityDirectionAlignmentLoss` (line 84)
- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization).
- **Implementation:** `src/ham/training/losses.py:95–112`. Computes cosine similarity using `safe_norm` (Euclidean $L^2$) and Euclidean dot product: $\cos\theta = \sum_i \hat{W}^i \hat{v}^i$.
- **Verdict:** WARNING
- **Notes:** Both $W$ and $v_{\text{lat}}$ live in a space equipped with a Riemannian metric $H$. The geometrically correct cosine similarity is:
  $$\cos_H(\theta) = \frac{\langle W, v \rangle_H}{\|W\|_H \cdot \|v\|_H}$$
  The code uses the flat Euclidean inner product instead, ignoring $H$. This is correct only when $H \approx I$, which is approximately true early in training (due to `MetricAnchorLoss`) but breaks as $H$ deviates from identity.

---

### 6. `ContrastiveAlignmentLoss` (line 116)
- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization).
- **Implementation:** `src/ham/training/losses.py:128–131`. Computes $-\langle W, \log_p(c) \rangle_L$ where $\log_p(c)$ is the manifold log map.
- **Verdict:** WARNING
- **Notes:** Uses `_minkowski_dot` unconditionally. If the manifold is not a hyperboloid (e.g., Euclidean latent space), this computes the wrong inner product. No guard or dispatch based on manifold type.

---

### 7. `MetricAnchorLoss` (line 134)
- **Spec Reference:** N/A (regularisation heuristic).
- **Implementation:** `src/ham/training/losses.py:143–152`. Computes $\|H - I\|_F^2 = \text{mean}((H_{ij} - \delta_{ij})^2)$.
- **Verdict:** OK
- **Notes:** Standard Frobenius-norm regularisation. Anchors $H$ near identity to prevent degenerate metrics.

---

### 8. `MetricSmoothnessLoss` (line 156)
- **Spec Reference:** N/A (regularisation heuristic).
- **Implementation:** `src/ham/training/losses.py:166–170`. Computes $\|\text{Jac}(W)\|_F^2 = \text{mean}((\partial W^i / \partial z^j)^2)$.
- **Verdict:** OK
- **Notes:** Standard Jacobian penalty for spatial smoothness of the wind field. Mathematically correct.

---

### 9. `_solve_and_integrate_impl` (line 171)
- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 (Energy Functional), § 2.1 (Euler-Lagrange).
- **Implementation:** `src/ham/training/losses.py:172–183`. Discretises $\mathcal{E}[\gamma] = \int_0^1 E(\gamma, \dot\gamma)\, dt$ as:
  $$\mathcal{E} \approx \sum_{k=0}^{N-2} E\!\left(x_k, \frac{x_{k+1} - x_k}{dt}\right) \cdot dt, \quad dt = \frac{1}{N-1}$$
- **Verdict:** OK
- **Notes:** Valid first-order (left-point) quadrature. By 2-homogeneity of $E$ in $v$, this is equivalent to $(N-1)\sum_k E(x_k, \Delta x_k)$. For a geodesic where $E$ is constant in time, the quadrature is exact. Uses left-point evaluation rather than midpoint; midpoint would give better accuracy for non-geodesic paths, but is acceptable for a training loss.

---

### 10. `LongTrajectoryAlignmentLoss` (line 196)
- **Spec Reference:** N/A (data-fitting loss).
- **Implementation:** `src/ham/training/losses.py:218–224`. Solves BVP between endpoints, then $L = \text{mean}((z_{\text{geo}} - z_{\text{obs}})^2)$.
- **Verdict:** OK
- **Notes:** Standard MSE alignment between BVP geodesic and observed trajectory in latent space. No Finsler math issues. Depends on correctness of the AVBD solver.

---

### 11. `EulerLagrangeResidualLoss` (line 227) — **PRIMARY FINDING**
- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (Euler-Lagrange Equations), § 5 (Zermelo/Randers).
- **Literature Reference:** Bao, Chern, Shen, "An Introduction to Riemann-Finsler Geometry" (Springer, 2000), Chapter 2; Shen, "Lectures on Finsler Geometry" (World Scientific, 2001), §2.3.
- **Implementation:** `src/ham/training/losses.py:261–309`.

#### 11a. Lagrangian formula mismatch — **CRITICAL**

The smoothed Lagrangian at line 275–283 is:
```python
v_norm_sq = jnp.dot(v_pt, jnp.dot(H_pt, v_pt))
v_norm_eps = jnp.sqrt(v_norm_sq + self.epsilon**2)
W_dot_v = jnp.dot(W_pt, jnp.dot(H_pt, v_pt))
F = v_norm_eps - W_dot_v
return 0.5 * (F**2)
```

This implements the "naïve" Randers form:
$$L_{\text{code}} = \frac{1}{2}\left(\sqrt{\|v\|_H^2 + \epsilon^2} - \langle W, v \rangle_H\right)^2$$

The **actual** Randers metric in `Randers.metric_fn` (`src/ham/geometry/zoo.py:116–131`) implements the correct Zermelo navigation formula from `spec/MATH_SPEC.md` § 5:
$$F(x, v) = \frac{\sqrt{\lambda\,\|v\|_H^2 + \langle W, v\rangle_H^2} \;-\; \langle W, v\rangle_H}{\lambda}, \quad \lambda = 1 - \|W\|_H^2$$

The two formulas differ in three ways:
1. The discriminant: $\|v\|_H^2$ vs. $\lambda\|v\|_H^2 + \langle W,v\rangle_H^2$.
2. The denominator: $1$ vs. $\lambda$.
3. The $\epsilon$-smoothing location.

These are different functions of $(z, v)$ for any non-zero $W$. Consequently, the Euler-Lagrange residual $R = \frac{d}{dt}\frac{\partial L}{\partial v} - \frac{\partial L}{\partial z}$ enforces the geodesic equations of a **different metric** than the one used by the solver, spray computation, and all other geometric operations.

**Recommended Action:** Replace the hand-coded Lagrangian with a call to `model.metric.energy(z_pt, v_pt)`, which correctly delegates to $E = \frac{1}{2}F^2$ using the actual `metric_fn`. Apply epsilon-smoothing in a manner consistent with `spec/MATH_SPEC.md` § 6.1 (i.e., $F_\epsilon(x,v) = \sqrt{F^2(x,v) + \epsilon^2}$ or directly regularise the argument of `metric_fn`).

#### 11b. E-L expansion correctness — OK

The chain-rule expansion of $\frac{d}{dt}\frac{\partial L}{\partial v^i}$ at line 289–296:
```python
_, hess_v_a = jax.jvp(lambda v_arg: grad_v_fn(z, v_arg), (v,), (a,))
_, mixed_term = jax.jvp(lambda z_arg: grad_v_fn(z_arg, v), (z,), (v,))
grad_z = jax.grad(L_smooth, argnums=0)(z, v)
residual = hess_v_a + mixed_term - grad_z
```

This correctly computes $R^i = \frac{\partial^2 L}{\partial v^i\partial v^j}a^j + \frac{\partial^2 L}{\partial z^k\partial v^i}v^k - \frac{\partial L}{\partial z^i}$, which is the exact expansion of the E-L residual vector. The use of `jax.jvp` for Hessian-vector products is efficient and mathematically correct.

#### 11c. Residual norm with stop-gradient — **STRONG**

At line 302–307:
```python
H_frozen = jax.lax.stop_gradient(H_frozen)
return jnp.dot(residual, jnp.dot(H_frozen, residual))
```

Using `stop_gradient` on $H$ prevents the metric from collapsing to reduce the apparent residual norm. This is a well-considered design choice.

#### 11d. Boundary artifacts from `jnp.gradient` — WARNING

`src/ham/training/losses.py:258–260`. `jnp.gradient` uses centered differences in the interior but one-sided (forward/backward) differences at the first and last points. This gives $O(h)$ accuracy at boundaries vs. $O(h^2)$ in the interior. The acceleration $a$ (second derivative) is especially noisy at boundary points, which can produce spurious E-L residual spikes. No boundary-point exclusion is applied before `jnp.mean`.

**Recommended Action:** Exclude the first and last 1–2 points from the residual mean, or use interior-only indexing: `el_violations[1:-1]`.

---

### 12. `AVBDPathEnergyLoss` (line 317)
- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 (Energy Functional).
- **Implementation:** Delegates to `_solve_and_integrate`, which computes $\int_0^1 E(\gamma, \dot\gamma)\,dt$.
- **Verdict:** OK
- **Notes:** Mathematically straightforward delegation. Correctness depends on the AVBD solver and the `_solve_and_integrate_impl` quadrature (reviewed above).

---

### 13. `WindThermodynamicLoss` (line 333)
- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo constraint $\|W\|_H < 1$).
- **Implementation:** `src/ham/training/losses.py:340–341`. Computes $\|W\|_H^2 = W^T H W$.
- **Verdict:** OK
- **Notes:** Correct Riemannian squared norm of the wind. Penalising this is thermodynamically motivated (wind "energy cost") and also helps enforce the Zermelo causality bound.

---

### 14. `KinematicPriorLoss` (line 347)
- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (Geodesic spray / distance).
- **Implementation:** `src/ham/training/losses.py:357–362`. Computes $\max(\|v\|_2 - m, 0)^2$ where $v = \log_{z_s}(z_e)$.
- **Verdict:** WARNING
- **Notes:** The log map $v = \log_{z_s}(z_e)$ gives the initial tangent vector whose Finsler norm $F(z_s, v)$ equals the geodesic distance. But the penalty uses the **Euclidean** norm $\|v\|_2$ via `safe_norm`, not the Finsler norm $F(z_s, v)$. In curved spaces where $F \neq \|\cdot\|_2$, the margin $m$ has no geometric meaning. For a geometrically consistent kinematic prior, use $F(z_s, v) - m$ instead.

---

### 15. `FinslerActionMatchingLoss` (line 369)
- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 (Energy Functional).
- **Implementation:** `src/ham/training/losses.py:383–390`. Computes $E(z_s, z_e - z_s)$ using `model.metric.energy`.
- **Verdict:** NOTE
- **Notes:** Approximates the tangent vector $v \approx z_e - z_s$ (flat-space log map). Geometrically exact only for flat manifolds; for curved spaces the correct vector is $v = \log_{z_s}(z_e)$. This is explicitly acknowledged in the code as a design choice to avoid ODE integration, and is acceptable for training when curvature is moderate.

---

### 16. `WindAssistedTrajectoryAlignmentLoss` (line 395)
- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (Geodesic ODE), § 2.2 (Spray implementation).
- **Implementation:** `src/ham/training/losses.py:414–428`. Shoots geodesic via `ExponentialMap.shoot(metric, z_start, v_init)` then compares decoded endpoint with observed $x_{\text{end}}$.
- **Verdict:** OK
- **Notes:** Mathematically sound. The exponential map $\exp_{z_s}(v)$ follows the geodesic ODE from $z_s$ with initial velocity $v$ for unit time. In flat space, $\exp_{z_s}(v) = z_s + v$, recovering $z_e$. The loss operates in data space (MSE on decoded outputs), which is a valid alignment objective.

---

### 17. `FinslerianFlowMatchingLoss` (line 435)
- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Navigation).
- **Implementation:** `src/ham/training/losses.py:462–487`. Computes alignment $a = \langle W, \hat{v} \rangle_H$ and loss $1 - a$.
- **Verdict:** WARNING
- **Notes:** The maximum achievable alignment is $a_{\max} = \|W\|_H$ (by Cauchy-Schwarz in the $H$-inner product). Since the Zermelo constraint enforces $\|W\|_H < 1$, the loss $1 - a \geq 1 - \|W\|_H > 0$ has an **irreducible positive residual** that can never reach zero. This creates a gradient signal pushing $\|W\|_H \to 1$ (toward the causality boundary), potentially conflicting with the `WindThermodynamicLoss` and the Zermelo safety margin. A normalised formulation $1 - \langle \hat{W}, \hat{v} \rangle_H$ (cosine similarity in the $H$-metric) would decouple direction alignment from wind magnitude.

---

## Spec Errata Noted During Review

While verifying the implementation against the spec, the following issue in `spec/MATH_SPEC.md` was observed:

### Spray coefficient formula (§ 2.1)
- **Spec states:** $G^i(x, v) = \frac{1}{4} g^{il} \left( 2 \frac{\partial^2 E}{\partial v^l \partial x^k} v^k - \frac{\partial E}{\partial x^l} \right)$
- **Correct formula** (Bao-Chern-Shen, eq. 2.3.5, using $F^2 = 2E$): $G^i = \frac{1}{4} g^{il}\left(2\frac{\partial^2 E}{\partial v^l \partial x^k}v^k - \mathbf{2}\frac{\partial E}{\partial x^l}\right)$
- The factor of 2 in front of $\frac{\partial E}{\partial x^l}$ is missing in the spec. The standard reference formula is $G^i = \frac{1}{4}g^{il}\left(\frac{\partial^2[F^2]}{\partial x^k \partial v^l}v^k - \frac{\partial[F^2]}{\partial x^l}\right)$, and since $F^2 = 2E$, both terms pick up a factor of 2.
- **Impact on code:** None. The code (`metric.py:52–61`) uses the implicit-solve formulation from § 2.2, which correctly derives $G$ from the E-L equations without using the explicit formula from § 2.1. The sign and factor algebra in `spray()` are verified correct.

---

## Open Questions

1. **EulerLagrangeResidualLoss Lagrangian:** Was the simplified Randers form $F_\epsilon = \sqrt{\|v\|_H^2 + \epsilon^2} - \langle W, v\rangle_H$ chosen intentionally as an approximation (e.g., for small-wind regimes where $\lambda \approx 1$), or is it an oversight? If intentional, the valid regime ($\|W\|_H \ll 1$) should be documented.

2. **Minkowski vs. Riemannian inner product:** Several losses (`ZermeloAlignmentLoss`, `ContrastiveAlignmentLoss`) use `_minkowski_dot`/`_minkowski_norm` for alignment computations. Is the library intended to run exclusively on hyperboloid manifolds, or should these losses dispatch on manifold type?

3. **GeodesicSprayLoss semantics:** Is the intent to regularise curvature (push geometry toward flatness), or to encourage observed trajectories to follow geodesics? The current implementation does the former; the latter requires a trajectory-dependent residual $\|\ddot{z} + 2G\|^2$.

4. **Boundary handling in EulerLagrangeResidualLoss:** Should the first/last trajectory points be excluded from the residual mean to avoid finite-difference boundary artifacts?
