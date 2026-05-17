# Math Review: Randers Inverse Problem — Lagrangian vs. Eulerian Formulations
**Reviewer:** Math Reviewer Agent  
**Date:** May 17, 2026  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

The seven mathematical points under review concern the theoretical equivalence and practical subtleties of solving the Randers metric inverse problem via HAMTools's Lagrangian (geodesic ODE) approach versus Gahtan et al.'s Eulerian (eikonal PDE) approach. The core duality (eikonal ↔ geodesic) is **mathematically correct** in the smooth setting. However, several formulas in the existing codebase and in the proposed `ArrivalTimeLoss` pipeline contain **discretization-level issues** that could introduce systematic bias. The `EulerLagrangeResidualLoss` has a **critical formula error** in its local Lagrangian that does not match the Zermelo–Randers energy from `spec/MATH_SPEC.md` § 5. Two additional warnings concern caustics in the geodesic fan strategy and the non-equivalence of feasibility enforcement mechanisms under gradient flow.

**Overall verdict: Minor Issues** (1 CRITICAL in existing code, 2 WARNINGs in proposed pipeline, 4 NOTEs).

---

## Formula-by-Formula Audit

### 1. Eikonal ↔ Geodesic Duality

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (geodesic spray), § 5 (Zermelo parameterization)
- **Literature Reference:** Bao, Chern & Shen, *An Introduction to Riemann–Finsler Geometry*, Springer GTM 200, Theorem 6.2.1; Shen, *Lectures on Finsler Geometry*, World Scientific, Ch. 4.
- **Statement under review:** The eikonal characteristics are the geodesics of the Randers metric. Arrival time $T(x) = \inf_\gamma \int F(\gamma, \dot\gamma)\,dt = d_F(\text{source}, x)$.
- **Verdict:** **CORRECT** (with NOTE)
- **Analysis:**

  In the smooth Finsler setting, the eikonal equation
  $$F^*(x, \nabla T) = 1$$
  (where $F^*$ is the dual/co-Finsler norm) is the static Hamilton–Jacobi equation for the Hamiltonian $H(x,p) = \frac{1}{2}(F^*(x,p))^2$. Its characteristics satisfy the Euler–Lagrange equations of $\int F(\gamma,\dot\gamma)\,dt$, which are exactly the geodesic spray equations $\ddot{x}^i + 2G^i(x,\dot{x}) = 0$ from `spec/MATH_SPEC.md` § 2.1. This duality is exact when:
  
  1. $F$ is smooth and strongly convex (ensured by the $\lambda > 0$ constraint, i.e., $\|W\|_h < 1$).
  2. The geodesics are minimizing (no conjugate points between source and $x$).
  3. There is a unique minimizing geodesic from source to $x$ (no cut locus).

  For the **Randers eikonal** specifically, Gahtan writes:
  $$(\nabla T - b)^\top G^{-1}(\nabla T - b) = 1$$
  This is the Randers co-metric applied to $\nabla T$, which is indeed equivalent to $F^*(x, \nabla T) = 1$ under the standard Randers Legendre transform (see Bao–Robles–Shen, *Zermelo navigation on Riemannian manifolds*, J. Diff. Geom., 2004, Prop. 3.1).

  **Discretization subtlety:** The duality is *exact* in the continuum. For discrete solvers:
  - Gahtan's fast sweeping discretizes the eikonal PDE on a grid with upwind finite differences — this is a monotone scheme and converges to the viscosity solution (the correct weak solution past the cut locus).
  - HAMTools's AVBD/IVP discretizes the geodesic ODE — this finds a *single* geodesic path and is oblivious to the cut locus. Beyond the cut locus, the geodesic found by AVBD may not be the global minimizer. The arrival time computed via arc length of a non-minimizing geodesic would be **too large**.

- **Notes:**
  - The eikonal viscosity solution automatically selects the global minimum over all geodesics (it encodes the cut locus). The Lagrangian approach does not, unless explicitly minimized over multiple geodesics or initial conditions.
  - For the inverse problem, if the observation domain is within the injectivity radius of the source, both approaches give identical results. For distant observations past the cut locus, the Lagrangian approach may overestimate $T_i^{\text{pred}}$, biasing the learned metric.

---

### 2. Inverse Problem Formulation Equivalence

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1, § 5
- **Literature Reference:** Gahtan et al. (arXiv:2603.00035), Section 3
- **Implementations:**
  - Gahtan: $\min_\theta \sum_i (T_i(\theta) - \hat{T}_i)^2 + \lambda R(\theta)$, with $T_i$ from eikonal PDE
  - HAMTools: $\min_\theta \sum_i (d_F(\text{source}, x_i; \theta) - \hat{T}_i)^2 + \lambda R(\theta)$, with $d_F$ from AVBD/IVP
- **Verdict:** **CORRECT** (with WARNING)
- **Analysis:**

  The two formulations are **mathematically identical** in the smooth setting, since $T_i = d_F(\text{source}, x_i)$ by definition (Point 1). The optimization landscapes are the same at the continuous level.

  **Discretization differences introduce non-equivalent loss surfaces:**
  
  1. **Gradient bias:** Gahtan's implicit differentiation computes $\partial T / \partial \theta$ via the adjoint of the converged PDE. This gives the exact gradient of the *discrete* objective. HAMTools's autodiff through AVBD gives the exact gradient of the *discrete AVBD objective*, which is a different discretization. The two gradients converge to the same continuous gradient as discretization refines, but for fixed resolution they are different, and the convergence basins of the two optimizations may differ.
  
  2. **Amortization vs. sparse evaluation:** Gahtan computes $T$ at *all* grid points simultaneously ($O(n)$); the loss is evaluated at a subset. HAMTools solves individual BVPs; the loss is evaluated only at the solved points. If the observation set $\Omega_{\text{obs}}$ is dense, Gahtan is far more efficient. If sparse, HAMTools is competitive.
  
  3. **Non-minimizing geodesic risk:** As noted in Point 1, AVBD may converge to a local energy minimizer rather than the global one. For strongly heterogeneous metrics where the Randers indicatrix is highly elongated, multiple geodesics between source and target may exist, and AVBD's linear initialization may find the wrong one.

- **Notes:** The mathematical equivalence is exact. The discretization-level differences are a practical WARNING, not a theoretical concern.

---

### 3. ArrivalTimeLoss — Arc Length Computation

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 (positive homogeneity), § 1.2 (energy)
- **Implementation:** [metric.py](src/ham/geometry/metric.py#L163-L192)
- **Code (arc_length):**
  ```python
  def segment_length(x1, x2):
      v = x2 - x1
      midpoint = self.manifold.project(0.5 * (x1 + x2))
      return self.metric_fn(midpoint, v)
  return jnp.sum(jax.vmap(segment_length)(gamma[:-1], gamma[1:]))
  ```
- **Verdict:** **WARNING**
- **Analysis:**

  The discrete arc length formula is:
  $$L_{\text{disc}} = \sum_{k=0}^{N-2} F\!\left(\frac{x_k + x_{k+1}}{2},\; x_{k+1} - x_k\right)$$

  This is a valid first-order quadrature for $\int_0^1 F(\gamma(t), \dot\gamma(t))\,dt$ **only if the path is parameterized with $t \in [0,1]$ and $N$ equally-spaced time samples**. The key concern is:

  **The velocity vector $v_k = x_{k+1} - x_k$ is not normalized by $\Delta t = 1/(N-1)$.** By the positive 1-homogeneity of $F$, we have:
  $$F(x, v_k) = F\!\left(x, \frac{x_{k+1} - x_k}{\Delta t}\right) \cdot \Delta t$$
  
  So the current formula computes:
  $$\sum_k F(x_{\text{mid},k}, v_k) = \sum_k F(x_{\text{mid},k}, \dot\gamma_k) \cdot \Delta t$$
  
  which, by homogeneity, is indeed $\sum_k F(x_{\text{mid},k}, \dot\gamma_k) \cdot \Delta t$ — the correct midpoint-rule quadrature. **The homogeneity of $F$ makes the $\Delta t$ factor implicit.** The formula is therefore correct.

  However, there is a subtlety: **the midpoint projection** `self.manifold.project(0.5 * (x1 + x2))` uses the ambient midpoint, not the geodesic midpoint. On curved manifolds this introduces $O(h^2)$ error in each segment where $h = \|x_{k+1} - x_k\|$. For a path with $N$ segments, the total error is $O(h)$, making this a first-order scheme. This is documented in the docstring and is acceptable for the intended use.

  **Should we use energy $E$ and then take $\sqrt{2ET}$?** No. For a geodesic parameterized with constant speed (which AVBD approximately produces), $F(\gamma, \dot\gamma) = c$ (constant), so $L = c \cdot T_{\text{param}}$ and $\mathcal{E} = \frac{1}{2}c^2 \cdot T_{\text{param}}$. Thus $L = \sqrt{2\mathcal{E} \cdot T_{\text{param}}}$. The direct $F$-summation is correct and avoids the square root. However, for paths that are *not* constant-speed (e.g., a partially-converged AVBD solution), the energy integral $\int E\,dt$ and the length integral $\int F\,dt$ differ by the Cauchy–Schwarz gap:
  $$\left(\int F\,dt\right)^2 \leq T_{\text{param}} \int F^2\,dt = 2T_{\text{param}} \int E\,dt$$
  with equality iff $F$ is constant along the path. Using $F$ directly (arc length) is the correct choice for arrival time prediction, since arrival time = arc length (not energy) for $F$-length-parameterized geodesics.

  **Recommended Action:** The current implementation is correct for arrival time computation. Document the implicit $\Delta t$ cancellation via homogeneity to avoid future confusion.

---

### 4. Geodesic Shooting Fan (Strategy C) — Caustics and Differentiability

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (spray ODE), § 3 (Berwald connection)
- **Literature Reference:** Bao–Chern–Shen, *Introduction to Riemann–Finsler Geometry*, Ch. 9 (Jacobi fields); Shen, *Differential Geometry of Spray and Finsler Spaces*, Ch. 5 (conjugate points)
- **Implementation:** [geodesic.py](src/ham/solvers/geodesic.py) (`ExponentialMap`)
- **Verdict:** **WARNING**
- **Analysis:**

  **When do caustics form in Randers geometry?**

  Caustics (conjugate points) form when the exponential map $\exp_p: T_pM \to M$ fails to be a local diffeomorphism. This occurs when a Jacobi field $J(t)$ along a geodesic $\gamma(t)$ vanishes: $J(t_c) = 0$ for $t_c > 0$. In Randers geometry, the Jacobi equation is:
  $$\frac{D^2 J^i}{dt^2} + R^i{}_j(\dot\gamma) J^j = 0$$
  where $D/dt$ is the Berwald covariant derivative along $\gamma$ and $R^i{}_j$ is the Jacobi endomorphism (the Riemann curvature contracted with the velocity).

  HAMTools has the full machinery for this diagnostic:
  - `curvature.py` computes $R^i{}_{jk}$ via `riemann_curvature_tensor()` ([curvature.py](src/ham/geometry/curvature.py#L147))
  - `transport.py` provides Berwald parallel transport
  - The Jacobi ODE could be integrated alongside the geodesic ODE via `ExponentialMap`

  However, **HAMTools currently does not implement Jacobi field integration or conjugate point detection**. The `ExponentialMap.shoot()` method has no mechanism to flag when the geodesic passes through a conjugate point.

  **For the Randers metric specifically,** caustics depend on the flag curvature $K(x, v, w)$ computed in [curvature.py](src/ham/geometry/curvature.py#L130-L160). By the Cartan–Hadamard theorem generalized to Finsler spaces (Auslander, 1955; Bao–Chern–Shen, Theorem 9.4.1), if $K \leq 0$ everywhere, no conjugate points exist and the geodesic fan is globally injective. For mixed-sign curvature (typical in heterogeneous media), caustics will generically form at distances $\sim 1/\sqrt{K_{\max}}$.

  **Is the "nearest ray" assignment differentiable?**

  The assignment $x_i \mapsto \arg\min_j \|x_i - \gamma_j(t)\|$ involves a discrete $\arg\min$, which is piecewise-constant and hence has zero gradient almost everywhere. This makes the naive nearest-ray strategy **non-differentiable** with respect to the initial shooting directions. Remedies include:
  
  1. **Soft assignment:** Replace $\arg\min$ with a softmin: $T_i^{\text{pred}} = \sum_j w_j T_j$ where $w_j \propto \exp(-\|x_i - \gamma_j\|^2 / \sigma^2)$.
  2. **Optimal transport:** Use a Sinkhorn-regularized assignment between observation points and ray points.
  3. **Implicit differentiation of the assignment:** Treat the assignment as a fixed point of the Voronoi partition and differentiate through it.

  **Recommended Action:** If Strategy C is pursued, implement: (a) Jacobi field integration in `ExponentialMap` to detect conjugate points; (b) soft assignment with temperature annealing for differentiability. Flag any geodesic fan result where conjugate points are detected as unreliable.

---

### 5. Feasibility Constraint Equivalence

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo parameterization)
- **Implementation (HAMTools):** [randers.py](src/ham/geometry/zoo/randers.py#L50-L62)
  ```python
  w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))
  w_norm = jnp.sqrt(jnp.maximum(w_norm_sq, GRAD_EPS))
  max_speed = 1.0 - self.epsilon
  scale = jnp.where(w_norm < 0.5, 1.0, (max_speed * jnp.tanh(w_norm)) / (w_norm + GRAD_EPS))
  W_safe = W_raw * scale
  ```
- **Gahtan (from paper, Section 3.2):** Eigenvalue clamping $\lambda_i \leftarrow \max(\lambda_i, \epsilon)$ on $G$ to ensure positive-definiteness, followed by norm rescaling $b \leftarrow b \cdot \min(1, (1-\epsilon)/\|b\|_{G^{-1}})$ to enforce $\|b\|_{G^{-1}} < 1$.
- **Verdict:** **NOTE**
- **Analysis:**

  Both mechanisms enforce the same constraint: the Randers metric $F = \sqrt{v^\top G v} + b \cdot v$ is positive-definite iff $\|b\|_{G^{-1}} < 1$ (equivalently, $\|W\|_h < 1$ in the Zermelo parameterization). However, the two approaches are **not equivalent under gradient flow**:

  1. **HAMTools ($\tanh$ gating):** The constraint is enforced by a smooth reparameterization. The gradient $\partial \mathcal{L}/\partial W_{\text{raw}}$ flows through $\tanh$, which provides a natural saturation: as $\|W\|_h$ approaches 1, the gradient is damped by $\text{sech}^2(\|W\|_h)$. This makes the constraint boundary a soft attractor, and the optimization can approach it but never cross it. The downside is vanishing gradients near the boundary.

  2. **Gahtan (projection):** The constraint is enforced by a hard projection after each gradient step. The gradient $\partial \mathcal{L}/\partial b$ is unconstrained, and the projection clips to the feasible set. This allows full gradient magnitude even near the boundary, but introduces a discontinuity in the gradient at the constraint surface (the projection is not smooth).

  **Reparameterization equivalence:** The two are equivalent *as constraint sets* (both produce the interior of the unit ball in the $G^{-1}$-norm) but not as *optimization geometries*. The $\tanh$ approach implicitly defines a different Riemannian metric on the parameter space (the Fisher information of the $\tanh$ transform), which could accelerate or decelerate convergence depending on where the true parameters lie relative to the constraint boundary.

  **Additional subtlety in HAMTools:** The `jnp.where(w_norm < 0.5, 1.0, ...)` branch means that for $\|W\|_h < 0.5$, no gating is applied at all. This creates a non-smooth transition at $\|W\|_h = 0.5$, though the scale function is continuous (both branches yield 1.0 at the transition). The derivative, however, has a jump discontinuity at this point, which may cause optimizer instability in rare cases.

  **Recommended Action:** No mathematical issue — both are valid feasibility mechanisms. The note is purely about gradient flow differences that could affect convergence speed and stability.

---

### 6. Multi-Source Extension

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1
- **Literature Reference:** Sethian, *Level Set Methods and Fast Marching Methods*, Cambridge, 1999, Ch. 8 (multiple sources)
- **Verdict:** **CORRECT** (with NOTE)
- **Analysis:**

  With multiple sources $\{s_1, \dots, s_S\}$, the arrival time is:
  $$T(x) = \min_{s \in \{s_1, \dots, s_S\}} d_F(s, x)$$

  In the Eulerian approach, this is trivially handled: initialize the eikonal solver with $T(s_k) = t_k$ for each source $s_k$ (with ignition time $t_k$) and sweep once. The fast sweeping naturally computes the min over all sources.

  In the Lagrangian approach, the naive implementation requires solving $S \times K$ BVPs (from each source to each observation). The $\min$ over sources is then:
  $$T_i^{\text{pred}} = \min_s d_F(s, x_i)$$

  **More efficient formulations:**

  1. **Reverse BVP:** Instead of shooting from each source to each observation, solve $K$ BVPs from each observation $x_i$ back to the nearest source. This requires the *reverse* Randers metric $\tilde{F}(x, v) = F(x, -v)$, which for a Randers metric $F = \alpha + \beta$ gives $\tilde{F} = \alpha - \beta$. AVBD can solve this if the metric is modified, reducing the problem to $K$ solves (source-independent). However, the reverse Randers metric has $\tilde{W} = -W$, and the arrival time in the original metric is not the same as the arrival time in the reverse metric (Finsler asymmetry).

  2. **Geodesic fan from each source (Strategy C variant):** Shoot $N_\theta$ geodesics from each source, then assign each observation to the nearest ray over all sources. This is $S \times N_\theta$ IVP solves — efficient if $N_\theta$ is moderate and $S$ is small.

  3. **Voronoi pre-assignment:** First, assign each observation to its nearest source (using Euclidean distance as a proxy), then solve BVPs only from the assigned source. This is $K$ BVP solves, but the assignment may be incorrect when the metric is far from isotropic.

  The $\min$ operation introduces non-differentiability at points equidistant from two sources (the "wavefront collision" locus). The smooth relaxation $T^{\text{pred}}_i = -\tau \log \sum_s \exp(-d_F(s, x_i)/\tau)$ (log-sum-exp softmin) provides a differentiable approximation with temperature $\tau$.

  **Recommended Action:** For practical implementation, Strategy C (geodesic fans from all sources) with softmin assignment is the most natural Lagrangian analog of the multi-source eikonal. The $O(S \times N_\theta \times T_{\text{steps}})$ cost is acceptable for moderate $S$.

---

### 7. Euler–Lagrange Residual as Alternative Supervision

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 (energy), § 2.1 (Euler–Lagrange), § 5 (Zermelo Randers)
- **Implementation:** [losses.py](src/ham/training/losses.py#L248-L338) (`EulerLagrangeResidualLoss`)
- **Code under review:**
  ```python
  def L_smooth(z_pt, v_pt):
      H_pt, W_pt, _ = model.metric._get_zermelo_data(z_pt)
      v_norm_sq = jnp.dot(v_pt, jnp.dot(H_pt, v_pt))
      v_norm_eps = jnp.sqrt(jnp.maximum(v_norm_sq, 0.0) + self.epsilon**2)
      W_dot_v = jnp.dot(W_pt, jnp.dot(H_pt, v_pt))
      F = v_norm_eps - W_dot_v
      return 0.5 * (F**2)
  ```
- **Verdict:** **CRITICAL**
- **Analysis:**

  The local Lagrangian $L_{\text{smooth}}$ used in `EulerLagrangeResidualLoss` does **not match** the Zermelo–Randers metric from `spec/MATH_SPEC.md` § 5 or the `Randers.metric_fn()` implementation in [randers.py](src/ham/geometry/zoo/randers.py#L73-L90).

  **Spec formula (§ 5):**
  $$F(x, v) = \frac{\sqrt{\lambda \|v\|_h^2 + \langle W, v\rangle_h^2} - \langle W, v\rangle_h}{\lambda}$$
  where $\lambda = 1 - \|W\|_h^2$ and $\langle W, v\rangle_h = W^\top H v$.

  **`Randers.metric_fn()` implementation:**
  ```python
  discriminant = lam * v_sq_h + W_dot_v**2
  cost = (jnp.sqrt(jnp.maximum(discriminant, GRAD_EPS)) - W_dot_v) / lam
  ```
  This correctly implements the spec formula.

  **`EulerLagrangeResidualLoss.L_smooth()` implementation:**
  ```python
  F = v_norm_eps - W_dot_v
  ```
  This computes $F_{\text{wrong}} = \sqrt{\|v\|_h^2 + \epsilon^2} - \langle W, v\rangle_h$, which is:
  - **Missing the $\lambda$ factor** ($1 - \|W\|_h^2$) in both the discriminant and the denominator.
  - **Missing the $\langle W, v\rangle_h^2$ term** inside the square root.
  - **Using a simplified Randers-like formula** $F \approx \alpha - \beta$ rather than the full Zermelo formula.

  The simplified formula $F_{\text{wrong}} = \|v\|_h - \langle W, v\rangle_h$ (setting $\epsilon = 0$) is actually the standard Randers metric $F = \alpha + \beta$ in the convention where $\beta_i = -H_{ij}W^j$ (the 1-form $\beta = -W^\flat$). This is **not** the same as the Zermelo-navigated Randers metric unless $\|W\|_h = 0$ (in which case $\lambda = 1$ and both formulas agree). The Zermelo navigation produces a specific *non-linear* relation between $(h, W)$ and the Randers data $(\alpha, \beta)$:

  $$\alpha_{ij} = \frac{\sqrt{\lambda}\,(h_{ij}\lambda + W_i^\flat W_j^\flat)}{\lambda^2}, \quad \beta_i = -\frac{W_i^\flat}{\lambda}$$

  (Bao–Robles–Shen, 2004, Prop. 2.1). The E-L residual computed from $L_{\text{wrong}} = \frac{1}{2}(\|v\|_h - \langle W, v\rangle_h)^2$ yields the geodesic equations of the *wrong* Randers metric.

  **Consequence:** The Euler–Lagrange residual loss penalizes deviations from the geodesics of a simplified Randers metric, not the Zermelo-parameterized one that `Randers.metric_fn()` and the spray/AVBD solvers use. Paths that are geodesics of the true metric will have nonzero residual under this loss, and vice versa. The discrepancy is $O(\|W\|_h^2)$ — small for weak winds, but potentially large for strong drift ($\|W\|_h \to 1$).

  **Recommended Action:** Replace the local Lagrangian in `EulerLagrangeResidualLoss` with $L(z, v) = \frac{1}{2} F_{\text{Zermelo}}^2(z, v)$, using the same formula as `Randers.metric_fn()`:
  ```python
  def L_smooth(z_pt, v_pt):
      H_pt, W_pt, lam = model.metric._get_zermelo_data(z_pt)
      v_sq_h = jnp.dot(v_pt, jnp.dot(H_pt, v_pt))
      W_dot_v = jnp.dot(W_pt, jnp.dot(H_pt, v_pt))
      discriminant = lam * v_sq_h + W_dot_v**2
      F = (jnp.sqrt(jnp.maximum(discriminant, 0.0) + self.epsilon**2) - W_dot_v) / lam
      return 0.5 * F**2
  ```
  Or, more simply, call `model.metric.energy(z_pt, v_pt)` directly (which is guaranteed to match the metric) and add epsilon regularization externally.

  **Can the E-L residual be adapted for the arrival-time inverse problem?** Yes, in principle. Given candidate paths $\gamma_i$ from source to each observation $x_i$, the residual $R[\gamma_i] = \frac{d}{dt}(\partial_v L) - \partial_x L$ measures how far $\gamma_i$ is from a geodesic. If the metric is correct, the true arrival-time paths should have $R \approx 0$. This provides a *necessary* condition for metric correctness but not a *sufficient* one — the residual being zero means the path is a geodesic, but it does not directly constrain the arrival time to match the observation. The E-L residual is therefore best used as a **regularizer** alongside an arrival-time loss, not as a replacement. A path that is straight (in coordinates) but is not a geodesic will have $R \neq 0$, and this residual provides a gradient signal to adjust the metric so that the straight path becomes more geodesic. The convergence of this approach depends on the initial path quality; for strongly heterogeneous media, straight-line candidates may be poor approximations, and the E-L residual gradient may not improve the metric in the right direction.

---

## Open Questions

1. **Cut locus handling (Point 1–2):** For the wildfire inverse problem, how frequently do observations lie beyond the cut locus of the source? In 2D with smooth Randers metrics, the cut locus is generically a tree-like 1D set. If observations are dense, some will inevitably lie near the cut locus, where the Lagrangian approach may select a suboptimal geodesic. Empirical testing on Gahtan's synthetic benchmarks would quantify this risk.

2. **AVBD local minima (Point 2):** The AVBD solver uses a linear initialization and gradient descent. For metrics with multiple geodesics between two points (e.g., around an obstacle), AVBD may converge to a local energy minimizer. Does the solver's convergence behavior on strongly anisotropic Randers metrics need characterization before the inverse problem is attempted?

3. **Reparameterization invariance of E-L residual (Point 7):** The Euler–Lagrange residual $R$ depends on the parameterization of the candidate path. For the energy Lagrangian $L = \frac{1}{2}F^2$, the E-L equations are satisfied by *affinely parameterized* geodesics (constant $F$ along the path). If the candidate path is parameterized non-affinely, $R \neq 0$ even if the path is geometrically a geodesic. The current implementation uses `jnp.gradient` for finite differences, which assumes uniform parameterization. This is consistent with the straight-line candidate (uniform spacing), but for adaptively refined paths it may introduce spurious residuals.

4. **Epsilon placement in arc length (Point 3):** The `arc_length` function uses `metric_fn` directly without epsilon regularization. For paths passing through regions where $v \approx 0$ (e.g., near endpoints with deceleration), the Finsler metric $F(x, 0)$ may be singular. The epsilon regularization from `spec/MATH_SPEC.md` § 6.1 is applied at the metric level in training but not in `arc_length`. Should it be?

5. **Jacobi field implementation priority (Point 4):** Implementing Jacobi field integration is mathematically straightforward (it is a linear ODE coupled to the geodesic ODE), but the Berwald connection coefficients $^B\Gamma^i_{jk}$ involve 3rd derivatives of the energy — computationally expensive and numerically delicate. Is the Jacobi diagnostic worth the implementation cost for the initial inverse problem experiments, or should it be deferred?

6. **Softmin temperature schedule (Point 6):** The softmin approximation $T^{\text{pred}} = -\tau \log \sum_s \exp(-d_F(s, x_i)/\tau)$ has a well-known bias: $T_{\text{softmin}} \geq T_{\min} - \tau \log S$, where $S$ is the number of sources. For a temperature schedule $\tau \to 0$, the bias vanishes but the gradients become sharp (approaching the non-differentiable $\min$). What temperature schedule balances bias and gradient stability for the wildfire multi-source setting?
