# MATH_SPEC.md: Mathematical Foundations of the HAM Library (Berwald Edition)

**Version:** 1.2.0 (Berwald Revision + Eikonal Duality)
**Date:** June 2026
**Project:** Holonomic Association Model (HAM)

## Abstract

This document defines the mathematical specifications for `HAMTools`, a differentiable geometry library designed for Finslerian representation learning. We construct a strict hierarchy of geometric spaces—Euclidean, Riemannian, and Randers—unified under the energy-based formalism. We rigorously derive the *Geodesic Spray* coefficients using the Euler-Lagrange equations. Crucially, we adopt the **Berwald Connection** for parallel transport, as it is the unique connection directly induced by the geodesic spray, providing a natural framework for analyzing path stability and parallel vector fields without enforcing artificial metric compatibility.

---

## 1. General Finsler Geometry

### 1.1. Definition
A **Finsler Manifold** is a pair $(M, F)$, where $M$ is a differentiable manifold and $F: TM \to [0, \infty)$ is a continuous function on the tangent bundle, satisfying:

1.  **Regularity:** $F$ is $C^\infty$ on $TM \setminus \{0\}$.
2.  **Positive Homogeneity:** $F(x, \lambda v) = \lambda F(x, v)$ for $\lambda > 0$.
3.  **Strong Convexity:** The **Fundamental Tensor** $g_{ij}$ is positive definite:
    $$
    g_{ij}(x, v) := \frac{1}{2} \frac{\partial^2 F^2}{\partial v^i \partial v^j}(x, v)
    $$

### 1.2. The Energy Functional
We define the **Lagrangian** (Energy) as:
$$
E(x, v) = \frac{1}{2} F^2(x, v)
$$
This scalar functional is the root of our computational graph. All geometric objects are derivatives of $E$.

---

## 2. Dynamics: The Geodesic Spray

The "Physics Engine" of the manifold is determined by the **Geodesic Spray**. This vector field describes the inertial flow of particles.

### 2.1. Derivation
Minimizing the energy functional $\mathcal{E}[\gamma] = \int E(x, \dot{x}) dt$ yields the Euler-Lagrange equations:
$$
\frac{d}{dt} \left( \frac{\partial E}{\partial v^i} \right) - \frac{\partial E}{\partial x^i} = 0
$$

Expanding this yields the equation of motion:
$$
\ddot{x}^i + 2G^i(x, \dot{x}) = 0
$$

Where $G^i$ are the **Spray Coefficients**. They are given by:
$$
G^i(x, v) = \frac{1}{2} g^{il}(x, v) \left( \frac{\partial^2 E}{\partial x^k \partial v^l} v^k - \frac{\partial E}{\partial x^l} \right)
$$

### 2.2. JAX Implementation (Implicit Solve)
We avoid inverting $g_{ij}$ explicitly. Instead, we compute $G^i$ by solving the linear system:
$$
\text{Hess}_v(E) \cdot (-2G) = \nabla_x E - \text{Jac}_x(\nabla_v E) \cdot v
$$

---

## 3. Kinematics: The Berwald Connection

For parallel transport and covariant differentiation, we employ the **Berwald Connection**. Unlike the Chern connection, the Berwald connection is defined purely by the non-linearity of the Spray.

### 3.1. Definition
The Berwald connection coefficients $^B\Gamma^i_{jk}$ are defined as the second partial derivatives of the spray coefficients with respect to velocity:

$$
^B\Gamma^i_{jk}(x, v) = \frac{\partial^2 G^i}{\partial v^j \partial v^k}(x, v)
$$

**Key Properties:**
* **Torsion-Free:** Symmetric in $j, k$.
* **Spray-Induced:** If the Spray is quadratic in $v$ (Riemannian case), $\Gamma$ depends only on $x$ (it becomes the Levi-Civita connection).
* **Non-Metric:** In general Finsler spaces, the Berwald connection does *not* preserve the Finsler norm ($D g \neq 0$). This is a feature, not a bug: it tracks the "affine" deformation of the space rather than forcing rigid rotation.

### 3.2. Parallel Transport
A vector field $X(t)$ is **Berwald Parallel** along a curve $\gamma(t)$ (with velocity $\dot{\gamma}$) if:

$$
\frac{d X^i}{dt} + \ ^B\Gamma^i_{jk}(\gamma, \dot{\gamma}) \dot{\gamma}^j X^k = 0
$$

In JAX, this is simply an ODE integration where the `force` term uses the Hessian of the `spray` function.

### 3.3. Holonomy and Projection-Based Transport

**Remark (Ambient vs. Intrinsic Convention):**
When the metric is defined in ambient coordinates as $g(x) = I_n$ (the identity), the Berwald connection satisfies $\Gamma^i_{jk} = 0$ everywhere. In this case, the parallel transport reduces to a pure tangent-space projection at each discrete step:

$$X_{k+1} = \Pi_{T_{\gamma_{k+1}}\mathcal{M}} \left( X_k - \Gamma^i_{jk} \dot\gamma^j X^k \Delta t \right) = \Pi_{T_{\gamma_{k+1}}\mathcal{M}}(X_k)$$

This is a valid approximation of the Levi-Civita connection via the Gauss equation ($\nabla^M_X Y = \Pi_{TM}(\bar\nabla_X Y)$), but it produces a holonomy angle that is the complement of the standard solid-angle formula. Here $\theta$ is the **colatitude** (polar angle measured from the north pole), so the transport circle is at constant $\theta$ and the enclosed spherical cap subtends solid angle $\Omega = 2\pi(1-\cos\theta)$:

| Mechanism | Holonomy angle for colatitude $\theta$ on $S^2$ |
|---|---|
| Projection-based ($\Gamma = 0$, ambient coords) | $2\pi\cos\theta$ |
| Intrinsic Levi-Civita ($\Gamma \neq 0$, chart coords) | $2\pi(1 - \cos\theta)$ |

The intrinsic row is the textbook result: parallel transport around a circle of colatitude $\theta$ rotates a vector by the enclosed solid angle $2\pi(1-\cos\theta)$. (In terms of *latitude* $\varphi = \tfrac{\pi}{2}-\theta$ this reads $2\pi(1-\sin\varphi)$.) Both rows are equivalent modulo $2\pi$ as elements of $SO(2)$ — since $\cos(2\pi\cos\theta) = \cos\!\big(2\pi(1-\cos\theta)\big)$ — so they describe the same physical rotation. The implementation uses the projection-based approach when the metric is position-independent in ambient coordinates.

For metrics defined in intrinsic coordinates where $g(x)$ is position-dependent (e.g., the Poincaré half-plane $ds^2 = (dx^2+dy^2)/y^2$), the Berwald connection is non-trivially non-zero and the ODE integration genuinely drives the transport.

---

## 4. The Geometric Hierarchy

We verify that our implementation generalizes standard geometries.

| Geometry | Metric Function $F(x, v)$ | Spray $G^i$ | Berwald Connection $\Gamma^i_{jk}$ |
| :--- | :--- | :--- | :--- |
| **Euclidean** | $\sqrt{v^T v}$ | $0$ | $0$ |
| **Riemannian** | $\sqrt{v^T g(x) v}$ | Quadratic in $v$ | $\Gamma^i_{jk}(x)$ (Levi-Civita) |
| **Berwald** | General Finsler | Quadratic in $v$ | $\Gamma^i_{jk}(x)$ (Indep. of $v$) |
| **Randers** | $\sqrt{v^T M v} + \beta \cdot v$ | Non-quadratic | $\Gamma^i_{jk}(x, v)$ (Dep. on $v$) |
| **Hyperboloid** | Minkowskian $\sqrt{\langle v, v\rangle_L}$ | Quadratic in $v$ | Levi-Civita equivalent |

### 4.1. Surface Formulations and Instabilities
`HAMTools` provides strict analytical sub-manifolds (e.g., `Sphere`, `Hyperboloid`, `Torus`, `Paraboloid`). The Hyperboloid models the upper sheet in Minkowski space and features exact $\cosh$/$\sinh$ exponential and logarithmic maps.
*Critical Note:* Integrating these exact geometric maps (especially for the Sphere and Hyperboloid) with deep neural learning loops inside the VAE currently causes severe numerical instability resulting in solver collapse.

---

## 5. The Zermelo Parameterization (Randers)

To learn valid Randers metrics, we parameterize $F$ via Zermelo's Navigation Problem.

**Inputs:**
* Riemannian metric (Sea): $h_{ij}(x)$
* Wind field: $W^i(x)$ with constraint $\|W\|_h < 1$.

**Resulting Randers Metric:**
$$
F(x, v) = \frac{\sqrt{\lambda \|v\|_h^2 + \langle W, v \rangle_h^2} - \langle W, v \rangle_h}{\lambda}
$$
where $\lambda = 1 - \|W\|_h^2$.

**Note:** We use the minus sign convention for $\beta$ here to align with "Headwind increases cost."

### 5.1. Enforcing the Causal Bound $\|W\|_h < 1$

The strong-convexity (weak-wind) condition $\|W\|_h < 1$ keeps $\lambda > 0$ and the
fundamental tensor positive-definite. A learned $W(x)$ is unconstrained, so it must
be projected into the causal ball. We require the projection to (i) preserve
*physically-valid* winds without distortion, (ii) be $C^\infty$ so the spray and
Berwald connection remain smooth, and (iii) guarantee $\|W\|_h < 1-\varepsilon$
strictly. Writing $r = \|W\|_h$ and $c = 1-\varepsilon$, we scale $W \mapsto sW$ with
$s = \varphi(r)/r$, where $\varphi$ is the temperature-controlled smooth minimum

$$
\varphi(r) \;=\; r \;-\; \tfrac{1}{\kappa}\,\mathrm{softplus}\!\big(\kappa\,(r-c)\big),
\qquad
\varphi'(r) = 1 - \sigma\!\big(\kappa(r-c)\big) \in (0,1),
\qquad
\sup_r \varphi(r) = c .
$$

$\varphi$ is the identity to within $\sim e^{-\kappa(c-r)}/\kappa$ for $r < c$, so a
requested wind of, say, $r=0.5$ is returned as $0.5$ (to $\sim 10^{-6}$), and bending
is confined to a shell of width $\sim 1/\kappa$ around the causal boundary, where it
is unavoidable. The stiffness $\kappa$ defaults to `ham.utils.WIND_STIFFNESS`.

> **Historical note.** Earlier versions used $s = (1-\varepsilon)\tanh(r)/r$. Because
> $\tanh$ has slope $<1$ at the origin, that map bent *every* wind — e.g.
> $0.5 \mapsto 0.46$ — silently distorting valid currents. The smooth-min above
> removes this distortion while retaining $C^\infty$ regularity and the strict bound.

For **trusted, prescribed** fields (e.g. a known ocean current already satisfying
$\|W\|_h < 1$), `Randers(..., wind_mode="raw")` bypasses the clamp entirely and passes
$W$ through bit-exact, flooring only $\lambda$ as a NaN guard. The default
`wind_mode="soft"` applies $\varphi$ and is the correct choice for learned winds.

---

## 6. Numerical Stability

### 6.1. Epsilon Regularization
The spray and Berwald connection involve high-order derivatives of the energy that are sensitive to the singularity at $v = 0$ and to near-degenerate directions (e.g. the Randers boundary, where the fundamental tensor's smallest eigenvalue $\to 0$). Two complementary safeguards are used:

**Norm smoothing.** Where a raw norm would be non-differentiable at the origin, we evaluate a smoothed surrogate
$$
F_\epsilon(x, v) = \sqrt{F^2(x, v) + \epsilon^2},
$$
and the `Randers.metric_fn` additionally clamps the discriminant and substitutes a shifted velocity for $\|v\| \approx 0$ so that $F$ and $\nabla F$ stay finite.

**Trace-scaled Tikhonov on the spray solve.** Rather than inverting the bare velocity-Hessian $H = \mathrm{Hess}_v(E)$, the spray solves
$$
\big(H + \epsilon\,\tfrac{\mathrm{tr}\,H}{D}\,I\big)\,(-2G) = \nabla_x E - \mathrm{Jac}_x(\nabla_v E)\,v .
$$
Scaling the regularizer by the mean eigenvalue $\mathrm{tr}\,H / D$ keeps the *relative* perturbation constant across metrics of different magnitude — avoiding over-regularizing small metrics and under-regularizing large ones.

### 6.2. Homogeneity Enforcement
Neural Networks approximating $F(x, v)$ may violate positive homogeneity ($F(\lambda v) \neq \lambda F(v)$).
**Fix:** We enforce homogeneity by construction:
$$
F_{net}(x, v) = \|v\| \cdot \text{NN}(x, v / \|v\|)
$$
This ensures the Berwald coefficients (which depend on homogeneity) remain well-defined.

### 6.3. Secant Scaling for Logarithmic Maps
The projected secant $\Pi_{T_xM}(y - x)$ can have a shorter ambient length than the chord $y - x$ on highly curved manifolds, which can cause topological shortcuts. To correct this, we rescale the tangent projection by the chord length:
$$
\log_x(y) \approx \frac{\|y - x\|}{\|\Pi_{T_xM}(y - x)\|} \cdot \Pi_{T_xM}(y - x)
$$
This preserves the direction but scales the magnitude correctly to avoid optimizer exploitation of the manifold's interior.

---

## 7. The Eikonal Dual: Arrival Times

Sections 2–4 describe the **primal** picture: a path $\gamma$ and its cost $\int F(\gamma, \dot\gamma)\,dt$. For *one-source-to-everywhere* problems it is far cheaper to solve the **dual** problem directly — the field of minimal arrival times $T(x)$ from a source set — without ever enumerating paths. This is the Finsler **eikonal equation**, and HAM's eikonal solvers implement it.

### 7.1. Hamilton–Jacobi Form

The minimal-cost (Finsler distance) field $T(x)$ from a source obeys the static Hamilton–Jacobi equation that the **dual norm** of its gradient is unit:
$$
F^*\!\big(x,\, \nabla T(x)\big) = 1, \qquad T\big|_{\text{source}} = 0,
$$
where $F^*$ is the polar (Legendre) dual of the Finsler norm $F$. Intuitively, $T$ rises at unit rate per unit Finsler cost, so its gradient lies exactly on the dual indicatrix.

### 7.2. Randers Duality

For a Randers norm written in the affine form $F(x, v) = \sqrt{v^\top G(x)\, v} + B(x)^\top v$, the unit ball $\{F = 1\}$ is an ellipsoid shifted by the drift $B$. Its polar dual is again a shifted ellipsoid, so $F^*(\nabla T) = 1$ becomes a concrete anisotropic eikonal PDE:
$$
\big(\nabla T - B\big)^\top G^{-1} \big(\nabla T - B\big) = 1 .
$$
The pair $(G, B)$ is an algebraic function of the Zermelo navigation data $(H, W, \lambda)$ of § 5. In the 3-D solver the equivalent **dual** operators are formed directly,
$$
Q = \lambda\,\big(H^{-1} - W W^\top\big), \qquad B_{\text{dual}} = -\,\frac{H W}{\lambda},
$$
giving $(\nabla T - B_{\text{dual}})^\top Q\,(\nabla T - B_{\text{dual}}) = 1$ — the same PDE, since $Q = G^{-1}$ and $B_{\text{dual}} = B$ (one can verify $\sqrt{v^\top G v} + B^\top v$ reproduces the Zermelo formula of § 5 exactly). The drift term $B$ is exactly what makes the metric **asymmetric**: it shifts the dual indicatrix off-center, so arrival time grows faster against the wind than with it.

### 7.3. Numerical Scheme

The PDE is discretized with an upwind **Godunov Hamiltonian** and solved by the **Fast Sweeping Method**: alternating Gauss–Seidel sweeps in every diagonal direction ($2^d$ orderings in $d$ dimensions) propagate causal information along characteristics until $T$ reaches a fixed point. Because the drift $B$ and off-diagonal couplings of $G^{-1}$ break axis-alignment, the stencil must enumerate all signed upwind donor configurations per axis — omitting them produces causality violations for non-zero wind.

### 7.4. Differentiability

The solvers expose O(1)-memory gradients $\partial T / \partial (G, B)$ via `jax.custom_vjp`. Rather than differentiating through every sweep, the backward pass runs an **adjoint fixed-point iteration** at the converged solution (implicit-function theorem on the discrete steady-state operator). This lets an entire eikonal solve sit inside a training loss — e.g. matching observed arrival times to a learned metric (`ArrivalTimeLoss`) — at constant memory cost regardless of sweep count.