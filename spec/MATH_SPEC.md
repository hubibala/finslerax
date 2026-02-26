# MATH_SPEC.md: Mathematical Foundations of the HAMTools Library (Berwald Edition)

**Version:** 1.1.0 (Berwald Revision)
**Date:** December 2, 2025
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
G^i(x, v) = \frac{1}{4} g^{il}(x, v) \left( 2 \frac{\partial^2 E}{\partial v^l \partial x^k} v^k - \frac{\partial E}{\partial x^l} \right)
$$

### 2.2. JAX Implementation (Implicit Solve)
We avoid inverting $g_{ij}$ explicitly. Instead, we compute $G^i$ by solving the linear system:
$$
\text{Hess}_v(E) \cdot (2G) = \nabla_x E - \text{Jac}_x(\nabla_v E) \cdot v
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

---

## 6. Numerical Stability

### 6.1. Epsilon Regularization
The Berwald connection involves 3rd derivatives of the Energy (2nd derivatives of Spray). This is highly sensitive to the singularity at $v=0$.
$$
F_\epsilon(x, v) = \sqrt{F^2(x, v) + \epsilon^2}
$$
We perform all derivations on $F_\epsilon$ during training.

### 6.2. Homogeneity Enforcement
Neural Networks approximating $F(x, v)$ may violate positive homogeneity ($F(\lambda v) \neq \lambda F(v)$).
**Fix:** We enforce homogeneity by construction:
$$
F_{net}(x, v) = \|v\| \cdot \text{NN}(x, v / \|v\|)
$$
This ensures the Berwald coefficients (which depend on homogeneity) remain well-defined.