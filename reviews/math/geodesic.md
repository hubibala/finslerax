# Math Review: `geodesic.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** [src/ham/solvers/geodesic.py](src/ham/solvers/geodesic.py)

## Summary

The geodesic IVP solver in `geodesic.py` is **mathematically correct in its core ODE formulation and RK4 scheme**. The phase-space ODE $(\dot{x}, \dot{v}) = (v, -2G)$ faithfully implements the geodesic equation from `spec/MATH_SPEC.md` § 2.1, and the RK4 update rule is standard. However, three issues compromise mathematical fidelity: (1) hardcoded velocity and acceleration clamping silently alters the ODE for high-speed geodesics, (2) manifold projection applied only after full RK4 steps — not at intermediate stages — degrades the method from 4th-order to at best 2nd-order on curved submanifolds, and (3) the `step_size` constructor parameter is dead code, never consumed by any integration method. No boundary-value or shooting method is present in this file; BVP solving is delegated to [src/ham/solvers/avbd.py](src/ham/solvers/avbd.py).

---

## Formula-by-Formula Audit

### 1. Geodesic ODE — Phase-Space Formulation

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1, equation $\ddot{x}^i + 2G^i(x, \dot{x}) = 0$
- **Literature Reference:** Bao–Chern–Shen, *Introduction to Riemann–Finsler Geometry* (2000), §5.3; standard first-order reduction of second-order geodesic ODE.
- **Implementation:** [geodesic.py:32–38](src/ham/solvers/geodesic.py#L32-L38)
  ```python
  dx = safe_v
  dv = metric.geod_acceleration(curr_x, safe_v)  # -2G
  ```
- **Verdict:** OK
- **Notes:** The geodesic equation $\ddot{x}^i + 2G^i = 0$ is reduced to the first-order system:

  $$\dot{x}^i = v^i, \qquad \dot{v}^i = -2G^i(x, v)$$

  The code sets `dx = v`, `dv = geod_acceleration(x, v)`, where `geod_acceleration` returns $-2G$ (verified in [math review of metric.py](../math/metric.md)). The phase-space vector $y = (x, v)$ is concatenated and evolved as a single system. Correct.

---

### 2. RK4 Integration Scheme

- **Spec Reference:** Not in `spec/MATH_SPEC.md` (numerical method choice is implementation detail).
- **Literature Reference:** Standard explicit Runge–Kutta 4th-order method; Hairer–Nørsett–Wanner, *Solving Ordinary Differential Equations I* (1993), §II.1.
- **Implementation:** [geodesic.py:41–48](src/ham/solvers/geodesic.py#L41-L48)
  ```python
  k1 = dynamics(y0)
  k2 = dynamics(y0 + 0.5 * dt * k1)
  k3 = dynamics(y0 + 0.5 * dt * k2)
  k4 = dynamics(y0 + dt * k3)
  y_next = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
  ```
- **Verdict:** OK
- **Notes:** This is the classical 4-stage, 4th-order Runge–Kutta formula:

  $$y_{n+1} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

  with $k_1 = f(y_n)$, $k_2 = f(y_n + \frac{\Delta t}{2}k_1)$, $k_3 = f(y_n + \frac{\Delta t}{2}k_2)$, $k_4 = f(y_n + \Delta t\, k_3)$. Correctly implemented.

---

### 3. Velocity Clamping — Modification of the ODE

- **Spec Reference:** `spec/MATH_SPEC.md` § 6 (numerical stability), but no specific clamping prescription.
- **Implementation:** [geodesic.py:31–33](src/ham/solvers/geodesic.py#L31-L33)
  ```python
  curr_v_norm = jnp.linalg.norm(curr_v) + 1e-12
  safe_v = jnp.where(curr_v_norm > 10.0, curr_v * (10.0 / curr_v_norm), curr_v)
  ```
- **Verdict:** WARNING
- **Notes:** This replaces $v$ with $\hat{v} = v \cdot \min(1,\; 10 / \|v\|)$ before evaluating both $\dot{x} = \hat{v}$ and $\dot{v} = -2G(x, \hat{v})$. Consequences:

  1. **ODE alteration:** For $\|v\| > 10$, the system being integrated is *not* the geodesic equation. Both the position update and the spray evaluation use the clamped velocity, so the trajectory diverges from the true geodesic.
  2. **Energy non-conservation:** Along a true geodesic, $E(x, v) = \frac{1}{2}F^2(x, v)$ is conserved. Clamping breaks this invariant.
  3. **Hardcoded threshold:** The value $10.0$ is arbitrary. For manifolds where natural velocities are $O(10^2)$ (e.g., large-curvature Randers metrics), this clamp activates spuriously. For manifolds where velocities are $O(10^{-1})$, it never activates and is wasted computation.

  **Recommended Action:** Make the velocity threshold a configurable parameter (or remove it, relying instead on adaptive step-size control or the regularization in `metric.spray()`).

---

### 4. Acceleration Clamping — Modification of the ODE

- **Spec Reference:** None.
- **Implementation:** [geodesic.py:37–38](src/ham/solvers/geodesic.py#L37-L38)
  ```python
  dv_norm = jnp.linalg.norm(dv) + 1e-12
  safe_dv = jnp.where(dv_norm > 100.0, dv * (100.0 / dv_norm), dv)
  ```
- **Verdict:** WARNING
- **Notes:** Clamps $\ddot{x}$ to $\|\ddot{x}\| \le 100$. Same concerns as velocity clamping: the integrated ODE is no longer the geodesic equation when the clamp is active. Particularly dangerous because near singularities (e.g., near $v = 0$, or at high-curvature points), the true acceleration can legitimately be large — clamping it will produce a trajectory that drifts off the geodesic.

  **Recommended Action:** Make the threshold configurable. Log or flag when clamping activates so the user knows the solution is approximate.

---

### 5. Manifold Projection After RK4 Steps — Order Reduction

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1 (submanifold formulations).
- **Literature Reference:** Hairer–Lubich–Wanner, *Geometric Numerical Integration* (2006), §IV.4 (projection methods for ODEs on manifolds).
- **Implementation:** [geodesic.py:50–52](src/ham/solvers/geodesic.py#L50-L52)
  ```python
  x_proj = metric.manifold.project(x_next)
  v_proj = metric.manifold.to_tangent(x_proj, v_next)
  ```
- **Verdict:** WARNING
- **Notes:** Projection is applied *after* the full RK4 step but *not* at the intermediate stages $k_2, k_3, k_4$. This has two consequences:

  1. **Off-manifold intermediate evaluations:** The RK4 midpoints $y_0 + \frac{\Delta t}{2} k_1$, etc., lie off the manifold. The spray $G^i(x, v)$ evaluated at these off-manifold points may be inaccurate or ill-defined (the metric $g_{ij}$ may not be positive-definite off-manifold).
  2. **Order reduction:** For an ODE constrained to a manifold $M$, the standard projected RK4 scheme (project only at the end of each step) reduces the global convergence order from $O(\Delta t^4)$ to $O(\Delta t^2)$ when the constraint curvature is non-zero. This is a well-known result (Hairer–Lubich–Wanner, Theorem IV.4.1).

  For `Euclidean` manifolds (where `project` is the identity), this is a non-issue. For `Sphere`, `Hyperboloid`, or `Torus`, the order loss is real.

  **Recommended Action:** For submanifold-constrained geodesics, consider projecting at each intermediate RK4 stage (Munthe-Kaas method), or using an intrinsic integrator that avoids ambient-space drift altogether.

---

### 6. Symplecticity — Energy Drift for Long Integrations

- **Spec Reference:** Not addressed in `spec/MATH_SPEC.md`.
- **Literature Reference:** Hairer–Lubich–Wanner, *Geometric Numerical Integration* (2006), §VI (backward error analysis of symplectic integrators).
- **Implementation:** Entire `_step_rk4` method.
- **Verdict:** NOTE
- **Notes:** Geodesic flow is a Hamiltonian system with Hamiltonian $H(x, p) = E(x, v)$ (where $p_i = \frac{\partial E}{\partial v^i}$ and $E$ is conserved along geodesics). RK4 is *not* symplectic, so $E(x_n, v_n)$ will drift over long integrations. The drift is $O(\Delta t^4)$ per step, accumulating to $O(T \cdot \Delta t^4)$ over time $T$ — bounded but growing linearly. A symplectic integrator (e.g., Störmer–Verlet / leapfrog) would bound the energy error uniformly in $T$.

  For the current use case (`max_steps = 200`, `step_size = 0.01`, so $T = 2$), the drift is likely negligible. For long-time transport or topological analysis, it could matter.

---

### 7. Exponential Map — `shoot()`

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (geodesic as exponential map).
- **Literature Reference:** Standard definition: $\exp_x(v) = \gamma(1)$ where $\gamma$ solves $\ddot{\gamma}^i + 2G^i = 0$ with $\gamma(0) = x$, $\dot{\gamma}(0) = v$.
- **Implementation:** [geodesic.py:55–63](src/ham/solvers/geodesic.py#L55-L63)
  ```python
  def shoot(self, metric, x0, v0, t_max=1.0):
      n_steps = self.max_steps
      dt = t_max / n_steps
      init_state = GeodesicState(x0, v0, 0.0)
      def body_fn(i, s):
          return self._step_rk4(metric, s, dt)
      final_state = jax.lax.fori_loop(0, n_steps, body_fn, init_state)
      return final_state.x
  ```
- **Verdict:** OK
- **Notes:** Integrates the geodesic ODE from $(x_0, v_0)$ over $[0, t_{\max}]$ with $N$ uniform steps and returns $\gamma(t_{\max})$. When $t_{\max} = 1$, this computes $\exp_{x_0}(v_0)$ as defined. The use of `fori_loop` is appropriate for forward-only computation where intermediate states are not needed. Mathematically correct.

---

### 8. Trajectory Tracing — `trace()`

- **Spec Reference:** Same as `shoot()`.
- **Implementation:** [geodesic.py:65–78](src/ham/solvers/geodesic.py#L65-L78)
  ```python
  def step_fn(s, _):
      s_new = self._step_rk4(metric, s, dt)
      return s_new, (s_new.x, s_new.v)
  init_state = GeodesicState(x0, v0, 0.0)
  _, (xs, vs) = jax.lax.scan(step_fn, init_state, None, length=n_steps)
  full_xs = jnp.concatenate([x0[None, :], xs], axis=0)
  full_vs = jnp.concatenate([v0[None, :], vs], axis=0)
  ```
- **Verdict:** OK
- **Notes:** Uses `jax.lax.scan` to collect the full trajectory $\{(x_k, v_k)\}_{k=0}^N$. The initial state $(x_0, v_0)$ is correctly prepended to the output of `scan` (which returns states $k = 1, \ldots, N$). The output shape is $(N+1, D)$ for both positions and velocities. Mathematically correct.

---

### 9. Dead Parameter — `step_size`

- **Spec Reference:** N/A.
- **Implementation:** [geodesic.py:20–22](src/ham/solvers/geodesic.py#L20-L22)
  ```python
  def __init__(self, step_size: float = 0.01, max_steps: int = 200):
      self.step_size = step_size
      self.max_steps = max_steps
  ```
  vs. [geodesic.py:57–58](src/ham/solvers/geodesic.py#L57-L58):
  ```python
  n_steps = self.max_steps
  dt = t_max / n_steps
  ```
- **Verdict:** WARNING
- **Notes:** The `step_size` attribute is set in `__init__` but never read by `shoot()` or `trace()`. Both methods compute `dt = t_max / self.max_steps` independently. This means:
  - Changing `step_size` has no effect on the integration.
  - The actual step size is $\Delta t = t_{\max} / N$, which for defaults gives $\Delta t = 1.0 / 200 = 0.005$, not the documented default of $0.01$.

  **Recommended Action:** Either remove the `step_size` parameter (breaking change) or use it: e.g., compute `n_steps = int(t_max / self.step_size)` and derive the number of steps from the desired step size.

---

## Upstream Dependency: `metric.spray()` and `geod_acceleration()`

The mathematical correctness of this solver is contingent on the correctness of `metric.geod_acceleration()`, which returns $-2G(x, v)$. This was verified in the [metric.py math review](../math/metric.md). Key points:

- `spray()` solves $\operatorname{Hess}_v(E) \cdot \text{acc} = \nabla_x E - \operatorname{Jac}_x(\nabla_v E) \cdot v$ and returns $G = -\frac{1}{2}\text{acc}$. **Correct.**
- `geod_acceleration()` returns $-2G$. **Correct.**
- A Tikhonov regularization of $\epsilon = 10^{-4}$ is applied to the Hessian (see [metric.py math review](../math/metric.md), Finding #4). This introduces a systematic bias in $G$ that propagates into the geodesic trajectory computed here.

---

## Cross-File Consistency

| Downstream consumer | How it uses `geodesic.py` | Consistency |
|---|---|---|
| [solvers/avbd.py](src/ham/solvers/avbd.py) | Does **not** use `geodesic.py`; solves BVP via direct energy minimization | Independent; no consistency issue |
| [bio/vae.py](src/ham/bio/vae.py) | Calls `metric.spray()` directly for the spray loss, not the geodesic solver | Consistent: both use the same `spray()` |
| [training/losses.py](src/ham/training/losses.py#L79) | `GeodesicSprayLoss` uses `metric.spray()` | Consistent |
| [geometry/transport.py](src/ham/geometry/transport.py) | Differentiates `metric.spray()` for Berwald coefficients | Consistent: spray is the shared source of truth |

---

## Summary of Findings

| # | Location | Severity | Finding |
|---|---|---|---|
| 1 | [geodesic.py:32–38](src/ham/solvers/geodesic.py#L32-L38) | OK | Phase-space ODE $(\dot{x}, \dot{v}) = (v, -2G)$ correctly implements geodesic equation |
| 2 | [geodesic.py:41–48](src/ham/solvers/geodesic.py#L41-L48) | OK | RK4 formula is standard and correctly implemented |
| 3 | [geodesic.py:31–33](src/ham/solvers/geodesic.py#L31-L33) | WARNING | Velocity clamp at $\|v\| = 10$ silently alters ODE for fast geodesics |
| 4 | [geodesic.py:37–38](src/ham/solvers/geodesic.py#L37-L38) | WARNING | Acceleration clamp at $\|\ddot{x}\| = 100$ silently alters ODE near high curvature |
| 5 | [geodesic.py:50–52](src/ham/solvers/geodesic.py#L50-L52) | WARNING | Post-step-only manifold projection reduces RK4 from $O(\Delta t^4)$ to $O(\Delta t^2)$ on curved submanifolds |
| 6 | [geodesic.py:25–48](src/ham/solvers/geodesic.py#L25-L48) | NOTE | RK4 is non-symplectic; energy drift $O(T \cdot \Delta t^4)$ over long integrations |
| 7 | [geodesic.py:55–63](src/ham/solvers/geodesic.py#L55-L63) | OK | `shoot()` correctly computes $\exp_x(v)$ via forward integration |
| 8 | [geodesic.py:65–78](src/ham/solvers/geodesic.py#L65-L78) | OK | `trace()` correctly collects full trajectory including initial conditions |
| 9 | [geodesic.py:20–22](src/ham/solvers/geodesic.py#L20-L22) | WARNING | `step_size` parameter is dead code; actual $\Delta t$ is $t_{\max}/N$, not `step_size` |

---

## Open Questions

1. **Clamping activation frequency:** How often do the velocity ($\|v\| > 10$) and acceleration ($\|\ddot{x}\| > 100$) clamps actually activate during typical training runs? If frequently, the learned geodesics are solutions to a modified (non-geodesic) ODE.

2. **Order verification on submanifolds:** Has the empirical convergence order been tested on `Sphere` or `Hyperboloid` manifolds? The predicted reduction from $O(\Delta t^4)$ to $O(\Delta t^2)$ due to end-of-step-only projection could be validated by a Richardson extrapolation test.

3. **Energy conservation monitoring:** Is $E(x_n, v_n)$ monitored during integration? Significant drift would indicate either (a) too-large step sizes, (b) clamping activation, or (c) accumulated manifold-projection error. This would be a useful diagnostic.

4. **Interaction with Hessian regularization:** The $\epsilon = 10^{-4}$ Tikhonov regularization in `metric.spray()` introduces a systematic $O(\epsilon)$ error in $G$ at each evaluation. Over $N = 200$ RK4 steps (each with 4 spray evaluations = 800 total), does this bias accumulate or average out? For directionally-stable geodesics the error should remain bounded, but near conjugate points it could amplify.
