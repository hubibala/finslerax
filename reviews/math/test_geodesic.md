# Math Review: `test_geodesic.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** [tests/test_geodesic.py](tests/test_geodesic.py)

## Summary

The test file verifies four properties of the geodesic solver: Euclidean ballistic motion, great-circle trajectory on $S^2$, energy conservation under a Randers metric, and manifold adherence. The energy-conservation and manifold-adherence tests are well-chosen mathematical invariants. However, the suite has two significant gaps: (1) two of four tests invoke `Sphere(1.0)`, which passes `1.0` to the `intrinsic_dim` parameter instead of `radius`, creating a semantically incorrect manifold that only works by accident, and (2) no test exercises the geodesic spray on a manifold with non-trivial curvature—the Euclidean-metric tests have identically zero spray and derive all curvature from the solver's post-step projection, so the core ODE $\ddot{x}^i + 2G^i = 0$ is never verified against a known closed-form Riemannian geodesic. **Verdict: Minor Issues.**

---

## Formula-by-Formula Audit

### 1. `test_euclidean_ballistic` — Straight-line geodesic in flat space

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1; § 4 (Euclidean row: $G^i = 0$).
- **Implementation:** [tests/test_geodesic.py:29–38](tests/test_geodesic.py#L29-L38)
  ```python
  x_final = self.solver.shoot(metric, x0, v0)
  expected = x0 + v0
  np.testing.assert_allclose(x_final, expected, atol=1e-5)
  ```
- **Verdict:** CORRECT
- **Notes:** For the Euclidean metric $F(x,v) = \|v\|$, the energy is $E = \frac{1}{2}\|v\|^2$. Auto-differentiation gives $\nabla_x E = 0$, $\text{Hess}_v E = I$, mixed term $= 0$, so the spray $G^i = 0$ and $\ddot{x}^i = 0$. The solution is $x(t) = x_0 + v_0 t$, hence $x(1) = x_0 + v_0$. The assertion is mathematically correct.

  **NOTE** (tolerance): Because the ODE is linear ($\dot{v} = 0$), RK4 reproduces the exact solution up to floating-point roundoff ($\sim 10^{-16}$ in float64). The tolerance `atol=1e-5` is five orders of magnitude looser than necessary. Not wrong, but does not fully exploit the precision available from double-precision arithmetic. No action required.

---

### 2. `test_sphere_great_circle` — Quarter great circle on $S^2$

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (geodesic equation), § 4 (Riemannian row).
- **Literature Reference:** do Carmo, *Riemannian Geometry* (1992), Prop. 3.2 (geodesics on $S^n$ are great circles).
- **Implementation:** [tests/test_geodesic.py:40–57](tests/test_geodesic.py#L40-L57)
  ```python
  sphere = Sphere(radius=1.0)
  metric = Euclidean(sphere)
  x0 = jnp.array([1.0, 0.0, 0.0])
  v0 = jnp.array([0.0, 0.0, speed])   # speed = π/2
  x_final = self.solver.shoot(metric, x0, v0)
  expected = jnp.array([0.0, 0.0, 1.0])
  np.testing.assert_allclose(x_final, expected, atol=1e-3)
  ```
- **Verdict:** CORRECT (with caveats)
- **Notes:**
  1. **Tangency verified:** $x_0 \cdot v_0 = 0$ ✓.
  2. **Expected endpoint:** The great circle $\gamma(t) = \cos(\|v_0\| t)\,\hat{x}_0 + \sin(\|v_0\| t)\,\hat{v}_0$ at $t=1$ with $\|v_0\| = \pi/2$ gives $\gamma(1) = (0, 0, 1)$ ✓.
  3. **Tolerance 1e-3:** Appropriate. The projection-based integrator (Euclidean spray $= 0$, then project) is first-order in the constraint enforcement, yielding $O(h)$ drift per step even though RK4 is $O(h^4)$ for the unconstrained ODE. With $h = 10^{-3}$ and 1000 steps, cumulative error $\sim O(10^{-3})$ is expected.

  **WARNING** (spray is not tested): The `Euclidean` metric has $G^i \equiv 0$; the trajectory curves only because of the post-step projection $x \mapsto x/\|x\|$ in the solver. This means the test does **not** exercise the geodesic ODE $\ddot{x}^i + 2G^i = 0$ from `spec/MATH_SPEC.md` § 2.1. It only validates the projection-retraction mechanism. To test the spray, one should use a `Riemannian` metric $F(x,v) = \sqrt{v^T g(x) v}$ with a known $g(x)$ whose geodesics have closed-form solutions (e.g., the Poincaré disk, or the sphere's induced metric via pullback).

  **Recommended Action:** Add a companion test that uses a Riemannian metric with non-zero spray and verifies the endpoint against an analytically known geodesic.

---

### 3. `test_energy_conservation_randers` — Hamiltonian invariant

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 ($E = \frac{1}{2}F^2$) and § 2.1 (Euler-Lagrange $\Rightarrow$ energy conservation).
- **Literature Reference:** Bao–Chern–Shen, *Introduction to Riemann–Finsler Geometry* (2000), §5.3: along a geodesic of $F$, the Lagrangian energy $E(x, \dot{x}) = \frac{1}{2}F^2$ is a first integral.
- **Implementation:** [tests/test_geodesic.py:59–85](tests/test_geodesic.py#L59-L85)
  ```python
  energies = jax.vmap(metric.energy)(xs, vs)
  max_deviation = jnp.max(jnp.abs(energies - e_start))
  self.assertLess(max_deviation, 1e-4)
  ```
- **Verdict:** CORRECT
- **Notes:** The conservation of $E$ along geodesic flow follows from the 2-homogeneity of $E$ in $v$. By Euler's theorem, $\frac{\partial E}{\partial v^i} v^i = 2E$. Differentiating $E(\gamma, \dot\gamma)$ along the flow and applying Euler-Lagrange:

  $$\frac{dE}{dt} = \frac{\partial E}{\partial x^i}\dot{x}^i + \frac{\partial E}{\partial v^i}\ddot{x}^i = \frac{d}{dt}\!\left(\frac{\partial E}{\partial v^i}\right)\! v^i + \frac{\partial E}{\partial v^i}\dot{v}^i = \frac{d}{dt}\!\left(\frac{\partial E}{\partial v^i} v^i\right) = 2\frac{dE}{dt}$$

  which forces $\frac{dE}{dt} = 0$. The property is fundamental and correctly tested.

  **Tolerance 1e-4** is consistent with the Hessian regularisation $\text{Hess}_v(E) + 10^{-4} I$ in `metric.spray()` ([src/ham/geometry/metric.py:63](src/ham/geometry/metric.py#L63)), which introduces an $O(10^{-4})$ perturbation to the spray. This is the dominant error source, not the RK4 discretisation. The tolerance correctly tracks this.

- **STRONG:** This is an exemplary invariant-based test. Energy conservation probes the entire chain (metric $\to$ spray $\to$ ODE) in a single scalar check.

---

### 4. `test_manifold_adherence` — Constraint preservation on $S^2$

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1 (submanifold formulations).
- **Implementation:** [tests/test_geodesic.py:87–97](tests/test_geodesic.py#L87-L97)
  ```python
  radii = safe_norm(xs, axis=1)
  max_err = jnp.max(jnp.abs(radii - 1.0))
  self.assertLess(max_err, 1e-6)
  ```
- **Verdict:** CORRECT
- **Notes:** On a sphere of radius $R = 1$, the constraint is $\|x\| = 1$. The solver projects $x \mapsto x / \|x\|$ at every step, so the residual should be at machine precision ($\sim 10^{-16}$). The tolerance $10^{-6}$ is conservative and appropriate.

  **Initial condition verified:** $x_0 = (0.6, 0.8, 0)$, $\|x_0\|^2 = 0.36 + 0.64 = 1$ ✓. $v_0 = (0, 0, 1)$, $x_0 \cdot v_0 = 0$ ✓ (tangent).

- **STRONG:** Constraint preservation is a necessary condition for the solver to be geometrically meaningful. Good to have as a standalone test.

---

## Cross-Cutting Issues

### 5. `Sphere(1.0)` passes `1.0` to `intrinsic_dim`, not `radius`

- **Implementation:** [tests/test_geodesic.py:62](tests/test_geodesic.py#L62), [tests/test_geodesic.py:89](tests/test_geodesic.py#L89)
  ```python
  sphere = Sphere(1.0)   # test_energy_conservation_randers
  sphere = Sphere(1.0)   # test_manifold_adherence
  ```
- **Sphere constructor:** [src/ham/geometry/surfaces.py:28](src/ham/geometry/surfaces.py#L28)
  ```python
  def __init__(self, intrinsic_dim: int = 2, radius: float = 1.0):
  ```
- **Verdict:** WARNING
- **Notes:** `Sphere(1.0)` binds `intrinsic_dim = 1.0` (a float, not even `int`) and leaves `radius = 1.0` at its default. This creates a semantic $S^1$ (circle in $\mathbb{R}^2$) rather than $S^2$ (sphere in $\mathbb{R}^3$). The tests pass by accident because `Sphere.project()` and `Sphere.to_tangent()` are dimension-agnostic—they normalise whatever array is passed in. However:
  - The `ambient_dim` property returns `2.0` (a float), which would crash any code path that uses it for array shapes.
  - The docstring in `test_sphere_great_circle` says "Unit Sphere" but two sibling tests silently create a circle.

  Contrast with `test_sphere_great_circle`, which correctly uses `Sphere(radius=1.0)` (keyword argument).

  **Recommended Action:** Change `Sphere(1.0)` to `Sphere(radius=1.0)` on lines 62 and 89.

---

### 6. `step_size` constructor argument is dead code

- **Implementation:** [tests/test_geodesic.py:21](tests/test_geodesic.py#L21)
  ```python
  self.solver = ExponentialMap(step_size=0.002, max_steps=1000)
  ```
- **Verdict:** NOTE
- **Notes:** As documented in the [geodesic.py math review](geodesic.md), `step_size` is stored but never used. The actual timestep is `t_max / max_steps`. The effective $\Delta t = 1.0 / 1000 = 0.001$, not `0.002`. This is cosmetically misleading but does not affect correctness.

---

## Coverage Gaps

### 7. No test of the spray equation on a manifold with non-trivial curvature

- **Verdict:** WARNING
- **Notes:** All tests that verify trajectory geometry (`test_euclidean_ballistic`, `test_sphere_great_circle`, `test_manifold_adherence`) use the `Euclidean` metric, which has $G^i \equiv 0$. The non-trivial spray is only indirectly tested via energy conservation in the Randers test. There is no test that compares the solver output against a closed-form geodesic on a curved Riemannian manifold (e.g., Poincaré disk: $g_{ij} = \frac{4\delta_{ij}}{(1-\|x\|^2)^2}$, where geodesics are circular arcs).

  **Recommended Action:** Add a test with a `Riemannian` metric that has known closed-form geodesics (e.g., constant-curvature spaces) and assert endpoint agreement.

### 8. No zero-velocity edge case

- **Verdict:** NOTE
- **Notes:** For $v_0 = 0$, the geodesic should satisfy $x(t) = x_0$ for all $t$. This is a degenerate but important edge case, especially given the $v = 0$ singularity discussed in `spec/MATH_SPEC.md` § 6.1 and the `safe_norm` regularisation. No test covers it.

### 9. No speed-preservation test for Riemannian geodesics

- **Verdict:** NOTE
- **Notes:** Along a Riemannian geodesic, $F(\gamma(t), \dot\gamma(t)) = \text{const}$ (equivalently, $\|\dot\gamma\|_g = \text{const}$). The energy test (finding 3) checks $E = \frac{1}{2}F^2 = \text{const}$, which implies $F = \text{const}$, so this is implicitly covered. No action needed.

### 10. No reversibility test

- **Verdict:** NOTE
- **Notes:** The property $\text{Exp}_{x_0}(v_0) = x_1 \implies \text{Exp}_{x_1}(-\dot\gamma(1)) \approx x_0$ is a valuable sanity check for symmetric (Riemannian) metrics. For the asymmetric Randers case, the reverse geodesic is genuinely different, so reversibility would need the reverse metric $\tilde{F}(x,v) = F(x, -v)$. Not tested.

---

## Open Questions

1. **Is the Hessian regularisation scale ($10^{-4}$) in `metric.spray()` the intended floor for energy-conservation accuracy, or should the test tolerance be decoupled from it?** If the regularisation is later reduced (e.g., to $10^{-6}$), the test tolerance should tighten accordingly.

2. **Should the `Plane` class in `test_euclidean_ballistic` inherit from `Manifold`?** Currently it is a bare class with `project`/`to_tangent` methods, relying on duck typing. This works but may break if `FinslerMetric` or Equinox later enforce the type annotation `manifold: Manifold`.

3. **The projection-based integrator (Euclidean spray + project) converges to a great circle, but at what order?** The test uses `atol=1e-3`. A theoretical convergence-rate analysis (expected $O(h)$ for naive projection, $O(h^2)$ with tangent correction) would inform whether the tolerance should be tighter with 1000 steps.
