# Math Review: test_solver

**Reviewer:** Math Reviewer Agent
**Date:** 2025-05-15
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Minor Issues.** The three tests in `tests/test_solver.py` assert geometrically meaningful properties (torus wrapping, Randers asymmetry, paraboloid dipping) and all expected values are analytically correct. However, the test suite has significant coverage gaps: there is no test of the IVP solver (`ExponentialMap` in `geodesic.py`), no energy-conservation check, no verification of the geodesic ODE $\ddot{x}^i + 2G^i = 0$ itself, and no flat-space baseline. Additionally, the `tol` parameter set in `setUp` has no effect on solver behaviour, making the apparent precision guarantee illusory.

---

## Formula-by-Formula Audit

### 1. Torus surface constraint (`test_torus_topology`, lines 33–37)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 (Geometric Hierarchy, Torus not explicitly listed but falls under the manifold framework)
- **Literature Reference:** Standard torus embedding: $(\sqrt{x^2+y^2}-R)^2 + z^2 = r^2$
- **Implementation:**
  ```python
  xy = safe_norm(traj.xs[:,:2], axis=1)
  dist = jnp.abs((xy - torus.R)**2 + traj.xs[:,2]**2 - torus.r**2)
  ```
  Tests that $|(\sqrt{x_i^2 + y_i^2} - R)^2 + z_i^2 - r^2| < 0.1$ for all path points.
- **Verdict:** CORRECT
- **Notes:** The implicit surface equation is standard. The tolerance $0.1$ is generous but appropriate for a discrete BVP solver with only 20 path segments. No mathematical error.

### 2. Torus topology assertion (`test_torus_topology`, lines 40–42)

- **Spec Reference:** N/A (topological, not in spec)
- **Literature Reference:** Geodesic from outer equator $(R+r, 0, 0)$ to inner equator $(R-r, 0, 0)$ on a torus must wrap over or under the tube.
- **Implementation:**
  ```python
  max_z = jnp.max(jnp.abs(traj.xs[:, 2]))
  self.assertGreater(max_z, 0.5, ...)
  ```
- **Verdict:** CORRECT
- **Notes:** The start point $(3, 0, 0) = (R+r, 0, 0)$ and end point $(1, 0, 0) = (R-r, 0, 0)$ are diametrically opposite on the tube cross-section. Any on-surface path connecting them must attain $|z| \geq r = 1.0$ (it must clear the tube), so the threshold $0.5$ is a conservative lower bound. Correct.

### 3. Test point coordinates on the sphere (`test_sphere_zermelo`, lines 55–57)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo/Randers)
- **Implementation:**
  ```python
  p_west = jnp.array([-0.5, 0.0, 0.866])
  p_east = jnp.array([ 0.5, 0.0, 0.866])
  ```
- **Verdict:** CORRECT
- **Notes:** $\sqrt{0.5^2 + 0^2 + 0.866^2} = \sqrt{0.25 + 0.75} = 1.0$. Both points lie on $S^2(1)$; the `sphere.project` call is a no-op for these inputs.

### 4. Randers wind direction and energy asymmetry (`test_sphere_zermelo`, lines 51–53, 68)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, Zermelo navigation formula
- **Literature Reference:** Bao, Robles, Shen, "Zermelo navigation on Riemannian manifolds," *J. Diff. Geom.* 66(3), 2004.
- **Implementation:**
  ```python
  h_net = lambda x: jnp.eye(3)
  w_net = lambda x: jnp.array([0.5, 0.0, 0.0])
  ...
  self.assertLess(traj_down.energy, traj_up.energy)
  ```
  The Zermelo–Randers metric from `spec/MATH_SPEC.md` § 5:
  $$F(x,v) = \frac{\sqrt{\lambda\|v\|_h^2 + \langle W,v\rangle_h^2} - \langle W,v\rangle_h}{\lambda}, \quad \lambda = 1 - \|W\|_h^2$$
  When $v$ is aligned with $W$ (downwind), $\langle W,v\rangle_h > 0$, the $-\langle W,v\rangle_h$ term reduces $F$. Conversely, upwind travel increases $F$.
- **Verdict:** CORRECT
- **Notes:** The sign convention matches `spec/MATH_SPEC.md` § 5 ("minus sign convention … headwind increases cost"). The Randers `metric_fn` in [src/ham/geometry/zoo.py](src/ham/geometry/zoo.py#L131-L140) faithfully implements this formula. The ambient-space identity matrix $H = I_3$ correctly induces the round metric on the sphere.

### 5. Randers wind causality ($\|W\|_h < 1$) (`test_sphere_zermelo`, line 53)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 — "$\|W\|_h < 1$"
- **Implementation:** `w_net = lambda x: jnp.array([0.5, 0.0, 0.0])` with `h_net = lambda x: jnp.eye(3)`.
  After tangent-plane projection by `_get_zermelo_data` ([src/ham/geometry/zoo.py](src/ham/geometry/zoo.py#L92)), the tangential component satisfies $\|W_\text{tan}\| \leq 0.5 < 1$.
- **Verdict:** CORRECT
- **Notes:** Causality is preserved everywhere on the sphere. No issue.

### 6. Paraboloid constraint function (`test_paraboloid_implicit`, lines 79–80)

- **Spec Reference:** N/A (Paraboloid $z = x^2 + y^2$ is standard)
- **Implementation:**
  ```python
  def para_c(x): return x[2] - (x[0]**2 + x[1]**2)
  ```
- **Verdict:** CORRECT
- **Notes:** The zero-level set $c(x) = 0$ is exactly $z = x^2 + y^2$. The start $(-1, 0, 1)$ and end $(1, 0, 1)$ satisfy $c = 0$. ✓

### 7. Paraboloid midpoint dip assertion (`test_paraboloid_implicit`, lines 88–90)

- **Spec Reference:** N/A (geometric reasoning)
- **Literature Reference:** The paraboloid $z = x^2 + y^2$ restricted to the $y = 0$ plane yields a 1D curve with induced metric $ds^2 = (1 + 4x^2)\,dx^2$. In any 1D Riemannian manifold all paths are geodesics up to reparameterisation, so the geodesic from $x = -1$ to $x = 1$ passes through $x = 0 \Rightarrow z = 0$.
- **Implementation:**
  ```python
  mid = traj.xs[10]    # index 10 of 21 points (n_steps=20)
  self.assertLess(mid[2], 0.2, "Path did not dip to follow surface.")
  ```
- **Verdict:** CORRECT
- **Notes:** By symmetry the geodesic lies in the $y = 0$ plane and must pass through the vertex $(0, 0, 0)$. The threshold $z < 0.2$ is loose but analytically justified.

### 8. Paraboloid constraint redundancy (`test_paraboloid_implicit`, line 86)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.2 (implicit solve strategy)
- **Implementation:**
  ```python
  traj = self.solver.solve(metric, start, end, n_steps=20, constraints=[para_c])
  ```
  The AVBD solver already enforces the manifold constraint via `retract` → `project` at every vertex update ([src/ham/solvers/avbd.py](src/ham/solvers/avbd.py#L133)). Passing `constraints=[para_c]` adds an augmented-Lagrangian penalty for the *same* constraint.
- **Verdict:** NOTE
- **Notes:** This double enforcement is not mathematically wrong but makes the test ambiguous: it is unclear whether the test verifies the retraction-based constraint enforcement or the augmented-Lagrangian path. A separate test isolating each mechanism would be more informative.

---

## Structural / Coverage Findings

### 9. `tol` parameter is unused — convergence tolerance is illusory

- **File:** [tests/test_solver.py](tests/test_solver.py#L17), [src/ham/solvers/avbd.py](src/ham/solvers/avbd.py#L161-L162)
- **Severity:** WARNING
- **Details:** The test sets `tol=1e-6`, but the AVBD solver's loop condition is simply `s.step < self.iterations` ([src/ham/solvers/avbd.py](src/ham/solvers/avbd.py#L161)). Neither `tol` nor `energy_tol` is checked anywhere in the iteration loop. The solver always runs for exactly 200 iterations regardless of convergence. This means the test does not actually verify convergence to any mathematical precision bound.
- **Recommended Action:** Either wire `tol` into the solver's convergence check, or remove it from the test to avoid implying a precision guarantee that does not exist.

### 10. No test of the IVP solver (`ExponentialMap` / `geodesic.py`)

- **File:** [src/ham/solvers/geodesic.py](src/ham/solvers/geodesic.py) (entire module)
- **Severity:** WARNING
- **Details:** The `ExponentialMap` class implements RK4 integration of the geodesic spray ODE $\ddot{x}^i + 2G^i(x, \dot{x}) = 0$ (`spec/MATH_SPEC.md` § 2.1). This is a fundamentally different solver path from the BVP-based `AVBDSolver` tested here. No test verifies the correctness of the RK4 stepper, the spray evaluation via `geod_acceleration`, or the endpoint accuracy of the exponential map.
- **Recommended Action:** Add at least one IVP test (e.g., shoot a geodesic on the sphere with known initial velocity and verify the endpoint matches the analytical exponential map $\exp_x(v)$).

### 11. No energy-conservation test along geodesics

- **File:** N/A
- **Severity:** WARNING
- **Details:** A fundamental property of the geodesic flow is conservation of the Finsler energy $E(x, \dot{x})$ along a solution of $\ddot{x}^i + 2G^i = 0$ (`spec/MATH_SPEC.md` § 2.1). This is a strong diagnostic: if the integrator introduces spurious energy drift, the geodesic is inaccurate. No test checks $E(\gamma(t_k), \dot{\gamma}(t_k)) \approx \text{const}$ along a trajectory.
- **Recommended Action:** After tracing a geodesic with `ExponentialMap.trace`, compute $E$ at each step and assert $\max_k |E_k - E_0| / E_0 < \epsilon$ for an appropriate $\epsilon$.

### 12. No flat-space (Euclidean) baseline test

- **File:** N/A
- **Severity:** WARNING
- **Details:** In flat Euclidean space, geodesics are straight lines. This is the simplest possible sanity check: solve for a geodesic between two points in $\mathbb{R}^n$ and verify the path is linear to machine precision. Its absence means there is no baseline confirming the solver works for the trivial case before testing on curved manifolds.
- **Recommended Action:** Add a test that solves for a Euclidean geodesic and asserts each interior point lies on the line segment $p_\text{start} + t(p_\text{end} - p_\text{start})$ within $O(10^{-6})$.

### 13. No test of the geodesic ODE residual

- **File:** N/A
- **Severity:** NOTE
- **Details:** None of the tests verify the geodesic equation $\ddot{x}^i + 2G^i = 0$ directly. The BVP tests check geometric properties (wrapping, asymmetry, dipping) which are necessary but not sufficient for verifying the underlying ODE. A residual test would compute $\ddot{x}^i + 2G^i$ at sample points along the trajectory and check it is near zero.

### 14. No test for degenerate/edge cases

- **File:** N/A
- **Severity:** NOTE
- **Details:** Missing edge-case tests include:
  - **Zero-length geodesic** ($p_\text{start} = p_\text{end}$): should return a constant path.
  - **Antipodal points on the sphere**: geodesic is non-unique; solver behaviour should be well-defined.
  - **Near-zero velocity**: the spray $G^i$ is singular at $v = 0$ (`spec/MATH_SPEC.md` § 6.1); the test suite does not exercise this regime.

---

## Strong Points

### S1. Correct use of Zermelo navigation for Randers testing

- **File:** [tests/test_solver.py](tests/test_solver.py#L44-L72)
- **Severity:** STRONG
- **Details:** The wind asymmetry test is the right way to verify a Randers metric: it checks the *qualitative* physical prediction (downwind < upwind) that distinguishes Finsler from Riemannian geometry, rather than comparing to a hard-coded numerical value that would be fragile.

### S2. Correct topological assertion for torus

- **File:** [tests/test_solver.py](tests/test_solver.py#L40-L42)
- **Severity:** STRONG
- **Details:** Checking that $\max|z| > 0.5$ is a robust topological invariant that will not break under minor solver parameter changes, unlike a pointwise coordinate comparison.

---

## Open Questions

1. **Is the 200-iteration budget sufficient for convergence on all tested geometries?** Since `tol` is unused (Finding 9), there is no programmatic evidence that the solver has actually converged. The test pass/fail depends entirely on whether 200 iterations happen to be enough.

2. **Should the Randers asymmetry test include a quantitative bound?** Currently it checks only `E_\text{down} < E_\text{up}`. A stronger check would verify the ratio or difference against the analytical prediction for a constant-wind Randers metric on the sphere, if such a closed-form is available.

3. **Why is the IVP solver (`ExponentialMap`) entirely untested?** If it is unused in the current pipeline, it may still harbour latent bugs that surface when integrated later (e.g., for shooting-based training or exponential-map layers).
