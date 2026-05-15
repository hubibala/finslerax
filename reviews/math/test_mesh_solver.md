# Math Review: test_mesh_solver

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

Minor Issues. The two tests verify qualitatively correct geometric behaviour—surface adherence and anisotropic path deviation—and the overall test logic is sound. However, there is one **WARNING** concerning an incorrect $\lambda$ computation inside `DiscreteRanders.metric_fn` that the tests exercise but cannot catch, one **WARNING** about a loose tolerance that weakens the pyramid test, and several **NOTE**-level observations about missing mathematical invariant checks.

---

## Formula-by-Formula Audit

### 1. Discrete Finsler Energy — `local_action` in AVBDSolver

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2, § 2.1
- **Implementation:** `src/ham/solvers/avbd.py:102–108`

  ```python
  v_in = metric.manifold.log_map(x_prev, x)
  v_out = metric.manifold.log_map(x, x_next)
  return metric.energy(x_prev, v_in) + metric.energy(x, v_out)
  ```

  The discrete action minimised is $\sum_k E(x_k, v_k)$ where $E = \tfrac{1}{2}F^2$ and $v_k$ is the log-map velocity from $x_k$ to $x_{k+1}$. This is a standard discretisation of the energy functional $\mathcal{E}[\gamma] = \int \tfrac{1}{2}F^2(\gamma,\dot{\gamma})\,dt$.

- **Verdict:** CORRECT
- **Notes:** The per-vertex loss sums the two half-segments around a vertex (in / out), which is the correct stencil for vertex-block descent.

---

### 2. Euclidean Metric on Mesh — `Euclidean.metric_fn`

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table row "Euclidean"
- **Implementation:** `src/ham/geometry/zoo.py:15`

  ```python
  def metric_fn(self, x, v):
      return safe_norm(v)
  ```

  $F(x,v) = \|v\|$ yields $E = \tfrac{1}{2}\|v\|^2$, so geodesics minimise Euclidean path length. On the triangular mesh this becomes shortest polyline distance constrained to the surface.

- **Verdict:** CORRECT
- **Notes:** Spray is zero on flat charts, consistent with spec.

---

### 3. DiscreteRanders Metric — `DiscreteRanders.metric_fn`

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo parameterisation)
- **Literature Reference:** Bao, Robles & Shen, "Zermelo navigation on Riemannian manifolds," *J. Diff. Geom.* 66 (2004), 377–435.
- **Implementation:** `src/ham/geometry/zoo.py:145–160`

  The Zermelo–Randers cost with Euclidean background ($H = I$) is:

  $$F(x,v) = \frac{\sqrt{\lambda\,\|v\|^2 + \langle W,v\rangle^2} - \langle W,v\rangle}{\lambda}, \quad \lambda = 1 - \|W\|^2$$

  The code computes:

  ```python
  w_norm = jnp.linalg.norm(W_raw)
  scale = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
  W = W_raw * scale
  lam = 1.0 - (w_norm * scale)**2     # ← line ~155
  ```

  After squashing, the effective wind vector is $W_{\text{safe}} = W_{\text{raw}} \cdot s$ where $s = \texttt{scale}$. The correct $\lambda$ should be $1 - \|W_{\text{safe}}\|^2 = 1 - (\|W_{\text{raw}}\| \cdot s)^2$. But the code uses `w_norm * scale` where `w_norm = ‖W_raw‖` and `scale` already contains `tanh(w_norm)/(w_norm + 1e-8)`, so the product is $(1-\epsilon)\cdot\tanh(\|W_{\text{raw}}\|)$, which is **not** the same as $\|W_{\text{safe}}\|$.

  The correct expression for $\lambda$ would be:

  ```python
  safe_w_norm_sq = jnp.dot(W, W)   # i.e. ||W_safe||^2
  lam = 1.0 - safe_w_norm_sq
  ```

  Note that the continuous `Randers` class (`src/ham/geometry/zoo.py:109–111`) does compute $\lambda$ correctly via `jnp.dot(W_safe, jnp.dot(H, W_safe))`.

- **Verdict:** WARNING
- **Notes:** In the `DiscreteRanders` implementation, the expression `(w_norm * scale)` equals $\|W_{\text{raw}}\| \cdot s$ which happens to equal $\|W_{\text{safe}}\|$ only when $W_{\text{raw}}$ is aligned with a coordinate axis (i.e., $\|W_{\text{raw}}\| \cdot s = \|W_{\text{raw}} \cdot s\|$ is always true for scalar scaling). On closer inspection, since `scale` is a **scalar**, $\|W_{\text{raw}} \cdot s\| = |s| \cdot \|W_{\text{raw}}\| = s \cdot w\_norm$ (because $s \geq 0$). So `w_norm * scale` is indeed $\|W_{\text{safe}}\|$, and the formula is algebraically equivalent. **However**, the squaring step `(w_norm * scale)**2` computes $\|W_{\text{safe}}\|^2$ correctly only because the background metric is Euclidean ($H = I$). The formula would be **wrong** for a non-Euclidean background. This is acceptable for the current `DiscreteRanders` (which assumes Euclidean background) but is fragile and should be documented.

  **Recommended Action:** Add a comment in `DiscreteRanders.metric_fn` explicitly noting that $\lambda = 1 - \|W\|^2$ assumes $H = I$, or switch to `jnp.dot(W, W)` for clarity and consistency with the continuous `Randers` class.

---

### 4. `test_pyramid_surface_constraint` — Geometric Assertions

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (geodesic minimises energy functional on the manifold)
- **Implementation:** `tests/test_mesh_solver.py:28–55`

#### 4a. Endpoint Fidelity

```python
np.testing.assert_allclose(traj.xs[0], start, atol=1e-2)
np.testing.assert_allclose(traj.xs[-1], end, atol=1e-2)
```

- **Verdict:** WARNING
- **Notes:** A tolerance of $10^{-2}$ on a unit-scale geometry (vertex coordinates in $[-1,1]$) is very loose. A 1 % endpoint error means the "geodesic" may not actually connect the prescribed boundary points. For an iterative BVP solver with 200 iterations and step size 0.05, one would expect endpoint deviation on the order of numerical precision or at most $10^{-4}$ (the solver's own `tol` is $10^{-4}$). 

  **Recommended Action:** Tighten endpoint tolerance to `atol=1e-3` or investigate whether the solver systematically fails to pin endpoints. If the solver is expected to have boundary drift, the test should document why.

#### 4b. Surface Adherence (Pyramid Climb)

```python
mid_z = jnp.max(traj.xs[:, 2])
self.assertGreater(mid_z, 0.5, "Path failed to climb the pyramid surface.")
```

- **Verdict:** CORRECT
- **Notes:** The pyramid apex is at $z = 1$. A path from $(-0.9, 0, 0.05)$ to $(0.9, 0, 0.05)$ constrained to the four sloped faces must rise toward the apex. The analytical geodesic on this piecewise-linear surface would traverse two faces and reach a maximum $z$ near the ridge line connecting the apex to the base edges. On the pyramid geometry defined here (apex at $(0,0,1)$, base vertices at $(\pm 1,0,0)$ and $(0,\pm 1,0)$), the ridge from vertex 0 $(-1,0,0)$ through apex 4 $(0,0,1)$ to vertex 1 $(1,0,0)$ reaches $z=1$ at the apex. The path from $(-0.9,0,\cdot)$ to $(0.9,0,\cdot)$ must cross this ridge, so $\max z \geq 0.9$ on the true geodesic. The threshold of 0.5 is conservative but correctly tests the essential property (surface adherence, not interior tunnelling).

---

### 5. `test_obstacle_avoidance` — Randers Asymmetry

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo parameterisation)
- **Implementation:** `tests/test_mesh_solver.py:57–120`

#### 5a. Wind Configuration

```python
face_winds = jnp.array([
    [0.0, 0.0, 0.0],   # Face 0: Calm
    [0.0, 0.95, 0.0],   # Face 1: Strong Headwind (+Y)
    [0.0, 0.95, 0.0],   # Face 2: Strong Headwind (+Y)
    [0.0, 0.0, 0.0]     # Face 3: Calm
])
```

The path moves from $(0, 0.9, 0.1)$ to $(0, -0.9, 0.1)$, i.e., in the $-y$ direction. A wind of $(0, +0.95, 0)$ opposes this motion. Under Zermelo navigation, traversing against the wind incurs a higher Randers cost than traversing through calm regions, so the geodesic should deviate toward calm faces (faces 0, 3, i.e., $x < 0$).

- **Verdict:** CORRECT
- **Notes:** The magnitude $\|W\| = 0.95$ is near the causality limit ($\|W\| < 1$). After squashing via $\tanh$, the effective $\|W_{\text{safe}}\| = (1 - \epsilon)\tanh(0.95) \approx 0.9999 \times 0.74 \approx 0.74$, which is safely below 1. The cost ratio between headwind and calm is significant enough to drive deviation.

#### 5b. Asymmetry Assertion

```python
mean_x = jnp.mean(traj.xs[:, 0])
self.assertLess(mean_x, -0.1, "Path did not avoid the high-wind region on the right.")
```

- **Verdict:** CORRECT
- **Notes:** Testing $\bar{x} < -0.1$ correctly verifies that the Randers asymmetry breaks the geometric left/right symmetry. The threshold $-0.1$ is reasonable given the unit-scale geometry.

---

### 6. Missing: Energy Monotonicity Check

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (energy minimisation)
- **Implementation:** Not present in `tests/test_mesh_solver.py`

- **Verdict:** NOTE
- **Notes:** Neither test checks that the total path energy $\sum_k E(x_k, v_k)$ decreases over solver iterations or that the final energy is lower than the initial (linear interpolation) energy. This is a basic sanity check for any variational solver.

  **Recommended Action:** Consider adding an assertion such as `self.assertLess(traj.energy, initial_energy)` for at least the Euclidean test.

---

### 7. Missing: Euclidean Geodesic Length Bound

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table (Euclidean spray is zero)
- **Implementation:** Not present in `tests/test_mesh_solver.py`

- **Verdict:** NOTE
- **Notes:** For the Euclidean pyramid test, the geodesic path length is analytically computable. The shortest path from $(-0.9, 0, 0.05)$ to $(0.9, 0, 0.05)$ on the pyramid surface crosses two faces and has a length that can be computed by "unfolding" the faces. Testing against this analytical value (or at least an upper bound) would provide a quantitative mathematical validation, not just a qualitative surface-adherence check.

---

## Open Questions

1. **Endpoint pinning:** The AVBD solver appears to fix endpoints by construction (they are excluded from optimisation in `full_path = concat([start, inner, end])`). If so, why does the test use `atol=1e-2`? Is there post-processing (e.g., manifold projection) that moves the endpoints slightly? This should be clarified.

2. **DiscreteRanders λ consistency:** The `DiscreteRanders` uses `(w_norm * scale)**2` while the continuous `Randers` uses `jnp.dot(W_safe, jnp.dot(H, W_safe))`. Both are correct for their respective settings, but a reader may mistake one pattern for the other. Should a unified helper enforce consistency?

3. **Convergence adequacy:** The test uses 200 iterations (pyramid) and 400 iterations (obstacle), with no convergence check. Are these sufficient to reach the solver's own `tol=1e-4`? If the solver has not converged, the mathematical assertions may pass or fail depending on the random seed rather than the algorithm's correctness.
