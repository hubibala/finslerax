# Math Review: test_geodesic_learning

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

The test file is **internally consistent** and tests a valid learning signal (wind field recovery from paired trajectory data). However, the core loss function conflates the Zermelo wind velocity $W(x)$ with a finite-time geodesic displacement $\log_x(y)$, which is only directionally correct (saved by cosine similarity being scale-invariant). The synthetic data generators are mathematically sound for all four manifolds. Two medium-severity issues involve evaluation methodology (train vs. held-out) and missing coverage of fundamental Finsler metric properties. No critical formula errors were found.

## Formula-by-Formula Audit

### 1. DirectWindAlignmentLoss — Wind ≈ Displacement identification

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization)
- **Literature Reference:** Bao–Robles–Shen, "Zermelo navigation on Riemannian manifolds," *J. Diff. Geom.* 66 (2004), §2
- **Implementation:** [tests/test_geodesic_learning.py](tests/test_geodesic_learning.py#L137-L145)
  ```python
  v_true = model.manifold.log_map(start, end)
  _, W, _ = model._get_zermelo_data(start)
  return jnp.mean((W - v_true) ** 2) * self.weight
  ```
- **Verdict:** WARNING
- **Notes:** In the Zermelo navigation picture, $W(x)$ is a *velocity field* (drift per unit time). The loss trains $W(x) = \log_x(y)$, which is the *displacement* over a finite time step $\delta t$. These differ by a factor of $\delta t$:

  $$\log_x\bigl(\operatorname{retract}(x, W(x)\,\delta t)\bigr) \approx W(x)\,\delta t$$

  The downstream assertions use cosine similarity, which is scale-invariant, so the *directional* content is preserved. However, the learned $\|W\|$ will be proportional to $\delta t$ rather than representing the physical wind speed. Any future test that checks magnitudes or uses $W$ to compute actual Randers geodesic costs will be wrong without rescaling.

  **Recommended Action:** Document that the learned $W$ encodes displacement (not velocity) and that cosine similarity is required for comparison, or normalise $v_{\mathrm{true}}$ by the data-generation time step.

---

### 2. generate_river_data — Constant flow generation

- **Spec Reference:** N/A (synthetic data)
- **Implementation:** [tests/test_geodesic_learning.py](tests/test_geodesic_learning.py#L41-L47)
  ```python
  starts = jax.random.uniform(key, (n, 2), minval=-2.0, maxval=2.0)
  flow = jnp.array([1.0, 0.0])
  noise = jax.random.normal(key, (n, 2)) * 0.03
  ends = starts + flow * 0.5 + noise
  ```
- **Verdict:** NOTE
- **Notes:** The same `PRNGKey(0)` is used for both `starts` and `noise`, introducing correlation between positions and noise. This does not invalidate the test but weakens its statistical power — noise is deterministically coupled to position rather than being independent.

  **Recommended Action:** Split the key: `k1, k2 = jax.random.split(key)`, then use `k1` for starts and `k2` for noise.

---

### 3. generate_vortex_data — Rotation matrix

- **Spec Reference:** N/A (synthetic data)
- **Implementation:** [tests/test_geodesic_learning.py](tests/test_geodesic_learning.py#L50-L57)
  ```python
  R = jnp.array([[c, -s], [s, c]])
  ends = jnp.dot(starts, R.T)
  ```
- **Verdict:** CORRECT
- **Notes:** For row-vector convention, $x R^T = (R x^T)^T$ applies the CCW rotation by angle $\delta t = 0.3$ to each row vector. The displacement field $\Delta x = (R - I)x$ is position-dependent with purely rotational structure. ✓

---

### 4. generate_sphere_vortex — Tangent wind on $S^2$

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 (Geometric Hierarchy — Sphere)
- **Implementation:** [tests/test_geodesic_learning.py](tests/test_geodesic_learning.py#L64-L67)
  ```python
  v = jnp.array([-x[1], x[0], 0.0])
  mag = 1.5 * jnp.exp(-5.0 * x[2] ** 2)
  return manifold.to_tangent(x, mag * v)
  ```
- **Verdict:** CORRECT
- **Notes:** The vector $(-x_1, x_0, 0)$ is the infinitesimal generator of rotation about the $z$-axis. For a point $x$ on the unit sphere, $\langle x, v \rangle = x_0(-x_1) + x_1 x_0 + x_2 \cdot 0 = 0$, so $v$ is *already* tangent — the `to_tangent` projection is a no-op here. The Gaussian attenuation $\exp(-5x_2^2)$ damps the field near the poles where the rotation axis meets the sphere. ✓

---

### 5. generate_hyperboloid_vortex — Tangent wind on $\mathbb{H}^2$

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 (Geometric Hierarchy — Hyperboloid)
- **Implementation:** [tests/test_geodesic_learning.py](tests/test_geodesic_learning.py#L96-L98)
  ```python
  v_rot = jnp.array([0.0, -x[2], x[1]])
  return manifold.to_tangent(x, 0.5 * v_rot)
  ```
- **Verdict:** CORRECT
- **Notes:** The Minkowski inner product $\langle x, v_{\mathrm{rot}} \rangle_L = -x_0 \cdot 0 + x_1(-x_2) + x_2 x_1 = 0$, so the rotation vector is tangent to the hyperboloid at $x$ and the `to_tangent` call (which adds $\langle x, v\rangle_L \cdot x$) is again a no-op. ✓

  Training data is sampled near the tip (tangent vectors with $\sigma = 0.8$), which keeps points in a region where the hyperboloid curvature is moderate.

---

### 6. Randers metric_fn (cross-reference)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, Zermelo navigation formula
- **Implementation:** [src/ham/geometry/zoo.py](src/ham/geometry/zoo.py#L132-L142)
  ```python
  discriminant = lam * v_sq_h + W_dot_v**2
  cost = (jnp.sqrt(jnp.maximum(discriminant, 0.0) + 1e-9) - W_dot_v) / lam
  ```
- **Verdict:** CORRECT
- **Notes:** This correctly implements:
  $$F(x,v) = \frac{\sqrt{\lambda \|v\|_h^2 + \langle W, v\rangle_h^2} - \langle W, v\rangle_h}{\lambda}, \quad \lambda = 1 - \|W\|_h^2$$
  The `1e-9` inside `sqrt` is consistent with `spec/MATH_SPEC.md` § 6.1 (epsilon regularisation). ✓

---

### 7. cosine_similarity

- **Spec Reference:** N/A (standard metric)
- **Implementation:** [tests/test_geodesic_learning.py](tests/test_geodesic_learning.py#L213-L218)
  ```python
  dots = jnp.sum(a * b, axis=-1)
  return float(jnp.mean(dots / (na * nb + 1e-8)))
  ```
- **Verdict:** CORRECT
- **Notes:** Standard formula $\cos\theta = \frac{a \cdot b}{\|a\|\,\|b\|}$, averaged over samples. The `1e-8` additive guard prevents division by zero when both vectors vanish simultaneously. ✓

---

### 8. MetricIdentityLoss — H(x) anchoring

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo — the "sea" metric $h_{ij}$)
- **Implementation:** [tests/test_geodesic_learning.py](tests/test_geodesic_learning.py#L163-L172)
  ```python
  H, _, _ = model._get_zermelo_data(start)
  I = jnp.eye(H.shape[-1])
  return jnp.mean((H - I) ** 2) * self.weight
  ```
- **Verdict:** CORRECT
- **Notes:** Anchoring $H \approx I$ with `weight=5.0` (vs. `weight=1.0` for wind alignment) effectively constrains the sea to remain Euclidean, reducing the Randers metric to $F(x,v) \approx \frac{\sqrt{\lambda\|v\|^2 + (W \cdot v)^2} - W \cdot v}{\lambda}$. This is appropriate for tests focused on wind recovery, but means the tests do **not** verify joint learning of a non-trivial Riemannian sea and wind.

---

### 9. test_vortex_direction — Evaluation on training data

- **Spec Reference:** N/A (test methodology)
- **Implementation:** [tests/test_geodesic_learning.py](tests/test_geodesic_learning.py#L299-L302)
  ```python
  true_disp = jax.vmap(manifold.log_map)(dataset.starts[:100], dataset.ends[:100])
  pred_disp = jax.vmap(get_w)(dataset.starts[:100])
  cos_sim = cosine_similarity(true_disp, pred_disp)
  ```
- **Verdict:** WARNING
- **Notes:** This evaluates the learned $W$ on *training points* (`dataset.starts[:100]`), measuring memorisation rather than generalisation. The river, hyperboloid, and sphere tests correctly evaluate on held-out random points, but this vortex test does not. A neural network with sufficient capacity could achieve high cosine similarity on training data without learning the correct rotational structure.

  **Recommended Action:** Evaluate on fresh random points (as in the other three tests) and compare $W(x)$ against the analytic displacement $(R - I)x$.

---

### 10. test_hyperboloid_vortex_direction — Distribution mismatch

- **Spec Reference:** N/A (test methodology)
- **Implementation:** [tests/test_geodesic_learning.py](tests/test_geodesic_learning.py#L315-L318)
  ```python
  eval_pts = jax.vmap(manifold.random_sample, in_axes=(0, None))(
      jax.random.split(jax.random.PRNGKey(999), 200), ()
  )
  ```
- **Verdict:** WARNING
- **Notes:** Training data is concentrated near the hyperboloid tip (tangent vectors $\sim \mathcal{N}(0, 0.8^2)$ from origin), while `random_sample` draws tangent vectors $\sim \mathcal{N}(0, 1)$. Points at $\|v\| \gg 1$ lie in an extrapolation regime where the neural network has seen no training signal. The lenient threshold ($\cos > 0.60$) partially compensates, but makes the test susceptible to flakiness if the evaluation distribution shifts further from training.

  **Recommended Action:** Sample evaluation points from the same distribution as training data (small tangent vectors from origin), or use a kernel-based interpolation to ensure the test is deterministic.

---

### 11. Test tolerances

- **Spec Reference:** N/A
- **Implementation:** Various `assertGreater` calls
- **Verdict:** NOTE
- **Notes:** Summary of thresholds:

  | Test | Threshold | Assessment |
  |------|-----------|------------|
  | River (cos) | 0.85 | Reasonable for constant field |
  | Vortex (cos) | 0.80 | Somewhat lenient for training-set evaluation |
  | Hyperboloid (cos) | 0.60 | Very lenient; see Finding 10 |
  | Sphere (cos) | 0.70 | Acceptable given non-uniform manifold curvature |
  | Loss decrease | 50% | Reasonable sanity check |

  The hyperboloid threshold of 0.60 is notably lower than others. A cosine similarity of 0.60 allows up to ~53° angular deviation, which may not meaningfully validate directional recovery.

---

## Missing Coverage

### M1. Finsler metric validity — no assertion on positive homogeneity

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1, property (2): $F(x, \lambda v) = \lambda F(x, v)$ for $\lambda > 0$
- **Verdict:** WARNING
- **Notes:** No test checks that the *learned* Randers metric satisfies the fundamental axiom of positive homogeneity after training. Homogeneity is enforced by construction in the Randers formula, but a test assertion would catch regressions in `metric_fn` or `_get_zermelo_data`.

  **Recommended Action:** Add an assertion: for several $(x, v)$ pairs, check $F(x, 2v) \approx 2 F(x, v)$.

### M2. Causality constraint $\|W\|_H < 1$ — never asserted

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (strict convexity requires $\|W\|_h < 1$)
- **Implementation:** Enforced via `tanh` squasher in [src/ham/geometry/zoo.py](src/ham/geometry/zoo.py#L100-L105), but never tested.
- **Verdict:** WARNING
- **Notes:** The `_get_zermelo_data` squasher enforces $\|W\|_H < 1 - \epsilon$, but no test asserts that $\lambda > 0$ after training. A negative $\lambda$ would produce imaginary metric values.

  **Recommended Action:** After training, evaluate $\lambda(x)$ on a grid and assert $\lambda > 0$ everywhere.

### M3. No geodesic-level validation

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (geodesic equation $\ddot{x}^i + 2G^i = 0$)
- **Verdict:** NOTE
- **Notes:** The test suite is named "Geodesic Learning" but contains no test that actually computes a Randers geodesic under the learned metric and checks it connects the paired endpoints. This is an integration gap — the tests verify the wind field proxy, not the full geodesic pipeline. This is acceptable given the separate `test_geodesic.py` file, but worth noting.

## Open Questions

1. **Intent of magnitude**: Is the learned $W(x)$ intended to represent a physical wind velocity ($\mathrm{length}/\mathrm{time}$) or a displacement ($\mathrm{length}$)? If the former, the loss should divide by $\delta t$. The downstream use in biological applications (Weinreb experiment) may depend on this distinction.

2. **Joint H–W learning**: All tests anchor $H \approx I$ with high weight. Is there a deliberate omission of tests for jointly learned $(H, W)$ due to known instability (cf. `spec/MATH_SPEC.md` § 4.1 "Critical Note"), or is this planned future coverage?

3. **Epoch counts**: The hyperboloid test runs for 3000 epochs while others run 60–1000. Is the hyperboloid case known to converge slowly, or could the high epoch count mask a convergence issue in the loss landscape?
