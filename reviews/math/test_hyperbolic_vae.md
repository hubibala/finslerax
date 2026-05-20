# Math Review: test_hyperbolic_vae

**Reviewer:** Math Reviewer Agent  
**Date:** 2026-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Minor Issues.** The test file is primarily a collection of gradient-propagation and finiteness smoke tests. The mathematical constants and tangent vectors used in the tests are correct. However, several tests miss opportunities to verify core geometric invariants (norm preservation, tangency, round-trip identities), and the KL divergence formula tested via `test_vae_forward_pass` is mathematically incomplete in the source—a fact the test cannot detect because it only asserts finiteness.

---

## Formula-by-Formula Audit

### 1. Hyperboloid origin and tangent vector (`test_exp_map_gradients`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4 (Table), § 4.1
- **Literature Reference:** Ratcliffe, *Foundations of Hyperbolic Manifolds*, Ch. 3
- **Implementation:** `tests/test_hyperbolic_vae.py:50–51`
  ```python
  x = jnp.array([1.0, 0.0, 0.0])  # Origin
  v = jnp.array([0.0, 0.5, 0.5])  # Tangent vector
  ```
- **Verdict:** CORRECT
- **Notes:** The origin satisfies $\langle x, x \rangle_L = -1^2 + 0 + 0 = -1$ (on $\mathbb{H}^2$). The tangent vector satisfies $\langle x, v \rangle_L = -1 \cdot 0 + 0 \cdot 0.5 + 0 \cdot 0.5 = 0$ (tangent to $x$). Both are analytically correct.

### 2. Exponential map gradient test (`test_exp_map_gradients`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1
- **Literature Reference:** $\exp_x(v) = \cosh(\|v\|_L)\, x + \frac{\sinh(\|v\|_L)}{\|v\|_L}\, v$; see Robbin & Salamon, *Introduction to Differential Geometry* (2022)
- **Implementation:** `tests/test_hyperbolic_vae.py:53–60`
  ```python
  def loss_fn(v_in):
      z = self.manifold.exp_map(x, v_in)
      return jnp.sum(z**2)
  grads = grad_fn(v)
  self.assertTrue(jnp.all(jnp.isfinite(grads)))
  self.assertFalse(jnp.all(grads == 0))
  ```
- **Verdict:** WARNING
- **Notes:** The test verifies gradient propagation but does **not** assert the defining mathematical property: that $\exp_x(v)$ lies on $\mathbb{H}^2$, i.e., $\langle \exp_x(v), \exp_x(v) \rangle_L = -1$. The source implementation (`src/ham/geometry/surfaces.py:349–357`) is correct, but the test would pass even if the exponential map formula were numerically wrong. **Recommended Action:** Add an assertion `self.assertAlmostEqual(self.manifold._minkowski_dot(z, z), -1.0, places=10)` for the computed point.

### 3. Parallel transport gradient test (`test_parallel_transport_gradients`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 3.2
- **Literature Reference:** Nickel & Kiela, "Learning Continuous Hierarchies in the Lorentz Model" (ICML 2018). Closed-form: $P_{x \to y}(v) = v + \frac{\langle y, v \rangle_L}{1 - \langle x, y \rangle_L}(x + y)$.
- **Implementation:** `tests/test_hyperbolic_vae.py:62–76`
- **Verdict:** WARNING
- **Notes:** The test only asserts gradient finiteness. It does **not** verify the two defining properties of parallel transport on $\mathbb{H}^n$:
  1. **Tangency:** $\langle y, P_{x \to y}(v) \rangle_L = 0$
  2. **Norm preservation:** $\|P_{x \to y}(v)\|_L = \|v\|_L$

  These are the properties that distinguish a correct parallel transport from an arbitrary linear map. The source implementation (`src/ham/geometry/surfaces.py:368–374`) is analytically correct (verified by substitution), but the test would pass with a broken implementation.

  **Recommended Action:** Add assertions for tangency and norm preservation with tolerance `places=10` (64-bit precision is enabled).

### 4. Projection of `y_raw` onto hyperboloid (`test_parallel_transport_gradients`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1
- **Implementation:** `tests/test_hyperbolic_vae.py:67`
  ```python
  y_raw = jnp.array([2.0, 1.0, 0.0])
  y = self.manifold.project(y_raw)
  ```
- **Verdict:** CORRECT
- **Notes:** For `y_raw = [2, 1, 0]`: $\langle y_{\text{raw}}, y_{\text{raw}} \rangle_L = -4 + 1 = -3 < 0$ and $y_0 = 2 > 0$, so `project` uses the scaling branch, yielding $y = y_{\text{raw}} / \sqrt{3}$. Check: $-(2/\sqrt{3})^2 + (1/\sqrt{3})^2 = -4/3 + 1/3 = -1$. ✓

### 5. Wrapped normal manifold constraint (`test_wrapped_normal_sampling`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1
- **Literature Reference:** Nagano et al., "A Wrapped Normal Distribution on Hyperbolic Space for Gradient-Based Learning" (AISTATS 2019)
- **Implementation:** `tests/test_hyperbolic_vae.py:78–93`
  ```python
  norm_sq = self.manifold._minkowski_dot(z[i], z[i])
  self.assertAlmostEqual(norm_sq, -1.0, places=5)
  self.assertGreater(z[i, 0], 0.0)
  ```
- **Verdict:** CORRECT
- **Notes:** The two assertions correctly verify the defining constraints of the upper-sheet hyperboloid: $\langle z, z \rangle_L = -1$ and $z_0 > 0$. The tolerance `places=5` ($10^{-5}$) is appropriate—conservative enough to catch formula errors while allowing for floating-point accumulation through `exp_map` and `parallel_transport`. With 64-bit precision, actual errors should be $\sim 10^{-14}$.

### 6. Wrapped normal sampling procedure (source cross-reference)

- **Spec Reference:** Not in `spec/MATH_SPEC.md` (VAE-specific)
- **Literature Reference:** Nagano et al. (2019), Algorithm 1
- **Implementation:** `src/ham/bio/vae.py:27–40`
  ```python
  origin = origin.at[..., 0].set(1.0)
  v_origin = jnp.concatenate([jnp.zeros(shape + (1,)), v_flat], axis=-1)
  v_at_mean = self.manifold.parallel_transport(origin, self.mean, v_origin)
  z = self.manifold.exp_map(self.mean, v_at_mean)
  ```
- **Verdict:** CORRECT
- **Notes:** The sampling procedure follows the standard recipe: (1) sample in the tangent space at the origin $o = [1, 0, \ldots, 0]$; (2) embed as $v_o = [0, v_{\text{flat}}]$ which is tangent to $o$ since $\langle o, v_o \rangle_L = -1 \cdot 0 = 0$; (3) parallel transport to the mean $\mu$; (4) apply $\exp_\mu$. This matches Nagano et al. (2019).

### 7. KL divergence formula (source cross-reference, tested via `test_vae_forward_pass`)

- **Spec Reference:** Not in `spec/MATH_SPEC.md`
- **Literature Reference:** Nagano et al. (2019), Theorem 1; Mathieu et al. (2019), "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders"
- **Implementation:** `src/ham/bio/vae.py:42–44`
  ```python
  kl = -jnp.log(self.scale + 1e-6) + (self.scale**2) / 2.0 - 0.5
  return jnp.sum(kl, axis=-1)
  ```
- **Verdict:** CRITICAL
- **Notes:** The formula computes:
  $$KL = \sum_{i=1}^{d} \left(-\log \sigma_i + \frac{\sigma_i^2}{2} - \frac{1}{2}\right)$$
  This is the KL divergence $KL\bigl(\mathcal{N}(0, \text{diag}(\sigma^2)) \;\|\; \mathcal{N}(0, I)\bigr)$ in **flat** Euclidean space, and it is missing two terms required for the wrapped normal on $\mathbb{H}^n$:
  1. **Geodesic distance term:** $\frac{1}{2}d_{\mathbb{H}}(\mu, o)^2$ penalizing the posterior mean's distance from the prior origin.
  2. **Log-determinant Jacobian correction:** $(n-1)\log\frac{\sinh r}{r}$ arising from the volume element distortion of the exponential map (Nagano et al. 2019, Theorem 1).

  Without these terms, the KL penalty does not regularize the posterior mean toward the origin, and the posterior entropy estimate is biased.

  The test `test_vae_forward_pass` (`tests/test_hyperbolic_vae.py:112–126`) only checks `jnp.isfinite(k)` and therefore **cannot detect this formula error**.

  **Recommended Action:**
  - Correct the KL formula in `src/ham/bio/vae.py:42–44` to include $d(\mu, o)^2/2$ and the log-det Jacobian term.
  - Add a test that verifies $KL = 0$ when $\mu = o$ and $\sigma = 1$ (the prior equals the posterior).
  - Add a test that verifies $KL > 0$ for $\mu \neq o$ or $\sigma \neq 1$.

### 8. VAE forward-pass smoke test (`test_vae_forward_pass`)

- **Spec Reference:** N/A
- **Implementation:** `tests/test_hyperbolic_vae.py:95–126`
- **Verdict:** WARNING
- **Notes:** The test checks that `ReconstructionLoss`, `KLDivergenceLoss`, and `ZermeloAlignmentLoss` all return finite scalars. This is necessary but insufficient: it would pass even if all three losses returned the constant $0.0$. No mathematical property of any loss is verified. **Recommended Action:** At minimum, check that the reconstruction loss is positive (MSE $\geq 0$) and the KL loss is non-negative ($KL \geq 0$ by Gibbs' inequality).

### 9. VAE gradient test (`test_vae_gradients`)

- **Spec Reference:** N/A
- **Implementation:** `tests/test_hyperbolic_vae.py:128–150`
- **Verdict:** NOTE
- **Notes:** The gradient finiteness check is a valid differentiability smoke test. It confirms that the composition of encoder → wrapped normal → exp_map → decoder is end-to-end differentiable through JAX. No mathematical concern, but the test does not check for vanishing gradients, only infinite/NaN ones.

### 10. Missing round-trip identity tests

- **Spec Reference:** `spec/MATH_SPEC.md` § 4.1
- **Literature Reference:** Standard property of Riemannian exp/log maps
- **Verdict:** WARNING
- **Notes:** No test verifies the fundamental identities:
  - $\exp_x(\log_x(y)) = y$ for $y \in \mathbb{H}^n$
  - $\log_x(\exp_x(v)) = v$ for $v \in T_x\mathbb{H}^n$
  - $d_{\mathbb{H}}(x, \exp_x(v)) = \|v\|_L$

  These are the primary correctness criteria for the exponential and logarithmic maps. The source implementations (`src/ham/geometry/surfaces.py:349–367`) are analytically correct, but no test guards against regression.

  **Recommended Action:** Add a test that picks several $(x, v)$ pairs and asserts the round-trip identities hold to $\sim 10^{-12}$ tolerance (64-bit).

### 11. MockMetric uses `_minkowski_norm` via `ZermeloAlignmentLoss`

- **Spec Reference:** `spec/MATH_SPEC.md` § 5
- **Implementation:** `src/ham/training/losses.py:55–59`
  ```python
  norm_w = model.manifold._minkowski_norm(W)
  norm_v = model.manifold._minkowski_norm(u_lat)
  ```
- **Verdict:** NOTE
- **Notes:** The `ZermeloAlignmentLoss` normalizes tangent vectors using `_minkowski_norm` (the Lorentzian norm for spacelike vectors). This is correct for vectors tangent to $\mathbb{H}^n$, which are spacelike ($\langle v, v \rangle_L > 0$). The `MockMetric._get_zermelo_data` returns $W = 0$, so the test exercises a degenerate case only (alignment loss $= 0$).

---

## Open Questions

1. **Is the missing $d(\mu, o)^2$ term in the KL divergence intentional?** Some hyperbolic VAE implementations use the flat-space approximation for simplicity, accepting bias when the mean is far from the origin. If intentional, this should be documented with a comment citing the approximation and its regime of validity. If not, it is a formula error (see Finding 7).

2. **Should the test suite include a dedicated distance-function test?** The geodesic distance $d_{\mathbb{H}}(x, y) = \text{arccosh}(-\langle x, y \rangle_L)$ is never directly tested, yet it is implicitly used by the log map via $d = \text{arcsinh}(\|u\|_L)$ (which is equivalent). An explicit distance test would provide independent verification.

3. **The `test_wrapped_normal_sampling` test uses a loop over 10 samples.** This is adequate for a unit test but provides no statistical coverage of the distribution's shape. A follow-up integration test could verify that the empirical Fréchet mean of a large sample set converges to `mean` and the empirical variance converges to `scale^2`.
