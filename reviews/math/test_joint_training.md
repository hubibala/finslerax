# Math Review: test_joint_training (tests/test_joint_training.py)

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Minor Issues.** The file tests the modular `HAMPipeline` by exercising two-phase training (Phase 1: VAE manifold, Phase 2: metric/wind). The tests are structurally sound smoke tests—they verify that the pipeline runs and that parameters move, but they do not verify any mathematical invariant (loss convergence, metric positive-definiteness, alignment sign, geodesic equation satisfaction). The `MockMetric` provides a flat Euclidean geometry ($g_{ij} = \delta_{ij}$, $G^i = 0$) with a constant wind $W^i = 0.1$, which is geometrically consistent but violates the Zermelo interface contract by returning an identity inverse metric even though it is not a proper Randers metric. Two **WARNING**-level issues and three **NOTE**-level observations are detailed below.

---

## Formula-by-Formula Audit

### 1. `MockMetric.inner_product` — Euclidean Inner Product (line 47)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1, § 4 (Euclidean row of Geometric Hierarchy table).
- **Implementation:**
  ```python
  def inner_product(self, x, u, v=None, w=None):
      if v is None: v = u
      return jnp.sum(u * v, axis=-1)
  ```
- **Verdict:** NOTE
- **Notes:** Implements $\langle u, v \rangle = \sum_i u^i v^i$, i.e., the flat Euclidean inner product. This is mathematically correct as a Riemannian inner product with $g_{ij} = \delta_{ij}$. However, the `w` parameter is silently ignored; calling code that passes four positional arguments (e.g., `inner_product(x, v, spray, spray)` in `GeodesicSprayLoss`) will compute `jnp.sum(v * spray)` rather than `jnp.sum(spray * spray)`. Since the spray is zero (see below), the result is 0.0 either way and is numerically harmless in this test, but it would be wrong for any non-trivial metric.

---

### 2. `MockMetric.spray` — Zero Spray (line 51)

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1, § 4 — Euclidean spray: $G^i = 0$.
- **Implementation:**
  ```python
  def spray(self, x, v):
      return jnp.zeros_like(v)
  ```
- **Verdict:** CORRECT
- **Notes:** For Euclidean space the spray coefficients vanish identically ($G^i = 0$), yielding straight-line geodesics $\ddot{x}^i = 0$. This is the correct Euclidean limit per the Geometric Hierarchy table.

---

### 3. `MockMetric._get_zermelo_data` — Constant Wind on Flat Space (line 54)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization).
- **Implementation:**
  ```python
  def _get_zermelo_data(self, x):
      dim = x.shape[-1]
      W = jnp.ones(dim) * 0.1
      return jnp.eye(dim), W, jnp.eye(dim)
  ```
- **Verdict:** WARNING
- **Notes:** Returns $(H, W, \Lambda)$ where $H = I_d$, $W = 0.1 \cdot \mathbf{1}$, and $\Lambda = I_d$. Per the spec, $\lambda = 1 - \|W\|_H^2$. For $d$-dimensional identity metric and $W = 0.1 \cdot \mathbf{1}$, we have $\|W\|_H^2 = 0.01 d$. With $d = \text{latent\_dim} = 2$, $\lambda = 1 - 0.02 = 0.98$, which is a scalar. Returning $\Lambda = I_d$ (a matrix) instead of $\lambda = 0.98$ (a scalar) is inconsistent with the spec's definition and with the `Randers._get_zermelo_data` implementation (which returns a scalar $\lambda$). The downstream losses that consume the third return value (e.g., `ZermeloAlignmentLoss`) never use $\Lambda$ in these tests, so this mismatch is dormant. However, any test that exercises $\lambda$-dependent formulas (e.g., the full Randers metric $F = (\sqrt{\lambda \|v\|_h^2 + \langle W,v\rangle_h^2} - \langle W,v\rangle_h)/\lambda$) against this mock would get incorrect results.

  **Recommended Action:** Return the correct scalar $\lambda = 1 - \|W\|_H^2$ as the third element of the tuple to match the `Randers` class interface and `spec/MATH_SPEC.md` § 5.

---

### 4. `test_phase1_manifold` — Phase 1 Assertion (lines 64–82)

- **Spec Reference:** N/A (test infrastructure, not a formula).
- **Implementation:**
  ```python
  old_dec = self.vae.decoder_net.layers[0].weight.copy()
  # ... training ...
  new_dec = trained.decoder_net.layers[0].weight
  self.assertFalse(jnp.allclose(old_dec, new_dec), "Decoder weights should change")
  ```
- **Verdict:** NOTE
- **Notes:** The test asserts only that decoder weights changed, not that the reconstruction loss decreased. This is a minimal liveness check, not a mathematical correctness check. It does not verify that $\mathcal{L}_{\text{recon}} = \mathbb{E}[\|x - \hat{x}\|^2]$ is actually minimized. For a 2-epoch run with 50 samples this is acceptable as a smoke test, but it provides no confidence in mathematical convergence properties.

---

### 5. `test_phase2_metric` — Phase 2: Contrastive + Anchor Losses (lines 84–100)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo); contrastive alignment from `losses.py`.
- **Implementation:**
  ```python
  losses=[
      ContrastiveAlignmentLoss(weight=1.0),
      MetricAnchorLoss(weight=1.0),
  ],
  ```
- **Verdict:** WARNING
- **Notes:** The `ContrastiveAlignmentLoss` computes $\mathcal{L} = -\langle W, \log_{z_p}(z_c)\rangle_L$, using the Minkowski inner product on the Hyperboloid. For the `MockMetric`, however, the wind vector $W = 0.1 \cdot \mathbf{1}$ is a constant Euclidean vector, not projected onto the tangent space of the Hyperboloid at $z_p$. The `log_map` output is a tangent vector in the Lorentzian tangent space $T_{z_p}\mathbb{H}^n$, while $W$ lives in $\mathbb{R}^{d+1}$. The Minkowski dot product $\langle W, v_{\tan}\rangle_L = -W^0 v^0 + \sum_{i=1}^d W^i v^i$ applied to an un-projected $W$ is geometrically meaningless—it mixes ambient-space and tangent-space quantities.

  With only 2 training epochs and an `assertIsInstance` check, this never surfaces as a numerical failure, but any test that verifies the sign or magnitude of the alignment loss would be invalid.

  **Recommended Action:** Either (a) project $W$ onto $T_z\mathbb{H}^n$ inside `_get_zermelo_data` using `manifold.to_tangent`, or (b) note explicitly in the test that the mock is only valid for flat (non-Hyperboloid) manifolds.

---

### 6. `test_phase2_metric` — MetricAnchorLoss (line 91)

- **Spec Reference:** Not in `MATH_SPEC.md`; regularization term.
- **Implementation (from `losses.py` lines 140–155):**
  ```python
  H_out, _, _ = model.metric._get_zermelo_data(parent_z)
  dim = H_out.shape[-1]
  I = jnp.eye(dim)
  return jnp.mean((H_out - I)**2) * self.weight
  ```
- **Verdict:** CORRECT
- **Notes:** The anchor loss penalizes $\|H(z) - I\|_F^2$, preventing the metric tensor from degenerating. For the `MockMetric` which returns $H = I$ identically, this loss is exactly zero at all times. This is self-consistent: the mock metric is already at the anchor target. Equal weighting (`weight=1.0`) alongside the contrastive loss is reasonable for a smoke test.

---

### 7. `test_full_pipeline` — Two-Phase Sequential Run (lines 102–126)

- **Spec Reference:** N/A.
- **Implementation:**
  ```python
  phase1 = TrainingPhase(
      name="Manifold",
      epochs=2,
      losses=[ReconstructionLoss(weight=1.0)],
      ...
  )
  phase2 = TrainingPhase(
      name="Metric",
      epochs=2,
      losses=[ContrastiveAlignmentLoss(weight=1.0)],
      ...
  )
  trained = pipeline.fit(self.dataset, phases=[phase1, phase2], batch_size=5, seed=123)
  self.assertIsInstance(trained, GeometricVAE)
  ```
- **Verdict:** NOTE
- **Notes:** Phase 1 uses only `ReconstructionLoss` without `KLDivergenceLoss`, unlike `test_phase1_manifold`. This means the latent space has no regularization pressure during full-pipeline Phase 1, so posterior samples can drift arbitrarily far from the prior. Phase 2 then trains the metric on these unregularized latent points. For a 2-epoch smoke test this is acceptable, but for any convergence test this would introduce a confound: the metric phase operates on a potentially degenerate latent space. The sole assertion (`assertIsInstance`) checks only that the return type is correct—no mathematical property is validated.

---

### 8. `setUp` — Dataset Construction (lines 61–73)

- **Spec Reference:** N/A (data generation).
- **Implementation:**
  ```python
  X = jax.random.normal(self.key, (self.N, self.data_dim))
  V = jax.random.normal(self.key, (self.N, self.data_dim))
  ```
- **Verdict:** NOTE
- **Notes:** The same PRNG key is used for both $X$ and $V$, meaning `X == V` exactly. This means the RNA velocity vectors are identical to the expression vectors—a degenerate scenario that would never occur in real biological data. While not a mathematical error per se, it means the tests cannot distinguish whether the pipeline correctly handles the $x \to v$ mapping. For mathematical testing of alignment losses (which rely on the relationship between $x$ and $v$), using `jax.random.split(self.key)` to generate independent $X$ and $V$ would be more informative.

---

## Open Questions

1. **Test coverage of mathematical invariants:** None of the three tests verify any mathematical property (loss monotonicity, metric PD-ness, alignment sign, geodesic equation satisfaction). Should additional assertions be added, e.g., `loss_final < loss_initial` after 10+ epochs?

2. **MockMetric tangent-space consistency on Hyperboloid:** The `MockMetric` implements a flat Euclidean geometry but is paired with a `Hyperboloid` manifold. The Hyperboloid's `log_map` and `_minkowski_dot` expect vectors in Lorentzian signature, while the mock metric's wind vector $W = 0.1 \cdot \mathbf{1}$ is not tangent to $\mathbb{H}^n$. Is this intentional (testing robustness to mismatched signatures) or an oversight?

3. **$\Lambda$ return value contract:** The third element returned by `_get_zermelo_data` is $\Lambda = I_d$ in the mock but $\lambda \in \mathbb{R}$ (scalar) in the production `Randers` class. Which is the canonical interface? This should be settled and documented.
