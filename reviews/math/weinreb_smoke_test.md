# Math Review: weinreb_smoke_test

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source File:** `examples/weinreb_smoke_test.py`

## Summary

The smoke test is structurally sound and correctly invokes the geometric pipeline from `weinreb_experiment.py`. No formulas are reimplemented here — the file delegates all mathematical work to the library and experiment modules. The numerical assertions are reasonable but two checks are mathematically imprecise, and one test omits a relevant property. Overall verdict: **Minor Issues**.

## Formula-by-Formula Audit

### 1. `two_segment_energy` usage (`check_two_segment_energy`, line 161–167)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 (Energy Functional), § 5 (Zermelo / Randers)
- **Implementation:**
  ```python
  energy = two_segment_energy(vae.metric, z_s, z_m, z_e)
  assert not jnp.isnan(energy)
  assert energy >= 0
  ```
  The called function (`weinreb_experiment.py:255–260`) computes:
  ```python
  metric.energy(z2, z4 - z2) + metric.energy(z4, z6 - z4)
  ```
  where `metric.energy(x, v) = 0.5 * F(x, v)**2` (`src/ham/geometry/metric.py:33`).

- **Verdict:** CORRECT
- **Notes:** The discrete two-segment action $E = \frac{1}{2}F(z_2, z_4 - z_2)^2 + \frac{1}{2}F(z_4, z_6 - z_4)^2$ is the standard discrete-time approximation to the continuous geodesic energy $\int \frac{1}{2}F^2(\gamma, \dot\gamma)\,dt$. The assertion `energy >= 0` is mathematically guaranteed since $F \ge 0$ by definition (§ 1.1).

### 2. `check_encode_mean` (line 141–145)

- **Spec Reference:** N/A (encoder, not a geometric operation)
- **Implementation:**
  ```python
  z = encode_mean(vae, x)
  assert z.shape == (vae.latent_dim,)
  assert not jnp.any(jnp.isnan(z))
  ```
- **Verdict:** CORRECT
- **Notes:** Shape and NaN checks are appropriate for a smoke test of the deterministic encoder mean.

### 3. `check_project_control` (line 147–153)

- **Spec Reference:** Pushforward of tangent vectors via the encoder Jacobian. This is a standard differential-geometric operation: if $\mu: \mathbb{R}^D \to \mathbb{R}^d$ is the encoder mean map, then $v_{\text{lat}} = D\mu(x) \cdot v_{\text{RNA}}$.
- **Implementation:**
  ```python
  z_mean, v_lat = vae.project_control(x, u)
  assert z_mean.shape == (vae.latent_dim,)
  assert v_lat.shape == (vae.latent_dim,)
  assert not jnp.any(jnp.isnan(v_lat))
  ```
  The underlying code (`src/ham/bio/vae.py:113–119`) uses `jax.jvp` followed by `manifold.to_tangent`, which is the correct JVP-based pushforward.
- **Verdict:** CORRECT
- **Notes:** No mathematical issue.

### 4. `check_zermelo_data` (line 155–161)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization)
- **Implementation:**
  ```python
  H, W, lam = vae.metric._get_zermelo_data(z)
  assert H.shape == (vae.latent_dim, vae.latent_dim)
  assert W.shape == (vae.latent_dim,)
  ```
- **Verdict:** WARNING
- **Severity:** WARNING
- **Notes:** The test checks shapes of $H$ and $W$ but does not check the returned $\lambda$ value. More importantly, it does not verify the Zermelo causality constraint $\|W\|_H < 1$ (equivalently $\lambda > 0$), which is the central mathematical invariant of the Randers parameterization (§ 5). The `_get_zermelo_data` method in `src/ham/geometry/zoo.py:79–108` enforces this via the tanh squasher, but a smoke test should verify the output satisfies the constraint, not just trust the squasher.
- **Recommended Action:** Add assertions at `examples/weinreb_smoke_test.py:160`:
  ```python
  assert lam > 0, "Zermelo causality violated: lambda must be > 0"
  w_norm_sq = jnp.dot(W, jnp.dot(H, W))
  assert w_norm_sq < 1.0, "Wind norm must be < 1 in H-metric"
  ```

### 5. `check_two_segment_energy` — `energy >= 0` assertion (line 166)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2, § 5
- **Implementation:**
  ```python
  assert energy >= 0
  ```
- **Verdict:** WARNING
- **Severity:** WARNING
- **Notes:** The assertion `energy >= 0` is necessary but not sufficient. Since the three points are distinct random embeddings, the energy should be strictly positive ($E > 0$). An energy of exactly zero would indicate either (a) $z_s = z_m = z_e$ (degenerate path) or (b) a bug where the metric returns zero. A stronger check `energy > 0` (or `energy > 1e-12`) would be more diagnostic.
- **Recommended Action:** Change assertion at `examples/weinreb_smoke_test.py:166` to:
  ```python
  assert energy > 1e-12, f"Energy is suspiciously close to zero: {energy}"
  ```

### 6. `check_full_validation` — results structure (line 179–194)

- **Spec Reference:** The validation logic in `weinreb_experiment.py:300–390` computes $E_{\text{cf}} / E_{\text{obs}}$ where both energies are computed via `two_segment_energy`.
- **Implementation:**
  ```python
  results, raw = run_validation(...)
  required_keys = ['randers', 'riemannian']
  for k in required_keys:
      if k in results:
          pass
  ```
- **Verdict:** WARNING
- **Severity:** WARNING
- **Notes:** The check iterates over required keys but performs no assertion — the `if k in results: pass` block is a no-op. The smoke test does not verify that the energy ratios $E_{\text{cf}}/E_{\text{obs}}$ are finite and positive, which is the core mathematical output of the validation pipeline. For a smoke test, verifying that the ratios are finite and non-negative is the minimum bar.
- **Recommended Action:** Replace the no-op loop at `examples/weinreb_smoke_test.py:189–191` with:
  ```python
  for k in required_keys:
      assert k in results, f"Missing key '{k}' in validation results"
      assert results[k]['energy_ratio_mean'] > 0, f"Non-positive mean ratio for {k}"
      assert np.isfinite(results[k]['energy_ratio_mean']), f"Non-finite ratio for {k}"
  ```

### 7. `build_smoke_model` — metric construction (line 94–113)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo / Randers)
- **Implementation:**
  ```python
  metric = PullbackRanders(
      manifold, decoder=decoder_net, key=k1, hidden_dim=16, depth=2
  )
  ```
- **Verdict:** CORRECT
- **Notes:** The initial model uses `PullbackRanders` with a learnable wind field. After Phase 1 training, the metric is replaced by `DataDrivenPullbackRanders` via `attach_datadriven_randers_metric`. The `DataDrivenPullbackRanders` class (`src/ham/models/learned.py:98–120`) correctly composes `PullbackGNet` ($H = J^T J + \epsilon I$, which is the pullback metric) with `KernelWindField` (Nadaraya-Watson smoother). The mathematical construction is sound: $h_{ij}(z) = \sum_\alpha \frac{\partial f^\alpha}{\partial z^i} \frac{\partial f^\alpha}{\partial z^j}$ where $f$ is the decoder.

### 8. `smoke_train` — Phase 1 losses (line 115–130)

- **Spec Reference:** Standard VAE loss: $\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{KL}(q \| p)$.
- **Implementation:**
  ```python
  losses=[
      ReconstructionLoss(weight=1.0),
      KLDivergenceLoss(weight=1e-4),
  ]
  ```
- **Verdict:** CORRECT
- **Notes:** No geometric formula is involved in Phase 1 — it is a standard $\beta$-VAE objective. The low KL weight ($10^{-4}$) is a common choice for avoiding posterior collapse.

## Open Questions

1. **Degenerate velocity vectors**: `check_project_control` tests with `dataset.V[0]` which is constructed to be non-zero by design (line 63: `V[:half, 0] = 0.5 + noise`). However, no test exercises the $v = 0$ edge case, which is the singularity discussed in `spec/MATH_SPEC.md` § 6.1. Should the smoke test include a zero-velocity check to verify the $\epsilon$-regularization path?

2. **Riemannian baseline symmetry**: The validation comment in `weinreb_experiment.py:247` states that the Riemannian ratio should be $\approx 1$ because the metric is symmetric. The smoke test does not verify this expected property of the Riemannian baseline (i.e., that `results['riemannian']['energy_ratio_mean']` is close to 1.0). This could serve as a mathematical sanity check.
