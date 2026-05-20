# Math Review: train_geodesic

**Reviewer:** Math Reviewer Agent
**Date:** 2025-05-15
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

**Minor Issues.** The file implements Phase 2 of a two-phase metric learning pipeline: given pre-encoded lineage pairs $(z_p, z_c)$, it solves the boundary-value geodesic problem via AVBD and trains the Randers metric parameters by minimising the geodesic action plus a metric regulariser. The core mathematical framework—discrete variational geodesic regression with an anchor regularisation—is sound. No **CRITICAL** errors were found. Two **WARNING**-level issues affect the numerical fidelity of the energy functional and a stochastic encoding concern. Two **NOTE**-level observations concern regularisation conventions.

---

## Formula-by-Formula Audit

### 1. Discrete Action (Energy) Loss
- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 (Energy Functional), § 2.1 (Euler-Lagrange / Spray derivation).
- **Literature Reference:** Discrete geodesic calculus: Rumpf & Wirth, *Variational time discretization of geodesic calculus*, IMA J. Numer. Anal. 35(3), 2015.
- **Implementation:** `src/ham/bio/train_geodesic.py:19–28`.
  ```python
  solve_fn = jax.vmap(lambda s, e: self.solver.solve(m.metric, s, e, n_steps=8, train_mode=True))
  trajectories = solve_fn(z_parent, z_child)
  action_loss = jnp.mean(trajectories.energy)
  ```
  The AVBD solver (`src/ham/solvers/avbd.py:173–175`) computes:
  ```python
  vels = jax.vmap(metric.manifold.log_map)(full_new[:-1], full_new[1:])
  total_E = jnp.sum(jax.vmap(metric.energy)(full_new[:-1], vels))
  ```
  This produces $\displaystyle \sum_{i=0}^{N-1} E(x_i,\, v_i)$ where $v_i = \log_{x_i}(x_{i+1})$ and $E(x,v) = \tfrac{1}{2}F^2(x,v)$.
- **Verdict:** WARNING
- **Notes:** The sum $\sum_i E(x_i, v_i)$ is not the standard discrete approximation to the continuous action $\mathcal{E}[\gamma] = \int_0^1 E(\gamma, \dot\gamma)\, dt$. With $N$ segments of parameter length $\Delta t = 1/N$ and actual velocity $\dot\gamma \approx v_i / \Delta t$, the proper quadrature is:
  $$\mathcal{E} \approx \sum_i E\!\bigl(x_i,\, v_i/\Delta t\bigr)\,\Delta t = \frac{1}{2\Delta t}\sum_i F^2(x_i, v_i)$$
  using 1-homogeneity of $F$. The implemented sum therefore equals $\Delta t \cdot \mathcal{E}$, i.e. it is proportional to the true discrete action by the factor $1/(N-1)$. Since the geodesic BVP minimiser (AVBD) uses the same un-normalised sum in its own energy gradient (`local_action` at `avbd.py:106–110`), the minimising path is still the correct discrete geodesic. However, the *magnitude* of `action_loss` scales linearly with `n_steps`, which means the effective balance between `action_loss` and `reg_loss` silently changes when `n_steps` is varied. At the current hardcoded `n_steps=8`, the ratio is fixed and the system works. If `n_steps` were ever changed, the regularisation weight `1.0` would need manual re-tuning.

  **Recommended Action:** Normalise by dividing the total energy by `n_steps` (or equivalently multiply by $\Delta t$) so that loss weighting is independent of discretisation resolution.

---

### 2. Metric Anchor Regularisation
- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization).
- **Implementation:** `src/ham/bio/train_geodesic.py:32–39`.
  ```python
  def regularize_metric(x):
      M, W, _ = m.metric._get_zermelo_data(x)
      dim = M.shape[-1]
      return jnp.mean((M - jnp.eye(dim))**2) + 0.1 * jnp.mean(W**2)
  ```
- **Verdict:** NOTE
- **Notes:**
  1. The $H$-anchor term $\frac{1}{d^2}\|H(x) - I\|_F^2$ is the standard Frobenius-norm penalty on the Riemannian component of the Zermelo parameterisation. This prevents the degenerate solution $H \to 0$ that would trivially minimise the action. Correct.
  2. The wind penalty uses the Euclidean norm $\|W\|_2^2 = \sum_i W_i^2$, whereas the natural norm in the Zermelo formulation is $\|W\|_H^2 = W^T H W$ (spec § 5). Since $H$ is simultaneously anchored to $I$, the two norms are close when the anchor is effective. This is a minor inconsistency rather than an error. Using $W^T H W$ would be more geometrically principled but is not required for correctness.

---

### 3. Trajectory Sampling for Regularisation
- **Spec Reference:** N/A (implementation choice).
- **Implementation:** `src/ham/bio/train_geodesic.py:38–39`.
  ```python
  sample_pts = trajectories.xs[:, ::2, :]
  reg_loss = jnp.mean(jax.vmap(jax.vmap(regularize_metric))(sample_pts))
  ```
- **Verdict:** OK
- **Notes:** Evaluating the anchor loss at points along the solved geodesic (subsampled at every 2nd node) is a valid spatial sampling strategy. The regularisation is purely pointwise in $x$ (it only queries $H(x)$ and $W(x)$) and does not depend on the trajectory geometry, so the coupling with the solver output is benign.

---

### 4. Directionality of Log Maps (Finsler Consistency)
- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (Geodesic ODE), § 5 (Randers / Zermelo).
- **Literature Reference:** Bao, Chern, Shen, *An Introduction to Riemann-Finsler Geometry*, Springer GTM 200, Ch. 11.
- **Implementation:** Inherited from `src/ham/solvers/avbd.py:106–110`:
  ```python
  v_in  = metric.manifold.log_map(x_prev, x)      # log_{x_prev}(x)
  v_out = metric.manifold.log_map(x, x_next)        # log_x(x_{next})
  ```
- **Verdict:** OK
- **Notes:** For a Randers metric ($F(x,v) \neq F(x,-v)$), the direction of the log map determines the cost. Here, both velocities point "forward" along the path direction, which is the correct convention for minimising the Randers action in the causal (parent $\to$ child) direction. The sum $E(x_{prev}, v_{in}) + E(x, v_{out})$ correctly captures the two-segment energy at vertex $x$.

---

### 5. Differentiable Solver Unrolling
- **Spec Reference:** `spec/MATH_SPEC.md` § 2.2 (JAX Implementation—Implicit Solve).
- **Implementation:** `src/ham/bio/train_geodesic.py:19–21` invokes AVBD with `train_mode=True`; `src/ham/solvers/avbd.py:161–163`:
  ```python
  if train_mode:
      final_state, _ = jax.lax.scan(step_fn, state, None, length=self.iterations)
  ```
- **Verdict:** WARNING
- **Notes:** The AVBD solver is unrolled for a fixed number of iterations (default `iterations=15`, overridden to `15` in the trainer constructor) with `n_steps=8` inner path points. The outer loss gradient $\nabla_\theta \mathcal{L}$ is obtained by differentiating through the entire unrolled solve via `eqx.filter_value_and_grad`. This is mathematically valid (it computes the exact gradient of the truncated optimisation) but the gradient quality degrades if the inner solver has not converged. With only 15 block-descent iterations, each with `step_size=0.1`, convergence is not guaranteed—especially for non-convex Randers energies.

  If the inner solve is far from the true geodesic, the outer gradient $\nabla_\theta \mathcal{L}$ reflects sensitivity of a *suboptimal* path to the metric, not sensitivity of the *geodesic*. By the envelope theorem, at convergence these coincide; away from convergence, they can diverge.

  **Recommended Action:** Validate empirically that the AVBD solver reaches a low gradient-norm at the given iteration budget. Alternatively, consider implicit differentiation (solving the linear system at convergence) instead of unrolling, which would provide exact geodesic gradients independent of the iteration count.

---

### 6. Stochastic Encoding with Fixed PRNG Key
- **Spec Reference:** N/A (statistical, not geometric).
- **Implementation:** `src/ham/bio/train_geodesic.py:57–58`:
  ```python
  z_all = jax.vmap(lambda x: self.model.encode(x, jax.random.PRNGKey(0)))(X)
  z_all = jax.lax.stop_gradient(z_all)
  ```
- **Verdict:** WARNING
- **Notes:** All $N$ data points are encoded with the identical PRNG key `PRNGKey(0)`. If `model.encode` performs the reparameterisation trick ($z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0,I)$), every sample gets the same noise vector $\epsilon$. This introduces a systematic bias: high-variance latent dimensions are all shifted in the same direction. The `stop_gradient` means this does not affect optimisation gradients for the metric, but the latent point cloud $\{z_i\}$ on which geodesics are regressed is a biased representation of the posterior.

  **Recommended Action:** Use `jax.vmap` with split keys: `keys = jax.random.split(jax.random.PRNGKey(0), len(X))` and encode as `jax.vmap(model.encode)(X, keys)`. Alternatively, if deterministic encoding is intended, use only the posterior mean (no sampling).

---

### 7. Overall Variational Framework
- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 (Energy Functional), § 2.1 (Euler-Lagrange equations), § 5 (Zermelo Parameterization).
- **Literature Reference:** Arvanitidis, Hansen, Hauberg, *Latent Space Oddity: on the Curvature of Deep Generative Models*, ICLR 2018.
- **Implementation:** The full `loss_fn` at `src/ham/bio/train_geodesic.py:17–40`:
  $$\mathcal{L}(\theta) = \underbrace{\frac{1}{B}\sum_{b=1}^{B} \sum_{i=0}^{N-1} E_\theta\!\bigl(x_i^{(b)},\, v_i^{(b)}\bigr)}_{\text{action loss}} + \underbrace{\frac{1}{|\mathcal{S}|}\sum_{x \in \mathcal{S}} \bigl[\|H_\theta(x) - I\|_F^2 + 0.1\|W_\theta(x)\|_2^2\bigr]}_{\text{anchor regularisation}}$$
  where $x_i^{(b)}$ are the AVBD-solved path nodes and $\mathcal{S}$ is the subsampled trajectory point set.
- **Verdict:** OK
- **Notes:** The competition between action minimisation (which incentivises $H \to 0$) and anchor regularisation (which penalises $H \neq I$) is the standard tension in metric learning. The balance is well-posed: there exists a non-trivial equilibrium where the metric captures the anisotropy of the lineage transitions while remaining bounded. The formulation is consistent with the variational principle in the spec.

---

## Open Questions

1. **Convergence of the inner AVBD solver:** Has it been empirically verified that 15 iterations with `step_size=0.1` produce near-converged geodesics for the typical Randers metrics encountered during training? If not, the outer gradients may be unreliable (see § 5 above).

2. **Deterministic vs. stochastic encoding intent:** Is `model.encode(x, key)` intended to return the posterior mean or a reparameterised sample? If deterministic, the fixed-key issue (§ 6) is moot; if stochastic, it should be fixed.

3. **Sensitivity of loss balance to `n_steps`:** The action loss magnitude scales with `n_steps` (§ 1). Has the regularisation weight `1.0` been tuned specifically for `n_steps=8`, and is this documented?
