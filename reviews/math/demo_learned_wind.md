# Math Review: `examples/demo_learned_wind.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

The demo implements a Zermelo-inverse problem: given observed wind vectors $W(x)$ on $S^2$, recover $W$ by learning a `NeuralRanders` model from energy evaluations. The overall formulation is **mathematically sound** with one **CRITICAL** issue in the loss function that causes the optimiser to minimise a quantity that does not uniquely identify the ground-truth wind, and one **WARNING** regarding an implicit assumption in the evaluation metrics. Two additional notes are recorded.

---

## Formula-by-Formula Audit

### 1. Finsler Energy Evaluation (`loss_energy`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 (Energy), § 5 (Zermelo / Randers)
- **Implementation:** [demo_learned_wind.py](../examples/demo_learned_wind.py#L55-L56)
  ```python
  energies = jax.vmap(m.energy)(x, v)
  loss_energy = jnp.mean(energies)
  ```
  where `energy` is $E(x,v) = \tfrac{1}{2}F^2(x,v)$ and $F$ is the Randers metric
  $$F(x,v) = \frac{\sqrt{\lambda\,\|v\|_H^2 + \langle W,v\rangle_H^2} - \langle W,v\rangle_H}{\lambda}$$
  with $\lambda = 1 - \|W\|_H^2$.

- **Verdict:** **CRITICAL**

- **Notes:**
  The loss $\mathcal{L}_E = \mathbb{E}_{(x,v)} \bigl[\tfrac{1}{2}F^2(x, v)\bigr]$ is minimised by the learner. However, the data pairs $(x_i, v_i)$ are *fixed samples* where $v_i = W_{\text{true}}(x_i)$. For the Zermelo energy to be a valid reconstruction objective, we need $F(x, W(x)) = 0$ when the learned wind $W$ matches the true wind exactly — but this is **not the case**. The Randers cost $F(x, v)$ is defined for traveller velocities, not for wind vectors. The energy $E(x, W(x))$ evaluated at $v = W(x)$ is:

  $$E(x, W) = \tfrac{1}{2}\left(\frac{\sqrt{\lambda\|W\|_H^2 + \langle W, W\rangle_H^2} - \langle W, W\rangle_H}{\lambda}\right)^2$$

  This equals zero only when $W = 0$. Thus the loss $\mathcal{L}_E$ has a trivial global minimum at $W_{\text{learned}} \equiv 0, H \equiv 0$, not at $W_{\text{learned}} = W_{\text{true}}$. The $H$-regularization term (see item 2) anchors $H \approx I$, which prevents full collapse, but the energy loss still *pushes the wind toward zero* rather than toward the ground truth.

  **What is likely intended** is a *velocity-matching* or *geodesic-direction* loss, for example:
  - Supervised wind loss: $\mathcal{L} = \mathbb{E}\bigl[\|W_{\text{learned}}(x) - v\|^2\bigr]$.
  - Or the *drift-corrected* cost: $F(x, v - W(x)) = \|v - W(x)\|_H$ should be minimised.
  - Or the Randers energy treated as cost-to-traverse the *observed displacement*, where the optimum occurs when $W$ aligns with $v$ and shortens the perceived cost. In this reading the energy loss is a *proxy* objective that couples both $H$ and $W$ — it can still produce reasonable results because the Jacobian regularisation and metric anchor constrain the solution space. However, the loss landscape admits infinitely many local minima and the global minimum does **not** correspond to $W = W_{\text{true}}$.

  **Recommended Action:** Replace or supplement the energy loss with a term that directly penalises the discrepancy between the learned wind vector $W_{\text{learned}}(x)$ and the observed velocity $v$, e.g.
  ```python
  loss_wind = jnp.mean(jnp.sum((W_learned - v)**2, axis=-1))
  ```
  Alternatively, reformulate the energy loss as $E(x, v - W(x))$ under the Riemannian part only, which achieves its minimum when $v = W$.

---

### 2. Metric Regularisation (`loss_h_reg`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo parameterisation: $h_{ij}$ is the Riemannian sea)
- **Implementation:** [demo_learned_wind.py](../examples/demo_learned_wind.py#L58-L60)
  ```python
  H_vals = jax.vmap(m.h_net)(x)
  I = jnp.eye(3)
  loss_h_reg = jnp.mean((H_vals - I)**2)
  ```
- **Verdict:** **NOTE**
- **Notes:**
  The regulariser $\mathcal{L}_H = \mathbb{E}\bigl[\|H(x) - I_3\|_F^2\bigr]$ anchors the Riemannian sea to the Euclidean metric in ambient $\mathbb{R}^3$, which on $S^2$ corresponds to the round metric when restricted to the tangent plane. This is a reasonable prior for learning on the unit sphere and is mathematically well-defined. No issues.

---

### 3. Jacobian Regularisation (`loss_smooth`)

- **Spec Reference:** Not in `MATH_SPEC.md` (regularisation heuristic, not a geometric formula).
- **Literature Reference:** Standard Jacobian penalty, see e.g. Sokolic et al. (arXiv:1706.08500).
- **Implementation:** [demo_learned_wind.py](../examples/demo_learned_wind.py#L62-L70)
  ```python
  def get_w(pt):
      _, W, _ = m._get_zermelo_data(pt)
      return W
  jac_fn = jax.jacfwd(get_w)
  jacobians = jax.vmap(jac_fn)(x)  # Shape (N, 3, 3)
  loss_smooth = jnp.mean(jacobians**2)
  ```
- **Verdict:** **WARNING**
- **Notes:**
  The Jacobian $\partial W^i / \partial x^j$ is computed in ambient $\mathbb{R}^3$ coordinates, not in intrinsic coordinates on $S^2$. For points on $S^2$, the wind $W$ should only have tangential components and the constraint surface has codimension 1. Computing $\mathrm{Jac}_{x} W \in \mathbb{R}^{3 \times 3}$ therefore includes a normal–normal component that does not correspond to any tangential variation. This is not *incorrect* per se (it still penalises rapid tangential variation), but it also penalises the normal projection component $\partial W_\perp / \partial x_\perp$, which is geometrically spurious and may bias the optimiser.

  **Recommended Action:** Consider projecting the Jacobian to the tangent–tangent block: $J_T = P\, J\, P$ where $P = I - n\,n^T$ and $n = x / \|x\|$, or accept the current form as a conservative regulariser and document the trade-off.

---

### 4. Randers `metric_fn` Formula

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Randers metric)
- **Implementation:** [zoo.py](../src/ham/geometry/zoo.py#L120-L130)
  ```python
  discriminant = lam * v_sq_h + W_dot_v**2
  cost = (jnp.sqrt(jnp.maximum(discriminant, 0.0) + 1e-9) - W_dot_v) / lam
  ```
- **Verdict:** **CORRECT**
- **Notes:**
  Expanding the spec formula $F = (\sqrt{\lambda\|v\|_H^2 + \langle W,v\rangle_H^2} - \langle W,v\rangle_H)/\lambda$, the implementation matches exactly with $\lambda = 1 - \|W\|_H^2$ (`lam`), $\|v\|_H^2$ (`v_sq_h`), $\langle W,v\rangle_H$ (`W_dot_v`). The epsilon inside the sqrt is a numerical guard consistent with `spec/MATH_SPEC.md` § 6.1.

---

### 5. Zermelo Causality Constraint (`_get_zermelo_data`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, constraint $\|W\|_h < 1$.
- **Implementation:** [zoo.py](../src/ham/geometry/zoo.py#L86-L99)
  ```python
  squash_factor = (max_speed * jnp.tanh(w_norm)) / (w_norm + 1e-8)
  ```
- **Verdict:** **CORRECT**
- **Notes:**
  The tanh squashing ensures $\|W_{\text{safe}}\|_H = (1-\epsilon)\tanh(\|W_{\text{raw}}\|_H) < 1 - \epsilon$, strictly satisfying the Zermelo causality bound. Well-designed.

---

### 6. Rossby-Haurwitz Stream Function

- **Spec Reference:** Not in `MATH_SPEC.md`.
- **Literature Reference:** Williamson et al. (1992), "A Standard Test Set for Numerical Approximations to the Shallow Water Equations in Spherical Geometry"; Rossby (1939).
- **Implementation:** [fields.py](../src/ham/sim/fields.py#L49-L72)
  ```python
  ψ = -ω sin(lat) + K cos^R(lat) sin(lat) cos(R lon)
  v = ∇ψ × x
  ```
- **Verdict:** **CORRECT**
- **Notes:**
  The stream function $\psi = -\omega\sin\phi + K\cos^R\phi\,\sin\phi\,\cos(R\lambda)$ is a standard Rossby–Haurwitz test case (with minor parameter simplifications: $K=1$ default). The cross product $v = \nabla\psi \times x$ correctly yields a divergence-free, tangential vector field on the unit sphere. The use of complex arithmetic for $\cos(R\lambda)$ via $(\hat{xy})^R$ is a clean differentiable trick.

---

### 7. Cosine Similarity Evaluation

- **Spec Reference:** N/A (evaluation metric).
- **Implementation:** [demo_learned_wind.py](../examples/demo_learned_wind.py#L95-L99)
  ```python
  cos_sim_train = jnp.mean(
      jnp.sum(vecs_true * vecs_pred, axis=-1) /
      (jnp.linalg.norm(vecs_true, axis=-1) * jnp.linalg.norm(vecs_pred, axis=-1) + 1e-8)
  )
  ```
- **Verdict:** **WARNING**
- **Notes:**
  The cosine similarity is computed using Euclidean inner products in ambient $\mathbb{R}^3$. Since both `vecs_true` and `vecs_pred` are tangent to $S^2$ (the `_get_zermelo_data` projects $W$ to the tangent space), the Euclidean inner product is a valid proxy for the tangent-space inner product under the round metric $h = I|_{T_xS^2}$. However, the $H$-regulariser only *approximately* keeps $H \approx I$; if $H$ deviates, the cosine similarity in $\mathbb{R}^3$ no longer faithfully reflects angular alignment in the learned metric. This is acceptable for a demo but should be noted.

  **Recommended Action:** Document that this evaluation metric assumes $H \approx I$ (round sphere).

---

### 8. Wind Scaling Factor

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, $\|W\|_h < 1$.
- **Implementation:** [demo_learned_wind.py](../examples/demo_learned_wind.py#L24)
  ```python
  w_true = lambda x: 0.8 * true_wind_fn(x)
  ```
- **Verdict:** **STRONG**
- **Notes:**
  Scaling the ground-truth wind by 0.8 ensures $\|W_{\text{true}}\|$ stays strictly below 1 in the Euclidean/$H \approx I$ regime, respecting the Zermelo causality bound. Good practice. (The Rossby-Haurwitz field has maximum $\|v\| \approx 1$ on the unit sphere, so $0.8 \times 1 = 0.8 < 1$.)

---

## Open Questions

1. **Identifiability of the inverse problem:** The energy loss alone does not uniquely determine $W$. Has the team verified (e.g. via a supervised wind loss term) that the learned $W$ converges to $W_{\text{true}}$, or is the current pipeline intended as a proof-of-concept where approximate recovery suffices?

2. **Tangent-space fidelity of Jacobian regularisation:** Is the ambient-$\mathbb{R}^3$ Jacobian norm a deliberate design choice (for simplicity/speed) or an oversight? If intrinsic Jacobian control is desired, the projected version should be tested.

3. **Interaction of loss weights:** The weight schedule (1.0 energy, 1.0 H-reg, 0.1 smoothness) may mask the CRITICAL issue above by implicitly balancing the wind-zeroing gradient from $\mathcal{L}_E$ against the H-anchor. Ablation of these weights with a ground-truth wind error metric would clarify.
