# Math Review: demo_zermelo

**Reviewer:** Math Reviewer Agent
**Date:** 2025-05-15
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

The Zermelo navigation problem and Randers metric construction are **mathematically correct**. The Randers formula in `src/ham/geometry/zoo.py` exactly matches `spec/MATH_SPEC.md` § 5 and the literature (Bao–Robles–Shen 2004). The wind field is a valid tangent vector field on $S^2$ satisfying the causality constraint $\|W\|_h < 1$. Two issues require attention: (1) the `Sphere` constructor is called with a wrong positional argument, producing incorrect manifold metadata; (2) several inline comments contradict the code constants.

## Formula-by-Formula Audit

### 1. Sphere Construction

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, Table row "Randers"
- **Implementation:** `examples/demo_zermelo.py:13`
  ```python
  sphere_cont = Sphere(radius)
  ```
  The `Sphere.__init__` signature is `(self, intrinsic_dim: int = 2, radius: float = 1.0)` ([surfaces.py](src/ham/geometry/surfaces.py#L27)). Passing `radius = 1.0` as the first positional argument sets `intrinsic_dim = 1` (i.e. $S^1$) instead of `intrinsic_dim = 2` (i.e. $S^2$). Other demos (`demo_vortex.py:27`, `demo_learned_wind.py:22`, `demo_trajectories.py:78`) correctly use `Sphere(radius=1.0)`.
- **Verdict:** WARNING
- **Notes:** The `project`, `to_tangent`, `exp_map`, and `log_map` methods operate on whatever-dimensional arrays they receive and do not branch on `intrinsic_dim`, so the numerical output is unaffected for this demo. However, any downstream code querying `manifold.ambient_dim` or `manifold.intrinsic_dim` will get wrong values (2 and 1 instead of 3 and 2).
- **Recommended Action:** Change to `Sphere(radius=radius)` or `Sphere(2, radius)`.

### 2. Wind Vector Field $W(x)$

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, Zermelo inputs
- **Literature Reference:** Bao, Robles, Shen, "Zermelo navigation on Riemannian manifolds," *J. Differential Geom.* 66 (2004), 377–435.
- **Implementation:** `examples/demo_zermelo.py:17–19`
  ```python
  def w_net(x):
      base = jnp.array([-x[1], x[0], 0.0])
      return 0.9 * base
  ```
  This defines $W(x) = 0.9\,(\hat{z} \times x)$, the infinitesimal generator of counter-clockwise rotation about the $z$-axis.

  **Tangency check:** $\langle W(x), x \rangle = 0.9(-x_1 x_0 + x_0 x_1) = 0$ for all $x$, so $W \in T_x S^2$. ✓

  **Causality check:** With $h = I_3$ (ambient Euclidean), $\|W\|_h^2 = 0.81(x_0^2 + x_1^2)$. On $S^2$, $x_0^2 + x_1^2 = 1 - x_2^2 \le 1$, so $\|W\|_h \le 0.9 < 1$. ✓

- **Verdict:** CORRECT

### 3. Comment–Code Mismatches

- **Implementation:** `examples/demo_zermelo.py:16`
  ```python
  # Strength 0.8 at equator. Counter-Clockwise.
  ```
  The code uses coefficient `0.9`, not `0.8`. (Note: the companion file `examples/demo_discrete_zermelo.py:17` uses `0.8`, matching its comment.)
- **Implementation:** `examples/demo_zermelo.py:25`
  ```python
  # --- 2. Mission: South -> North ---
  ```
  Start point `[1, 0, 0]` is on the equator, not the south pole ($[0, 0, -1]$).
- **Verdict:** NOTE
- **Notes:** These do not affect correctness but reduce readability.
- **Recommended Action:** Update the comment to `Strength 0.9 at equator` and `Mission: Equator -> North Pole`.

### 4. Riemannian Metric $h_{ij}(x)$

- **Spec Reference:** `spec/MATH_SPEC.md` § 5
- **Implementation:** `examples/demo_zermelo.py:21`
  ```python
  def h_net(x): return jnp.eye(3)
  ```
  Uses the ambient Euclidean metric $\delta_{ij}$ in $\mathbb{R}^3$. When restricted to tangent vectors of $S^2$, this is the standard round metric on the unit sphere. The `Randers._get_zermelo_data` method ([zoo.py](src/ham/geometry/zoo.py#L85–L117)) symmetrizes and regularizes $H$, so the identity passes through unchanged (up to a $+0.005 I$ shift).
- **Verdict:** CORRECT

### 5. Randers Metric Formula $F(x, v)$

- **Spec Reference:** `spec/MATH_SPEC.md` § 5
- **Literature Reference:** Bao–Robles–Shen (2004), Theorem 1.1
- **Implementation:** `src/ham/geometry/zoo.py:119–132`
  ```python
  v_sq_h = jnp.sum(v_safe * Hv, axis=-1)          # ||v||_h^2
  W_dot_v = jnp.sum(v_safe * HW, axis=-1)          # <W, v>_h
  discriminant = lam * v_sq_h + W_dot_v**2
  cost = (jnp.sqrt(...discriminant...) - W_dot_v) / lam
  ```
  This computes:
  $$F(x, v) = \frac{\sqrt{\lambda\,\|v\|_h^2 + \langle W, v\rangle_h^2} \;-\; \langle W, v\rangle_h}{\lambda}, \qquad \lambda = 1 - \|W\|_h^2$$
  which matches the spec exactly.

  **Sign convention:** Headwind ($\langle W, v\rangle_h < 0$) increases cost, tailwind decreases it. This is the correct Zermelo navigation convention. ✓

  **Inner product:** $\langle W, v \rangle_h = v^T H W$. Since $H$ is symmetric, $v^T H W = W^T H v$. ✓

  **Discriminant non-negativity:** $\lambda > 0$ (ensured by squashing) and $\|v\|_h^2 \ge 0$, so $\lambda\|v\|_h^2 + \langle W,v\rangle_h^2 \ge 0$. ✓
- **Verdict:** CORRECT

### 6. Zermelo Causality Squasher

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, constraint $\|W\|_h < 1$
- **Implementation:** `src/ham/geometry/zoo.py:103–112`
  ```python
  squash_factor = (max_speed * jnp.tanh(w_norm)) / (w_norm + 1e-8)
  W_safe = W_raw * squash_factor
  ```
  For the demo's equatorial wind ($w_\text{norm} = 0.9$), $\tanh(0.9) \approx 0.716$, giving an effective norm $\approx 0.716$ instead of the nominal $0.9$. The squasher is applied unconditionally, even when the raw wind already satisfies $\|W\|_h < 1$.
- **Verdict:** WARNING
- **Notes:** The squasher is a safety mechanism in the `Randers` class, not a bug in the demo. However, the effective wind used for geodesic computation is ~20 % weaker than the `0.9` specified in `w_net`. If the demo is intended to demonstrate strong wind deflection, this attenuation may be misleading. This is a property of `zoo.Randers`, not of `demo_zermelo.py`.
- **Recommended Action:** (For `zoo.py`, not this demo) Consider a passthrough mode or document that the nominal wind strength differs from the effective strength after squashing.

### 7. Energy Comparison

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2, $E = \frac{1}{2}F^2$
- **Implementation:** `examples/demo_zermelo.py:36–38`
  ```python
  batch_energy = jax.vmap(metric_randers.energy)
  e_riem = batch_energy(traj_riem.xs[:-1], traj_riem.vs).sum()
  e_rand = batch_energy(traj_rand.xs[:-1], traj_rand.vs).sum()
  ```
  Both trajectories are evaluated with the **Randers** energy $E_{\text{Randers}}$. This is a valid comparison: the Randers-optimal trajectory should have $E_{\text{Randers}}(\gamma_{\text{Randers}}) \le E_{\text{Randers}}(\gamma_{\text{Riemannian}})$. However, the variable name `e_riem` is misleading—it is the Randers energy of the Riemannian-optimal path, not the Riemannian energy.
- **Verdict:** NOTE
- **Notes:** Mathematically sound comparison; naming is a readability concern only.

### 8. Randers Energy $E(x, v) = \frac{1}{2} F^2(x, v)$

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2
- **Implementation:** `src/ham/geometry/metric.py:36`
  ```python
  def energy(self, x, v):
      return 0.5 * self.metric_fn(x, v)**2
  ```
- **Verdict:** CORRECT

### 9. Spray and Geodesic ODE

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1–2.2
- **Implementation:** `src/ham/geometry/metric.py:48–66`
  The spray is computed via the implicit solve:
  $$g_{ij}\,\text{acc}^j = \nabla_{x^i} E - \bigl[\text{Jac}_x(\nabla_v E)\cdot v\bigr]_i, \qquad G^i = -\tfrac{1}{2}\,\text{acc}^i$$
  Substituting yields $G^i = \frac{1}{2}g^{il}\bigl(\partial^2_{x^k v^l} E\; v^k - \partial_{x^l} E\bigr)$, matching the standard spray formula (Shen, "Lectures on Finsler Geometry," §5.1). The solver uses these spray coefficients indirectly through the `log_map`-based discrete energy minimization in `AVBDSolver`.
- **Verdict:** CORRECT

### 10. Tangent Projection of Wind in `_get_zermelo_data`

- **Spec Reference:** `spec/MATH_SPEC.md` § 5, "Wind field: $W^i(x)$"
- **Implementation:** `src/ham/geometry/zoo.py:100`
  ```python
  W_raw = self.manifold.to_tangent(z, W_raw)
  ```
  For the demo's wind $W(x) = 0.9(-x_1, x_0, 0)$, which is already tangent ($\langle W, x\rangle = 0$), this is a no-op. For general learned wind fields this is essential.
- **Verdict:** STRONG — Good defensive practice ensuring $W \in T_x M$ regardless of `w_net` output.

### 11. Riemannian Baseline via Zero Wind

- **Spec Reference:** `spec/MATH_SPEC.md` § 4, table row "Riemannian"
- **Implementation:** `examples/demo_zermelo.py:22`
  ```python
  metric_riem = Randers(sphere_cont, h_net, lambda x: jnp.zeros(3))
  ```
  With $W = 0$, $\lambda = 1$, and $F(x,v) = \sqrt{\|v\|_h^2} = \|v\|_h$. This correctly degenerates to the Riemannian metric induced by $h$. ✓
- **Verdict:** CORRECT

## Open Questions

1. The `Sphere(radius)` positional-argument bug also appears in `examples/demo_discrete_zermelo.py:12`. Both demos should be fixed together.
2. The `+0.005 I$ regularization in `_get_zermelo_data` ([zoo.py](src/ham/geometry/zoo.py#L94)) shifts the identity metric to $1.005\,I$, slightly inflating all h-norms. This is negligible for $h = I$ but should be documented, as it breaks exact unit-sphere geometry at $O(10^{-3})$.
3. The `+1e\text{-}4\,I$ regularization in the spray Hessian solve ([metric.py](src/ham/geometry/metric.py#L64)) introduces a small bias in the geodesic equation. For $3\times 3$ ambient systems with a rank-2 tangent constraint, this regularization is necessary but its effect on spray accuracy deserves quantification.
