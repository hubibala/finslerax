# Math Review: plot_publication_figs

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

No critical formula errors. The pullback metric determinant, wind projection, and arc-length discretisation are all implemented correctly. There are two **WARNING**-level issues: (1) the annotation label "$E$" collides with the spec's energy functional $E = \tfrac{1}{2}F^2$ and will confuse readers familiar with the spec; (2) the wind quiver plot displays the *raw* wind from `w_net`, not the causality-squashed wind that the Randers metric actually uses. Several minor notes on omitted regularisation in labels and redundant computation.

---

## Formula-by-Formula Audit

### 1. Pullback metric determinant — `compute_logdet_grid`

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 (fundamental tensor), § 5 (Zermelo / Randers)
- **Implementation:** [examples/plot_publication_figs.py](examples/plot_publication_figs.py#L56-L67)

```python
J    = jax.jacfwd(v_mod.decode)(z)            # (data_dim, latent_dim)
G    = jnp.dot(J.T, J) + 1e-6 * jnp.eye(latent_dim)
sign, ld = jnp.linalg.slogdet(G)
```

The pullback metric of the decoder is $G_{ij}(z) = \sum_a \frac{\partial f^a}{\partial z^i}\frac{\partial f^a}{\partial z^j} = (J^T J)_{ij}$. The code correctly computes $J = \mathrm{Jac}(f)$, forms $G = J^T J + \epsilon I$, and takes $\log\det$ via `slogdet`. The sign guard (`sign > 0`) correctly discards degenerate grid points.

- **Verdict:** CORRECT

---

### 2. Panel A title label

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 ($E = \tfrac{1}{2}F^2$)
- **Implementation:** [examples/plot_publication_figs.py](examples/plot_publication_figs.py#L213-L214)

```python
ax_det.set_title("(A)  Pullback Metric Determinant\nlog det $G(z) = \\log \\det J^T J$",
                 fontsize=11, pad=8)
```

The displayed $G$ is actually $J^T J + 10^{-6} I$ (line 62). For publication the regulariser term should either be mentioned or acknowledged as negligible.

- **Verdict:** NOTE
- **Notes:** Consider "log det $(J^T J + \epsilon I)$" or a footnote mentioning the regularisation.

---

### 3. Wind field evaluation — `compute_wind_grid`

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Navigation), strict constraint $\|W\|_h < 1$
- **Literature Reference:** Bao–Robles–Shen, *J. Differential Geom.* 66 (2004), Zermelo navigation on Randers spaces
- **Implementation:** [examples/plot_publication_figs.py](examples/plot_publication_figs.py#L86-L88)

```python
def get_winds(m_mod, pts):
    return m_mod.w_net(pts)
```

`w_net` returns the **raw, unsquashed** wind vector (a `KernelWindField` kernel-smoother output). The actual Randers metric uses the causality-squashed wind $W_{\text{safe}}$ produced by `_get_zermelo_data` in [src/ham/geometry/zoo.py](src/ham/geometry/zoo.py#L100-L108):

$$W_{\text{safe}} = W_{\text{raw}} \cdot \frac{(1 - \epsilon)\tanh(\|W_{\text{raw}}\|_H)}{\|W_{\text{raw}}\|_H + 10^{-8}}$$

The quiver plot therefore shows a wind field that may violate the Zermelo causality constraint $\|W\|_H < 1$, misrepresenting the metric that the model actually uses.

- **Verdict:** WARNING
- **Recommended Action:** Replace the `w_net` call with `_get_zermelo_data` and display the squashed $W_{\text{safe}}$, e.g.:

  ```python
  def get_wind_safe(m_mod, pt):
      _, W_safe, _ = m_mod._get_zermelo_data(pt)
      return W_safe
  ```

---

### 4. Wind projection onto 2-D PCA plane

- **Spec Reference:** (linear algebra, not Finsler-specific)
- **Implementation:** [examples/plot_publication_figs.py](examples/plot_publication_figs.py#L93-L95)

```python
components = pca2.components_[:2]          # (2, D), orthonormal rows
winds_2d = winds_full @ components.T       # (N, 2)
magnitudes = np.linalg.norm(winds_full, axis=1)
```

The projection $W_{2D} = W_D \, C^T$ with orthonormal $C$ is a correct orthogonal projection onto the PCA subspace. Arrow directions are faithful.

However, `magnitudes` is computed in the full $D$-dimensional space while arrows are 2-D projections. Arrow length and colour (full-$D$ magnitude) will be inconsistent whenever the wind has large out-of-plane components.

- **Verdict:** NOTE
- **Notes:** The colour encodes the $D$-dimensional $\|W\|_2$ while the arrow length reflects only the in-plane component. Consider using `np.linalg.norm(winds_2d, axis=1)` if colour should match arrow length, or add a caption clarifying the distinction.

---

### 5. Arc-length discretisation — `arc_length_segment` and `FinslerMetric.arc_length`

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 (positive 1-homogeneity of $F$)
- **Implementation:**
  - [examples/plot_publication_figs.py](examples/plot_publication_figs.py#L102-L106) (`arc_length_segment`)
  - [src/ham/geometry/metric.py](src/ham/geometry/metric.py#L69-L79) (`arc_length`)

The Finsler length of a curve is:
$$L[\gamma] = \int_0^1 F\!\bigl(\gamma(t),\,\dot\gamma(t)\bigr)\,dt$$

The code discretises with $N=20$ equally-spaced points:
$$L \approx \sum_{i=0}^{N-2} F\!\!\left(\frac{x_i + x_{i+1}}{2},\; x_{i+1} - x_i\right)$$

Because $F$ is 1-homogeneous in $v$, $F(x, \Delta x) = \|\Delta x\| \cdot F(x, \Delta x / \|\Delta x\|)$, so the step size is implicitly absorbed. The midpoint evaluation gives a second-order quadrature. Mathematically correct.

- **Verdict:** CORRECT

---

### 6. Cost label "$E$" in Panel C

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2 — $E(x,v) = \tfrac{1}{2}F^2(x,v)$
- **Implementation:** [examples/plot_publication_figs.py](examples/plot_publication_figs.py#L354-L358)

```python
ax_traj.text(mid_obs[0] + 0.05, mid_obs[1], f"$E$={e_obs1 + e_obs2:.2f}",
             fontsize=8, color="limegreen", fontweight="bold", zorder=8)
```

and [examples/plot_publication_figs.py](examples/plot_publication_figs.py#L383-L384):
```python
ax_traj.set_title("(C)  Exemplar Trajectory vs Counterfactual\n"
                  "($E$ = Randers arc length cost)", fontsize=11, pad=8)
```

The quantity plotted is $L[\gamma] = \sum F(\cdot)$ (Finsler *length*, 1-homogeneous), **not** $E = \tfrac{1}{2}F^2$ (Finsler *energy*, 2-homogeneous). Using the symbol "$E$" for arc length directly contradicts the spec notation defined in § 1.2 and will mislead any reader familiar with Finsler geometry.

- **Verdict:** WARNING
- **Recommended Action:** Replace "$E$" with "$L$" or "$\ell$" (Finsler length / arc-length cost) in both the annotation text and the Panel C title, e.g. `($L$ = Randers arc length)`.

---

### 7. Straight-line vs. geodesic path cost

- **Spec Reference:** `spec/MATH_SPEC.md` § 2.1 (geodesic ODE $\ddot x^i + 2G^i = 0$)
- **Implementation:** [examples/plot_publication_figs.py](examples/plot_publication_figs.py#L102-L106)

```python
gamma = z_start + t * (z_end - z_start)
```

The cost is computed along a *straight line* in latent space, not along the Randers geodesic. This is a valid upper bound on geodesic distance but is not the geodesic distance itself. The Panel C annotations and title do not make this distinction.

- **Verdict:** NOTE
- **Notes:** Consider adding a caption or footnote: "Cost computed along straight latent-space segments, not geodesics." If geodesic distance is intended, the spray-based ODE solver from `spec/MATH_SPEC.md` § 2.1 should be used instead.

---

### 8. Redundant computation — `e_cf1`

- **Implementation:** [examples/plot_publication_figs.py](examples/plot_publication_figs.py#L346-L349)

```python
e_obs1 = float(arc_fn(vae_randers.metric, z2, z4))
e_obs2 = float(arc_fn(vae_randers.metric, z4, z6))
e_cf1  = float(arc_fn(vae_randers.metric, z2, z4))   # identical to e_obs1
e_cf2  = float(arc_fn(vae_randers.metric, z4, z6_cf))
```

`e_cf1 == e_obs1` by construction (same arguments). Not a mathematical error, but the total cost label for each path sums both segments: the observed and counterfactual totals differ only in their second segment ($z_4 \to z_6$ vs. $z_4 \to z_{6,\text{cf}}$). The annotation is therefore correct in value, but the redundant call wastes computation.

- **Verdict:** NOTE

---

### 9. `PullbackGNet` vs. `compute_logdet_grid` consistency

- **Implementation:**
  - [examples/plot_publication_figs.py](examples/plot_publication_figs.py#L59-L62) (`compute_logdet_grid`)
  - [src/ham/models/learned.py](src/ham/models/learned.py#L123-L128) (`PullbackGNet.__call__`)

Both compute $G = J^T J + \epsilon I$, but with different $\epsilon$ values: `compute_logdet_grid` uses $10^{-6}$, while `PullbackGNet` uses $10^{-4}$. The heatmap therefore visualises a slightly different matrix from the one used in the actual Randers metric.

- **Verdict:** NOTE
- **Notes:** For strict consistency, use the same $\epsilon$ as `PullbackGNet` ($10^{-4}$), or call `vae.metric.h_net(z)` directly instead of recomputing the Jacobian.

---

## Open Questions

1. **Wind display intent:** Is the quiver plot *intended* to show the raw data-driven velocity field (before squashing), or the effective Randers wind? If the former, the title should say "Data velocity field" rather than "Randers Wind Field $W(z)$."
2. **Straight-line cost vs. geodesic distance:** Is the straight-line Finsler cost the intended quantity for the publication, or should geodesic distances be computed? The former is cheaper but only an upper bound.
3. **Magnitude colour convention:** Should quiver colour encode in-plane ($\|W_{2D}\|$) or full-space ($\|W_D\|$) wind magnitude? Both are valid but convey different information.
