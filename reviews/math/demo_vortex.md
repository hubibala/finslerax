# Math Review: `examples/demo_vortex.py`

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)

## Summary

The demo is **mathematically correct** with one **WARNING**-level finding related to the visualisation of the wind field. The vortex field is a well-constructed tangent vector field on $S^2$, the geodesic distance formula is exact, and the Randers metric invoked through `zoo.Randers` matches `spec/MATH_SPEC.md` § 5.

## Formula-by-Formula Audit

### 1. Vortex centre normalisation

- **Spec Reference:** Standard differential geometry — points on $S^2(1) \subset \mathbb{R}^3$ satisfy $\|x\| = 1$.
- **Implementation:** `examples/demo_vortex.py:14` — `center = center / jnp.linalg.norm(center)`.
- **Verdict:** CORRECT
- **Notes:** Projects the user-supplied centre onto the unit sphere. Straightforward and correct.

### 2. Geodesic distance on $S^2$

- **Spec Reference:** For unit sphere, $d(x,c) = \arccos\bigl(\langle x, c \rangle\bigr)$, $\langle x, c \rangle \in [-1, 1]$.
- **Implementation:** `examples/demo_vortex.py:16–17`
  ```python
  cos_dist = jnp.dot(x, center)
  dist = jnp.arccos(jnp.clip(cos_dist, -1.0, 1.0))
  ```
- **Verdict:** CORRECT
- **Notes:** `clip` to $[-1,1]$ is standard numerical safety for `arccos`. At $\cos\theta = \pm 1$ the `clip` may zero out the JAX gradient of `arccos`, but this is benign because the wind magnitude factor $(c \times x)$ vanishes at those same points (see § 3 below), so the product rule produces finite derivatives.

### 3. Tangent rotation vector $v_{\mathrm{rot}} = c \times x$

- **Spec Reference:** For any $x \in S^2(1)$ and pole $c \in S^2(1)$, the cross product $c \times x$ lies in $T_x S^2$ because $\langle c \times x,\, x \rangle = 0$ identically. This is the standard generator of rotation about the $c$-axis.
- **Implementation:** `examples/demo_vortex.py:18` — `v_rot = jnp.cross(center, x)`.
- **Verdict:** **STRONG** — The cross-product construction guarantees tangency without requiring an explicit projection step. $\|c \times x\| = \sin d(x,c)$, so the field vanishes at both the centre ($d=0$) and the antipode ($d=\pi$), which is the physically correct vortex-eye behaviour.

### 4. Gaussian magnitude envelope

- **Spec Reference:** No specific spec entry; this is a user-defined wind profile. The profile $m(x) = s\, e^{-\alpha\, d^2}$ is a standard isotropic Gaussian on the sphere (geodesic-distance kernel).
- **Literature Reference:** Gaussian kernels on $S^n$ via geodesic distance, see e.g. Borodachov–Hardin–Saff, *Discrete Energy on Rectifiable Sets*, Springer (2019), Ch. 14.
- **Implementation:** `examples/demo_vortex.py:19` — `magnitude = strength * jnp.exp(-decay * (dist**2))`.
- **Verdict:** CORRECT
- **Notes:** Combined with $\|c \times x\| = \sin d$, the wind magnitude is $\|W(x)\| = s\, e^{-\alpha d^2} \sin d$, which peaks at $d_* = \frac{1}{2}\sqrt{1/\alpha}$ for small $\alpha$ and decays smoothly. Well-behaved everywhere on $S^2$.

### 5. Randers metric setup

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 — Zermelo parameterisation:
  $$F(x,v)=\frac{\sqrt{\lambda\,\|v\|_h^2+\langle W,v\rangle_h^2}-\langle W,v\rangle_h}{\lambda},\qquad \lambda=1-\|W\|_h^2$$
- **Implementation:** `examples/demo_vortex.py:27–29`
  ```python
  h_net = lambda x: jnp.eye(3)
  metric = Randers(sphere, h_net, w_net)
  ```
  The Randers class (`src/ham/geometry/zoo.py:116–134`) implements the spec formula verbatim: `discriminant = lam * v_sq_h + W_dot_v**2`, `cost = (sqrt(discriminant) - W_dot_v) / lam`.
- **Verdict:** CORRECT
- **Notes:** Using $h_{ij} = \delta_{ij}$ (the ambient Euclidean metric restricted to $T_x S^2$) correctly reproduces the round metric on the unit sphere for tangent vectors, since $v^T I\, v = \|v\|^2 = \|v\|_{S^2}^2$ when $\langle v, x\rangle = 0$.

### 6. Causality constraint $\|W\|_h < 1$ (delegated to `Randers`)

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 — $\|W\|_h < 1$ is required for the Randers norm to be positive definite.
- **Implementation:** The demo sets `strength=10.0` (`examples/demo_vortex.py:27`). The raw wind magnitude can reach $\|W\| \approx 1.89$ (at $d_* \approx 0.32$ with `decay=5.0`), which violates $\|W\|_h < 1$. The `Randers._get_zermelo_data` method (`src/ham/geometry/zoo.py:97–103`) squashes the wind via $\tanh$ so that $\|W_{\text{safe}}\|_h < 1 - \epsilon$. The metric is therefore mathematically well-defined.
- **Verdict:** **WARNING**
- **Notes:** See Finding W-1 below.

### 7. Endpoint geometry

- **Spec Reference:** Points must satisfy $\|x\| = 1$ for $S^2(1)$.
- **Implementation:**
  - `start = jnp.array([1.0, 0.0, 0.0])` — norm 1. ✓ (`examples/demo_vortex.py:31`)
  - `end = jnp.array([-0.99, 0.1, 0.0]); end = end / jnp.linalg.norm(end)` — explicitly normalised. ✓ (`examples/demo_vortex.py:32–33`)
- **Verdict:** CORRECT

---

## Findings

### W-1 — Visualised wind ≠ metric wind (WARNING)

**File:** `examples/demo_vortex.py:47–48`  
**Lines:**
```python
wind_vecs = np.array(jax.vmap(w_net)(pts))
plot_vector_field(ax, pts, wind_vecs, scale=0.2, color='cyan', alpha=0.3)
```

The plot renders the **raw** output of `w_net` (pre-squash, $\|W\| \gg 1$ near the vortex peak), but the `Randers` metric internally squashes the wind to $\|W_{\text{safe}}\|_h < 0.95$ via a $\tanh$ nonlinearity (`src/ham/geometry/zoo.py:97–103`). With `strength=10.0` the raw and effective fields differ by up to a factor of $\sim 5\times$, so the cyan arrows misrepresent the actual anisotropy encoded in the Finsler metric. This does not affect numerical results but is misleading for geometric interpretation.

**Recommended Action:** Visualise the squashed wind by extracting it from `metric._get_zermelo_data(x)` instead of calling `w_net` directly, or add a note to the plot title clarifying that arrows show the raw (pre-squash) field.

---

## Open Questions

1. **Peak-location formula.** The wind magnitude profile $\|W\| = s\,e^{-\alpha d^2}\sin d$ peaks where $\cos d = 2\alpha\, d\, \sin d$. For the demo parameters ($\alpha = 5$) the peak is at $d_* \approx 0.32\,\text{rad}$ ($\approx 18°$). Is this the intended vortex radius, or should `decay` be tuned to a specific angular scale? This is a modelling choice, not a correctness issue.
