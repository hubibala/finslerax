# Math Review: fields

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source File:** `src/ham/sim/fields.py`

## Summary

**Verdict: Minor Issues.** The file implements vector/wind fields for use in Zermelo navigation on spheres (and 2D vortices). All core formulas—stream-function construction, Rossby–Haurwitz wave, Lamb–Oseen vortex, Rankine vortex—are analytically correct and match their standard literature definitions. One WARNING is raised for an incorrect epsilon-regularisation pattern that is currently benign because the affected vector always has unit norm.

---

## Formula-by-Formula Audit

### 1. `get_stream_function_flow` — Stream-function to divergence-free tangential field

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (Zermelo Parameterization — wind field $W^i(x)$).
- **Literature Reference:** Standard construction on $S^2$; see Williamson et al. (1992), "A Standard Test Set for Numerical Approximations to the Shallow Water Equations in Spherical Geometry," §3.
- **Implementation:** `src/ham/sim/fields.py:6–18`
  ```python
  grad_psi = jax.grad(psi_fn)(x)       # ∇ψ in ℝ³
  v = jnp.cross(grad_psi, x)           # v = ∇ψ × x
  ```
- **Verdict:** OK
- **Notes:**
  1. **Tangentiality:** $\mathbf{v} \cdot \mathbf{x} = (\nabla\psi \times \mathbf{x}) \cdot \mathbf{x} = 0$ by the scalar triple product identity. ✓
  2. **Divergence-free on $S^2$:** Follows from the stream-function construction. ✓
  3. **Ambient vs. surface gradient:** `jax.grad` returns the ambient gradient $\nabla_{\mathbb{R}^3}\psi$, not the surface gradient $\nabla_{S^2}\psi$. This is correct: the normal component $(\nabla\psi \cdot \hat{n})\hat{n}$ is parallel to $\mathbf{x}$, so $(\nabla\psi \cdot \hat{n})\hat{n} \times \mathbf{x} = 0$, and both formulations give the same cross product.
  4. **Sign convention:** The code uses $\nabla\psi \times \mathbf{x}$, whereas some references use $\hat{n} \times \nabla\psi = \mathbf{x} \times \nabla\psi = -\nabla\psi \times \mathbf{x}$. This is a valid convention choice (reverses handedness). Consistent within the codebase.

---

### 2. `tilted_rotation` — Constant rotation around a tilted axis

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (wind field for Randers metric).
- **Literature Reference:** Williamson et al. (1992), Test Case 1 (advection over the pole with tilted axis).
- **Implementation:** `src/ham/sim/fields.py:21–36`
  ```python
  axis = jnp.array([jnp.sin(alpha), 0.0, jnp.cos(alpha)])
  axis = axis / jnp.linalg.norm(axis + 1e-10)   # ← WARNING
  ...
  psi(x) = jnp.dot(axis, x)   # linear stream function
  ```
  The resulting flow is $\mathbf{v} = \hat{a} \times \mathbf{x}$ (uniform rotation about $\hat{a}$).
- **Verdict:** WARNING
- **Notes:**
  The axis vector $[\sin\alpha, 0, \cos\alpha]$ always has unit norm ($\sin^2\alpha + \cos^2\alpha = 1$), so the normalisation is redundant. More importantly, the epsilon is applied incorrectly: `jnp.linalg.norm(axis + 1e-10)` adds $\epsilon$ to each **component** before computing the norm, producing $\|\hat{a} + \epsilon\mathbf{1}\|$ instead of $\|\hat{a}\| + \epsilon$. For a unit vector, the perturbation is $O(\epsilon)$ and has no practical effect, but the pattern is mathematically wrong.

  **Recommended Action:** Replace with `axis / jnp.maximum(jnp.linalg.norm(axis), 1e-10)`, or remove the normalisation entirely since the norm is identically 1.

---

### 3. `rossby_haurwitz` — Rossby–Haurwitz wave

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (wind field).
- **Literature Reference:** Haurwitz (1940); Williamson et al. (1992), Test Case 6. The stream function is:
  $$\psi = -\omega\sin\phi + K\cos^R\!\phi\;\sin\phi\;\cos(R\lambda)$$
  where $\phi$ is latitude and $\lambda$ is longitude.
- **Implementation:** `src/ham/sim/fields.py:39–64`
  ```python
  z = x[2]                               # sin(lat)
  rho_xy = jnp.sqrt(jnp.sum(x[:2]**2) + 1e-10)  # cos(lat)
  cos_R_lon = jnp.real(xy_unit ** R)     # cos(R·lon) via De Moivre
  term1 = -omega * z                     # −ω sin(lat)
  term2 = K * (rho_xy ** R) * z * cos_R_lon  # K cos^R(lat) sin(lat) cos(R·lon)
  ```
- **Verdict:** OK
- **Notes:**
  1. **Coordinate mapping:** $\sin\phi = z$, $\cos\phi = \sqrt{x^2+y^2}$ (unit sphere), $\cos(R\lambda) = \Re[(e^{i\lambda})^R]$ via De Moivre's theorem. All correct. ✓
  2. **Epsilon placement:** `rho_xy` uses $\sqrt{x^2+y^2+10^{-10}}$ (epsilon inside sqrt), while `xy_norm` uses $\max(|x+iy|, 10^{-10})$. Both are valid regularisations at the poles. The two separate epsilons do not interact problematically because at the poles $\rho_{xy}^R \to 0$ and the entire term2 vanishes regardless.
  3. The `R` parameter is typed as `int`, which is correct: non-integer $R$ would make $\cos(R\lambda)$ lose its spherical-harmonic character.

---

### 4. `harmonic_vortices` — Cellular vortex flow

- **Spec Reference:** None directly (utility field for experiments).
- **Literature Reference:** Inspired by spherical harmonics $Y_l^m(\theta, \lambda)$; the docstring correctly states "associated Legendre-**like** structure."
- **Implementation:** `src/ham/sim/fields.py:67–92`
  ```python
  cos_lat = jnp.sqrt(1.0 - z**2 + 1e-10)
  cos_lat_m = cos_lat ** m
  z_poly = jnp.sin(l * jnp.pi * z)      # latitudinal oscillation
  cos_m_lon = jnp.real(xy_unit ** m)     # cos(m·lon)
  return cos_lat_m * z_poly * cos_m_lon
  ```
- **Verdict:** OK
- **Notes:**
  1. This is **not** an exact spherical harmonic (the latitudinal part uses $\sin(l\pi z)$ instead of an associated Legendre polynomial $P_l^m(\cos\theta)$). The docstring is clear about this ("can be replaced with real Legendre"). It is simply a valid smooth stream function on $S^2$ that generates divergence-free tangential flow. ✓
  2. The epsilon $\sqrt{1 - z^2 + 10^{-10}}$ prevents a zero under the radical at the poles ($z = \pm 1$). ✓

---

### 5. `lamb_oseen_vortex` — 2D Lamb–Oseen (smoothed point) vortex

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (wind field in $\mathbb{R}^2$).
- **Literature Reference:** Lamb (1932), *Hydrodynamics*; Oseen (1911). The azimuthal velocity of the Lamb–Oseen vortex at distance $r$ from the center is:
  $$v_\theta(r) = \frac{\Gamma}{2\pi r}\left(1 - e^{-r^2/r_c^2}\right)$$
- **Implementation:** `src/ham/sim/fields.py:94–109`
  ```python
  r_sq = jnp.sum(r_vec**2) + 1e-10
  r = jnp.sqrt(r_sq)
  velocity_mag = (circulation / (2 * jnp.pi * r)) * (1.0 - jnp.exp(-r_sq / (core_radius**2)))
  v_x = -velocity_mag * r_vec[1] / r    # −v_θ sin θ
  v_y =  velocity_mag * r_vec[0] / r    #  v_θ cos θ
  ```
- **Verdict:** OK
- **Notes:**
  1. The formula matches the standard Lamb–Oseen velocity profile exactly. ✓
  2. **Direction:** With $\mathbf{r} = r[\cos\theta, \sin\theta]$, the Cartesian decomposition $v_x = -v_\theta \sin\theta$, $v_y = v_\theta \cos\theta$ gives counter-clockwise rotation for $\Gamma > 0$. ✓
  3. **Regularisation at origin:** $r_\text{sq} = r^2 + 10^{-10}$, so the exponential becomes $e^{-10^{-10}/r_c^2} \approx 1$, giving $v_\theta \approx 0$. The velocity correctly vanishes at the center. ✓
  4. **Dimensional note:** The standard Lamb–Oseen uses $r_c^2 = 4\nu t$ (viscous diffusion radius). Here $r_c$ is treated as a free parameter for a time-frozen snapshot, which is standard practice. ✓

---

### 6. `rankine_vortex` — 2D Rankine vortex

- **Spec Reference:** `spec/MATH_SPEC.md` § 5 (wind field in $\mathbb{R}^2$).
- **Literature Reference:** Rankine (1858). The piecewise-defined azimuthal velocity is:
  $$v_\theta(r) = \begin{cases} \dfrac{\Gamma\, r}{2\pi\, r_c^2} & r \le r_c \\[6pt] \dfrac{\Gamma}{2\pi\, r} & r > r_c \end{cases}$$
- **Implementation:** `src/ham/sim/fields.py:111–129`
  ```python
  v_theta = jnp.where(r <= core_radius,
                       (circulation * r) / (2 * jnp.pi * core_radius**2),
                       circulation / (2 * jnp.pi * r))
  ```
- **Verdict:** OK
- **Notes:**
  1. Both branches match the standard definition. ✓
  2. **Continuity at $r = r_c$:** Inside gives $\Gamma r_c / (2\pi r_c^2) = \Gamma/(2\pi r_c)$; outside gives $\Gamma/(2\pi r_c)$. Continuous. ✓
  3. **JAX `jnp.where` semantics:** Both branches are evaluated for all inputs (JAX traces both). At the origin ($r \approx 10^{-10}$), the outside branch computes $\Gamma/(2\pi \cdot 10^{-10})$, which is very large but finite. The inside branch is selected, so the forward pass is correct. Gradients through `jnp.where` are also correct (JAX selects the gradient of the chosen branch). No issue for first-order differentiation.
  4. **Direction decomposition:** Same as Lamb–Oseen (counter-clockwise for $\Gamma > 0$). ✓

---

## Open Questions

1. **Zermelo wind-norm constraint:** The Randers metric (spec § 5) requires $\|W\|_h < 1$. None of the field constructors enforce or check this bound. For `tilted_rotation` on the unit sphere with the round metric, $\|W\| = |\hat{a} \times \mathbf{x}|$ reaches 1 at the equator (relative to the rotation axis), violating the strict inequality. This is an upstream concern (the metric module should enforce the bound), but users combining these fields with Zermelo navigation should be aware.

2. **Missing `__init__.py` in `src/ham/sim/`:** The `sim` package has no `__init__.py`, which means `from ham.sim.fields import ...` works only with implicit namespace packages or a permissive `pyproject.toml` configuration. Not a mathematical concern, but could cause import failures.

3. **Differentiability of Rankine vortex:** The Rankine velocity profile has a discontinuous first derivative (vorticity jump) at $r = r_c$. If this field is used as a wind $W^i(x)$ inside a Finsler energy that is differentiated to 3rd order (Berwald connection, spec § 3.1), the resulting connection coefficients will contain distributional terms at $r = r_c$. Users requiring smooth Berwald transport should prefer `lamb_oseen_vortex`.
