# Open Issues & Roadmap

This document tracks known limitations, architectural questions, and planned enhancements for the HAMTools library. Issues are formatted to be easily ported to GitHub Issues if the repository is open-sourced.

---

## [ENHANCEMENT] Upgrade Parallel Transport Integrator to RK4
**Component:** `ham.geometry.transport`
**Status:** Open

**Description:**
Currently, `BerwaldConnection.parallel_transport` uses a 1st-order forward Euler integration scheme. While recent fixes to the time-step calculation and tangent space projection have significantly improved accuracy (allowing us to tighten holonomy tolerances), Euler integration inherently drifts on curved manifolds.

The geodesic IVP solver (`ExponentialMap`) already uses RK4. For long transport paths or highly curved manifolds, the parallel transport should be upgraded to RK4 or a symplectic Euler variant to prevent $O(1)$ global error accumulation.

**Proposed Action:**
- Implement RK4 within the `transport_ode` scan body.
- Benchmark the computational cost, as RK4 requires 4 evaluations of $\Gamma$ per step (which involves computationally expensive double `jacfwd` calls).

---

## [PERFORMANCE] 4th-Order Autodiff Compilation Cost in Curvature
**Component:** `ham.geometry.curvature`
**Status:** Open

**Description:**
The Riemann curvature tensor $R^i_{jk}$ computation involves 4th-order differentiation of the energy $E$ (Energy -> Spray -> Nonlinear Connection -> Riemann Tensor). For higher-dimensional manifolds ($D \geq 8$), XLA compilation of the `jacfwd` chain can take minutes. 

**Proposed Action:**
- Investigate if `jax.lax.stop_gradient` can be used to prevent redundant tracing in the horizontal derivative terms.
- Consider implementing an analytical expansion of $R^i_{jk}$ for Riemannian metrics to avoid the generic Finsler pipeline when possible.
- Provide a `jax.checkpoint`-wrapped variant of the curvature module to trade recomputation for memory pressure.

---

## [PERFORMANCE] Batched Transport for Jacobian Frames
**Component:** `ham.geometry.transport`
**Status:** Open

**Description:**
The current `parallel_transport` API accepts a single initial vector `vec_start`. While users can use `jax.vmap` over `vec_start` to transport an entire frame (multiple vectors) along the *same* path, this may cause JAX to redundantly compute the Christoffel symbols $\Gamma^i_{jk}$ for each vector in the batch. 

**Proposed Action:**
- Investigate whether XLA optimization and `jax.vmap` successfully hoists the $\Gamma$ computation out of the vector batch dimension.
- If not, refactor `parallel_transport` to accept `vec_start` of shape `(..., D)` and contract the einsum over arbitrary batch dimensions to ensure $\Gamma$ is computed strictly once per path point.

---

## [FEATURE] Explicit Path Parameterization
**Component:** `ham.geometry.transport`
**Status:** Open

**Description:**
`parallel_transport` assumes the input discrete curve `path_x` is parameterized uniformly over $t \in [0, 1]$. It hardcodes the time step as `dt = 1.0 / (len(path_x) - 1)`. If a path is generated with non-uniform time steps (e.g., via adaptive step size ODE solvers) or over a different time interval, the integration will be physically scaled incorrectly.

**Proposed Action:**
- Add an optional `t_array` or explicit `dt` array parameter to `parallel_transport`.

---

## [RESEARCH] Tikhonov Regularization Bias in Berwald Connection
**Component:** `ham.geometry.transport` / `ham.geometry.metric`
**Status:** Open

**Description:**
The Berwald connection $\Gamma$ is computed by taking the Hessian of `metric.spray` w.r.t velocity. The spray internally uses a Tikhonov regularization term (`spray_reg * I`) inside a linear solve. While differentiating through `jnp.linalg.solve` is fully supported by JAX, the regularization introduces an artificial shift in $G^i$. 

**Proposed Action:**
- Mathematically quantify the Tikhonov bias $\Delta \Gamma$ near the degeneracy boundary (e.g., when the Randers wind norm $\|W\|_h \to 1$).
- Determine if the condition number of the regularized Hessian needs to be actively monitored/clipped during the double `jacfwd` differentiation step.

---

## ~~[ENHANCEMENT] Test Transport with Non-Zero Intrinsic Connection~~
**Component:** `tests/test_transport.py`
**Status:** ✅ Resolved

**Resolution:**
Implemented `test_poincare_half_plane_transport` which defines a Poincaré half-plane metric ($ds^2 = (dx^2+dy^2)/y^2$) and:
1. Verifies the Christoffel symbols match the analytic values ($\Gamma^1_{12} = -1/y$, $\Gamma^2_{11} = 1/y$, etc.) within Tikhonov regularization tolerance.
2. Transports a vector along a vertical geodesic and compares against the analytic solution $X(t) = X(0) e^t$.
3. Confirms metric norm preservation ($\|X\|_g = 1$ at all points).
4. Includes a sanity check proving that $\Gamma = 0$ would give the wrong answer.

---

## ~~[DOCUMENTATION] Holonomy Angle Convention~~
**Component:** `tests/test_transport.py` / `spec/MATH_SPEC.md`
**Status:** ✅ Resolved

**Resolution:**
Added `MATH_SPEC.md § 3.3` documenting the ambient vs. intrinsic holonomy convention:
- Projection-based transport ($\Gamma = 0$) produces angle $2\pi\cos\theta$.
- Intrinsic Levi-Civita ($\Gamma \neq 0$) produces angle $2\pi(1 - \cos\theta)$.
- Both are equivalent modulo $2\pi$. The test docstrings now explicitly explain which mechanism is active.
---

## ~~[BUG] Curvature Module Bugs and Test Gaps~~
**Component:** `ham.geometry.curvature`
**Status:** ✅ Resolved

**Resolution:**
- Replaced `jnp.linalg.norm` with `safe_norm` to prevent NaN gradients in `flag_curvature_sample`.
- Replaced Euclidean Gram-Schmidt with metric-aware Gram-Schmidt using `metric.inner_product`.
- Renamed `scalar_curvature` to `flag_curvature_sample` (with backward-compat alias) to reflect direction-dependence in Finsler geometry.
- Added 17 exhaustive tests in `tests/test_curvature.py` covering zero curvature, curved spaces, antisymmetry, Euler homogeneity, and JAX transforms.
- Formalized the PRNG key API for stochastic curvature sampling.
