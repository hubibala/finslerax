# Documentation Review: `src/ham/sim/fields.py`
**Reviewer:** Doc Reviewer Agent
**Date:** May 15, 2026

## Summary
Overall documentation quality: **needs work**.

The module provides six public functions for generating analytical vector fields (spherical and planar). Four of the six functions have incomplete docstrings — missing `Args`, `Returns`, or both. Two functions (`lamb_oseen_vortex`, `rankine_vortex`) have only a single-line docstring with no parameter documentation at all. Mathematical formulas are either absent or use coordinate conventions (lat/lon) that differ from the Cartesian implementation, which will confuse both target audiences. The module also lacks a module-level docstring, and there is no mention of these functions in `spec/MATH_SPEC.md`, making it impossible to verify spec alignment. The ARCH_SPEC lists the file only as "Field abstractions for sim" (`spec/ARCH_SPEC.md § 5`), providing no further detail.

## Issue Tracker

| # | Severity | Symbol / Section | Issue | Suggested Text |
|---|----------|-----------------|-------|----------------|
| 1 | **MISSING** | Module level (`src/ham/sim/fields.py:1`) | No module-level docstring. Users browsing `help(ham.sim.fields)` get no overview, no listing of available fields, and no explanation of the spherical stream-function construction pattern shared by the first four functions. | `"""Analytical vector fields for simulation on manifolds.\n\nProvides divergence-free spherical flows via the stream-function construction\nv = ∇ψ × x (see get_stream_function_flow), as well as classical 2-D planar\nvortex models (Lamb–Oseen, Rankine).\n"""` |
| 2 | **MISSING** | `get_stream_function_flow` (`src/ham/sim/fields.py:5`) | No `Args` or `Returns` section. The sole required parameter `psi_fn` (its expected signature, input shape, output shape) is undocumented. | Add: `Args:\n    psi_fn: Scalar stream function ψ: ℝ³ → ℝ. Must be a differentiable\n        function accepting a single point x of shape (3,).\n\nReturns:\n    A vector field function (3,) → (3,) that maps a point on the sphere to\n    a divergence-free tangent vector v = ∇ψ × x.` |
| 3 | **UNCLEAR** | `get_stream_function_flow` (`src/ham/sim/fields.py:7–9`) | The docstring says "on the sphere" but does not clarify that $x$ must lie on the **unit** sphere for the tangency guarantee to hold. An ML engineer might call this on arbitrary 3-D points and be surprised. | Append: `Note: tangency (v · x = 0) is guaranteed only when x lies on the unit sphere (|x| = 1).` |
| 4 | **MISSING** | `rossby_haurwitz` (`src/ham/sim/fields.py:39`) | No `Args` or `Returns` section. Parameters `R` (wave number), `omega` (solid-body rotation amplitude), and `K` (wave amplitude) are entirely undocumented. | Add: `Args:\n    R: Azimuthal wave number (integer, default 4).\n    omega: Amplitude of solid-body rotation component.\n    K: Amplitude of the Rossby–Haurwitz wave component.\n\nReturns:\n    Tangential vector field (3,) → (3,) on the unit sphere.` |
| 5 | **INACCURATE** | `rossby_haurwitz` (`src/ham/sim/fields.py:42–43`) | The formula in the docstring uses geographic coordinates (`lat`, `lon`) but the implementation operates in Cartesian (x, y, z). The mapping between them (`z = sin(lat)`, `rho_xy = cos(lat)`, etc.) is never stated. A mathematician reading the docstring cannot directly verify the code. | Either rewrite the formula in Cartesian: `ψ(x) = −ω x₂ + K (x₀² + x₁²)^{R/2} x₂ cos(R·atan2(x₁,x₀))`, or add a coordinate dictionary: `Coordinate map: z ≡ sin(lat), ρ_xy ≡ cos(lat), lon ≡ atan2(y, x).` |
| 6 | **MISSING** | `lamb_oseen_vortex` (`src/ham/sim/fields.py:84`) | Only a one-line docstring. No `Args`, `Returns`, or mathematical formula. Three parameters (`center`, `core_radius`, `circulation`) are undocumented. The function operates in 2-D (optionally extended to higher dims) — this dimensional distinction from the spherical fields above is never stated. | Suggested docstring: `"""2-D Lamb–Oseen vortex (smoothed point vortex).\n\nv_θ(r) = (Γ / 2πr)(1 − exp(−r² / r_c²))\n\nThe velocity is purely azimuthal around 'center' and decays like 1/r\nin the far field, while going smoothly to zero at the origin.\n\nArgs:\n    center: Vortex center position, shape (2,) or (D,); only the first\n        two components are used.\n    core_radius: Viscous core radius r_c.\n    circulation: Total circulation Γ.\n\nReturns:\n    Vector field (D,) → (D,). Output dimension matches input.\n"""` |
| 7 | **MISSING** | `rankine_vortex` (`src/ham/sim/fields.py:99`) | Same problem as `lamb_oseen_vortex`: one-line docstring, no `Args`, `Returns`, or formula. | Suggested docstring: `"""2-D Rankine vortex (solid-body core, irrotational exterior).\n\nv_θ(r) = Γr / (2π r_c²)   for r ≤ r_c  (solid body)\nv_θ(r) = Γ / (2πr)         for r > r_c  (irrotational)\n\nArgs:\n    center: Vortex center, shape (2,) or (D,).\n    core_radius: Core radius r_c separating the two regimes.\n    circulation: Total circulation Γ.\n\nReturns:\n    Vector field (D,) → (D,). Output dimension matches input.\n"""` |
| 8 | **UNCLEAR** | `tilted_rotation` (`src/ham/sim/fields.py:23`) | The docstring says "Constant rotation around a tilted axis" but does not state what the axis direction is or how `alpha_deg` relates to it. A mathematician would expect the rotation axis to be identified explicitly. | Append to the one-liner: `The rotation axis is (sin α, 0, cos α) in ambient ℝ³, where α = alpha_deg. When α = 0 the axis is the north pole (z-axis).` |
| 9 | **UNCLEAR** | `harmonic_vortices` (`src/ham/sim/fields.py:68`) | The docstring mentions "associated Legendre-like structure" but the implementation uses `sin(l π z)` — a sinusoidal approximation, not an actual associated Legendre polynomial. This will mislead a mathematician expecting $P_l^m(\cos\theta)$. | Change "associated Legendre-like structure" to something like: `Cellular vortex flow using a sinusoidal latitudinal profile ψ ∝ cos^m(lat) · sin(lπ·sin(lat)) · cos(m·lon).` |
| 10 | **UNCLEAR** | All spherical fields (`src/ham/sim/fields.py`) | The ARCH_SPEC mandates batch-first design (`(B, ...)`), but every function in this module expects unbatched input of shape `(3,)` or `(2,)`. This is not documented anywhere. An ML engineer following ARCH_SPEC conventions will pass batched tensors and get silent wrong results via broadcasting. | Add a note to the module docstring and/or to `get_stream_function_flow`: `Note: These functions operate on single points, not batched inputs.\nUse jax.vmap to batch.` |
| 11 | **TYPO** | `rossby_haurwitz` (`src/ham/sim/fields.py:43`) | The formula line `v = ∇ψ × x` is redundant — it repeats exactly the `get_stream_function_flow` docstring. This adds clutter without new information. | Remove the redundant `v = ∇ψ × x` line, or replace with: `See get_stream_function_flow for the tangent-vector construction.` |
| 12 | **MISSING** | `rossby_haurwitz` (`src/ham/sim/fields.py:39`) | No reference to the Rossby–Haurwitz wave literature. Both audiences benefit from a citation (the mathematician for verification, the ML engineer for background). | Add: `Reference: Haurwitz, B. (1940). The motion of atmospheric disturbances on the spherical earth. J. Mar. Res., 3, 254–267.` |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:------------:|:---------------:|:-----------------:|:-------------:|:-------:|
| `get_stream_function_flow` | Yes | No | No | Partial ($v = \nabla\psi \times x$) | No |
| `tilted_rotation` | Yes | Yes | Yes | No | No |
| `rossby_haurwitz` | Yes | No | No | Yes (lat/lon formula) | No |
| `harmonic_vortices` | Yes | Yes | Yes | Misleading | No |
| `lamb_oseen_vortex` | One-liner | No | No | No | No |
| `rankine_vortex` | One-liner | No | No | No | No |

## Spec Alignment Notes

1. **ARCH_SPEC § 5** lists `sim/fields.py` as "Field abstractions for sim" but provides no further description. The actual module contains concrete analytical field generators, not abstractions. The ARCH_SPEC description should be updated to reflect the actual content (e.g., "Analytical vector fields for spherical and planar simulation").

2. **ARCH_SPEC § 1 (Batch-First)** — The module violates the batch-first convention without documenting the deviation. All functions operate on single points `(3,)` or `(2,)` rather than batched `(B, D)` tensors.

3. **MATH_SPEC** — The `sim/fields.py` module has no corresponding section in `spec/MATH_SPEC.md`. The stream-function construction ($v = \nabla\psi \times \mathbf{x}$), the Rossby–Haurwitz wave formula, and the vortex models are mathematically non-trivial and would benefit from a dedicated spec section (e.g., § 7 "Simulation Fields") to serve as a single source of truth.

4. **Missing `__init__.py`** — The `src/ham/sim/` directory has no `__init__.py`, relying on implicit namespace packages. While this works, it means `help(ham.sim)` and `dir(ham.sim)` will not surface `fields` as a submodule, making the API less discoverable.
