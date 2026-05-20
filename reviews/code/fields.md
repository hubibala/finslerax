# Code Review: `ham.sim.fields`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`fields.py` provides a clean set of analytical vector-field generators used in simulation demos.
The stream-function approach for sphere flows is well designed and inherently guarantees divergence-free tangent fields.
However, there are two concrete risks: an incorrect safe-normalization pattern that shifts the vector direction instead of guarding the denominator, and the absence of an `__init__.py` for the `sim/` package, which breaks consistency with every other HAM subpackage.
No function supports a leading batch dimension, violating the ARCH_SPEC batch-first convention, and two public functions (`rossby_haurwitz`, `harmonic_vortices`) lack any test coverage.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **RISK** | `src/ham/sim/fields.py:32` | Safe-normalization pattern is incorrect: `axis / jnp.linalg.norm(axis + 1e-10)`. The epsilon is added **to the vector** before computing the norm, which shifts the direction of `axis` rather than preventing a zero denominator. For this particular call site the input is always `[sin(α), 0, cos(α)]` (norm = 1), so the error is negligible (~$10^{-10}$), but the pattern is wrong and dangerous if copied elsewhere. | Replace with `axis / jnp.maximum(jnp.linalg.norm(axis), 1e-10)`, or simply remove the normalization since $\sin^2\alpha + \cos^2\alpha = 1$ guarantees unit norm. |
| 2 | **RISK** | `src/ham/sim/fields.py` (module) | The `src/ham/sim/` directory has **no `__init__.py`**. Every other subpackage under `src/ham/` (`geometry/`, `bio/`, `models/`, `nn/`, `solvers/`, `training/`, `utils/`, `vis/`) has one. The package currently works as an implicit namespace package (PEP 420), but this is inconsistent and may cause issues with certain packaging tools, editable installs, or IDE resolution. | Add an `__init__.py` to `src/ham/sim/` that re-exports the public API (e.g., the six factory functions). |
| 3 | **STYLE** | `src/ham/sim/fields.py:1–148` | None of the six public functions support a leading batch dimension `(B, ...)`. They all operate on single points via `reshape(-1)`. This violates the ARCH_SPEC § 1 batch-first principle. The returned closures are `vmap`-compatible (no side-effects, statically-shaped branches), so callers can `vmap` manually, but the convention is still breached. | Either document that these are single-point functions designed for `vmap` wrapping, or add explicit batch handling. |
| 4 | **STYLE** | `src/ham/sim/fields.py:14,37,49,82` | Multiple ad-hoc epsilon values (`1e-10`) are hardcoded throughout the module. `src/ham/utils/math.py` defines canonical constants (`GRAD_EPS = 1e-12`, `NORM_EPS = 1e-8`). Using inconsistent magic numbers undermines the consolidation goal stated in that module. | Import and use the canonical constants from `ham.utils.math`. |
| 5 | **STYLE** | `src/ham/sim/fields.py:73` | Parameter name `l` (lowercase L) in `harmonic_vortices(l=5, m=3)` is visually indistinguishable from `1` in many fonts. PEP 8 discourages single-character `l` for this reason. | Rename to `degree` or `ell`. |
| 6 | **STYLE** | `src/ham/sim/fields.py:108,112,132,136` | `if x.shape[0] > 2:` uses Python-level control flow to conditionally append zero components. While `x.shape` is statically known at JAX trace time (so JIT/grad/vmap work correctly), it couples the function's output shape to a silent runtime convention. | Consider using `jnp.zeros_like(x).at[:2].set(jnp.array([v_x, v_y]))` for clarity and uniform output shape, or document the shape contract explicitly. |

---

## Test Coverage Assessment

| Public Function | Tested? | Notes |
|---|---|---|
| `get_stream_function_flow` | **Yes** | Basic equatorial-rotation test (`ψ = z`). |
| `tilted_rotation` | **Yes** | Only `alpha_deg=0.0`. Does not test non-trivial tilt angles. |
| `rossby_haurwitz` | **No** | No test at all — gap. |
| `harmonic_vortices` | **No** | No test at all — gap. |
| `lamb_oseen_vortex` | **Yes** | Far-field and near-origin tests; 3D extension untested. |
| `rankine_vortex` | **Yes** | Inside/outside core boundary tested; 3D extension untested. |

**Gap analysis:**

- `rossby_haurwitz` and `harmonic_vortices` are entirely untested. Recommended action: add at least a smoke test (JIT-compiles, returns finite values, tangent to sphere).
- No test verifies `jax.jit` or `jax.vmap` compatibility for any field function.
- No test checks `jax.grad` through a field (e.g., differentiating a trajectory loss w.r.t. field parameters).
- The 3D-extension branches (`if x.shape[0] > 2`) of `lamb_oseen_vortex` and `rankine_vortex` are never exercised.
- `tilted_rotation` is only tested at the degenerate angle `0°`; a test at a non-trivial angle (e.g., `45°`) would exercise the full code path including the normalization on line 32.

---

## Positive Patterns

1. **Stream-function construction** (`get_stream_function_flow`): using $\mathbf{v} = \nabla\psi \times \mathbf{x}$ to generate divergence-free, sphere-tangent flows from an arbitrary scalar function is elegant and automatically correct. Auto-differentiating `psi_fn` via `jax.grad` avoids any manual gradient algebra.
2. **Complex-number trick for azimuthal phase**: computing $\cos(R\,\text{lon})$ via `jnp.real((x+iy)^R / |x+iy|^R)` avoids `atan2` and its branch-cut issues, and is fully differentiable. Used in both `rossby_haurwitz` and `harmonic_vortices`.
3. **Closure-based API**: the factory-function pattern (e.g., `lamb_oseen_vortex(center, ...) → Callable`) keeps the user-facing API clean and composable — fields can be added, scaled, or passed to solvers without wrapping in a class.
4. **Lamb-Oseen stability at origin**: `r_sq + 1e-10` prevents division by zero, and the `(1 - exp(-r²/a²))` envelope naturally drives velocity to zero near the center, so the epsilon only guards the gradient, not the forward value.
