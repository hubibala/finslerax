# HAM Verification Report

**Purpose.** This document records independent, closed-form correctness checks of
the HAM core: places where the geometry has a *known analytic answer* and the
library's autodiff / solver output is compared against it. Reference values are
textbook formulas (Christoffel symbols, Gaussian curvature, Zermelo travel times,
Poincaré distance) computed independently of the library.

**Reproduce.** All numbers below are produced by a single script:

```
JAX_PLATFORMS=cpu python spec/verify.py
```

Hardware: CPU, float32 (the library's default dtype). Errors are reported as
absolute deviation from the closed-form reference. Last run: all checks pass.

---

## 1. Geodesic spray — Poincaré upper half-plane

Metric `g = I / y²`. The geodesic acceleration `aᵏ = −Γᵏ_ij vⁱvʲ` with the
Poincaré Christoffel symbols (`Γˣ_xy = −1/y`, `Γʸ_xx = 1/y`, `Γʸ_yy = −1/y`) gives,
at `x = (0, 1)`, `v = (1, 0.5)`: `aˣ = 1.0`, `aʸ = −0.75`. The library's spray `G`
satisfies `a = −2G`.

| quantity | library | analytic | abs err |
|---|---|---|---|
| acceleration `aˣ` | +0.999900 | +1.000000 | 1.0e-4 |
| acceleration `aʸ` | −0.749925 | −0.750000 | 7.5e-5 |

Validates the implicit Euler–Lagrange spray solve (`geometry/metric.py`).

## 2. Randers / Zermelo cost — directional asymmetry

Identity sea, constant wind `w = 0.5` along `+x`. Zermelo time over a unit step is
`1/(1+w)` downwind, `1/(1−w)` upwind, `1/√(1−w²)` crosswind.

| direction | library | analytic | abs err |
|---|---|---|---|
| east (downwind) | +0.666667 | +0.666667 | 2.0e-8 |
| west (upwind) | +2.000000 | +2.000000 | 0.0 |
| north (crosswind) | +1.154701 | +1.154701 | 0.0 |

Confirms the asymmetric drift sign convention ("headwind increases cost",
MATH_SPEC §5).

## 3. Flag / sectional curvature — autodiff vs Gaussian curvature

Curvature is obtained by 4th-order autodiff of the energy and compared to the
closed-form Gaussian curvature `K`.

| metric | library `K` | analytic `K` | abs err |
|---|---|---|---|
| stereographic sphere `4/(1+|x|²)² I` | +0.999900 | +1.0 | 1.0e-4 |
| surface of revolution `diag(1, 1+x²)`, x=0.5 | −0.639941 | −0.64 | 5.9e-5 |
| Poincaré `I/y²` | −1.000000 | −1.0 | 1.2e-7 |

Both **signs** and **magnitudes** of curvature are recovered. (These replace an
earlier test that only asserted `|K| > 1e-4`, which could not distinguish sign or
magnitude — see `tests/test_curvature.py`.)

## 4. Causal wind clamp — identity in region, strict bound above

The Zermelo strong-convexity bound `‖W‖_H < 1` is enforced by the
identity-preserving soft-min `causal_wind_scale` (`utils/math.py`, MATH_SPEC §5.1),
which replaced the `tanh` squash that bent every wind (`0.5 → 0.46`).

| input `‖W‖` | clamped `‖W‖` | note |
|---|---|---|
| 0.5 | 0.500000 | identity (err 0.0) |
| 0.9 | 0.898380 | identity to 1.6e-3 (near boundary) |
| 1.5 | 0.999990 | strictly < 1 (strong convexity preserved) |
| 5.0 | 0.999990 | strictly < 1 |

## 5. Gauss–Newton geodesic — boundary-value length

Poincaré endpoints `(−0.5, 1)` and `(0.5, 1)`; analytic distance
`arccosh(1 + Δx²/(2y₁y₂)) = arccosh(1.5) = 0.962424`.

| solver | length | analytic | rel err |
|---|---|---|---|
| GaussNewtonGeodesic (n=64) | 0.962431 | 0.962424 | 7e-6 |

## 6. AVBD on the same BVP — *informational, honest limitation*

AVBD is block Gauss–Seidel gradient descent and converges slowly on stiff metrics
(documented in `spec/AVBD_LATENT_FINDINGS_2026-06-14.md`). On the same Poincaré BVP:

| solver | length | analytic | rel gap |
|---|---|---|---|
| GaussNewtonGeodesic | 0.962431 | 0.962424 | **0.001 %** |
| AVBDSolver (400 iter) | 0.9991 | 0.962424 | 3.8 % |

This is reported, not asserted — it is the expected behaviour and the reason the
docs recommend GaussNewton (+ continuation) as the primary BVP solver and AVBD as
the local/parallel option.

---

### Summary

| area | status |
|---|---|
| geodesic spray (implicit EL solve) | ✅ analytic match |
| Randers/Zermelo asymmetry & sign | ✅ analytic match |
| curvature (sign + magnitude, both signs) | ✅ analytic match |
| causal wind clamp (identity + strict bound) | ✅ verified |
| Gauss–Newton BVP geodesic | ✅ analytic match |
| AVBD convergence | ⚠️ slow on stiff metrics (documented, by design) |

The core differential-geometry machinery reproduces closed-form answers to
3–7 significant figures on CPU/float32. Solver-quality caveats (AVBD) are
documented rather than hidden.
