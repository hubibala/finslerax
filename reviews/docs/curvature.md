# Documentation Review: `src/ham/geometry/curvature.py`

**Reviewer:** Doc Reviewer Agent
**Date:** May 15, 2026

---

## Summary

Overall documentation quality: **needs work**.

`curvature.py` exposes four functions—two publicly via `__init__.py` (`sectional_curvature`, `scalar_curvature`) and two as module-level helpers (`nonlinear_connection`, `riemann_curvature_tensor`). All four have minimal docstrings that omit `Args`, `Returns`, `Raises`, and audience-appropriate explanations. The module itself lacks a module-level docstring. Neither `spec/MATH_SPEC.md` nor `spec/ARCH_SPEC.md` documents curvature quantities, creating a spec-coverage gap. There are no dedicated tests and no example scripts for curvature, so users have no tutorial entry-point.

---

## Issue Tracker

| # | Severity | Symbol / Section | Issue | Suggested Text |
|---|----------|-----------------|-------|----------------|
| 1 | **MISSING** | Module-level | `src/ham/geometry/curvature.py:1` — No module-level docstring explaining the purpose of the module, the mathematical objects it computes, or how it fits into the HAM geometry stack. | **Recommended Action:** Add a module docstring, e.g.: `"""Finsler curvature utilities. Computes the nonlinear connection, Riemann curvature tensor, sectional curvature and scalar curvature from a FinslerMetric, using automatic differentiation of the geodesic spray. All objects are derived from the spray coefficients G^i(x,v) defined in spec/MATH_SPEC.md § 2."""` |
| 2 | **MISSING** | `nonlinear_connection` | `src/ham/geometry/curvature.py:5` — No `Args`, `Returns`, or shape documentation. The one-line docstring states the formula but does not explain what the nonlinear connection *is* for either audience (geometers or ML engineers). | **Recommended Action:** Expand docstring: `"""Computes the nonlinear connection N^i_j = ∂G^i / ∂v^j.\n\nThe nonlinear connection characterises how the geodesic spray varies with direction. It is the building block for the Finsler Riemann curvature tensor.\n\nArgs:\n    metric: A FinslerMetric instance providing the spray.\n    x: Position on the manifold, shape (D,).\n    v: Tangent vector at x, shape (D,).\n\nReturns:\n    N: The nonlinear connection matrix, shape (D, D), where N[i, j] = ∂G^i/∂v^j.\n"""` |
| 3 | **MISSING** | `riemann_curvature_tensor` | `src/ham/geometry/curvature.py:11` — Missing `Args` and `Raises` documentation. No computational-audience explanation of what the tensor measures or how it is used downstream. | **Recommended Action:** Add `Args` section (same signature as `nonlinear_connection`) and a prose sentence: *"Measures how infinitesimally nearby geodesics diverge — the Finslerian generalisation of the Riemannian Riemann tensor."* |
| 4 | **UNCLEAR** | `riemann_curvature_tensor` | `src/ham/geometry/curvature.py:14-16` — The displayed formula uses mixed notation (some terms with superscripts, some without). The inline comment on line 39 (`# N is (l, j) ? No, N^i_j means N[i, j].`) reads as development scratchpad rather than documentation. | **Recommended Action:** Remove the questioning comment and replace with a clean index legend: `# N[i, j] ≡ N^i_j; dN_dv[i, k, l] ≡ ∂N^i_k/∂v^l`. |
| 5 | **MISSING** | `sectional_curvature` | `src/ham/geometry/curvature.py:50` — Missing `Args` section. Parameters `metric`, `x`, `v1`, `v2` are undocumented. The docstring does not explain that in Finsler geometry $K$ depends on the flag pole direction $v_1$, which distinguishes it from the Riemannian case — this is critical for the mathematician audience. | **Recommended Action:** Add `Args` block and note: *"Unlike Riemannian sectional curvature, the Finsler flag curvature K(x, v1, v2) depends on the choice of flag pole v1. The curvature tensor R^i_{jk} is evaluated at (x, v1)."* |
| 6 | **UNCLEAR** | `sectional_curvature` | `src/ham/geometry/curvature.py:56` — Formula in docstring uses $g_{im}$ but the code calls `metric.inner_product(x, v1, R_i, v2)` which computes `w1^T H_v(E) w2` — this is $g_{ij}(x,v_1) R^i v_2^j$, i.e., the contraction via the fundamental tensor. The index-heavy formula and the code use different notation paths, which may confuse readers trying to verify correctness. | **Recommended Action:** Add a bridge sentence: *"Numerically, the numerator is computed as `inner_product(x, v1, R_contracted, v2)` where `R_contracted^i = R^i_{jk} v1^j v2^k`, which is equivalent to the contraction $g_{im} R^m_{jk} v_1^j v_2^k v_2^i$."* |
| 7 | **MISSING** | `sectional_curvature` | `src/ham/geometry/curvature.py:50` — No `Returns` documentation. The return value is a scalar (or scalar array) representing the sectional/flag curvature, but this is not stated. The safe-division behavior (returns `0.0` when denominator < `1e-12`) is also undocumented. | **Recommended Action:** Add `Returns:\n    Scalar flag curvature K. Returns 0.0 when v1 and v2 are nearly collinear (denominator < 1e-12).` |
| 8 | **MISSING** | `scalar_curvature` | `src/ham/geometry/curvature.py:77` — Missing `Args` and `Returns` sections. The docstring says "averaged … averaging sectional curvatures" but the implementation only evaluates a *single* pair of random tangent vectors — it does not average over anything. This is misleading. | **Recommended Action:** Correct the docstring to say *"Approximates scalar curvature by evaluating the sectional curvature for a single random orthonormal tangent pair (seeded with key 42). This is a diagnostic utility, not a rigorous scalar curvature."* Add `Args: metric, x` and `Returns: Scalar approximate curvature at x.` |
| 9 | **INACCURATE** | `scalar_curvature` | `src/ham/geometry/curvature.py:80` — Docstring says *"As a simple metric"* — should likely read *"As a simple heuristic"*. The word "metric" here is overloaded and confusing in a geometry library. | **Recommended Action:** Replace "As a simple metric" with "As a simple heuristic". |
| 10 | **UNCLEAR** | `scalar_curvature` | `src/ham/geometry/curvature.py:77` — The hard-coded `PRNGKey(42)` makes the function deterministic but non-configurable. This design choice is not documented — users may expect reproducibility to be caller-controlled. | **Recommended Action:** Document the fixed seed in the docstring, or accept an optional `key` parameter. At minimum note: *"Uses a fixed JAX PRNGKey(42) for reproducibility."* |
| 11 | **MISSING** | `nonlinear_connection`, `riemann_curvature_tensor` | `src/ham/geometry/__init__.py:7` — These two functions are **not** exported in `__all__`, yet they are importable from `ham.geometry.curvature`. Their public/private status is ambiguous. No leading underscore signals internal use. | **Recommended Action:** Either (a) export them and document them, or (b) prefix with `_` to signal internal use. |
| 12 | **MISSING** | All functions | No docstring examples or pointers to example scripts in any of the four functions. `spec/ARCH_SPEC.md § 5` does not list `curvature.py` in the module tree. | **Recommended Action:** Add at minimum a `See Also` or `Example` block in `sectional_curvature` showing usage with a `Riemannian` or `Euclidean` metric from `ham.geometry.zoo`. |
| 13 | **TYPO** | `riemann_curvature_tensor` | `src/ham/geometry/curvature.py:17` — Docstring says `R[i, j, k] corresponds to R^i_{jk}` but does not clarify whether `j` and `k` are antisymmetric or which pair carries the skew-symmetry. In standard Finsler literature the tensor is $R^i{}_k = R^i{}_{jk} v^j$ (flag curvature form); the full 3-index object should note its skew property in $j \leftrightarrow k$. | **Recommended Action:** Add: *"R is antisymmetric in the last two indices: R^i_{jk} = −R^i_{kj}."* |
| 14 | **UNCLEAR** | `nonlinear_connection` | `src/ham/geometry/curvature.py:5-8` — For ML engineers unfamiliar with spray coefficients, the docstring gives no intuition. A one-sentence bridge like *"Think of it as the Jacobian of the geodesic acceleration with respect to velocity"* would help. | **Recommended Action:** Add a computational-audience sentence after the formula. |

---

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---|---|---|---|---|---|
| `nonlinear_connection` | Yes (1 line) | No | No | Partial (formula only) | No |
| `riemann_curvature_tensor` | Yes (multi-line) | No | Partial (shape only) | Yes (full formula) | No |
| `sectional_curvature` | Yes (multi-line) | No | No | Yes (formula) | No |
| `scalar_curvature` | Yes (multi-line) | No | No | No | No |

---

## Spec Alignment Notes

1. **`spec/MATH_SPEC.md`** — Does not contain a section on curvature. The nonlinear connection $N^i_j$ is not defined there, nor is the Riemann curvature tensor $R^i{}_{jk}$, sectional curvature, or scalar curvature. The curvature module therefore has **no spec backing**. This is a documentation gap at the spec level: `curvature.py` implements meaningful geometric objects that should be specified.
   - **Recommended Action:** Add a section to `MATH_SPEC.md` (e.g., § 7 "Curvature") defining $N^i_j = \partial G^i / \partial v^j$ and the Finsler Riemann tensor $R^i{}_{jk}$ in terms of the nonlinear connection, consistent with the implementation.

2. **`spec/ARCH_SPEC.md § 5`** — The module tree omits `curvature.py` entirely. It is listed neither in the source tree diagram nor in the "Completed & Validated" status section (§ 6).
   - **Recommended Action:** Add `├── curvature.py  # Finsler curvature: nonlinear connection, Riemann tensor, sectional & scalar curvature` to the module tree, and note its validation status.

3. **`spec/ARCH_SPEC.md § 1`** — States *"everything else (geodesics, curvature, transport) is auto-differentiated from [the Metric]"* — this is consistent with the implementation (curvature is derived from `metric.spray`), but the claim is not elaborated anywhere in the architecture spec.

4. **Tests** — No dedicated test file `tests/test_curvature.py` exists. The only curvature mention in tests is an inline comment in `tests/test_transport.py:69`. This means the two publicly exported functions (`sectional_curvature`, `scalar_curvature`) have **no test coverage** visible from the test suite.
