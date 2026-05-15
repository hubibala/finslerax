# Documentation Review: `src/ham/utils/math.py`

**Reviewer:** Doc Reviewer Agent
**Date:** 2026-05-15

## Summary

Overall documentation quality: **needs work**.

The module is small (27 lines) and exports five public symbols: four canonical epsilon constants and one function (`safe_norm`). The function has a serviceable docstring, but it lacks Args/Returns documentation, mathematical context linking it to the spec's epsilon regularisation strategy (`spec/MATH_SPEC.md § 6.1`), and type annotations in the docstring. The four constants have only inline comments — they are **not** individually documented with any prose explaining their intended downstream use or how they relate to the spec. The module itself has no module-level docstring. Given that this module is the **canonical** source of numerical stability primitives (referenced by `spec/ARCH_SPEC.md § 5`), the documentation gap is significant.

## Issue Tracker

| # | Severity | Symbol / Section | Location | Issue | Suggested Text |
|---|----------|-----------------|----------|-------|----------------|
| 1 | **MISSING** | Module docstring | `src/ham/utils/math.py:1` | No module-level docstring. This module is the canonical repository for numerical stability constants and primitives (`spec/ARCH_SPEC.md § 5`). Users browsing `help(ham.utils.math)` or auto-generated API docs will see no description. | `"""Canonical numerical stability primitives for HAMTools.\n\nThis module centralises all epsilon constants and gradient-safe\nnumerical operations used throughout the library.  Every constant\ncorresponds to a specific stability role described in\nspec/MATH_SPEC.md § 6 and should be imported rather than\nre-defining magic numbers in downstream modules.\n"""` |
| 2 | **MISSING** | `GRAD_EPS` | `src/ham/utils/math.py:7` | Exported public constant has no docstring — only an inline comment. No link to the mathematical motivation ($F_\epsilon(x,v) = \sqrt{F^2 + \epsilon^2}$, `spec/MATH_SPEC.md § 6.1`). ML engineers may not understand *why* `1e-12` is chosen or *where* to use it vs. `NORM_EPS`. | Add a block comment or a `GRAD_EPS: float` annotation with a docstring explaining: "Guard for `jnp.sqrt` backward pass at zero. Used inside `safe_norm`. Ref: `spec/MATH_SPEC.md § 6.1`." |
| 3 | **MISSING** | `NORM_EPS` | `src/ham/utils/math.py:8` | Same as #2. The inline comment "Epsilon for norm-based comparisons/thresholds" does not distinguish it from `GRAD_EPS` for a newcomer. | Clarify: "Threshold for deciding whether a vector is numerically zero (e.g., `norm < NORM_EPS` ⟹ degenerate). Used in forward-pass branching, **not** in backward-pass guards (use `GRAD_EPS` for that)." |
| 4 | **MISSING** | `PSD_EPS` | `src/ham/utils/math.py:9` | Exported but **never imported** by any code in `src/ham/` outside of `__init__.py`. Multiple code review findings (`reviews/code/metric.md`, `reviews/code/networks.md`, `reviews/code/learned.md`, `reviews/code/zoo.md`) flag that downstream modules hardcode `1e-4` instead of importing `PSD_EPS`. The documentation should explicitly state that this is the canonical regularisation constant for $G + \varepsilon I$ and that all PSD regularisation should reference it. | Add prose: "Canonical positive-definite regularisation floor: $G \leftarrow G + \texttt{PSD\_EPS} \cdot I$. All modules that regularise metric matrices should import this constant rather than hardcoding `1e-4`." |
| 5 | **MISSING** | `TAYLOR_EPS` | `src/ham/utils/math.py:10` | No documentation beyond the inline comment. Heavily used in `surfaces.py` for Taylor-series fallback switching but the comment does not explain this role. | Add: "When a quantity (e.g. geodesic distance, angle) is below `TAYLOR_EPS`, implementations switch to a Taylor expansion to avoid catastrophic cancellation. See `surfaces.py` `Sphere.log_map`, `Hyperboloid.log_map`." |
| 6 | **MISSING** | `safe_norm` — Args section | `src/ham/utils/math.py:17-25` | The docstring describes the *what* and *why* but omits structured Args/Returns/Raises documentation. For a function imported in 7+ modules across the codebase, this is a notable gap. | **Recommended Action:** Add an Args/Returns block: `Args:\n    x: Input array of arbitrary shape.\n    axis: Axis along which to compute the norm. Default -1.\n    keepdims: Whether to keep the reduced axis. Default False.\n    eps: Guard epsilon; defaults to GRAD_EPS (1e-12).\n\nReturns:\n    Array of L2 norms with the same dtype as x.` |
| 7 | **UNCLEAR** | `safe_norm` — math context | `src/ham/utils/math.py:17-25` | The docstring says "gradient-safe L2 norm" but does not explain *why* gradients of `jnp.sqrt` are problematic at zero or how the `max(·, eps)` pattern fixes it. A mathematician may wonder whether this changes the forward value; an ML engineer may not realise this is specifically about JAX's autodiff. | Add one sentence: "Without the guard, `jax.grad(jnp.sqrt)(0.0)` returns `NaN` because the derivative $1/(2\sqrt{x})$ diverges at $x=0$. The `max` clamp does not affect the forward value for any $\|x\| > \sqrt{\epsilon} \approx 10^{-6}$." |
| 8 | **INACCURATE** | `safe_norm` — notation vs. implementation | `src/ham/utils/math.py:22` | The docstring says `sqrt(max(sum(x²), eps))`. The spec (`spec/MATH_SPEC.md § 6.1`) defines the regularisation as $F_\epsilon = \sqrt{F^2 + \epsilon^2}$ (additive, not max-based). These are different strategies: `max` clamps while the spec uses additive smoothing. The docstring should acknowledge this distinction so readers don't confuse the two. | Add: "Note: this uses a `max`-clamp strategy, not the additive $\sqrt{F^2 + \epsilon^2}$ regularisation from `spec/MATH_SPEC.md § 6.1`. The additive variant is applied at the metric level (see `losses.py`)." |
| 9 | **UNCLEAR** | `safe_norm` — default `eps` | `src/ham/utils/math.py:17` | The default `eps=GRAD_EPS` is `1e-12`. For `float32` (JAX default), the machine epsilon is ~`1.2e-7`, so `1e-12` is well below the precision floor for squared sums. The docstring does not state the expected dtype or whether `1e-12` is suitable for `float32` vs. `float64`. | Add: "The default `eps=1e-12` is chosen for `float64`. For `float32` workloads, consider `eps=1e-8` or use `NORM_EPS`." |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|--------------|-----------------|-------------------|---------------|---------|
| `GRAD_EPS` | Inline comment only | N/A (constant) | N/A | No | No |
| `NORM_EPS` | Inline comment only | N/A (constant) | N/A | No | No |
| `PSD_EPS` | Inline comment only | N/A (constant) | N/A | No | No |
| `TAYLOR_EPS` | Inline comment only | N/A (constant) | N/A | No | No |
| `safe_norm` | Yes (partial) | No | No | Partial (`sqrt(max(...))`) | No |

## Spec Alignment Notes

1. **`spec/ARCH_SPEC.md § 5`** lists `utils/math.py` as providing "safe_norm, numerical stability primitives." The module fulfils this role, but the documentation does not explicitly call out its canonical status — downstream modules frequently re-invent constants rather than importing from here (evidenced by code review findings in `reviews/code/zoo.md`, `reviews/code/networks.md`, `reviews/code/learned.md`, `reviews/code/metric.md`). Stronger documentation of the canonical role would help enforce the spec's intent.

2. **`spec/MATH_SPEC.md § 6.1`** defines epsilon regularisation as $F_\epsilon = \sqrt{F^2 + \epsilon^2}$. The `safe_norm` implementation uses `sqrt(max(sum(x²), eps))` instead. These are mathematically distinct: the additive form smooths the function everywhere, while the `max`-clamp is a hard threshold. Neither the docstring nor the module docs clarify this distinction or explain when each strategy is appropriate. This is the most important spec alignment gap.

3. **`spec/MATH_SPEC.md § 6.2`** describes homogeneity enforcement via $F_{net}(x,v) = \|v\| \cdot \text{NN}(x, v/\|v\|)$, which relies on a safe norm. The connection between `safe_norm` and this pattern is not documented anywhere in `math.py`.

4. **`PSD_EPS`** is exported but unused in source code — it exists only in the constant definition and re-exports. The spec and code reviews both confirm it *should* be the canonical source for PSD regularisation, but adoption has not occurred. The documentation should note this intended role prominently.
