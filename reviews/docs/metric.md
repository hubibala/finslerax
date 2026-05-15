# Documentation Review: `src/ham/geometry/metric.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2026-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** [src/ham/geometry/metric.py](src/ham/geometry/metric.py)

## Summary

Overall documentation quality: **needs work**.

The module defines `FinslerMetric`, the central abstraction of the entire library (see `spec/ARCH_SPEC.md` § 2.2). Despite its importance, the file lacks a module-level docstring, three of six public methods have no docstrings, and the existing docstrings omit return types, raises, and shape information. Mathematical notation is mostly absent — the dual audience (differential geometers + ML engineers) cannot read the API reference alone and reconstruct the semantics. The existing `spray()` docstring is the strongest but still omits argument documentation.

---

## Issue Tracker

| # | Severity | Symbol / Section | Location | Issue | Suggested Text |
|---|----------|-----------------|----------|-------|----------------|
| 1 | **MISSING** | Module docstring | [metric.py:1](src/ham/geometry/metric.py#L1) | No module-level docstring. This is the single most important module in the library (`spec/ARCH_SPEC.md` § 2.2: "The heart of the library"). Users browsing `help(ham.geometry.metric)` see nothing. | See §Suggested Text 1 below. |
| 2 | **MISSING** | `energy()` | [metric.py:33–34](src/ham/geometry/metric.py#L33-L34) | No docstring. The Energy functional $E = \tfrac{1}{2}F^2$ is the root of the entire computational graph (`spec/MATH_SPEC.md` § 1.2). Undocumented despite being the foundation for every derived quantity. | See §Suggested Text 2 below. |
| 3 | **MISSING** | `inner_product()` | [metric.py:36–41](src/ham/geometry/metric.py#L36-L41) | No docstring. The function computes $\langle w_1, w_2 \rangle_v = w_1^T\, g_{ij}(x,v)\, w_2$ using the fundamental tensor. This is a non-trivial Finsler concept (the inner product depends on a reference direction $v$), yet is entirely undocumented. `spec/ARCH_SPEC.md` § 2.2 explicitly prescribes a docstring for this method. | See §Suggested Text 3 below. |
| 4 | **MISSING** | `geod_acceleration()` | [metric.py:68–69](src/ham/geometry/metric.py#L68-L69) | No docstring. The relationship $\ddot{x} = -2G$ (`spec/MATH_SPEC.md` § 2.1) is critical for users who want to integrate the geodesic ODE but do not know the sign/scaling convention. | See §Suggested Text 4 below. |
| 5 | **UNCLEAR** | `FinslerMetric` class docstring | [metric.py:9–13](src/ham/geometry/metric.py#L9-L13) | Mentions `eqx.Module` / JAX PyTrees but says nothing about the mathematical role. An ML engineer unfamiliar with Finsler geometry gets no orientation; a geometer gets no computational guidance. `spec/ARCH_SPEC.md` § 2.2 specifies this class "Defines the geometry via the Finsler energy function." | Expand to: *"Abstract base class for all Finsler metrics. Defines the geometry of a manifold via the fundamental cost function $F(x, v)$ and its derived energy $E = \\tfrac{1}{2}F^2$. All downstream geometry (geodesic spray, curvature, parallel transport) is auto-differentiated from `metric_fn`. Inherits from `eqx.Module` so subclasses are valid JAX PyTrees."* |
| 6 | **UNCLEAR** | `metric_fn()` docstring | [metric.py:19–30](src/ham/geometry/metric.py#L19-L30) | Missing return shape annotation. For an ML engineer, knowing the output is a scalar (shape `()`) vs. a batch (`(B,)`) is essential. `spec/ARCH_SPEC.md` § 1 states "Batch-First: All operations assume a leading batch dimension `(B, ...)`", yet the signature and docstring are silent on batching. | Add to Returns: *"Scalar Finsler cost $F(x, v) \geq 0$. Shape: `()`."* Add a note on batching: *"Implementations may be vmapped externally; this method operates on single points."* |
| 7 | **UNCLEAR** | `spray()` docstring | [metric.py:43–48](src/ham/geometry/metric.py#L43-L48) | Docstring documents the *equation solved* but omits `Args`, `Returns`, `Raises`. A user cannot determine from the docstring what the return value represents (spray coefficients $G^i$) or its shape. | See §Suggested Text 5 below. |
| 8 | **INACCURATE** | `spray()` docstring | [metric.py:46](src/ham/geometry/metric.py#L46) | Docstring says `Hess_v(E) * (-2G) = …`. Per the math review ([reviews/math/metric.md](reviews/math/metric.md)) and the code itself (`acc = solve(hess_v, rhs); return -0.5 * acc`), the system solved is $\text{Hess}_v(E)\cdot\text{acc} = \text{rhs}$ where $\text{acc} = -2G$. The docstring omits the crucial minus sign: the system is $\text{Hess}_v(E)\cdot(-2G) = \nabla_x E - \text{Jac}_x(\nabla_v E)\cdot v$. This happens to match the code, but the sign convention in `spec/MATH_SPEC.md` § 2.2 writes $\text{Hess}_v(E)\cdot(2G) = …$ (positive $2G$, different RHS sign), which contradicts both the code and this docstring. The docstring should explicitly state the convention used. | Clarify to: *"Solves the linear system $\\text{Hess}_v(E)\\cdot(-2G) = \\nabla_x E - \\text{Jac}_x(\\nabla_v E)\\cdot v$ for the spray coefficients $G^i$, per `spec/MATH_SPEC.md` § 2.2."* |
| 9 | **UNCLEAR** | `spray()` inline comment | [metric.py:63](src/ham/geometry/metric.py#L63) | Comment says *"Regularize hessian slightly to avoid singular matrices (Randers ill-conditioning near boundary)"*. This is helpful context, but the hardcoded `1e-4` Tikhonov constant is undocumented in the docstring. Users and downstream reviewers need to know this modifies the true spray. | Add a `Note` section to the docstring: *"A Tikhonov regularization term $\\epsilon I$ ($\\epsilon = 10^{-4}$) is added to the Hessian to prevent singular-matrix errors near degenerate directions (e.g., Randers metrics near the wind-speed boundary). See `spec/MATH_SPEC.md` § 6.1."* |
| 10 | **UNCLEAR** | `arc_length()` docstring | [metric.py:71–82](src/ham/geometry/metric.py#L71-L82) | Says *"continuous path gamma (N, D)"* — but `gamma` is discrete (an array of waypoints). The word "continuous" is misleading. Also missing: `Args`, `Returns`, and a note that the midpoint rule is first-order. | See §Suggested Text 6 below. |
| 11 | **MISSING** | `arc_length()` — no example | [metric.py:71–82](src/ham/geometry/metric.py#L71-L82) | `arc_length` is a user-facing utility. A one-liner example in the docstring (or a pointer to `examples/demo_trajectories.py`) would help ML engineers. | Add: *"Example: ``metric.arc_length(jnp.linspace(start, end, 50))``"* |
| 12 | **TYPO** | `metric_fn()` docstring | [metric.py:22](src/ham/geometry/metric.py#L22) | Type hint uses `jnp.ndarray` (deprecated alias). JAX recommends `jax.Array`. Minor but inconsistent with modern JAX practice. | Replace `jnp.ndarray` → `jax.Array` throughout the file. |

---

## Suggested Texts

### §Suggested Text 1 — Module docstring

```python
"""
Finsler Metric base class — the core geometric abstraction of HAMTools.

This module defines :class:`FinslerMetric`, from which all concrete
metrics (Euclidean, Riemannian, Randers, learned neural metrics) inherit.
The metric provides the fundamental cost function F(x, v) and its derived
energy E = ½ F². Every downstream geometric object — geodesic spray,
curvature, parallel transport — is auto-differentiated from ``metric_fn``.

Mathematical reference: spec/MATH_SPEC.md §§ 1–2.
Architecture reference: spec/ARCH_SPEC.md § 2.2.

See Also:
    ham.geometry.zoo : Concrete metric implementations.
    ham.geometry.transport : Berwald parallel transport built on this class.
"""
```

### §Suggested Text 2 — `energy()`

```python
def energy(self, x: jax.Array, v: jax.Array) -> jax.Array:
    """
    Lagrangian energy E(x, v) = ½ F²(x, v).

    This scalar is the root of the computational graph: the geodesic
    spray, fundamental tensor, and inner product are all derived from
    automatic differentiation of this function.

    Args:
        x: Position on the manifold. Shape ``(D,)``.
        v: Tangent vector at x. Shape ``(D,)``.

    Returns:
        Scalar energy. Shape ``()``.

    Reference:
        spec/MATH_SPEC.md § 1.2.
    """
```

### §Suggested Text 3 — `inner_product()`

```python
def inner_product(self, x: jax.Array, v: jax.Array,
                  w1: jax.Array, w2: jax.Array) -> jax.Array:
    """
    Finsler inner product <w1, w2>_v using the fundamental tensor g_ij(x, v).

    Computes  w1ᵀ · g(x, v) · w2  where g_ij = ∂²E/∂vⁱ∂vʲ (the Hessian
    of the energy with respect to velocity).  Note: unlike Riemannian
    geometry, the inner product depends on a *reference direction* v.

    Args:
        x: Position on the manifold. Shape ``(D,)``.
        v: Reference tangent direction for the fundamental tensor. Shape ``(D,)``.
        w1: First tangent vector. Shape ``(D,)``.
        w2: Second tangent vector. Shape ``(D,)``.

    Returns:
        Scalar inner product. Shape ``()``.

    Reference:
        spec/MATH_SPEC.md § 1.1 (fundamental tensor);
        spec/ARCH_SPEC.md § 2.2.
    """
```

### §Suggested Text 4 — `geod_acceleration()`

```python
def geod_acceleration(self, x: jax.Array, v: jax.Array) -> jax.Array:
    """
    Geodesic acceleration  ẍⁱ = −2 Gⁱ(x, v).

    Returns the acceleration vector that, combined with velocity v,
    defines the geodesic ODE:  dx/dt = v,  dv/dt = −2G(x, v).

    Args:
        x: Position on the manifold. Shape ``(D,)``.
        v: Velocity (tangent vector). Shape ``(D,)``.

    Returns:
        Acceleration vector. Shape ``(D,)``.

    Reference:
        spec/MATH_SPEC.md § 2.1.
    """
```

### §Suggested Text 5 — `spray()` (expanded)

```python
def spray(self, x: jax.Array, v: jax.Array) -> jax.Array:
    """
    Geodesic spray coefficients Gⁱ(x, v).

    Solves the implicit linear system derived from Euler-Lagrange:
        Hess_v(E) · (−2G) = ∇ₓE − Jacₓ(∇ᵥE) · v

    Args:
        x: Position on the manifold. Shape ``(D,)``.
        v: Velocity (tangent vector). Shape ``(D,)``.

    Returns:
        Spray vector G(x, v). Shape ``(D,)``.

    Note:
        A Tikhonov term ε·I (ε = 1e-4) is added to the Hessian to
        regularize near-degenerate directions (Randers boundary).
        See spec/MATH_SPEC.md § 6.1.

    Reference:
        spec/MATH_SPEC.md § 2.2.
    """
```

### §Suggested Text 6 — `arc_length()` (expanded)

```python
def arc_length(self, gamma: jax.Array) -> jax.Array:
    """
    Approximate Finsler arc length of a discrete path.

    Uses the midpoint rule: each segment [γᵢ, γᵢ₊₁] is evaluated at
    the midpoint with velocity v = γᵢ₊₁ − γᵢ.  This is a first-order
    quadrature; accuracy improves with finer discretization.

    Args:
        gamma: Waypoints of the path. Shape ``(N, D)`` where N ≥ 2.

    Returns:
        Total arc length (scalar). Shape ``()``.

    Example:
        >>> path = jnp.linspace(start, end, num=50)
        >>> length = metric.arc_length(path)
    """
```

---

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:---:|:---:|:---:|:---:|:---:|
| `FinslerMetric` (class) | Yes | N/A | N/A | No | No |
| `metric_fn()` | Yes | Yes | Partial (no shape) | Partial ($F(x,v)$, homogeneity) | No |
| `energy()` | **No** | — | — | — | — |
| `inner_product()` | **No** | — | — | — | — |
| `spray()` | Yes | **No** | **No** | Partial (equation only) | No |
| `geod_acceleration()` | **No** | — | — | — | — |
| `arc_length()` | Yes | Partial (no `Args:`) | **No** | No | No |

---

## Spec Alignment Notes

1. **`spec/ARCH_SPEC.md` § 2.2** prescribes docstrings for `energy()`, `spray()`, and `inner_product()`. The code's `energy()` and `inner_product()` have none; `spray()` has a partial one. The ARCH_SPEC's suggested `spray()` docstring says *"Returns: Acceleration vector"*, but the actual method returns the spray $G^i$, **not** the acceleration $-2G^i$. The code is consistent with the mathematical convention; the ARCH_SPEC description is misleading. **Recommended Action:** Update `spec/ARCH_SPEC.md` § 2.2 `spray()` Returns to read *"Spray coefficients $G^i$"*.

2. **`spec/MATH_SPEC.md` § 2.2** writes the implicit system with a positive $2G$ on the left-hand side. The code and its docstring use $-2G$. This sign discrepancy is already flagged in [reviews/math/metric.md](reviews/math/metric.md) (Finding §3, WARNING). The docstring should note which convention is implemented to avoid confusion for geometers cross-referencing the spec.

3. **`spec/ARCH_SPEC.md` § 1** states *"Batch-First: All operations assume a leading batch dimension"*. The `FinslerMetric` methods all operate on single points (unbatched). No docstring mentions that vmapping is required for batched use. This is a cross-cutting documentation gap; at minimum the class docstring should note the convention.

4. **`spec/MATH_SPEC.md` § 6.1** documents $\epsilon$-regularization of $F$ itself ($F_\epsilon = \sqrt{F^2 + \epsilon^2}$), but the code regularizes the Hessian via Tikhonov (`hess_v + 1e-4 * I`). These are different strategies. The docstring should clarify which is used and why.
