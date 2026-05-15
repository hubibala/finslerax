# Documentation Review: `src/ham/geometry/manifold.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2026-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** [src/ham/geometry/manifold.py](src/ham/geometry/manifold.py)

## Summary

Overall documentation quality: **adequate**.

The `Manifold` base class ([manifold.py](src/ham/geometry/manifold.py)) is the topological foundation of the library (`spec/ARCH_SPEC.md` § 2.1). Its class docstring and abstract method docstrings are reasonably clear and correctly state the "topology, not geometry" design intent. However, the file is missing a module-level docstring, several methods lack shape annotations and `Raises` sections, the private helper `_safe_norm_ratio_jvp` is entirely undocumented from a mathematical standpoint, and the non-trivial `log_map` scaling correction has no spec or literature reference. The type annotations consistently use the deprecated `jnp.ndarray` alias instead of `jax.Array`.

---

## Issue Tracker

| # | Severity | Symbol / Section | Location | Issue | Suggested Text |
|---|----------|-----------------|----------|-------|----------------|
| 1 | **MISSING** | Module docstring | [manifold.py:1](src/ham/geometry/manifold.py#L1) | No module-level docstring. A user running `help(ham.geometry.manifold)` or browsing the source sees only imports. `spec/ARCH_SPEC.md` § 5 lists this as `manifold.py  # Abstract Base Class` but users need orientation text. | See §Suggested Text 1 below. |
| 2 | **UNCLEAR** | `_safe_norm_ratio_jvp` | [manifold.py:9–40](src/ham/geometry/manifold.py#L9-L40) | The function docstring says *"Computes \|\|x\|\| / \|\|y\|\| safely"*, but does not explain **why** a custom JVP is needed, what edge case it protects against ($\|y\| \to 0$), or where it is consumed. A geometer reading the file will not understand the computational motivation; an ML engineer will not understand the mathematical motivation. The JVP derivation in the `defjvp` block (`_safe_norm_ratio_jvp_def`) has inline comments that sketch the quotient rule but do not state the full derivative formula in a way that can be independently verified. | Add a paragraph: *"A custom JVP is registered because the naive quotient $\|x\|/\|y\|$ has an undefined derivative when $\|y\| = 0$. The custom rule clamps the tangent to zero in that regime, ensuring clean AD through `log_map` (see below). The derivative follows the quotient rule: $d(\|x\|/\|y\|) = (\|y\|\,d\|x\| - \|x\|\,d\|y\|) / \|y\|^2$."* |
| 3 | **MISSING** | `_safe_norm_ratio_jvp` — no spec reference | [manifold.py:9](src/ham/geometry/manifold.py#L9) | This function is a numerical stability primitive, yet it is not referenced by `spec/MATH_SPEC.md` § 6 (Numerical Stability) or anywhere in the spec. The relationship to the $\epsilon$-regularization strategy in § 6.1 is unclear. | Add a `Reference` note: *"See `spec/MATH_SPEC.md` § 6 for the general numerical stability strategy. This JVP guard complements the $\epsilon$-regularization of $F$."* |
| 4 | **UNCLEAR** | `Manifold` class docstring | [manifold.py:44–52](src/ham/geometry/manifold.py#L44-L52) | The docstring correctly states the topology-only role and references `ARCH_SPEC.md § 2.1`. However, it does not mention that `Manifold` inherits from `eqx.Module`, which makes subclasses valid JAX PyTrees — a critical detail for ML engineers who need to know they can `jit`/`vmap`/`grad` through manifold operations. | Append: *"Inherits from `eqx.Module`, making all subclasses valid JAX PyTrees composable with `jax.jit`, `jax.vmap`, and `jax.grad`."* |
| 5 | **MISSING** | `project()` — return shape | [manifold.py:68–76](src/ham/geometry/manifold.py#L68-L76) | The `Returns` section says `x_proj: Point on the manifold M` but does not specify the shape. Per `spec/ARCH_SPEC.md` § 1, the library is batch-first, so users need to know whether `project` handles batched inputs. | Change to: *"x_proj: Point on the manifold $M$. Shape: same as input `x`."* |
| 6 | **MISSING** | `to_tangent()` — return shape | [manifold.py:78–89](src/ham/geometry/manifold.py#L78-L89) | Same issue as `project()`. The return shape is not specified. | Add to Returns: *"Shape: same as input `v`."* |
| 7 | **MISSING** | `retract()` — Args / Returns | [manifold.py:91–104](src/ham/geometry/manifold.py#L91-L104) | `retract()` has a detailed mathematical description of the contract (zeroth and first-order conditions) and lists common implementation choices, but lacks `Args` and `Returns` sections entirely. A user cannot determine parameter names, types, or shapes from the docstring. | Add `Args:` (`x: Base point on $M$, shape `(D,)`; `delta: Tangent vector in $T_x M$, shape `(D,)`) and `Returns:` (`Point on $M$, shape `(D,)`). |
| 8 | **UNCLEAR** | `exp_map()` — relationship to `retract()` | [manifold.py:106–112](src/ham/geometry/manifold.py#L106-L112) | Docstring says *"The default implementation falls back to the exact retraction"* — the word "exact" is misleading. A retraction is by definition a first-order approximation of the exponential map, not exact (unless the subclass overrides it with the true exp). The current phrasing could mislead a geometer into thinking the base class provides the true Riemannian/Finslerian exponential. | Replace *"the exact retraction"* with *"the retraction defined by `retract()`, which is a first-order approximation. Subclasses (e.g., `Sphere`, `Hyperboloid`) should override this with the closed-form exponential map when available."* |
| 9 | **MISSING** | `exp_map()` — Args / Returns | [manifold.py:106–112](src/ham/geometry/manifold.py#L106-L112) | No `Args` or `Returns` documentation. | Add `Args:` (`x: Base point on $M$; `v: Tangent vector in $T_x M$`) and `Returns:` (`Point on $M$ reached by following the geodesic from $x$ with initial velocity $v$.`). |
| 10 | **MISSING** | `log_map()` — Args / Returns | [manifold.py:114–131](src/ham/geometry/manifold.py#L114-L131) | No `Args` or `Returns` sections despite being a non-trivial method with a subtle scaling correction. | Add `Args:` (`x: Source point on $M$; `y: Target point on $M$`) and `Returns:` (`Tangent vector $v \in T_x M$ such that `exp_map(x, v) ≈ y`. Shape: `(D,)`.`). |
| 11 | **UNCLEAR** | `log_map()` — scaling correction rationale | [manifold.py:121–129](src/ham/geometry/manifold.py#L121-L129) | The inline comment block (lines 121–127) explains the scaling correction well for a code reader, but the docstring itself says only *"provides a mathematically rigorous first-order approximation"* without defining what "rigorous" means here. The comment mentions preventing "topological shortcuts through the interior of curved objects (like the hole of a Torus)" — this is excellent context that belongs in the docstring, not buried in a comment. No spec or literature reference is given. | Promote the comment to a `Note` section in the docstring: *"The projected secant $\Pi_{T_xM}(y - x)$ can have shorter ambient length than $y - x$ on highly curved manifolds, causing the solver to exploit interior 'shortcuts'. The scaling factor $\|y - x\| / \|\Pi_{T_xM}(y - x)\|$ corrects for this, preserving the ambient chord length as a proxy for intrinsic distance."* |
| 12 | **MISSING** | `log_map()` — no spec reference | [manifold.py:114–131](src/ham/geometry/manifold.py#L114-L131) | The scaling correction in `log_map` is a non-trivial numerical design decision. It is not documented in `spec/MATH_SPEC.md` or `spec/ARCH_SPEC.md`. This makes it impossible for a reviewer to audit against a specification. | **Recommended Action:** Either add a section to `spec/MATH_SPEC.md` § 6 documenting the secant-scaling heuristic, or add a docstring reference to the commit or design note that introduced it. |
| 13 | **MISSING** | `random_sample()` — Raises | [manifold.py:133–145](src/ham/geometry/manifold.py#L133-L145) | The docstring does not document any `Raises` behavior. Subclasses may raise on invalid shapes or non-PRNG keys. Even if the base class doesn't raise, it should document the expected contract. | Add: *"Raises: Behavior on invalid `key` or negative `shape` entries is subclass-defined."* |
| 14 | **TYPO** | All methods | Throughout | Type hints use `jnp.ndarray` (deprecated alias). JAX ≥ 0.4 recommends `jax.Array`. This is inconsistent with modern JAX practice and may confuse IDE users who see deprecation warnings. | Replace `jnp.ndarray` → `jax.Array` throughout. Also update `Tuple` import to use `tuple` (Python 3.9+). |
| 15 | **MISSING** | `Manifold` — no usage example or pointer | [manifold.py:44](src/ham/geometry/manifold.py#L44) | The class docstring has no example or pointer to example scripts. Since `Manifold` is abstract, a pointer to a concrete subclass or example script would orient new users. Examples in [examples/](examples/) use concrete manifolds (`Sphere`, `EuclideanSpace`, `Hyperboloid`) but `Manifold` never references them. | Add to class docstring: *"See `ham.geometry.surfaces` for concrete implementations (Sphere, Torus, Hyperboloid, etc.) and `examples/demo_trajectories.py` for usage."* |
| 16 | **INACCURATE** | `log_map()` docstring | [manifold.py:118](src/ham/geometry/manifold.py#L118) | Docstring says `retract(x, v) = y` as an equivalent condition to `exp_map(x, v) = y`. This is only approximately true — the base-class `exp_map` delegates to `retract`, but subclasses may override `exp_map` with the true exponential map while keeping `retract` as a cheaper first-order approximation. The docstring should state the approximation clearly. | Replace *"such that retract(x, v) = y (or exp_map(x, v) = y)"* with *"approximately satisfying `exp_map(x, v) ≈ y`. Accuracy depends on the subclass implementation of `exp_map`/`retract`."* |

---

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:---:|:---:|:---:|:---:|:---:|
| `Manifold` (class) | Yes | N/A | N/A | No | No |
| `ambient_dim` | Yes | N/A | Yes | Yes ($N$) | No |
| `intrinsic_dim` | Yes | N/A | Yes | Yes ($D$) | No |
| `project()` | Yes | Yes | Partial (no shape) | No | No |
| `to_tangent()` | Yes | Yes | Partial (no shape) | Yes ($T_x M$) | No |
| `retract()` | Partial (no Args/Returns) | No | No | Yes (conditions) | No |
| `exp_map()` | Partial (no Args/Returns) | No | No | Yes ($\text{Exp}_x(v)$) | No |
| `log_map()` | Partial (no Args/Returns) | No | No | Yes ($T_x M$) | No |
| `random_sample()` | Yes | Yes | Yes | No | No |
| `_safe_norm_ratio_jvp` (private) | Partial | No | No | Partial | No |

---

## Spec Alignment Notes

1. **`spec/ARCH_SPEC.md` § 2.1 vs. code:** The spec prescribes `random_sample(key: jax.random.PRNGKey, shape: tuple)`. The implementation uses `key: jax.Array` and `shape: Tuple[int, ...]` from `typing`. The `Tuple` import is outdated (Python 3.9+ supports `tuple[int, ...]` natively), and the type of `key` differs from the spec. Minor inconsistency.

2. **`spec/ARCH_SPEC.md` § 2.1 vs. code — extra methods:** The spec defines `project`, `to_tangent`, and `random_sample` as the abstract interface. The implementation adds `retract`, `exp_map`, and `log_map` — which are not in the spec's `Manifold` listing (they appear conceptually under § 4.4 for IVP solvers). These are useful additions but represent undocumented spec drift. **Recommended Action:** Update `spec/ARCH_SPEC.md` § 2.1 to include `retract`, `exp_map`, and `log_map` in the `Manifold` interface.

3. **`spec/MATH_SPEC.md` — no coverage of `log_map` scaling:** The secant-scaling correction in `log_map` (lines 121–129) is a non-trivial numerical design decision with no corresponding spec entry. `spec/MATH_SPEC.md` § 6 covers $\epsilon$-regularization and homogeneity enforcement but not this heuristic. **Recommended Action:** Add a subsection to `spec/MATH_SPEC.md` § 6 documenting this technique.

4. **`_safe_norm_ratio_jvp` not in spec:** This custom JVP helper is a numerical stability primitive that should be referenced in `spec/MATH_SPEC.md` § 6, alongside the `safe_norm` utility from `utils/math.py`.

---

## Suggested Texts

### §Suggested Text 1 — Module docstring

```python
"""
Abstract Manifold base class — the topological domain for HAMTools.

This module defines :class:`Manifold`, the abstract base class that
specifies the domain M and its constraints (projection, tangent spaces,
retraction). The manifold does *not* define distance or geodesics;
those are the responsibility of :class:`~ham.geometry.metric.FinslerMetric`.

Subclasses must implement :meth:`project`, :meth:`to_tangent`,
:meth:`retract`, and :meth:`random_sample`. Optionally override
:meth:`exp_map` and :meth:`log_map` with closed-form expressions
for better accuracy (see :mod:`ham.geometry.surfaces`).

Architecture reference: spec/ARCH_SPEC.md § 2.1.
"""
```

### §Suggested Text 2 — `retract()` Args/Returns

```python
def retract(self, x: jax.Array, delta: jax.Array) -> jax.Array:
    """
    Retraction: maps a tangent vector delta in T_x M back to a point on M.

    ...existing mathematical description...

    Args:
        x: Base point on the manifold M. Shape: ``(D,)`` or ``(N,)``
           in ambient coordinates.
        delta: Tangent vector in T_x M. Shape: same as ``x``.

    Returns:
        Point on M. Shape: same as ``x``.
    """
```

### §Suggested Text 3 — `exp_map()` full docstring

```python
def exp_map(self, x: jax.Array, v: jax.Array) -> jax.Array:
    """
    Exponential map Exp_x(v): follow the geodesic from x with velocity v.

    The default implementation delegates to :meth:`retract`, which is a
    first-order approximation. Subclasses (e.g., Sphere, Hyperboloid)
    should override with the closed-form exponential when available.

    Args:
        x: Base point on M. Shape: ``(D,)`` in ambient coordinates.
        v: Tangent vector in T_x M. Shape: same as ``x``.

    Returns:
        Point on M reached by following the geodesic. Shape: same as ``x``.
    """
```
