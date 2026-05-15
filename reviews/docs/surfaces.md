# Documentation Review: `src/ham/geometry/surfaces.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** 2026-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** [src/ham/geometry/surfaces.py](src/ham/geometry/surfaces.py)

## Summary

Overall documentation quality: **needs work**.

The file provides five concrete `Manifold` subclasses (`Sphere`, `Torus`, `Paraboloid`, `Hyperboloid`, `EuclideanSpace`) plus two private helper functions with custom JVPs (`_safe_minkowski_self_norm`, `_safe_arccos`). The module-level docstring lists review-driven changes but does not orient the reader to the file's purpose or contents. The `Hyperboloid` class has a class-level docstring; all other classes and the vast majority of public methods have **no docstrings at all**. The two methods that do have docstrings (`Sphere.exp_map`, `_safe_minkowski_self_norm`) are incomplete: they lack `Args`, `Returns`, and `Raises` sections. Mathematical formulae are provided only for `Sphere.exp_map`; the `Hyperboloid` exponential/logarithmic maps — which implement non-trivial $\cosh$/$\sinh$ closed forms referenced in `spec/MATH_SPEC.md` § 4.1 — are entirely undocumented. There are no usage examples or pointers to example scripts anywhere in the file.

---

## Issue Tracker

| # | Severity | Symbol / Section | Location | Issue | Suggested Text |
|---|----------|-----------------|----------|-------|----------------|
| 1 | **MISSING** | Module docstring | [surfaces.py:1–11](src/ham/geometry/surfaces.py#L1-L11) | The module docstring is a changelog of review fixes (P0-1, P1-6, etc.), not a user-facing orientation. A user running `help(ham.geometry.surfaces)` or browsing the source sees implementation notes, not what the module provides or how to use it. `spec/ARCH_SPEC.md` § 5 lists this as `surfaces.py  # Sphere, Torus, Hyperboloid, Paraboloid, EuclideanSpace`. | Replace the current docstring with: *"Concrete manifold implementations for analytical sub-manifolds.\n\nClasses:\n    Sphere — $S^n(r)$ embedded in $\\mathbb{R}^{n+1}$.\n    Torus — $T^2$ embedded in $\\mathbb{R}^3$.\n    Paraboloid — $z = x^2 + y^2$ embedded in $\\mathbb{R}^3$.\n    Hyperboloid — Upper sheet of the two-sheeted hyperboloid in Minkowski space.\n    EuclideanSpace — Flat $\\mathbb{R}^N$.\n\nSee also: `spec/ARCH_SPEC.md` § 5, `spec/MATH_SPEC.md` § 4.1."* Move the changelog to an internal `# Changelog` comment block below. |
| 2 | **MISSING** | `Sphere` (class) | [surfaces.py:21](src/ham/geometry/surfaces.py#L21) | No class docstring. The class implements $S^n(r)$ embedded in $\mathbb{R}^{n+1}$ — this is stated only in the section comment, not in the class itself. | Add: *"The $n$-sphere $S^n(r)$ of radius $r$, embedded in $\\mathbb{R}^{n+1}$.\n\nThe sphere uses the standard round metric inherited from ambient Euclidean space. Exponential and logarithmic maps are exact (closed-form geodesic formulae). See `spec/MATH_SPEC.md` § 4, row 'Hyperboloid' (analogous structure).\n\nArgs:\n    intrinsic_dim: Dimension $n$ of the sphere. Default: 2 ($S^2$).\n    radius: Radius $r$. Default: 1.0."* |
| 3 | **MISSING** | `Sphere.project()` | [surfaces.py:38–44](src/ham/geometry/surfaces.py#L38-L44) | No docstring. Projects ambient points onto $S^n(r)$ via normalization. Zero-vector fallback to the north pole is an important edge-case detail that should be documented. | Add: *"Projects $x \\in \\mathbb{R}^{n+1}$ onto $S^n(r)$ by normalizing: $\\pi(x) = r \\cdot x / \\|x\\|$.\n\nFor zero-length inputs ($\\|x\\| < \\epsilon$), defaults to the north pole $(0, \\ldots, 0, r)$.\n\nArgs:\n    x: Point in ambient space, shape `(..., n+1)`.\n\nReturns:\n    Projected point on $S^n(r)$, shape `(..., n+1)`."* |
| 4 | **MISSING** | `Sphere.to_tangent()` | [surfaces.py:46–48](src/ham/geometry/surfaces.py#L46-L48) | No docstring. Implements orthogonal projection $v - \langle x, v \rangle x / r^2$ onto $T_x S^n$. | Add: *"Projects ambient vector $v$ onto $T_x S^n(r)$: $\\Pi(v) = v - \\frac{\\langle x, v \\rangle}{r^2} x$.\n\nArgs:\n    x: Base point on $S^n(r)$, shape `(..., n+1)`.\n    v: Ambient vector, shape `(..., n+1)`.\n\nReturns:\n    Tangent vector in $T_x S^n(r)$, shape `(..., n+1)`."* |
| 5 | **UNCLEAR** | `Sphere.exp_map()` | [surfaces.py:50–66](src/ham/geometry/surfaces.py#L50-L66) | The docstring states the formula $\gamma(1) = \cos(\theta) x + \frac{\sin(\theta)}{\theta} v$ correctly but lacks `Args`, `Returns`, shape information, and does not mention the Taylor-series branch for small $\theta$ which is the key numerical detail. An ML engineer cannot determine input/output shapes; a mathematician cannot verify the small-angle approximation from the docstring alone. | Extend to: *"Args:\n    x: Base point on $S^n(r)$, shape `(..., n+1)`.\n    v: Tangent vector in $T_x S^n(r)$, shape `(..., n+1)`.\n\nReturns:\n    Point on $S^n(r)$, shape `(..., n+1)`.\n\nNote:\n    For $\\theta < \\text{TAYLOR\\_EPS}$, $\\sin(\\theta)/\\theta$ and $\\cos(\\theta)$ are replaced by their second-order Taylor expansions to avoid numerical instability."* |
| 6 | **MISSING** | `Sphere.retract()` | [surfaces.py:68–70](src/ham/geometry/surfaces.py#L68-L70) | Docstring says only "Retraction (delegates to exp_map)" with no `Args` or `Returns`. Since this delegates to `exp_map`, it should state that explicitly and cross-reference. | Add: *"Args:\n    x: Base point on $S^n(r)$.\n    delta: Tangent vector in $T_x S^n(r)$.\n\nReturns:\n    Point on $S^n(r)$. Equivalent to `exp_map(x, delta)`."* |
| 7 | **MISSING** | `Sphere.log_map()` | [surfaces.py:72–93](src/ham/geometry/surfaces.py#L72-L93) | No docstring. This implements the exact inverse exponential map on $S^n(r)$ using the arccos-based formula — a non-trivial computation with three safety guards (clipped dot product, safe $\sin$, Taylor branch). The inline comments are helpful but a docstring is required for public API. | Add: *"Inverse exponential map on $S^n(r)$.\n\nComputes $v \\in T_x S^n(r)$ such that $\\text{Exp}_x(v) = y$. Uses the formula $v = \\frac{\\theta}{\\sin \\theta}(y - \\cos\\theta \\cdot x)$ where $\\theta = \\arccos(\\langle x, y \\rangle / r^2)$.\n\nArgs:\n    x: Source point on $S^n(r)$, shape `(..., n+1)`.\n    y: Target point on $S^n(r)$, shape `(..., n+1)`.\n\nReturns:\n    Tangent vector in $T_x S^n(r)$, shape `(..., n+1)`."* |
| 8 | **MISSING** | `Sphere.parallel_transport()` | [surfaces.py:95–101](src/ham/geometry/surfaces.py#L95-L101) | No docstring. Implements parallel transport via the bisector reflection (Schild's ladder equivalent). The inline comment "Reflection through the bisector of x and y" is correct but insufficient for a public method. The formula and its assumptions (points not antipodal, etc.) should be documented. | Add: *"Parallel transports $v \\in T_x S^n(r)$ to $T_y S^n(r)$ along the geodesic from $x$ to $y$.\n\nUses the closed-form reflection formula: $P_{x \\to y}(v) = v - \\frac{\\langle y, v \\rangle}{r^2 + \\langle x, y \\rangle}(x + y)$.\n\nArgs:\n    x: Source point on $S^n(r)$, shape `(..., n+1)`.\n    y: Target point on $S^n(r)$, shape `(..., n+1)`.\n    v: Tangent vector at $x$, shape `(..., n+1)`.\n\nReturns:\n    Transported vector at $y$, shape `(..., n+1)`.\n\nWarning:\n    Degenerates when $x$ and $y$ are antipodal ($\\langle x, y \\rangle \\to -r^2$)."* |
| 9 | **MISSING** | `Sphere.random_sample()` | [surfaces.py:103–105](src/ham/geometry/surfaces.py#L103-L105) | No docstring. Uses the Gaussian projection method (Muller, 1959) for uniform sampling — worth stating. | Add: *"Samples uniformly on $S^n(r)$ via Gaussian projection.\n\nArgs:\n    key: JAX PRNG key.\n    shape: Batch shape.\n\nReturns:\n    Points on $S^n(r)$, shape `(*shape, n+1)`."* |
| 10 | **MISSING** | `Torus` (class) | [surfaces.py:112](src/ham/geometry/surfaces.py#L112) | No class docstring. The class implements $T^2$ embedded in $\mathbb{R}^3$ with major radius $R$ and minor radius $r$, using the standard parametric embedding $(R + r\cos v)\cos u, (R + r\cos v)\sin u, r\sin v)$. | Add: *"The 2-torus $T^2$ embedded in $\\mathbb{R}^3$.\n\nParametrized by major radius $R$ (distance from center to tube center) and minor radius $r$ (tube radius): $(x, y, z) = ((R + r\\cos v)\\cos u,\\; (R + r\\cos v)\\sin u,\\; r\\sin v)$.\n\nArgs:\n    major_R: Major radius $R$. Default: 2.0.\n    minor_r: Minor radius $r$. Default: 1.0."* |
| 11 | **MISSING** | `Torus.project()` | [surfaces.py:130–146](src/ham/geometry/surfaces.py#L130-L146) | No docstring. Non-trivial projection onto the torus surface with zero-rho fallback — important to document. | Add docstring with `Args`, `Returns`, and note about the fallback for $\rho < \text{TAYLOR\_EPS}$. |
| 12 | **MISSING** | `Torus.to_tangent()` | [surfaces.py:148–162](src/ham/geometry/surfaces.py#L148-L162) | No docstring. | Add docstring with `Args`, `Returns`. |
| 13 | **INACCURATE** | `Torus.exp_map()` | [surfaces.py:164–166](src/ham/geometry/surfaces.py#L164-L166) | Docstring says *"Approximate exp map via projected retraction"* — this is accurate but the parameter is named `delta` while the parent class `exp_map` signature uses `v`. This inconsistency may confuse users reading both the base class and subclass. The same inconsistency appears in `Paraboloid.exp_map()`. | Rename parameter to `v` to match the base class signature, or document the difference: *"Args:\n    x: Base point.\n    delta: Tangent vector (alias for `v` in the base class)."* |
| 14 | **MISSING** | `Torus.retract()`, `Torus.random_sample()` | [surfaces.py:168–178](src/ham/geometry/surfaces.py#L168-L178) | No docstrings on either method. `random_sample` uses the standard parametric form with uniform $(u, v)$ — this produces a non-uniform area distribution (denser near inner equator) which should be noted. | Add docstrings. For `random_sample`, note: *"Samples from uniform angular coordinates. Note: this does NOT produce area-uniform samples on $T^2$; points near $u$ where $\\cos v \\approx -1$ are over-represented."* |
| 15 | **MISSING** | `Torus.log_map()` | N/A | `Torus` does not override `log_map()`, so it inherits the base class's first-order approximation. This is not documented anywhere on the class. A user comparing `Sphere` (exact `log_map`) with `Torus` (approximate `log_map`) has no way to know the difference without reading source. | Add a class-level note: *"Note: `log_map` uses the base class first-order approximation (projected secant with scaling correction). No closed-form inverse exponential map is implemented."* |
| 16 | **MISSING** | `Torus.parallel_transport()` | N/A | `Torus` does not implement `parallel_transport()`, yet the base class `Manifold` does not define it either (it is not abstract). If a user calls `torus.parallel_transport(x, y, v)` they will get an `AttributeError`. This gap should be documented. | Add a class-level note listing which optional methods are and are not available. |
| 17 | **MISSING** | `Paraboloid` (class) | [surfaces.py:183](src/ham/geometry/surfaces.py#L183) | No class docstring. Implements $z = x^2 + y^2$ in $\mathbb{R}^3$, but this is stated only in the section comment. | Add: *"The paraboloid of revolution $z = x^2 + y^2$, embedded in $\\mathbb{R}^3$."* |
| 18 | **MISSING** | `Paraboloid.project()`, `Paraboloid.to_tangent()` | [surfaces.py:192–199](src/ham/geometry/surfaces.py#L192-L199) | No docstrings. `project` discards the input $z$ and recomputes $z = x^2 + y^2$; `to_tangent` computes the surface normal $n = (-2x, -2y, 1)/\|n\|$ and projects. | Add docstrings with `Args` and `Returns`. |
| 19 | **INACCURATE** | `Paraboloid.exp_map()` | [surfaces.py:201–203](src/ham/geometry/surfaces.py#L201-L203) | Docstring says *"Approximate exp map via exact retraction"* — the word "exact" is misleading. The retraction on the Paraboloid is a projected retraction $\pi(x + \delta)$, which is a first-order approximation, not exact. The phrase "exact retraction" is self-contradictory in differential geometry. | Replace with: *"Approximate exp map via projected retraction."* |
| 20 | **MISSING** | `Paraboloid.retract()`, `Paraboloid.random_sample()` | [surfaces.py:205–213](src/ham/geometry/surfaces.py#L205-L213) | No docstrings. | Add docstrings with `Args`, `Returns`. |
| 21 | **MISSING** | `Paraboloid.log_map()`, `Paraboloid.parallel_transport()` | N/A | Same gap as `Torus`: no override, no documentation of which base-class defaults apply. | Document in class-level note. |
| 22 | **MISSING** | `_safe_minkowski_self_norm()` — Args / Returns / math derivation | [surfaces.py:220–228](src/ham/geometry/surfaces.py#L220-L228) | The docstring describes the computation but does not document the custom JVP semantics, does not list `Args` or `Returns`, and does not reference `spec/MATH_SPEC.md` § 4 (Hyperboloid) or § 6 (Numerical Stability). The P1-7 note about restricting to self-norm is good but insufficient. | Add: *"Args:\n    x: Vector in Minkowski space, shape `(..., n+1)`.\n\nReturns:\n    Minkowski self-norm $\\sqrt{-x_0^2 + x_1^2 + \\cdots + x_n^2}$, shape `(...)`.\n\nNote:\n    A custom JVP is registered to handle the $\\|x\\|_L = 0$ singularity.\n    See `spec/MATH_SPEC.md` § 4 (Hyperboloid row) and § 6 (Numerical Stability)."* |
| 23 | **MISSING** | `_safe_minkowski_self_norm_jvp()` | [surfaces.py:231–243](src/ham/geometry/surfaces.py#L231-L243) | No docstring on the JVP rule. The derivative formula $d\sqrt{\langle x,x\rangle_L} = \langle x, \dot{x}\rangle_L / \sqrt{\langle x,x\rangle_L}$ is stated in a comment but should be in a docstring for auditability. | Add docstring: *"Custom JVP for `_safe_minkowski_self_norm`. Derivative: $\\frac{\\langle x, \\dot{x}\\rangle_L}{\\|x\\|_L}$, clamped to 0 when $\\|x\\|_L < \\text{GRAD\\_EPS}$."* |
| 24 | **MISSING** | `_safe_arccos()` — Args / Returns | [surfaces.py:246–248](src/ham/geometry/surfaces.py#L246-L248) | Docstring says *"Safely computes arccos(x) avoiding infinite gradients at \|x\|=1"* but has no `Args` or `Returns`. | Add `Args:` / `Returns:` sections. |
| 25 | **MISSING** | `_safe_arccos_jvp()` | [surfaces.py:251–258](src/ham/geometry/surfaces.py#L251-L258) | No docstring. | Add: *"Custom JVP for `_safe_arccos`. Derivative: $-\\dot{x} / \\sqrt{1 - x^2}$, regularized by $\\text{GRAD\\_EPS}$."* |
| 26 | **MISSING** | `Hyperboloid` — class docstring incomplete | [surfaces.py:264–268](src/ham/geometry/surfaces.py#L264-L268) | The class docstring states the constraint $-x_0^2 + x_1^2 + \cdots + x_n^2 = -1,\; x_0 > 0$ correctly, but lacks: constructor `Args`, spec references, the relationship to the Minkowski inner product, and the fact that `exp_map`/`log_map` are exact closed-form (unlike `Torus`/`Paraboloid`). `spec/MATH_SPEC.md` § 4 and § 4.1 explicitly reference this class. | Extend to include `Args`, spec references, and a note about exact maps. |
| 27 | **MISSING** | `Hyperboloid._minkowski_dot()`, `Hyperboloid._minkowski_norm()` | [surfaces.py:282–285](src/ham/geometry/surfaces.py#L282-L285) | No docstrings on either private helper. While private, these define the fundamental inner product used by all Hyperboloid operations and are called from outside via the public methods. Documenting the Minkowski signature $\eta = \text{diag}(-1, +1, \ldots, +1)$ would help both audiences. | Add brief docstrings. |
| 28 | **MISSING** | `Hyperboloid.project()` | [surfaces.py:287–299](src/ham/geometry/surfaces.py#L287-L299) | No docstring. Implements a two-branch projection: (1) if the point already has negative Minkowski self-norm and $x_0 > 0$, rescale; (2) otherwise, lift spatial coordinates. This branching logic is subtle and undocumented. | Add docstring explaining both branches with `Args` and `Returns`. |
| 29 | **MISSING** | `Hyperboloid.to_tangent()` | [surfaces.py:301–303](src/ham/geometry/surfaces.py#L301-L303) | No docstring. Implements $\Pi(v) = v + \langle x, v \rangle_L x$ (note the $+$ sign due to Minkowski signature). | Add docstring. |
| 30 | **MISSING** | `Hyperboloid.exp_map()` | [surfaces.py:305–316](src/ham/geometry/surfaces.py#L305-L316) | No docstring. Implements the exact exponential map $\text{Exp}_x(v) = \cosh(\|v\|_L) x + \frac{\sinh(\|v\|_L)}{\|v\|_L} v$. This is a critical method referenced by `spec/MATH_SPEC.md` § 4.1 and should have a full docstring with the formula, Args, Returns, and a note about the Taylor branch for small $\|v\|_L$. | Add full docstring. |
| 31 | **MISSING** | `Hyperboloid.log_map()` | [surfaces.py:352–369](src/ham/geometry/surfaces.py#L352-L369) | No docstring. Implements the inverse exponential map using $\text{arcsinh}$. Uses a scaling correction for small norms. | Add full docstring with formula, `Args`, `Returns`. |
| 32 | **MISSING** | `Hyperboloid.parallel_transport()` | [surfaces.py:371–378](src/ham/geometry/surfaces.py#L371-L378) | No docstring. Implements the closed-form transport $P_{x \to y}(v) = v + \frac{\langle y, v \rangle_L}{1 - \langle x, y \rangle_L}(x + y)$. Note the $+$ sign (vs. $-$ on the Sphere) due to Minkowski signature. | Add docstring analogous to `Sphere.parallel_transport()`. |
| 33 | **INACCURATE** | `Hyperboloid.parallel_transport()` — denominator guard | [surfaces.py:376](src/ham/geometry/surfaces.py#L376) | The denominator is guarded as `jnp.maximum(1.0 - xy, 2.0)`. The `2.0` floor is suspicious — for points on the hyperboloid, $\langle x, y \rangle_L \leq -1$, so $1 - \langle x, y \rangle_L \geq 2$ always. The guard does nothing for valid inputs but the docstring (which doesn't exist) should explain this. If the intent is a safety clamp, it should use a small epsilon, not `2.0`. Either way, the **absence of any documentation** makes it impossible to determine whether this is correct or a bug. | Document the denominator guard and its mathematical rationale. |
| 34 | **MISSING** | `Hyperboloid.retract()` | [surfaces.py:380–385](src/ham/geometry/surfaces.py#L380-L385) | No docstring. Implements a clamped retraction with `max_norm = 10.0` — a hard-coded stability threshold that should be documented and potentially made configurable. | Add docstring explaining the norm-clamping strategy. |
| 35 | **MISSING** | `Hyperboloid.random_sample()` | [surfaces.py:387–399](src/ham/geometry/surfaces.py#L387-L399) | No docstring. Samples via exponential map from the origin with Gaussian tangent vectors. | Add docstring. |
| 36 | **MISSING** | `Hyperboloid.metric_tensor()` | [surfaces.py:401–404](src/ham/geometry/surfaces.py#L401-L404) | No docstring. Returns the ambient Minkowski metric $\eta = \text{diag}(-1, 1, \ldots, 1)$ — but note this is the **ambient** metric, not the induced metric on the hyperboloid. The method name is potentially misleading without documentation clarifying this distinction. The base class `Manifold` does not define `metric_tensor()`, so this is an ad-hoc extension. | Add docstring: *"Returns the ambient Minkowski metric tensor $\\eta = \\text{diag}(-1, 1, \\ldots, 1)$.\n\nNote: This is the ambient space metric, not the induced Riemannian metric on the hyperboloid."* |
| 37 | **MISSING** | `EuclideanSpace` — class docstring incomplete | [surfaces.py:411](src/ham/geometry/surfaces.py#L411) | Docstring says *"Flat Euclidean space R^N"* — correct but lacking `Args`, spec reference, or the fact that all geometric operations are trivial (identity). | Extend: *"Args:\n    dim: Dimension $N$.\n\nAll geometric maps (exp, log, transport, retract) are identity/trivial."* |
| 38 | **MISSING** | All `EuclideanSpace` methods | [surfaces.py:420–444](src/ham/geometry/surfaces.py#L420-L444) | None of the 7 public methods (`project`, `to_tangent`, `exp_map`, `retract`, `log_map`, `parallel_transport`, `random_sample`) have docstrings. While their implementations are trivial, they are public API. | Add brief one-line docstrings. |
| 39 | **UNCLEAR** | Notation inconsistency: `delta` vs `v` | Multiple | `Torus.exp_map`, `Torus.retract`, `Paraboloid.exp_map`, `Paraboloid.retract` use `delta` as the tangent vector parameter name; `Sphere.exp_map`, `Hyperboloid.exp_map`, `EuclideanSpace.exp_map` use `v`. The base class `Manifold` uses `v` in `exp_map` and `delta` in `retract`. This inconsistency is confusing for both audiences. | Standardize: use `v` for `exp_map` and `delta` for `retract` everywhere, matching the base class. |
| 40 | **TYPO** | Type annotations | Throughout | All type hints use `jnp.ndarray` (deprecated alias in JAX ≥ 0.4). The `random_sample` methods use `jax.Array` for the `key` parameter but `jnp.ndarray` for return types — inconsistent within the same signature. | Replace `jnp.ndarray` → `jax.Array` throughout, or at minimum be consistent within each method. |
| 41 | **MISSING** | No example pointers | Entire file | None of the classes reference example scripts. `examples/demo_trajectories.py`, `examples/demo_vortex.py`, `examples/demo_zermelo.py` appear to use these manifolds. Cross-references would help new users. | Add to each class docstring: *"Example: See `examples/demo_trajectories.py`."* (or the appropriate script). |
| 42 | **UNCLEAR** | `Hyperboloid.log_map()` — arcsinh vs acosh | [surfaces.py:357](src/ham/geometry/surfaces.py#L357) | The log map uses `jnp.arcsinh(norm_u)` to compute the geodesic distance. The standard formula for hyperbolic distance is $d(x, y) = \text{acosh}(-\langle x, y \rangle_L)$. While $\text{arcsinh}(\|u\|_L)$ where $u = y + \langle x, y \rangle_L x$ is mathematically equivalent (since $\sinh(d) = \|u\|_L$), this non-standard form should be documented so a geometer can verify the equivalence. | Add a note: *"Uses `arcsinh(||u||_L)` rather than `acosh(-<x,y>_L)` for improved numerical stability when $x \\approx y$. The two are equivalent: $\\sinh(d(x,y)) = \\|y + \\langle x,y \\rangle_L x\\|_L$."* |

---

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:---:|:---:|:---:|:---:|:---:|
| `Sphere` (class) | No | No | N/A | No | No |
| `Sphere.__init__()` | No | No | No | No | No |
| `Sphere.project()` | No | No | No | No | No |
| `Sphere.to_tangent()` | No | No | No | No | No |
| `Sphere.exp_map()` | Partial | No | No | Yes ($\theta$, $\gamma$) | No |
| `Sphere.retract()` | Partial | No | No | No | No |
| `Sphere.log_map()` | No | No | No | No | No |
| `Sphere.parallel_transport()` | No | No | No | No | No |
| `Sphere.random_sample()` | No | No | No | No | No |
| `Torus` (class) | No | No | N/A | No | No |
| `Torus.__init__()` | No | No | No | No | No |
| `Torus.project()` | No | No | No | No | No |
| `Torus.to_tangent()` | No | No | No | No | No |
| `Torus.exp_map()` | Partial | No | No | No | No |
| `Torus.retract()` | No | No | No | No | No |
| `Torus.random_sample()` | No | No | No | No | No |
| `Paraboloid` (class) | No | No | N/A | No | No |
| `Paraboloid.project()` | No | No | No | No | No |
| `Paraboloid.to_tangent()` | No | No | No | No | No |
| `Paraboloid.exp_map()` | Partial | No | No | No | No |
| `Paraboloid.retract()` | No | No | No | No | No |
| `Paraboloid.random_sample()` | No | No | No | No | No |
| `_safe_minkowski_self_norm()` | Partial | No | No | Partial | No |
| `_safe_arccos()` | Partial | No | No | No | No |
| `Hyperboloid` (class) | Partial | No | N/A | Yes (constraint) | No |
| `Hyperboloid._minkowski_dot()` | No | No | No | No | No |
| `Hyperboloid._minkowski_norm()` | No | No | No | No | No |
| `Hyperboloid.project()` | No | No | No | No | No |
| `Hyperboloid.to_tangent()` | No | No | No | No | No |
| `Hyperboloid.exp_map()` | No | No | No | No | No |
| `Hyperboloid.log_map()` | No | No | No | No | No |
| `Hyperboloid.parallel_transport()` | No | No | No | No | No |
| `Hyperboloid.retract()` | No | No | No | No | No |
| `Hyperboloid.random_sample()` | No | No | No | No | No |
| `Hyperboloid.metric_tensor()` | No | No | No | No | No |
| `EuclideanSpace` (class) | Partial | No | N/A | No | No |
| `EuclideanSpace.project()` | No | No | No | No | No |
| `EuclideanSpace.to_tangent()` | No | No | No | No | No |
| `EuclideanSpace.exp_map()` | No | No | No | No | No |
| `EuclideanSpace.retract()` | No | No | No | No | No |
| `EuclideanSpace.log_map()` | No | No | No | No | No |
| `EuclideanSpace.parallel_transport()` | No | No | No | No | No |
| `EuclideanSpace.random_sample()` | No | No | No | No | No |

---

## Spec Alignment Notes

| Spec Section | Spec Statement | File Status |
|---|---|---|
| `spec/ARCH_SPEC.md` § 2.1 | `Manifold` defines `project`, `to_tangent`, `random_sample` as abstract. | All five classes implement these three. **Aligned.** |
| `spec/ARCH_SPEC.md` § 5 | `surfaces.py` listed as containing `Sphere, Torus, Hyperboloid, Paraboloid, EuclideanSpace`. | All five present. **Aligned.** |
| `spec/MATH_SPEC.md` § 4 | Hyperboloid listed with "Minkowskian $\sqrt{\langle v, v\rangle_L}$" metric and "Levi-Civita equivalent" connection. | `Hyperboloid.metric_tensor()` returns the ambient Minkowski metric. `parallel_transport()` implements closed-form Levi-Civita transport. However, **none of this is documented** in docstrings. |
| `spec/MATH_SPEC.md` § 4.1 | *"Hyperboloid models the upper sheet in Minkowski space and features exact $\cosh$/$\sinh$ exponential and logarithmic maps."* | Implementation present. **No docstring references the spec.** |
| `spec/MATH_SPEC.md` § 4.1 | *"Critical Note: Integrating these exact geometric maps... with deep neural learning loops inside the VAE currently causes severe numerical instability."* | This caveat is **not documented anywhere** in the `surfaces.py` docstrings. Users integrating `Hyperboloid` into training loops will not be warned. |
| `spec/ARCH_SPEC.md` § 6 (Known Limitations) | *"Hyperboloid VAE: Joint training on complex curved manifolds (Sphere, Hyperboloid) with the full VAE pipeline remains numerically sensitive."* | **Not documented** in `Sphere` or `Hyperboloid` class docstrings. |
| `spec/ARCH_SPEC.md` § 1 | Batch-first design: all operations assume leading batch dimension `(B, ...)`. | `Torus.project()` and `Torus.to_tangent()` use `x[:2]` (unbatched indexing), inconsistent with the batch-first spec. This is a code issue (Code Reviewer's domain), but the **documentation does not clarify** whether `Torus` supports batched inputs. |

---

## Priority Summary

- **MISSING**: 36 findings — the dominant issue. The vast majority of public symbols lack any docstring.
- **INACCURATE**: 3 findings — `Paraboloid.exp_map` "exact retraction" oxymoron, `Torus.exp_map` parameter naming, `Hyperboloid.parallel_transport` undocumented guard.
- **UNCLEAR**: 3 findings — parameter naming inconsistency, `Hyperboloid.log_map` arcsinh convention, batch support ambiguity.
- **TYPO**: 1 finding — deprecated `jnp.ndarray` type annotations.

**Recommended Action:** A documentation pass adding class-level and method-level docstrings is the single highest-impact improvement. Priority should go to `Hyperboloid` (most complex, spec-referenced, numerically sensitive) followed by `Sphere` (most commonly used).
