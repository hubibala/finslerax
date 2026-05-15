# Documentation Review: `src/ham/geometry/transport.py`

**Reviewer:** Doc Reviewer Agent  
**Date:** May 15, 2026

## Summary

Overall documentation quality: **needs work**.

The module contains three public symbols (`Connection`, `BerwaldConnection`, `berwald_transport`) all exported via `ham.geometry.__init__` and `ham.__init__`. The base class has no docstrings on its abstract methods, the key `christoffel_symbols` override is undocumented, and the public convenience function `berwald_transport` has no docstring at all. Parameter shapes, return shapes, and audience-bridging prose are absent throughout. The ARCH_SPEC shows a different function signature for `berwald_transport` than what is implemented, creating a spec-documentation-code mismatch.

---

## Issue Tracker

| # | Severity | Symbol / Section | Issue | Suggested Text |
|---|----------|-----------------|-------|----------------|
| 1 | **MISSING** | Module level (`transport.py:1`) | No module-level docstring. A reader opening this file has no orientation. | `"""Berwald parallel-transport integrator.\n\nImplements the Berwald connection (spec/MATH_SPEC.md § 3) for transporting\ntangent vectors along geodesics on Finsler manifolds. The connection\ncoefficients are derived by differentiating the geodesic spray twice\nw.r.t. velocity.\n"""` |
| 2 | **MISSING** | `Connection.__init__` (`transport.py:7-8`) | No docstring on `__init__`. The `metric` parameter is not documented. | `"""Initialise connection from a Finsler metric.\n\nArgs:\n    metric: The FinslerMetric whose spray defines this connection.\n"""` |
| 3 | **MISSING** | `Connection.christoffel_symbols` (`transport.py:10-11`) | Abstract method has no docstring — only a bare `raise NotImplementedError`. Users subclassing `Connection` have no contract to follow. | `"""Compute the connection coefficients at (x, v).\n\nArgs:\n    x: Position on the manifold, shape (D,).\n    v: Tangent vector at x, shape (D,).\n\nReturns:\n    Connection coefficients Γ^i_{jk}, shape (D, D, D).\n\nRaises:\n    NotImplementedError: Subclasses must override.\n"""` |
| 4 | **MISSING** | `Connection.parallel_transport` (`transport.py:13-14`) | Abstract method has no docstring. | `"""Transport a vector along a discrete path.\n\nArgs:\n    path_x: Positions along the curve, shape (T, D).\n    path_v: Velocities along the curve, shape (T, D).\n    vec_start: Initial tangent vector to transport, shape (D,).\n\nReturns:\n    Transported vectors at each point, shape (T, D).\n\nRaises:\n    NotImplementedError: Subclasses must override.\n"""` |
| 5 | **UNCLEAR** | `BerwaldConnection` class docstring (`transport.py:17-23`) | The class docstring gives the formula $\Gamma^i_{jk} = \partial^2 G^i / \partial v^j \partial v^k$ but does not explain what this means computationally (ML-engineer audience). It also omits the spec's $^B\Gamma$ notation that distinguishes Berwald from Chern/Cartan connections. | Add a sentence such as: *"Computationally, the coefficients are the Hessian of the `metric.spray` function with respect to the velocity argument. See `spec/MATH_SPEC.md § 3.1`."* |
| 6 | **MISSING** | `BerwaldConnection.christoffel_symbols` (`transport.py:25-31`) | Override has no docstring — only inline comments. This is a public method on a public class. Args, Returns, and shape information are absent. | `"""Berwald connection coefficients via Hessian of the spray.\n\nComputes $^B\\Gamma^i_{jk}(x,v) = \\partial^2 G^i / \\partial v^j \\partial v^k$\nusing two nested `jax.jacfwd` calls on `metric.spray`.\n\nArgs:\n    x: Position, shape (D,).\n    v: Tangent vector, shape (D,).\n\nReturns:\n    Coefficients tensor, shape (D, D, D).\n"""` |
| 7 | **MISSING** | `BerwaldConnection.parallel_transport` (`transport.py:33-57`) | Docstring is present but incomplete — `Args`, `Returns`, and `Raises` sections are all absent. Shape information is not documented. | Add sections: `Args:\n    path_x: Discrete positions along the curve, shape (T, D).\n    path_v: Velocities at each position, shape (T, D).\n    vec_start: Initial tangent vector, shape (D,).\n\nReturns:\n    Transported vectors aligned with path_x, shape (T, D).\n    The first row equals vec_start; the last row is the\n    vector at path_x[-2] (one-step lag from scan).\n` |
| 8 | **UNCLEAR** | `BerwaldConnection.parallel_transport` (`transport.py:37`) | The docstring equation `dX_dt + Gamma^i_jk(x, v) * v^j * X^k = 0` uses ASCII notation without distinguishing the Berwald connection from a generic $\Gamma$. Should use $^B\Gamma$ to match `spec/MATH_SPEC.md § 3.2`. | Replace with: `$$\\frac{dX^i}{dt} + \\,^B\\Gamma^i_{jk}(\\gamma, \\dot\\gamma)\\,\\dot\\gamma^j X^k = 0$$` |
| 9 | **INACCURATE** | `BerwaldConnection.parallel_transport` return (`transport.py:55-56`) | The return value is not what a naïve reader expects. `result = jnp.concatenate([vec_start[None, :], transported_vecs[:-1]], axis=0)` means the output is shifted by one step: the vector at index $i$ is the transported vector *arriving at* `path_x[i]`, not *departing from* it. This offset semantics is not documented and could surprise both audiences. | Document the alignment convention explicitly: *"The returned array has the same leading dimension as `path_x`. Entry `i` is the transported vector at `path_x[i]`, computed from integrating over the segment ending at that point. In particular, `result[0] == vec_start`."* |
| 10 | **MISSING** | `berwald_transport` (`transport.py:60-64`) | Public convenience function has **no docstring at all**. It is exported in `ham.__init__` and `ham.geometry.__init__`, making it a primary entry point. | `"""Transport a vector along a path using the Berwald connection.\n\nConvenience wrapper around ``BerwaldConnection.parallel_transport``.\nSee ``spec/MATH_SPEC.md § 3.2`` for the transport equation.\n\nArgs:\n    metric: FinslerMetric whose spray defines the connection.\n    path_x: Discrete positions along the curve, shape (T, D).\n    path_v: Velocities at each position, shape (T, D).\n    vec_start: Initial tangent vector to transport, shape (D,).\n\nReturns:\n    Transported vectors at each path point, shape (T, D).\n\nExample::\n\n    vecs = berwald_transport(metric, path_x, path_v, v0)\n"""` |
| 11 | **INACCURATE** | `berwald_transport` signature vs ARCH_SPEC (`transport.py:60` ↔ `spec/ARCH_SPEC.md § 4.3`) | The ARCH_SPEC shows signature `berwald_transport(metric, path, v0)` (3 positional args). The implementation takes `(metric, path_x, path_v, vec_start)` (4 positional args). The spec also describes the integration variable as `dv/dt`, while the code uses `dX/dt` for the transported vector — a different symbol. Either the spec or the code should be updated. | **Recommended Action:** Update `spec/ARCH_SPEC.md § 4.3` to match the implemented 4-argument signature and clarify variable naming. |
| 12 | **UNCLEAR** | `BerwaldConnection.parallel_transport` (`transport.py:49`) | `dt = 1.0 / len(path_x)` assumes unit-interval parameterisation $t \in [0,1]$. This is not stated anywhere. If the user supplies a path with a different time parameterisation (e.g., arc-length), the integration step size will be wrong. This assumption should be documented. | Add inline comment or docstring note: *"Assumes the curve is parameterised over [0, 1] with uniform spacing."* |
| 13 | **TYPO** | `BerwaldConnection` class docstring (`transport.py:22`) | Double backslash before `partial` (`\\partial`) — renders as a literal backslash in most docstring viewers. Should use single backslash or raw string. | Use `r"""..."""` raw docstring or single backslashes. |

---

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---------------|:---:|:---:|:---:|:---:|:---:|
| `Connection` | Yes (1-line) | No | N/A | No | No |
| `Connection.__init__` | No | No | N/A | N/A | No |
| `Connection.christoffel_symbols` | No | No | No | No | No |
| `Connection.parallel_transport` | No | No | No | No | No |
| `BerwaldConnection` | Yes | N/A | N/A | Yes (partial) | No |
| `BerwaldConnection.christoffel_symbols` | No | No | No | No | No |
| `BerwaldConnection.parallel_transport` | Partial | No | No | Yes (ASCII only) | No |
| `berwald_transport` | No | No | No | No | No |

---

## Spec Alignment Notes

1. **Signature mismatch** (`spec/ARCH_SPEC.md § 4.3`): The spec pseudocode shows `berwald_transport(metric, path, v0)` with 3 positional arguments. The implementation at [transport.py](src/ham/geometry/transport.py#L60) accepts `(metric, path_x, path_v, vec_start)` — 4 positional arguments. The separation of positions and velocities is arguably better design (velocities may not be trivially recoverable from discrete positions), but the spec must be updated to reflect it.

2. **Variable naming** (`spec/ARCH_SPEC.md § 4.3`): The spec describes the ODE as `dv/dt = -Gamma * v * dx/dt`, reusing `v` for both the transported vector and path velocity. The implementation correctly uses distinct names (`carry_vec` / `X` for the transported vector, `v` for path velocity), matching `spec/MATH_SPEC.md § 3.2` which uses $X^i$ and $\dot\gamma^j$. The ARCH_SPEC should adopt the same distinction.

3. **Connection notation** (`spec/MATH_SPEC.md § 3.1`): The spec uses the decorated notation $^B\Gamma^i_{jk}$ to distinguish the Berwald connection. The code docstrings use undecorated $\Gamma^i_{jk}$, which could be confused with the Levi-Civita or Chern connection by a differential-geometer audience.

4. **No example scripts**: Neither the `examples/` directory nor any docstring references an example of parallel transport. Given that transport is a validated feature (`spec/ARCH_SPEC.md § 6 — "Parallel Transport: Berwald connection verified"`), a minimal example would be valuable for both audiences.
