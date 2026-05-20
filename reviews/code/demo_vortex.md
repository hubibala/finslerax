# Code Review: `examples/demo_vortex.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0

## Summary

A clean, focused demo script that compares AVBD solver stiffness on a Randers sphere with a strong vortex wind field. The code is readable and correctly uses the HAM API. There are no bugs or crash-level issues. Findings are limited to an unused import, minor clarity improvements, and a numerical-stability edge case in the vortex helper.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **STYLE** | `examples/demo_vortex.py:9` | `Euclidean` is imported from `ham.geometry.zoo` but never used anywhere in the script. | Remove `Euclidean` from the import: `from ham.geometry.zoo import Randers`. |
| 2 | **RISK** | `examples/demo_vortex.py:16` | `jnp.cross(center, x)` returns the zero vector when `x` is parallel or anti-parallel to `center`. The exponential decay drives `magnitude` to near-zero at `center` itself, but at the antipode ($\text{dist} = \pi$) the decay factor is $e^{-5\pi^2} \approx 0$, so the product `magnitude * v_rot` is `0 * 0`. Numerically this is harmless for this demo's parameters, but for lower `decay` values the zero-cross-product produces a discontinuous field. | Add a comment noting the assumption that `decay` is large enough to suppress the antipodal singularity, or normalize `v_rot` with a safe norm before scaling. |
| 3 | **STYLE** | `examples/demo_vortex.py:5-6` | `config.update("jax_enable_x64", True)` is a module-level global side-effect. Any script that `import`s this module (even accidentally) will switch JAX to 64-bit globally. | Move inside the `if __name__ == "__main__":` guard, or add a comment explaining the intentional global scope. |
| 4 | **STYLE** | `examples/demo_vortex.py:28-29` | `end` is assigned on two consecutive lines — first to a raw array, then overwritten with its normalized form. This is correct but reads as if line 28 is dead code at first glance. | Combine into a single expression: `end = jnp.array([-0.99, 0.1, 0.0]); end = end / jnp.linalg.norm(end)`, or use a temporary variable name like `end_raw`. |
| 5 | **STYLE** | `examples/demo_vortex.py:26` | `h_net = lambda x: jnp.eye(3)` — the unused parameter `x` may trigger linter warnings. Minor, but an underscore convention is clearer. | Use `h_net = lambda _x: jnp.eye(3)` or `h_net = lambda x: jnp.eye(3)  # constant flat metric`. |

## Test Coverage Assessment

This is a standalone demo script, not a library module; it has no associated test file. No public API is defined. Coverage assessment is not applicable.

However, the library components it exercises (`Randers`, `AVBDSolver`, `Sphere`, `vis` utilities) are tested in their respective test files (`test_zoo.py`, `test_solver.py`, `test_metric.py`).

## Positive Patterns

- **Clean main-guard structure** — all work is inside `main()` with a proper `if __name__` guard.
- **Good use of `jnp.clip` inside `jnp.arccos`** (line 15) — prevents NaN from floating-point rounding beyond ±1. This is a numerically defensive best practice.
- **Clear comments** — each section (Setup, Mission, Solver Comparison, Visualization) is labelled, making the demo easy to follow.
- **Correct use of `jax.vmap`** (line 44) for batch-evaluating the wind field over icosphere points — idiomatic JAX.
- **Solver comparison is pedagogically effective** — contrasting `beta=20` vs `beta=0.5` directly demonstrates the stiffness concept.
