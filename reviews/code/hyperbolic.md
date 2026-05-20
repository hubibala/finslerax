# Code Review: `ham.vis.hyperbolic`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

`src/ham/vis/hyperbolic.py` is a small (98-line) visualization utility providing Hyperboloid → Poincaré Ball projection and a Matplotlib plotting helper. The projection math is implemented correctly and handles arbitrary batch shapes via `...` indexing. However, the module has **zero test coverage**, a missing-figure-handle return path, and no input guards against invalid hyperboloid points that would cause division-by-zero. None of these are crashers under normal usage, but the lack of tests and missing exports leave gaps.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `src/ham/vis/hyperbolic.py:53` | When `ax is None`, a `fig` object is created (`fig, ax = plt.subplots(...)`) but the function only returns `ax`. The caller loses the figure handle and cannot call `fig.savefig()` without resorting to `plt.gcf()`. The example in `examples/demo_weinreb_vis.py` works around this by passing its own `ax`. | Return `(fig, ax)` when the figure is created internally, or always return `ax` but document that the caller must provide their own `fig, ax`. |
| 2 | **RISK** | `src/ham/vis/hyperbolic.py:13-14` | `project_to_poincare` divides by `(1.0 + x0)`. For valid upper-sheet hyperboloid points $x_0 \geq 1$, so $1+x_0 \geq 2$ and this is safe. However, there is no validation; passing lower-sheet points ($x_0 < 0$), or the degenerate value $x_0 = -1$, would produce a division by zero. | Add a `jnp.clip(x0, a_min=eps)` guard or validate `x0 > 0` with a warning/assertion at the call site. |
| 3 | **RISK** | `src/ham/vis/hyperbolic.py:30` | `project_vector_to_poincare` divides by `(1.0 + x0)**2`. Same unguarded denominator concern as #2, squared. | Same fix as #2 — guard the denominator or validate inputs. |
| 4 | **RISK** | `src/ham/vis/hyperbolic.py:75` | `np.random.choice(len(points), 500, replace=False)` uses the global NumPy RNG without a seed. Subsampled quiver arrows are non-reproducible across runs, making visual debugging harder. | Accept an optional `rng` or `seed` parameter, or use `np.random.default_rng(seed)`. |
| 5 | **STYLE** | `src/ham/vis/__init__.py:1-2` | `project_to_poincare` and `project_vector_to_poincare` are not exported in `__init__.py` or `__all__`, despite being useful public utilities (e.g., for users building custom plots). | Add both to the `__init__.py` imports and `__all__`. |
| 6 | **STYLE** | `src/ham/vis/hyperbolic.py:6-31` | `project_to_poincare` and `project_vector_to_poincare` have no mention of the batch-first convention (`(B, D+1)` → `(B, D)`). The `...` indexing handles arbitrary leading dims correctly, but per `spec/ARCH_SPEC.md § 1` ("Batch-First"), the docstrings should explicitly state the batch shape. | Add `(B, D+1) -> (B, D)` to the docstring `Args/Returns` section. |
| 7 | **STYLE** | `src/ham/vis/hyperbolic.py:1-4` | `jax.numpy` is imported but the two projection functions could work with plain NumPy. In `plot_poincare_disk`, the JAX array returned by `project_to_poincare` is immediately wrapped in `np.array(...)`. For a pure visualization module, this forces a JAX dependency at import time even when users only want to plot pre-computed data. | Minor: could accept both array types, or document that JAX is required. Not a blocking issue. |

## Test Coverage Assessment

| Public Function | Tested? | Notes |
|-----------------|---------|-------|
| `project_to_poincare` | **NO** | No test file exists for `vis/hyperbolic.py`. Should verify round-trip with known hyperboloid points (e.g., origin $(1,0,0)$ → $(0,0)$), boundary behaviour, and `jit`/`vmap` compatibility. |
| `project_vector_to_poincare` | **NO** | Should verify pushforward at origin produces identity-scaled tangent, and that result is consistent with the Riemannian metric conformal factor. |
| `plot_poincare_disk` | **NO** | Smoke test should verify it runs without error for various argument combos (`vectors=None`, `lineage_pairs=None`, custom `ax`, etc.) and check returned `ax` type. |

**Gap:** This module has **0 % test coverage**. Recommended action: create `tests/test_hyperbolic_vis.py` covering at least:
- Origin projection: $(1, 0, 0) \mapsto (0, 0)$.
- Known point: $(cosh(r), sinh(r), 0) \mapsto (tanh(r/2), 0)$.
- Vector pushforward at origin.
- `jax.jit(project_to_poincare)` works.
- `jax.vmap(project_to_poincare)` works.
- `plot_poincare_disk` smoke test with and without optional args.

## Positive Patterns

1. **Correct ellipsis indexing** — `x[..., 0:1]` and `x[..., 1:]` correctly handle arbitrary batch dimensions without hardcoding shape, which is idiomatic JAX.
2. **Clean pushforward derivation** — `project_vector_to_poincare` implements the differential of stereographic projection correctly with a clear inline comment.
3. **Sensible defaults** — `plot_poincare_disk` draws the unit disk boundary, sets equal aspect ratio, and auto-subsamples dense quiver plots for readability.
4. **Separation of concerns** — Projection math (pure JAX functions) is cleanly separated from Matplotlib plotting logic.
