# Code Review: `examples/plot_publication_figs.py`

**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-15
**Arch Spec Version:** 1.1.0

## Summary

This plotting script is well-structured for a publication figure generator, with clear panel separation and helpful docstrings. However, it contains several unused imports and a dead function, a redundant JIT-compiled computation, a dead-code variable, broad exception handlers that silently swallow errors, and missing figure cleanup. No critical correctness bugs were found, but one finding (redundant arc-length computation hiding a copy-paste error pattern) warrants careful attention.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **STYLE** | `examples/plot_publication_figs.py:26` | `FancyArrowPatch` is imported but never used anywhere in the file. | Remove: `from matplotlib.patches import FancyArrowPatch`. |
| 2 | **STYLE** | `examples/plot_publication_figs.py:38` | `GeometricVAE` is imported but never referenced. | Remove from the import line. |
| 3 | **STYLE** | `examples/plot_publication_figs.py:39` | `DataDrivenPullbackRanders` is imported but never referenced (the metric is attached via `attach_datadriven_randers_metric`). | Remove from the import line. |
| 4 | **STYLE** | `examples/plot_publication_figs.py:40` | `FinslerMetric` is imported but never referenced. | Remove from the import line. |
| 5 | **STYLE** | `examples/plot_publication_figs.py:41` | `build_diagnostic_vae` is imported from `weinreb_vae` but never called. | Remove from the import list. |
| 6 | **STYLE** | `examples/plot_publication_figs.py:118` | `find_exemplar_triple` is defined (lines 118–135) but never called. Panel C re-implements the exemplar search inline in `main()`. | Either delete the dead function or refactor Panel C to call it. |
| 7 | **BUG** | `examples/plot_publication_figs.py:253` | `quiv_colors = cm.copper(mag_norm(mags_grid.ravel()))` computes a color array that is never used. The subsequent `ax_wind.quiver(...)` call passes `mags_grid` and `cmap="copper"` directly, so `quiv_colors` is dead code. | Remove the `quiv_colors` line. |
| 8 | **RISK** | `examples/plot_publication_figs.py:349` | `e_cf1 = float(arc_fn(vae_randers.metric, z2, z4))` duplicates the computation of `e_obs1` (line 347). While mathematically correct (both paths share the z2→z4 segment), the redundant JIT call is wasteful and the naming pattern (`e_cf1`) suggests the author may have intended to compute a different segment. If the paths ever diverge from z2, this will silently produce wrong annotations. | Replace with `e_cf1 = e_obs1` and add a comment explaining that the first segment is shared. |
| 9 | **RISK** | `examples/plot_publication_figs.py:217`, `:258` | Bare `except Exception as e` blocks silently swallow all errors (including `RuntimeError`, `MemoryError`, shape mismatches) and only print a one-line warning. A user would not know their figure panel is blank due to, e.g., an OOM error. | Catch specific exceptions (e.g., `jax.errors.TracerArrayConversionError`, `ValueError`) or at minimum log the full traceback with `traceback.print_exc()`. |
| 10 | **RISK** | `examples/plot_publication_figs.py:70` | `logdet_at` is called in a Python `for` loop (line 72: `[float(logdet_at(vae, jnp.array(z))) for z in pts_full]`), processing one grid point per iteration. For `n_grid=35` this is $35^2 = 1225$ individual JIT dispatch calls, which is extremely slow. | Refactor `logdet_at` with `eqx.filter_vmap` to process all grid points in a single batched call, analogous to the pattern already used in `compute_wind_grid`. |
| 11 | **RISK** | `examples/plot_publication_figs.py:301–306` | `get_means` is defined with `@eqx.filter_jit` / `@eqx.filter_vmap` inside a loop body, re-creating the closure each outer iteration. While JAX's tracing cache typically prevents re-compilation, defining JIT-compiled functions inside loops is fragile and confusing. Additionally, each inner-loop iteration (lines 308–315) encodes only 1 cell at a time despite the `filter_vmap` decorator — the batch dimension is always size 1. | Hoist `get_means` out of both loops. Batch-encode all candidate cells at once before the loop, then index into the results. |
| 12 | **STYLE** | `examples/plot_publication_figs.py:383` | `from matplotlib.lines import Line2D` is imported inside `main()` instead of at module top. | Move to the module-level imports section. |
| 13 | **RISK** | `examples/plot_publication_figs.py:397` | No `plt.close(fig)` call after `plt.savefig(...)`. With `matplotlib.use("Agg")`, the figure object and its backing buffers are never released. In a one-shot script this is benign, but if `main()` is ever called in a loop or from a notebook, it will leak memory. | Add `plt.close(fig)` after `plt.savefig(...)`. |
| 14 | **STYLE** | `examples/plot_publication_figs.py:36` | `sys.path.insert(0, os.path.dirname(__file__))` is a brittle path hack. If the file is invoked from a different working directory or via symlink, the relative imports from `weinreb_vae` and `weinreb_experiment` may still fail. | Consider using a `pyproject.toml` `[project.scripts]` entry-point or relative imports from a package structure instead. |
| 15 | **STYLE** | `examples/plot_publication_figs.py:140–142` | File paths (`CHECKPOINT`, `PREPROCESSED`, `TEST_TRIPLES`) are hardcoded relative strings. The script will fail silently or with a confusing error if run from any directory other than the project root. | Either resolve relative to `__file__` (e.g., `os.path.join(os.path.dirname(__file__), "..", "data", ...)`) or accept paths via `argparse`. |
| 16 | **RISK** | `examples/plot_publication_figs.py:66` | `latent_dim` is captured as a closure variable by `logdet_at` but originates from a non-JAX Python scope (`Z_all.shape[1]`). This works only because `latent_dim` is a static integer known at trace time. If it were ever a traced value, `jnp.eye(latent_dim)` would fail inside JIT. The implicit closure over `latent_dim` is fragile. | Pass `latent_dim` as an explicit argument or use `jax.numpy.eye(z.shape[0])` to derive the dimension from the traced input. |

## Test Coverage Assessment

This is an `examples/` script, not a library module, so there is no corresponding test file in `tests/`. This is acceptable for a plotting script, but the helper functions (`compute_logdet_grid`, `compute_wind_grid`, `arc_length_segment`) perform non-trivial JAX computations that would benefit from unit tests:

| Function | Tested? | Notes |
|----------|---------|-------|
| `compute_logdet_grid` | No | Performs JIT'd Jacobian + slogdet computation; a smoke test with a toy model would catch shape/dtype issues. |
| `compute_wind_grid` | No | Uses `filter_vmap`; a minimal test would verify output shapes. |
| `arc_length_segment` | No | Delegates to `metric.arc_length`; a test with a known analytic metric would verify correctness. |
| `find_exemplar_triple` | No | Dead code (never called). |
| `main` | No | Integration-level; an end-to-end smoke test with mock data would catch regressions. |

## Positive Patterns

- **Clear panel structure.** The script is logically organized with section headers and print statements that make it easy to follow execution flow and diagnose which panel failed.
- **Percentile-based axis ranges.** Using `np.percentile([2, 98])` and `np.percentile([3, 97])` for grid extents (lines 52–53, 82–83) avoids outlier-dominated axis ranges — a best practice for biological data visualization.
- **Subsampling for speed.** The 20k-cell cap (line 178) is a pragmatic choice that prevents the plotting script from being bottlenecked by the full dataset.
- **Robust slogdet usage.** The sign check in `logdet_at` (line 68: `jnp.where(sign > 0, ld, -20.0)`) gracefully handles non-positive-definite Jacobian products, which can occur at the PCA grid boundary.
- **Good docstrings.** Both the module docstring and the helper functions have clear, concise documentation of inputs and outputs.
