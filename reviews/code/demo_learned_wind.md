# Code Review: `examples/demo_learned_wind.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2025-05-15  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

This demo script trains a `NeuralRanders` metric to recover a Rossby–Haurwitz wind field on the sphere from sampled trajectory data. Overall it is well-structured and readable for an example script. Two substantive issues stand out: (1) the loss is computed **twice** per training step because `eqx.filter_grad` is used instead of `eqx.filter_value_and_grad`, doubling the forward-pass cost; and (2) evaluation metrics labelled "Training set" are actually computed on an icosphere grid, not on the training data, producing misleading diagnostics.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `examples/demo_learned_wind.py:68–71` | `loss_fn(model)` is called a second time on line 71 after already being evaluated inside `eqx.filter_grad` on line 68. This **doubles the forward-pass cost** (including the expensive `jacfwd` + `vmap` Jacobian computation) for every training step. | Replace `eqx.filter_grad` with `eqx.filter_value_and_grad` and return the loss from the same call: `loss, grads = eqx.filter_value_and_grad(loss_fn)(model)`. |
| 2 | **RISK** | `examples/demo_learned_wind.py:100–103` | The variables `cos_sim_train` and `mse_train` and the comment `# Training set` are misleading. These metrics are computed on `grid_pts` from `generate_icosphere`, which is an evaluation grid — not the actual training points `X`. This could lead a user to misinterpret generalisation performance. | Rename to `cos_sim_grid` / `mse_grid` and update the comment and print statement to say "Grid evaluation" instead of "Training set". Alternatively, actually compute metrics on the training set `X`. |
| 3 | **RISK** | `examples/demo_learned_wind.py:61, 96` | Direct access to the private method `_get_zermelo_data` (underscore-prefixed). While this is a pattern used elsewhere in the codebase, it couples the demo to an internal implementation detail. If `Randers` refactors this method, all examples break silently. | Consider exposing a public `get_wind(x)` convenience method on `NeuralRanders`, or at minimum add a comment acknowledging the private-API dependency. |
| 4 | **RISK** | `examples/demo_learned_wind.py:64–66` | `jax.jacfwd` is called inside `jax.vmap` inside `eqx.filter_jit`. For 512 samples × 3×3 Jacobians this works, but scaling `N_samples` up will cause OOM because `vmap(jacfwd(...))` materialises the full batched Jacobian tensor eagerly. No warning or guard is present. | Add a comment noting the memory scaling, or chunk the Jacobian computation for larger sample counts. |
| 5 | **STYLE** | `examples/demo_learned_wind.py:7–8` | `config.update("jax_enable_x64", True)` is executed at module-import time as a top-level side effect. While acceptable for a standalone script, it can cause surprising behaviour if the module is ever imported from elsewhere. | Move the `config.update` call inside `main()` before any JAX operations, or guard it with `if __name__ == "__main__"`. |
| 6 | **STYLE** | `examples/demo_learned_wind.py:1–8` | Imports are not grouped according to PEP 8. `numpy`, `equinox`, and `optax` are third-party but interleaved with stdlib-adjacent JAX imports. The `from jax import config` + immediate call breaks the import block. | Group imports: (1) stdlib, (2) third-party (`jax`, `jnp`, `numpy`, `equinox`, `optax`, `matplotlib`), (3) local (`ham.*`). Place the `config.update` call after all imports or inside `main()`. |
| 7 | **STYLE** | `examples/demo_learned_wind.py:75` | The comment `# 5. Visualization` skips section 4, suggesting a deleted or renumbered section. | Renumber to sequential `# 4. Visualization` or add the missing `# 4.` section header (e.g., for the training loop). |

## Test Coverage Assessment

This is a demo/example script, so it does not have a dedicated test file. Coverage considerations:

| Aspect | Status |
|--------|--------|
| `NeuralRanders` unit tests | Covered by `tests/test_learned_metric.py` |
| `rossby_haurwitz` unit tests | Covered by `tests/test_fields.py` |
| `Sphere.random_sample` | Covered by `tests/test_surfaces.py` |
| Visualization helpers (`setup_3d_plot`, etc.) | No dedicated tests (visual utilities) |
| End-to-end smoke test of this script | **Gap** — no CI job or test runs this demo to check for regressions |

**Recommended Action:** Add a lightweight smoke test (e.g., run 10 training steps and assert the loss is finite) to prevent import or API breakage from going unnoticed.

## Positive Patterns

- **Clean `main()` guard:** The script is properly wrapped in `if __name__ == "__main__": main()`, preventing side effects on import (modulo the `config.update` issue).
- **Held-out evaluation:** The script generates a separate test set (`X_test`) with fresh PRNG keys to evaluate generalisation, which is good scientific practice for a demo.
- **Numerical safety in cosine similarity:** Both cosine-similarity computations include a `+ 1e-8` denominator guard against division by zero.
- **Jacobian regularisation:** The smoothness loss via `jacfwd` is a well-motivated and correctly implemented regulariser for the wind field — a good pedagogical pattern.
- **Proper PRNG threading:** The `key` is split correctly at each stage with no key reuse.
