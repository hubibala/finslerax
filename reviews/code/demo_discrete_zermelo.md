# Code Review: `examples/demo_discrete_zermelo.py`

**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-15  
**Arch Spec Version:** 1.1.0

## Summary

This demo script is a clear, well-structured example showing continuous-vs-discrete Finsler geodesics on $S^2$ with Zermelo wind. It successfully demonstrates the library's end-to-end workflow. However, it has a cross-metric energy evaluation bug that will produce misleading output, a missing `plt.close()` resource leak, and several style issues that could confuse new users.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `examples/demo_discrete_zermelo.py:35` | `batch_energy = jax.vmap(metric_randers.energy)` is used to evaluate the Riemannian trajectory's energy (`e_riem`). This computes the Randers energy of the Riemannian path, not the Riemannian energy of the Riemannian path. The printed comparison is therefore not apples-to-apples: it evaluates both paths under the Randers cost, which may be the intent (to show the Randers path is cheaper under the Randers metric), but the label `"Energy Riemannian path"` is misleading and will confuse users. | Either (a) rename the print label to `"Randers energy of Riemannian path"`, or (b) compute `e_riem` using `metric_riem.energy` and add a comment explaining the comparison semantics. |
| 2 | **RISK** | `examples/demo_discrete_zermelo.py:55` | `batch_energy(traj_mesh.xs[:-1], traj_mesh.vs)` evaluates discrete mesh trajectory points through the *continuous* Randers energy function (`metric_randers.energy`). Mesh trajectory points may not lie exactly on the unit sphere, causing the continuous metric evaluation to be inaccurate. This cross-domain energy comparison is fragile. | Add a comment explaining the cross-evaluation intent, or compute discrete energy using `metric_discrete_randers.energy`. |
| 3 | **RISK** | `examples/demo_discrete_zermelo.py:14–17` | `w_net` and `h_net` are plain Python functions, not `eqx.Module`s. While this works for non-learned metrics, passing bare closures makes this example non-portable to JIT-compiled workflows. If a user copies this pattern and tries `jax.jit(solver.solve)(...)`, these closures will be traced as abstract values and may cause errors or recompilation. | Add a brief comment: `# Note: for JIT, wrap these in eqx.Module or use functools.partial`. |
| 4 | **STYLE** | `examples/demo_discrete_zermelo.py:1–4` | Imports do not follow PEP 8 grouping: `jax.numpy` and `jax` (third-party) are mixed, and `numpy` appears after `matplotlib`. Standard convention: stdlib → third-party → local, with blank lines between groups. | Reorder to: `import numpy as np` / `import jax` / `import jax.numpy as jnp` / `import matplotlib.pyplot as plt` / blank line / `from ham...` |
| 5 | **STYLE** | `examples/demo_discrete_zermelo.py:76` | `plt.savefig("zermelo_demo.png")` saves to the current working directory with no indication to the user. If run from a different directory, the output location is surprising. No `plt.close(fig)` follows `plt.show()`, leaking the figure resource. | Add `plt.close(fig)` after `plt.show()`, and consider using `pathlib.Path(__file__).parent / "visualizations" / "zermelo_demo.png"` or printing the save path. |
| 6 | **STYLE** | `examples/demo_discrete_zermelo.py:60` | The f-string uses `\\n` for a newline in the title, but this renders as a literal backslash-n in some backends (e.g., Agg). Use `\n` (actual newline) instead. | Change to: `ax.set_title(f"Zermelo S^2: Discrete Matches Continuous\nEnergy: {e_rand:.2f} (Cont) vs {e_mesh:.2f} (Disc)")`. |
| 7 | **STYLE** | `examples/demo_discrete_zermelo.py:20` | `lambda x: jnp.zeros(3)` hard-codes dimension 3. Although this is a sphere demo and correct, it's fragile if users adapt for other manifolds. | Use `lambda x: jnp.zeros_like(x)` for dimension-agnostic zero wind. |
| 8 | **STYLE** | `examples/demo_discrete_zermelo.py:1–77` | No module docstring. Example scripts are the primary onramp for new users; a top-level docstring explaining what this demo does (continuous vs. discrete Finsler geodesics under Zermelo wind on $S^2$) and how to run it would significantly improve discoverability. | Add a module docstring at line 1. |
| 9 | **STYLE** | `examples/demo_discrete_zermelo.py:19` | `if __name__ == "__main__":` guard is missing. Running this file on import (e.g., for testing or introspection) will trigger computation and open a plot window. | Wrap lines 11–77 in an `if __name__ == "__main__":` block. |

## Test Coverage Assessment

This is an example/demo script, so it has no dedicated test file. However:

| Aspect | Covered? | Gap |
|--------|----------|-----|
| `AVBDSolver.solve` | Yes, in `tests/test_solver.py` | — |
| `Randers` metric | Yes, in `tests/test_zoo.py` | — |
| `DiscreteRanders` metric | Partially, in `tests/test_zoo.py` | `DiscreteRanders` is not imported in test_zoo.py; no dedicated test exists. |
| `TriangularMesh` | Yes, in `tests/test_mesh.py` | — |
| `generate_icosphere` | No dedicated test | Gap: no test for vertex count, normals, or face consistency. |
| End-to-end discrete geodesic pipeline | No | Gap: no integration test exercises the full discrete Zermelo pipeline. |

## Positive Patterns

- **Clear section headers** (`# --- 1. Continuous Physics ---`, etc.) make the script easy to follow.
- **Side-by-side comparison** of continuous Riemannian, continuous Randers, and discrete Randers paths is an excellent pedagogical structure.
- **Correct use of `jax.vmap`** for batched energy computation.
- **Wind visualization** on the equator provides good physical intuition for what the Randers term does.
- **Indicatrix plotting** shows the anisotropy, reinforcing the Finsler concept visually.
