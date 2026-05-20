# Code Review: `examples/weinreb_vae.py`

**Reviewer:** Code Reviewer Agent
**Date:** 2025-05-15
**Arch Spec Version:** 1.1.0

## Summary

The script is a well-structured, purpose-built VAE training harness with clear docstrings and deliberate design choices (tanh decoder, cyclic KL, deterministic geometric path). However, it contains one likely **BUG** in data-dimension inference, several numerical **RISK** items (division-by-near-zero patterns, unbounded PCA inverse, unguarded `jnp.where` under `grad`), and a handful of **STYLE** issues (import placement, dead code, missing `__all__`). No architectural violations of `spec/ARCH_SPEC.md` were found.

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `examples/weinreb_vae.py:436–439` | `data_dim` is computed from `dataset.X.shape[0]` first (gives the number of *cells*, not features, for 2D arrays), then corrected on the next line. If `dataset.X` is ever 1-D (edge case) the first assignment silently wins and `data_dim` becomes the feature count by accident, but with a misleading code path. More critically, the first expression `dataset.X.shape[0]` is always wrong for the normal 2-D layout and only works because it is immediately overwritten. | Remove the dead first assignment; use `data_dim = dataset.X.shape[-1]` directly, which is correct for both 1-D and 2-D. |
| 2 | **RISK** | `examples/weinreb_vae.py:299` | `VelocityConsistencyLoss.__call__` computes `cos_sim_data = jnp.sum(u_i * u_j, axis=-1) / (safe_norm(u_i) * safe_norm(u_j))` where `safe_norm` adds `1e-8` inside the sqrt but the *product* of two small norms can still underflow to ≈0, producing `inf` or `NaN` in the division. Under `jax.grad` the backward pass through the division amplifies the instability. | Use `jnp.where(denom > eps, ratio, 0.0)` with `eps = 1e-6`, or compute via `jnp.arctan2`-based angle. |
| 3 | **RISK** | `examples/weinreb_vae.py:210–213` | `TrajectoryCoherenceLoss`: `norm_early` and `norm_full` use `+ 1e-8` inside `jnp.sqrt`, which is correct for forward evaluation. However, when all three time-points collapse to the same latent point (early training), both vectors are zero and `v_en`, `v_fn` are `0/sqrt(1e-8)` — a random direction with magnitude ≈ 3162. The cosine similarity becomes meaningless, and the gradient is large. | Gate with `jnp.where(norm > eps, normalised, 0.0)` to zero out the contribution from degenerate samples. |
| 4 | **RISK** | `examples/weinreb_vae.py:537–541` | The lineage-triple indexing uses `step * batch_size % n_lin` which can wrap mid-epoch, creating overlapping indices across steps and non-uniform sampling. When `n_lin < steps * batch_size`, some triples are visited multiple times per epoch while others are skipped. | Use a full permutation with wrap-around similar to the KNN triplet branch, or simply resample with `rng.choice` each step. |
| 5 | **RISK** | `examples/weinreb_vae.py:248` | `AnnealedKLLoss.__call__` accepts `beta` as an extra kwarg beyond the `LossComponent.__call__` signature `(self, model, batch, key)`. This works at the call-site inside `train_step` but breaks the generic loss-component contract. If the pipeline ever calls `kl_loss(model, batch, key)` without `beta`, the KL contribution is silently zero because `beta` defaults to `0.0`. | Make `beta` a mutable attribute set before the step (e.g. `kl_loss = eqx.tree_at(lambda l: l.weight, kl_loss, beta)`) or wrap the call in a partial. |
| 6 | **RISK** | `examples/weinreb_vae.py:640–641` | `compute_pullback_det`: `pts_full = pca2.inverse_transform(pts2d)` projects 2-D grid points back into the full PCA space, but the inverse only recovers the 2-component subspace; all other components are set to the training-data mean. For a latent dim ≫ 2 this can place grid points far from the data manifold, where the decoder Jacobian (and hence `det G`) is unreliable. No warning or clamping is applied. | Add a note/warning in the output, or project grid points onto the nearest data point's full-latent coordinates via KNN lookup. |
| 7 | **RISK** | `examples/weinreb_vae.py:649` | `logdet_at` adds `1e-6 * jnp.eye(latent_dim)` regularisation to `G`. This shifts all eigenvalues by `1e-6`, which is fine numerically but changes `det G` by a factor of `(1 + 1e-6/λ_i)` for each eigenvalue `λ_i`. For near-singular Jacobians (small `λ_i`), this multiplicatively inflates `log det G`, masking actual degeneracies. The regularisation should be documented or made controllable. | Add a parameter `jitter` and document the bias it introduces in the diagnostic. |
| 8 | **RISK** | `examples/weinreb_vae.py:654–656` | `logdet_at` is called in a Python `for` loop over `pts_full` (up to 625 points for `n_grid=25`). Each call triggers a separate JIT compilation or dispatch. This is extremely slow compared to `vmap`. | Use `eqx.filter_vmap` over the grid points. |
| 9 | **STYLE** | `examples/weinreb_vae.py:1–14` | Module docstring refers to the file as `weinreb_vae_diagnostic.py`, but the actual filename is `weinreb_vae.py`. | Update the docstring header to match the filename. |
| 10 | **STYLE** | `examples/weinreb_vae.py:771` | `from matplotlib.lines import Line2D` is imported inside `plot_diagnostics`. All other matplotlib imports are at module level. | Move to the top-level import block for consistency. |
| 11 | **STYLE** | `examples/weinreb_vae.py:791–793` | `from sklearn.metrics import silhouette_samples` is imported inside the `try` block of `plot_diagnostics`. It is already partially imported at module level (`silhouette_score`). | Move to the top-level import block alongside `silhouette_score`. |
| 12 | **STYLE** | `examples/weinreb_vae.py:362` | `build_diagnostic_vae` imports `from ham.bio.vae import GeometricVAE` at function scope. This is the only deferred import in the file; all other HAM imports are top-level. | Move to top-level imports for consistency, unless there is a circular-import reason (none apparent). |
| 13 | **STYLE** | `examples/weinreb_vae.py:73` | `make_tanh_mlp` is a thin one-liner wrapper around `eqx.nn.MLP`. It is called exactly once (line 377). Consider inlining or documenting why the wrapper exists for discoverability. | Inline at the call-site, or add a comment explaining the wrapper is kept for potential reuse. |
| 14 | **STYLE** | `examples/weinreb_vae.py:419–424` | `get_trainable_mask` builds the mask with a nested `make_true` helper that manually checks `eqx.is_array`. The Equinox-idiomatic approach is `eqx.is_array` as the filter spec to `eqx.partition` directly with `eqx.is_array`. | Replace with `eqx.partition(model, eqx.is_array)` if the intent is to train all arrays, or use `filter_spec` on the sub-trees. |

## Test Coverage Assessment

This is an `examples/` script, not a library module, so it has no dedicated test file in `tests/`. However, several of the custom loss classes defined here (`KNNTripletLoss`, `TrajectoryCoherenceLoss`, `AnnealedKLLoss`, `CellTypeClassificationLoss`, `VelocityConsistencyLoss`, `ReconstructionLossDeterministic`) are **not tested anywhere** in the test suite. They are only exercised through end-to-end runs of this script.

| Symbol | Tested? | Gap |
|--------|---------|-----|
| `make_tanh_mlp` | No | Trivial wrapper — low priority |
| `build_knn_triplet_indices` | No | Complex logic (KNN + sampling) — should have a unit test |
| `KNNTripletLoss` | No | Should verify margin semantics and gradient flow |
| `TrajectoryCoherenceLoss` | No | Should verify degenerate-triple handling (Issue #3) |
| `AnnealedKLLoss` | No | Should verify `beta=0` ⇒ zero loss, `beta>0` ⇒ positive |
| `CellTypeClassificationLoss` | No | Should verify log-softmax stability with small-init weights |
| `VelocityConsistencyLoss` | No | Should verify zero-velocity masking and norm stability |
| `ReconstructionLossDeterministic` | No | Should verify both stochastic and deterministic paths |
| `build_diagnostic_vae` | No | Integration-level — exercised by `main()` |
| `cyclic_beta` | No | Pure function — easy to unit-test ramp and plateau phases |
| `train_vae` | No | Integration-level |
| `encode_all` | No | Should verify batching and concatenation |
| `compute_pullback_det` | No | Should verify grid construction and Jacobian computation |
| `knn_preservation_score` | No | Pure function — easy to unit-test |

**Recommended action:** Extract `KNNTripletLoss`, `TrajectoryCoherenceLoss`, `VelocityConsistencyLoss`, `ReconstructionLossDeterministic`, and `cyclic_beta` into `src/ham/training/losses.py` (or a dedicated `bio_losses.py`) and add unit tests. This would also resolve the style issue of defining library-grade loss components inside an example script.

## Positive Patterns

1. **Deterministic encoding for geometric losses** (line 161 and throughout): Using `dist.mean` instead of `dist.sample` for all non-reconstruction losses eliminates sampling noise from geometry-sensitive computations. This is a well-motivated design decision.
2. **Classifier weight scaling** (lines 386–392): Scaling classifier weights by 0.01 at init prevents `log_softmax → -inf → NaN` on the first forward pass. This is a practical fix for a common failure mode.
3. **Cyclic KL annealing** (lines 229–235): The saw-tooth β schedule is a well-known technique to prevent posterior collapse. The implementation is clean and easy to tune.
4. **Velocity scaling** (lines 930–931): Dividing `V_pca` by `scaler.scale_` (without mean subtraction) correctly preserves the velocity zero-mean property. This shows careful reasoning about the data semantics.
5. **Deferred pullback metric** (lines 396–401): Training with `Euclidean` metric and attaching `PullbackRiemannian` only at evaluation avoids expensive `jacfwd` in the training loop. This is a pragmatic performance decision, well-documented.
6. **Per-sample normalisation in `TrajectoryCoherenceLoss`** (lines 210–215): The fix comment and implementation correctly address the original global-normalisation bug.
