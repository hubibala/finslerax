# Code Review: wildfire CNN encoder (`wildfire.py`, `experiment_wildfire_flat.py`, test suite)
**Reviewer:** Code Reviewer Agent
**Date:** 2026-05-18
**Arch Spec Version:** 1.1.0

---

## Summary

The `LocalTerrainCNN` encoder and its integration into `CovariateConditionedRanders` are broadly well-structured: the convolutional architecture is JAX/Equinox-idiomatic, the SPD and norm-bound projection helpers remain vmap/grad-safe, and the `precompute_metric_field` + bilinear-interpolation design cleanly separates expensive per-scene CNN inference from cheap per-point lookups. However, **two BUG-severity issues were found that actively corrupt training**. The most critical is that `precompute_metric_field` does not apply `stop_gradient` to terrain rasters; because the main `train_scene` loop passes a *terrain-bound* model as the differentiable argument to `eqx.filter_value_and_grad`, the optimizer computes and applies updates to `elev_raster`, `slope_raster`, `aspect_raster`, and `canopy_raster` on every training step, silently corrupting the scene data. A second BUG is that `scene_origin_xy` lacks the `stop_gradient` guard that is already correctly applied to `pixel_spacing_m`. Together these issues mean the Phase W1 training loop as currently implemented does **not** train purely the CNN/MLP weights; it also fits the terrain rasters, which undermines the scientific validity of the results.

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | [src/ham/models/wildfire.py](../src/ham/models/wildfire.py#L474-L492) (precompute_metric_field), [examples/experiment_wildfire_flat.py](../examples/experiment_wildfire_flat.py#L247-L252) (fire_loss / _single_fire_grad) | **Terrain rasters corrupted by optimizer.** `precompute_metric_field` feeds `self.elev_raster`, `self.slope_raster`, `self.aspect_raster`, and `self.canopy_raster` into the CNN without `stop_gradient`. In `make_batched_train_step`, `eqx.filter_value_and_grad(fire_loss)(metric)` is called on the *terrain-bound* model (lines 252 and 758–804). Because the rasters are `float64` leaves of `metric`, JAX propagates $\partial \mathcal{L} / \partial \text{raster}$ through the CNN input → the optimizer accumulates this gradient and applies it at line 289 via `eqx.apply_updates(metric, updates)`. After each step, `metric.elev_raster` etc. are corrupted with gradient-descent updates. The CNN weights see a moving target and the scene data is silently destroyed. | Apply `jax.lax.stop_gradient` to all raster inputs inside `precompute_metric_field`: `raster_stack = jnp.stack([jax.lax.stop_gradient(self.elev_raster), jax.lax.stop_gradient(self.slope_raster), jnp.sin(jax.lax.stop_gradient(self.aspect_raster)), jnp.cos(jax.lax.stop_gradient(self.aspect_raster)), jax.lax.stop_gradient(self.canopy_raster)], axis=0)`. This is the same pattern already used for `pixel_spacing_m` in the bilinear interp methods. |
| 2 | **BUG** | [src/ham/models/wildfire.py](../src/ham/models/wildfire.py#L502-L505) `_bilinear_interp`, [src/ham/models/wildfire.py](../src/ham/models/wildfire.py#L539-L542) `_bilinear_interp_field` | **`scene_origin_xy` not protected by `stop_gradient`.** `pixel_spacing_m` is correctly wrapped: `spacing = jax.lax.stop_gradient(self.pixel_spacing_m)` (lines 502, 539). But `scene_origin_xy` is used raw: `px = (x_world[0] - self.scene_origin_xy[0]) / spacing`. Since `scene_origin_xy` is a `float64` leaf, spurious gradients flow to it in the terrain-bound training path (same scenario as Issue 1). The optimizer updates the world-space grid origin during training. | Wrap in the same way: `origin = jax.lax.stop_gradient(self.scene_origin_xy)` and replace `self.scene_origin_xy[0/1]` with `origin[0/1]` in both methods. The class docstring already documents the intent for `pixel_spacing_m` (line 249); apply the same rationale to `scene_origin_xy`. |
| 3 | **RISK** | [src/ham/models/wildfire.py](../src/ham/models/wildfire.py#L569) `_get_params` | **`metric_field` not guarded against `None`.** `_get_params` calls `self._bilinear_interp_field(self.metric_field, x_world)` unconditionally. If `metric_fn` is invoked before `precompute_metric_field()` (e.g., during a smoke-test or after `bind_scene` resets the field), JAX will attempt dynamic indexing into `None` inside a JIT context, producing an opaque XLA error rather than a clear `AssertionError`. | Add a guard at the top of `_get_params`: `if self.metric_field is None: raise RuntimeError("precompute_metric_field() must be called before metric_fn()")`. Since this check is on a pytree-structural property it is safe outside JIT; inside a JIT context the error will surface at trace time. |
| 4 | **RISK** | [src/ham/models/wildfire.py](../src/ham/models/wildfire.py#L138-L145) `LocalTerrainCNN.__init__` | **`dtype=jnp.float64` hardcoded without x64 requirement documented.** All four `Conv2d` layers specify `dtype=jnp.float64`. Under default JAX (`jax_enable_x64=False`), float64 weights are silently downcast to float32 at JIT boundaries, causing dtype mismatches with float64 raster inputs. The docstring does not mention the x64 requirement. | Add a note to the class docstring: *"Requires `jax.config.update('jax_enable_x64', True)` before construction."* Optionally add a runtime assertion in `__init__`: `assert jax.config.x64_enabled, "LocalTerrainCNN requires x64 mode"`. |
| 5 | **RISK** | [src/ham/models/wildfire.py](../src/ham/models/wildfire.py#L506) `_bilinear_interp`, [src/ham/models/wildfire.py](../src/ham/models/wildfire.py#L543) `_bilinear_interp_field` | **Magic constant `1.001` in boundary clip causes edge-pixel aliasing.** `px = jnp.clip(px, 0.0, W - 1.001)` means a point at exactly the right edge is mapped to `W - 1.001`, so `x0 = W - 2` and the interpolation weights `fx = 0.999`. The boundary pixel is never cleanly sampled. For a 1-pixel grid (`W = 1`), the upper bound becomes `-0.001`, which after clipping to `[0, -0.001]` is undefined behaviour (upper < lower). | Replace `W - 1.001` and `H - 1.001` with `W - 1 - 1e-6` / `H - 1 - 1e-6` and document the rationale: *"Ensures floor(px) ≤ W-2 so x1 = x0+1 is always in-bounds."* Add a degenerate-grid assert: `assert H >= 2 and W >= 2` in `precompute_metric_field`. |
| 6 | **STYLE** | [src/ham/models/wildfire.py](../src/ham/models/wildfire.py#L310) `__init__` | **`fuel_embedding` zero-initialized.** `self.fuel_embedding = jnp.zeros((13, fuel_emb_dim), dtype=jnp.float64)` initialises all fuel-type embeddings to the same vector. This creates a gradient symmetry that delays the embedding from learning fuel-type distinctions early in training. | Initialise with small random noise: `self.fuel_embedding = jax.random.normal(k_emb, (13, fuel_emb_dim), dtype=jnp.float64) * 0.01` (requires splitting an extra key). |
| 7 | **STYLE** | [src/ham/models/wildfire.py](../src/ham/models/wildfire.py#L494-L526) `_bilinear_interp`, [src/ham/models/wildfire.py](../src/ham/models/wildfire.py#L528-L556) `_bilinear_interp_field` | **Duplicated pixel-coordinate logic.** Both methods repeat ~10 identical lines for `spacing → px/py → clip → floor → x0/y0/x1/y1/fx/fy`. If either method's boundary arithmetic changes, the other must be updated manually. | Extract `_world_to_pixel_coords(x_world) -> tuple[int, int, int, int, float, float]` as a private helper. |
| 8 | **STYLE** | [examples/experiment_wildfire_flat.py](../examples/experiment_wildfire_flat.py#L436-L480) `train_one_fire` | **`train_one_fire` is dead code.** This function implements the *correct* training pattern (binds scene inside the loss, no raster-gradient issue), but is never called in `train_scene`. The main training loop exclusively uses `make_batched_train_step` (the path with Bug #1). | Either integrate `train_one_fire`'s bind-inside-loss pattern into `make_batched_train_step`, or remove it. If kept as a reference implementation, add a comment marking it as unused and explaining why. |
| 9 | **STYLE** | [tests/test_covariate_randers.py](../tests/test_covariate_randers.py) `test_metric_fn_riemannian_matches_sqrt_Gv` | **Misleading docstring.** The docstring states *"F(x,v)^2 == v^T G v"*, but the assertion compares `F(x,v)` (not squared) against `sqrt(v^T G v)`. The code is correct; the docstring is not. | Change docstring to *"F(x,v) == sqrt(v^T G v) for Riemannian case (use_wind=False)"*. |

---

## Test Coverage Assessment

| Public Symbol | Tested? | Notes |
|---|---|---|
| `project_spd` | ✓ | `TestProjectSPD`: eigenvalue range, vmap, grad, already-SPD identity |
| `project_b_norm` | ✓ | `TestProjectBNorm`: norm bound, already-within unchanged, grad |
| `LocalTerrainCNN.__call__` | ✓ | `TestLocalTerrainCNN`: shape, dtype, finite output, grad through weights |
| `CovariateConditionedRanders.bind_scene` | Partial | Tested via `_bound_model`; `test_bind_scene_resets_field` verifies `metric_field → None`. No test for incorrect raster shapes or dtype coercion. |
| `CovariateConditionedRanders.bind_scene_rasters` | **MISSING** | No direct test. The `bind_scene_rasters` + `bind_weather` composition (the vmap-per-fire path) is entirely untested. |
| `CovariateConditionedRanders.bind_weather` | **MISSING** | Not tested independently. |
| `CovariateConditionedRanders.precompute_metric_field` | ✓ | `TestPrecomputeMetricField`: field shape, dtype, finite, grad to CNN, bind resets field |
| `CovariateConditionedRanders.metric_fn` | ✓ | Positivity, 1-homogeneity, Riemannian limit, JIT, vmap, grad w.r.t. v, grad w.r.t. x, near-zero v, fuel embedding grad |
| `CovariateConditionedRanders._get_params` | Partial | Exercised via `metric_fn`; no test for `metric_field is None` path (expected crash) |
| `CovariateConditionedRanders._bilinear_interp_field` | Partial | No boundary-pixel test; no test for out-of-bounds query point |
| `CovariateConditionedRanders._bilinear_interp` | Partial | Same gaps as `_bilinear_interp_field` |

### Critical gap: terrain-bound gradient path

No test exercises the `make_batched_train_step` scenario — i.e., calling `eqx.filter_value_and_grad` on a *terrain-bound* model — and checks that `elev_raster` / `slope_raster` / etc. remain unchanged after the update. Adding such a test would have caught Bug #1 immediately:

```python
def test_rasters_not_updated_by_optimizer():
    """Raster values must be identical before and after a gradient step."""
    import optax, equinox as eqx

    model = _make_model()
    scene = _make_scene()
    terrain_bound = model.bind_scene(**scene)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(terrain_bound, eqx.is_inexact_array))

    x = jnp.array([150.0, 150.0], dtype=jnp.float64)
    v = jnp.array([1.0, 0.5], dtype=jnp.float64)

    def loss(m):
        m2 = m.precompute_metric_field()
        return m2.metric_fn(x, v)

    grads = eqx.filter_grad(loss)(terrain_bound)
    updates, _ = optimizer.update(
        eqx.filter(grads, eqx.is_inexact_array), opt_state,
        eqx.filter(terrain_bound, eqx.is_inexact_array),
    )
    updated = eqx.apply_updates(terrain_bound, updates)

    # Rasters must be unchanged
    np.testing.assert_array_equal(updated.elev_raster, terrain_bound.elev_raster)
    np.testing.assert_array_equal(updated.slope_raster, terrain_bound.slope_raster)
```

### Higher-order gradient coverage

No test uses `jax.test_util.check_grads` or `jax.jacfwd` / `jax.jacrev` to verify second-order derivatives of `metric_fn`, which are required by the Euler-Lagrange `spray` computation. This is a gap if `CovariateConditionedRanders` is ever passed to the spray/Hessian routines in `FinslerMetric`.

---

## Positive Patterns

- **`project_spd` via trace/discriminant**: avoids `jnp.linalg.eigh`, keeping the projection vmap-safe and grad-safe. The analytic eigenvector angle via `jnp.arctan2` is numerically robust for 2×2 matrices.
- **`project_b_norm` analytic G⁻¹-norm**: computes $\|b\|_{G^{-1}}^2 = (b_1^2 g_{22} - 2b_1 b_2 g_{12} + b_2^2 g_{11})/\det G$ without matrix inversion; `det_G = jnp.maximum(det_G, 1e-8)` prevents division by zero.
- **`bind_scene` / `bind_scene_rasters` correctly reset `metric_field = None`**: stale CNN output is never silently reused after a scene change.
- **`_bilinear_interp_field` propagates spatial gradient**: gradients flow through `fx = px - x0` (and `fy = py - y0`) to `x_world`, correctly populating $\partial F/\partial x$ for the Euler-Lagrange equations.
- **`stop_gradient` on `pixel_spacing_m`**: the grid resolution is correctly treated as a non-trainable constant in both bilinear interp methods.
- **`eqx.field(static=True)` for scalar hyperparameters** (`eps_G`, `max_G`, `max_b_norm`, `use_wind`, `fuel_emb_dim`, `cnn_channels`): these are compile-time constants from JAX's perspective, preventing unnecessary re-tracing.
- **`fuel_code_raster` clipped before indexing**: `jnp.clip(self.fuel_code_raster, 0, 12)` guards `fuel_embedding[fuel_codes]` against out-of-bounds integer indices at no gradient cost.
- **`test_grad_flows_to_cnn_weights`** uses the correct bind-inside-loss pattern (unbound model, `bind_scene` inside `loss_fn`), ensuring the CNN weight gradient test is itself correct even though the main training loop has a bug.
- **`test_identity_metric_distance` fix**: normalising `t_obs = t_raw / t_raw.max()` correctly matches the `ArrivalTimeLoss` normalisation convention; the commit comment explains the reasoning clearly.
- **7×7 receptive-field claim** in `LocalTerrainCNN` docstring is correct: three stride-1 3×3 conv layers (same padding) produce a receptive field of $3 + 2 \times 2 = 7$ pixels in each spatial dimension.
- **`LocalTerrainCNN` head transpose** `raw_local.transpose(1, 2, 0)` correctly maps equinox's channel-first layout `(5, H, W)` to the `(H, W, 5)` format expected by `_bilinear_interp_field` and `_get_params`.
