# Code Review: `examples/experiment_wildfire_flat.py` — Training Loop Refactor
**Reviewer:** Code Reviewer Agent  
**Date:** 2026-05-18  
**Arch Spec Version:** 1.1.0 (April 2026)

## Summary

This review covers three interacting changes in `experiment_wildfire_flat.py`: (1) the
replacement of `jax.vmap`-based batching with a Python-level `for`-loop gradient
accumulation in `make_batched_train_step`; (2) the `AVBDSolver` configuration change to
`grad_clip=100.0` and `implicit_diff=True`; and (3) the IoU normalisation change in
`evaluate_fire`.

The Python-loop gradient accumulation is **mathematically correct** (equivalent to the
vmap mean gradient in exact arithmetic) but introduces **two bugs** — a mis-normalised
evaluation metric causing persistent IoU@50=0, and an implicit-diff/grad-clip combination
that violates the IFT fixed-point assumption and is the likely root cause of NaN loss at
epoch 9 — plus two performance risks.  The JIT compilation and lambda-capture questions
raised in the review request are resolved as non-issues (see Issues 5 and 6).

---

## Issue Tracker

| # | Severity | Location | Description | Suggested Fix |
|---|----------|----------|-------------|---------------|
| 1 | **BUG** | `examples/experiment_wildfire_flat.py:537–538` | **IoU@50 = 0 throughout training.**  `gt_ref = max(gt_arrival_finite)` where `gt_arrival` is sampled from `scenario.arrival_times`, which the data loader normalises to [0,1] at load time (confirmed by the inline comment at line 544 and the `iou_at_50` docstring, `src/ham/data/wildfire.py:180`).  Therefore `gt_ref ≤ 1.0`.  Dividing `pred_arrivals` (geodesic arc lengths in metres, O(100–10,000)) by a value ≤ 1.0 produces `pred_norm >> 1` for every pixel.  `iou_at_50` then evaluates `(pred_raster <= 0.5)`, which is `False` everywhere, giving intersection=0 and IoU=0 regardless of metric quality.  The comment at line 534 incorrectly states the intent: it says "normalise both pred and gt by the same reference (gt.max())" but the two quantities have incompatible units and cannot share a reference. | Normalise predictions by their own maximum: `pred_finite = pred_arrivals[np.isfinite(pred_arrivals)]; pred_ref = float(pred_finite.max()) if pred_finite.size > 0 and pred_finite.max() > 1e-8 else 1.0; pred_norm = pred_arrivals / pred_ref`.  This maps predictions to [0,1] on the same relative scale as the GT, making the 0.5 threshold comparable across both arrays. |
| 2 | **BUG** | `examples/experiment_wildfire_flat.py:302,305` | **`grad_clip=100.0` + `implicit_diff=True` violates the IFT fixed-point assumption.**  The IFT adjoint in `_implicit_forward_pass_bwd` (`src/ham/solvers/avbd.py:149–186`) is valid only when the solver has converged to a true fixed point `x*` satisfying `G(x*, θ) = 0` (the discrete Euler-Lagrange residual).  With `grad_clip=100.0` (10× the default of 10.0), each AVBD vertex update can move up to `step_size × grad_clip = 0.05 × 100 = 5 m/step` — 10× larger than before.  With the same `avbd_iters=50`, the iterative path optimisation is less likely to converge within the iteration budget.  At a non-converged path, `dG_dx = jax.jacobian(...)` (`avbd.py:154`) is not the correct Hessian of the fixed-point residual; the `jnp.linalg.lstsq` solve (`avbd.py:159`) can return a large `lam`, magnifying the upstream gradient by O(cond(dG_dx)²).  This is the most likely **root cause of NaN gradients reaching metric parameters by epoch 9**: the IFT produces unstable gradient estimates → Adam accumulates large second moments → parameters diverge → `metric.metric_fn` returns NaN → loss is NaN (compounded by Issue #1 in `losses_arrival_time_v2.md`). | **Option A (recommended):** Revert `grad_clip` to the default 10.0. The original step budget was validated against this value.  **Option B:** If the larger step size is needed to navigate high-curvature regions, increase `avbd_iters` to ~200 to restore convergence quality, or add a post-solve residual check: if `jnp.max(jnp.abs(_el_residual(x_inner, metric, p_start, p_end))) > tol`, fall back to unrolled-backprop gradients. |
| 3 | **RISK** | `examples/experiment_wildfire_flat.py:267` | `total_loss_val += float(loss_i)` inside the per-fire loop performs a **synchronous device-to-host transfer** on every iteration.  `float()` on a JAX array calls `.block_until_ready()` implicitly, serialising XLA dispatch: fire `i+1`'s JIT kernel cannot be dispatched until fire `i`'s result has been copied to the CPU.  For B=16 fires this eliminates all GPU pipeline overlap and makes the loop approximately B× slower than necessary.  The loss value is used only for logging, so gradient correctness is unaffected. | Accumulate as a JAX array: `total_loss_val = total_loss_val + loss_i` (where `total_loss_val` is initialised as `jnp.zeros(())`), and call `float(total_loss_val / B)` once after the loop. |
| 4 | **RISK** | `examples/experiment_wildfire_flat.py:244–250` | `_single_fire_grad` is `@eqx.filter_jit`.  The `metric` pytree passed into it on each iteration has the **same structure** (terrain-bound weights; `eqx.apply_updates` is structure-preserving) but different values.  Equinox reuses the compiled XLA computation for the same structure — so there is **one JIT compilation**, not B.  However, if the `metric` structure ever changes between steps (e.g. because `bind_weather` modifies the pytree and a refactor changes how it's bound), this will silently cause B retraces.  The current code passes weather inside `_single_fire_grad` (via `bind_weather`), not as a structural change to the metric, so this is safe *today*. | No immediate action required.  Add an assertion or a shape check: `assert jax.tree_util.tree_structure(metric) == _expected_structure` at the top of `batched_step` to catch future regressions. |
| 5 | **RISK** | `examples/experiment_wildfire_flat.py:271–273` | The `lambda a, b: a + b` inside the accumulation `tree_map` is a pure function — `b` is a parameter, not captured from the loop scope — so there is **no lambda-capture bug**.  However, `accumulated_grads` grows from the raw gradient of fire 0 (not averaged) to the sum of all B gradients before the `/ B` division at line 276–278.  This is correct arithmetic (sum then divide), but any inspection of `accumulated_grads` inside the loop would show un-normalised values, which could mislead debugging.  The first iteration sets `accumulated_grads = grads_0` (not `grads_0 / B`), which is intentional and correct. | No correctness fix needed.  Consider renaming `accumulated_grads` to `grad_sum` inside the loop for clarity. |
| 6 | **STYLE** | `examples/experiment_wildfire_flat.py:251–289` | `batched_step` is not `@eqx.filter_jit`.  The Python loop, `tree_map` calls, and `optimizer.update` all run at the Python/NumPy level, with per-iteration XLA dispatch.  The original vmap-based batching compiled the entire B-fire gradient into a single XLA program, which is both faster and friendlier to XLA's memory allocator.  The refactor trades correctness-for-memory but incurs a significant runtime cost. | If memory is the actual constraint, consider `jax.checkpoint` on `single_arrival_time` inside the loss or reducing `solver_steps`.  If the loop is retained, wrap `optimizer.update` and the `tree_map` division in a `jax.jit`-compiled helper. |
| 7 | **STYLE** | `examples/experiment_wildfire_flat.py:302` | `grad_clip=100.0` comment: *"Large trust-region (max move 5m/step) for bending mobility."*  The original default gives 0.5 m/step.  The rationale for a 10× increase is not documented in the commit message or spec.  Without justification, this looks like an ad-hoc workaround for solver non-convergence that was better addressed differently. | Add a spec reference or remove the increase.  See Issue #2. |

---

## Answers to Specific Review Questions

**Q1. Is the Python loop equivalent to vmap-based batching?**  
Yes, mathematically.  Both compute $\hat{g} = \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta L_i$ where $L_i$ is the loss for fire $i$.  The loop version evaluates each $\nabla_\theta L_i$ sequentially with independent JIT dispatches; vmap evaluates them in one vectorised XLA kernel.  The gradient identity is exact (not an approximation).

**Q2. Does `float(loss_i)` detach the loss from the computation graph?**  
The loss value has **already been detached** from the graph: `eqx.filter_value_and_grad` returns `(loss_scalar, grads_pytree)`, where `loss_scalar` is a JAX scalar that carries no gradient tape itself (gradients are in `grads_pytree`).  `float(loss_i)` is therefore safe for correctness but harmful for performance (see Issue #3).

**Q3. Is there a scale inconsistency in gradient accumulation on the first iteration?**  
No.  The accumulation is `sum = grads_0 + grads_1 + … + grads_{B-1}`, then `mean = sum / B`.  Setting `accumulated_grads = grads_0` on the first iteration is the correct initialisation of a running sum, not an incorrect averaging.  The final division by `B` is at line 276–278.

**Q4. Does `implicit_diff=True` interact poorly with the Python loop?**  
The IFT is **per-fire and stateless**: each call to `_single_fire_grad` computes one forward IFT solve and one backward IFT adjoint independently.  The Python loop does not share IFT state between fires.  The interaction problem is not *loop-specific* but rather the `grad_clip=100.0` undermining the fixed-point assumption the IFT requires (Issue #2 above).

**Q5. JAX/Equinox-specific pitfalls in the new pattern?**  
Issue #3 (device sync) and the potential re-trace in Issue #4.  Additionally: if `batched_step` is ever wrapped in an outer JIT (e.g. for a multi-scene parallel loop), the Python-level `for` loop would unroll B iterations into the outer JIT's trace, creating a B× larger XLA program.  The current code avoids this only because `batched_step` is not itself JIT-compiled.

**Q6. Lambda capture inside loop?**  
Not a bug.  `b` in `lambda a, b: a + b` is a function parameter, not a free variable captured from the enclosing scope.  `tree_map(lambda a, b: a + b, pytree1, pytree2)` passes the corresponding leaves as `a` and `b`.  This is idiomatic and correct.

**Q7. Does B=16 JIT calls cause B compilations?**  
No.  `eqx.filter_jit` traces on pytree *structure*, not values.  Since `metric` has the same structure on every call (verified: `eqx.apply_updates` is structure-preserving; `bind_weather` is called *inside* `fire_loss`, not on the argument to `_single_fire_grad`), only **one compilation** is produced.  All 16 calls dispatch to the same compiled XLA computation.

---

## Test Coverage Assessment

| Function | Tested? | Notes |
|---|---|---|
| `make_batched_train_step` (entire) | **No** | No unit test for the accumulation loop, optimizer update, or gradient equivalence to vmap |
| `make_batched_train_step` gradient equivalence | **No** | No test comparing loop gradients to vmap gradients on a known metric |
| `make_solver` (`implicit_diff=True`) | **No** | `test_implicit_diff_avbd.py` exists but does not test `grad_clip=100.0` or combined use with `ArrivalTimeLoss` |
| `evaluate_fire` (IoU path) | **No** | No test verifying `pred_norm` is in [0,1] before calling `iou_at_50` |
| `evaluate_fire` (Pearson r) | **No** | No test verifying Pearson r = 1.0 for a perfect metric |

---

## Positive Patterns

- **Gradient accumulation arithmetic** is correct: sequential accumulation followed by
  division is mathematically equivalent to vmap-averaged gradients.
- **Weather binding inside the JIT-compiled function** (`_single_fire_grad` at line 247):
  `bound = m.bind_weather(weather)` is called inside `fire_loss(m)`, which is differentiated
  inside the JIT.  This correctly propagates gradients through the bound weather into the
  shared MLP weights.  Binding outside the JIT would have been a correctness error.
- **Terrain binding once per scene** (lines 757–762 in `train_scene`): Binding the
  terrain rasters once and only varying `weather_vec` per fire correctly amortises the
  raster preprocessing cost.
