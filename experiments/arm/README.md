# HAM Robot-Arm Geodesics — Asymmetric C-Space Cost and the Exact-Drift Gauge

A physically-grounded experiment built on the HAM framework (the AVBD twin of
`experiments/marine`): **energy-optimal motion planning as geodesics of a
layered Randers/Finsler metric on configuration space**, with gravity as a
joint-space "wind", obstacles folded into the metric, exact task constraints
via the Augmented Lagrangian, and — the capstone — a *theorem-grade* treatment
of what demonstrations can and cannot teach about the cost.

It is also provider-agnostic: everything depends on four protocols
(`Robot / Scene / DistanceField / DemoSource` in `interfaces.py`), so the
synthetic planar arm swaps for a URDF robot without touching the science.

---

## The framing (what is actually novel)

A gravity-loaded arm's cost of motion is **directional**: lifting the links
costs more than lowering them. No symmetric Riemannian metric can express
this; a **Randers metric** `F(q, q̇) = √(q̇ᵀM(q)q̇) + β(q)·q̇` can. The layers:

1. **Kinetic base** — the mechanical metric `M(q)` (mass matrix, autodiff of
   the kinematics).
2. **Gravity drift** — the Zermelo wind `W = −gₛ·M⁻¹(q)∇U(q)`: the velocity
   drift the gravity *force* induces under the mobility `M⁻¹`. The raising
   matters: it makes the induced Randers 1-form `β = −(MW)♭/λ = gₛ·dU/λ`
   **exact**, which is the geometric heart of Stage D.
3. **Obstacle conformal warp** — `ρ(q)F` with `ρ = 1 + α/(dist − δ)₊` over an
   injected clearance field (Region-Avoiding Metrics); realized as
   `H → ρ²H, W → W/ρ`, which leaves the mild-wind bound invariant.

**The capstone result (Stage D).** Recovering the drift from demonstrated
*paths* is not merely hard — for the gravity drift it is **provably impossible**:
adding an exact 1-form to a cost changes the value of every path by a boundary
term and the choice of path not at all. This one theorem is Finsler
*projective invariance*, *potential-based reward shaping* (Ng et al. 1999;
Skalse et al. 2023), and the exact component of a Hodge split, verbatim
(`spec/exact_drift_gauge_equivalence.md`). The gauge is broken by the
complementary observation channel — **cost/timing** (Bucataru–Muzsnay
parametrization-rigidity) — and the cleanest such observation is the
experiment's own headline: the **directed round-trip asymmetry**
`cost(γ) − cost(γ⁻¹) = 2∫β = 2gₛ(U(B) − U(A))`, which turns drift recovery
into a *convex least-squares* for the potential. The negative result becomes
the estimator.

## Physical grounding

| Quantity | Model | Mechanical basis |
|---|---|---|
| Base metric `M(q)` | `Σᵢ mᵢ JᵢᵀJᵢ` from tip Jacobians | kinetic energy of point masses at link tips |
| Potential `U(q)` | `g Σᵢ mᵢ yᵢ` | gravitational height of each mass |
| Drift `W` | `−gₛ·M⁻¹∇U`, causal soft clamp `‖W‖_M < 1` | force→velocity via mobility; Randers well-posedness |
| Obstacles | conformal `ρ²H` over C-space clearance | Region-Avoiding Metrics (IROS 2023) |
| Task constraint | `c(q) = Σθᵢ − φ★ = 0` via AVBD's ALM | carry a cup level through the motion |
| Configuration space | Clifford `FlatTorus(n)` or intrinsic angles | revolute joints; seam-safe vs eikonal-grid-able |

## Results (reproduced by the scripts)

* **Stage A — forward planning with demonstrable intent.** The task is chosen
  so the obstacle *constrains the optimum*: the straight swing drives the arm
  through the disk (clearance −0.29 mid-motion), and the avoided geodesic's
  tightest clearance (+0.23) occurs mid-swing, hugging the barrier margin —
  the signature of a cost trade-off. The route is validated globally: the
  eikonal characteristic on the *same barrier metric* (no initialization, no
  basins) takes the same corridor at matching cost (AVBD 7.30 vs eikonal 7.39,
  grid-resolution agreement). Cross-solver triangle on the smooth metric ~2%;
  the identical code plans for a 5-DoF arm (−0.10 → +0.41).
* **Stage B — asymmetric cost (the headline).** The gravity-Randers geodesic
  costs A→B vs B→A differ by ~21% (choosing the cheap *direction* is the
  actionable payoff), while the gravity-aware and gravity-blind *plans coincide
  as point sets* (deviation ~0.01 rad, true-cost gap 0.2%) — the drift is big
  in the ledger, invisible on the map. That is projective invariance previewing
  Stage D, not a planner defect.
* **Stage C — exact constraints.** The ALM holds the cup orientation to ~1e-3
  where a well-tuned pure penalty plateaus (the classic `1/μ`
  ill-conditioning) and the unconstrained geodesic tilts freely.
* **Stage D — the exact-drift gauge.**
  - *Scaling law:* point-set bending of geodesics grows as **strength²** for
    the gravity drift (`dev/gₛ² ≈ 1.6–2.0`, the second-order Zermelo leak) but
    as **strength¹** for a rotational vortex drift — the theorem made visible.
  - *A-priori diagnostic:* a discrete projective-metrizability score separates
    the regimes **before any fit** (pure gravity ≈ 0.007 vs +vortex ≈ 0.047).
  - *Channel experiment:* with the same scalar-potential hypothesis, the
    geometry loss L-B recovers essentially nothing of the potential at any
    demo count (R² erratic, ≤ 0.35), while the segment-timing loss L-T sits at
    **R² ≈ 1.000**.
  - *Convex recovery:* round-trip cost asymmetries give a linear LSQ that
    recovers `gₛ·U` at **R² = 0.998** on pure gravity, and the **entire drift
    1-form at cosine 0.996** on gravity+vortex (its Hodge *split* from a finite
    demo graph is soft — U R² 0.96, vortex cosine ~0.5 — and reported as such).
  - *De-circularization:* the eikonal arrival field of the timing-recovered
    metric reproduces a held-out demo's cost to ~2% (a solver independent of
    the AVBD demos).
* **Higher dimensions (notebook §7).** The convex estimators are
  dimension-blind: the 5-DoF arm's gravity potential is recovered from timed
  round trips at **R² ≈ 0.93** using data-adaptive RBF centers (subsampled demo
  vertices) — no grid anywhere. This is the operating regime of a *learned
  latent metric*: trajectories and costs are the only observables, and they
  are exactly what the estimators consume. Grid-based instrumentation (eikonal
  cross-checks, the a-priori diagnostic) is honestly low-DoF-only.

## Layout

```
interfaces.py   the provider seam: Robot / Scene / DistanceField / DemoSource
providers/      synthetic.py (analytic planar arm, circle scenes, demos), real.py (URDF stub)
medium.py       ArmMetric — kinetic + gravity-Randers + conformal barrier; zermelo_cost
fields.py       learnable fields: MLPDistance, BoundedWind, and the Hodge-structured
                PotentialWind / StreamWind / HodgeWind (+ VortexWind ground truth)
constraints.py  upright-cup / waypoint equality constraints + AVBD bridge
planners.py     AVBDPlanner (barrier continuation) ; EikonalPlanner (2-DoF cross-check)
learn.py        loss menu by observation channel (L-A/L-B/L-C/L-D shape; L-T timing)
                + convex estimators recover_potential_lsq / recover_form_lsq
evaluate.py     path metrics, spray oracle, polyline_deviation, shape_identifiability,
                Hodge-split and form-cosine recovery metrics
validate.py     the runnable validation ladder L0–L7e (CI-style gate)
run_stage_a_forward.py / run_stage_b_asymmetric.py / run_stage_c_constraint.py /
run_stage_d_inverse.py
arm_geodesics.ipynb   publication-grade walkthrough (interactive 2-D/3-D + animation
                      players, plus the 5-DoF latent-regime recovery); build_notebook.py
```

## Running

```bash
JAX_PLATFORMS=cpu python -m experiments.arm.validate               # ladder L0–L7e
JAX_PLATFORMS=cpu python -m experiments.arm.run_stage_a_forward
JAX_PLATFORMS=cpu python -m experiments.arm.run_stage_b_asymmetric
JAX_PLATFORMS=cpu python -m experiments.arm.run_stage_c_constraint
JAX_PLATFORMS=cpu python -m experiments.arm.run_stage_d_inverse
JAX_PLATFORMS=cpu python -m experiments.arm.build_notebook --run   # build the walkthrough
JAX_PLATFORMS=cpu pytest tests/test_arm.py                         # test suite
```
Figures are written to `experiments/arm/visualizations/`.

## Caveats (honest)

* **First-order gauge, second-order leak.** Projective invisibility of the
  exact drift is exact in the additive Randers form; the Zermelo
  representation leaks at `O(gₛ²)` (measured: `dev/gₛ² ≈ 1.7`). At `gₛ = 0.15`
  that residual is real but an order of magnitude below the rotational signal —
  and it is precisely what makes the geometry-channel fits *erratic* rather
  than exactly zero.
* **Vertex spacing is timing in disguise.** Solver-generated demos are
  constant-F-speed sampled, so "path-only" losses that consume raw vertices
  quietly smuggle parametrization information (this is Bucataru–Muzsnay
  rigidity, not a bug). The shape/timing split is made explicit: the
  diagnostic and the scaling law quotient parametrization out
  (`polyline_deviation`), the timing losses use it deliberately.
* **Hodge split softness.** Cost asymmetries observe `∫β` along the demo
  graph, which pins the *form* but determines its exact/co-exact split only up
  to near-harmonic leakage on a bounded region with finite coverage.
* **Solver-in-the-loop shape inversion is fragile.** Gradient descent through
  the geodesic BVP (L-A) on these weak, long-wavelength signals is
  seed-sensitive and can basin-hop — the historical "identifiability frontier"
  observation, now explained by this gauge structure rather than reported as a
  mystery. The convex estimators replace it wherever the theory allows.
* **Barrier scale matters — and saturates.** The conformal barrier is a
  *localized* wall only for `α ~ 0.2·(typical clearance)`; beyond `α ≈ 1` it
  defeats itself: the softplus wall's hardness is fixed by its width, so larger
  `α` only inflates free space until cutting through the wall beats detouring.
  Stage scripts use `α = 0.15–0.6`.
* **Tunneling is real, and vertex clearance lies about it.** AVBD's midpoint-
  rule energy lets a long segment straddle a thin obstacle body with no barrier
  gradient at all; the barrier even *repels vertices* into that configuration
  (found at 3 DoF: a smooth path ending in a 3-rad "teleport" through the
  obstacle, reported clear by vertex clearance). Defenses, now standard here:
  `min_clearance` evaluates a densified path; segment length must stay well
  below obstacle body thickness (48 segments at 3 DoF); and
  `AVBDPlanner(resample_between=True)` re-equalizes spacing between
  continuation stages.
* **Fixed-step stiffness limit.** The 5-link mass matrix is stiff enough that
  AVBD's step 0.05 diverges (energy grows with iterations); 0.01–0.02
  converges. The same limit is why the constrained (ALM) solve needs step 0.02.
* **Scope.** Energy/cost geometry only: torque and velocity limits are
  inequality constraints, outside AVBD's exact-equality ALM; timing data here
  is synthetic per-segment cost (a real robot would log it from execution).

## Extending the frame

* **Real robot:** implement `providers/real.py` against a URDF (inertia via a
  rigid-body library, clearance from a mesh SDF); everything downstream is
  unchanged.
* **Torus topology:** on the Clifford `FlatTorus` the harmonic 1-forms
  (`H¹ ≠ 0`) are closed but not exact — loop demos around the torus would
  expose the topological corner of the gauge story (§5.1 of the note).
* **More constraints:** any equality `c(q) = 0` plugs into the ALM; inequality
  physics belongs in a penalty/continuation layer as in the marine experiment.
