# HAM Marine Navigation — Time-Dependent Zermelo Routing for a 3D Underwater Glider

A physically-grounded experiment built on the HAM framework: globally-aware,
**time-optimal path planning through a time-varying, depth-stratified ocean
current**, for a buoyancy-driven autonomous underwater glider.

It is also a **dimension-agnostic frame** (Medium / Vehicle / Constraints /
Planners / Evaluate). The 3D glider is instantiated here; a 2D surface vessel is a
drop-in next-session target (see *Extending the frame*).

---

## The framing (what is actually novel)

Time-optimal navigation through a current `W` over a Riemannian "sea" `H` is
**Zermelo's problem**, and its solutions are exactly the geodesics of a **Randers
metric** `F(x,v) = √(vᵀH v) + β·v`, valid in the *mild-wind* regime `‖W‖_H < 1`
([Bao–Robles–Shen; Caponio et al.](https://arxiv.org/abs/2304.00478)). HAM's
`Randers` metric + differentiable fast-sweeping eikonal solver (arXiv:2603.00035)
implement this directly. Two corrections to the naïve "inverse-Zermelo" pitch:

1. **Drifters give the current directly.** Passive drifter buoys float *with* the
   flow — their tracks are integral curves of `W`, not time-optimal geodesics —
   so reconstructing the current is **direct velocity regression** (Stage B), not
   inverse optimal control. (Drifter flow inference is famously underdetermined.)
2. **The novelty is time-dependence.** The eikonal `F(x,∇T)=1` is *stationary* and
   breaks when the current evolves. Planning time-optimal routes through an
   *evolving* medium is the open problem (active 2024–2026: RRT*, NSGA-II, deep-RL
   that "struggles with stability"; level-set 3D solvers are heavy). HAM's
   differentiable Randers machinery gives a clean, gradient-based alternative.

## Physical grounding (conceptually grounded, not capability-for-its-own-sake)

| Quantity | Model | Oceanographic basis |
|---|---|---|
| Sea `H(x)` | `I / (s_max·speed_factor(x))²` | quadratic drag vs power budget; a cold/dense "lens" lowers achievable speed |
| Current `W` (geostrophic) | `∇^⊥ψ`, divergence-free | geostrophy: current = curl of sea-surface-height stream function `ψ` |
| Vertical structure | two-layer (thermocline): adverse surface over reversed deep layer | baroclinic mode; the depth-riding opportunity |
| Current `W` (Ekman) | curl-free, divergent, surface-trapped | wind-driven ageostrophic drift; the stream-function blind spot |
| Time variation | meander phase advects; deep window opens/closes (period `τ`) | mesoscale evolution; breaks the stationary eikonal |
| Glider | `s_max ≈ 0.85` (basin units), no propeller, near `‖W‖_H → 1` | buoyancy-driven AUV, ~0.25–0.5 m/s, advances by choosing depth |

The slow glider operates **near the Zermelo mild-wind boundary** — a *featured*
regime: strong eddy cores are genuinely non-navigable (`λ = 1 − ‖W‖²_H < 0`) and
must be ridden or avoided. The `tanh` squash in `Randers` caps `‖W_safe‖ < 1` so
planning stays well-posed there.

## Results (reproduced by the scripts)

* **Stage A — forward planning.** Volumetric-eikonal arrival field + AVBD route,
  cross-validated against the time-lifted solver and an analytic shooting solution.
  Depth-riding (diving to the reversed deep layer) saves **~10%** vs a
  surface-locked plan.
* **Stage B — reconstruction.** On-track cosine **~0.94** for both the geostrophic
  stream-function fit and the kernel baseline; off-track drops to **~0.55–0.62**
  (the honest identifiability frontier). The divergence-free prior is data-
  efficient at low coverage; the flexible kernel overtakes with dense data,
  because the true field's divergent Ekman part is *structurally* invisible to any
  stream function.
* **Stage C — time-dependent (the novelty).** Executing both plans under the true
  evolving current, the **time-aware route is ~7% faster** than the frozen-field
  route: it delays its dive until the deep favorable window opens mid-transit. This
  stage assumes a *perfect forecast* (it knows the whole future of the current).
* **Stage D — closed-loop replanning (MPC).** The honest operating mode: the glider
  only ever sees a forecast that is exact now and decays with lead time, and it
  re-plans the remainder each time it surfaces (receding horizon). Holding the
  controller fixed and varying only the forecast, a realistic **decaying forecast
  reaches the destination ~7% faster than a no-skill persistence forecast** — that
  margin is the value of forecast skill, and it is stable across runs. Closed-loop
  does not reach the perfect-foreknowledge ideal (a commitment overhead): the honest
  price of not knowing the future, and the reason a real glider both forecasts *and*
  replans.

## Layout

```
medium.py       OceanMedium (geostrophic ψ + baroclinic + Ekman + time), randers_cost, FrozenMedium
vehicle.py      Glider — sea tensor H(x) and operating constraints
constraints.py  unified Constraint layer (eq/ineq, segment, time-aware) + AVBD bridge
planners.py     StationaryPlanner (eikonal/volumetric + AVBD) ; TimeLiftedPlanner (clock-threaded)
forecast.py     belief models — Perfect / Persistence / Decaying forecasts
mpc.py          receding-horizon (MPC) controller: replan from current state, warm-started
drifters.py     passive-drifter simulation + stream-function / kernel reconstruction
evaluate.py     executed-plan timing, recovery metrics, shooting validation, navigability
run_stage_a_forward.py / run_stage_b_reconstruct.py / run_stage_c_timedependent.py / run_stage_d_mpc.py
marine_navigation.ipynb  publication-grade walkthrough (interactive 3-D + time players); build_notebook.py
```

### The unified constraint layer (and the AVBD question)

`AVBDSolver` enforces **equality** `c(x)=0` via its Augmented Lagrangian — used for
waypoints / fixed-depth legs via `avbd_equality_constraints`. But the grounded
glider physics are **inequality / per-segment / time-aware** (depth envelope,
glide-angle kinematics, seafloor clearance, moving no-go zones), and the
time-dependent clock is not a local metric, so AVBD cannot run the novelty stage.
Every `Constraint` is therefore also enforced as a differentiable penalty
(`constraint_penalty`, with a continuation schedule) inside `TimeLiftedPlanner`.
**Adding new physics = construct one `Constraint`; both planners pick it up.**

## Running

```bash
JAX_PLATFORMS=cpu python -m experiments.marine.run_stage_a_forward
JAX_PLATFORMS=cpu python -m experiments.marine.run_stage_b_reconstruct
JAX_PLATFORMS=cpu python -m experiments.marine.run_stage_c_timedependent
JAX_PLATFORMS=cpu python -m experiments.marine.run_stage_d_mpc
JAX_PLATFORMS=cpu python -m experiments.marine.build_notebook --run   # build the walkthrough
JAX_PLATFORMS=cpu pytest tests/test_marine.py        # validation suite (8 tests)
```
Figures are written to `experiments/marine/visualizations/`.

## Caveats (honest)

* **Mild-wind cap.** `tanh` squash distorts truly non-navigable (super-vehicle)
  currents; the medium is kept navigable on the corridor and the regime is mapped.
* **Local minima.** The time-lifted BVP is non-convex; we warm-start from the
  stationary route and use multi-start. Reported numbers are local optima. For the
  same reason the open-loop *persistence* plan is a fragile baseline (it sometimes
  discovers a good dive on its own), so Stage D's headline is the stable, apples-to-
  apples **closed-loop** comparison (decaying vs persistence forecast), not an
  open-loop "worst case".
* **Forecast optimism.** Stage C's perfect-foreknowledge gain is an *upper bound*;
  Stage D shows closed-loop replanning realizes part of it, and only because the
  forecast has short-horizon skill — replanning alone, without skill, buys nothing.
* **Identifiability.** Sparse drifters underdetermine `W` off-track; the
  stream-function prior helps but cannot recover the divergent Ekman drift.
* **Geometry scope.** The time-lifted path's effective metric is time-parameterized
  along the route, so static Randers transport/curvature apply per-snapshot only.
* **Vertical current neglected** (`W_z ≈ 0`, physically justified); full glider
  sawtooth/glide kinematics are a `Constraint`/`Vehicle` extension.

## Extending the frame

* **2D surface vessel (next session):** reuse Planners/Constraints/Evaluate; swap
  `EuclideanSpace(2)` + `EikonalSolver`, add an **anisotropic speed-polar `H`**
  (wave added resistance — a directional Finsler indicatrix) and Ekman-dominated
  surface currents.
* **More physics as `Constraint`s:** bathymetry obstacles, density-from-(T,S) drag,
  glide-angle sawtooth, moving exclusion zones (`time_varying_nogo` is provided).
* **Depth-resolved reconstruction:** profiling floats → a depth-dependent `W`.
