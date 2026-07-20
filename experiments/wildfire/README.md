# Wildfire cross-scene generalization — identifiable, transferable Randers fire spread

Companion experiment to Gahtan, Shpund & Bronstein (arXiv:2603.00035), whose
differentiable Randers eikonal solver is architecturally the same as
`ham.solvers.EikonalSolver`. Their cross-scene generalization collapses
(IoU@50 0.609 → 0.295 while correlation barely moves); this package decomposes
that failure with the HAM identifiability dictionary and repairs it:

1. **The IoU collapse is dominated by a low-dimensional scene gauge** — an
   absolute speed scale `s` (and wind coupling `c`) that correlation metrics
   discard by construction and IoU@50 bills in full. A **2-parameter convex
   recalibration** from the first hours of a new fire closes most of the gap
   (`recover.recalibrate`: `s` is closed-form given `c`, `c` is a 1-D search).
2. **Wind is odd, terrain is even, and a fire burns each pixel once** — one
   slowness equation `1/speed(n) = sqrt(n^T G n) + B.n` against five unknowns.
   The odd channel (wind) is only identified by *signed* front diversity:
   an antiparallel front pair separates `B` exactly, single-signed orthogonal
   fronts never do (`medium.odd_coverage`, the pre-fit identifiability
   ledger; the sign-blind structure tensor `direction_coverage` measures the
   even channel and empirically does NOT predict wind error).
3. **Measured wind transfers trivially because it is an input** —
   `wind_mode="coupled"` in `CovariateConditionedRanders` sets
   `b = c * measured_wind` with one learned scalar, so the encoder only has
   to generalize the even channel.

Theory guard for all writing: arrival times are the *ledger* channel (T is a
value function — exact drift parts ARE visible in it, unlike path shapes);
the binding constraint here is parity/direction coverage, not projective
invariance. Keep the two failure modes distinct.

## Layout

| file | contents |
|---|---|
| `medium.py` | conventions + `GridZermelo`, arrival solves, paper metrics (absolute IoU@50), dense two-gauge loss, TV, coverage masks |
| `synthetic.py` | known-truth scenes (smooth sea + wind), fire simulation |
| `recover.py` | free-field `(H, W)` recovery through the solver; `recalibrate` (s, c) |
| `ingest.py` | Sim2Real-Fire loading: hours kept absolute, pixel solver frame, raw wind, FBFM13 non-burnable remap, mask/raster resolution reconciliation |
| `train.py` | covariate-encoder training (per-scene JIT, vmapped fire batches) |
| `evaluate.py` | per-fire zero-shot + recalibrated evaluation |
| `run_stage_a_bridge.py` | **W-A** gates W0–W4 (forward, gradients, multi-source recovery, single-source confounding, recalibration exactness) |
| `run_stage_b_mask.py` | **W-B** odd-coverage predictivity + parity-aware shrinkage |
| `run_stage_c_crossscene.py` | **W-C** LOSO baseline: reproduce the collapse |
| `run_stage_d_headline.py` | **W-D** recalibration curves + coupled-wind ablation (the email figure) |

Unit gates: `tests/test_wildfire.py` (fast); model-level tests in
`tests/test_covariate_randers.py` (coupled mode).

## Conventions (bound once — see `medium.py` docstring)

- Solver frame is **pixel coordinates**, arrival times in **hours**; the
  encoder is bound with `pixel_spacing_m=1.0` (G's eigenvalue box [0.1, 10]
  then spans slowness 0.3–3.2 h/px, the physical range; in metres it cannot
  represent realistic speeds).
- Wind is a **velocity, downwind cheaper** at every API surface. Inside
  `CovariateConditionedRanders` the drift one-form `b` points downwind
  (`b·v > 0` cheaper — Zermelo form); inside `GridZermelo` the primal `B`
  points upwind. Do not mix the two.
- IoU@50 thresholds both fields at 0.5 × the **ground-truth** duration in
  hours; predictions are never normalized by their own scale.
- Weather 4-vector `[T_air, humidity, wind_x, wind_y]` is standardized for
  the encoder; the **raw** wind velocity rides separately (`measured_wind`)
  because z-scoring destroys physical direction.

## Data

Stages C and D need the **Sim2Real-Fire** dataset (Sun et al.), downloaded
separately and unpacked under `data/sim2real_fire/` (the `data/` tree is
gitignored). Stages A and B are fully synthetic and need no download.

`data/sim2real_fire/` scenes 0001–0005, 0012, 0013, 0014_00426 (zips
0006–0011 and 0026–0031 are corrupt/truncated). Scenes differ in raster size
(100–430 px native, auto-downsampled to ≤148) and some have mask/raster
resolution mismatches (resampled at ingest). FBFM13 rasters carry
non-burnable codes 91–99 → fuel-embedding row 13.

## Running

```bash
python -m experiments.wildfire.run_stage_a_bridge      # ~5 min, synthetic
python -m experiments.wildfire.run_stage_b_mask        # ~10 min, synthetic
python -m experiments.wildfire.run_stage_c_crossscene  # ~25 min, 8 LOSO folds
python -m experiments.wildfire.run_stage_d_headline    # ~45 min, needs stage C
```

All stage scripts are resumable (JSON rows / checkpoints per fold) and write
figures + gate ledgers to `visualizations/`.
