# HAM UAV Energy Ledger — the Exact-Drift Gauge on Real Flight Data

The real-data instance of the HAM gauge/identifiability program (the data-only
companion to `experiments/arm`): **a multirotor's cost of motion is a
direction-asymmetric energy geometry, and its exact (gravity) part is invisible
in the flight paths and recoverable from the energy ledger by a convex
estimator** — with the autopilot's own wind estimate as independent ground truth
for the stretch channel.

It is provider-agnostic in the same spirit as marine/arm: everything downstream
of `ingest.py` operates on a single **segment-table schema** (`medium.REQUIRED_COLUMNS`),
so a synthetic fleet and a directory of real PX4 logs are the same object to the
estimator. See `spec/uav_energy_gauge_PLAN.md` for the full plan and
`spec/exact_drift_gauge_equivalence.md` for the theorem.

---

## The framing (what is actually novel)

Integrate a multirotor's electrical power over a path segment `Δx` flown in a
*measured* time `Δt`. The segment energy splits by **parity under segment
reversal** (`Δx → −Δx` at fixed duration, length, speed profile):

```
E_seg ≈ [ c₀·Δt + c₁·len + c₂·len²/Δt + s·|Δz| ]  +  [ k·Δz + b·Δxy ]
        └────────── even (nuisance) ───────────┘     └──── odd: β ────┘
```

- **`k = mg(1/η↑ + η↓)/2`** is the **climb-cost ledger** — the exact 1-form
  `β_grav = k·dz = d(k z)` (potential-energy exchange). This is the object the
  gauge theorem is about.
- **`b`** is the horizontal wind coupling; for quadratic drag `b = −2c₂·W`
  *exactly* (the `|W|²` remainder is even and lands in `c₀`), so the fitted drag
  coefficient converts the recovered form back to a wind vector `Ŵ = −b̂/2ĉ₂`.
- The **even bracket is a nuisance model** — hover, drag, speed profile, and the
  *symmetric* half `s·|Δz|` of the vertical cost. It is fitted, never
  interpreted; even and odd features are L²-orthogonal on any direction-balanced
  segment set, so the nuisance cannot leak into the ledger.

**Why this is the honest real-data setting.** Potential-energy exchange is
*parametrization-free*: `mgΔz` does not care how fast the segment was flown, so
the ledger estimator survives the two things that break naive cost models on
real data — non-1-homogeneity (hover burns power at `v=0`) and arbitrary speed
profiles. The even bracket absorbs them; the ledger signal never touches them.
The Randers/gauge language applies *exactly* to the odd part, which is a 1-form
on trajectories regardless of homogeneity.

## Physical grounding and its honest limits

| Quantity | Model | Basis |
|---|---|---|
| Segment energy | `∫V·I dt` (trapezoid on the aligned grid) | measured electrical energy, never throttle |
| Ledger `k` | odd coefficient of `Δz` | `mg(1/η↑ + η↓)/2`, per-log free (m, η unlogged) |
| Vertical overhead `s` | even coefficient of `|Δz|` | `mg(1/η↑ − η↓)/2`, climbing *and* descending cost extra |
| Wind coupling `b` | odd coefficient of `Δxy` | `−2c₂·W` for quadratic drag |
| Frame | PX4 NED → ENU (`dz>0` = climb, wind = E/N) | `vehicle_local_position` is z-**down** |

Stated up front (and enforced by the filters, not hidden):

1. **Not a Finsler metric at `v→0`.** Hover has cost at zero velocity; the model
   works at the *segment-energy* level with measured `Δt`, where 1-homogeneity is
   not assumed. Only the odd part is a genuine 1-form.
2. **Descent is not energy recovery.** `η↓ ∈ [0, small]`; multirotors don't
   regen. `k` is still well-defined (half the up/down specific-power gap).
3. **`m`, `η` are not logged** → `k` is a per-log free parameter. The science is
   in its *consistency* (same airframe ⇒ same `k`) and *predictive power*
   (held-out directed energy), never in matching a book value.
4. **Vortex-ring / acro / ground effect** are outside the model → filtered
   (cruise-band on 3-D speed, `|a|` cap, AGL floor, battery sanity), not explained.

## The pipeline

```
ingest.py     ULog → per-flight segment table: parse_ulog (lazy pyulog) →
              align_flight (NED→ENU, uniform grid) → segment_track (densified
              length, ∫V·I dt) → apply_filters (+ survivorship census).
              download_logs bulk driver; simulate_raw_flight offline generator.
medium.py     the cost model: even/odd feature maps, schema contract, the
              reversal involution, spatial RBF wind atoms.
synthetic.py  the U2 bridge: fleets from the §2 power model with exact k, W.
estimate.py   convex fits: per-log ridge LSQ, Schur-pooled hierarchical fit
              (per-log nuisance + shared drift), negative control, implied wind.
evaluate.py   held-out directed-energy R², k-consistency, wind cosine vs EKF,
              direction-balance / even-odd leakage, the two-bin pilot ledger.
validate.py   the U0–U6 ladder as one runnable gate.
run_stage_{a,b,c,d}_*.py   the four stage scripts (below).
```

## Results (reproduced by the scripts, on the synthetic bridge)

* **Stage A — pilot GO/NO-GO** (`run_stage_a_pilot`). On a 50-log corpus run
  through the *real* ingest pipeline: the up/down asymmetry has the right sign in
  **100%** of judged logs (bar ≥90%), the crude two-bin `k̂` has within-airframe
  CV **0.05** (bar <50%) and lands on the truth, and a survivorship census
  reports what the quality gates keep. **GO.**
* **Stage B — the ledger and the gauge** (`run_stage_b_ledger`). Per-log ridge
  LSQ recovers `k` at within-airframe CV **0.07**; held-out **directed-energy
  R² = 0.99**; pooled wind cosine **1.00**. The gauge is shown concretely: two
  fleets flown identically with different `k` have **bit-identical trajectories**
  but different energy ledgers, and a profile-likelihood over `k` is flat for the
  path channel and sharply peaked for the energy channel. Negative control (fake
  east-west potential) recovers **2%** of `k`. **Success bar met.**
* **Stage C — the wind channel vs the EKF** (`run_stage_c_wind`). The horizontal
  drift recovered from energy asymmetries matches the autopilot's own wind
  estimate — an instrument the fit never saw — at global cosine **0.98** (spatial
  field) and median **0.96** per flight above a wind-speed floor.
* **Stage D — few-shot transfer** (`run_stage_d_transfer`). Carrying a model
  A→B by reusing A's hover+drag shape and re-fitting only the vertical ledger
  (+ a scale) reaches held-out R² ≥ 0.9 at **K=4** calibration segments, where a
  from-scratch fit is still starved (K=8) — the concrete "exact part transfers as
  a coordinate" story for the holonomy pitch.

## The real-data result (`run_real.py`, 2026-07)

Run on **129 real PX4 flights of 33 physical multirotors** (public Flight Review
corpus, mission-mode curated, identifiability-gated): **climb costs more (k > 0)
in 98% of flights**, median k = 7.9 J/m, and **k is a per-vehicle constant** —
within-vehicle CV 0.35 vs cross-vehicle CV 2.09 (7 vehicles with ≥3 flights;
grouping by `sys_uuid`). Negative control at 10% of k. Five of these flights are
frozen as a CC-BY test fixture (`tests/fixtures/uav/real_segments.csv`), so the
suite checks the real-data claim offline. Full account and survivorship numbers:
`spec/uav_energy_gauge_OUTCOMES.md`.

## Validation

Runnable ladder `python -m experiments.uav.validate` (rungs **U0–U6**) and the
test suite `pytest tests/test_uav.py`: ingest integrity (NED→ENU round-trip,
`∫V·I dt` balance, densified length ≥ chord); segmentation gates + honest census;
estimator recovery of a *known* `k`/`W` (the mandatory synthetic bridge);
direction-balance / even-odd orthogonality; the Stage-A pilot gates; the Stage-B
success bar; and the negative control. All rungs green. The U0/U1 rungs drive the
real ingest pipeline with a synthetic PX4-convention raw generator, so the suite
is offline and deterministic (no pyulog, no network, no files).

## Running on real data

Everything above runs on synthetic flights so every number is checkable against
ground truth. The switch to real PX4 logs is a data-source change, not a rewrite:

```bash
pip install 'hamtools[uav]'          # pyulog, pyarrow, requests
python -c "from experiments.uav import download_logs; \
          download_logs('data/uav', max_logs=800, airframe_types=('Quadrotor',))"
```

Then, in a stage script, replace the synthetic fleet with the corpus loader:

```python
from experiments.uav import load_corpus
df = load_corpus("data/uav")   # a dir of *.ulg (parsed) or *.parquet (preprocessed)
```

`load_corpus` returns the same segment-table schema the synthetic generator
produces, so `fit_per_log`, `fit_pooled`, `directed_energy_r2`,
`wind_field_cosine`, and the transfer estimator are all unchanged. On real logs
the EKF wind columns come straight from the `wind` topic, so the Stage C
cross-check validates against a genuinely independent sensor.

**Data & licensing.** PX4 Flight Review logs are **CC-BY**; any real log or
preprocessed table committed under `tests/fixtures/uav/` must carry its `log_id`
and uploader attribution (see that folder's README).

## Reuse from the framework

- The convex `recover_potential_lsq` / `recover_form_lsq` pattern from
  `experiments/arm/learn.py`, with RBF atoms swapped for flight features
  (`medium.rbf_wind_features`, `estimate.fit_pooled(wind="spatial")`).
- The odd/even split is `segment_asymmetry` thinking applied to energy.
- The validation-ladder + stage-script + walkthrough-notebook house pattern
  (marine/arm), and the honesty patterns that earned their keep: densified
  geometry (the tunneling lesson), a-priori gates before fits, negative controls,
  and "figures must demonstrate intent unambiguously".
