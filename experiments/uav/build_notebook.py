"""Assemble (and optionally execute) the UAV energy-ledger walkthrough notebook.

Usage:
    python -m experiments.uav.build_notebook          # write unexecuted
    python -m experiments.uav.build_notebook --run    # write + execute

Produces ``experiments/uav/uav_energy.ipynb``. The notebook reuses the experiment
package (no duplicated physics) and develops one idea end to end: a multirotor's
segment energy splits by parity into an even nuisance and an odd drift 1-form; the
gravity ledger is the exact part — invisible in the flight paths, recovered from
the energy by a convex solve, and cross-checked against the onboard EKF wind.
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "uav_energy.ipynb"

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def co(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ===========================================================================
md(r"""
# The Exact-Drift Gauge on Real Flight Data — a UAV Energy Ledger

This notebook is a walkthrough of `experiments/uav`, the **real-data instance**
of the HAM gauge/identifiability program (the data-only companion to
`experiments/arm`). One claim, developed carefully and tested against ground
truth at every step:

> A multirotor's cost of motion is a direction-**asymmetric** energy geometry.
> Its exact (gravity) part — the climb-cost ledger `k` — is **invisible in the
> flight paths** (a gauge symmetry) and **recoverable from the energy ledger by a
> convex least-squares**. The autopilot's own wind estimate is independent ground
> truth for the complementary (stretch) channel.

The plan:

1. **The model** — why segment energy splits by *parity under reversal* into an
   even nuisance and an odd drift 1-form, and why that split is robust to the
   two things that break naive cost models on real data.
2. **From logs to segments** — a flight through the real ingest pipeline: PX4
   NED → ENU, densified geometry, `∫V·I dt` energy, quality gates.
3. **The ledger** — a convex per-log fit; held-out *directed-energy* skill.
4. **The gauge** — same flights, different `k`: identical trajectories, different
   energy ledgers; the profile-likelihood that is flat for shape, sharp for energy.
5. **The wind channel** — recover the horizontal drift and validate it against the
   onboard EKF, an instrument the fit never saw.
6. **The real fleet** — the claim on the physical world: 129 crowd-sourced PX4
   flights of 33 multirotors; climb costs more in 98%, and `k` is a per-vehicle
   constant.
7. **Transfer, validation, and honest limits.**

Sections 2-5 run on synthetic flights with known `k`, `W` — the mandatory bridge
(`spec/uav_energy_gauge_PLAN.md` §4) — so every number is checkable against ground
truth *before* §6 points the identical estimator at real logs.
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Setup

We import the experiment package and HAM's plotting palette. The estimators are
pure NumPy/pandas convex least-squares — no JAX, no solver in the loop, no seeds.
""")

co(r"""
import pathlib
import sys

_p = pathlib.Path.cwd()
while not (_p / "experiments").exists() and _p != _p.parent:
    _p = _p.parent
sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from ham.vis.style import PALETTE, use_ham_style

from experiments.uav import (
    EVEN_NAMES, IngestConfig, PowerModel, RawFlightSpec, align_flight,
    directed_energy_r2, even_features, fit_per_log, fit_pooled, fleet_predictor,
    implied_wind, implied_wind_field, k_consistency, k_series, ledger_conditioned,
    make_vortex, negative_control, odd_features, reverse_segments, segment_track,
    simulate_raw_flight, split_segments, synthesize_fleet, wind_field_cosine,
)

pio.renderers.default = "plotly_mimetype"
use_ham_style()


def quiver(P, V, color, name, sc=1.0, width=2):
    xs, ys = [], []
    for p, v in zip(P, V):
        xs += [p[0], p[0] + sc * v[0], None]
        ys += [p[1], p[1] + sc * v[1], None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width),
                      name=name, hoverinfo="skip")


print("ready")
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. The model: energy splits by parity into nuisance + drift

Integrate a multirotor's electrical power over a segment `Δx` flown in a
*measured* time `Δt`. Grouping terms by how they transform when the segment is
flown backwards (`Δx → -Δx`, same duration/length/speed):

$$E_{\text{seg}} \approx \underbrace{c_0\,\Delta t + c_1\,\ell + c_2\,\ell^2/\Delta t + s\,|\Delta z|}_{\text{even (reversal-invariant): hover, drag, vertical overhead}} \;+\; \underbrace{k\,\Delta z + b\cdot\Delta xy}_{\text{odd (reversal-negating): the drift 1-form }\beta}$$

- **`k = mg(1/\eta_\uparrow + \eta_\downarrow)/2`** is the **climb-cost ledger** —
  the exact form `β_grav = k\,dz = d(kz)`. This is what the gauge theorem is about.
- **`b = -2c_2 W`** (exact for quadratic drag) is the horizontal wind coupling.
- The even bracket is a **nuisance model**: fitted, never interpreted. It absorbs
  hover (cost at `v=0`), drag, the speed profile, and the *symmetric* half `s|Δz|`
  of the vertical cost (climbing **and** descending both cost extra over cruise).

The key robustness fact: potential-energy exchange `mgΔz` is *parametrization-free*
— it does not care how fast you flew the segment — so the ledger survives
non-1-homogeneity and arbitrary speed profiles. Below: even features are invariant
under reversal, odd features negate. That orthogonality is what keeps the nuisance
out of the ledger.
""")

co(r"""
model = PowerModel()                      # mass 1.5 kg, hover 250 W, quad drag, eta_up 0.7
df, truth = synthesize_fleet(24, model=model, wind=(3.0, -2.0), noise=0.02,
                             mass_cv=0.08, n_legs=(10, 16), seed=7)
print(f"{df['log_id'].nunique()} flights, {len(df)} segments; "
      f"true ledger k = {model.k:.2f} J/m,  vertical overhead s = {model.s_even:.2f} J/m")

rev = reverse_segments(df)
print("\nparity check (max abs difference over all segments):")
print(f"  even features  |f(reverse) - f(forward)| = "
      f"{np.abs(even_features(rev) - even_features(df)).max():.2e}   (invariant)")
print(f"  odd  features  |f(reverse) + f(forward)| = "
      f"{np.abs(odd_features(rev) + odd_features(df)).max():.2e}   (negates)")
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. From logs to segments: the ingest pipeline

Real logs are PX4 ULog files. The pipeline (`ingest.py`) is four pure-pandas
stages — align asynchronous topics onto a uniform grid, convert **NED → ENU**
(`vehicle_local_position` is z-*down*; getting this wrong silently inverts the
ledger), slice into segments with **densified** length (`Σ|Δpos|` over the fine
samples, always ≥ the chord — the tunneling guard) and honest `∫V·I dt` energy,
then apply the quality gates. Here we drive it with a synthetic PX4-convention
raw flight so it runs offline; on real data only the *parser* changes.
""")

co(r"""
spec = RawFlightSpec(
    waypoints=np.array([[0, 0, 0], [0, 0, 45], [140, 0, 55], [140, 90, 45],
                        [0, 90, 55], [0, 0, 45], [0, 0, 0]], float),
    speed=12.0, climb_speed=3.0, climb_current_gain=0.5, wind=(4.0, 0.0),
)
raw = simulate_raw_flight(spec, log_id="demo")
track = align_flight(raw)
seg = segment_track(track, raw, IngestConfig(trim_s=0.0, seg_dt=4.0))
chord = np.sqrt(seg[["dx", "dy", "dz"]].pow(2).sum(1))
print(f"aligned {len(track)} grid samples -> {len(seg)} segments")
print(f"densified length >= chord for every segment: {(seg['length'] >= chord - 1e-6).all()}")
print(f"climb segments have dz>0 and outdraw descents: "
      f"{(seg[seg.dz>2].energy/seg[seg.dz>2].dt).mean():.0f} W vs "
      f"{(seg[seg.dz<-2].energy/seg[seg.dz<-2].dt).mean():.0f} W")

t = np.array(track["t"]); p = track[["e", "n", "u"]].to_numpy()
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]],
                    column_widths=[0.55, 0.45],
                    subplot_titles=("the flight (ENU, up recovered from NED)",
                                    "electrical power  P = V·I"))
fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                           line=dict(color=PALETTE["primary"], width=4), name="path"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=np.array(track["power"]), mode="lines",
                         line=dict(color=PALETTE["rose"], width=2), name="P [W]"),
              row=1, col=2)
fig.update_layout(height=440, paper_bgcolor="white",
                  scene=dict(xaxis_title="east", yaxis_title="north", zaxis_title="up"),
                  margin=dict(l=0, r=0, t=40, b=0))
fig.update_xaxes(title="time [s]", row=1, col=2)
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. The ledger: a convex fit with held-out skill

A per-log ridge least-squares of `[even nuisance | k·Δz | wind]` on the segment
energies recovers each vehicle's `k`. We never trust in-sample fit; the headline
is **held-out directed-energy R²**: match each climb to a similar descent and
score the predicted energy *difference*. Matching makes the even bracket nearly
cancel, so the score is dominated by the odd channel — a model with the right
nuisance but no ledger keeps only the residue that imperfect matching leaves
behind, far below the full model.
""")

co(r"""
fits = fit_per_log(df)
k_hat = k_series(fits).sort_index()
k_true = truth.k_by_log.sort_index()
train, test = split_segments(df, test_frac=0.3, seed=0)
de_r2, n_pairs = directed_energy_r2(test, fleet_predictor(fit_per_log(train)))
print(f"within-airframe ledger consistency:  CV = {k_consistency(k_hat):.3f}")
print(f"held-out directed-energy R² = {de_r2:.3f}  ({n_pairs} climb/descent pairs)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(k_true), y=np.array(k_hat), mode="markers",
                         marker=dict(size=9, color=PALETTE["primary"]), name="per flight"))
lim = [float(k_true.min()) - 0.5, float(k_true.max()) + 0.5]
fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                         line=dict(color=PALETTE["ink"], dash="dash"), name="identity"))
fig.update_layout(height=430, width=520, paper_bgcolor="white",
                  title="Per-flight ledger recovery (mass jitter along the diagonal)",
                  xaxis_title="true k [J/m]", yaxis_title="recovered k̂ [J/m]")
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. The gauge: the ledger is invisible in the paths

Here is the piece of science. The gravity ledger `k·dz` is an **exact** 1-form, so
adding it to the cost changes the value of every path by a boundary term
`k(z_B - z_A)` — the same for all paths between the same endpoints — and the choice
of path **not at all**. That is Finsler projective invariance (= potential-based
reward shaping = the exact part of a Hodge split, `spec/exact_drift_gauge_equivalence.md`).

We show it concretely on flight hardware: fly two fleets **identically** but with
different `k` (different efficiency). The trajectories come out *bit-identical*; the
energy ledgers differ. A profile-likelihood over an assumed `k` — refitting the even
nuisance at each value — is a sharp U for the **energy** channel (it pins `k`) and
exactly flat for the **path** channel (the paths never depended on `k`).
""")

co(r"""
mA, mB = PowerModel(eta_up=0.70), PowerModel(eta_up=0.45)   # k ~ 12 vs ~ 17
dfA, _ = synthesize_fleet(24, model=mA, wind=(3.0, -2.0), noise=0.0, n_legs=(10, 16), seed=7)
dfB, _ = synthesize_fleet(24, model=mB, wind=(3.0, -2.0), noise=0.0, n_legs=(10, 16), seed=7)
path_diff = np.abs(dfA[["dx","dy","dz"]].to_numpy() - dfB[["dx","dy","dz"]].to_numpy()).max()
kA = k_series(fit_per_log(dfA, ridge=0.0)).mean()
kB = k_series(fit_per_log(dfB, ridge=0.0)).mean()
print(f"same flights, different k:  max trajectory difference = {path_diff:.1e} m")
print(f"  energy differs; the convex solve separates them: k̂_A={kA:.2f} (true {mA.k:.2f}), "
      f"k̂_B={kB:.2f} (true {mB.k:.2f})")

# profile-likelihood over assumed k (per-log energy misfit, refit even+wind at each k)
k_grid = np.linspace(0.3*model.k, 1.8*model.k, 41)
curves = []
for _, sub in df.groupby("log_id"):
    A = np.hstack([even_features(sub), odd_features(sub, ("dx","dy"))])
    dz = sub["dz"].to_numpy(float); y = sub["energy"].to_numpy(float)
    rms = np.array([np.sqrt(np.mean((y - k*dz - A @ np.linalg.lstsq(A, y - k*dz, rcond=None)[0])**2))
                    for k in k_grid])
    curves.append(rms / rms.min())
curves = np.array(curves)

fig = go.Figure()
for c in curves:
    fig.add_trace(go.Scatter(x=k_grid, y=c, mode="lines",
                             line=dict(color=PALETTE["primary"], width=1),
                             opacity=0.25, showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=k_grid, y=curves.mean(0), mode="lines",
                         line=dict(color=PALETTE["primary"], width=3), name="energy channel"))
fig.add_hline(y=1.0, line=dict(color=PALETTE["rose"], width=3),
              annotation_text="path-shape channel (k-invariant)")
fig.add_vline(x=model.k, line=dict(color=PALETTE["green"], dash="dash"),
              annotation_text="true k")
fig.update_layout(height=440, paper_bgcolor="white",
                  title="Energy identifies k; path shape cannot (projective invariance)",
                  xaxis_title="assumed ledger k [J/m]",
                  yaxis_title="relative misfit (1 = best-fit k)", yaxis_range=[0.9, 2.5])
fig.show()
""")

md(r"""
And the falsification check: regress the odd part against a **fake** east-west
potential `k'·dx`. With no real east-west drift it must come back ≈ 0, while the
genuine vertical ledger is large — the estimator is not manufacturing signal.
""")

co(r"""
df0, _ = synthesize_fleet(24, model=model, wind=(0.0, 0.0), noise=0.02, n_legs=(10, 16), seed=8)
k_fake = float(np.abs(negative_control(df0)).median())
k_real = float(k_series(fit_per_log(df0, odd=("dz",))).median())
print(f"negative control:  |k'(dx, fake)| = {k_fake:.3f}  vs  k(dz, real) = {k_real:.3f}  "
      f"({100*k_fake/k_real:.0f}% of k)")
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. The wind channel, validated against the EKF

The odd part carries a horizontal drift alongside the vertical ledger. Recovering
it with gradient + curl RBF atoms (the `recover_form_lsq` pattern) and converting
to a wind vector (`Ŵ = -b̂/2ĉ₂`) gives a spatial wind field — which we check against
the autopilot's **own EKF wind estimate**, an instrument the energy fit never saw.
""")

co(r"""
wind_fn = make_vortex(center=(40.0, -30.0), strength=5.0, radius=160.0)
dfw, _ = synthesize_fleet(30, wind=wind_fn, noise=0.02, n_legs=(10, 16),
                          wind_meas_noise=0.4, seed=11)
g = np.linspace(-260, 260, 5); C1, C2 = np.meshgrid(g, g, indexing="ij")
centers = np.stack([C1.ravel(), C2.ravel()], -1)
pooled = fit_pooled(dfw, wind="spatial", centers=centers, width=190.0)
w_field = implied_wind_field(pooled)
print(f"spatial wind field vs onboard EKF: global cosine = {wind_field_cosine(w_field, dfw):.3f}")

gx = np.linspace(dfw.mid_x.min(), dfw.mid_x.max(), 11)
gy = np.linspace(dfw.mid_y.min(), dfw.mid_y.max(), 11)
GX, GY = np.meshgrid(gx, gy); pts = np.stack([GX.ravel(), GY.ravel()], -1)
true = np.stack([wind_fn(p) for p in pts]); rec = np.stack([w_field(p) for p in pts])
fig = go.Figure()
fig.add_trace(quiver(pts, true, PALETTE["ink"], "true / EKF field", sc=14, width=3))
fig.add_trace(quiver(pts, rec, PALETTE["rose"], "recovered from energy", sc=14, width=1.5))
fig.update_layout(height=520, width=560, paper_bgcolor="white",
                  title="Wind field from energy asymmetry vs the onboard EKF",
                  xaxis_title="east [m]", yaxis_title="north [m]")
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. The real fleet: the ledger on the physical world

Everything above ran on the synthetic bridge, so every number was checkable.
Now the **same estimator — not one line changed** — on real crowd-sourced PX4
Flight Review logs (CC-BY). Five flights of three physical vehicles are frozen
as a test fixture (`tests/fixtures/uav/real_segments.csv`), so this section
always runs offline; when the full curated corpus
(`data/uav_fleet/segments.parquet`) is present, it reproduces the fleet headline:

> **From 129 crowd-sourced flights of 33 multirotors: climb costs more (k > 0)
> in 98%**, median `k` = 7.9 J/m (physical for a ~1 kg quad at `η↑ ≈ 0.65`), and
> `k` is a **per-vehicle constant** — median within-vehicle CV 0.35 against
> cross-vehicle CV 2.09, a six-fold separation. And by §5, none of it was
> readable from the flight paths.

Real data is also where the honest boundaries live: ~59% of public logs carry no
current sensor, most of the rest never climb — the ledger is unidentifiable
without vertical dynamic range, so `ledger_conditioned` gates each flight on its
*geometry*, before any fit. The survivorship path (421k public logs → 129
conditioned flights) is recorded in `spec/uav_energy_gauge_OUTCOMES.md`.
""")

co(r"""
fixture = pd.read_csv(_p / "tests" / "fixtures" / "uav" / "real_segments.csv")
ks_fix = k_series(fit_per_log(fixture)).sort_values()
print(f"frozen fixture: {fixture['log_id'].nunique()} real flights, "
      f"{len(fixture)} segments, {fixture['vehicle_uuid'].nunique()} physical vehicles")
print("per-flight ledger k̂ [J/m]:",
      "  ".join(f"{str(l)[:8]}…={k:.1f}" for l, k in ks_fix.items()))

# One real flight, seen the way the estimator sees it: altitude vs time,
# each segment colored by its electrical power — the climbs burn hot.
lid = fixture["log_id"].value_counts().idxmax()
g = fixture[fixture["log_id"] == lid].sort_values("t0")
fig = go.Figure(go.Scatter(
    x=np.array(g["t0"]), y=np.array(g["mid_z"]), mode="markers+lines",
    line=dict(color="rgba(120,120,120,0.3)", width=1),
    marker=dict(size=8, color=np.array(g["energy"] / g["dt"]), colorscale="Inferno",
                colorbar=dict(title="P [W]")),
))
fig.update_layout(height=400, paper_bgcolor="white",
                  title=f"A real PX4 flight ({str(lid)[:8]}…, k̂ = {ks_fix[lid]:.1f} J/m): "
                        "climbs burn hot, descents run cool",
                  xaxis_title="time in log [s]", yaxis_title="segment altitude [m]")
fig.show()
""")

co(r"""
corpus = _p / "data" / "uav_fleet" / "segments.parquet"
if not corpus.exists():
    print("full corpus not on disk (needs pyarrow + the curated pull) — "
          "the frozen numbers live in spec/uav_energy_gauge_OUTCOMES.md")
else:
    fleet = pd.read_parquet(corpus)
    cond = ledger_conditioned(fleet)
    sub = fleet[fleet["log_id"].astype(str).isin(cond[cond].index)].reset_index(drop=True)
    ks = k_series(fit_per_log(sub))
    per_veh = {u: k_series(fit_per_log(g)) for u, g in sub.groupby("vehicle_uuid")
               if g["log_id"].nunique() >= 3}
    within = float(np.median([k_consistency(v) for v in per_veh.values()]))
    cross = k_consistency(ks)
    print(f"{len(ks)} conditioned flights / {sub['vehicle_uuid'].nunique()} vehicles:  "
          f"k>0 in {100 * (ks > 0).mean():.0f}%,  median k = {ks.median():.1f} J/m")
    print(f"k is a per-vehicle constant: within-vehicle CV = {within:.2f}  "
          f"vs  cross-vehicle CV = {cross:.2f}")

    # arcsinh(k/30) = a symlog axis: the 2 J/m clusters and the 300 J/m
    # vehicles stay readable together, with nothing clipped.
    tr = lambda k: np.arcsinh(np.asarray(k, float) / 30.0)
    ticks = np.array([-30, -10, 0, 10, 30, 100, 300])
    lo, hi = -20.0, 80.0
    n_off = int(((ks < lo) | (ks > hi)).sum())
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], subplot_titles=(
        f"the fleet ledger: k > 0 in {100 * (ks > 0).mean():.0f}% of {len(ks)} flights",
        f"k clusters by physical vehicle ({len(per_veh)} with ≥3 flights)"))
    fig.add_trace(go.Histogram(x=np.array(ks[(ks >= lo) & (ks <= hi)]),
                               xbins=dict(start=lo, end=hi, size=2.5),
                               marker_color=PALETTE["primary"], showlegend=False), 1, 1)
    fig.add_vline(x=float(ks.median()), line=dict(color=PALETTE["rose"], dash="dash"),
                  annotation_text=f"median {ks.median():.1f} J/m", row=1, col=1)
    fig.add_vline(x=0.0, line=dict(color=PALETTE["ink"], width=1), row=1, col=1)
    colors = [PALETTE[c] for c in
              ("primary", "accent", "teal", "violet", "rose", "green", "ink")]
    for i, (u, v) in enumerate(sorted(per_veh.items(), key=lambda kv: -len(kv[1]))):
        fig.add_trace(go.Scatter(x=[i] * len(v), y=tr(v), mode="markers",
                                 marker=dict(size=9, color=colors[i % len(colors)], opacity=0.65),
                                 name=f"{len(v)} flights", showlegend=False), 1, 2)
        fig.add_trace(go.Scatter(x=[i - 0.25, i + 0.25], y=tr([v.median()] * 2), mode="lines",
                                 line=dict(color=PALETTE["ink"], width=3), showlegend=False), 1, 2)
    fig.update_xaxes(title=f"per-flight k̂ [J/m]   ({n_off} flights off-scale, up to "
                           f"{ks.max():.0f})", row=1, col=1)
    fig.update_yaxes(title="# flights", row=1, col=1)
    fig.update_xaxes(title="vehicle (sorted by #flights)",
                     tickvals=list(range(len(per_veh))),
                     ticktext=[f"{len(v)} fl" for _, v in
                               sorted(per_veh.items(), key=lambda kv: -len(kv[1]))], row=1, col=2)
    fig.update_yaxes(title="per-flight k̂ [J/m]  (symlog)", tickvals=tr(ticks),
                     ticktext=[f"{t:g}" for t in ticks], row=1, col=2)
    fig.update_layout(height=430, paper_bgcolor="white",
                      title="The real PX4 fleet: the climb-cost ledger is real, "
                            "correctly signed, and a per-vehicle constant")
    fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Transfer, validation, and honest limits

**Transfer.** The exact part is a single scalar coordinate on the vehicle family,
so carrying a model across airframes should be *gauge + a convex recalibration*:
reuse the source's hover+drag shape, re-fit only the vertical ledger (and a scale)
from a handful of the target's segments. `run_stage_d_transfer` shows this reaches
held-out R² ≥ 0.9 at K=4 calibration segments, where a from-scratch fit is starved.

**Running on new real data.** §7 used the curated fleet corpus; pointing the
pipeline at any fresh directory of logs is one line:

```python
from experiments.uav import load_corpus
df = load_corpus("data/uav")   # a dir of *.ulg (parsed) or *.parquet (preprocessed)
```

after `pip install 'hamtools[uav]'` and `download_logs(...)`. `load_corpus` returns
the same schema, so every fit above is unchanged — and on logs that carry the
`wind` topic, the §6 EKF cross-check runs against a genuinely independent sensor
(none of the pulled corpus had it; that channel remains open on real data).

**Validation.** The runnable ladder `python -m experiments.uav.validate` (U0-U6)
and `pytest tests/test_uav.py` gate the machinery: ingest integrity (NED→ENU, energy
balance, densified length), the segmentation gates, recovery of a *known* k/W (the
mandatory bridge), even/odd orthogonality, the pilot gates, the Stage-B bar, and the
negative control.

**Honest limits.**

- **Not a Finsler metric at `v→0`.** Hover costs power at zero velocity; the model
  works at the segment-energy level with measured `Δt`, where 1-homogeneity is not
  assumed. Only the odd part is a genuine 1-form.
- **Descent is not regen.** `η↓` is small; `k` is still half the up/down gap.
- **`k` is per-log** (mass/efficiency unlogged). The science is consistency and
  held-out prediction, never a book value.
- **The uniform-wind read anti-aligns on rotational fields** (a per-flight single
  vector can't represent a vortex a flight loops around) — visible as the outliers
  in Stage C; the spatial field handles it, and we show both.
- **Filtered, not explained:** vortex-ring, acro, and ground effect are removed by
  the quality gates (cruise-band on 3-D speed, `|a|` cap, AGL floor, battery sanity),
  and survivorship is reported, never hidden.
""")

# ===========================================================================
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")

if "--run" in sys.argv:
    from nbclient import NotebookClient

    print("executing...")
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO)}})
    client.execute()
    OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"executed and saved {OUT}")
