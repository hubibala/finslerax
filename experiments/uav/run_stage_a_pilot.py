"""Stage A — the pilot GO/NO-GO gate on a crowd-scale corpus (§4).

Before any estimator is trusted, three cheap questions decide whether the energy
channel is clean enough to carry the ledger claim, answered on a corpus run
through the *real* ingest pipeline (``simulate_raw_flight`` → ``ingest_flight``,
so the frame/energy/filter machinery is exercised, not bypassed):

1. **U-A1 — the asymmetry has the right sign.** In ≥ 90% of logs, climbing burns
   more specific power than descending. This is the raw physical signal the whole
   experiment rests on; if it is not there, the battery telemetry is too dirty to
   proceed (and the plan says investigate the signal before abandoning).
2. **U-A2 — the crude ledger is consistent.** The two-bin ``k̂`` (up/down
   specific-power gap over climb-rate gap, computed only on vertical-dominant
   climb/descent segments) has within-airframe CV < 50% — a deliberately crude
   bar that Stage B tightens.
3. **Data-quality census.** What fraction of segments survive the cruise-band /
   accel-cap / AGL-floor gates, reported per gate — survivorship shown, not hidden.

Until the real corpus is downloaded (``ingest.download_logs``), the corpus here
is synthetic PX4-convention raw flights with a known climb-cost ledger, so the
gate is exercised deterministically. Swapping in real ``.ulg`` paths is a
one-line change (``parse_ulog`` in place of ``simulate_raw_flight``).

Run:  python -m experiments.uav.run_stage_a_pilot
Writes experiments/uav/visualizations/stage_a_pilot.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import climb_costs_more, two_bin_k
from .ingest import IngestConfig, RawFlightSpec, ingest_flight, simulate_raw_flight

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

N_LOGS = 50
AIRFRAME = "quad_a"
GAIN = 0.5            # climb-current gain [A per m/s] ⇒ ledger k ≈ V·GAIN ≈ 11 J/m
CFG = IngestConfig(min_duration=30.0, seg_dt=4.0, cruise_speed=(3.0, 22.0),
                   accel_cap=6.0, agl_floor=5.0, trim_s=2.0)


def _corpus(n=N_LOGS, seed=0):
    """A fleet of varied waypoint missions as raw PX4-convention flights."""
    rng = np.random.default_rng(seed)
    flights = []
    for i in range(n):
        alt = float(rng.uniform(35.0, 70.0))
        reach = float(rng.uniform(90.0, 200.0))
        # climb-out, a few cruise legs at altitude, descent.
        wps = [[0.0, 0.0, 0.0], [0.0, 0.0, alt]]
        for _ in range(int(rng.integers(4, 8))):
            th = rng.uniform(0, 2 * np.pi)
            wps.append([reach * np.cos(th), reach * np.sin(th),
                        float(np.clip(alt + rng.uniform(-12, 12), 25, 85))])
        wps.append([0.0, 0.0, alt])
        wps.append([0.0, 0.0, 0.0])
        spec = RawFlightSpec(
            waypoints=np.array(wps),
            speed=float(rng.uniform(9.0, 15.0)),
            climb_speed=float(rng.uniform(2.5, 4.0)),  # drones climb slower than cruise
            climb_current_gain=GAIN * float(rng.uniform(0.9, 1.1)),  # mild fleet spread
            wind=(float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4))),
        )
        flights.append(simulate_raw_flight(spec, log_id=f"{AIRFRAME}-{i:03d}",
                                           airframe=AIRFRAME, seed=i))
    return flights


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    flights = _corpus()

    tables, census = [], []
    for raw in flights:
        seg, cen = ingest_flight(raw, CFG)
        census.append(cen)
        if len(seg):
            tables.append(seg)
    census = pd.DataFrame(census)
    df = pd.concat(tables, ignore_index=True)

    print("Stage A — pilot gate on a crowd-scale corpus "
          f"({len(flights)} logs, {len(df)} segments after filtering)")

    # --- U-A1: asymmetry sign -------------------------------------------------
    signs = {lid: climb_costs_more(sub) for lid, sub in df.groupby("log_id")}
    judged = [v for v in signs.values() if v is not None]
    frac_sign = float(np.mean(judged))
    a1 = frac_sign >= 0.90
    print(f"  U-A1 asymmetry sign: climb costs more in {100 * frac_sign:.0f}% of "
          f"{len(judged)} judged logs  [{'PASS' if a1 else 'FAIL'} ≥90%]")

    # --- U-A2: crude ledger consistency --------------------------------------
    kc = pd.Series({lid: two_bin_k(sub) for lid, sub in df.groupby("log_id")}).dropna()
    cv = float(kc.std(ddof=1) / kc.mean())
    a2 = cv < 0.50
    print(f"  U-A2 two-bin k̂: mean={kc.mean():.2f} J/m  CV={cv:.2f}  "
          f"[{'PASS' if a2 else 'FAIL'} <50%]  (V·GAIN≈{22.0 * GAIN:.1f})")

    # --- data-quality census --------------------------------------------------
    survived = census["n_out"].sum() / max(census["n_in"].sum(), 1)
    print(f"  census: {survived:.0%} of segments survive filters  "
          f"(cruise {census['pass_cruise'].mean():.2f}, "
          f"accel {census['pass_accel'].mean():.2f}, agl {census['pass_agl'].mean():.2f})")

    go = a1 and a2
    print(f"\n  VERDICT: {'GO — proceed to Stage B' if go else 'NO-GO — inspect battery signal (§4)'}")

    _figure(df, kc, cv, census, frac_sign)
    return 0 if go else 1


def _figure(df, kc, cv, census, frac_sign):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # (a) the raw signal: specific power vs climb rate, climb bin outdraws descent.
    ax = axes[0]
    vz = df["dz"] / df["dt"]
    vh = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2) / df["dt"]
    sp = df["energy"] / df["dt"]
    vert = np.abs(vz) > vh
    ax.scatter(vz[vert], sp[vert], s=6, alpha=0.3, color="C0", label="vertical-dominant seg")
    ax.scatter(vz[~vert], sp[~vert], s=5, alpha=0.12, color="0.6", label="cruise (excluded)")
    up, dn = vert & (vz > 0.3), vert & (vz < -0.3)
    for m, c, lbl in [(up, "crimson", "climb mean"), (dn, "navy", "descent mean")]:
        ax.plot(vz[m].mean(), sp[m].mean(), "D", color=c, ms=11, mec="k", zorder=5, label=lbl)
    ax.set_title(f"Raw asymmetry: climb burns more\n(right sign in {100 * frac_sign:.0f}% of logs)")
    ax.set_xlabel("climb rate  ż = dz/dt  [m/s]"), ax.set_ylabel("specific power  E/dt  [W]")
    ax.axvline(0, color="k", lw=0.7), ax.legend(fontsize=8, loc="upper left")

    # (b) per-log crude ledger: a tight cluster around V·GAIN.
    ax = axes[1]
    ax.hist(kc, bins=18, color="C0", alpha=0.85, edgecolor="k")
    ax.axvline(kc.mean(), color="crimson", lw=2, label=f"mean {kc.mean():.1f} J/m")
    ax.axvline(22.0 * GAIN, color="green", lw=2, ls="--", label=f"truth V·GAIN={22.0 * GAIN:.1f}")
    ax.set_title(f"Crude two-bin ledger k̂ across the fleet\n(within-airframe CV={cv:.2f}, bar <0.50)")
    ax.set_xlabel("two-bin k̂  [J/m]"), ax.set_ylabel("# logs"), ax.legend(fontsize=8)

    # (c) survivorship census: what the quality gates keep.
    ax = axes[2]
    gates = ["cruise", "accel", "agl"]
    vals = [census[f"pass_{g}"].mean() for g in gates]
    overall = census["n_out"].sum() / max(census["n_in"].sum(), 1)
    labels = [*gates, "overall"]
    heights = [*vals, overall]
    bars = ax.bar(labels, heights, color=["C0", "C0", "C0", "crimson"])
    for b, v in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0%}", ha="center", va="bottom")
    ax.set_ylim(0, 1.15), ax.set_ylabel("fraction of segments passing")
    ax.set_title("Data-quality census\n(survivorship reported, not hidden)")

    fig.suptitle("Stage A — pilot gate: the energy channel carries a clean, "
                 "consistent climb-cost asymmetry", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "stage_a_pilot.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    sys.exit(main())
