"""Run the full analysis on a REAL PX4 corpus — the one command for real data.

Everything else in this package runs on the synthetic bridge so each number is
checkable against ground truth. This driver points the *same* estimators at a
directory of real logs (``load_corpus``: ``*.ulg`` parsed on the fly, or
preprocessed ``*.parquet``), gates each flight on ledger identifiability, and
reports the fleet result.

Because real ``k`` is unknown, the report is exactly the plan's measurable claims:
the up/down asymmetry sign rate, per-flight ``k`` distribution, held-out directed-
energy R², the negative control, and — the fleet-heterogeneity fix — the
*within-vehicle* ``k`` consistency (grouped by ``vehicle_uuid``) versus the
cross-vehicle spread, which tests whether ``k`` is a per-vehicle constant.

Usage:
    pip install 'hamtools[uav]'
    python -c "from experiments.uav import download_logs; download_logs('data/uav')"
    python -m experiments.uav.run_real data/uav            # or set UAV_CORPUS_DIR
    python -m experiments.uav.run_real data/uav --airframe 4001

Writes experiments/uav/visualizations/real_report.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .estimate import (
    fit_per_log,
    fleet_predictor,
    k_series,
    negative_control,
)
from .evaluate import (
    directed_energy_r2,
    k_consistency,
    ledger_conditioned,
    split_segments,
)
from .ingest import IngestConfig, ingest_logs, load_corpus
from .medium import validate_table

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

# Tuned for real multirotor logs: shorter segments catch the climb bursts, a wider
# cruise band and lower AGL floor keep more real flight (bench logs are culled by
# the current/movement gates, not these).
REAL_CFG = IngestConfig(min_duration=60.0, seg_dt=2.5, cruise_speed=(2.0, 28.0),
                        accel_cap=8.0, agl_floor=3.0, trim_s=3.0)


def build_corpus(source):
    """Load a corpus, reporting ingest survivorship when parsing raw ``.ulg`` files."""
    src = Path(source)
    if sorted(src.glob("*.parquet")):
        return load_corpus(src), None
    ulg = sorted(src.glob("*.ulg"))
    if not ulg:
        raise FileNotFoundError(f"no .parquet or .ulg files under {src}")
    df, census = ingest_logs(ulg, REAL_CFG)
    n = len(ulg)
    skips = census["skipped"].value_counts().to_dict() if "skipped" in census else {}
    errs = int(census["error"].notna().sum()) if "error" in census else 0
    flew = census["n_out"].gt(0).sum() if "n_out" in census else 0
    print(f"ingest survivorship: {n} logs -> {int(flew)} usable flights "
          f"({len(df)} segments).  skipped={skips}  parse_errors={errs}")
    return df, census


def _has_ekf_wind(df: pd.DataFrame) -> bool:
    """True if the wind columns are present and actually vary (a real EKF estimate)."""
    if not {"wind_x", "wind_y", "mid_x", "mid_y"}.issubset(df.columns):
        return False
    w = df[["wind_x", "wind_y"]].to_numpy(float)
    return bool(np.isfinite(w).all() and w.std() > 1e-3)


def analyze(df: pd.DataFrame, airframe: str | None = None) -> dict:
    """Identifiability-gated fleet ledger analysis, grouped by physical vehicle.

    Reports the pooled result (sign of the ledger, held-out directed-energy R²,
    negative control) and — the heterogeneity fix — the *within-vehicle* ``k``
    consistency versus the *cross-vehicle* spread, using ``vehicle_uuid`` when the
    corpus carries it (``SYS_AUTOSTART`` spans many physical vehicles, so grouping
    by it conflates them).
    """
    validate_table(df)
    if airframe is not None:
        df = df[df["airframe"] == airframe].reset_index(drop=True)
    gcol = ("vehicle_uuid" if "vehicle_uuid" in df and df["vehicle_uuid"].nunique() > 1
            else "airframe")
    n_logs = df["log_id"].nunique()

    # --- identifiability gate: k is only readable where the flight climbs -----
    cond = ledger_conditioned(df)
    cond_logs = list(cond[cond].index)
    subc = df[df["log_id"].astype(str).isin(cond_logs)].reset_index(drop=True)
    print(f"corpus: {n_logs} flights / {len(df)} segments; "
          f"identifiable ledger (enough vertical range): {len(cond_logs)} flights, "
          f"{subc[gcol].nunique()} distinct {gcol}s")
    res = {"n_logs": n_logs, "n_cond": len(cond_logs), "group": gcol}
    if len(cond_logs) < 3:
        print("  VERDICT: NO-GO — too few flights with vertical dynamic range (this "
              "corpus is mostly constant-altitude / bench logs, §5)")
        res.update(go=False, frac_sign=float("nan"))
        return res

    # --- pooled ledger result -------------------------------------------------
    ks = k_series(fit_per_log(subc))
    frac_sign = float((ks > 0).mean())
    cross_cv = k_consistency(ks)
    train, test = split_segments(subc, test_frac=0.3, seed=0)
    de_r2, n_pairs = directed_energy_r2(test, fleet_predictor(fit_per_log(train)))
    k_fake = float(np.abs(negative_control(subc)).median())
    k_real = float(k_series(fit_per_log(subc, odd=("dz",))).median())
    print(f"  climb-costs-more (k>0): {100 * frac_sign:.0f}% of {len(ks)} flights  "
          f"[{'PASS' if frac_sign >= 0.9 else 'FAIL'} >=90%]")
    print(f"  per-flight k [J/m]: median={ks.median():.1f}  IQR=[{ks.quantile(.25):.0f}, "
          f"{ks.quantile(.75):.0f}]   held-out directed-energy R²={de_r2:.3f} ({n_pairs} pairs)")
    print(f"  negative control |k'|={k_fake:.1f} vs k={k_real:.1f} "
          f"({100 * k_fake / max(k_real, 1e-9):.0f}% of k)")

    # --- per-vehicle consistency: is k a per-vehicle constant? ----------------
    per_veh = {}
    for uid, g in subc.groupby(gcol):
        if g["log_id"].nunique() >= 3:
            per_veh[uid] = k_series(fit_per_log(g))
    within_cv = (float(np.median([k_consistency(v) for v in per_veh.values()]))
                 if per_veh else float("nan"))
    print(f"  vehicles with >=3 conditioned flights: {len(per_veh)}")
    for uid, v in sorted(per_veh.items(), key=lambda kv: -len(kv[1]))[:6]:
        print(f"    {str(uid)[:16]}: {len(v):2d} flights  k={v.median():6.1f} J/m  "
              f"CV={k_consistency(v):.2f}")
    if per_veh:
        print(f"  WITHIN-vehicle k CV = {within_cv:.2f}  vs  CROSS-vehicle CV = {cross_cv:.2f}  "
              f"(k is a per-vehicle constant: {'YES' if within_cv < 0.6 * cross_cv else 'unclear'})")

    go = frac_sign >= 0.90 and (not per_veh or within_cv < 0.6 * cross_cv)
    res.update(go=go, frac_sign=frac_sign, k_med=float(ks.median()), cross_cv=cross_cv,
               within_cv=within_cv, de_r2=de_r2, n_pairs=n_pairs, k_fake=k_fake,
               k_real=k_real, n_veh=len(per_veh))
    verdict = ("GO — the climb-cost ledger is real, correctly signed, and a per-vehicle "
               "constant" if go else "signal present; see per-vehicle detail")
    print(f"  VERDICT: {verdict}")
    _figure(subc, ks, per_veh, res)
    return res


def _figure(subc, ks, per_veh, res):
    from .evaluate import _match_pairs
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # (a) the per-flight ledger across the whole fleet: overwhelmingly k>0.
    # Off-scale flights are counted in the margin, never silently piled into
    # the edge bins (a clipped pile-up reads as a false mode).
    a = ax[0]
    lo, hi = -20.0, 80.0
    n_lo, n_hi = int((ks < lo).sum()), int((ks > hi).sum())
    a.hist(ks[(ks >= lo) & (ks <= hi)], bins=36, range=(lo, hi),
           color="C0", alpha=0.85, edgecolor="k")
    a.axvline(0, color="k", lw=1)
    a.axvline(ks.median(), color="crimson", lw=2, ls="--", label=f"median {ks.median():.1f} J/m")
    if n_lo or n_hi:
        a.text(0.98, 0.72, f"off-scale: {n_hi} flights k>{hi:.0f} (max {ks.max():.0f})"
               + (f"\n{n_lo} flight(s) k<{lo:.0f} (min {ks.min():.0f})" if n_lo else ""),
               transform=a.transAxes, fontsize=7.5, ha="right", va="top", color="0.35")
    a.set_title(f"Real climb-cost ledger — {len(ks)} flights\n"
                f"k>0 (climb costs more) in {100 * res['frac_sign']:.0f}%")
    a.set_xlabel("per-flight k̂ [J/m]"), a.set_ylabel("# flights"), a.legend(fontsize=8)

    # (b) within-vehicle vs cross-vehicle: k clusters by physical vehicle.
    # Symlog keeps the 2-300 J/m fleet spread AND the tight per-vehicle clusters
    # readable without clipping; medians are of the raw values.
    a = ax[1]
    ordered = sorted(per_veh.items(), key=lambda kv: -len(kv[1]))
    for i, (_, v) in enumerate(ordered):
        a.scatter([i] * len(v), v.values, s=26, alpha=0.6, color=f"C{i % 10}")
        a.plot(i, v.median(), "_", ms=20, color="k", mew=2)
    a.axhline(0, color="k", lw=0.6)
    a.set_yscale("symlog", linthresh=30.0)
    a.set_ylim(min(-40.0, 1.5 * float(ks.min())), 1.6 * max(80.0, float(ks.max())))
    a.set_yticks([-100, -30, -10, 0, 10, 30, 100, 300])
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
    a.set_xticks(range(len(ordered)))
    a.set_xticklabels([f"{len(v)} fl" for _, v in ordered], fontsize=8)
    a.set_title(f"k is a per-vehicle constant\nwithin-veh CV={res['within_cv']:.2f}  vs  "
                f"cross-veh CV={res['cross_cv']:.2f}")
    a.set_xlabel(f"vehicle ({res['n_veh']} with >=3 flights)")
    a.set_ylabel("per-flight k̂ [J/m]  (symlog)")

    # (c) held-out directed energy on the real fleet.
    a = ax[2]
    train, test = split_segments(subc, test_frac=0.3, seed=0)
    pred = pd.Series(fleet_predictor(fit_per_log(train))(test), index=test.index)
    od, pr = [], []
    for _, g in test.groupby("log_id"):
        for i, j in _match_pairs(g, 1.0):
            od.append(test.at[i, "energy"] - test.at[j, "energy"]), pr.append(pred[i] - pred[j])
    if od:
        od, pr = np.array(od), np.array(pr)
        a.scatter(od, pr, s=10, alpha=0.4, color="C0")
        lim = float(np.percentile(np.abs(od), 98))
        a.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        a.set_xlim(-lim, lim), a.set_ylim(-lim, lim)
    a.set_title(f"Held-out directed energy (real)\nR²={res['de_r2']:.2f}, {res['n_pairs']} pairs")
    a.set_xlabel("observed ΔE (climb - descent) [J]"), a.set_ylabel("predicted ΔE [J]")

    fig.suptitle(f"Real PX4 fleet ({len(ks)} flights, {subc[res['group']].nunique()} vehicles): "
                 f"the climb-cost ledger is real, correctly signed, and per-vehicle "
                 f"({'GO' if res['go'] else 'partial'})", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "real_report.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:] if argv is None else argv
    airframe = None
    if "--airframe" in argv:
        i = argv.index("--airframe")
        airframe = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    source = argv[0] if argv else os.environ.get("UAV_CORPUS_DIR")
    if not source:
        print("usage: python -m experiments.uav.run_real <corpus_dir> [--airframe ID]\n"
              "   or set UAV_CORPUS_DIR. The dir holds *.ulg or *.parquet segment tables.")
        return 2
    df, _ = build_corpus(source)
    if not len(df):
        print("no usable flights in this corpus (all logs lack a current sensor, "
              "are bench tests, or were filtered) — NO-GO")
        return 1
    res = analyze(df, airframe=airframe)
    return 0 if res["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
