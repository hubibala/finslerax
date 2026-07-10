"""The UAV validation ladder as a runnable gate — machinery before the science.

Run ``python -m experiments.uav.validate`` to check each rung and print a table.
The posture mirrors ``experiments/arm/validate.py`` and ``experiments/marine``:
the experiment ships its own ground-truth cross-checks, not just a headline
number. Every rung that touches real data must first be green on synthetic
flights whose ``k``/``W`` are known exactly (§4: "U2 MUST be green before
touching real-data conclusions"), so the ladder runs entirely on the
synthetic bridge and the offline ingest generator.

    U0  ingest integrity — NED→ENU sign, ∫V·I dt balance, densified length ≥ chord.
    U1  segmentation — cruise-band / AGL gates and an honest survivorship census.
    U2  estimator math — recover known k and W within tolerance (the bridge).
    U3  direction balance / even-odd orthogonality diagnostics.
    U4  Stage-A pilot gates frozen — asymmetry sign ≥ 90%, two-bin k CV < 50%.
    U5  Stage-B success bar — held-out directed-energy R² ≥ 0.7, k CV ≤ 20%.
    U6  negative control — a fake east-west potential recovers ≈ 0.

Each rung returns ``(passed, detail)``; the exit code is nonzero if any fails,
so the stage scripts can depend on this as a CI-style gate.
"""

from __future__ import annotations

import sys

import numpy as np

from .estimate import (
    fit_per_log,
    fleet_predictor,
    implied_wind,
    k_series,
    negative_control,
)
from .evaluate import (
    climb_costs_more,
    directed_energy_r2,
    direction_balance,
    even_odd_leakage,
    k_consistency,
    split_segments,
    two_bin_k,
)
from .ingest import (
    IngestConfig,
    RawFlightSpec,
    align_flight,
    apply_filters,
    segment_track,
    simulate_raw_flight,
)
from .medium import reverse_segments
from .synthetic import PowerModel, synthesize_fleet

# Pinned synthetic corpus so U4/U5/U6 are deterministic.
MODEL = PowerModel()
WIND = (3.0, -2.0)


def _fleet(n=20, *, mass_cv=0.0, noise=0.02, wind=WIND, seed=0):
    """A pinned multirotor fleet with ≥5-min flights (n_legs sized for it)."""
    return synthesize_fleet(
        n, model=MODEL, wind=wind, noise=noise, mass_cv=mass_cv,
        n_legs=(10, 16), seed=seed,
    )


def _box_raw(**spec_kw):
    wps = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 40.0], [120.0, 0.0, 55.0], [120.0, 90.0, 45.0],
        [0.0, 90.0, 55.0], [0.0, 0.0, 40.0], [0.0, 0.0, 0.0],
    ])
    return simulate_raw_flight(RawFlightSpec(waypoints=wps, **spec_kw))


# ---------------------------------------------------------------------------
def u0_ingest_integrity() -> tuple[bool, str]:
    """NED→ENU sign, segment energy sums to the flight integral, length ≥ chord."""
    from .ingest import energy_balance

    raw = _box_raw()
    track = align_flight(raw)
    cfg = IngestConfig(trim_s=0.0, seg_dt=4.0)
    seg = segment_track(track, raw, cfg)
    bal = energy_balance(track)
    chord = np.sqrt(seg[["dx", "dy", "dz"]].pow(2).sum(axis=1).to_numpy())
    climb = seg[seg["dz"] > 2.0]
    e_bal = abs(seg["energy"].sum() - bal["energy_j"]) / bal["energy_j"]
    len_ok = bool((seg["length"].to_numpy() >= chord - 1e-6).all())
    ok = len(climb) > 0 and e_bal < 0.02 and len_ok
    return ok, f"climb dz>0 ✓ | Σseg vs ∫flight rel={e_bal:.3f} | length≥chord={len_ok}"


def u1_segmentation() -> tuple[bool, str]:
    """The cruise-band and AGL gates each drop the right segments; census is sane."""
    raw = _box_raw(speed=12.0)
    seg = segment_track(align_flight(raw), raw, IngestConfig(trim_s=0.0, seg_dt=3.0))
    kept, census = apply_filters(seg, IngestConfig(agl_floor=10.0, cruise_speed=(6.0, 20.0)))
    agl_ok = bool((kept["agl_min"] >= 10.0).all())
    speed_ok = bool((kept["speed"] >= 6.0).all())
    dropped = census["n_out"] < census["n_in"]
    ok = agl_ok and speed_ok and dropped and 0.0 < census["kept_frac"] <= 1.0
    return ok, f"kept {census['n_out']}/{census['n_in']} (agl {census['pass_agl']:.2f}, cruise {census['pass_cruise']:.2f})"


def u2_estimator_math() -> tuple[bool, str]:
    """Recover the known ledger and wind from a noisy synthetic fleet (the bridge)."""
    df, _ = _fleet(20, noise=0.02, seed=1)
    fits = fit_per_log(df)
    ks = k_series(fits)
    k_err = abs(ks.mean() - MODEL.k) / MODEL.k
    w_hat = np.mean([implied_wind(f) for f in fits.values()], axis=0)
    w_cos = float(np.dot(w_hat, WIND) / (np.linalg.norm(w_hat) * np.linalg.norm(WIND)))
    ok = k_err < 0.05 and w_cos > 0.95
    return ok, f"k̂/k err={k_err:.3f}  wind cos={w_cos:.3f}  (|Ŵ|={np.linalg.norm(w_hat):.2f})"


def u3_balance_orthogonality() -> tuple[bool, str]:
    """Reversal-balanced sets have zero even/odd leakage; missions are vert-balanced."""
    import pandas as pd

    df, _ = _fleet(6, seed=2)
    balanced = pd.concat([df, reverse_segments(df)], ignore_index=True)
    leak = even_odd_leakage(balanced)
    vbal = float(direction_balance(df)["vertical"].mean())
    ok = leak < 1e-10 and vbal < 0.15
    return ok, f"balanced leakage={leak:.1e}  vertical imbalance={vbal:.3f}"


def u4_pilot_gates() -> tuple[bool, str]:
    """Stage-A frozen: asymmetry sign correct in ≥90% of logs, two-bin k CV < 50%."""
    df, _ = _fleet(30, mass_cv=0.05, noise=0.03, seed=3)
    signs = [climb_costs_more(sub) for _, sub in df.groupby("log_id")]
    frac = np.mean([s for s in signs if s is not None])
    kc = np.array([two_bin_k(sub) for _, sub in df.groupby("log_id")])
    cv = float(np.nanstd(kc, ddof=1) / np.nanmean(kc))
    ok = frac >= 0.9 and cv < 0.5
    return ok, f"asymmetry-sign correct {100 * frac:.0f}%  two-bin k CV={cv:.2f}"


def u5_stage_b_bar() -> tuple[bool, str]:
    """Stage-B success bar: held-out directed-energy R² ≥ 0.7, within-airframe k CV ≤ 20%."""
    df, _ = _fleet(20, mass_cv=0.08, noise=0.02, seed=4)
    train, test = split_segments(df, test_frac=0.3, seed=0)
    fits = fit_per_log(train)
    r2, n = directed_energy_r2(test, fleet_predictor(fits))
    cv = k_consistency(k_series(fit_per_log(df)))
    ok = r2 >= 0.7 and cv <= 0.20
    return ok, f"held-out directed-energy R²={r2:.3f} ({n} pairs)  within-airframe k CV={cv:.2f}"


def u6_negative_control() -> tuple[bool, str]:
    """A fake east-west potential recovers ≈ 0 (|k'| ≤ 10% of the true k)."""
    df, _ = _fleet(20, noise=0.02, wind=(0.0, 0.0), seed=5)
    k_fake = float(np.abs(negative_control(df)).median())
    k_true = float(k_series(fit_per_log(df, odd=("dz",))).median())
    ok = k_fake < 0.10 * MODEL.k and k_true > 5.0 * k_fake
    return ok, f"|k'_fake|={k_fake:.3f}  k={k_true:.3f}  ratio={k_true / max(k_fake, 1e-9):.0f}x"


LADDER = [
    ("U0  Ingest integrity", u0_ingest_integrity),
    ("U1  Segmentation gates", u1_segmentation),
    ("U2  Estimator math", u2_estimator_math),
    ("U3  Balance / orthogonality", u3_balance_orthogonality),
    ("U4  Stage-A pilot gates", u4_pilot_gates),
    ("U5  Stage-B success bar", u5_stage_b_bar),
    ("U6  Negative control", u6_negative_control),
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # ladder detail uses ✓/²/× glyphs
    print("UAV energy-ledger validation ladder\n" + "=" * 72)
    all_ok = True
    for name, fn in LADDER:
        try:
            ok, detail = fn()
        except Exception as exc:  # report any rung failure verbatim
            ok, detail = False, f"ERROR: {type(exc).__name__}: {exc}"
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:28s} {detail}")
    print("=" * 72)
    print("ALL RUNGS PASSED" if all_ok else "SOME RUNGS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
