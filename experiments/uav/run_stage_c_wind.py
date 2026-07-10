"""Stage C — the wind channel, validated against the onboard EKF (§4, the stretch).

The odd part of the segment energy is ``k·dz + b·dxy``: alongside the vertical
ledger sits a *horizontal* drift, the wind coupling. Recovering ``b`` and
converting it to a wind vector (``Ŵ = −b̂/2ĉ₂``, exact for quadratic drag) is
``recover_form_lsq`` on flight hardware — and the payoff is that multirotor logs
carry the autopilot's **own EKF wind estimate**, an instrument the energy fit
never sees. Agreement is the strongest version of the claim: the drift channel,
recovered from energy asymmetries alone, matches an independent sensor.

Two recoveries:

1. **Uniform wind** (headline): one ``(Wx, Wy)`` per flight from the pooled fit,
   compared to the flight-mean EKF wind.
2. **Spatial wind field** (the ``recover_form_lsq`` generalization): a rotational
   field recovered with gradient + curl RBF atoms and compared to the EKF wind
   at every segment midpoint — cosine ≥ 0.6 on flights above a wind-speed floor.

Runs on the synthetic bridge (a known vortex field, EKF columns = true wind +
sensor noise). On real logs the EKF columns come straight from the ``wind``
topic, so this script is unchanged.

Run:  python -m experiments.uav.run_stage_c_wind
Writes experiments/uav/visualizations/stage_c_wind.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .estimate import fit_per_log, fit_pooled, implied_wind, implied_wind_field
from .evaluate import cosine, wind_field_cosine
from .synthetic import make_vortex, synthesize_fleet

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

N_FLIGHTS = 30
SEED = 11
VORTEX = {"center": (40.0, -30.0), "strength": 5.0, "radius": 160.0}
FLOOR = 1.5  # wind-speed floor [m/s] above which the cosine bar applies


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    wind_fn = make_vortex(**VORTEX)
    df, _ = synthesize_fleet(N_FLIGHTS, wind=wind_fn, noise=0.02,
                             n_legs=(10, 16), wind_meas_noise=0.4, seed=SEED)
    print(f"Stage C — wind channel vs EKF ({N_FLIGHTS} flights, {len(df)} segments, "
          f"vortex peak≈{VORTEX['strength']:.1f} m/s)")

    # --- 1. uniform per-flight wind vs flight-mean EKF ------------------------
    fits = fit_per_log(df)
    uni_cos, mags = [], []
    for lid, sub in df.groupby("log_id"):
        w_hat = implied_wind(fits[lid])
        w_ekf = sub[["wind_x", "wind_y"]].to_numpy(float).mean(0)
        mags.append(float(np.linalg.norm(w_ekf)))
        uni_cos.append(cosine(w_hat, w_ekf))
    uni_cos, mags = np.array(uni_cos), np.array(mags)
    above = mags >= FLOOR
    print(f"  1. uniform wind: median cosine vs EKF = {np.median(uni_cos[above]):.3f} "
          f"on {above.sum()}/{len(uni_cos)} flights with ‖wind‖≥{FLOOR}")

    # --- 2. spatial field vs EKF at every midpoint ---------------------------
    g = np.linspace(-260.0, 260.0, 5)
    C1, C2 = np.meshgrid(g, g, indexing="ij")
    centers = np.stack([C1.ravel(), C2.ravel()], -1)
    pooled = fit_pooled(df, wind="spatial", centers=centers, width=190.0)
    w_field = implied_wind_field(pooled)
    field_cos = wind_field_cosine(w_field, df)
    print(f"  2. spatial field: global cosine vs EKF = {field_cos:.3f}  "
          f"[{'PASS' if field_cos >= 0.6 else 'FAIL'} ≥0.6]")

    passed = field_cos >= 0.6 and np.median(uni_cos[above]) >= 0.6
    print(f"\n  VERDICT: {'wind channel validated against independent EKF' if passed else 'wind too noisy at fleet scale (honest boundary, §5)'}")

    _figure(df, wind_fn, w_field, field_cos, uni_cos, mags, above)
    return 0 if passed else 1


def _figure(df, wind_fn, w_field, field_cos, uni_cos, mags, above):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # (a) recovered wind field vs the true/EKF field over the flown region.
    ax = axes[0]
    gx = np.linspace(df["mid_x"].min(), df["mid_x"].max(), 11)
    gy = np.linspace(df["mid_y"].min(), df["mid_y"].max(), 11)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.stack([GX.ravel(), GY.ravel()], -1)
    true = np.stack([wind_fn(p) for p in pts])
    rec = np.stack([w_field(p) for p in pts])
    ax.quiver(pts[:, 0], pts[:, 1], true[:, 0], true[:, 1], color="0.35",
              alpha=0.9, width=0.005, label="true / EKF field")
    ax.quiver(pts[:, 0], pts[:, 1], rec[:, 0], rec[:, 1], color="crimson",
              alpha=0.9, width=0.003, label="recovered from energy")
    ax.set_title(f"Wind field from energy asymmetry\n(global cosine {field_cos:.3f} vs EKF)")
    ax.set_xlabel("east [m]"), ax.set_ylabel("north [m]")
    ax.set_aspect("equal"), ax.legend(fontsize=8, loc="upper right")

    # (b) recovered vs EKF wind components at midpoints (the cross-check scatter).
    ax = axes[1]
    sub = df.dropna(subset=["wind_x", "wind_y"])
    mids = sub[["mid_x", "mid_y"]].to_numpy(float)
    pred = np.stack([w_field(m) for m in mids])
    ekf = sub[["wind_x", "wind_y"]].to_numpy(float)
    ax.scatter(ekf[:, 0], pred[:, 0], s=5, alpha=0.3, color="C0", label="east")
    ax.scatter(ekf[:, 1], pred[:, 1], s=5, alpha=0.3, color="crimson", label="north")
    lim = 1.1 * np.abs(ekf).max()
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
    ax.set_xlabel("EKF wind component [m/s]"), ax.set_ylabel("recovered [m/s]")
    ax.set_title("Recovered vs onboard EKF\n(the fit never saw the EKF)")
    ax.legend(fontsize=8)

    # (c) per-flight cosine vs wind magnitude: skill grows above the floor.
    ax = axes[2]
    ax.scatter(mags[above], uni_cos[above], s=30, color="C0", label=f"‖wind‖≥{FLOOR}")
    ax.scatter(mags[~above], uni_cos[~above], s=25, color="0.6",
               label=f"‖wind‖<{FLOOR} (below floor)")
    ax.axhline(0.6, color="crimson", ls=":", lw=1.5, label="cosine bar 0.6")
    ax.set_xlabel("flight-mean ‖EKF wind‖ [m/s]"), ax.set_ylabel("uniform-wind cosine")
    ax.set_ylim(-1.05, 1.05), ax.set_title("Recovery quality grows with wind strength")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Stage C — the wind channel: horizontal drift recovered from energy "
                 "asymmetries matches the onboard EKF", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "stage_c_wind.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    sys.exit(main())
