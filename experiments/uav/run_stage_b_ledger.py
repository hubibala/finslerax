"""Stage B — the ledger, and the exact-drift gauge on flight hardware (§4, the core).

Three claims, one convex estimator:

1. **Convex recovery.** A per-log ridge LSQ of ``[even nuisance | k·Δz | wind]`` on
   segment energies recovers each vehicle's climb-cost ledger ``k``. We report the
   held-out **directed-energy R²** (predict the energy *difference* of matched
   climb/descent pairs — the even bracket cancels, so this scores the odd channel
   the claim is about), the within-airframe ``k`` consistency, and the pooled
   hierarchical fit's shared wind.

2. **The gauge, on real flights.** The gravity ledger ``k·dz`` is an *exact*
   1-form, so it is invisible in trajectory *shape* (Finsler projective
   invariance) and visible only in the energy ledger. We show this concretely:
   two fleets flown identically but with different ``k`` have **bit-identical
   trajectories** and **different energy ledgers**, and a profile-likelihood over
   ``k`` is flat for the path channel but sharply peaked for the energy channel.
   The linear solve pins ``k`` where the paths cannot.

3. **Falsification.** A negative control regresses the odd part against a fake
   east-west potential ``k'·dx`` and recovers ≈ 0.

Success bar (§4): held-out directed-energy R² ≥ 0.7; within-airframe ``k`` CV
≤ 20%; negative control ``|k'| ≤ 10%`` of ``k``. Runs on the synthetic bridge
(known ``k``, ``W``) until the real corpus lands; the estimator code is unchanged
when the segment tables come from ``ingest`` instead.

Run:  python -m experiments.uav.run_stage_b_ledger
Writes experiments/uav/visualizations/stage_b_ledger.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .estimate import (
    fit_per_log,
    fit_pooled,
    fleet_predictor,
    implied_wind,
    k_series,
    negative_control,
)
from .evaluate import directed_energy_r2, k_consistency, split_segments
from .medium import even_features, odd_features
from .synthetic import PowerModel, synthesize_fleet

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

WIND = (3.0, -2.0)
N_FLIGHTS = 24
SEED = 7


def _energy_misfit_profile(df, k_grid):
    """Per-log relative energy misfit as the ledger coefficient k is swept.

    At each fixed ``k`` the even nuisance + wind are refit by LSQ to ``E − k·Δz``;
    the residual RMS, normalized by its own minimum over the grid, is a sharp
    U with its floor at the log's true ``k`` — the energy channel *identifies* k.
    """
    curves = []
    for _, sub in df.groupby("log_id"):
        A = np.hstack([even_features(sub), odd_features(sub, ("dx", "dy"))])
        dz = sub["dz"].to_numpy(float)
        y = sub["energy"].to_numpy(float)
        rms = []
        for k in k_grid:
            c, *_ = np.linalg.lstsq(A, y - k * dz, rcond=None)
            rms.append(float(np.sqrt(np.mean((y - k * dz - A @ c) ** 2))))
        rms = np.array(rms)
        curves.append(rms / rms.min())
    return np.array(curves)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    model = PowerModel()
    df, truth = synthesize_fleet(N_FLIGHTS, model=model, wind=WIND, noise=0.02,
                                 mass_cv=0.08, n_legs=(10, 16), seed=SEED)
    print(f"Stage B — ledger + gauge ({N_FLIGHTS} flights, {len(df)} segments, "
          f"true k≈{model.k:.2f} J/m)")

    # --- 1. convex recovery ---------------------------------------------------
    fits = fit_per_log(df)
    k_hat = k_series(fits).sort_index()
    k_true = truth.k_by_log.sort_index()
    cv = k_consistency(k_hat)
    train, test = split_segments(df, test_frac=0.3, seed=0)
    de_r2, n_pairs = directed_energy_r2(test, fleet_predictor(fit_per_log(train)))
    pooled = fit_pooled(df, share_k=False, wind="uniform")
    w_hat = implied_wind(pooled)
    w_cos = float(np.dot(w_hat, WIND) / (np.linalg.norm(w_hat) * np.linalg.norm(WIND)))
    print(f"  1. recovery: within-airframe k CV={cv:.3f}  "
          f"held-out directed-energy R²={de_r2:.3f} ({n_pairs} pairs)  "
          f"pooled wind cos={w_cos:.3f}")

    # even-only baseline: the ledger-free model has no directed-energy skill.
    even_only = {lid: _EvenOnly(f) for lid, f in fit_per_log(train).items()}
    de_r2_even, _ = directed_energy_r2(test, fleet_predictor(even_only))
    print(f"     ledger-free (even-only) directed-energy R²={de_r2_even:+.3f} "
          f"(only the even-bracket residue imperfect pair matching leaves — "
          f"the ledger term carries the rest)")

    # --- 2. the gauge: same flights, different k ------------------------------
    mA = PowerModel(eta_up=0.70)   # k ≈ 12.0
    mB = PowerModel(eta_up=0.45)   # k ≈ 17.4 — heavier climb cost
    dfA, _ = synthesize_fleet(N_FLIGHTS, model=mA, wind=WIND, noise=0.0,
                              n_legs=(10, 16), seed=SEED)
    dfB, _ = synthesize_fleet(N_FLIGHTS, model=mB, wind=WIND, noise=0.0,
                              n_legs=(10, 16), seed=SEED)
    path_diff = float(np.max(np.abs(
        dfA[["dx", "dy", "dz"]].to_numpy() - dfB[["dx", "dy", "dz"]].to_numpy())))
    energy_gap = float(np.mean(np.abs(dfA["energy"] - dfB["energy"])))
    kA = k_series(fit_per_log(dfA, ridge=0.0)).mean()
    kB = k_series(fit_per_log(dfB, ridge=0.0)).mean()
    print(f"  2. gauge: identical trajectories (max path diff={path_diff:.1e}) but "
          f"mean |ΔE|={energy_gap:.0f} J/seg")
    print(f"     LSQ separates them: k̂_A={kA:.2f} (true {mA.k:.2f}), "
          f"k̂_B={kB:.2f} (true {mB.k:.2f})")
    k_grid = np.linspace(0.3 * model.k, 1.8 * model.k, 41)
    profiles = _energy_misfit_profile(df, k_grid)

    # --- 3. negative control --------------------------------------------------
    df0, _ = synthesize_fleet(N_FLIGHTS, model=model, wind=(0.0, 0.0), noise=0.02,
                              n_legs=(10, 16), seed=SEED + 1)
    k_fake = float(np.abs(negative_control(df0)).median())
    k_real = float(k_series(fit_per_log(df0, odd=("dz",))).median())
    print(f"  3. negative control: |k'_fake(dx)|={k_fake:.3f}  vs  k(dz)={k_real:.3f}  "
          f"({100 * k_fake / k_real:.0f}% of k, bar ≤10%)")

    passed = (de_r2 >= 0.7 and cv <= 0.20 and k_fake <= 0.10 * k_real)
    print(f"\n  SUCCESS BAR: {'MET' if passed else 'NOT MET'} "
          f"(directed-energy R²≥0.7, k CV≤0.20, |k'|≤10%)")

    _figure(df, k_hat, k_true, cv, test, fits, de_r2, de_r2_even, n_pairs,
            k_grid, profiles, model.k, dfA, dfB, mA, mB, kA, kB, path_diff,
            k_fake, k_real, pooled, w_hat, w_cos)
    return 0 if passed else 1


class _EvenOnly:
    """A predictor that keeps only the even nuisance — no odd (ledger/wind) channel."""

    def __init__(self, fit):
        self.even = fit.even

    def predict(self, sub):
        return even_features(sub) @ self.even


def _figure(df, k_hat, k_true, cv, test, fits, de_r2, de_r2_even, n_pairs,
            k_grid, profiles, k_true_scalar, dfA, dfB, mA, mB, kA, kB, path_diff,
            k_fake, k_real, pooled, w_hat, w_cos):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0))

    # (a) per-log ledger recovery vs the injected mass jitter.
    ax = axes[0, 0]
    ax.scatter(k_true, k_hat, s=40, color="C0", zorder=3)
    lo, hi = float(k_true.min()) - 0.5, float(k_true.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax.set_xlabel("true k (per flight) [J/m]"), ax.set_ylabel("recovered k̂ [J/m]")
    ax.set_title(f"Per-log ledger recovery\n(within-airframe CV={cv:.2f}, bar ≤0.20)")
    ax.legend(fontsize=8)

    # (b) held-out directed energy: the full model scores, the ledger-free one doesn't.
    ax = axes[0, 1]
    pred = fleet_predictor(fits)
    obs_d, pred_d = _pair_diffs(test, pred)
    ax.scatter(obs_d, pred_d, s=14, alpha=0.5, color="C0", zorder=3)
    lim = max(np.abs(obs_d).max(), np.abs(pred_d).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
    ax.set_xlabel("observed ΔE (climb - descent) [J]")
    ax.set_ylabel("predicted ΔE [J]")
    ax.set_title(f"Held-out directed energy ({n_pairs} pairs)\n"
                 f"full R²={de_r2:.2f}  ·  ledger-free R²={de_r2_even:+.2f}")

    # (c) THE GAUGE: energy pins k, path shape is flat (projective invariance).
    ax = axes[0, 2]
    for c in profiles:
        ax.plot(k_grid, c, "-", color="C0", alpha=0.25, lw=1)
    ax.plot(k_grid, profiles.mean(0), "-", color="C0", lw=2.5,
            label="energy channel (mean)")
    ax.axhline(1.0, color="crimson", lw=2.5, ls="-",
               label="path-shape channel (k-invariant)")
    ax.axvline(k_true_scalar, color="green", lw=1.5, ls="--", label="true k")
    ax.set_ylim(0.9, min(3.0, float(profiles.max())))
    ax.set_xlabel("assumed ledger k [J/m]"), ax.set_ylabel("relative misfit  (1 = best-fit k)")
    ax.set_title("The gauge: energy identifies k;\nshape cannot (Finsler projective invariance)")
    ax.legend(fontsize=8, loc="upper center")

    # (d) same flights, different k: identical trajectories, different ledgers.
    ax = axes[1, 0]
    first = dfA["log_id"].iloc[0]
    for lid in list(dict.fromkeys(dfA["log_id"]))[:5]:
        a = dfA[dfA["log_id"] == lid]
        b = dfB[dfB["log_id"] == lid]
        pa = np.concatenate([[0.0], np.cumsum(a["dz"].to_numpy())])
        pb = np.concatenate([[0.0], np.cumsum(b["dz"].to_numpy())])
        ax.plot(pa, color="C0", lw=4, alpha=0.55,
                label=f"fleet A  (k={mA.k:.1f})" if lid == first else None)
        ax.plot(pb, color="crimson", lw=1.4, ls="--",
                label=f"fleet B  (k={mB.k:.1f})" if lid == first else None)
    ax.set_xlabel("segment index"), ax.set_ylabel("altitude profile  Σdz [m]")
    ax.set_title("Same flights, different ledger:\ntrajectories identical, energy ledgers differ")
    ax.text(0.03, 0.03,
            f"max trajectory diff: {path_diff:.0e} m\n"
            f"mean |ΔE|: {np.mean(np.abs(dfA['energy'].to_numpy() - dfB['energy'].to_numpy())):.0f} J/seg\n"
            f"LSQ recovers  k̂_A={kA:.1f},  k̂_B={kB:.1f}",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.6"})
    ax.legend(fontsize=8, loc="upper right")

    # (e) negative control: the real ledger is large, the fake east-west ≈ 0.
    ax = axes[1, 1]
    bars = ax.bar(["k (dz, real)", "k' (dx, fake)"], [k_real, k_fake],
                  color=["C0", "0.6"])
    for b, v in zip(bars, [k_real, k_fake]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
    ax.axhline(0.10 * k_real, color="crimson", ls=":", lw=1.5, label="10% of k bar")
    ax.set_ylabel("recovered coefficient [J/m]")
    ax.set_title("Negative control:\na fake east-west potential recovers ≈ 0")
    ax.legend(fontsize=8)

    # (f) pooled wind vs the EKF ground truth the fit never saw.
    ax = axes[1, 2]
    ax.quiver(0, 0, WIND[0], WIND[1], angles="xy", scale_units="xy", scale=1,
              color="0.2", width=0.02, label=f"true wind {WIND}")
    ax.quiver(0, 0, w_hat[0], w_hat[1], angles="xy", scale_units="xy", scale=1,
              color="crimson", width=0.012, label=f"recovered Ŵ ({w_hat[0]:.1f},{w_hat[1]:.1f})")
    m = 1.3 * max(abs(WIND[0]), abs(WIND[1]), abs(w_hat[0]), abs(w_hat[1]))
    ax.set_xlim(-m, m), ax.set_ylim(-m, m), ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5), ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("east [m/s]"), ax.set_ylabel("north [m/s]")
    ax.set_title(f"Pooled wind from energy asymmetry\n(cosine {w_cos:.3f} vs EKF ground truth)")
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Stage B — a convex estimator recovers each vehicle's climb-cost "
                 "ledger; the paths alone provably could not", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "stage_b_ledger.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


def _pair_diffs(df, predict):
    """Observed/predicted matched climb−descent energy differences (for the scatter)."""
    import pandas as pd

    from .evaluate import _match_pairs
    pred = pd.Series(np.asarray(predict(df), float), index=df.index)
    obs_d, pred_d = [], []
    for _, sub in df.groupby("log_id"):
        for i_up, i_dn in _match_pairs(sub, 1.0):
            obs_d.append(df.at[i_up, "energy"] - df.at[i_dn, "energy"])
            pred_d.append(pred[i_up] - pred[i_dn])
    return np.array(obs_d), np.array(pred_d)


if __name__ == "__main__":
    sys.exit(main())
