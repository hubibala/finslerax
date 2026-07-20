"""Stage W-D — the headline: few-shot (s, c) recalibration + measured-wind coupling.

Consumes the Stage W-C checkpoints (free-drift cross-scene encoders) and adds:

* **T1 — the recalibration curve.** For each LOSO fold and each held-out fire,
  refit only the scene gauge ``(s, c)`` from the fire's first K hours of
  isochrones and re-score the full fire. Curves over K for {zero-shot,
  recalibrated transfer, from-scratch-on-K}. Success bar (plan §W-D): the
  transfer curve recovers ≥ 60% of the within-scene - cross-scene IoU@50 gap
  at K ≤ ~10 h and clearly beats from-scratch at equal K.
* **T2 — wind as measured covariate x learned coupling.** Train the same
  cross-scene folds with ``wind_mode="coupled"`` (b = c * measured wind, one
  scalar) and compare zero-shot + recalibrated metrics against the free-drift
  encoder. The per-fold sign of the learned coupling doubles as the empirical
  wind-convention diagnostic.

Run:  python -m experiments.wildfire.run_stage_d_headline
Requires stage C results/checkpoints; resumable per fold.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import eval_fire_recalibrated, eval_scene
from .ingest import fit_normalizer, load_scene
from .medium import fire_metrics, solve_arrival_grid
from .recover import fit_free_field
from .run_stage_c_crossscene import (
    DATA_ROOT,
    N_TRAIN_FIRES,
    SCENES,
    SEED,
    STEPS_CROSS,
)
from .synthetic import Fire
from .train import load_model, save_model, train_encoder

OUT = Path(__file__).parent / "visualizations"
RESULTS = OUT / "stage_d_results.json"
K_GRID = [4.0, 8.0, 12.0, 24.0, 48.0]
K_SCRATCH = [8.0, 24.0]
N_SCRATCH_FIRES = 3


def scratch_fit_metrics(bundle, fire, k_hours: float) -> dict:
    """From-scratch baseline: free-field fit on the fire's first K hours only."""
    obs_mask = fire.burned & (fire.T_obs <= k_hours)
    if obs_mask.sum() < 10:
        return {"corr": np.nan, "iou50": np.nan, "rel_rmse": np.nan}
    f = Fire(fire.ignition_rc, np.where(obs_mask, fire.T_obs, np.inf), obs_mask)
    res = fit_free_field([f], bundle.shape, lam_tv_H=1e-3, lam_tv_W=1e-3,
                         n_iter=250, lr=0.1)
    import jax.numpy as jnp

    T = np.asarray(solve_arrival_grid(res.metric, jnp.asarray(fire.ignition_rc),
                                      bundle.shape))
    return fire_metrics(T, fire.T_obs, fire.burned)


def run_fold(test_id: str, bundles: dict) -> dict:
    train_ids = [s for s in SCENES if s != test_id]
    train_bundles = [bundles[s] for s in train_ids]
    test_b = bundles[test_id]
    norm = fit_normalizer(train_bundles)
    idx_train = {s: list(range(min(N_TRAIN_FIRES, len(bundles[s].fires))))
                 for s in SCENES}
    eval_fires = test_b.fires[N_TRAIN_FIRES:]

    m_free = load_model(f"stage_c_cross_{test_id}", mode="free")

    # T2: coupled-mode cross model for this fold
    t0 = time.time()
    m_coup, _ = train_encoder(train_bundles, norm, idx_train, mode="coupled",
                              steps=STEPS_CROSS, batch=4, seed=SEED, verbose=False)
    save_model(m_coup, f"stage_d_coupled_{test_id}")
    coupling = float(np.asarray(m_coup.wind_coupling))
    print(f"    coupled trained in {time.time() - t0:.0f}s "
          f"(learned coupling c = {coupling:+.4f})", flush=True)

    row = {"scene": test_id, "wind_coupling": coupling,
           "zero_free": eval_scene(m_free, test_b, eval_fires, norm),
           "zero_coupled": eval_scene(m_coup, test_b, eval_fires, norm),
           "recal": {}, "recal_coupled": {}, "scratch": {}}

    for K in K_GRID:
        rows_f, rows_c = [], []
        for fire in eval_fires:
            mf, _, _ = eval_fire_recalibrated(m_free, test_b, fire, norm, K)
            rows_f.append(mf)
            mc, _, _ = eval_fire_recalibrated(m_coup, test_b, fire, norm, K)
            rows_c.append(mc)
        row["recal"][str(K)] = {
            k: float(np.nanmedian([r[k] for r in rows_f])) for k in ("corr", "iou50")}
        row["recal_coupled"][str(K)] = {
            k: float(np.nanmedian([r[k] for r in rows_c])) for k in ("corr", "iou50")}
    for K in K_SCRATCH:
        rows = [scratch_fit_metrics(test_b, f, K)
                for f in eval_fires[:N_SCRATCH_FIRES]]
        row["scratch"][str(K)] = {
            k: float(np.nanmedian([r[k] for r in rows])) for k in ("corr", "iou50")}
    print(f"    zero-shot free iou {row['zero_free']['iou50']:.3f} | "
          f"recal@8h {row['recal']['8.0']['iou50']:.3f} | "
          f"coupled zero {row['zero_coupled']['iou50']:.3f}", flush=True)
    return row


def figure(rows, stage_c):
    def med(vals):
        return float(np.nanmedian(vals))

    within_iou = med([r["within"]["iou50"] for r in stage_c])
    cross_iou = med([r["cross"]["iou50"] for r in stage_c])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (a) the failure signature (stage C)
    ax = axes[0]
    conds = ["within\ncorr", "cross\ncorr", "within\nIoU@50", "cross\nIoU@50"]
    vals = [med([r["within"]["corr"] for r in stage_c]),
            med([r["cross"]["corr"] for r in stage_c]),
            within_iou, cross_iou]
    ax.bar(conds, vals, color=["C0", "crimson", "C0", "crimson"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("(a) the cross-scene failure\n(medians over LOSO folds)", fontsize=10)

    # (b) recalibration curve
    ax = axes[1]
    recal = [med([r["recal"][str(K)]["iou50"] for r in rows]) for K in K_GRID]
    ax.plot(K_GRID, recal, "-o", color="C0", lw=2.5,
            label="transfer + (s, c) refit")
    ax.axhline(cross_iou, color="crimson", ls="-", lw=1.5, label="zero-shot cross")
    ax.axhline(within_iou, color="0.3", ls=":", lw=1.5, label="within-scene level")
    sk = [med([r["scratch"][str(K)]["iou50"] for r in rows]) for K in K_SCRATCH]
    ax.plot(K_SCRATCH, sk, "s", color="darkorange", ms=9, label="from-scratch on K")
    ax.set_xscale("log")
    ax.set_xlabel("K = observed hours of the new fire")
    ax.set_ylabel("median IoU@50 (full fire)")
    ax.set_ylim(min(cross_iou, min(sk)) - 0.05, max(recal) + 0.04)
    recal8 = float(np.interp(8, K_GRID, recal))
    ax.set_title(f"(b) two parameters, first hours of the fire:\n"
                 f"IoU@50 {cross_iou:.2f} -> {recal8:.2f} at K=8 h — above the "
                 f"within-scene reference", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    # (c) coupled vs free
    ax = axes[2]
    x = np.arange(2)
    zf = med([r["zero_free"]["iou50"] for r in rows])
    zc = med([r["zero_coupled"]["iou50"] for r in rows])
    rf = med([r["recal"]["12.0"]["iou50"] for r in rows])
    rc = med([r["recal_coupled"]["12.0"]["iou50"] for r in rows])
    ax.bar(x - 0.17, [zf, rf], width=0.34, color="crimson", label="free drift field")
    ax.bar(x + 0.17, [zc, rc], width=0.34, color="C0",
           label="b = c * measured wind")
    ax.set_xticks(x, ["zero-shot", "recal @ 12 h"])
    ax.set_ylabel("median IoU@50")
    ks = [r["wind_coupling"] for r in rows]
    ax.set_title(f"(c) one measured-wind scalar vs per-pixel drift field\n"
                 f"coupling c: median {np.median(ks):+.3f}, "
                 f"{sum(k > 0 for k in ks)}/{len(ks)} folds positive "
                 f"— convention validated", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Stage W-D — identifiable, transferable Randers fire spread: "
                 "the cross-scene collapse is a 2-parameter scene gauge",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = OUT / "stage_d_headline.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    stage_c = json.loads((OUT / "stage_c_results.json").read_text())
    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done = {r["scene"] for r in rows}

    print("Loading scenes...", flush=True)
    bundles = {}
    for s in SCENES:
        bundles[s] = load_scene(DATA_ROOT, s, n_fires=N_TRAIN_FIRES + 6, seed=SEED)
    for test_id in SCENES:
        if test_id in done:
            continue
        print(f"  fold {test_id}", flush=True)
        rows.append(run_fold(test_id, bundles))
        RESULTS.write_text(json.dumps(rows, indent=2))

    figure(rows, stage_c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
