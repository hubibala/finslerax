"""Stage W-B — the odd-channel identifiability mask + parity-aware shrinkage.

The mask (`medium.odd_coverage`) claims: *where signed front-direction
diversity is low, the wind is not determined by the data* — an antiparallel
front pair separates ``B`` exactly (``s(n) - s(-n) = 2 B.n``), single-signed
fronts let the even channel absorb it. Two findings shaped the gates:

* The sign-blind structure tensor (``direction_coverage``) is EVEN-channel
  coverage and does not predict wind trust — kept only as a contrast.
* The error of a TV-regularised global fit does NOT track local
  identifiability either (measured: pooled spearman ≈ -0.1): TV interpolates
  wind across unidentified patches by design. That result is reported as an
  honest negative, not gated. The mask's actual claim is about **flat
  directions of the observation operator**, so that is what B1 tests.

Gates:

* **B1a curvature** — perturb the TRUTH fields inside one block along the
  mask's null direction (``B += d*u`` with its even compensation
  ``G -= 2d*(coef@u)``) and re-solve: the loss increase must track the mask
  (pooled Spearman ≥ 0.6 over seeds).
* **B1b confounding, blockwise** — in the lowest-coverage quartile, the
  compensated perturbation must cost ≤ 20% of the *uncompensated* one
  (the even channel absorbs the wind exactly where the mask says so).
* **B2 shrinkage** — on a designed scene (clustered ignitions -> strong
  coverage contrast; rotating truth wind -> TV extrapolation into the
  uncovered half is wrong), the coverage-weighted penalty is SELECTIVE:
  low-coverage wind shrunk ≥ 40% while high-coverage wind is retained ≥ 70%,
  at ≤ 15% total-error cost. Shrinkage does not improve accuracy — it
  converts unknowable wind into an explicit zero instead of a confident
  extrapolation ("shrunk, not hallucinated").

Run:  python -m experiments.wildfire.run_stage_b_mask
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .medium import (
    GridZermelo,
    dense_arrival_loss,
    front_normals,
    godunov_to_grid,
    grid_to_godunov,
    odd_null_data,
    solve_arrival_grid,
)
from .recover import fit_free_field
from .synthetic import make_scene, simulate_fires

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

SHAPE = (48, 48)
BLOCK = 8
SEEDS = [0, 1, 2, 3]
N_FIRES = 10
BURN_Q = 0.45
DELTA = 0.15  # wind-perturbation magnitude [px/h]

GATES = []


def gate(name, ok, detail):
    GATES.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def block_mean(a: np.ndarray, k: int = BLOCK) -> np.ndarray:
    H, W = a.shape
    return a[: H // k * k, : W // k * k].reshape(H // k, k, W // k, k).mean(axis=(1, 3))


def perturbed_loss(sc, fires, dG_params, dB) -> float:
    """Mean absolute dense loss of the perturbed truth over all fires."""
    G0, B0 = grid_to_godunov(sc.H_grid, sc.W_grid)
    H_p, W_p = godunov_to_grid(G0 + jnp.asarray(dG_params), B0 + jnp.asarray(dB))
    metric = GridZermelo(H_p, W_p)
    vals = []
    for f in fires:
        T = solve_arrival_grid(metric, jnp.asarray(f.ignition_rc), sc.shape)
        vals.append(float(dense_arrival_loss(
            T, jnp.asarray(f.T_obs), jnp.asarray(f.burned), 1.0)))
    return float(np.mean(vals))


def curvature_scan(sc, fires, cov, u, coef_u):
    """Loss increase per block: compensated (null) vs uncompensated wind bump."""
    Hb, Wb = cov.shape
    d_null = np.full((Hb, Wb), np.nan)
    d_raw = np.full((Hb, Wb), np.nan)
    for bi in range(Hb):
        for bj in range(Wb):
            if not np.any(u[bi, bj]):
                continue
            dB = np.zeros((2, *SHAPE))
            dG = np.zeros((3, *SHAPE))
            sl = (slice(bi * BLOCK, (bi + 1) * BLOCK),
                  slice(bj * BLOCK, (bj + 1) * BLOCK))
            dB[0][sl], dB[1][sl] = DELTA * u[bi, bj]
            d_raw[bi, bj] = perturbed_loss(sc, fires, dG, dB)
            for k in range(3):
                dG[k][sl] = -2.0 * DELTA * coef_u[bi, bj, k]
            d_null[bi, bj] = perturbed_loss(sc, fires, dG, dB)
    return d_null, d_raw


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("B1 — curvature along the mask's null directions (the flatness test)")
    cov_all, dnull_all, draw_all, fiterr_all = [], [], [], []
    first = None
    for seed in SEEDS:
        sc = make_scene(SHAPE, seed=seed, wind_rel_mag=0.4, anisotropy=True,
                        wind_uniform=False)
        fires = simulate_fires(sc, N_FIRES, seed=seed + 100, burn_quantile=BURN_Q)
        normals = [front_normals(f.T_obs, f.burned) for f in fires]
        cov, u, coef_u = odd_null_data(normals, block=BLOCK)
        d_null, d_raw = curvature_scan(sc, fires, cov, u, coef_u)

        # honest-negative companion: TV-fit wind error vs the mask
        res = fit_free_field(fires, SHAPE, lam_tv_H=1e-4, lam_tv_W=1e-4,
                             n_iter=1200, lr=0.1)
        dW = np.asarray(res.model.W_grid) - np.asarray(sc.W_grid)
        fit_err = block_mean(np.hypot(dW[0], dW[1]))

        ok = np.isfinite(d_null)
        rho = spearmanr(cov[ok], d_null[ok]).statistic
        print(f"  seed {seed}: spearman(mask, curvature) = {rho:+.3f}", flush=True)
        cov_all.append(cov[ok])
        dnull_all.append(d_null[ok])
        draw_all.append(d_raw[ok])
        fiterr_all.append(fit_err[ok])
        if first is None:
            first = (sc, fires, cov, d_null, res)

    cov_all = np.concatenate(cov_all)
    dnull_all = np.concatenate(dnull_all)
    draw_all = np.concatenate(draw_all)
    fiterr_all = np.concatenate(fiterr_all)

    rho = spearmanr(cov_all, dnull_all).statistic
    gate("B1a mask predicts loss curvature", rho >= 0.6,
         f"pooled spearman {rho:+.3f} over {len(cov_all)} blocks")
    lo = cov_all <= np.quantile(cov_all, 0.25)
    absorb = np.median(dnull_all[lo]) / max(np.median(draw_all[lo]), 1e-12)
    gate("B1b even channel absorbs wind where mask=0", absorb <= 0.20,
         f"compensated/uncompensated loss ratio {absorb:.2f} in low-coverage quartile")
    rho_fit = spearmanr(cov_all, fiterr_all).statistic
    print(f"  (reported, not gated: TV-fit wind error vs mask spearman "
          f"{rho_fit:+.3f} — TV interpolates across unidentified patches, "
          f"so fit error is NOT a local identifiability readout)")

    print("B2 — parity-aware shrinkage (designed scene: clustered ignitions)")
    import jax.numpy as jnp

    from .synthetic import Fire

    sc2 = make_scene(SHAPE, seed=2, wind_rel_mag=0.4, anisotropy=True,
                     wind_uniform=False)
    rng = np.random.default_rng(0)
    igns = np.stack([rng.uniform(4, 18, 8), rng.uniform(4, 18, 8)], axis=1)
    fires2 = []
    for ign in igns:
        T = np.asarray(solve_arrival_grid(sc2.metric, jnp.asarray(ign), SHAPE))
        fires2.append(Fire(ign, np.where(T < 1e4, T, np.inf), T < 1e4))
    normals2 = [front_normals(f.T_obs, f.burned) for f in fires2]
    cov2, _, _ = odd_null_data(normals2, block=BLOCK)
    cov_norm = np.clip(cov2 / max(np.quantile(cov2, 0.9), 1e-9), 0, 1)
    cov_px = np.kron(cov_norm, np.ones((BLOCK, BLOCK)))
    res_unreg = fit_free_field(fires2, SHAPE, lam_tv_H=1e-4, lam_tv_W=1e-4,
                               n_iter=1200, lr=0.1)
    res_reg = fit_free_field(fires2, SHAPE, coverage_weight=1e-5, coverage=cov_px,
                             lam_tv_H=1e-4, lam_tv_W=1e-4, n_iter=1200, lr=0.1)
    W_true = np.asarray(sc2.W_grid)
    lo1 = cov_norm.ravel() <= np.quantile(cov_norm, 0.25)
    hi1 = cov_norm.ravel() >= np.quantile(cov_norm, 0.75)

    def block_wind_norm(model):
        W = np.asarray(model.W_grid)
        return block_mean(np.hypot(W[0], W[1])).ravel()

    n_unreg, n_reg = block_wind_norm(res_unreg.model), block_wind_norm(res_reg.model)
    shrink = 1.0 - n_reg[lo1].mean() / max(n_unreg[lo1].mean(), 1e-9)
    retain = n_reg[hi1].mean() / max(n_unreg[hi1].mean(), 1e-9)
    gate("B2 low-coverage wind shrunk", shrink >= 0.4,
         f"|W| in lowest-coverage quartile reduced by {shrink:.0%}")
    gate("B2 high-coverage wind retained", retain >= 0.7,
         f"|W| in highest-coverage quartile kept at {retain:.0%}")

    def total_err(model):
        d = np.asarray(model.W_grid) - W_true
        return float(np.mean(np.hypot(d[0], d[1])))

    e_u, e_r = total_err(res_unreg.model), total_err(res_reg.model)
    gate("B2 no material degradation", e_r <= 1.15 * e_u,
         f"wind err {e_u:.3f} -> {e_r:.3f} (shrinkage is conservatism, "
         f"not accuracy)")

    # ---- figure ----
    _sc0, _fires0, cov0, d_null0, _res0 = first
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.4))
    im = axes[0].imshow(cov0, cmap="viridis")
    plt.colorbar(im, ax=axes[0], shrink=0.85)
    axes[0].set_title(f"odd coverage mask (block {BLOCK})\nseed {SEEDS[0]}, "
                      "pre-fit, observations only", fontsize=10)
    im = axes[1].imshow(d_null0, cmap="magma")
    plt.colorbar(im, ax=axes[1], shrink=0.85)
    axes[1].set_title("loss curvature along the null direction\n"
                      "(perturb truth wind + even compensation)", fontsize=10)
    ax = axes[2]
    ax.scatter(cov_all, draw_all, s=12, alpha=0.4, color="0.6",
               label="wind bump, uncompensated")
    ax.scatter(cov_all, dnull_all, s=12, alpha=0.6, color="C0",
               label="with even compensation")
    ax.set_yscale("log")
    ax.set_xlabel("block odd coverage")
    ax.set_ylabel("loss increase (log)")
    ax.set_title(f"the mask predicts flatness: spearman {rho:+.2f}\n"
                 f"(TV-fit error, by contrast: {rho_fit:+.2f})", fontsize=10)
    ax.legend(fontsize=8)
    x = np.arange(2)
    truth_blocks = block_mean(np.hypot(W_true[0], W_true[1])).ravel()
    axes[3].bar(x - 0.15, [n_unreg[lo1].mean(), n_unreg[hi1].mean()], width=0.3,
                label="free fit", color="crimson")
    axes[3].bar(x + 0.15, [n_reg[lo1].mean(), n_reg[hi1].mean()], width=0.3,
                label="coverage-weighted", color="C0")
    axes[3].plot(x, [truth_blocks[lo1].mean(), truth_blocks[hi1].mean()], "k_",
                 ms=26, mew=2, label="truth |W|")
    axes[3].set_xticks(x, ["low-coverage quartile", "high-coverage quartile"])
    axes[3].set_ylabel("mean recovered |W|")
    axes[3].set_title("clustered-ignition scene: shrunk where\nunidentifiable, "
                      "kept where identified", fontsize=10)
    axes[3].legend(fontsize=8)
    fig.suptitle("Stage W-B — the odd-channel ledger: signed front diversity is the "
                 "wind's identifiability currency (flat-direction test)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = OUT / "stage_b_mask.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")

    n_fail = sum(1 for _, ok, _ in GATES if not ok)
    (OUT / "stage_b_gates.json").write_text(
        json.dumps([{"gate": n, "pass": ok, "detail": d} for n, ok, d in GATES],
                   indent=2))
    print(f"\nStage W-B: {len(GATES) - n_fail}/{len(GATES)} gates passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
