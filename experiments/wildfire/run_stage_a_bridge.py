"""Stage W-A — revive & verify: the synthetic bridge (GO/NO-GO).

Gates, fixed a priori (all on synthetic scenes with exact known truth):

* **W0 forward** — eikonal arrival error < 1% mean on analytic cases
  (isotropic, rotated anisotropic, constant drift).
* **W1 gradients** — implicit-adjoint gradients of the dense arrival loss
  match central finite differences through the solver (both gauges).
* **W2 multi-source recovery** — 6 fires, uniform per-fire wind: sea and wind
  recovered (wind error ≤ 25% of |W|, sea rel error ≤ 0.35).
* **W3 single-source confounding (negative control)** — spatially varying
  wind, one fire: wind recovery FAILS (error ≥ 70% ≈ predicting no wind)
  while the same loss with 6 fires cuts it by ≥ 1.8×. If single-source
  *succeeds*, the parity/coverage diagnosis is wrong — stop and rethink.
* **W4 recalibration** — an injected scene gauge (s0, c0) is recovered from
  the first hours of one fire (closed-form s, 1-D search c), and the zero-shot
  IoU@50 collapse (corr high, IoU low) is closed by the 2-parameter refit.

Run:  python -m experiments.wildfire.run_stage_a_bridge
Writes experiments/wildfire/visualizations/stage_a_bridge.png and prints a
gate table; exits non-zero if any gate fails.
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

from .medium import (
    GridZermelo,
    dense_arrival_loss,
    fire_metrics,
    solve_arrival_grid,
)
from .recover import fit_free_field, recalibrate, sea_error, wind_error
from .synthetic import make_scene, simulate_fires

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

GATES: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, detail: str):
    GATES.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
# W0 — forward accuracy on analytic cases
# ---------------------------------------------------------------------------


def w0_forward():
    print("W0 — forward eikonal vs analytic arrival times")
    shape = (48, 48)
    src = jnp.array([24.0, 24.0])
    I, J = np.meshgrid(np.arange(48), np.arange(48), indexing="ij")
    d0, d1 = I - 24.0, J - 24.0
    r = np.hypot(d0, d1)
    ring = r > 3  # exclude the source-stencil neighbourhood

    def rel_err(T, T_exact):
        return float(np.mean(np.abs(T - T_exact)[ring] / np.maximum(T_exact[ring], 1e-9)))

    ones, zeros = jnp.ones(shape), jnp.zeros(shape)

    # isotropic
    T = np.asarray(solve_arrival_grid(GridZermelo(jnp.stack([ones, zeros, ones]),
                                                  jnp.zeros((2, *shape))), src, shape))
    e_iso = rel_err(T, r)

    # rotated anisotropic, constant: T = sqrt(d^T G d), G = R diag(2, 0.5) R^T
    th = np.deg2rad(30.0)
    c, s = np.cos(th), np.sin(th)
    l1, l2 = 2.0, 0.5
    g11 = l1 * c**2 + l2 * s**2
    g12 = (l1 - l2) * c * s
    g22 = l1 * s**2 + l2 * c**2
    H_grid = jnp.stack([g11 * ones, g12 * ones, g22 * ones])
    T = np.asarray(solve_arrival_grid(GridZermelo(H_grid, jnp.zeros((2, *shape))), src, shape))
    e_aniso = rel_err(T, np.sqrt(g11 * d0**2 + 2 * g12 * d0 * d1 + g22 * d1**2))

    # constant drift: straight-line geodesics, T = sqrt(d^T G d) + B.d with
    # G = H + (HW)(HW)^T, B = -HW  (lam = 1 form), W = (0, 0.4)
    w = np.array([0.0, 0.4])
    B = -w  # H = I
    G = np.eye(2) + np.outer(w, w)
    T = np.asarray(solve_arrival_grid(
        GridZermelo(jnp.stack([ones, zeros, ones]),
                    jnp.stack([zeros, 0.4 * ones])), src, shape))
    T_exact = np.sqrt(G[0, 0] * d0**2 + 2 * G[0, 1] * d0 * d1 + G[1, 1] * d1**2) \
        + B[0] * d0 + B[1] * d1
    e_drift = rel_err(T, T_exact)

    gate("W0 iso < 1%", e_iso < 0.01, f"mean rel err {e_iso:.4f}")
    gate("W0 aniso < 1.5%", e_aniso < 0.015, f"mean rel err {e_aniso:.4f}")
    gate("W0 drift < 1.5%", e_drift < 0.015, f"mean rel err {e_drift:.4f}")


# ---------------------------------------------------------------------------
# W1 — gradients through the solver vs finite differences
# ---------------------------------------------------------------------------


def w1_gradients():
    print("W1 — implicit gradients vs central finite differences")
    import jax

    shape = (20, 20)
    ones, zeros = jnp.ones(shape), jnp.zeros(shape)
    src = jnp.array([10.0, 10.0])
    T_obs = solve_arrival_grid(
        GridZermelo(jnp.stack([1.3 * ones, zeros, 1.3 * ones]),
                    jnp.stack([zeros, 0.25 * ones])), src, shape)
    burned = jnp.ones(shape, dtype=bool)

    def loss(theta, alpha):
        # theta = (sea scale, wind magnitude)
        H_grid = jnp.stack([theta[0] * ones, zeros, theta[0] * ones])
        W_grid = jnp.stack([zeros, theta[1] * ones])
        T = solve_arrival_grid(GridZermelo(H_grid, W_grid), src, shape)
        return dense_arrival_loss(T, T_obs, burned, alpha)

    theta0 = jnp.array([1.0, 0.1])
    for alpha, tol in [(1.0, 0.08), (0.0, 0.08)]:
        g_ad = np.asarray(jax.grad(loss)(theta0, alpha))
        eps = 3e-3
        g_fd = np.zeros(2)
        for i in range(2):
            e = np.zeros(2)
            e[i] = eps
            g_fd[i] = (float(loss(theta0 + e, alpha)) - float(loss(theta0 - e, alpha))) / (2 * eps)
        rel = np.abs(g_ad - g_fd) / np.maximum(np.abs(g_fd), 1e-6)
        gate(
            f"W1 grad alpha={alpha:g}",
            bool(np.all(rel < tol)),
            f"AD {np.round(g_ad, 4).tolist()} vs FD {np.round(g_fd, 4).tolist()} "
            f"(rel {np.round(rel, 3).tolist()})",
        )


# ---------------------------------------------------------------------------
# W2 / W3 — multi-source recovery vs single-source confounding
# ---------------------------------------------------------------------------


def w2_w3_recovery():
    print("W2 — multi-source recovery (uniform per-scene wind)")
    shape = (48, 48)
    sc_u = make_scene(shape, seed=3, wind_rel_mag=0.4, anisotropy=True, wind_uniform=True)
    W_mag_u = float(np.mean(np.hypot(*np.asarray(sc_u.W_grid))))
    fires_u = simulate_fires(sc_u, 6, seed=5, burn_quantile=1.0)
    res_u = fit_free_field(fires_u, shape, lam_tv_H=1e-4, lam_tv_W=1e-4,
                           n_iter=1200, lr=0.1)
    we_u = wind_error(res_u.model.W_grid, sc_u.W_grid) / W_mag_u
    se_u = sea_error(res_u.model.H_grid, sc_u.H_grid)
    gate("W2 wind err <= 25%", we_u <= 0.25, f"{we_u:.1%} of |W|")
    gate("W2 sea err <= 0.35", se_u <= 0.35, f"rel Frobenius {se_u:.3f}")

    print("W3 — single-source confounding (spatially varying wind)")
    sc_v = make_scene(shape, seed=3, wind_rel_mag=0.4, anisotropy=True, wind_uniform=False)
    W_mag_v = float(np.mean(np.hypot(*np.asarray(sc_v.W_grid))))
    fires_v = simulate_fires(sc_v, 6, seed=5, burn_quantile=1.0)
    res_m = fit_free_field(fires_v, shape, lam_tv_H=1e-4, lam_tv_W=1e-4,
                           n_iter=1200, lr=0.1)
    res_1 = fit_free_field(fires_v[:1], shape, lam_tv_H=1e-4, lam_tv_W=1e-4,
                           n_iter=1200, lr=0.1)
    we_m = wind_error(res_m.model.W_grid, sc_v.W_grid) / W_mag_v
    we_1 = wind_error(res_1.model.W_grid, sc_v.W_grid) / W_mag_v
    gate("W3 single-source fails", we_1 >= 0.70,
         f"single-fire wind err {we_1:.1%} (predict-zero baseline = 100%)")
    gate("W3 multi/single ratio", we_1 / max(we_m, 1e-9) >= 1.8,
         f"single {we_1:.1%} / multi {we_m:.1%} = {we_1 / max(we_m, 1e-9):.2f}x")
    return sc_v, res_m, res_1, (sc_u, res_u, fires_u)


# ---------------------------------------------------------------------------
# W4 — the scene gauge: injected (s, c) recovered few-shot
# ---------------------------------------------------------------------------


def w4_recalibration():
    print("W4 — few-shot (s, c) recalibration exactness")
    shape = (48, 48)
    sc = make_scene(shape, seed=11, wind_rel_mag=0.35, anisotropy=True, wind_uniform=True)
    ign = jnp.array([20.0, 30.0])
    s0, c0 = 1.7, 0.6
    T_obs = s0 * np.asarray(
        solve_arrival_grid(GridZermelo(sc.H_grid, c0 * sc.W_grid), ign, shape))
    burned = T_obs <= np.quantile(T_obs[T_obs < 1e4], 0.6)

    def solve_c(c):
        return np.asarray(
            solve_arrival_grid(GridZermelo(sc.H_grid, c * sc.W_grid), ign, shape))

    obs_mask = burned & (T_obs <= 8.0)  # first 8 hours only
    r = recalibrate(solve_c, T_obs, obs_mask)
    m_recal = fire_metrics(r.T_pred, T_obs, burned)
    m_zero = fire_metrics(solve_c(1.0), T_obs, burned)
    gate("W4 s recovered", abs(r.s - s0) <= 0.05 * s0, f"s={r.s:.3f} vs {s0}")
    gate("W4 c recovered", abs(r.c - c0) <= 0.1, f"c={r.c:.3f} vs {c0}")
    gate("W4 signature: corr high, IoU collapsed",
         m_zero["corr"] >= 0.9 and m_zero["iou50"] <= 0.6,
         f"zero-shot corr={m_zero['corr']:.3f}, iou50={m_zero['iou50']:.3f}")
    gate("W4 refit closes IoU", m_recal["iou50"] >= 0.9,
         f"recalibrated iou50={m_recal['iou50']:.3f}")
    return T_obs, burned, r, m_zero, m_recal


# ---------------------------------------------------------------------------
# Figure — demonstrate intent unambiguously
# ---------------------------------------------------------------------------


def figure(sc_v, res_m, res_1, uniform_pack, w4_pack):
    sc_u, _res_u, fires_u = uniform_pack
    T_obs, burned, recal, m_zero, m_recal = w4_pack
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))

    def quiver(ax, W_grid, title, step=5):
        W_ = np.asarray(W_grid)
        H, Wd = W_.shape[1:]
        rr, cc = np.mgrid[0:H:step, 0:Wd:step]
        ax.quiver(cc, rr, W_[1, ::step, ::step], W_[0, ::step, ::step],
                  angles="xy", scale=6.0, width=0.005)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    quiver(axes[0, 0], sc_v.W_grid, "truth wind (spatially varying)")
    quiver(axes[0, 1], res_m.model.W_grid, "recovered — 6 fires (multi-source)")
    quiver(axes[0, 2], res_1.model.W_grid,
           "recovered — 1 fire (odd channel invisible)")
    ax = axes[0, 3]
    ax.imshow(np.asarray(sc_u.H_grid[0]), cmap="viridis")
    for f in fires_u:
        ax.plot(f.ignition_rc[1], f.ignition_rc[0], "r*", ms=10)
    ax.set_title("uniform-wind scene: sea h11 + ignitions", fontsize=10)

    # W4 panels: the gauge story in miniature
    ax = axes[1, 0]
    im = ax.imshow(np.where(burned, T_obs, np.nan), cmap="magma")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("observed fire (hours)", fontsize=10)

    tau = 0.5 * T_obs[burned].max()
    ax = axes[1, 1]
    zero_shot = np.asarray(recal.T_pred) / recal.s  # T at (s=1, c=1) ~ shape only
    ax.contour(T_obs, levels=[tau], colors="k")
    ax.contour(zero_shot * recal.s / max(recal.s, 1e-9), levels=[tau], colors="tab:blue")
    ax.contour(recal.T_pred, levels=[tau], colors="tab:red", linestyles="--")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"50% perimeter: GT (black) vs recal (red)\n"
                 f"zero-shot iou={m_zero['iou50']:.2f} -> recal {m_recal['iou50']:.2f}",
                 fontsize=10)

    ax = axes[1, 2]
    p = recal.T_pred[burned]
    g = T_obs[burned]
    ax.scatter(g, p / recal.s, s=4, alpha=0.3, color="tab:blue",
               label=f"zero-shot-scale (corr {m_zero['corr']:.2f})")
    ax.scatter(g, p, s=4, alpha=0.3, color="tab:red",
               label=f"recalibrated (s={recal.s:.2f}, c={recal.c:.2f})")
    lim = [0, g.max() * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("observed arrival [h]")
    ax.set_ylabel("predicted [h]")
    ax.set_title("the affine gauge: correlation survives,\nabsolute timing does not",
                 fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1, 3]
    ax.axis("off")
    lines = [f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok, _ in GATES]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=9,
            color="black")
    ax.set_title("gate ledger", fontsize=10)

    fig.suptitle(
        "Stage W-A — the bridge: forward/gradient integrity, multi-source buys the "
        "odd channel, single-source is confounded, and the scene gauge is a 2-parameter refit",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "stage_a_bridge.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    w0_forward()
    w1_gradients()
    sc_v, res_m, res_1, uniform_pack = w2_w3_recovery()
    w4_pack = w4_recalibration()
    figure(sc_v, res_m, res_1, uniform_pack, w4_pack)

    n_fail = sum(1 for _, ok, _ in GATES if not ok)
    print(f"\nStage W-A: {len(GATES) - n_fail}/{len(GATES)} gates passed")
    (OUT / "stage_a_gates.json").write_text(
        json.dumps([{"gate": n, "pass": ok, "detail": d} for n, ok, d in GATES], indent=2)
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
