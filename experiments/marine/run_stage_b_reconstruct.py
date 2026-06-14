"""Stage B — reconstruct the surface current from passive drifters.

Drifters measure the flow directly (direct regression, not inverse optimal
control). We compare two reconstructions and report the honest identifiability
picture:

* **Geostrophic stream function** ``W = ∇^⊥ψ`` — divergence-free by construction,
  extrapolates smoothly *off-track*, but structurally cannot recover the divergent
  Ekman drift (its blind spot).
* **Kernel smoother** (Nadaraya–Watson) — captures the divergence near data but
  decays to zero away from the drifter tracks.

A coverage ablation sweeps the number of drifters to show where each method's
recovery saturates.

Run:  python -m experiments.marine.run_stage_b_reconstruct
Writes experiments/marine/visualizations/stage_b_reconstruct.png
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .drifters import fit_kernel, fit_streamfunction, simulate_drifters
from .evaluate import recovery_metrics
from .medium import OceanMedium

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)


def _grid(n=22):
    ax = np.linspace(1.0, 9.0, n)
    pts = jnp.array([[x, y, 0.0] for x in ax for y in ax])
    return ax, pts


def _split_on_off_track(grid_pts, obs_pos, radius=0.9):
    d = jnp.linalg.norm(grid_pts[:, None, :2] - obs_pos[None, :, :], axis=-1)
    dmin = jnp.min(d, axis=1)
    return dmin < radius


def main():
    # Static current (drifters reconstruct a snapshot); Ekman left divergent.
    medium = OceanMedium(meander_c=0.0, eddy_drift=0.0, ekman_omega=0.0)

    def true_current(x):
        return medium.physical_current(x, jnp.asarray(0.0))

    ax, grid_pts = _grid()
    print("Simulating drifters...")
    obs = simulate_drifters(medium, n_drifters=20, t_span=14.0, noise=0.01,
                            key=jax.random.PRNGKey(7))
    print(f"  {obs.positions.shape[0]} velocity observations")

    print("Fitting geostrophic stream function + kernel baseline...")
    psi_field = fit_streamfunction(obs, key=jax.random.PRNGKey(3), iters=1500)
    kernel = fit_kernel(obs, sigma=0.8)

    on = _split_on_off_track(grid_pts, obs.positions)
    for name, fn in [("stream-function", psi_field), ("kernel", kernel)]:
        m_all = recovery_metrics(fn, true_current, grid_pts)
        m_on = recovery_metrics(fn, true_current, grid_pts[on])
        m_off = recovery_metrics(fn, true_current, grid_pts[~on])
        print(f"  {name:16s} cosine all/on/off = "
              f"{m_all['cosine']:.3f} / {m_on['cosine']:.3f} / {m_off['cosine']:.3f}")

    # --- coverage ablation (averaged over drifter-placement seeds) ---
    print("Coverage ablation (3 seeds/point)...")
    counts = [4, 8, 16, 32]
    n_seeds = 3
    cov = {"stream-function": [], "kernel": []}
    for n in counts:
        psi_vals, ker_vals = [], []
        for seed in range(n_seeds):
            o = simulate_drifters(medium, n_drifters=n, t_span=14.0, noise=0.01,
                                  key=jax.random.PRNGKey(100 + 17 * n + seed))
            pf = fit_streamfunction(o, key=jax.random.PRNGKey(3), iters=1000)
            kf = fit_kernel(o, sigma=0.8)
            psi_vals.append(recovery_metrics(pf, true_current, grid_pts)["cosine"])
            ker_vals.append(recovery_metrics(kf, true_current, grid_pts)["cosine"])
        cov["stream-function"].append(float(np.mean(psi_vals)))
        cov["kernel"].append(float(np.mean(ker_vals)))
        print(f"  n={n:2d}: psi={cov['stream-function'][-1]:.3f}  kernel={cov['kernel'][-1]:.3f}")

    # --- figure ---
    fig = plt.figure(figsize=(14, 4.5))
    X, Y = np.meshgrid(ax, ax, indexing="ij")

    def quiver(axp, fn, title):
        W = np.array(jax.vmap(fn)(grid_pts)).reshape(len(ax), len(ax), 3)
        axp.quiver(X, Y, W[..., 0], W[..., 1], np.linalg.norm(W[..., :2], axis=-1),
                   cmap="viridis", scale=12)
        axp.plot(np.asarray(obs.positions)[:, 0], np.asarray(obs.positions)[:, 1],
                 ".", color="crimson", ms=2, alpha=0.5)
        axp.set_title(title), axp.set_aspect("equal")
        axp.set_xlabel("east"), axp.set_ylabel("north")

    quiver(fig.add_subplot(1, 4, 1), true_current, "True surface current\n(red: drifter pings)")
    quiver(fig.add_subplot(1, 4, 2), psi_field, "Geostrophic ψ reconstruction\n(divergence-free)")
    quiver(fig.add_subplot(1, 4, 3), kernel, "Kernel reconstruction\n(decays off-track)")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.plot(counts, cov["stream-function"], "o-", color="navy", label="stream-function")
    ax4.plot(counts, cov["kernel"], "s--", color="darkorange", label="kernel")
    ax4.set_xlabel("number of drifters"), ax4.set_ylabel("cosine similarity")
    ax4.set_title("Coverage ablation"), ax4.legend(), ax4.grid(alpha=0.3)

    fig.suptitle("Stage B — reconstructing the current from sparse drifters (direct regression)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "stage_b_reconstruct.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
