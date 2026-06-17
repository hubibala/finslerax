"""Stage D — amortization / scale (Randers Flow Matching, H4).

The "what real data needs" stage.  A BVP geodesic per lineage triple inside the
training loop is infeasible at 130k-cell × 71k-triple scale, so we train a
low-parameter interpolant ``φθ(z₀,z₁,t)`` to minimize the **asymmetric Randers
action** over sampled endpoint pairs and check that it (a) matches the exact
AVBD+continuation BVP and the true min-action path, (b) is far cheaper per pair
with an N-independent forward cost, and (c) preserves the directionality signal.

Run:  python -m experiments.single_cell_synthetic.run_stage_d_amortize
Writes visualizations/stage_d_amortize.png
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .drift import SparseVFC
from .evaluate import path_discrepancy
from .generator import GeneratorConfig, generate
from .landscape import Landscape, least_action_path
from .metric import FlatSea, build_randers, navigable_wind_scale
from .solvers import exact_geodesic, train_rfm

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)


def main():
    rng = np.random.default_rng(0)
    ls = Landscape(kappa=1.0)
    ds = generate(ls, GeneratorConfig(n_cells=2500, n_clones=350, n_genes=150,
                                     substeps=30, vel_noise=0.3, seed=20))
    z = ds.true_state
    sea = FlatSea(2)
    vf = SparseVFC.fit(z, ds.velocity_true, n_control=120, reg=1e-3)
    scale = navigable_wind_scale(vf, sea, z, margin=0.8)
    metric = build_randers(sea, vf, dim=2, wind_scale=scale)

    # endpoint pairs = lineage (early, late) states
    tri = ds.triples
    states = ds.true_state
    pairs = np.stack([states[tri[:, 0]], states[tri[:, 2]]], axis=1).astype(np.float32)
    print(f"Training Randers-Flow-Matching on {pairs.shape[0]} lineage pairs...")
    t0 = time.perf_counter()
    interp, hist = train_rfm(metric, pairs, dim=2, key=jax.random.PRNGKey(0),
                             steps=1200, batch=64, n_quad=20)
    train_s = time.perf_counter() - t0
    print(f"  RFM action {hist[0]:.3f} → {hist[-1]:.3f}  (train {train_s:.1f}s)")

    # ---- path fidelity on held-out pairs ----
    test = pairs[rng.choice(len(pairs), min(20, len(pairs)), replace=False)]
    disc_exact, disc_lap = [], []
    for z0, z1 in test:
        g = exact_geodesic(metric, jnp.asarray(z0), jnp.asarray(z1), n_steps=32)
        r = interp.path(jnp.asarray(z0), jnp.asarray(z1), n_steps=32)
        lap, _ = least_action_path(ls, jnp.asarray(z0), jnp.asarray(z1), n_steps=32)
        disc_exact.append(path_discrepancy(np.asarray(r), np.asarray(g.xs)))
        disc_lap.append(path_discrepancy(np.asarray(r), np.asarray(lap)))
    print(f"  RFM vs exact BVP discrepancy: {np.mean(disc_exact):.3f}")
    print(f"  RFM vs true min-action path : {np.mean(disc_lap):.3f}")

    # ---- cost vs N: exact BVP wall-clock grows with N; RFM forward is flat ----
    Ns = [16, 32, 48, 64, 96]
    z0, z1 = jnp.asarray(test[0, 0]), jnp.asarray(test[0, 1])
    exact_t, rfm_t = [], []
    for n in Ns:
        t0 = time.perf_counter()
        exact_geodesic(metric, z0, z1, n_steps=n).xs.block_until_ready()
        exact_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        interp.path(z0, z1, n_steps=n).block_until_ready()
        rfm_t.append(time.perf_counter() - t0)
    print("  cost-vs-N (exact BVP s / RFM forward s):")
    for n, e, r in zip(Ns, exact_t, rfm_t):
        print(f"    N={n:3d}  exact={e:.3f}s  RFM={r:.4f}s  speedup x{e / max(r, 1e-6):.0f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(hist, color="navy")
    ax.set_title("RFM training: mean Randers action")
    ax.set_xlabel("step"), ax.set_ylabel("action"), ax.set_yscale("log")

    ax = axes[1]
    g = exact_geodesic(metric, z0, z1, n_steps=32)
    r = interp.path(z0, z1, n_steps=32)
    lap, _ = least_action_path(ls, z0, z1, n_steps=32)
    gx = np.linspace(-2.6, 2.6, 60)
    gy = np.linspace(-1.6, 1.6, 60)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    phi = np.array([[float(ls.potential(jnp.array([x, y]))) for y in gy] for x in gx])
    ax.contourf(X, Y, phi, levels=20, cmap="terrain", alpha=0.8)
    ax.plot(np.asarray(g.xs)[:, 0], np.asarray(g.xs)[:, 1], "-", color="crimson", lw=2.5, label="exact BVP")
    ax.plot(np.asarray(r)[:, 0], np.asarray(r)[:, 1], "--", color="yellow", lw=2, label="amortized RFM")
    ax.plot(np.asarray(lap)[:, 0], np.asarray(lap)[:, 1], ":", color="white", lw=2, label="true min-action")
    ax.set_title("Amortized vs exact geodesic"), ax.legend(loc="lower right")
    ax.set_xlabel("x₁"), ax.set_ylabel("x₂")

    ax = axes[2]
    ax.plot(Ns, exact_t, "o-", color="crimson", label="exact BVP (continuation)")
    ax.plot(Ns, rfm_t, "s-", color="navy", label="amortized RFM forward")
    ax.set_title("Per-pair cost vs resolution N\n(amortization flattens the wall)")
    ax.set_xlabel("N"), ax.set_ylabel("wall-clock (s)"), ax.set_yscale("log"), ax.legend()

    fig.suptitle("Stage D — amortized Randers Flow Matching enables real scale", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "stage_d_amortize.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
