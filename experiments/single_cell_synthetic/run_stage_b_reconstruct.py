"""Stage B — reconstruction & the identifiability / collapse frontier (H2).

From destructive snapshots + noisy velocity + clones, run the recovery pipeline
and map how its quality depends on the things the *real* run cannot control:

* **velocity SNR** (Gaussian noise + fraction of sign-flipped arrows) — how much
  corruption the sparseVFC drift estimator tolerates;
* **lineage sampling** (clones / snapshots) — the cross-time supervision frontier;
* **latent dimension d** — the *collapse curve*: the pullback base ``H = JᵀJ``
  degenerates (its conditioning blows up / smallest eigenvalue → 0) beyond some
  ``d*``, while the density-conformal base ``H = c(z)I`` stays well-conditioned —
  the documented Finsler collapse (Pouplin TMLR'23), measured against truth.

Run:  python -m experiments.single_cell_synthetic.run_stage_b_reconstruct
Writes visualizations/stage_b_reconstruct.png
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .drift import SparseVFC
from .embedding import _lognorm_np, train_autoencoder
from .evaluate import mean_cosine
from .generator import GeneratorConfig, generate
from .landscape import Landscape
from .metric import conformal_sea, pullback_sea

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)


def _recover_cosine(ds, **fit_kw):
    """sparseVFC recovery cosine vs the true drift (oracle-frame velocity).

    Uses ``velocity_true`` so the SNR/sampling axes are isolated from the
    embedding-frame rotation (the latter is the collapse study below).
    """
    z = ds.true_state  # recover in the true 2-D latent (isolates the SNR axis)
    vf = SparseVFC.fit(z, ds.velocity_true, **fit_kw)
    pred = np.asarray(jax.vmap(vf)(jnp.asarray(z, jnp.float32)))
    return mean_cosine(pred, ds.true_drift)


def _metric_conditioning(sea_fn, points):
    """Mean log10 condition number and min eigenvalue of H over points."""
    def stats(z):
        H = sea_fn(z)
        ev = jnp.linalg.eigvalsh(0.5 * (H + H.T))
        ev = jnp.clip(ev, 1e-12, None)
        return jnp.array([jnp.log10(ev[-1] / ev[0]), ev[0]])
    s = jax.vmap(stats)(jnp.asarray(points, jnp.float32))
    return float(jnp.mean(s[:, 0])), float(jnp.mean(s[:, 1]))


def main():
    base_ls = Landscape(kappa=1.0)

    # ---- (1) velocity SNR frontier ----
    # Sparse cells + few control points so corruption actually bites (with dense
    # sampling sparseVFC recovers almost perfectly at any noise — itself a finding).
    print("Velocity SNR frontier (d=2):")
    noises = [0.0, 1.0, 2.0, 4.0]
    flips = [0.0, 0.15, 0.3, 0.45]
    snr_noise, snr_flip = [], []
    for nz in noises:
        ds = generate(base_ls, GeneratorConfig(n_cells=400, n_clones=120, n_genes=120,
                                               substeps=25, vel_noise=nz, vel_flip_frac=0.1, seed=2))
        c = _recover_cosine(ds, n_control=40, reg=1e-2)
        snr_noise.append(c)
        print(f"  vel_noise={nz:.2f}  recovery cosine={c:.3f}")
    for fl in flips:
        ds = generate(base_ls, GeneratorConfig(n_cells=400, n_clones=120, n_genes=120,
                                               substeps=25, vel_noise=1.0, vel_flip_frac=fl, seed=3))
        c = _recover_cosine(ds, n_control=40, reg=1e-2)
        snr_flip.append(c)
        print(f"  flip_frac={fl:.2f}  recovery cosine={c:.3f}")

    # ---- (2) lineage sampling frontier ----
    print("Lineage sampling frontier:")
    cell_counts = [100, 250, 600, 1500]
    samp = []
    for ncell in cell_counts:
        ds = generate(base_ls, GeneratorConfig(n_cells=ncell, n_clones=max(20, ncell // 8),
                                               n_genes=120, substeps=25, vel_noise=1.5, seed=4))
        c = _recover_cosine(ds, n_control=min(60, ncell // 4), reg=1e-2)
        samp.append(c)
        print(f"  n_cells={ncell:4d}  recovery cosine={c:.3f}  (triples={ds.triples.shape[0]})")

    # ---- (3) collapse curve: pullback vs conformal base across latent dim d ----
    # The data is intrinsically 2-D (a 2-D landscape decoded to gene space). For
    # a working latent of dim d, we train an AE z(d) -> gene-space and put the
    # pullback sea H = JᵀJ (d×d) on that latent. Because the decoder Jacobian
    # stays rank ~2, the extra d-2 latent directions are near-null, so JᵀJ gains
    # ever-smaller eigenvalues as d grows — its condition number explodes (the
    # documented Finsler collapse). The density-conformal base H = c(z)I is a
    # scaled identity and stays perfectly conditioned at any d.
    print("Collapse frontier (base-metric conditioning vs latent dim d):")
    ds = generate(base_ls, GeneratorConfig(n_cells=2000, n_clones=300, n_genes=200,
                                           substeps=25, vel_noise=0.3, seed=5))
    Y = _lognorm_np(ds.X_counts, 4000.0)  # (n, D_gene) gene-space target
    dims = [2, 4, 8, 16, 32]
    pull_cond, conf_cond = [], []
    for d in dims:
        enc, dec, _ = train_autoencoder(Y, d=d, key=jax.random.PRNGKey(d), steps=800)
        zc = np.asarray(jax.vmap(enc)(jnp.asarray(Y[:300], jnp.float32)))
        pc, _ = _metric_conditioning(pullback_sea(dec, dim=d), zc)
        cc, _ = _metric_conditioning(conformal_sea(zc, dim=d, sigma=0.8), zc)
        pull_cond.append(pc)
        conf_cond.append(cc)
        print(f"  d={d:2d}  pullback log10(cond)={pc:.2f}  conformal log10(cond)={cc:.2f}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.plot(noises, snr_noise, "o-", label="vs Gaussian noise")
    ax.plot(flips, snr_flip, "s--", label="vs flipped-arrow fraction")
    ax.set_title("Velocity-SNR frontier\n(sparseVFC drift recovery)")
    ax.set_xlabel("noise level"), ax.set_ylabel("recovery cosine"), ax.legend()

    ax = axes[1]
    ax.plot(cell_counts, samp, "o-", color="purple")
    ax.set_title("Sampling frontier\n(recovery vs cell count)")
    ax.set_xlabel("cells"), ax.set_ylabel("recovery cosine")

    ax = axes[2]
    ax.plot(dims, pull_cond, "o-", color="crimson", label="pullback H=JᵀJ")
    ax.plot(dims, conf_cond, "s-", color="navy", label="conformal H=c(z)I")
    ax.set_title("Collapse curve\n(base-metric conditioning vs d)")
    ax.set_xlabel("latent dim d"), ax.set_ylabel("mean log₁₀ cond(H)"), ax.legend()
    ax.set_xscale("log", base=2)

    fig.suptitle("Stage B — reconstruction & the identifiability/collapse frontier", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "stage_b_reconstruct.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
