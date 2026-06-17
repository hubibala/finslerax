"""Stage C — asymmetry pays (the headline, H1).

As a function of the flux fraction ``κ``, compare the learned **non-conservative
Randers drift** against the symmetric-Riemannian control (``β=0``), the
**potential-only** negative control (Hodge-predicted to fail), and a
CellRank-style Markov fate baseline.  The claim is that every asymmetry signal —
flux recovery, directionality, and fate accuracy — is **monotone in κ** and the
margin over the symmetric/potential-only controls *grows with κ*, exactly as the
Hodge-decomposition argument predicts (PLAN §1, Stage C).

Run:  python -m experiments.single_cell_synthetic.run_stage_c_asymmetry
Writes visualizations/stage_c_asymmetry.png
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .baselines import cellrank_fate, potential_only_randers, symmetric_riemannian
from .drift import SparseVFC, helmholtz_hodge_rbf
from .evaluate import (
    directionality_score,
    fate_accuracy,
    fate_by_geodesic,
    global_cosine,
)
from .generator import GeneratorConfig, generate
from .landscape import Landscape
from .metric import FlatSea, build_randers, navigable_wind_scale

OUT = Path(__file__).parent / "visualizations"
OUT.mkdir(exist_ok=True)

KAPPAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def _fit_drift(ds):
    """sparseVFC drift + Hodge split in the true 2-D latent.

    Uses the oracle-frame ``velocity_true`` so the recovered field and its Hodge
    split live in the same frame as ``true_drift``/``true_sol`` (Stage C isolates
    the *asymmetry* question; the embedding-frame confound is Stage B's subject).
    """
    z = ds.true_state
    vf = SparseVFC.fit(z, ds.velocity_true, n_control=120, reg=1e-3)
    hf = helmholtz_hodge_rbf(vf, z, n_control=120, reg=1e-2)
    return vf, hf


def main():
    rng = np.random.default_rng(0)
    rows = []
    for k in KAPPAS:
        ls = Landscape(kappa=k)
        ds = generate(ls, GeneratorConfig(n_cells=2500, n_clones=350, n_genes=150,
                                          substeps=30, vel_noise=0.3, vel_flip_frac=0.1, seed=10))
        vf, hf = _fit_drift(ds)
        z = ds.true_state
        pts = jnp.asarray(z, jnp.float32)
        sea = FlatSea(2)

        # flux recovery cosine (learned solenoidal part vs truth)
        sol_pred = np.asarray(jax.vmap(hf.sol_part)(pts))
        flux_cos = global_cosine(sol_pred, ds.true_sol)

        # metrics: full Randers / symmetric / potential-only (flat sea, true latent)
        scale = navigable_wind_scale(vf, sea, z, margin=0.8)
        m_full = build_randers(sea, vf, dim=2, wind_scale=scale)
        m_sym = symmetric_riemannian(sea, dim=2)
        m_pot = potential_only_randers(sea, hf, dim=2, points=z, margin=0.8)

        # directionality on the transverse commitment route (constant x₁): no
        # gradient-tilt asymmetry there, so it is exactly 1 at κ=0 and only the
        # non-conservative flux can break it — isolating what the gradient
        # one-form (potential-only) and the symmetric metric *cannot* represent.
        term = ls.terminal_states()
        TA = jnp.array([0.8, -0.8], jnp.float32)
        TB = jnp.array([0.8, 0.8], jnp.float32)
        dir_full = directionality_score(m_full, TA, TB, n_steps=20)
        dir_pot = directionality_score(m_pot, TA, TB, n_steps=20)
        dir_sym = directionality_score(m_sym, TA, TB, n_steps=20)

        # fate accuracy on late, committed cells (subsampled). Only the relative
        # geodesic-cost ordering matters, so use a light solver here.
        light = {"schedule": (8, 16), "avbd_iters": 120, "gn_iters": 8}
        late = (ds.time_point == ds.time_point.max()) & (np.abs(z[:, 1]) > 0.3)
        idx = np.nonzero(late)[0]
        if idx.size > 16:
            idx = rng.choice(idx, 16, replace=False)
        src, truef = z[idx], ds.fate_label[idx]
        pf, _ = fate_by_geodesic(m_full, src, np.asarray(term), n_steps=16, **light)
        ps, _ = fate_by_geodesic(m_sym, src, np.asarray(term), n_steps=16, **light)
        acc_full, acc_sym = fate_accuracy(pf, truef), fate_accuracy(ps, truef)

        # CellRank baseline
        m0 = late & (z[:, 1] < 0)
        m1 = late & (z[:, 1] > 0)
        ab = cellrank_fate(ds.X_pca, ds.velocity_pca, [m0, m1], k=25)
        acc_cr = fate_accuracy(ab[idx].argmax(1), truef)

        rows.append({"kappa": k, "flux_cos": flux_cos, "dir_full": dir_full, "dir_pot": dir_pot,
                         "dir_sym": dir_sym, "acc_full": acc_full, "acc_sym": acc_sym, "acc_cr": acc_cr})
        print(f"kappa={k:.2f}  flux_cos={flux_cos:.3f}  dir[full/pot/sym]="
              f"{dir_full:.2f}/{dir_pot:.2f}/{dir_sym:.2f}  "
              f"fate[full/sym/CR]={acc_full:.2f}/{acc_sym:.2f}/{acc_cr:.2f}")

    ks = [r["kappa"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(ks, [r["flux_cos"] for r in rows], "o-", color="crimson")
    ax.set_title("Flux recovery grows with κ\n(learned solenoidal part vs truth)")
    ax.set_xlabel("κ (flux fraction)"), ax.set_ylabel("flux cosine")

    ax = axes[1]
    ax.plot(ks, [r["dir_full"] for r in rows], "o-", label="full Randers")
    ax.plot(ks, [r["dir_pot"] for r in rows], "s--", label="potential-only")
    ax.plot(ks, [r["dir_sym"] for r in rows], "^:", label="symmetric (β=0)")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_title("Directionality: only the full drift\nbreaks symmetry as κ grows")
    ax.set_xlabel("κ"), ax.set_ylabel("cost(rev)/cost(fwd)"), ax.legend()

    ax = axes[2]
    ax.plot(ks, [r["acc_full"] for r in rows], "o-", label="full Randers")
    ax.plot(ks, [r["acc_sym"] for r in rows], "^:", label="symmetric (β=0)")
    ax.plot(ks, [r["acc_cr"] for r in rows], "d--", label="CellRank-Markov")
    ax.set_title("Terminal-fate accuracy vs κ")
    ax.set_xlabel("κ"), ax.set_ylabel("fate accuracy"), ax.legend()

    fig.suptitle("Stage C — asymmetry pays: the headline as a function of flux κ", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / "stage_c_asymmetry.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    return rows


if __name__ == "__main__":
    main()
