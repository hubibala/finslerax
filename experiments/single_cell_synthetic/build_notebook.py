"""Assemble (and optionally execute) the single-cell walkthrough notebook.

Usage:
    python -m experiments.single_cell_synthetic.build_notebook        # write unexecuted
    python -m experiments.single_cell_synthetic.build_notebook --run  # write + execute

Produces ``experiments/single_cell_synthetic/single_cell_synthetic.ipynb``. The
notebook reuses the experiment package (no duplicated science) and develops one
idea carefully: a *known* Waddington geometry with a tunable irreversible flux
``κ`` lets us prove the learned asymmetric Randers drift recovers something a
symmetric metric cannot — before touching real data.
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "single_cell_synthetic.ipynb"

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def co(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ===========================================================================
md(r"""
# Synthetic Single-Cell Waddington Geometry — a ground-truth precursor

This notebook walks through the `experiments/single_cell_synthetic` study. It
develops one idea carefully: **differentiation is irreversible**, the irreversible
part is a *non-conservative flux*, and HAM's learned **asymmetric Randers** metric
can recover it — something a symmetric metric provably cannot.

Because the geometry is *synthetic*, every quantity (the drift, its exact Hodge
split, the fate of every cell, the true minimum-action path) is known, so we can
*score* recovery instead of guessing. The knob `κ` interpolates from a purely
conservative landscape (`κ=0`, where symmetric ties) to a strongly
non-conservative one (`κ>0`, where only the asymmetric drift keeps up).
""")

co(r"""
import jax, jax.numpy as jnp, numpy as np
import matplotlib.pyplot as plt
from experiments.single_cell_synthetic import (
    Landscape, GeneratorConfig, generate, build_true_metric,
    SparseVFC, helmholtz_hodge_rbf, exact_geodesic, least_action_path,
)
jax.config.update("jax_platform_name", "cpu")
""")

md(r"""
## 1. The landscape and its exact Hodge split

`φ` is a double-well-along-a-channel Waddington potential (a pitchfork at the
bifurcation `x_b`); the flux `κ·∇^⊥ψ` is a divergence-free circulation at the
saddle. The total drift `f = -∇φ + κ·∇^⊥ψ` splits *exactly* into a reversible
(gradient) and an irreversible (solenoidal) part.
""")

co(r"""
ls = Landscape(kappa=1.0)
gx = np.linspace(-2.6, 2.6, 40); gy = np.linspace(-1.6, 1.6, 40)
X, Y = np.meshgrid(gx, gy, indexing="ij")
phi = np.array([[float(ls.potential(jnp.array([x,y]))) for y in gy] for x in gx])
F = np.array([[np.asarray(ls.drift(jnp.array([x,y]))) for y in gy] for x in gx])
fig, ax = plt.subplots(figsize=(6,4))
ax.contourf(X, Y, phi, levels=24, cmap="terrain", alpha=0.85)
ax.quiver(X[::2,::2], Y[::2,::2], F[::2,::2,0], F[::2,::2,1], color="k", alpha=0.6)
ax.set_title("Waddington potential + total drift (κ=1)")
ax.set_xlabel("x₁ (developmental)"); ax.set_ylabel("x₂ (fate)"); fig.tight_layout()
""")

md(r"""
## 2. A destructively-sampled, gene-space dataset

The SDE is integrated from a progenitor cloud; cells are observed at discrete days
(never twice), decoded to negative-binomial gene counts with dropout, and tagged
with clonal barcodes. RNA velocity is the latent drift pushed to PCA space, then
corrupted with a realistic noise model.
""")

co(r"""
ds = generate(ls, GeneratorConfig(n_cells=2000, n_clones=300, n_genes=150, substeps=30, seed=0))
print("cells", ds.n_cells, "genes", ds.n_genes, "zero-fraction", round((ds.X_counts==0).mean(),3))
print("lineage triples", ds.triples.shape[0])
fig, axes = plt.subplots(1,2, figsize=(11,4))
sc = axes[0].scatter(ds.true_state[:,0], ds.true_state[:,1], c=ds.time_point, cmap="viridis", s=6)
axes[0].set_title("true latent, coloured by day"); fig.colorbar(sc, ax=axes[0])
axes[1].scatter(ds.X_pca[:,0], ds.X_pca[:,1], c=ds.fate_label, cmap="coolwarm", s=6)
axes[1].set_title("PCA embedding, coloured by fate"); fig.tight_layout()
""")

md(r"""
## 3. Recover the drift (sparseVFC) and its flux (Hodge)

The Dynamo-style sparseVFC RKHS fit reconstructs a smooth drift from the noisy
velocity (its EM step rejects the sign-flipped arrows); a matrix-valued
Helmholtz-Hodge decomposition splits it into gradient + solenoidal parts. The
recovered solenoidal part is compared to the known truth.
""")

co(r"""
from experiments.single_cell_synthetic.evaluate import global_cosine, mean_cosine
z = ds.true_state
vf = SparseVFC.fit(z, ds.velocity_pca, n_control=120, reg=1e-3)
hf = helmholtz_hodge_rbf(vf, z, n_control=120, reg=1e-2)
pred = np.asarray(jax.vmap(vf)(jnp.asarray(z)))
sol = np.asarray(jax.vmap(hf.sol_part)(jnp.asarray(z)))
print("full-drift recovery cosine:", round(mean_cosine(pred, ds.true_drift),3))
print("flux (solenoidal) recovery cosine:", round(global_cosine(sol, ds.true_sol),3))
""")

md(r"""
## 4. Asymmetry pays — sweep κ

Flux recovery and the directionality ratio on the transverse commitment route both
grow with `κ`, while the symmetric baseline stays at 1. Re-run
`run_stage_c_asymmetry` for the full multi-baseline figure.
""")

co(r"""
from experiments.single_cell_synthetic.evaluate import directionality_score
TA, TB = jnp.array([0.8,-0.8]), jnp.array([0.8,0.8])
pts = jnp.asarray(np.random.default_rng(0).uniform([-2.5,-1.5],[2.5,1.5],size=(400,2)),jnp.float32)
ks, dirs = [0.0,0.5,1.0,2.0], []
for k in ks:
    m = build_true_metric(Landscape(kappa=k), pts, margin=0.8)
    dirs.append(directionality_score(m, TA, TB, n_steps=20))
plt.plot(ks, dirs, "o-"); plt.axhline(1, color="gray", ls=":")
plt.xlabel("κ"); plt.ylabel("cost(against)/cost(with)"); plt.title("directionality grows with κ"); plt.show()
print(dict(zip(ks, [round(d,3) for d in dirs])))
""")

md(r"""
## 5. LAP ↔ Randers, and amortization

The Randers geodesic recovers the true Onsager-Machlup minimum-action path (H3),
and a trained Randers-Flow-Matching interpolant matches the exact BVP at a fraction
of the cost (H4) — see `run_stage_a_solver` and `run_stage_d_amortize`.
""")


def build():
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "ham-venv", "language": "python", "name": "ham-venv"}
    if "--run" in sys.argv:
        from nbclient import NotebookClient
        print("Executing notebook (this runs the full pipeline)...")
        NotebookClient(nb, timeout=1200, kernel_name="ham-venv",
                       resources={"metadata": {"path": str(REPO)}}).execute()
    with open(OUT, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
