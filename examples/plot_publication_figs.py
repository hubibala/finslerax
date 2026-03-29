"""
plot_publication_figs.py
=========================
Generates a 3-panel publication-quality figure:

  Panel A — Latent metric determinant heatmap (log det G) over 2D PCA of Z,
             overlaid with fate-colored cell scatter.

  Panel B — Wind quiver field over the same PCA plane, colored by wind
             magnitude, overlaid with pseudotime-colored cell scatter.

  Panel C — Exemplar observed vs. counterfactual trajectories with
             per-segment Randers cost annotations.

All panels share the same 2D PCA projection of the 8-dimensional latent space.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax
import jax.numpy as jnp
import equinox as eqx
import anndata

sys.path.insert(0, os.path.dirname(__file__))
from ham.bio.data import BioDataset
from ham.bio.vae import GeometricVAE
from ham.models.learned import DataDrivenPullbackRanders
from ham.geometry.metric import FinslerMetric
from weinreb_vae import build_diagnostic_vae, encode_all, TARGET_FATES
from experiment_h2_directional import attach_datadriven_randers_metric, load_phase1_vae


# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_logdet_grid(vae, Z_all: np.ndarray, pca2: PCA, n_grid: int = 40):
    """
    Evaluate log det J^T J on a 2D grid in the PCA(Z) plane.
    Returns (GX, GY, logdet) all shape (n_grid, n_grid).
    """
    z2d = pca2.transform(Z_all)
    x_range = np.percentile(z2d[:, 0], [2, 98])
    y_range = np.percentile(z2d[:, 1], [2, 98])
    gx = np.linspace(*x_range, n_grid)
    gy = np.linspace(*y_range, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    pts2d = np.stack([GX.ravel(), GY.ravel()], axis=1)
    pts_full = pca2.inverse_transform(pts2d).astype(np.float32)
    latent_dim = Z_all.shape[1]
    pts_full = pts_full[:, :latent_dim]

    @eqx.filter_jit
    def logdet_at(v_mod, z):
        J    = jax.jacfwd(v_mod.decode)(z)
        G    = jnp.dot(J.T, J) + 1e-6 * jnp.eye(latent_dim)
        sign, ld = jnp.linalg.slogdet(G)
        return jnp.where(sign > 0, ld, jnp.array(-20.0))

    log_dets = np.array([float(logdet_at(vae, jnp.array(z))) for z in pts_full])

    return GX, GY, log_dets.reshape(n_grid, n_grid)


def compute_wind_grid(vae, pca2: PCA, Z_all: np.ndarray, n_grid: int = 20):
    """
    Evaluate the wind vector W(z) on a 2D grid in the PCA(Z) plane.
    Projects the D-dim wind back to 2D for display.
    Returns (GX, GY, Wx2d, Wy2d, magnitudes).
    """
    z2d = pca2.transform(Z_all)
    x_range = np.percentile(z2d[:, 0], [3, 97])
    y_range = np.percentile(z2d[:, 1], [3, 97])
    gx = np.linspace(*x_range, n_grid)
    gy = np.linspace(*y_range, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    pts2d = np.stack([GX.ravel(), GY.ravel()], axis=1)
    pts_full = pca2.inverse_transform(pts2d).astype(np.float32)[:, :Z_all.shape[1]]

    # Use eqx.filter_vmap and pass the metric module explicitly
    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0))
    def get_winds(m_mod, pts):

        return m_mod.w_net(pts)

    winds_full = np.array(get_winds(vae.metric, jnp.array(pts_full)))  # (n_grid^2, D)


    # Project wind into 2D PCA plane
    components = pca2.components_[:2]  # (2, D)
    winds_2d = winds_full @ components.T   # (n_grid^2, 2)
    magnitudes = np.linalg.norm(winds_full, axis=1)

    Wx2d = winds_2d[:, 0].reshape(n_grid, n_grid)
    Wy2d = winds_2d[:, 1].reshape(n_grid, n_grid)
    mags = magnitudes.reshape(n_grid, n_grid)
    return GX, GY, Wx2d, Wy2d, mags


def arc_length_segment(metric, z_start: jax.Array, z_end: jax.Array, steps: int = 20) -> float:
    """Finsler arc length of the straight line from z_start to z_end."""
    t = jnp.linspace(0, 1, steps)[:, None]
    gamma = z_start + t * (z_end - z_start)
    return metric.arc_length(gamma)


def find_exemplar_triple(Z_all, labels_np, fate_names, test_triples,
                         target_fate: str, fallback_fate: str) -> tuple:
    """Return (z2, z4, z6_obs, z6_counter) for a single exemplar trajectory."""
    if target_fate not in fate_names or fallback_fate not in fate_names:
        return None
    fidx_target = fate_names.index(target_fate)
    fidx_fallback = fate_names.index(fallback_fate)
    cf_centroid = jnp.array(Z_all[labels_np == fidx_fallback].mean(axis=0))

    for triple in test_triples:
        i2, i4, i6 = triple
        if labels_np[i6] == fidx_target:
            z2 = jnp.array(Z_all[i2])
            z4 = jnp.array(Z_all[i4])
            z6 = jnp.array(Z_all[i6])
            return z2, z4, z6, cf_centroid
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    LATENT_DIM   = 8
    OUTPUT       = "weinreb_publication_figure.png"

    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    if not os.path.exists(CHECKPOINT):
        print("Checkpoint not found.")
        return

    # ── Load raw data ────────────────────────────────────────────────────────
    print("Loading data ...")
    adata = anndata.read_h5ad(PREPROCESSED)
    X_pca = np.array(adata.obsm["X_pca"], dtype=np.float32)
    V_pca = np.array(adata.obsm["velocity_pca"], dtype=np.float32)
    ct_series  = adata.obs["Cell type annotation"].astype("category")
    fate_names = list(ct_series.cat.categories)
    labels_np  = ct_series.cat.codes.values.astype(np.int32)
    n_types    = len(fate_names)
    time_point = adata.obs["time_point"].values.astype(float)

    scaler  = StandardScaler()
    X_norm  = scaler.fit_transform(X_pca).astype(np.float32)
    V_norm  = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)
    dataset = BioDataset(X=jnp.array(X_norm), V=jnp.array(V_norm),
                         labels=jnp.array(labels_np), lineage_pairs=None)

    test_triples = np.load(TEST_TRIPLES)

    # ── Load VAE ─────────────────────────────────────────────────────────────
    print("Loading VAE & attaching Randers metric ...")
    key = jax.random.PRNGKey(42)
    vae_p1      = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM, n_types, key)
    vae_randers = attach_datadriven_randers_metric(vae_p1, dataset, n_anchors=2000, sigma=0.4)

    # ── Encode full dataset ──────────────────────────────────────────────────
    print("Encoding full dataset ...")
    # Use smaller sample for speed — 20k cells
    sample_size = min(20000, X_norm.shape[0])
    rng = np.random.default_rng(0)
    idx_sample = rng.choice(X_norm.shape[0], sample_size, replace=False)
    idx_sample_sorted = np.sort(idx_sample)

    Z_all   = encode_all(vae_randers, jnp.array(X_norm[idx_sample_sorted]))
    labs    = labels_np[idx_sample_sorted]
    tpoints = time_point[idx_sample_sorted]

    pca2 = PCA(n_components=2).fit(Z_all)
    Z2d  = pca2.transform(Z_all)

    # ── Figure setup ────────────────────────────────────────────────────────
    print("Building figure ...")
    fig = plt.figure(figsize=(21, 7))
    gs = GridSpec(1, 3, figure=fig, wspace=0.28)
    ax_det   = fig.add_subplot(gs[0, 0])
    ax_wind  = fig.add_subplot(gs[0, 1])
    ax_traj  = fig.add_subplot(gs[0, 2])

    # ---- Shared cell-type colormap ----------------------------------------
    fate_cmap = cm.get_cmap("tab20", n_types)
    fate_colors_list = {fname: fate_cmap(i) for i, fname in enumerate(fate_names)}

    # ──────────────────────────────────────────────────────────────────────────
    # Panel A: log det G heatmap
    # ──────────────────────────────────────────────────────────────────────────
    print("  Panel A: log det G heatmap ...")
    try:
        GX, GY, logdet = compute_logdet_grid(vae_randers, Z_all, pca2, n_grid=35)
        logdet_clipped = np.clip(logdet, np.percentile(logdet, 2), np.percentile(logdet, 98))
        cf = ax_det.contourf(GX, GY, logdet_clipped, levels=24, cmap="plasma", alpha=0.85, zorder=0)
        divider = make_axes_locatable(ax_det)
        cax = divider.append_axes("right", size="4%", pad=0.06)
        fig.colorbar(cf, cax=cax, label="log det  G(z)")
    except Exception as e:
        print(f"  Warning: log det computation failed: {e}")

    # Overlay fate scatter (sample for speed)
    for fname in TARGET_FATES:
        if fname not in fate_names: continue
        fidx = fate_names.index(fname)
        mask = labs == fidx
        ax_det.scatter(Z2d[mask, 0], Z2d[mask, 1], s=3, alpha=0.55,
                       color=fate_colors_list[fname], label=fname, zorder=3)

    # Background (grey for non-target)
    bg_mask = ~np.isin(labs, [fate_names.index(f) for f in TARGET_FATES if f in fate_names])
    ax_det.scatter(Z2d[bg_mask, 0], Z2d[bg_mask, 1], s=1, alpha=0.08, color="white", zorder=1)
    ax_det.set_title("(A)  Pullback Metric Determinant\nlog det $G(z) = \\log \\det J^T J$",
                     fontsize=11, pad=8)
    ax_det.set_xlabel("Latent PC1"); ax_det.set_ylabel("Latent PC2")
    ax_det.legend(fontsize=8, markerscale=3, loc="upper right")
    ax_det.set_xticks([]); ax_det.set_yticks([])

    # ──────────────────────────────────────────────────────────────────────────
    # Panel B: wind quiver colored by magnitude
    # ──────────────────────────────────────────────────────────────────────────
    print("  Panel B: wind quiver ...")

    # Background: pseudotime / day colored
    pt_norm = mcolors.Normalize(vmin=tpoints.min(), vmax=tpoints.max())
    sc = ax_wind.scatter(Z2d[:, 0], Z2d[:, 1], s=2, c=tpoints,
                         cmap="viridis", norm=pt_norm, alpha=0.3, zorder=1)
    divider2 = make_axes_locatable(ax_wind)
    cax2 = divider2.append_axes("right", size="4%", pad=0.06)
    fig.colorbar(sc, cax=cax2, label="Pseudotime (day)")

    try:
        GX_w, GY_w, Wx, Wy, mags_grid = compute_wind_grid(vae_randers, pca2, Z_all, n_grid=18)
        mag_norm = mcolors.Normalize(vmin=0, vmax=np.percentile(mags_grid, 95))
        quiv_colors = cm.copper(mag_norm(mags_grid.ravel()))
        ax_wind.quiver(GX_w, GY_w, Wx, Wy,
                       mags_grid, cmap="copper", norm=mag_norm,
                       scale=None, scale_units="xy",
                       width=0.003, alpha=0.9, zorder=4)
    except Exception as e:
        print(f"  Warning: wind quiver failed: {e}")

    ax_wind.set_title("(B)  Randers Wind Field  $W(z)$\n(arrows colored by magnitude)",
                      fontsize=11, pad=8)
    ax_wind.set_xlabel("Latent PC1"); ax_wind.set_ylabel("Latent PC2")
    ax_wind.set_xticks([]); ax_wind.set_yticks([])

    # ──────────────────────────────────────────────────────────────────────────
    # Panel C: exemplar observed vs. counterfactual path
    # ──────────────────────────────────────────────────────────────────────────
    print("  Panel C: exemplar trajectories ...")

    # We need Z indices into the *full* cell set, not the subsample.
    # Re-use test_triples (which index into full X_norm).
    # Encode just the cells we need.
    ex_fates_pairs = [(TARGET_FATES[0], TARGET_FATES[1])]   # true vs. counterfactual
    if len(TARGET_FATES) < 2:
        ex_fates_pairs = [(TARGET_FATES[0], fate_names[0])]

    ax_traj.scatter(Z2d[:, 0], Z2d[:, 1], s=1, alpha=0.1, color="lightgray", zorder=0)

    # Draw fate centroids
    fate_centroid_2d = {}
    for fname in TARGET_FATES:
        if fname not in fate_names: continue
        fidx = fate_names.index(fname)
        # Encode cells of this fate from the subsample
        mask = labs == fidx
        if mask.sum() > 5:
            c2d = Z2d[mask].mean(axis=0)
            fate_centroid_2d[fname] = c2d
            ax_traj.scatter(*c2d, s=300, marker="*",
                            color=fate_colors_list[fname], zorder=7,
                            edgecolors="white", linewidths=0.8, label=f"{fname}")

    # Find an exemplar triple from the full dataset (index into X_norm)
    trj = None
    for fname_true, fname_cf in ex_fates_pairs:
        if fname_true not in fate_names or fname_cf not in fate_names: continue
        fidx_true = fate_names.index(fname_true)
        fidx_cf   = fate_names.index(fname_cf)
        cf_cells  = np.where(labels_np == fidx_cf)[0]
        @eqx.filter_jit
        @eqx.filter_vmap(in_axes=(None, 0))
        def get_means(v_mod, x):

            return v_mod._get_dist(x).mean

        for triple in test_triples[:5000]:
            i2, i4, i6 = triple
            if labels_np[i6] == fidx_true and len(cf_cells) > 0:
                z2 = get_means(vae_randers, jnp.array(X_norm[[i2]]))[0]
                z4 = get_means(vae_randers, jnp.array(X_norm[[i4]]))[0]
                z6 = get_means(vae_randers, jnp.array(X_norm[[i6]]))[0]
                z6_cf = get_means(vae_randers, jnp.array(X_norm[[cf_cells[0]]]))[0]
                trj = (fname_true, fname_cf, z2, z4, z6, z6_cf)
                break

        if trj: break

    if trj:
        fname_true, fname_cf, z2, z4, z6, z6_cf = trj
        pts_3d = {
            "z2": np.array(z2), "z4": np.array(z4),
            "z6": np.array(z6), "z6_cf": np.array(z6_cf),
        }
        pts_2d = {k: pca2.transform(v[None])[0] for k, v in pts_3d.items()}

        # ---- Draw observed path (green) --------------------------------------
        obs_path_pts = [pts_2d["z2"], pts_2d["z4"], pts_2d["z6"]]
        for pt_a, pt_b in zip(obs_path_pts[:-1], obs_path_pts[1:]):
            ax_traj.annotate("", xy=pt_b, xytext=pt_a,
                             arrowprops=dict(arrowstyle="-|>", color="limegreen",
                                             lw=2.0, mutation_scale=16), zorder=6)

        # ---- Draw counterfactual path (red dashed) ---------------------------
        cf_path_pts = [pts_2d["z2"], pts_2d["z4"], pts_2d["z6_cf"]]
        for pt_a, pt_b in zip(cf_path_pts[:-1], cf_path_pts[1:]):
            ax_traj.annotate("", xy=pt_b, xytext=pt_a,
                             arrowprops=dict(arrowstyle="-|>", color="tomato",
                                             lw=2.0, mutation_scale=16,
                                             linestyle="dashed"), zorder=6)

        # ---- Cost annotations ------------------------------------------------
        @eqx.filter_jit
        def arc_fn(m_mod, a, b):
            return arc_length_segment(m_mod, a, b, steps=20)
            
        e_obs1 = float(arc_fn(vae_randers.metric, z2, z4))
        e_obs2 = float(arc_fn(vae_randers.metric, z4, z6))
        e_cf1  = float(arc_fn(vae_randers.metric, z2, z4))
        e_cf2  = float(arc_fn(vae_randers.metric, z4, z6_cf))


        # Mid annotation for observed path
        mid_obs = (pts_2d["z4"] + pts_2d["z6"]) / 2
        ax_traj.text(mid_obs[0] + 0.05, mid_obs[1], f"$E$={e_obs1 + e_obs2:.2f}",
                     fontsize=8, color="limegreen", fontweight="bold", zorder=8)

        mid_cf = (pts_2d["z4"] + pts_2d["z6_cf"]) / 2
        ax_traj.text(mid_cf[0] + 0.05, mid_cf[1], f"$E$={e_cf1 + e_cf2:.2f}",
                     fontsize=8, color="tomato", fontweight="bold", zorder=8)

        # Key points
        for lbl, pt, col, mksz in [
            ("Day 2", pts_2d["z2"], "gold", 80),
            ("Day 4", pts_2d["z4"], "dodgerblue", 60),
            (fname_true, pts_2d["z6"], "limegreen", 80),
            (fname_cf + " (CF)", pts_2d["z6_cf"], "tomato", 80),
        ]:
            ax_traj.scatter(*pt, s=mksz, color=col, zorder=9,
                            edgecolors="white", linewidths=0.8)
            ax_traj.text(pt[0] + 0.03, pt[1] + 0.03, lbl,
                         fontsize=7, fontweight="bold", zorder=10)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="limegreen", lw=2, label=f"Obs. path → {fname_true}"),
            Line2D([0], [0], color="tomato", lw=2, linestyle="--", label=f"Counter. → {fname_cf}"),
        ]
        ax_traj.legend(handles=legend_elements, fontsize=8)

    ax_traj.set_title("(C)  Exemplar Trajectory vs Counterfactual\n"
                      "($E$ = Randers arc length cost)", fontsize=11, pad=8)
    ax_traj.set_xlabel("Latent PC1"); ax_traj.set_ylabel("Latent PC2")
    ax_traj.set_xticks([]); ax_traj.set_yticks([])

    # ── Save ────────────────────────────────────────────────────────────────
    plt.suptitle("Randers Metric — Weinreb Hematopoiesis\n"
                 "Geometric topology, directional wind, and fate-cost discrimination",
                 fontsize=13, y=1.02)
    plt.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    print(f"\n✓ Saved publication figure → {OUTPUT}")


if __name__ == "__main__":
    main()
