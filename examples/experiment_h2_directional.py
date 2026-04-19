"""
experiment_h2_short_segments_standalone.py
============================================
H2 - Directional Asymmetry on Short Observed Clonal Segments (Fully Self-Contained)

For each clonal triple (day2 → day4 → day6):
  - Solve BVP for day2→day4 and day4→day6 separately
  - Compute forward and backward length on the SAME geodesic for each segment
"""

import os
import numpy as np
from scipy import stats
import jax
import jax.numpy as jnp
import equinox as eqx
import anndata
from sklearn.preprocessing import StandardScaler

# ====================== Core HAM Imports ======================
from ham.bio.data import BioDataset
from ham.bio.vae import GeometricVAE
from ham.models.learned import DataDrivenPullbackRanders
from ham.solvers.avbd import AVBDSolver


def load_phase1_vae(checkpoint, d_in, d_lat, n_cls, key):
    """Minimal loader — adjust path if your builder is elsewhere."""
    from weinreb_vae import build_diagnostic_vae   # Keep only this one external import
    model = build_diagnostic_vae(d_in, d_lat, n_cls, key)
    return eqx.tree_deserialise_leaves(checkpoint, model)


def attach_datadriven_randers_metric(vae, dataset, n_anchors=2000, sigma=0.4, seed=42):
    """Self-contained attachment."""
    vel_norms = np.linalg.norm(np.array(dataset.V), axis=1)
    valid_mask = vel_norms > 1e-6
    X_valid = dataset.X[valid_mask]
    V_valid = dataset.V[valid_mask]

    rng = np.random.default_rng(seed)
    n_sample = min(n_anchors, len(X_valid))
    idx = rng.choice(len(X_valid), n_sample, replace=False)

    X_sample = X_valid[idx]
    V_sample = V_valid[idx]

    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def project_vel(v_mod, x, u):
        return v_mod.project_control(x, u)

    z_anchors, v_anchors = project_vel(vae, X_sample, V_sample)

    metric = DataDrivenPullbackRanders(
        manifold=vae.manifold,
        decoder=vae.decoder_net,
        anchors_z=z_anchors,
        anchors_v=v_anchors,
        sigma=sigma,
        use_wind=True
    )
    return eqx.tree_at(lambda m: m.metric, vae, metric)


def arc_length_forward_backward(metric, z_start, z_end, solver, steps=25):
    """Forward and backward length on the SAME geodesic curve."""
    traj = solver.solve(metric, z_start, z_end, n_steps=steps, train_mode=False)
    xs = traj.xs

    L_fwd = metric.arc_length(xs)
    L_bwd = metric.arc_length(xs[::-1])   # reversed on same points

    return float(L_fwd), float(L_bwd)


def main():
    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    LATENT_DIM   = 8
    MAX_PAIRS    = 800

    SIGMAS = [0.2, 0.4, 0.6]
    SEEDS  = [42, 101, 256, 1024, 2048]

    print("=" * 80)
    print("H2 - DIRECTIONAL ASYMMETRY (Short Observed Segments)")
    print("Forward vs Backward length on the SAME geodesic for day2→day4 and day4→day6")
    print("=" * 80)

    if not all(os.path.exists(p) for p in [CHECKPOINT, PREPROCESSED, TEST_TRIPLES]):
        print("Missing required files.")
        return

    # Load data
    adata = anndata.read_h5ad(PREPROCESSED)
    X_pca = np.array(adata.obsm['X_pca'], dtype=np.float32)
    V_pca = np.array(adata.obsm['velocity_pca'], dtype=np.float32)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_pca).astype(np.float32)
    V_norm = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)

    dataset = BioDataset(X=jnp.array(X_norm), V=jnp.array(V_norm),
                         labels=jnp.array(adata.obs['Cell type annotation'].cat.codes.values),
                         lineage_pairs=None)

    test_triples = np.load(TEST_TRIPLES)[:MAX_PAIRS]

    idx2 = test_triples[:, 0]
    idx4 = test_triples[:, 1]
    idx6 = test_triples[:, 2]

    X2 = X_norm[idx2]
    X4 = X_norm[idx4]
    X6 = X_norm[idx6]

    # Load base VAE
    key = jax.random.PRNGKey(42)
    vae_base = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM,
                               len(adata.obs['Cell type annotation'].cat.categories), key)

    # Encode
    Z2 = jax.vmap(lambda x: vae_base._get_dist(x).mean)(jnp.array(X2))
    Z4 = jax.vmap(lambda x: vae_base._get_dist(x).mean)(jnp.array(X4))
    Z6 = jax.vmap(lambda x: vae_base._get_dist(x).mean)(jnp.array(X6))

    # Use scan-based solver (train_mode=True) — much faster under vmap than while_loop.
    # 60 iterations is sufficient for short observed segments (day2→day4, day4→day6).
    solver = AVBDSolver(iterations=60)

    print("\n" + "─"*95)
    print(f"{'sigma':>6} {'seed':>6} {'L_fwd_mean':>12} {'L_bwd_mean':>12} {'ratio':>9} {'p-val':>12} {'frac<1':>8}")
    print("─"*95)

    @eqx.filter_jit
    def batch_arc_lengths(metric, zs_batch, ze_batch):
        def single_pair(zs, ze):
            # train_mode=True uses jax.lax.scan (fixed-length, vmap-friendly)
            # instead of while_loop which stalls on worst-case convergence
            traj = solver.solve(metric, zs, ze, n_steps=15, train_mode=True)
            xs = traj.xs
            Lf = metric.arc_length(xs)
            Lb = metric.arc_length(xs[::-1])
            return Lf, Lb
        return jax.vmap(single_pair)(zs_batch, ze_batch)

    sweep_results = {}
    for sigma in SIGMAS:
        per_sigma_ratios, per_sigma_pvals, per_sigma_fracs = [], [], []
        for seed in SEEDS:
            vae = attach_datadriven_randers_metric(vae_base, dataset, n_anchors=2000, sigma=jnp.array(sigma), seed=seed)

            # ── Batch BVP Solving (Chunked for memory) ──────────────────────────────
            Z_starts = jnp.concatenate([Z2, Z4], axis=0) # (2*N, D)
            Z_ends   = jnp.concatenate([Z4, Z6], axis=0) # (2*N, D)
            
            n_total = Z_starts.shape[0]
            chunk_size = 50  # Smaller chunks for faster first-result and lower memory
            n_chunks = (n_total + chunk_size - 1) // chunk_size
            print(f"  σ={sigma}, seed={seed}: solving {n_total} BVPs in {n_chunks} chunks of {chunk_size}...")
            
            Lf_list, Lb_list = [], []
            
            for ci, i in enumerate(range(0, n_total, chunk_size)):
                zs_chunk = Z_starts[i:i+chunk_size]
                ze_chunk = Z_ends[i:i+chunk_size]
                L_fwd_c, L_bwd_c = batch_arc_lengths(vae.metric, zs_chunk, ze_chunk)
                Lf_list.append(np.array(L_fwd_c))
                Lb_list.append(np.array(L_bwd_c))
                print(f"    chunk {ci+1}/{n_chunks} done ({i+len(zs_chunk)}/{n_total} pairs)", flush=True)

            Lf = np.concatenate(Lf_list)
            Lb = np.concatenate(Lb_list)
            valid = (Lf > 0) & (Lb > 0) & np.isfinite(Lf) & np.isfinite(Lb)
            Lf, Lb = Lf[valid], Lb[valid]

            if len(Lf) < 30:
                continue

            ratio = np.mean(Lf / Lb)
            frac = np.mean(Lf < Lb)
            _, pval = stats.wilcoxon(Lf, Lb, alternative='less')

            print(f"{sigma:6.2f} {seed:6d} {np.mean(Lf):12.4f} {np.mean(Lb):12.4f} "
                  f"{ratio:9.4f} {pval:12.2e} {frac:8.1%}")

            per_sigma_ratios.append(ratio)
            per_sigma_pvals.append(pval)
            per_sigma_fracs.append(frac)

        sweep_results[sigma] = {"ratio": per_sigma_ratios, "pval": per_sigma_pvals, "frac": per_sigma_fracs}

    # Summary
    print("\n" + "=" * 80)
    print("H2 ROBUSTNESS SUMMARY — Short Observed Segments")
    print(f"{'sigma':>8}  {'ratio mean±std':>18}  {'frac Lf<Lb':>12}  {'p<0.05 runs':>14}")
    print("-" * 80)

    for sigma, res in sweep_results.items():
        if not res["ratio"]:
            continue
        r = np.array(res["ratio"])
        f = np.array(res["frac"])
        p = np.array(res["pval"])
        p05_frac = np.mean(p < 0.05)
        print(f"  {sigma:8.2f}  {np.mean(r):8.4f}±{np.std(r):.4f}  "
              f"{np.mean(f):12.1%}  {p05_frac:14.1%}")

    print("\nH2 Short-Segment Evaluation Completed.")

if __name__ == "__main__":
    main()