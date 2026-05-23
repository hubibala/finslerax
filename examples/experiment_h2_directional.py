"""
experiment_h2_directional.py
============================================
H2 - Directional Asymmetry on Short Observed Clonal Segments (Fully Self-Contained)

For each clonal triple (day2 → day4 → day6):
  - Solve BVP for day2→day4 and day4→day6 separately
  - Compute forward and backward length on the SAME geodesic for each segment
"""

import os
import argparse
import numpy as np
from scipy import stats
import jax
import jax.numpy as jnp
import equinox as eqx
import anndata
import joblib

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


def attach_datadriven_randers_metric(vae, dataset, n_anchors=2000, sigma=0.4, seed=42, permute_wind=False):
    """Self-contained attachment."""
    # Filter velocities to training clones only implicitly via valid_mask
    vel_norms = np.linalg.norm(np.array(dataset.V), axis=1)
    valid_mask = vel_norms > 1e-6
    X_valid = dataset.X[valid_mask]
    V_valid = dataset.V[valid_mask]

    rng = np.random.default_rng(seed)
    n_sample = min(n_anchors, len(X_valid))
    idx = rng.choice(len(X_valid), n_sample, replace=False)

    X_sample = X_valid[idx]
    V_sample = V_valid[idx].copy()
    
    if permute_wind:
        rng.shuffle(V_sample) # Shuffle velocities to destroy wind signal

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


def matched_rank_biserial(x, y):
    """Compute matched-pairs rank biserial correlation for Wilcoxon signed-rank."""
    d = np.array(x) - np.array(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    R_plus = np.sum(ranks[d > 0])
    R_minus = np.sum(ranks[d < 0])
    return (R_plus - R_minus) / (R_plus + R_minus)

def bh_fdr(p_values):
    """Benjamini-Hochberg FDR correction."""
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    fdr_p = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    fdr_p = np.minimum(fdr_p, 1.0)
    original_idx_p = np.empty(n)
    original_idx_p[sorted_idx] = fdr_p
    return original_idx_p

def main():
    parser = argparse.ArgumentParser(description="H2: Directional Asymmetry Experiment")
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu', 'tpu'],
                        help='JAX device to use (default: cpu).')
    args = parser.parse_args()
    from ham.utils import configure_device
    configure_device(args.device)

    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    LATENT_DIM   = 8
    MAX_PAIRS    = 800
    SIGMAS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    SEEDS = [42, 101, 256, 1024, 2048]

    print("=" * 80)
    print("H2: Directional Asymmetry Experiment")
    print("=" * 80)

    if not all(os.path.exists(p) for p in [CHECKPOINT, PREPROCESSED, TEST_TRIPLES]):
        print(f"Missing required files. Looked for {CHECKPOINT}, {PREPROCESSED}, {TEST_TRIPLES}")
        return

    # Load data
    adata = anndata.read_h5ad(PREPROCESSED)
    X_pca = np.array(adata.obsm['X_pca'], dtype=np.float32)
    V_pca = np.array(adata.obsm['velocity_pca'], dtype=np.float32)

    scaler = joblib.load("data/weinreb_pca_scaler.joblib")
    X_norm = scaler.transform(X_pca).astype(np.float32)
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

    solver = AVBDSolver(iterations=60)

    print("\n" + "─"*130)
    print(f"{'sigma':>6} {'seed':>6} {'L_fwd':>10} {'L_bwd':>10} {'ratio':>8} {'p-val':>10} {'FDR p':>10} {'frac<1':>8} {'eff_size':>10} {'C_viol':>10} {'Energy':>10} {'Failed':>8}")
    print("─"*130)

    @eqx.filter_jit
    def batch_solve(metric, zs_batch, ze_batch):
        def single_pair(zs, ze):
            traj = solver.solve(metric, zs, ze, n_steps=15, train_mode=True)
            xs = traj.xs
            Lf = metric.arc_length(xs)
            Lb = metric.arc_length(xs[::-1])
            return Lf, Lb, traj.constraint_violation, traj.energy
        return jax.vmap(single_pair)(zs_batch, ze_batch)

    # We will collect all p-values for FDR correction
    all_pvals = []
    runs_info = []

    for sigma in SIGMAS:
        for seed in SEEDS:
            # We will run both Randers and Permuted-Wind Randers to provide a baseline
            for permute in [False, True]:
                vae = attach_datadriven_randers_metric(vae_base, dataset, n_anchors=2000, sigma=jnp.array(sigma), seed=seed, permute_wind=permute)

                Z_starts = jnp.concatenate([Z2, Z4], axis=0)
                Z_ends   = jnp.concatenate([Z4, Z6], axis=0)
                
                n_total = Z_starts.shape[0]
                chunk_size = 50
                n_chunks = (n_total + chunk_size - 1) // chunk_size
                
                Lf_list, Lb_list, C_list, E_list = [], [], [], []
                
                for ci, i in enumerate(range(0, n_total, chunk_size)):
                    zs_chunk = Z_starts[i:i+chunk_size]
                    ze_chunk = Z_ends[i:i+chunk_size]
                    L_fwd_c, L_bwd_c, C_c, E_c = batch_solve(vae.metric, zs_chunk, ze_chunk)
                    Lf_list.append(np.array(L_fwd_c))
                    Lb_list.append(np.array(L_bwd_c))
                    C_list.append(np.array(C_c))
                    E_list.append(np.array(E_c))

                Lf = np.concatenate(Lf_list)
                Lb = np.concatenate(Lb_list)
                C_viol = np.concatenate(C_list)
                Energy = np.concatenate(E_list)
                
                valid = (Lf > 0) & (Lb > 0) & np.isfinite(Lf) & np.isfinite(Lb)
                failed_frac = 1.0 - np.mean(valid)
                
                Lf, Lb = Lf[valid], Lb[valid]
                C_viol, Energy = C_viol[valid], Energy[valid]

                if len(Lf) < 30:
                    continue

                ratio = np.mean(Lf / Lb)
                frac = np.mean(Lf < Lb)
                _, pval = stats.wilcoxon(Lf, Lb, alternative='less')
                eff_size = matched_rank_biserial(Lf, Lb)
                mean_c = np.mean(C_viol)
                mean_e = np.mean(Energy)

                all_pvals.append(pval)
                runs_info.append({
                    'sigma': sigma, 'seed': seed, 'permute': permute,
                    'Lf': np.mean(Lf), 'Lb': np.mean(Lb), 'ratio': ratio,
                    'pval': pval, 'frac': frac, 'eff_size': eff_size,
                    'mean_c': mean_c, 'mean_e': mean_e, 'failed_frac': failed_frac
                })

    fdr_pvals = bh_fdr(all_pvals)
    
    for i, run in enumerate(runs_info):
        run['fdr_pval'] = fdr_pvals[i]
        tag = "PERM" if run['permute'] else "NORM"
        print(f"{run['sigma']:6.2f} {run['seed']:6d} {run['Lf']:10.3f} {run['Lb']:10.3f} "
              f"{run['ratio']:8.4f} {run['pval']:10.1e} {run['fdr_pval']:10.1e} {run['frac']:8.1%} "
              f"{run['eff_size']:10.3f} {run['mean_c']:10.2e} {run['mean_e']:10.3f} {run['failed_frac']:8.1%} [{tag}]")

    # Summary
    print("\n" + "=" * 80)
    print("H2 ROBUSTNESS SUMMARY — Short Observed Segments")
    print(f"{'sigma':>8}  {'type':>6}  {'ratio mean±std':>18}  {'frac Lf<Lb':>12}  {'p_fdr<0.05':>12}")
    print("-" * 80)

    for sigma in SIGMAS:
        for permute in [False, True]:
            sigma_runs = [r for r in runs_info if r['sigma'] == sigma and r['permute'] == permute]
            if not sigma_runs:
                continue
            r = np.array([run['ratio'] for run in sigma_runs])
            f = np.array([run['frac'] for run in sigma_runs])
            p = np.array([run['fdr_pval'] for run in sigma_runs])
            p05_frac = np.mean(p < 0.05)
            tag = "PERM" if permute else "NORM"
            print(f"  {sigma:8.2f}  {tag:6s}  {np.mean(r):8.4f}±{np.std(r):.4f}  "
                  f"{np.mean(f):12.1%}  {p05_frac:12.1%}")

    print("\nH2 Short-Segment Evaluation Completed.")

if __name__ == "__main__":
    main()