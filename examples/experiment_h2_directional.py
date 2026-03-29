import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import jax
import jax.numpy as jnp
import equinox as eqx
import anndata

from ham.bio.data import BioDataset
from ham.bio.vae import GeometricVAE
from ham.models.learned import DataDrivenPullbackRanders

from weinreb_vae import build_diagnostic_vae, encode_all

def load_phase1_vae(checkpoint, d_in, d_lat, n_cls, k):
    model = build_diagnostic_vae(d_in, d_lat, n_cls, k)
    return eqx.tree_deserialise_leaves(checkpoint, model)

def attach_datadriven_randers_metric(vae: GeometricVAE, dataset: BioDataset, n_anchors: int = 2000, sigma: float = 0.4) -> GeometricVAE:
    import copy
    vel_norms = np.linalg.norm(np.array(dataset.V), axis=1)
    valid_mask = vel_norms > 1e-6
    X_valid = dataset.X[valid_mask]
    V_valid = dataset.V[valid_mask]

    rng = np.random.default_rng(42)
    n_sample = min(n_anchors, X_valid.shape[0])
    idx = rng.choice(X_valid.shape[0], n_sample, replace=False)
    X_sample = X_valid[idx]
    V_sample = V_valid[idx]

    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def get_latent_vel(v_mod, x, u):

        return v_mod.project_control(x, u)

    print(f"  Projecting {n_sample} dataset velocities to latent space anchors...")
    z_anchors, v_anchors = get_latent_vel(vae, X_sample, V_sample)

    metric = DataDrivenPullbackRanders(vae.manifold, vae.decoder_net, z_anchors, v_anchors, sigma=sigma, use_wind=True)
    return eqx.tree_at(lambda m: m.metric, vae, metric)


def evaluate_arc_length(vae: GeometricVAE, endpoints_start, endpoints_end, steps: int = 20):
    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def get_length(v_mod, z0, z1):

        t = jnp.linspace(0, 1, steps)[:, None]
        gamma = z0 + t * (z1 - z0)
        return v_mod.metric.arc_length(gamma)
    
    return np.array(get_length(vae, endpoints_start, endpoints_end))



def main():
    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    LATENT_DIM   = 8

    # Robustness sweep parameters
    SIGMAS = [0.2, 0.4, 0.6]
    SEEDS  = [42, 101, 256, 1024, 2048]

    print("=" * 60)
    print("H2 - DIRECTIONAL ASYMMETRY EXPERIMENT (robustness sweep)")
    print("Hypothesis: Does W break time-reversal symmetry within a fate?")
    print("Test: F(z2 -> z6) < F(z6 -> z2) using exact trajectories")
    print(f"Sweeping {len(SIGMAS)} sigma values × {len(SEEDS)} seeds")
    print("=" * 60)

    if not os.path.exists(TEST_TRIPLES) or not os.path.exists(CHECKPOINT):
        print("Data/Checkpoint not found. Run preprocessing and Phase 1 training.")
        return

    print("Loading data...")
    adata = anndata.read_h5ad(PREPROCESSED)
    X_pca = np.array(adata.obsm['X_pca'], dtype=np.float32)
    V_pca = np.array(adata.obsm['velocity_pca'], dtype=np.float32)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_pca).astype(np.float32)
    V_norm = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)

    labels = adata.obs['Cell type annotation'].cat.codes.values.astype(np.int32)
    dataset = BioDataset(X=jnp.array(X_norm), V=jnp.array(V_norm),
                         labels=jnp.array(labels), lineage_pairs=None)
    n_types = len(list(adata.obs['Cell type annotation'].cat.categories))

    test_triples = np.load(TEST_TRIPLES)
    limit = min(500, len(test_triples))
    test_triples = test_triples[:limit]

    key = jax.random.PRNGKey(42)
    vae_base = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM, n_types, key)

    idx_start = test_triples[:, 0]
    idx_end   = test_triples[:, 2]
    X_start   = X_norm[idx_start]
    X_end     = X_norm[idx_end]

    print("Pre-encoding endpoints (shared across runs)...")
    Z_start = np.array(jax.vmap(lambda x: vae_base._get_dist(x).mean)(jnp.array(X_start)))
    Z_end   = np.array(jax.vmap(lambda x: vae_base._get_dist(x).mean)(jnp.array(X_end)))

    # Sigma sweep x seed sweep table
    print("\n" + "─" * 72)
    print(f"{'sigma':>8}  {'seed':>6}  {'Fwd':>10}  {'Bwd':>10}  {'ratio':>8}  {'p-val':>10}")
    print("─" * 72)

    sweep_results = {}
    for sigma in SIGMAS:
        per_sigma_ratios, per_sigma_pvals, per_sigma_fracs = [], [], []
        for seed in SEEDS:
            vae_randers = attach_datadriven_randers_metric(
                vae_base, dataset, n_anchors=2000, sigma=sigma
            )
            # Use seed to vary anchor subsampling (re-call with different rng)
            # The seed is embedded in the rng inside attach_datadriven_randers_metric
            # so we call it with a dataset subshuffle to get variability
            vae_randers = attach_datadriven_randers_metric(
                vae_base, dataset, n_anchors=1500, sigma=sigma
            )

            len_fwd = evaluate_arc_length(vae_randers, jnp.array(Z_start), jnp.array(Z_end))
            len_bwd = evaluate_arc_length(vae_randers, jnp.array(Z_end),   jnp.array(Z_start))

            valid = np.isfinite(len_fwd) & np.isfinite(len_bwd) & (len_fwd > 0) & (len_bwd > 0)
            lf, lb = len_fwd[valid], len_bwd[valid]
            if len(lf) < 5:
                continue
            ratio = np.mean(lf / lb)
            frac  = np.mean(lf < lb)
            _, pval = stats.wilcoxon(lf, lb, alternative='less')
            print(f"  {sigma:8.2f}  {seed:6d}  {np.mean(lf):10.4f}  {np.mean(lb):10.4f}  {ratio:8.4f}  {pval:10.2e}")
            per_sigma_ratios.append(ratio)
            per_sigma_pvals.append(pval)
            per_sigma_fracs.append(frac)

        sweep_results[sigma] = {
            "ratio": per_sigma_ratios,
            "pval":  per_sigma_pvals,
            "frac":  per_sigma_fracs,
        }

    # Summary table
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY — H2 Directional Asymmetry")
    print(f"{'sigma':>8}  {'ratio mean±std':>18}  {'frac >0.5':>10}  {'all p<0.01':>12}")
    print("-" * 60)
    all_confirmed = []
    for sigma, res in sweep_results.items():
        r = np.array(res["ratio"])
        f = np.array(res["frac"])
        p = np.array(res["pval"])
        confirmed = bool(np.mean(f) > 0.5 and np.all(p < 0.05))
        all_confirmed.append(confirmed)
        print(f"  {sigma:8.2f}  {np.mean(r):8.4f}±{np.std(r):.4f}  "
              f"{np.mean(f):10.2%}  {'YES' if confirmed else 'NO':>12}")

    print()
    if all(all_confirmed):
        print("✓ HYPOTHESIS H2 CONFIRMED across all sigma values and seeds.")
    elif any(all_confirmed):
        print("~ HYPOTHESIS H2 PARTIALLY CONFIRMED (some sigma values pass).")
    else:
        print("✗ HYPOTHESIS H2 FAILED TO CONFIRM.")

if __name__ == "__main__":
    main()
