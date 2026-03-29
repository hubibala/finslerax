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
from ham.solvers.avbd import AVBDSolver

from weinreb_vae import build_diagnostic_vae, encode_all, TARGET_FATES

# Reuse the datadriven attachment created for H2
from experiment_h2_directional import attach_datadriven_randers_metric

def load_phase1_vae(checkpoint, d_in, d_lat, n_cls, k):
    model = build_diagnostic_vae(d_in, d_lat, n_cls, k)
    return eqx.tree_deserialise_leaves(checkpoint, model)

def build_riemannian_fallback(vae: GeometricVAE) -> GeometricVAE:
    """
    Build a null (Riemannian, W=0) VAE with use_wind=False by directly
    setting the metric field on a copy of the frozen module.

    Uses object.__setattr__ to bypass eqx.Module's frozen-field guard,
    which is the canonical equinox approach for one-shot field replacement
    when the field contains a new static structure (use_wind).
    """
    old_m = vae.metric
    null_metric = DataDrivenPullbackRanders(
        manifold=old_m.manifold,
        decoder=old_m.decoder,
        anchors_z=old_m.w_net.anchors_z,
        anchors_v=old_m.w_net.anchors_v,
        sigma=old_m.w_net.sigma,
        use_wind=False,
    )
    return eqx.tree_at(lambda m: m.metric, vae, null_metric)


def arc_length_normalized(metric, z_start, z_end, solver, steps=20):
    """
    Computes \int F(gamma, u) dt along the ACTUAL geodesic computed via AVBD.
    This normalizes the Finsler arc length by the Euclidean length,
    removing the distance confound.
    """
    # Solve for the ACTUAL geodesic
    trajectory = solver.solve(metric, z_start, z_end, n_steps=steps, train_mode=False)
    
    # Integrated Finsler Length along the geodesic
    finsler_len = metric.arc_length(trajectory.xs)
    
    # Euclidean Length of the path (baseline for normalization)
    eucl_len = jnp.linalg.norm(z_end - z_start) + 1e-8
    
    # Unit speed equivalent
    return finsler_len / eucl_len

def two_segment_normalized_cost(metric, z2, z4, z6, solver):
    cost_1 = arc_length_normalized(metric, z2, z4, solver)
    cost_2 = arc_length_normalized(metric, z4, z6, solver)
    return cost_1 + cost_2

def geodesic_proximity(metric, z_start, z_end, z_target, solver, steps=20):
    """
    Computes how close the geodesic z_start -> z_end passes to z_target.
    Returns the minimum Euclidean distance.
    """
    trajectory = solver.solve(metric, z_start, z_end, n_steps=steps, train_mode=False)
    # trajectory.xs is (N, D)
    dists = jnp.linalg.norm(trajectory.xs - z_target, axis=-1)
    return jnp.min(dists)

def build_fate_attractors(vae, dataset, lineage_triples, cell_type_labels, fate_names, target_fates):
    attractors = {}
    day6_idx = lineage_triples[:, 2]
    day6_labels = cell_type_labels[day6_idx]
    for fate in target_fates:
        if fate not in fate_names: continue
        fidx = fate_names.index(fate)
        mask = day6_labels == fidx
        if mask.sum() == 0: continue
        X_fate = dataset.X[day6_idx[mask]]
        
        @eqx.filter_jit
        @eqx.filter_vmap(in_axes=(None, 0))
        def get_means(v_mod, x):

            return v_mod._get_dist(x).mean
            
        z_fate = get_means(vae, X_fate)
        attractors[fate] = jnp.mean(z_fate, axis=0)

    return attractors

def main():
    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    LATENT_DIM   = 8

    print("="*60)
    print("H3 - DISCRIMINATIVE GEOMETRY EXPERIMENT")
    print("Hypothesis: Does Randers metric assign lower cost to correct paths than wrong paths?")
    print("Test: E_cf / E_obs > 1 for trajectory energy, controlled for euclidean distance.")
    print("="*60)

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
    fate_names = list(adata.obs['Cell type annotation'].cat.categories)

    dataset = BioDataset(X=jnp.array(X_norm), V=jnp.array(V_norm), labels=jnp.array(labels), lineage_pairs=None)
    n_types = len(fate_names)

    triples = np.load(TEST_TRIPLES)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(triples), min(1000, len(triples)), replace=False)
    triples = triples[idx]

    key = jax.random.PRNGKey(42)
    vae_p1 = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM, n_types, key)
    vae_randers = attach_datadriven_randers_metric(vae_p1, dataset)
    vae_riemannian = build_riemannian_fallback(vae_randers)

    attractors = build_fate_attractors(vae_randers, dataset, triples, labels, fate_names, TARGET_FATES)

    # Encode path cells
    X2 = dataset.X[triples[:, 0]]
    X4 = dataset.X[triples[:, 1]]
    X6 = dataset.X[triples[:, 2]]
    
    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0))
    def get_means(v_mod, x):

        return v_mod._get_dist(x).mean
        
    z2s = get_means(vae_randers, X2)
    z4s = get_means(vae_randers, X4)
    z6s = get_means(vae_randers, X6)

    day6_labels = labels[triples[:, 2]]

    results = {
        'randers': [],
        'riemannian': []
    }

    print("\nRunning Null and Hypothesis Metric Loops...")
    
    # Pre-filter valid triples and prepare batched pairs
    valid_idxs, valid_z6_cfs = [], []
    for i in range(len(triples)):
        obs_fidx = day6_labels[i]
        obs_fname = fate_names[obs_fidx] if obs_fidx < len(fate_names) else None
        if obs_fname not in TARGET_FATES or obs_fname not in attractors:
            continue
        z2 = z2s[i]
        cf_candidates = [(fn, fz) for fn, fz in attractors.items() if fn != obs_fname]
        if not cf_candidates: continue
        z6_cf = min(cf_candidates, key=lambda fz: float(jnp.linalg.norm(fz[1] - z2)))[1]
        valid_idxs.append(i)
        valid_z6_cfs.append(z6_cf)
        
    if not valid_idxs:
        print("No valid trajectories found for targeted fates.")
        return
        
    bz2s = jnp.stack([z2s[i] for i in valid_idxs])
    bz4s = jnp.stack([z4s[i] for i in valid_idxs])
    bz6_obss = jnp.stack([z6s[i] for i in valid_idxs])
    bz6_cfs = jnp.stack(valid_z6_cfs)

    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, None, 0, 0, 0, 0, None))
    def batch_eval(m_rand, m_riem, z2, z4, z6_obs, z6_cf, solver):
        # 1. Energy Discriminative Test (z2 -> z4 -> z6)
        e_obs_rand = arc_length_normalized(m_rand, z2, z6_obs, solver)
        e_cf_rand = arc_length_normalized(m_rand, z2, z6_cf, solver)
        e_obs_riem = arc_length_normalized(m_riem, z2, z6_obs, solver)
        e_cf_riem = arc_length_normalized(m_riem, z2, z6_cf, solver)
        
        # 2. Proximity Test (Does geodesic z2 -> z6 pass through z4?)
        prox_rand = geodesic_proximity(m_rand, z2, z6_obs, z4, solver)
        prox_riem = geodesic_proximity(m_riem, z2, z6_obs, z4, solver)
        
        return jnp.array([e_obs_rand, e_cf_rand, e_obs_riem, e_cf_riem, prox_rand, prox_riem])
        
    solver = AVBDSolver(iterations=20)
    output = batch_eval(vae_randers.metric, vae_riemannian.metric, bz2s, bz4s, bz6_obss, bz6_cfs, solver)

    output = np.array(output)
    
    e_obs_rand, e_cf_rand, e_obs_riem, e_cf_riem, prox_rand, prox_riem = output.T
    
    # Compute Ratios
    r_rand = e_cf_rand / (e_obs_rand + 1e-8)
    r_riem = e_cf_riem / (e_obs_riem + 1e-8)
    
    # Filter valid
    mask = (e_obs_rand > 0) & (e_cf_rand > 0) & (e_obs_riem > 0) & (e_cf_riem > 0)
    r_rand = r_rand[mask]
    r_riem = r_riem[mask]

    print("\nRESULTS (Distance-Controlled):")
    print(f"  Valid Trajectories Evaluated: {len(r_rand)}")
    print(f"  Mean Ratio (Randers):         {np.mean(r_rand):.4f}")
    print(f"  Mean Ratio (Riemannian Null): {np.mean(r_riem):.4f}")
    print(f"  Randers Fraction > 1.0:       {np.mean(r_rand > 1.0):.2%}")
    print(f"  Riemannian Fraction > 1.0:    {np.mean(r_riem > 1.0):.2%}")

    w_stat, w_pval = stats.wilcoxon(r_rand, r_riem, alternative='greater')
    print(f"  Wilcoxon Paired p-value:      {w_pval:.2e}")

    if w_pval < 0.05 and np.mean(r_rand) > np.mean(r_riem):
        print("\n✓ HYPOTHESIS H3 CONFIRMED: Randers metric correctly discriminates fate trajectories beyond Euclidean distance.")
    else:
        print("\n✗ HYPOTHESIS H3 FAILED TO CONFIRM trajectory energy gap.")

    print("\nRESULTS (Geodesic Proximity to Midpoint z4):")
    print(f"  Mean Min-Dist (Randers):      {np.mean(prox_rand):.4f}")
    print(f"  Mean Min-Dist (Riemannian):   {np.mean(prox_riem):.4f}")
    
    prox_improved = np.mean(prox_rand < prox_riem)
    print(f"  Randers closer than Riem:      {prox_improved:.2%}")

    p_stat, p_pval = stats.wilcoxon(prox_rand, prox_riem, alternative='less')
    print(f"  Wilcoxon Proximity p-value:    {p_pval:.2e}")

    if p_pval < 0.05:
        print("\n✓ GEODESIC PROXIMITY CONFIRMED: Randers geodesics pass closer to observed intermediate states.")
    else:
        print("\n✗ GEODESIC PROXIMITY FAILED: Randers geodesics are not significantly closer to intermediate states.")

if __name__ == "__main__":
    main()
