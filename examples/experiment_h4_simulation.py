"""
experiment_h4_simulation.py
============================
H4 — Forward predictive simulation via stochastic integration.

Hypothesis: Starting from Day-2 latent positions, integrating the Randers
wind field W(z) forward produces endpoints closer to the true Day-6 fate
than the isotropic null (W=0).

Method: Euler-Maruyama SDE
    dZ = W(Z) dt + σ_noise √dt dW

Evaluation on held-out clonal triples (day2 → day6).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import jax
import jax.numpy as jnp
import equinox as eqx

import anndata

from ham.bio.data import BioDataset
from ham.bio.vae import GeometricVAE
from ham.models.learned import DataDrivenPullbackRanders

# Reuse helpers
from weinreb_vae import build_diagnostic_vae, encode_all, TARGET_FATES
from experiment_h2_directional import attach_datadriven_randers_metric, load_phase1_vae
from experiment_h3_discriminative import build_riemannian_fallback


# ─────────────────────────────────────────────────────────────────────────────
# SDE Simulation (Fixed & Clean)
# ─────────────────────────────────────────────────────────────────────────────

@eqx.filter_jit
def single_trajectory(wind_fn, z0, keys, dt, sigma, n_steps):
    """Run one stochastic trajectory using Euler-Maruyama."""
    def step(z, key):
        w = wind_fn(z)
        eps = jax.random.normal(key, z.shape)
        return z + w * dt + sigma * jnp.sqrt(dt) * eps, None

    z_final, _ = jax.lax.scan(step, z0, keys)
    return z_final


def simulate_sde_batch(
    z_starts: jnp.ndarray,      # (N, D)
    wind_fn,                    # callable: z -> w
    n_steps: int = 60,
    dt: float = 0.025,
    sigma: float = 0.04,
    n_trajectories: int = 15,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate n_trajectories from each of N starting points.
    Returns: (N, n_trajectories, D) final positions as numpy.
    """
    key = jax.random.PRNGKey(seed)
    N = z_starts.shape[0]

    # Generate all random keys
    all_keys = jax.random.split(key, N * n_trajectories * n_steps)
    all_keys = all_keys.reshape(N, n_trajectories, n_steps, 2)

    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def simulate_all_starts(w_fn, z0_point, keys_batch):
        # z0_point: (D,), keys_batch: (n_trajectories, n_steps, 2)
        # We want to run n_trajectories from the SAME z0_point
        def many_trajs(z0, traj_keys):
            return single_trajectory(w_fn, z0, traj_keys, dt, sigma, n_steps)
        
        return jax.vmap(many_trajs, in_axes=(None, 0))(z0_point, keys_batch)

    endpoints = simulate_all_starts(wind_fn, z_starts, all_keys)  # (N, n_traj, D)
    return np.array(endpoints)


# ─────────────────────────────────────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────────────────────────────────────

def main():
    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    LATENT_DIM   = 8
    N_STEPS      = 60
    N_TRAJ       = 20
    N_EVAL       = 400

    print("=" * 70)
    print("H4 — FORWARD PREDICTIVE SIMULATION")
    print("Does Randers wind improve fate prediction from Day-2 starts?")
    print("=" * 70)

    if not os.path.exists(CHECKPOINT) or not os.path.exists(TEST_TRIPLES):
        print("Missing checkpoint or test triples.")
        return

    # Load data
    adata = anndata.read_h5ad(PREPROCESSED)
    X_pca = np.array(adata.obsm["X_pca"], dtype=np.float32)
    V_pca = np.array(adata.obsm["velocity_pca"], dtype=np.float32)
    labels = adata.obs["Cell type annotation"].cat.codes.values.astype(np.int32)
    fate_names = list(adata.obs["Cell type annotation"].cat.categories)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_pca).astype(np.float32)
    V_norm = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)

    dataset = BioDataset(X=jnp.array(X_norm), V=jnp.array(V_norm),
                         labels=jnp.array(labels), lineage_pairs=None)

    test_triples = np.load(TEST_TRIPLES)[:N_EVAL]

    # Load VAE + Randers
    key = jax.random.PRNGKey(42)
    vae_p1 = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM, len(fate_names), key)
    vae_randers = attach_datadriven_randers_metric(vae_p1, dataset, n_anchors=2000, sigma=0.4)

    vae_null = build_riemannian_fallback(vae_randers)

    # Encode everything
    Z_all = encode_all(vae_randers, jnp.array(X_norm))

    # Fate centroids
    fate_centroids = {}
    for fname in TARGET_FATES:
        if fname not in fate_names: continue
        fidx = fate_names.index(fname)
        mask = labels == fidx
        if mask.sum() > 10:
            # Robust mean across potentially NaN-containing latent points
            z_cluster = Z_all[mask]
            fate_centroids[fname] = jnp.array(np.nanmean(z_cluster, axis=0))

    # Prepare starts and true ends
    idx_day2 = test_triples[:, 0]
    idx_day6 = test_triples[:, 2]
    Z_start = Z_all[idx_day2]
    Z_true  = Z_all[idx_day6]
    true_fate_names = [fate_names[labels[i]] for i in idx_day6]

    # Filter to target fates
    valid_mask = np.array([f in fate_centroids for f in true_fate_names])
    Z_start_v = Z_start[valid_mask]
    Z_true_v  = Z_true[valid_mask]
    true_fates_v = [true_fate_names[i] for i in np.where(valid_mask)[0]]

    print(f"Evaluating forward simulation on {len(Z_start_v)} valid trajectories.")

    # Auto-scale sigma_noise
    pairwise_dists = np.linalg.norm(Z_all[:2000, None] - Z_all[:2000, None, :], axis=-1)
    mask = (pairwise_dists > 1e-8) & (np.isfinite(pairwise_dists))
    if mask.any():
        mean_dist = float(np.mean(pairwise_dists[mask]))
    else:
        print("Warning: Could not compute mean latent distance. Using fallback dist=1.0.")
        mean_dist = 1.0
        
    base_sigma = mean_dist * 0.025
    print(f"Auto-scaled base sigma_noise ≈ {base_sigma:.4f} (mean latent dist = {mean_dist:.3f})")

    dt = 1.0 / N_STEPS

    # Sweep
    sigmas = [base_sigma * f for f in [0.5, 1.0, 1.5]]
    results = {}

    for sigma in sigmas:
        print(f"\n--- sigma = {sigma:.4f} ---")
        res = {"randers": 0.0, "null": 0.0}

        for name, vae in [("randers", vae_randers), ("null", vae_null)]:
            wind_fn = vae.metric.w_net
            endpoints = simulate_sde_batch(
                Z_start_v, wind_fn, n_steps=N_STEPS, dt=dt,
                sigma=sigma, n_trajectories=N_TRAJ, seed=0
            )  # (N, N_TRAJ, D)

            mean_endpoints = endpoints.mean(axis=1)   # (N, D)

            hits = 0
            for i, z_pred in enumerate(mean_endpoints):
                true_fate = true_fates_v[i]
                d_true = float(jnp.linalg.norm(z_pred - fate_centroids[true_fate]))

                d_cf = min(float(jnp.linalg.norm(z_pred - c))
                           for f, c in fate_centroids.items() if f != true_fate)

                if d_true < d_cf * 0.95:   # slight margin to avoid ties
                    hits += 1

            acc = hits / len(Z_start_v)
            print(f"  {name:8s} → accuracy = {acc:.1%}")
            res[name] = acc

        results[sigma] = res

    # Summary
    print("\n" + "="*60)
    print("H4 RESULTS — Predictive Accuracy")
    print(f"{'sigma':>10} {'Randers':>10} {'Null':>10} {'Delta':>8}")
    print("-" * 50)
    for sigma, r in results.items():
        delta = r["randers"] - r["null"]
        print(f"{sigma:10.4f} {r['randers']:10.1%} {r['null']:10.1%} {delta:+8.1%}")

    if np.mean([r["randers"] for r in results.values()]) > np.mean([r["null"] for r in results.values()]) + 0.02:
        print("\n✓ H4 PARTIALLY SUPPORTED: Randers wind improves fate prediction.")
    else:
        print("\n✗ H4 NOT SUPPORTED: No clear improvement from wind in forward simulation.")

    # Optional: save one exemplar trajectory plot
    print("\nGenerating example trajectory visualization...")
    # Clean Z_all for PCA
    Z_clean = Z_all[np.all(np.isfinite(Z_all), axis=1)]
    if len(Z_clean) < 10:
        print("Warning: Too many NaNs/Infs in latent space. Skipping PCA plot.")
        return

    pca2 = PCA(n_components=2).fit(Z_clean)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca2.transform(Z_all)[:,0], pca2.transform(Z_all)[:,1],
               s=1, alpha=0.15, color='lightgray')

    # Plot one example
    example_idx = 0
    z0 = Z_start_v[example_idx]
    wind_fn = vae_randers.metric.w_net
    endpoints = simulate_sde_batch(z0[None], wind_fn, n_steps=N_STEPS,
                                   dt=dt, sigma=sigmas[1], n_trajectories=8, seed=0)

    z0_2d = pca2.transform(np.array(z0)[None])[0]
    ax.scatter(*z0_2d, s=80, color='green', label='Day 2 start')

    for traj in endpoints[0]:
        traj_2d = pca2.transform(np.array(traj))
        ax.plot(traj_2d[:,0], traj_2d[:,1], alpha=0.6, lw=1.2, color='steelblue')

    ax.set_title("Example Forward Simulations with Randers Wind")
    ax.legend()
    plt.savefig("h4_example_trajectories.png", dpi=200, bbox_inches='tight')
    print("Saved → h4_example_trajectories.png")

if __name__ == "__main__":
    main()