"""
weinreb_experiment_v5.py
=========================
Loads the trained VAE checkpoint from weinreb_vae_diagnostic.py,
runs phase 2 wind training on the frozen encoder/decoder, then
performs the fate-stratified Randers energy validation.

Pipeline:
  [DONE]  Phase 1 — weinreb_vae_diagnostic.py produced a clean latent space
                     with Y-shaped trajectories and tight fate clusters.
                     Checkpoint: data/weinreb_vae_phase1.eqx

  Phase 2 — Freeze encoder + decoder. Train w_net only from clonal
             pseudo-velocity push-forward. No lineage pairs used.

  Validation — Fate-stratified energy comparison:
                For each observed trajectory (day2→day4→day6):
                  E_counterfactual / E_observed  under Randers metric
                If > 1: the metric penalises wrong-fate paths
                Riemannian is symmetric so ratio ≈ 1 always (null baseline)

Core hypothesis:
  "Randers metric assigns higher energy to wrong-fate paths than to
   observed correct-fate trajectories, when W is learned from clonal
   pseudo-velocity alone."
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import anndata

from ham.bio.data import BioDataset
from ham.bio.vae import GeometricVAE
from ham.geometry.surfaces import EuclideanSpace
from ham.models.learned import PullbackRanders, PullbackRiemannian
from ham.training.losses import LossComponent
from ham.solvers.avbd import AVBDSolver
from ham.training.pipeline import HAMPipeline, TrainingPhase

# Re-use the diagnostic VAE builder so the architecture is identical
from weinreb_vae import (
    build_diagnostic_vae,
    attach_pullback_metric,
    encode_all,
    TARGET_FATES,
    make_tanh_mlp,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Phase 2 losses
# ══════════════════════════════════════════════════════════════════════════════

class RNAVelocityWindLoss(LossComponent):
    """
    Aligns W(z) with clonal pseudo-velocity pushed forward into latent space.

    Uses model.project_control which calls jax.jvp on the encoder mean —
    deterministic, no sampling noise, no trajectory path information.

    batch[0] = x  (normalised PCA coords)
    batch[1] = u  (pseudo-velocity in normalised PCA space)
    """
    magnitude_weight: float = eqx.field(static=True)

    def __init__(self, weight: float = 1.0, magnitude_weight: float = 0.1):
        super().__init__(weight, "RNAVelWind")
        self.magnitude_weight = magnitude_weight

    def __call__(self, model, batch, key):
        x, u = batch[0], batch[1]

        u_norm = jnp.linalg.norm(u)
        valid  = u_norm > 1e-6

        # Deterministic push-forward of velocity into latent space
        z_mean, v_lat = model.project_control(x, u)

        if not hasattr(model.metric, '_get_zermelo_data'):
            return 0.0

        H, W, _ = model.metric._get_zermelo_data(z_mean)

        def h_norm(v):
            return jnp.sqrt(jnp.dot(v, jnp.dot(H, v)) + 1e-8)

        # Direction alignment in H-inner product
        W_unit = W     / h_norm(W)
        v_unit = v_lat / h_norm(v_lat)
        dir_loss = 1.0 - jnp.dot(W_unit, jnp.dot(H, v_unit))

        # Magnitude matching — soft, weighted low to avoid fighting squasher
        mag_loss = (h_norm(W) - h_norm(v_lat)) ** 2

        total = dir_loss + self.magnitude_weight * mag_loss

        # Zero out cells with unreliable velocity (zeros from non-cloned cells)
        return jnp.where(valid, total * self.weight, 0.0)


class WindSmoothnessLoss(LossComponent):
    """
    Frobenius penalty on Jacobian of W(z).
    Prevents W from memorising sparse velocity observations.
    """
    def __init__(self, weight: float = 0.05):
        super().__init__(weight, "WindSmooth")

    def __call__(self, model, batch, key):
        x = batch[0]
        z = model._get_dist(x).mean
        if not hasattr(model.metric, '_get_zermelo_data'):
            return 0.0

        def get_w(z_pt):
            _, W, _ = model.metric._get_zermelo_data(z_pt)
            return W

        jac = jax.jacfwd(get_w)(z)
        return jnp.mean(jac ** 2) * self.weight


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Model construction helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_filter_fn(selector):
    def filter_spec(model):
        base = jax.tree_util.tree_map(lambda _: False, model)
        tgts = selector(model)
        def make_true(n):
            return jax.tree_util.tree_map(
                lambda l: True if eqx.is_array(l) else False, n)
        mask = tuple(make_true(t) for t in tgts) \
               if isinstance(tgts, tuple) else make_true(tgts)
        return eqx.tree_at(selector, base, replace=mask)
    return filter_spec


def load_phase1_vae(
    checkpoint_path: str,
    data_dim: int,
    latent_dim: int,
    n_cell_types: int,
    key: jax.Array,
) -> GeometricVAE:
    """
    Rebuilds the VAE architecture (identical to diagnostic script) and
    loads weights from the checkpoint produced by weinreb_vae_diagnostic.py.
    """
    vae = build_diagnostic_vae(data_dim, latent_dim, n_cell_types, key)
    vae = eqx.tree_deserialise_leaves(checkpoint_path, vae)
    print(f"  Loaded phase-1 checkpoint from {checkpoint_path}")
    return vae


def attach_randers_metric(vae: GeometricVAE, key: jax.Array) -> GeometricVAE:
    """
    Replaces the metric with PullbackRanders:
      H(z) = J^T J  (frozen decoder Jacobian — same as PullbackRiemannian)
      W(z) = w_net(z)  (learnable wind field — initialised randomly)

    The decoder is shared by reference so H always reflects the trained decoder.
    """
    manifold = vae.manifold
    metric   = PullbackRanders(
        manifold,
        decoder   = vae.decoder_net,
        key       = key,
        hidden_dim= 64,
        depth     = 3,
    )
    return eqx.tree_at(lambda m: m.metric, vae, metric)


class KernelWindField(eqx.Module):
    anchors_z: jax.Array
    anchors_v: jax.Array
    sigma: float

    def __init__(self, anchors_z, anchors_v, sigma=0.5):
        self.anchors_z = anchors_z
        self.anchors_v = anchors_v
        self.sigma = sigma

    def __call__(self, z: jax.Array) -> jax.Array:
        dists_sq = jnp.sum((self.anchors_z - z)**2, axis=-1)
        weights = jax.nn.softmax(-dists_sq / (2 * self.sigma**2))
        return jnp.sum(weights[:, None] * self.anchors_v, axis=0)

class DataDrivenPullbackRanders(PullbackRanders):
    """
    Instead of a parameterized neural network, uses a kernel smoother
    over the dataset's exact RNA velocities projected into the latent space.
    """
    def __init__(self, manifold, decoder, anchors_z, anchors_v, sigma=0.5):
        self.dim = manifold.ambient_dim
        self.decoder = decoder
        self.manifold = manifold
        self.epsilon = 1e-5
        
        # Override w_net with the non-parametric kernel smoother
        self.w_net = KernelWindField(anchors_z, anchors_v, sigma)
        # We don't need to call super().__init__ if we fully define the properties, 
        # but to satisfy eqx.Module we define all fields exactly as they exist in PullbackRanders.
        # Actually PullbackRanders inherits from Randers. Let's just bypass standard initialization safely.
        
        # We must initialize h_net correctly to satisfy Randers base class signature.
        self.h_net = lambda x: x 

def attach_datadriven_randers_metric(vae: GeometricVAE, dataset: BioDataset, n_anchors: int = 5000, sigma: float = 0.5) -> GeometricVAE:
    """
    Builds a non-parametric wind field by projecting dataset velocities
    into the latent space and using a Nadaraya-Watson (softmax) kernel smoother.
    """
    # 1. Filter cells with valid velocity
    vel_norms = np.linalg.norm(dataset.V, axis=1)
    valid_mask = vel_norms > 1e-6
    X_valid = dataset.X[valid_mask]
    V_valid = dataset.V[valid_mask]
    
    # 2. Subsample anchors for computational efficiency
    rng = np.random.default_rng(42)
    n_sample = min(n_anchors, X_valid.shape[0])
    idx = rng.choice(X_valid.shape[0], n_sample, replace=False)
    X_sample = X_valid[idx]
    V_sample = V_valid[idx]
    
    # 3. Project to latent space using the encoder's Jacobian
    @jax.vmap
    def get_latent_vel(x, u):
        return vae.project_control(x, u)
    
    print(f"  Projecting {n_sample} dataset velocities to latent space anchors...")
    z_anchors, v_anchors = get_latent_vel(X_sample, V_sample)
    
    # 4. Attach Metric
    metric = DataDrivenPullbackRanders(vae.manifold, vae.decoder_net, z_anchors, v_anchors, sigma=sigma)
    return eqx.tree_at(lambda m: m.metric, vae, metric)



# ══════════════════════════════════════════════════════════════════════════════
# 4.  Validation: fate-stratified Randers energy comparison
#
#  For each observed triple (day2, day4, day6) whose day-6 cell is
#  annotated as one of TARGET_FATES:
#
#    E_obs = energy(z2→z4) + energy(z4→z6_obs)    observed correct-fate path
#    E_cf  = energy(z2→z4) + energy(z4→z6_cf)     counterfactual wrong-fate end
#
#  where z6_cf is the mean latent position of the hardest wrong-fate attractor
#  (closest wrong-fate centroid to z2 — most challenging comparison).
#
#  energy(z_a, z_b) = metric.energy(z_a, z_b - z_a)   two-segment discrete action
#
#  Randers metric is asymmetric: energy(z, v) ≠ energy(z, -v).
#  If W is aligned with correct-fate flow, E_obs < E_cf  →  ratio > 1.
#  Riemannian is symmetric: ratio ≈ 1 always (null baseline).
#
#  Primary test:  one-sided Wilcoxon  rand_ratios > riem_ratios
#  Secondary test: one-sample t-test  rand_ratios > 1.0
# ══════════════════════════════════════════════════════════════════════════════

def encode_mean(model: GeometricVAE, x: jnp.ndarray) -> jnp.ndarray:
    return model._get_dist(x).mean


def two_segment_energy(metric, z2, z4, z6):
    """
    Discrete action of path z2 → z4 → z6.
    E = F(z2, z4-z2)² / 2  +  F(z4, z6-z4)² / 2
    """
    return metric.energy(z2, z4 - z2) + metric.energy(z4, z6 - z4)


def build_fate_attractors(
    vae: GeometricVAE,
    dataset: BioDataset,
    lineage_triples: np.ndarray,
    cell_type_labels: np.ndarray,
    fate_names: list,
    target_fates: list,
) -> Dict[str, jnp.ndarray]:
    """
    Mean latent position of day-6 cells for each target fate.
    Used as counterfactual endpoints.
    """
    attractors = {}
    day6_idx    = lineage_triples[:, 2]
    day6_labels = cell_type_labels[day6_idx]

    for fate in target_fates:
        if fate not in fate_names:
            print(f"  WARNING: '{fate}' not in fate_names")
            continue
        fidx = fate_names.index(fate)
        mask = day6_labels == fidx
        if mask.sum() == 0:
            print(f"  WARNING: No day-6 cells found for fate '{fate}'")
            continue
        X_fate = dataset.X[day6_idx[mask]]
        z_fate = jax.vmap(lambda x: encode_mean(vae, x))(X_fate)
        attractors[fate] = jnp.mean(z_fate, axis=0)
        print(f"  Attractor '{fate}': {int(mask.sum())} day-6 cells  "
              f"|z|={float(jnp.linalg.norm(attractors[fate])):.3f}")
    return attractors


def run_validation(
    randers_vae: GeometricVAE,
    dataset: BioDataset,
    lineage_triples: jnp.ndarray,
    cell_type_labels: np.ndarray,
    fate_names: list,
    target_fates: list,
    key: jax.Array,
    n_pairs: int = 1000,
) -> Tuple[Dict, Dict]:

    riem_key      = jax.random.PRNGKey(77)
    riemannian_vae = build_riemannian_baseline(randers_vae, riem_key)

    print("\nBuilding fate attractors ...")
    attractors = build_fate_attractors(
        randers_vae, dataset,
        np.array(lineage_triples), cell_type_labels,
        fate_names, target_fates,
    )
    if len(attractors) < 2:
        raise ValueError("Need ≥ 2 fate attractors for counterfactual test.")

    # Shuffle and subsample
    rng = np.random.default_rng(int(key[0]))
    n   = min(n_pairs, len(lineage_triples))
    idx = rng.choice(len(lineage_triples), n, replace=False)
    triples = np.array(lineage_triples)[idx]

    X2 = dataset.X[triples[:, 0]]
    X4 = dataset.X[triples[:, 1]]
    X6 = dataset.X[triples[:, 2]]

    z2s = jax.vmap(lambda x: encode_mean(randers_vae, x))(X2)
    z4s = jax.vmap(lambda x: encode_mean(randers_vae, x))(X4)
    z6s = jax.vmap(lambda x: encode_mean(randers_vae, x))(X6)

    day6_labels = cell_type_labels[triples[:, 2]]

    # Accumulate per-model results
    raw = {
        'randers':    {'ratio': [], 'e_obs': [], 'e_cf': [], 'fate': []},
        'riemannian': {'ratio': [], 'e_obs': [], 'e_cf': [], 'fate': []},
    }

    fate_idx_map = {f: fate_names.index(f)
                    for f in target_fates if f in fate_names}

    for i in range(n):
        obs_fidx = day6_labels[i]
        obs_fname = fate_names[obs_fidx] if obs_fidx < len(fate_names) else None
        if obs_fname not in target_fates or obs_fname not in attractors:
            continue

        z2, z4, z6_obs = z2s[i], z4s[i], z6s[i]

        # Hardest counterfactual: closest wrong-fate attractor to z2
        cf_candidates = [(fn, fz) for fn, fz in attractors.items()
                         if fn != obs_fname]
        if not cf_candidates:
            continue
        z6_cf = min(cf_candidates,
                    key=lambda fz: float(jnp.linalg.norm(fz[1] - z2)))[1]

        for mname, model in [('randers', randers_vae),
                              ('riemannian', riemannian_vae)]:
            e_obs = float(two_segment_energy(model.metric, z2, z4, z6_obs))
            e_cf  = float(two_segment_energy(model.metric, z2, z4, z6_cf))

            if e_obs <= 0 or e_cf <= 0 or not np.isfinite(e_obs) or not np.isfinite(e_cf):
                continue

            raw[mname]['ratio'].append(e_cf / e_obs)
            raw[mname]['e_obs'].append(e_obs)
            raw[mname]['e_cf'].append(e_cf)
            raw[mname]['fate'].append(obs_fname)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    results = {}
    for mname in ['randers', 'riemannian']:
        r      = raw[mname]
        ratios = np.array(r['ratio'])
        fates  = np.array(r['fate'])
        if len(ratios) == 0:
            print(f"WARNING: no valid triples for {mname}")
            continue

        results[mname] = {
            'n':                   len(ratios),
            'energy_ratio_mean':   float(np.mean(ratios)),
            'energy_ratio_median': float(np.median(ratios)),
            'energy_ratio_std':    float(np.std(ratios)),
            # Fraction of triples where counterfactual is more expensive
            'correct_fate_frac':   float(np.mean(ratios > 1.0)),
            'mean_e_obs':          float(np.mean(r['e_obs'])),
            'mean_e_cf':           float(np.mean(r['e_cf'])),
        }
        for fate in target_fates:
            m = fates == fate
            if m.sum() > 0:
                results[mname][f'ratio_mean_{fate}']  = float(np.mean(ratios[m]))
                results[mname][f'correct_frac_{fate}'] = float(np.mean(ratios[m] > 1.0))

    # ── Statistical tests ─────────────────────────────────────────────────────
    if 'randers' in results and 'riemannian' in results:
        r_rand = np.array(raw['randers']['ratio'])
        r_riem = np.array(raw['riemannian']['ratio'])
        n_test = min(len(r_rand), len(r_riem))

        # Test 1: Randers ratios > Riemannian ratios (one-sided paired Wilcoxon)
        w_stat, w_pval = stats.wilcoxon(
            r_rand[:n_test], r_riem[:n_test], alternative='greater'
        )
        # Test 2: Randers ratios > 1.0 (one-sample t-test)
        t_stat, t_pval = stats.ttest_1samp(r_rand, popmean=1.0,
                                            alternative='greater')
        # Effect size: Cohen's d vs null (ratio=1)
        cohens_d = (np.mean(r_rand) - 1.0) / (np.std(r_rand) + 1e-8)

        results['stats'] = {
            'n_randers':           len(r_rand),
            'n_riemannian':        len(r_riem),
            'wilcoxon_stat':       float(w_stat),
            'wilcoxon_pvalue':     float(w_pval),
            'wilcoxon_sig_p05':    bool(w_pval < 0.05),
            'wilcoxon_sig_p01':    bool(w_pval < 0.01),
            'ttest_pvalue':        float(t_pval),
            'ttest_sig_p05':       bool(t_pval < 0.05),
            'cohens_d':            float(cohens_d),
        }

    return results, raw


def build_riemannian_baseline(vae: GeometricVAE, key: jax.Array) -> GeometricVAE:
    metric = PullbackRiemannian(vae.manifold, decoder=vae.decoder_net, key=key)
    return eqx.tree_at(lambda m: m.metric, vae, metric)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Visualization
# ══════════════════════════════════════════════════════════════════════════════

def save_wind_visualization(vae: GeometricVAE, dataset: BioDataset, save_path: str = "wind_latent_space.png"):
    Z = encode_all(vae, dataset.X)
    from sklearn.decomposition import PCA as skPCA
    pca2 = skPCA(n_components=2).fit(Z)
    z2d = pca2.transform(Z)

    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot cells
    scatter = ax.scatter(z2d[:, 0], z2d[:, 1], c=np.array(dataset.labels), s=6, alpha=0.5, cmap='Spectral')
    
    # Create grid
    x_min, x_max = np.percentile(z2d[:, 0], [1, 99])
    y_min, y_max = np.percentile(z2d[:, 1], [1, 99])
    gx = np.linspace(x_min, x_max, 25)
    gy = np.linspace(y_min, y_max, 25)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
    
    # Lift grid to full latent space
    grid_full = pca2.inverse_transform(grid).astype(np.float32)
    grid_full = grid_full[:, :vae.latent_dim]
    
    # Compute Wind on grid in full latent space
    try:
        W_full = np.array(jax.vmap(
            lambda z: vae.metric._get_zermelo_data(z)[1]
        )(jnp.array(grid_full)))
        
        # Project Wind back to 2D PCA space
        W_2d = np.dot(W_full, pca2.components_.T)
        
        # Quiver plot
        ax.quiver(grid[:, 0], grid[:, 1], W_2d[:, 0], W_2d[:, 1],
                  color='black', scale=15, width=0.003, alpha=0.8)
    except Exception as e:
        ax.text(0.5, 0.5, f"Wind plot failed:\n{e}", ha='center', va='center', transform=ax.transAxes)

    ax.set_title("Wind Field W(z) Trajectories on Latent PCA Space", fontsize=16)
    ax.set_xlabel("Latent PC1", fontsize=12)
    ax.set_ylabel("Latent PC2", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved standalone wind visualization → {save_path}")


def plot_results(
    vae: GeometricVAE,
    dataset: BioDataset,
    results: Dict,
    raw: Dict,
    target_fates: list,
    fate_names: list,
    phase2_history: Dict,
    save_path: str = "weinreb_results_v5.png",
):
    Z    = encode_all(vae, dataset.X)
    from sklearn.decomposition import PCA as skPCA
    pca2 = skPCA(n_components=2).fit(Z)
    z2d  = pca2.transform(Z)

    rand_ratios = np.array(raw['randers']['ratio'])
    riem_ratios = np.array(raw['riemannian']['ratio'])

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    axes = axes.ravel()

    fate_colors = ["steelblue", "tomato", "forestgreen", "darkorange"]

    # ── Panel 0: Latent space + wind field ────────────────────────────────────
    ax = axes[0]
    ax.scatter(z2d[:, 0], z2d[:, 1],
               c=np.array(dataset.labels), s=3, alpha=0.35, cmap='Spectral')
    x_range = np.percentile(z2d[:, 0], [5, 95])
    y_range = np.percentile(z2d[:, 1], [5, 95])
    gx = np.linspace(*x_range, 18)
    gy = np.linspace(*y_range, 18)
    GX, GY = np.meshgrid(gx, gy)
    grid   = np.stack([GX.ravel(), GY.ravel()], axis=1)
    grid_full = pca2.inverse_transform(grid).astype(np.float32)
    grid_full = grid_full[:, :vae.latent_dim]

    try:
        W_full = np.array(jax.vmap(
            lambda z: vae.metric._get_zermelo_data(z)[1]
        )(jnp.array(grid_full)))
        # Transform wind vectors to PCA space via dot product with components
        W_2d = np.dot(W_full, pca2.components_.T)
        
        ax.quiver(grid[:, 0], grid[:, 1], W_2d[:, 0], W_2d[:, 1],
                  color='black', scale=8, width=0.003, alpha=0.55)
    except Exception as e:
        ax.text(0.5, 0.5, f"Wind plot failed:\n{e}",
                ha='center', va='center', transform=ax.transAxes)
    ax.set_title("Latent space + Wind field W(z)", fontsize=11)
    ax.set_xlabel("PC1 of Z"); ax.set_ylabel("PC2 of Z")

    # ── Panel 1: Energy ratio distributions ───────────────────────────────────
    ax = axes[1]
    ax.hist(rand_ratios, bins=60, alpha=0.65, color='steelblue',
            label=f"Randers  μ={np.mean(rand_ratios):.3f}")
    ax.hist(riem_ratios, bins=60, alpha=0.65, color='coral',
            label=f"Riemannian  μ={np.mean(riem_ratios):.3f}")
    ax.axvline(1.0, color='black', linestyle='--', lw=1.5,
               label='Ratio=1  (no discrimination)')
    ax.axvline(np.mean(rand_ratios), color='steelblue', lw=2.0)
    ax.axvline(np.mean(riem_ratios), color='coral',     lw=2.0)
    ax.set_xlabel("E_counterfactual / E_observed", fontsize=10)
    ax.set_ylabel("Count")
    s = results.get('stats', {})
    wp = s.get('wilcoxon_pvalue', float('nan'))
    sig = '***' if wp < 0.001 else ('**' if wp < 0.01 else ('*' if wp < 0.05 else 'n.s.'))
    r   = results.get('randers', {})
    ax.set_title(
        f"Energy ratio  (Randers correct-fate frac: "
        f"{r.get('correct_fate_frac', float('nan')):.1%})\n"
        f"Wilcoxon p={wp:.4f} ({sig})",
        fontsize=10
    )
    ax.legend(fontsize=9)

    # ── Panel 2: Per-fate correct-fate fraction ────────────────────────────────
    ax = axes[2]
    fates_plot = [f for f in target_fates
                  if f'correct_frac_{f}' in results.get('randers', {})]
    rand_frac  = [results['randers'][f'correct_frac_{f}'] for f in fates_plot]
    riem_frac  = [results.get('riemannian', {}).get(f'correct_frac_{f}', 0.5)
                  for f in fates_plot]
    x_pos = np.arange(len(fates_plot))
    w     = 0.35
    ax.bar(x_pos - w/2, rand_frac, w, label='Randers',    color='steelblue', alpha=0.8)
    ax.bar(x_pos + w/2, riem_frac, w, label='Riemannian', color='coral',     alpha=0.8)
    ax.axhline(0.5, color='black', linestyle='--', lw=1.2, label='Chance (0.5)')
    ax.set_xticks(x_pos); ax.set_xticklabels(fates_plot, rotation=15, ha='right')
    ax.set_ylabel("Fraction  E_cf > E_obs")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-fate: does metric favour correct fate?", fontsize=11)
    ax.legend(fontsize=9)

    # ── Panel 3: Paired scatter Randers vs Riemannian ratio ───────────────────
    ax = axes[3]
    n_sc = min(len(rand_ratios), len(riem_ratios))
    fate_arr = np.array(raw['randers']['fate'][:n_sc])
    for fi, fname in enumerate(target_fates):
        m = fate_arr == fname
        if m.sum() > 0:
            ax.scatter(riem_ratios[:n_sc][m], rand_ratios[:n_sc][m],
                       s=6, alpha=0.4, color=fate_colors[fi % len(fate_colors)],
                       label=fname)
    lim = max(np.percentile(np.concatenate([rand_ratios, riem_ratios]), 97), 1.5)
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='Equal')
    ax.axhline(1.0, color='steelblue', lw=0.8, linestyle=':')
    ax.axvline(1.0, color='coral',     lw=0.8, linestyle=':')
    ax.set_xlabel("Riemannian energy ratio"); ax.set_ylabel("Randers energy ratio")
    ax.set_title("Paired comparison\n(above diagonal = Randers more discriminative)",
                 fontsize=10)
    ax.legend(fontsize=9)

    # ── Panel 4: Phase 2 training curves ─────────────────────────────────────
    ax = axes[4]
    if phase2_history is not None:
        ep = np.arange(len(phase2_history['total']))
        ax.plot(ep, phase2_history['total'],  color='steelblue', lw=1.5, label='total')
        ax.plot(ep, phase2_history['vel'],    color='tomato',    lw=1.5, label='vel align')
        ax.plot(ep, phase2_history['smooth'], color='forestgreen', lw=1.5, label='smooth')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.set_title("Phase 2 training — wind loss curves", fontsize=11)

    # ── Panel 5: |W|_H evolution during phase 2 ──────────────────────────────
    ax = axes[5]
    if phase2_history is not None:
        wn = phase2_history['w_norm']
        wn_ep = np.linspace(0, max(1, len(phase2_history['total']) - 1), len(wn))
        ax.plot(wn_ep, wn, color='darkorange', lw=2.0, marker='o', markersize=4)
    ax.axhline(0.95, color='red', linestyle='--', lw=1.2,
               label='Singularity threshold (0.95)')
    ax.axhline(0.0,  color='gray', linestyle=':', lw=1.0)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean |W|_H")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("Wind field magnitude during training\n"
                 "(must stay < 0.95 — Randers constraint)", fontsize=10)
    ax.legend(fontsize=9)

    plt.suptitle("Weinreb Experiment v5 — Randers Energy Validation", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"\nSaved figure → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TRIPLES      = "data/weinreb_lineage_triples.npy"
    LATENT_DIM   = 8

    for p in [CHECKPOINT, PREPROCESSED, TRIPLES]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found.\n"
                "Run weinreb_vae_diagnostic.py first and ensure it saves:\n"
                "  eqx.tree_serialise_leaves('data/weinreb_vae_phase1.eqx', vae)"
            )

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data ...")
    adata = anndata.read_h5ad(PREPROCESSED)

    X_pca = np.array(adata.obsm['X_pca'],        dtype=np.float32)
    V_pca = np.array(adata.obsm['velocity_pca'], dtype=np.float32)

    ct_series  = adata.obs['Cell type annotation'].astype('category')
    fate_names = list(ct_series.cat.categories)
    labels_np  = ct_series.cat.codes.values.astype(np.int32)
    n_types    = len(fate_names)

    # Same normalisation as phase 1 — MUST be identical
    print("Applying StandardScaler (same as phase 1) ...")
    scaler   = StandardScaler()
    X_norm   = scaler.fit_transform(X_pca).astype(np.float32)
    V_norm   = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)

    dataset = BioDataset(
        X      = jnp.array(X_norm),
        V      = jnp.array(V_norm),
        labels = jnp.array(labels_np),
        lineage_pairs=None,
    )

    lineage_triples = jnp.array(np.load(TRIPLES))

    print(f"Cells:            {X_norm.shape[0]}")
    print(f"Features (PCA):   {X_norm.shape[1]}")
    print(f"Cell types:       {n_types}  → {fate_names}")
    print(f"Lineage triples:  {len(lineage_triples)}")
    print(f"Nonzero velocity: "
          f"{float(jnp.mean(jnp.any(dataset.V != 0, axis=1))):.1%}")

    # ── Load phase 1 checkpoint ───────────────────────────────────────────────
    print(f"\nLoading phase-1 VAE checkpoint ...")
    key = jax.random.PRNGKey(2026)
    vae_p1 = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM, n_types, key)

    # ── Attach Data-Driven Randers metric ─────────────────────────────────────
    print("Attaching Data-Driven PullbackRanders metric using kernel smoothed RNA velocities ...")
    vae_randers = attach_datadriven_randers_metric(vae_p1, dataset, n_anchors=2000, sigma=0.4)

    # Verify |W|_H before validation
    sample_z = jax.vmap(lambda x: encode_mean(vae_randers, x))(dataset.X[:200])
    def wn_at(z):
        H, W, _ = vae_randers.metric._get_zermelo_data(z)
        return jnp.sqrt(jnp.dot(W, jnp.dot(H, W)))
    final_wn = float(np.mean(np.array(jax.vmap(wn_at)(sample_z))))
    print(f"  Mean |W|_H = {final_wn:.4f}  (must be < 0.95)")
    if final_wn >= 0.95:
        print("  WARNING: |W|_H near singularity — consider tuning sigma.")

    # Skip Phase 2 network training entirely
    vae_trained = vae_randers
    p2_history = None

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("VALIDATION: Fate-stratified Randers energy comparison")
    print("="*60)
    val_key = jax.random.PRNGKey(7)
    results, raw = run_validation(
        vae_trained, dataset,
        lineage_triples, labels_np,
        fate_names, TARGET_FATES,
        val_key, n_pairs=1000,
    )

    # ── Print results ──────────────────────────────────────────────────────────
    print("\nRESULTS:")
    for mname in ['randers', 'riemannian']:
        if mname not in results:
            continue
        print(f"\n  {mname.upper()}:")
        for k, v in results[mname].items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print("\n  STATISTICAL TESTS:")
    s = results.get('stats', {})
    for k, v in s.items():
        print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    s        = results.get('stats', {})
    w_sig    = s.get('wilcoxon_sig_p05', False)
    t_sig    = s.get('ttest_sig_p05', False)
    cf_frac  = results.get('randers', {}).get('correct_fate_frac', 0.5)
    ratio_m  = results.get('randers', {}).get('energy_ratio_mean', 1.0)
    cohens_d = s.get('cohens_d', 0.0)

    if w_sig or t_sig:
        print("✓ HYPOTHESIS SUPPORTED")
        print(f"  Randers metric assigns higher energy to wrong-fate paths")
        print(f"  correct_fate_frac = {cf_frac:.1%}  (chance = 50%)")
        print(f"  energy_ratio_mean = {ratio_m:.3f}  (null = 1.0)")
        print(f"  Cohen's d         = {cohens_d:.3f}")
        if s.get('wilcoxon_sig_p01'):
            print("  Strong evidence: Wilcoxon p < 0.01")
    else:
        print("✗ HYPOTHESIS NOT SUPPORTED at p < 0.05")
        print(f"  correct_fate_frac = {cf_frac:.1%}")
        print(f"  energy_ratio_mean = {ratio_m:.3f}")
        print(f"  Cohen's d         = {cohens_d:.3f}")
        print("\n  Diagnostic guidance:")
        if cf_frac > 0.55:
            print("  → Directional signal present (cf_frac > 0.55) but")
            print("    underpowered. Try n_pairs=2000 or more phase 2 epochs.")
        if final_wn < 0.1:
            print("  → |W|_H is very small — wind barely deforms the metric.")
            print("    Try increasing RNAVelocityWindLoss weight to 2.0.")
        if ratio_m < 1.02:
            print("  → Energy ratio ≈ 1. The metric is nearly symmetric.")
            print("    Check whether w_net weights updated (gradient flow).")
    print("="*60)

    plot_results(
        vae_trained, dataset, results, raw,
        TARGET_FATES, fate_names, p2_history,
        save_path="weinreb_results_v5.png",
    )
    
    save_wind_visualization(vae_trained, dataset, save_path="wind_latent_space.png")


if __name__ == "__main__":
    main()