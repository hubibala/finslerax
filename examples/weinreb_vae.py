"""
weinreb_vae_diagnostic.py
==========================
Focused diagnostic experiment: get the VAE geometry right FIRST.

Goal: produce a latent space where
  - Monocyte / Neutrophil / other fates cluster visibly
  - The pullback metric G = JᵀJ varies smoothly (no degeneracies)
  - Day-2 → day-4 → day-6 arcs are coherent, not scattered blobs

What is NOT in this file (deliberately):
  - No W / wind field (Phase 2 from the full experiment)
  - No geodesic / AVBD solver calls during training
  - No Randers metric — PullbackRiemannian only

Fixes applied vs. the original codebase:
  1. Decoder uses tanh activations throughout — prerequisite for a smooth,
     everywhere-differentiable Jacobian (ReLU / GeLU have kinks at 0).
  2. KL annealing — cyclical β schedule prevents posterior collapse while
     still regularising the latent space.
  3. Cluster-separation: neighbourhood triplet loss built from the KNN
     graph in PCA space (gene-space topology), NOT the current latent space.
     This directly requests cluster structure without label supervision.
  4. Optional weak label supervision (cross-entropy on cell-type codes).
  5. TrajectoryCoherenceLoss fixed: per-sample normalisation, not global.
  6. Deterministic encoding (encoder mean only) for all geometric losses —
     stochastic sampling only in the reconstruction term.
  7. Visualisation: 6-panel figure covering latent scatter, metric
     determinant heatmap, KNN graph preservation, trajectory arcs,
     per-cluster silhouette, and loss curves.

Usage
-----
    python weinreb_vae_diagnostic.py

Requires:
    data/weinreb_preprocessed.h5ad
    data/weinreb_lineage_triples.npy
    (produced by preprocess_weinreb_spring.py)
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import anndata

from ham.bio.data import BioDataset
from ham.geometry.surfaces import EuclideanSpace
from ham.models.learned import PullbackRiemannian
from ham.training.losses import LossComponent
from ham.geometry.zoo import Euclidean


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Encoder / Decoder with tanh activations
#     The decoder MUST use tanh (or softplus) throughout for the Jacobian
#     J = d(decode)/dz to be smooth and well-defined everywhere.
#     GeLU has a kink at the origin; ReLU is not differentiable at 0.
# ══════════════════════════════════════════════════════════════════════════════

def make_tanh_mlp(in_dim: int, out_dim: int, hidden: int, depth: int,
                  key: jax.Array) -> eqx.nn.MLP:
    """MLP with tanh activations — smooth Jacobian guaranteed."""
    return eqx.nn.MLP(
        in_dim, out_dim, hidden, depth,
        activation=jax.nn.tanh,          # ← key change
        key=key,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  KNN triplet loss — built from gene-space (PCA) topology
#
#     For each anchor cell a, we find:
#       positive p  = a neighbour of a in PCA space (same local structure)
#       negative n  = a cell from a DIFFERENT cluster (far in PCA space)
#
#     Then we push:   d_lat(a, p) + margin < d_lat(a, n)
#
#     This topology-preserving loss is the direct mechanism to request
#     cluster structure without hard label supervision.
# ══════════════════════════════════════════════════════════════════════════════

def build_knn_triplet_indices(X_pca: np.ndarray, labels: np.ndarray,
                               k: int = 15, n_triplets: int = 20_000,
                               seed: int = 0) -> np.ndarray:
    """
    Pre-compute (anchor, positive, negative) index triples in PCA space.
    Returns array of shape (n_triplets, 3).
    """
    rng = np.random.default_rng(seed)
    print(f"  Building KNN graph (k={k}) on {X_pca.shape[0]} cells ...")
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean",
                            algorithm="ball_tree", n_jobs=-1).fit(X_pca)
    distances, indices = nbrs.kneighbors(X_pca)
    # indices[:, 0] is self — drop it
    knn_indices = indices[:, 1:]          # (N, k)

    n = X_pca.shape[0]
    triplets = []
    attempts = 0
    max_attempts = n_triplets * 10

    while len(triplets) < n_triplets and attempts < max_attempts:
        anchors = rng.integers(0, n, size=512)
        for a in anchors:
            # positive: random KNN neighbour
            p = knn_indices[a, rng.integers(0, k)]
            # negative: random cell with a different label
            neg_pool = np.where(labels != labels[a])[0]
            if len(neg_pool) == 0:
                continue
            neg = rng.choice(neg_pool)
            triplets.append((a, p, neg))
            if len(triplets) >= n_triplets:
                break
        attempts += 512

    triplets = np.array(triplets[:n_triplets], dtype=np.int32)
    print(f"  Generated {len(triplets)} KNN triplets.")
    return triplets


class KNNTripletLoss(LossComponent):
    """
    Topology-preserving triplet loss.
    Margin is applied in Euclidean latent distance — no metric needed.
    batch[0] = anchor PCA coords
    batch[1] = positive PCA coords   (KNN neighbour)
    batch[2] = negative PCA coords   (different-label cell)
    """
    margin: float = eqx.field(static=True)

    def __init__(self, weight: float = 1.0, margin: float = 1.0):
        super().__init__(weight, "KNNTriplet")
        self.margin = margin

    def __call__(self, model, batch, key):
        xa, xp, xn = batch[0], batch[1], batch[2]

        v_get_dist = eqx.filter_vmap(model._get_dist)
        # Use deterministic encoder mean — no sampling noise here
        za = v_get_dist(xa).mean
        zp = v_get_dist(xp).mean
        zn = v_get_dist(xn).mean

        da = jnp.sum((za - zp) ** 2, axis=-1)
        dn = jnp.sum((za - zn) ** 2, axis=-1)

        loss = jax.nn.relu(da - dn + self.margin)
        return jnp.mean(loss) * self.weight


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Trajectory coherence loss (fixed)
#
#     Original bug: direction coherence normalised GLOBAL batch vectors,
#     not per-sample vectors — the dot product was therefore meaningless.
#     Fix: per-sample normalise before dot product.
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryCoherenceLoss(LossComponent):
    """
    Per-triple (day2, day4, day6) coherence:
      1. Midpoint: z4 ≈ 0.5*(z2 + z6)
      2. Direction: (z4-z2) · (z6-z2) > 0, normalised per sample.

    batch[0] = x_day2, batch[3] = x_day4, batch[4] = x_day6
    """
    def __init__(self, weight: float = 0.5):
        super().__init__(weight, "TrajCoherence")

    def __call__(self, model, batch, key):
        if len(batch) < 5 or batch[3] is None or batch[4] is None:
            return jnp.array(0.0)

        x2, x4, x6 = batch[0], batch[3], batch[4]
        v_get_dist = eqx.filter_vmap(model._get_dist)
        z2 = v_get_dist(x2).mean
        z4 = v_get_dist(x4).mean
        z6 = v_get_dist(x6).mean

        # 1. Midpoint coherence (per sample, then mean)
        z_mid    = 0.5 * (z2 + z6)
        mid_loss = jnp.mean(jnp.sum((z4 - z_mid) ** 2, axis=-1))

        # 2. Direction coherence — FIX: normalise per sample
        v_early = z4 - z2                                          # (B, D)
        v_full  = z6 - z2                                          # (B, D)
        norm_early = jnp.sqrt(jnp.sum(v_early ** 2, axis=-1, keepdims=True) + 1e-8)
        norm_full  = jnp.sqrt(jnp.sum(v_full  ** 2, axis=-1, keepdims=True) + 1e-8)
        v_en    = v_early / norm_early
        v_fn    = v_full  / norm_full
        dir_loss = jnp.mean(1.0 - jnp.sum(v_en * v_fn, axis=-1))  # (B,) → scalar

        return (mid_loss + dir_loss) * self.weight


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Cyclic KL annealing
#
#     β(epoch) follows a saw-tooth that resets every `cycle_len` epochs.
#     The key effect: early in each cycle β ≈ 0 so the encoder is free to
#     build cluster structure; late in the cycle β → β_max which regularises.
#     This prevents posterior collapse while preserving geometry.
# ══════════════════════════════════════════════════════════════════════════════

def cyclic_beta(epoch: int, cycle_len: int = 20,
                beta_min: float = 0.0, beta_max: float = 5e-4) -> float:
    phase = (epoch % cycle_len) / cycle_len   # 0 → 1 within each cycle
    # linear ramp for first 50% of cycle, flat at max for the rest
    ramp = min(phase * 2.0, 1.0)
    return beta_min + (beta_max - beta_min) * ramp


class AnnealedKLLoss(LossComponent):
    """KL loss whose weight is provided dynamically via __call__."""

    def __init__(self, beta_max: float = 5e-4):
        super().__init__(beta_max, "AnnealedKL")

    def __call__(self, model, batch, key, beta: float = 0.0):
        x    = batch[0]
        v_get_dist = eqx.filter_vmap(model._get_dist)
        dist = v_get_dist(x)
        return jnp.mean(dist.kl_divergence_std_normal()) * beta


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Lightweight cell-type classifier head
#     Weak supervision: cross-entropy on integer cell-type codes.
#     Acts on encoder MEAN to avoid label noise from sampling.
# ══════════════════════════════════════════════════════════════════════════════

class CellTypeClassificationLoss(LossComponent):
    n_classes: int = eqx.field(static=True)

    def __init__(self, n_classes: int, weight: float = 0.15):
        super().__init__(weight, "CellTypeClass")
        self.n_classes = n_classes

    def __call__(self, model, batch, key):
        x      = batch[0]
        labels = batch[2].astype(int)
        v_get_dist = eqx.filter_vmap(model._get_dist)
        z      = v_get_dist(x).mean
        v_classifier = eqx.filter_vmap(model.classifier_head)
        logits = v_classifier(z)
        log_p  = jax.nn.log_softmax(logits, axis=-1)
        oh     = jax.nn.one_hot(labels, self.n_classes)
        return -jnp.mean(jnp.sum(oh * log_p, axis=-1)) * self.weight


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Reconstruction loss — deterministic path for geometry,
#     stochastic path retained to avoid posterior collapse
# ══════════════════════════════════════════════════════════════════════════════

class VelocityConsistencyLoss(LossComponent):
    """
    For KNN pairs with similar velocities in PCA space,
    their latent velocities should also be similar.
    This regularizes the encoder Jacobian to preserve
    velocity directions smoothly across the latent space.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "VelConsistency")

    def __call__(self, model, batch, key):
        x_i, x_j = batch[0], batch[1]  # KNN neighbors
        u_i, u_j = batch[2], batch[3]  # their velocities
        
        def safe_norm(v):
            return jnp.sqrt(jnp.sum(v ** 2, axis=-1) + 1e-8)
        
        # Only for pairs with similar velocity directions
        cos_sim_data = jnp.sum(u_i * u_j, axis=-1) / (safe_norm(u_i) * safe_norm(u_j))
        
        # Push forward both velocities
        v_proj = eqx.filter_vmap(model.project_control)
        _, v_lat_i = v_proj(x_i, u_i)
        _, v_lat_j = v_proj(x_j, u_j)
        
        # Their latent velocities should be similar
        cos_sim_lat = jnp.sum(v_lat_i * v_lat_j, axis=-1) / (safe_norm(v_lat_i) * safe_norm(v_lat_j))
        
        # Penalize inconsistency weighted by data-space similarity
        weight = jax.nn.relu(cos_sim_data)
        
        # Also zero out loss for cells with zero velocity to avoid noise
        u_i_norm_sq = jnp.sum(u_i ** 2, axis=-1)
        valid_mask = (u_i_norm_sq > 1e-6)
        
        loss_vals = weight * (1.0 - cos_sim_lat)
        return jnp.mean(jnp.where(valid_mask, loss_vals, 0.0)) * self.weight


class ReconstructionLossDeterministic(LossComponent):
    """
    Uses BOTH the sampled z (for the ELBO) and the mean z (for geometric
    stability). The mean-path reconstruction anchors the decoder geometry
    so the pullback metric G = JᵀJ is not noisy.
    """
    stochastic_weight: float = eqx.field(static=True)
    deterministic_weight: float = eqx.field(static=True)

    def __init__(self, weight: float = 1.0,
                 stochastic_weight: float = 0.5,
                 deterministic_weight: float = 0.5):
        super().__init__(weight, "Recon")
        self.stochastic_weight  = stochastic_weight
        self.deterministic_weight = deterministic_weight

    def __call__(self, model, batch, key):
        x    = batch[0]
        v_get_dist = eqx.filter_vmap(model._get_dist)
        dist = v_get_dist(x)

        # Stochastic path (necessary for proper ELBO)
        keys = jax.random.split(key, x.shape[0])
        z_sample = jax.vmap(lambda d, k: d.sample(k))(dist, keys)
        v_decode = eqx.filter_vmap(model.decode)
        loss_stoch = jnp.mean((x - v_decode(z_sample)) ** 2)

        # Deterministic path (anchors decoder geometry)
        z_mean = dist.mean
        loss_det  = jnp.mean((x - v_decode(z_mean))  ** 2)

        return (self.stochastic_weight  * loss_stoch
                + self.deterministic_weight * loss_det) * self.weight


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Model builder
#     - encoder: gelu (standard, expressiveness matters here)
#     - decoder: tanh ONLY (smooth Jacobian for pullback metric)
#     - metric : PullbackRiemannian (pure G = JᵀJ, no wind yet)
# ══════════════════════════════════════════════════════════════════════════════

def build_diagnostic_vae(data_dim: int, latent_dim: int,
                          n_cell_types: int, key: jax.Array):
    """
    Returns a GeometricVAE with:
      - tanh decoder (smooth Jacobian for pullback metric at eval time)
      - gelu encoder (expressiveness)
      - EuclideanSpace metric during training (NO jacfwd in forward pass)
        The PullbackRiemannian is only instantiated after training for
        evaluation — running jacfwd inside every training step is
        extremely expensive and causes NaN on random init weights.
      - classifier_head with small weight init (Fix C)
    """
    from ham.bio.vae import GeometricVAE

    k1, k2, k3, k_cls = jax.random.split(key, 4)
    manifold = EuclideanSpace(dim=latent_dim)

    # Encoder: gelu — standard for expressiveness
    encoder_net = eqx.nn.MLP(
        data_dim, 2 * latent_dim, 256, 4,
        activation=jax.nn.gelu, key=k1
    )

    # Decoder: tanh — mandatory for smooth Jacobian at eval time
    decoder_net = make_tanh_mlp(latent_dim, data_dim, 256, 4, key=k2)

    metric = Euclidean(manifold)

    vae = GeometricVAE(
        data_dim, latent_dim, metric,
        key=k3,
        encoder_net=encoder_net,
        decoder_net=decoder_net,
    )

    # FIX C: small init on classifier — large random logits → -inf after
    # log_softmax on the first forward pass → NaN cross-entropy loss.
    classifier = eqx.nn.Linear(latent_dim, n_cell_types, key=k_cls)
    # Scale down weights by 0.01 so initial logits are O(0.01) not O(1)
    classifier = eqx.tree_at(
        lambda c: c.weight,
        classifier,
        classifier.weight * 0.01
    )
    vae = eqx.tree_at(
        lambda m: m.classifier_head, vae, classifier,
        is_leaf=lambda x: x is None
    )
    return vae


def attach_pullback_metric(vae, key: jax.Array):
    """
    Call AFTER training. Swaps in PullbackRiemannian so the trained
    decoder's Jacobian defines the metric for evaluation and geodesics.
    """
    manifold = vae.manifold
    metric   = PullbackRiemannian(manifold, decoder=vae.decoder_net, key=key)
    return eqx.tree_at(lambda m: m.metric, vae, metric)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Training loop — manual epoch loop so we can update β each epoch
# ══════════════════════════════════════════════════════════════════════════════

def get_trainable_mask(model):
    """Trainable = everything except metric.w_net (doesn't exist here, but future-safe)."""
    base = jax.tree_util.tree_map(lambda _: False, model)
    def make_true(node):
        return jax.tree_util.tree_map(
            lambda l: True if eqx.is_array(l) else False, node)
    targets = (model.encoder_net, model.decoder_net, model.classifier_head)
    masks   = tuple(make_true(t) for t in targets)
    return eqx.tree_at(
        lambda m: (m.encoder_net, m.decoder_net, m.classifier_head),
        base, replace=masks
    )


def train_vae(
    dataset: "BioDataset",
    triplet_indices: np.ndarray,
    lineage_triples: np.ndarray,
    labels_np: np.ndarray,
    n_cell_types: int,
    key: jax.Array,
    latent_dim: int = 8,
    epochs: int = 120,
    batch_size: int = 512,
    kl_cycle_len: int = 20,
    kl_beta_max: float = 5e-4,
    triplet_weight: float = 1.0,
    triplet_margin: float = 1.0,
    coherence_weight: float = 0.3,
    cls_weight: float = 0.15,
    vel_weight: float = 1.0,
):
    data_dim   = dataset.X.shape[0] if hasattr(dataset.X, "shape") else dataset.X.shape[1]
    # Defensive: handle (cells, features) layout
    if len(dataset.X.shape) == 2:
        data_dim = dataset.X.shape[1]

    print(f"Building VAE: data_dim={data_dim}, latent_dim={latent_dim}, "
          f"n_types={n_cell_types}")
    vae = build_diagnostic_vae(data_dim, latent_dim, n_cell_types, key)

    # ── Loss components ───────────────────────────────────────────────────────
    recon_loss = ReconstructionLossDeterministic(weight=1.0)
    kl_loss    = AnnealedKLLoss(beta_max=kl_beta_max)
    trip_loss  = KNNTripletLoss(weight=triplet_weight, margin=triplet_margin)
    coh_loss   = TrajectoryCoherenceLoss(weight=coherence_weight)
    cls_loss   = CellTypeClassificationLoss(n_classes=n_cell_types, weight=cls_weight)
    vel_loss   = VelocityConsistencyLoss(weight=vel_weight)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer  = optax.adamw(1e-3, weight_decay=1e-5)
    mask       = get_trainable_mask(vae)
    diff_model, static_model = eqx.partition(vae, mask)
    opt_state  = optimizer.init(diff_model)

    # ── Helper to assemble batch ───────────────────────────────────────────────
    X  = dataset.X
    labels = jnp.array(labels_np)

    # ── JIT-compiled step ─────────────────────────────────────────────────────
    @eqx.filter_jit
    def train_step(diff, static, opt_state, batch_main, batch_trip, batch_vel,
                   beta: float, step_key: jax.Array):
        def total_loss(dm):
            full = eqx.combine(dm, static)
            k1, k2, k3, k4, k5, k6 = jax.random.split(step_key, 6)

            # batch_main: (x, v, labels, x_day4, x_day6)
            l_recon = recon_loss(full, batch_main, k1)

            l_kl    = kl_loss(full, batch_main, k2, beta=beta)

            l_cls   = cls_loss(full, batch_main, k3)
            l_coh   = coh_loss(full, batch_main, k4)

            # batch_trip: (anchor, positive, negative)
            l_trip  = trip_loss(full, batch_trip, k5)
            
            # batch_vel: (anchor, positive, v_anchor, v_positive)
            l_vel   = vel_loss(full, batch_vel, k6)

            total   = l_recon + l_kl + l_cls + l_coh + l_trip + l_vel
            return total, {
                "recon": l_recon, "kl": l_kl, "cls": l_cls,
                "coh": l_coh, "triplet": l_trip, "vel_consist": l_vel
            }

        (loss, stats), grads = eqx.filter_value_and_grad(
            total_loss, has_aux=True)(diff)
        updates, new_opt = optimizer.update(grads, opt_state, diff)
        new_diff = eqx.apply_updates(diff, updates)
        return new_diff, new_opt, loss, stats

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {k: [] for k in ["total", "recon", "kl", "cls", "coh", "triplet", "vel_consist"]}
    n_cells    = X.shape[0]
    n_trip_total = triplet_indices.shape[0]
    n_lin      = lineage_triples.shape[0]

    for epoch in range(epochs):
        beta = cyclic_beta(epoch, kl_cycle_len, 0.0, kl_beta_max)
        key, sk = jax.random.split(key)

        # Shuffle main data
        perm_main = np.random.permutation(n_cells)
        # Shuffle triplet data
        perm_trip = np.random.permutation(n_trip_total)
        # Shuffle lineage triples (for coherence loss)
        perm_lin  = np.random.permutation(n_lin)

        steps     = max(1, n_cells // batch_size)
        ep_stats  = {k: 0.0 for k in history}

        for step in range(steps):
            idx_main = perm_main[step * batch_size : (step + 1) * batch_size]

            # Lineage triples batch (pad/repeat if smaller than batch_size)
            idx_lin  = perm_lin[step * batch_size % n_lin :
                                 (step * batch_size % n_lin) + batch_size]
            if len(idx_lin) < batch_size:
                idx_lin = np.concatenate([
                    idx_lin,
                    perm_lin[:(batch_size - len(idx_lin))]
                ])
            trip3    = lineage_triples[idx_lin]
            i2, i4, i6 = trip3[:, 0], trip3[:, 1], trip3[:, 2]

            batch_main = (
                X[idx_main],          # x
                dataset.V[idx_main],  # v (not used by these losses but kept for compat)
                labels[idx_main],     # cell-type labels
                X[i4],                # x_day4  (for coherence)
                X[i6],                # x_day6  (for coherence)
            )

            # KNN triplet batch
            idx_trip = perm_trip[step * batch_size % n_trip_total :
                                  (step * batch_size % n_trip_total) + batch_size]
            if len(idx_trip) < batch_size:
                idx_trip = np.concatenate([
                    idx_trip,
                    perm_trip[:(batch_size - len(idx_trip))]
                ])
            ta, tp, tn = (triplet_indices[idx_trip, 0],
                          triplet_indices[idx_trip, 1],
                          triplet_indices[idx_trip, 2])
            batch_trip = (X[ta], X[tp], X[tn])
            
            batch_vel = (
                X[ta], X[tp],
                dataset.V[ta], dataset.V[tp]
            )

            sk, step_key = jax.random.split(sk)
            diff_model, opt_state, loss, stats = train_step(
                diff_model, static_model, opt_state,
                batch_main, batch_trip, batch_vel, beta, step_key
            )

            ep_stats["total"] += float(loss)
            for k, v in stats.items():
                ep_stats[k] += float(v)

        for k in ep_stats:
            history[k].append(ep_stats[k] / steps)

        if epoch % 10 == 0 or epoch == epochs - 1:
            s = " | ".join(f"{k}: {history[k][-1]:.4f}" for k in history)
            print(f"  Epoch {epoch:03d}  β={beta:.2e}  |  {s}")
            if epoch == 0 and history["vel_consist"][-1] == 0.0:
                print("\n  [WARNING] VelocityConsistencyLoss is exactly 0.0 at epoch 0!")
                print("            This means velocity supervision is dead OR all velocities were zero.\n")

    vae = eqx.combine(diff_model, static_model)
    return vae, history


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Diagnostics — encode the full dataset once
# ══════════════════════════════════════════════════════════════════════════════

def encode_all(vae, X: jnp.ndarray) -> np.ndarray:
    """Batch-encode X to latent means, returns numpy (N, D)."""
    # Use eqx.filter_vmap and pass vae explicitly to avoid closure hashing
    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0))
    def batch_encode(v, x):

        return v._get_dist(x).mean

    batch = 1024
    parts = []
    for start in range(0, X.shape[0], batch):
        x_b = X[start : start + batch]
        z_b = batch_encode(vae, x_b)
        parts.append(np.array(z_b))



    return np.concatenate(parts, axis=0)



def compute_pullback_det(vae, Z: np.ndarray, n_grid: int = 20) -> tuple:
    """
    Evaluate log det G(z) on a 2D grid spanning the latent space.
    Uses the first two principal latent dimensions and pads with zeros.
    Returns (grid_x, grid_y, log_det_grid).
    """
    pca2 = PCA(n_components=2).fit(Z)
    z2d  = pca2.transform(Z)

    x_range = np.percentile(z2d[:, 0], [2, 98])
    y_range = np.percentile(z2d[:, 1], [2, 98])
    gx = np.linspace(*x_range, n_grid)
    gy = np.linspace(*y_range, n_grid)
    GX, GY = np.meshgrid(gx, gy)
    pts2d  = np.stack([GX.ravel(), GY.ravel()], axis=1)

    # Project back to full latent dim via PCA inverse
    latent_dim = Z.shape[1]
    pts_full   = pca2.inverse_transform(pts2d).astype(np.float32)
    # Clip to latent_dim
    pts_full   = pts_full[:, :latent_dim]

    @eqx.filter_jit
    def logdet_at(v_mod, z):
        J   = jax.jacfwd(v_mod.decode)(z)
        G   = jnp.dot(J.T, J) + 1e-6 * jnp.eye(latent_dim)
        sign, ld = jnp.linalg.slogdet(G)
        return jnp.where(sign > 0, ld, jnp.array(-20.0))

    log_dets = []
    for z in pts_full:
        log_dets.append(float(logdet_at(vae, jnp.array(z))))
    log_det_grid = np.array(log_dets).reshape(n_grid, n_grid)


    return GX, GY, log_det_grid, pca2


def knn_preservation_score(Z: np.ndarray, X: np.ndarray, k: int = 15) -> float:
    """
    Fraction of k-NN neighbours preserved between PCA space (X) and latent (Z).
    Higher = latent space preserved local topology.
    """
    nbrs_x = NearestNeighbors(n_neighbors=k + 1).fit(X)
    nbrs_z = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, ix  = nbrs_x.kneighbors(X)
    _, iz  = nbrs_z.kneighbors(Z)
    ix = ix[:, 1:]     # drop self
    iz = iz[:, 1:]
    preserved = sum(
        len(set(ix[i]) & set(iz[i])) / k
        for i in range(len(Z))
    ) / len(Z)
    return preserved


# ══════════════════════════════════════════════════════════════════════════════
# 10.  6-Panel diagnostic figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_diagnostics(
    vae,
    X_pca: np.ndarray,
    Z: np.ndarray,
    labels_np: np.ndarray,
    fate_names: list,
    lineage_triples: np.ndarray,
    history: dict,
    target_fates: list,
    save_path: str = "weinreb_vae_diagnostic.png",
):
    """
    6-panel figure:
      [0] Latent space coloured by cell type
      [1] Latent space coloured by day (if day info available, else by label)
      [2] Pullback metric log det G heatmap
      [3] Example trajectory arcs for target fates
      [4] Silhouette score per cluster
      [5] Training loss curves
    """
    pca2 = PCA(n_components=2).fit(Z)
    z2d  = pca2.transform(Z)

    unique_labels = np.unique(labels_np)
    n_types = len(unique_labels)
    cmap    = plt.cm.get_cmap("tab20", n_types)

    fig = plt.figure(figsize=(22, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # ── Panel 0: latent coloured by cell type ────────────────────────────────
    ax = axes[0]
    for i, lbl in enumerate(unique_labels):
        mask = labels_np == lbl
        name = fate_names[lbl] if lbl < len(fate_names) else str(lbl)
        ax.scatter(z2d[mask, 0], z2d[mask, 1], s=3, alpha=0.35,
                   color=cmap(i), label=name)
    ax.set_title("Latent space — cell type", fontsize=11)
    if n_types <= 12:
        ax.legend(markerscale=3, fontsize=6, loc="upper right",
                  ncol=2 if n_types > 8 else 1)
    ax.set_xlabel("PC1 of Z"); ax.set_ylabel("PC2 of Z")

    # ── Panel 1: target fates highlighted ────────────────────────────────────
    ax = axes[1]
    ax.scatter(z2d[:, 0], z2d[:, 1], s=2, alpha=0.15, color="lightgray")
    fate_colors = ["steelblue", "tomato", "forestgreen", "darkorange"]
    for fi, fname in enumerate(target_fates):
        if fname not in fate_names:
            continue
        fidx = fate_names.index(fname)
        mask = labels_np == fidx
        ax.scatter(z2d[mask, 0], z2d[mask, 1], s=5, alpha=0.6,
                   color=fate_colors[fi % len(fate_colors)], label=fname)
    ax.set_title("Latent space — target fates", fontsize=11)
    ax.legend(markerscale=3, fontsize=9)
    ax.set_xlabel("PC1 of Z"); ax.set_ylabel("PC2 of Z")

    # ── Panel 2: pullback metric log det G heatmap ───────────────────────────
    ax = axes[2]
    try:
        GX, GY, logdet, _ = compute_pullback_det(vae, Z, n_grid=25)
        im = ax.contourf(GX, GY, logdet, levels=20, cmap="plasma")
        plt.colorbar(im, ax=ax, label="log det G(z)")
        ax.scatter(z2d[:, 0], z2d[:, 1], s=1, alpha=0.2, color="white")
        ax.set_title("Pullback metric  log det G(z)", fontsize=11)
        ax.set_xlabel("PC1 of Z"); ax.set_ylabel("PC2 of Z")
    except Exception as e:
        ax.text(0.5, 0.5, f"Metric computation failed:\n{e}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("Pullback metric (error)", fontsize=11)

    # ── Panel 3: trajectory arcs for target fates ────────────────────────────
    ax = axes[3]
    ax.scatter(z2d[:, 0], z2d[:, 1], s=1.5, alpha=0.1, color="lightgray")

    # Encode lineage triples and draw arcs
    n_draw = min(200, len(lineage_triples))
    rng    = np.random.default_rng(42)
    draw_idx = rng.choice(len(lineage_triples), n_draw, replace=False)
    trip_sub = lineage_triples[draw_idx]

    z_d2 = pca2.transform(Z[trip_sub[:, 0]])
    z_d4 = pca2.transform(Z[trip_sub[:, 1]])
    z_d6 = pca2.transform(Z[trip_sub[:, 2]])

    day6_labels = labels_np[trip_sub[:, 2]]
    for fi, fname in enumerate(target_fates):
        if fname not in fate_names:
            continue
        fidx = fate_names.index(fname)
        mask = day6_labels == fidx
        if mask.sum() == 0:
            continue
        col = fate_colors[fi % len(fate_colors)]
        for i in np.where(mask)[0][:50]:
            xs = [z_d2[i, 0], z_d4[i, 0], z_d6[i, 0]]
            ys = [z_d2[i, 1], z_d4[i, 1], z_d6[i, 1]]
            ax.plot(xs, ys, "-o", color=col, alpha=0.3, linewidth=0.8,
                    markersize=1.5)

    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], color=fate_colors[fi % len(fate_colors)],
                          lw=1.5, label=fname)
                  for fi, fname in enumerate(target_fates) if fname in fate_names]
    ax.legend(handles=legend_els, fontsize=9)
    ax.set_title("Lineage arcs  day2→day4→day6", fontsize=11)
    ax.set_xlabel("PC1 of Z"); ax.set_ylabel("PC2 of Z")

    # ── Panel 4: silhouette score per cluster ────────────────────────────────
    ax = axes[4]
    try:
        # Only compute if enough samples
        n_uniq = len(np.unique(labels_np))
        if n_uniq >= 2 and len(Z) >= 2 * n_uniq:
            per_sample = silhouette_score(Z, labels_np, metric="euclidean",
                                           sample_size=min(5000, len(Z)))
            # Per-cluster breakdown
            from sklearn.metrics import silhouette_samples
            sil_vals = silhouette_samples(Z[:5000], labels_np[:5000])
            sil_by_type = []
            type_names  = []
            for lbl in unique_labels:
                m = labels_np[:5000] == lbl
                if m.sum() > 0:
                    sil_by_type.append(np.mean(sil_vals[m]))
                    type_names.append(
                        fate_names[lbl][:12] if lbl < len(fate_names) else str(lbl))
            y_pos = np.arange(len(type_names))
            colors = [cmap(i) for i in range(len(type_names))]
            ax.barh(y_pos, sil_by_type, color=colors, alpha=0.8)
            ax.axvline(per_sample, color="black", linestyle="--",
                       linewidth=1.5, label=f"Overall: {per_sample:.3f}")
            ax.axvline(0, color="gray", linestyle=":", linewidth=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(type_names, fontsize=7)
            ax.set_xlabel("Silhouette score")
            ax.set_title("Per-cluster silhouette  (higher = better)", fontsize=11)
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "Too few clusters/samples", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Silhouette (skipped)", fontsize=11)
    except Exception as e:
        ax.text(0.5, 0.5, f"Silhouette failed:\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title("Silhouette (error)", fontsize=11)

    # ── Panel 5: loss curves ──────────────────────────────────────────────────
    ax = axes[5]
    loss_colors = {
        "recon":   "steelblue",
        "kl":      "darkorange",
        "cls":     "forestgreen",
        "coh":     "mediumpurple",
        "triplet": "tomato",
        "vel_consist": "brown",
    }
    epochs_x = np.arange(len(history["total"]))
    for k, col in loss_colors.items():
        if k in history and len(history[k]) > 0:
            vals = np.array(history[k])
            # Plot on log scale if range is large
            ax.plot(epochs_x, vals, color=col, linewidth=1.5, label=k)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_yscale("log")
    ax.legend(fontsize=9, ncol=2)
    ax.set_title("Training loss curves", fontsize=11)

    plt.suptitle("Weinreb VAE diagnostic — geometry check", fontsize=14, y=1.01)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"\nSaved diagnostic figure → {save_path}")

    # ── Print key metrics ─────────────────────────────────────────────────────
    print("\n─── Latent space quality metrics ───")
    knn_score = knn_preservation_score(Z, X_pca[:, :Z.shape[1]], k=15)
    print(f"  KNN preservation (k=15): {knn_score:.3f}  "
          f"(>0.6 is good; measures local topology)")

    try:
        n_uniq = len(np.unique(labels_np))
        if n_uniq >= 2:
            sil = silhouette_score(Z, labels_np, metric="euclidean",
                                    sample_size=min(5000, len(Z)))
            print(f"  Silhouette (all types):  {sil:.4f}  "
                  f"(>0.2 = moderate; >0.5 = strong clustering)")
    except Exception:
        pass

    for fname in target_fates:
        if fname not in fate_names:
            continue
        fidx = fate_names.index(fname)
        mask = labels_np == fidx
        if mask.sum() > 1:
            z_fate   = Z[mask]
            centroid = z_fate.mean(axis=0)
            intra    = np.mean(np.linalg.norm(z_fate - centroid, axis=1))
            print(f"  Intra-cluster spread [{fname:12s}]: {intra:.4f}  "
                  f"(lower = tighter cluster)")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  Main
# ══════════════════════════════════════════════════════════════════════════════

TARGET_FATES = ["Monocyte", "Neutrophil"]


def main():
    preprocessed_path = "data/weinreb_preprocessed.h5ad"
    triples_path      = "data/weinreb_lineage_triples.npy"

    for p in [preprocessed_path, triples_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run preprocess_weinreb_spring.py first.")

    print("Loading data ...")
    adata = anndata.read_h5ad(preprocessed_path)

    X_pca = np.array(adata.obsm["X_pca"],        dtype=np.float32)
    V_pca = np.array(adata.obsm["velocity_pca"], dtype=np.float32)

    ct_col    = "Cell type annotation"
    ct_series  = adata.obs[ct_col].astype("category")
    fate_names = list(ct_series.cat.categories)
    labels_np  = ct_series.cat.codes.values.astype(np.int32)
    n_types    = len(fate_names)

    for f in TARGET_FATES:
        if f not in fate_names:
            print(f"  WARNING: '{f}' not found in annotations. Available: {fate_names}")

    # FIX A: StandardScaler normalisation — critical for tanh encoder.
    # PCA components have very different variances (PC1 >> PC50).
    # Without normalisation, many input dimensions are in [-10, 10] or
    # larger. The gelu encoder (and indirectly the tanh decoder) then
    # saturate on the first forward pass, producing NaN gradients.
    # We fit the scaler on X_pca and apply the SAME transform to V_pca
    # (velocity lives in the same feature space as position).
    print("Normalising PCA coordinates ...")
    scaler  = StandardScaler()
    X_pca_n = scaler.fit_transform(X_pca).astype(np.float32)
    # Velocity: same scale transform but no mean subtraction
    # (velocity has zero mean by construction; subtracting mean again
    # would shift it incorrectly — divide by std only)
    V_pca_n = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)
    print(f"  X_pca range after scaling: [{X_pca_n.min():.2f}, {X_pca_n.max():.2f}]")

    print(f"Cells: {X_pca.shape[0]}  |  PCA dims: {X_pca.shape[1]}  |  "
          f"Cell types: {n_types}")

    lineage_triples = np.load(triples_path).astype(np.int32)
    print(f"Lineage triples: {lineage_triples.shape[0]}")

    labels_j = jnp.array(labels_np)
    dataset  = BioDataset(
        X=jnp.array(X_pca_n),
        V=jnp.array(V_pca_n),
        labels=labels_j,
        lineage_pairs=None,
    )

    # ── Pre-compute KNN triplets in PCA space ─────────────────────────────────
    print("\nPre-computing KNN triplets in PCA space ...")
    knn_trip_idx = build_knn_triplet_indices(
        X_pca_n, labels_np, k=15, n_triplets=30_000, seed=42
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Training diagnostic VAE")
    print("="*60)
    t0  = time.time()
    key = jax.random.PRNGKey(2026)

    vae, history = train_vae(
        dataset         = dataset,
        triplet_indices = knn_trip_idx,
        lineage_triples = lineage_triples,
        labels_np       = labels_np,
        n_cell_types    = n_types,
        key             = key,
        latent_dim      = 8,
        epochs          = 120,
        batch_size      = 512,
        kl_cycle_len    = 20,
        kl_beta_max     = 5e-4,
        triplet_weight  = 1.0,
        triplet_margin  = 1.0,
        coherence_weight= 0.3,
        cls_weight      = 0.15,
        vel_weight      = 1.0,
    )
    print(f"\nTraining done in {(time.time()-t0)/60:.1f} min")
    # Save checkpoint for phase 2
    eqx.tree_serialise_leaves("data/weinreb_vae_phase1.eqx", vae)
    print("Saved checkpoint → data/weinreb_vae_phase1.eqx")

    # Attach PullbackRiemannian now that the decoder is trained and
    # its Jacobian is well-conditioned (tanh, smooth weights).
    print("Attaching pullback metric for evaluation ...")
    eval_key = jax.random.PRNGKey(99)
    vae_eval = attach_pullback_metric(vae, eval_key)

    # ── Encode & visualise ────────────────────────────────────────────────────
    print("\nEncoding full dataset ...")
    Z = encode_all(vae, dataset.X)
    print(f"Latent shape: {Z.shape}")

    plot_diagnostics(
        vae         = vae_eval,
        X_pca       = X_pca_n,
        Z           = Z,
        labels_np   = labels_np,
        fate_names  = fate_names,
        lineage_triples = lineage_triples,
        history     = history,
        target_fates= TARGET_FATES,
        save_path   = "weinreb_vae_diagnostic.png",
    )


if __name__ == "__main__":
    main()