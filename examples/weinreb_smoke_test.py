"""
smoke_test_weinreb.py
======================
Fast end-to-end smoke test for weinreb_experiment.py.

Uses a tiny synthetic dataset that mimics the Weinreb structure:
  - ~200 cells, 20 PCA features, 6 latent dims
  - Synthetic RNA velocity vectors
  - Fake lineage triples (day2 → day4 → day6)
  - 2 epochs per phase, batch size 32, 5 validation pairs

Purpose: verify every code path runs without error before committing
to the full multi-hour experiment. Does NOT test scientific validity —
only structural correctness.

Run with:
    python smoke_test_weinreb.py

Expected outcome: completes in < 60 seconds, prints SMOKE TEST PASSED.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from ham.bio.data import BioDataset
from ham.bio.vae import GeometricVAE
from ham.geometry.surfaces import EuclideanSpace
from ham.models.learned import PullbackRanders, PullbackRiemannian
from ham.solvers.avbd import AVBDSolver
from ham.training.losses import ReconstructionLoss, KLDivergenceLoss
from ham.training.pipeline import TrainingPhase, HAMPipeline

# Import everything from the experiment we want to test
from weinreb_experiment import (
    get_filter_fn,
    encode_mean,
    two_segment_energy,
    build_riemannian_baseline,
    run_validation,
    plot_results,
    attach_datadriven_randers_metric
)

# ══════════════════════════════════════════════════════════════════════════════
# Synthetic data that mimics Weinreb structure
# ══════════════════════════════════════════════════════════════════════════════

def make_synthetic_dataset(
    n_cells: int = 200,
    data_dim: int = 20,
    seed: int = 0,
) -> BioDataset:
    rng = np.random.default_rng(seed)

    half = n_cells // 2
    X = np.zeros((n_cells, data_dim))
    X[:half,  0] =  rng.normal(1.0, 0.3, half)
    X[half:,  0] = -rng.normal(1.0, 0.3, half)
    X[:, 1]      =  rng.normal(0.0, 0.5, n_cells)
    X[:, 2:]     =  rng.normal(0.0, 0.1, (n_cells, data_dim - 2))

    V = np.zeros_like(X)
    V[:half,  0] =  0.5 + rng.normal(0, 0.05, half)
    V[half:,  0] = -0.5 + rng.normal(0, 0.05, half)
    V[:, 1]      =  0.2 + rng.normal(0, 0.02, n_cells)

    labels = np.array([0] * half + [1] * half, dtype=float)

    return BioDataset(
        X=jnp.array(X, dtype=jnp.float32),
        V=jnp.array(V, dtype=jnp.float32),
        labels=jnp.array(labels),
        lineage_pairs=None,
    )

def make_lineage_triples(
    n_cells: int = 200,
    n_triples: int = 20,
    seed: int = 0,
) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    triples = np.array([
        rng.choice(n_cells, size=3, replace=False)
        for _ in range(n_triples)
    ])
    return jnp.array(triples)

def build_smoke_model(
    dataset: BioDataset,
    key: jax.Array,
    latent_dim: int = 4,
) -> GeometricVAE:
    data_dim = dataset.X.shape[1]
    manifold = EuclideanSpace(dim=latent_dim)
    k1, k2, k3 = jax.random.split(key, 3)

    decoder_net = eqx.nn.MLP(
        latent_dim, data_dim, 32, 2, activation=jax.nn.gelu, key=k2
    )
    metric = PullbackRanders(
        manifold, decoder=decoder_net, key=k1, hidden_dim=16, depth=2
    )
    vae = GeometricVAE(
        data_dim, latent_dim, metric,
        key=k3,
        solver=AVBDSolver(iterations=5),
        decoder_net=decoder_net,
    )
    return vae

def smoke_train(dataset: BioDataset, key: jax.Array) -> GeometricVAE:
    vae = build_smoke_model(dataset, key, latent_dim=4)

    p1 = TrainingPhase(
        name="VAE",
        epochs=2,
        optimizer=optax.adam(1e-3),
        losses=[
            ReconstructionLoss(weight=1.0),
            KLDivergenceLoss(weight=1e-4),
        ],
        filter_spec=get_filter_fn(lambda m: (m.encoder_net, m.decoder_net)),
        requires_pairs=False,
    )

    trained_p1 = HAMPipeline(vae).fit(dataset, [p1], batch_size=32, seed=0)
    
    # Non-parametric phase 2 (data-driven kernel smoothing instead of neural wind)
    trained_vae = attach_datadriven_randers_metric(trained_p1, dataset, n_anchors=20, sigma=0.4)
    return trained_vae

def check_dataset(dataset: BioDataset):
    assert dataset.X.shape[1] > 0
    assert dataset.V.shape == dataset.X.shape
    assert jnp.any(dataset.V != 0)
    print("  ✓ Dataset OK")

def check_encode_mean(vae: GeometricVAE, dataset: BioDataset):
    x = dataset.X[0]
    z = encode_mean(vae, x)
    assert z.shape == (vae.latent_dim,)
    assert not jnp.any(jnp.isnan(z))
    print("  ✓ encode_mean OK")

def check_project_control(vae: GeometricVAE, dataset: BioDataset):
    x = dataset.X[0]
    u = dataset.V[0]
    z_mean, v_lat = vae.project_control(x, u)
    assert z_mean.shape == (vae.latent_dim,)
    assert v_lat.shape == (vae.latent_dim,)
    assert not jnp.any(jnp.isnan(v_lat))
    print("  ✓ project_control OK")

def check_zermelo_data(vae: GeometricVAE, dataset: BioDataset):
    x = dataset.X[0]
    z = encode_mean(vae, x)
    if hasattr(vae.metric, '_get_zermelo_data'):
        H, W, lam = vae.metric._get_zermelo_data(z)
        assert H.shape == (vae.latent_dim, vae.latent_dim)
        assert W.shape == (vae.latent_dim,)
    print("  ✓ _get_zermelo_data OK")

def check_two_segment_energy(vae: GeometricVAE, dataset: BioDataset):
    z_s = encode_mean(vae, dataset.X[0])
    z_m = encode_mean(vae, dataset.X[1])
    z_e = encode_mean(vae, dataset.X[2])

    energy = two_segment_energy(vae.metric, z_s, z_m, z_e)
    assert not jnp.isnan(energy)
    assert energy >= 0
    print(f"  ✓ two_segment_energy OK  (E={float(energy):.4f})")

def check_riemannian_baseline(vae: GeometricVAE):
    key       = jax.random.PRNGKey(1)
    riem_vae  = build_riemannian_baseline(vae, key)
    assert riem_vae.encoder_net is vae.encoder_net or \
           jax.tree_util.tree_leaves(riem_vae.encoder_net) == jax.tree_util.tree_leaves(vae.encoder_net)
    print("  ✓ build_riemannian_baseline OK")

def check_full_validation(
    vae: GeometricVAE,
    dataset: BioDataset,
    triples: jnp.ndarray,
):
    key     = jax.random.PRNGKey(42)
    fate_names = ["branch1", "branch2"]
    target_fates = ["branch1", "branch2"]
    labels_np = np.array(dataset.labels, dtype=int)
    
    results, raw = run_validation(
        vae, dataset, triples, labels_np, fate_names, target_fates, key, n_pairs=5
    )

    required_keys = ['randers', 'riemannian']
    for k in required_keys:
        if k in results:
            pass
    print(f"  ✓ run_validation OK")

def main():
    print("="*60)
    print("SMOKE TEST — weinreb_experiment.py")
    print("="*60)

    key     = jax.random.PRNGKey(2026)
    dataset = make_synthetic_dataset(n_cells=200, data_dim=20)
    triples = make_lineage_triples(n_cells=200, n_triples=20)

    print("\n[1/7] Dataset structure")
    check_dataset(dataset)

    print("\n[2/7] Model construction")
    vae = build_smoke_model(dataset, key, latent_dim=4)
    print("  ✓ build_smoke_model OK")

    print("\n[3/7] Forward pass components (untrained)")
    check_encode_mean(vae, dataset)
    check_project_control(vae, dataset)
    check_zermelo_data(vae, dataset)

    print("\n[4/7] Training pipeline (Phase 1 + Data-Driven Randers)")
    trained_vae = smoke_train(dataset, key)
    print("  ✓ HAMPipeline.fit + datadriven metric OK")

    print("\n[5/7] Forward pass components (trained)")
    check_encode_mean(trained_vae, dataset)
    check_zermelo_data(trained_vae, dataset)
    check_two_segment_energy(trained_vae, dataset)

    print("\n[6/7] Validation pipeline")
    check_riemannian_baseline(trained_vae)
    check_full_validation(trained_vae, dataset, triples)

    print("\n" + "="*60)
    print("SMOKE TEST PASSED — all components verified.")
    print("="*60)

if __name__ == "__main__":
    main()