"""
End-to-End Geodesic Learning Visualization
==========================================

Trains the full HAM pipeline (VAE + NeuralRanders) on high-dimensional
noisy observations of a rotating circle, and visualizes the recovery.

Panels:
1. Data Space PCA (True Dynamics)
2. Latent Embedding (Color = Phase)
3. Recovered Dynamics (Pushforward vs. Observed)
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx
import optax
import os, sys
from sklearn.decomposition import PCA

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ham.geometry.surfaces import Sphere
from ham.models.learned import NeuralRanders
from ham.bio.vae import GeometricVAE
from ham.training.losses import ReconstructionLoss, KLDivergenceLoss, GeodesicSprayLoss
from ham.training.pipeline import TrainingPhase, HAMPipeline
from ham.utils.math import safe_norm

# Import logic from the test script
from tests.test_e2e_geodesic import (
    generate_rotating_circle_data,
    LatentVelocityAlignmentLoss,
    get_filter_fn,
    PipelineDataset,
    cosine_sim_batch
)

def train_and_vis():
    print("Generating High-Dimensional Synthetic Data...")
    latent_dim = 6
    data_dim = 50
    # Use the dense, non-axis-aligned generator
    synth = generate_rotating_circle_data(
        n=1000, 
        data_dim=data_dim, 
        latent_dim=latent_dim, 
        noise_level=0.03, 
        seed=42
    )
    
    # 1. Setup Model
    key = jax.random.PRNGKey(2025)
    k1, k2 = jax.random.split(key)
    manifold = Sphere(intrinsic_dim=latent_dim, radius=1.0)
    metric = NeuralRanders(manifold, k1, hidden_dim=64)
    model = GeometricVAE(data_dim=data_dim, latent_dim=latent_dim, metric=metric, key=k2)
    ds = PipelineDataset(synth)

    # 2. Pipeline Training
    print("\nTraining Phase 1: Manifold Learning...")
    p1 = TrainingPhase(
        name="Manifold", epochs=1500, optimizer=optax.adam(1e-4),
        losses=[ReconstructionLoss(1.0), KLDivergenceLoss(1e-4)],
        filter_spec=get_filter_fn(lambda m: (m.encoder_net, m.decoder_net)),
        requires_pairs=False,
    )
    model_p1 = HAMPipeline(model).fit(ds, [p1], batch_size=64, seed=2025)

    print("\nTraining Phase 2: Geodesic Metric Learning...")
    p2 = TrainingPhase(
        name="Metric", epochs=300, optimizer=optax.adam(1e-4),
        losses=[LatentVelocityAlignmentLoss(1.0), GeodesicSprayLoss(1)],
        filter_spec=get_filter_fn(lambda m: m.metric),
        requires_pairs=False,
    )
    model_p2 = HAMPipeline(model_p1).fit(ds, [p2], batch_size=64, seed=2025)

    # 3. Preparation for Visualization
    print("\nPreparing Visualizations...")
    # Encode all data
    keys = jax.random.split(jax.random.PRNGKey(0), synth.X.shape[0])
    z_enc = jax.vmap(model_p2.encode)(synth.X, keys)
    
    # PCA project high-dim data to 2D for visualization
    pca = PCA(n_components=2)
    x_2d = pca.fit_transform(np.array(synth.X))
    v_2d = pca.transform(np.array(synth.X + synth.V)) - x_2d
    
    # Phase (angle) for coloring
    # z_true is (n, 7), find its principal 2D plane to extract angle
    pca_z = PCA(n_components=2)
    z_true_2d = pca_z.fit_transform(np.array(synth.z_true))
    angles = np.arctan2(z_true_2d[:, 1], z_true_2d[:, 0])
    
    # Recovered Dynamics
    def predict_velocity(x):
        z_mean, _ = model_p2.project_control(x, jnp.zeros_like(x))
        _, W, _ = model_p2.metric._get_zermelo_data(z_mean)
        _, v_pred = jax.jvp(model_p2.decode, (z_mean,), (W,))
        return v_pred

    v_pred = np.array(jax.vmap(predict_velocity)(synth.X[:200]))
    v_pred_2d = pca.transform(np.array(synth.X[:200] + v_pred)) - x_2d[:200]
    
    # 4. Plotting
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "text.color": "#e2e8f0",
        "grid.color": "#334155",
        "font.family": "sans-serif"
    })
    
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    # Panel 1: Data Space
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(x_2d[:, 0], x_2d[:, 1], c=angles, cmap="hsv", s=15, alpha=0.3)
    q = ax1.quiver(x_2d[::5, 0], x_2d[::5, 1], v_2d[::5, 0], v_2d[::5, 1], 
                   color="cyan", alpha=0.6, width=0.003, label="True Dynamics")
    ax1.set_title("Data Space (PCA Projection)\nTrue Dynamical Field", fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: Latent Space
    # Since latent is 6D, we take the top 2 linear components for the circle signal
    pca_lat = PCA(n_components=2)
    z_2d_lat = pca_lat.fit_transform(np.array(z_enc))
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(z_2d_lat[:, 0], z_2d_lat[:, 1], c=angles, cmap="hsv", s=20, alpha=0.8)
    ax2.set_title("Latent Manifold Embedding\n(Principal Signal Plane)", fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])
    # Draw a circle border
    circle = plt.Circle((0, 0), np.max(np.abs(z_2d_lat)), color='white', fill=False, alpha=0.1, linestyle='--')
    ax2.add_artist(circle)

    # Panel 3: Recovery Quality
    ax3 = fig.add_subplot(gs[0, 2])
    # Plot True vs Pred quiver in PCA overlay
    ax3.scatter(x_2d[:200, 0], x_2d[:200, 1], color="gray", s=5, alpha=0.2)
    ax3.quiver(x_2d[:200:2, 0], x_2d[:200:2, 1], v_2d[:200:2, 0], v_2d[:200:2, 1], 
               color="cyan", alpha=0.4, scale=1, width=0.004, label="True")
    ax3.quiver(x_2d[:200:2, 0], x_2d[:200:2, 1], v_pred_2d[::2, 0], v_pred_2d[::2, 1], 
               color="magenta", alpha=0.8, scale=1, width=0.004, label="Recovered (HAM)")
    
    cos = float(jnp.mean(jnp.sum(synth.V[:200] * v_pred, axis=-1) / 
                         (safe_norm(synth.V[:200], axis=-1) * safe_norm(v_pred, axis=-1) + 1e-8)))
    ax3.set_title(f"Dynamics Recovery\nCosine Sim = {cos:.4f}", fontweight="bold")
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.legend(loc="upper right", fontsize=8)

    plt.suptitle("HAM: End-to-End Geodesic Learning on High-Dim Dense Signal", 
                 fontsize=16, fontweight="bold", y=1.05, color="#f8fafc")
    
    out_path = "e2e_geodesic_recovery.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✓ Saved visualization to {out_path}")
    plt.show()

if __name__ == "__main__":
    train_and_vis()
