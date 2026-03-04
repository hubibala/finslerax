"""
Geodesic Learning Visualization
================================
Trains NeuralRanders on four synthetic scenarios and produces:
  Row 1-2: 2D flat-space scenarios (River, Vortex)
  Row 3-4: Curved manifolds (Hyperboloid, Sphere)

Columns:
  A) True wind / displacement field
  B) Learned wind field W(x)
  C) Per-point cosine similarity heatmap
  D) Geodesic tangent comparison (curved cases only)

Usage:
    python scripts/visualize_geodesic_learning.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm
import os, sys

from ham.geometry.surfaces import EuclideanSpace, Hyperboloid, Sphere
from ham.models.learned import NeuralRanders
from ham.training.losses import LossComponent
from ham.training.pipeline import TrainingPhase, HAMPipeline
from ham.solvers.avbd import AVBDSolver
from ham.utils.math import safe_norm

# ── Import test scenarios ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from test_geodesic_learning import (
    generate_river_data,
    generate_vortex_data,
    generate_sphere_vortex,
    DirectWindAlignmentLoss,
    MetricIdentityLoss,
    WindRegularizationLoss,
    MetricModel,
    PairDataset,
    SyntheticDataset,
    cosine_similarity,
)


# ── Hyperboloid data generator ─────────────────────────────

def generate_hyperboloid_vortex(n: int = 400, noise: float = 0.0):
    """Rotational flow on H^2 (upper sheet of the two-sheeted hyperboloid).

    Samples points near the tip via exp_map so that the wind field has
    non-negligible magnitude everywhere in the training set — matching
    the approach used in test_geodesic_learning.py.
    """
    key = jax.random.PRNGKey(123)
    manifold = Hyperboloid(intrinsic_dim=2)

    # Sample points concentrated near the tip by using small tangent vectors
    origin = jnp.array([1.0, 0.0, 0.0])
    k1, k2 = jax.random.split(key)
    v_spatial = jax.random.normal(k1, (n, 2)) * 0.8  # scale controls spread
    v_tangent = jnp.concatenate([jnp.zeros((n, 1)), v_spatial], axis=1)
    starts = jax.vmap(manifold.exp_map, in_axes=(None, 0))(origin, v_tangent)

    def true_wind(x):
        # Rotation in the spatial (x1, x2) plane, constant magnitude
        v_rot = jnp.array([0.0, -x[2], x[1]])
        return manifold.to_tangent(x, 0.5 * v_rot)

    dt = 0.3
    noise_keys = jax.random.split(k2, n)

    def step(s, nk):
        tang = true_wind(s) * dt
        if noise > 0:
            raw_n = jax.random.normal(nk, (3,)) * noise
            tang = tang + manifold.to_tangent(s, raw_n)
        return manifold.retract(s, tang)

    ends = jax.vmap(step)(starts, noise_keys)
    return SyntheticDataset(starts, ends), true_wind


# ── Helpers ────────────────────────────────────────────────

def _filter_all(model):
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_array(leaf) else False, model
    )


def train(manifold, dataset, epochs=80, lr=5e-3, batch_size=64):
    key = jax.random.PRNGKey(2025)
    metric = NeuralRanders(manifold, key, hidden_dim=32)
    model = MetricModel(metric)
    ds = PairDataset(dataset.starts, dataset.ends)

    phase = TrainingPhase(
        name="WindAlignment",
        epochs=epochs,
        optimizer=optax.adam(lr),
        losses=[
            DirectWindAlignmentLoss(weight=1.0),
            MetricIdentityLoss(weight=5.0),
            WindRegularizationLoss(weight=0.01),
        ],
        filter_spec=_filter_all,
        requires_pairs=False,
    )
    pipeline = HAMPipeline(model)
    return pipeline.fit(ds, [phase], batch_size=batch_size, seed=2025)


def compute_geodesic_tangents(metric, starts, ends, n_eval=50):
    """
    Solve BVPs with the AVBD solver and return initial geodesic tangent vectors.
    This tests the FULL metric (H + W), not just W.
    """
    solver = AVBDSolver(step_size=0.05, iterations=30)

    def solve_one(s, e):
        traj = solver.solve(metric, s, e, n_steps=8, train_mode=False)
        return traj.vs[0]  # initial tangent = velocity at first segment

    # Only use a subset — BVP is expensive
    s_sub = starts[:n_eval]
    e_sub = ends[:n_eval]
    tangents = jax.vmap(solve_one)(s_sub, e_sub)
    true_disp = jax.vmap(metric.manifold.log_map)(s_sub, e_sub)
    return s_sub, tangents, true_disp


def per_point_cosine(a, b):
    """Per-point cosine similarity."""
    na = safe_norm(a, axis=-1)
    nb = safe_norm(b, axis=-1)
    dots = jnp.sum(a * b, axis=-1)
    return dots / (na * nb + 1e-8)


# ── 2D plotting ───────────────────────────────────────────

def plot_2d_row(axes, manifold, grid, true_W, pred_W, title):
    """3 panels: True | Learned | Cosine Similarity."""
    x, y = grid[:, 0], grid[:, 1]
    cos = per_point_cosine(true_W, pred_W)
    mean_cos = float(jnp.mean(cos))

    for ax in axes[:3]:
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.12)

    axes[0].quiver(x, y, true_W[:, 0], true_W[:, 1],
                   color="#60a5fa", pivot="mid", scale=18, width=0.004,
                   headwidth=4, headlength=5, alpha=0.9)
    axes[0].set_title(f"{title}\nTrue Displacement", fontsize=10, fontweight="bold")

    axes[1].quiver(x, y, pred_W[:, 0], pred_W[:, 1],
                   color="#f87171", pivot="mid", scale=18, width=0.004,
                   headwidth=4, headlength=5, alpha=0.9)
    axes[1].set_title(f"{title}\nLearned W(x)", fontsize=10, fontweight="bold")

    sc = axes[2].scatter(x, y, c=cos, cmap="RdYlGn", vmin=-1, vmax=1,
                         s=25, edgecolors="none", alpha=0.85)
    axes[2].set_title(f"Cosine Similarity\nmean = {mean_cos:.4f}",
                      fontsize=10, fontweight="bold")
    plt.colorbar(sc, ax=axes[2], shrink=0.75, pad=0.02)

    # 4th panel: N/A for flat space
    axes[3].text(0.5, 0.5, "N/A\n(flat space)",
                 ha="center", va="center", fontsize=12,
                 color="#64748b", transform=axes[3].transAxes)
    axes[3].set_title("Geodesic Tangents", fontsize=10, fontweight="bold", color="#64748b")
    axes[3].set_xticks([])
    axes[3].set_yticks([])


# ── 3D plotting ───────────────────────────────────────────

def _wireframe(ax, kind):
    """Draw a transparent manifold wireframe."""
    if kind == "sphere":
        u, v = np.mgrid[0:2*np.pi:25j, 0:np.pi:15j]
        ax.plot_wireframe(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                          color="gray", alpha=0.06, linewidth=0.3)
    elif kind == "hyperboloid":
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, 2.0, 15)
        U, V = np.meshgrid(u, v)
        X0 = np.cosh(V)
        X1 = np.sinh(V) * np.cos(U)
        X2 = np.sinh(V) * np.sin(U)
        ax.plot_wireframe(X1, X2, X0, color="gray", alpha=0.06, linewidth=0.3)


def _setup_3d(ax, kind):
    lim = 1.4 if kind == "sphere" else 2.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim if kind == "sphere" else 0.5, lim if kind == "sphere" else 4.0)
    ax.set_box_aspect([1, 1, 1])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.tick_params(labelsize=6)
    _wireframe(ax, kind)


def plot_3d_row(axes, manifold, grid, true_W, pred_W, title, kind,
                geo_pts=None, geo_tangents=None, geo_true=None):
    """4 panels: True | Learned | Cosine | Geodesic tangents."""
    # For 3D: pts are (x1, x2, x0) for hyperboloid or (x,y,z) for sphere
    if kind == "hyperboloid":
        px, py, pz = grid[:, 1], grid[:, 2], grid[:, 0]
        tw0, tw1, tw2 = true_W[:, 1], true_W[:, 2], true_W[:, 0]
        pw0, pw1, pw2 = pred_W[:, 1], pred_W[:, 2], pred_W[:, 0]
    else:
        px, py, pz = grid[:, 0], grid[:, 1], grid[:, 2]
        tw0, tw1, tw2 = true_W[:, 0], true_W[:, 1], true_W[:, 2]
        pw0, pw1, pw2 = pred_W[:, 0], pred_W[:, 1], pred_W[:, 2]

    cos = per_point_cosine(true_W, pred_W)
    mean_cos = float(jnp.mean(cos))

    # Panel A: True
    _setup_3d(axes[0], kind)
    axes[0].quiver(px, py, pz, tw0, tw1, tw2,
                   color="#60a5fa", length=0.25, normalize=False,
                   linewidth=0.7, arrow_length_ratio=0.3, alpha=0.85)
    axes[0].set_title(f"{title}\nTrue Wind", fontsize=10, fontweight="bold")

    # Panel B: Learned
    _setup_3d(axes[1], kind)
    axes[1].quiver(px, py, pz, pw0, pw1, pw2,
                   color="#f87171", length=0.25, normalize=False,
                   linewidth=0.7, arrow_length_ratio=0.3, alpha=0.85)
    axes[1].set_title(f"{title}\nLearned W(x)", fontsize=10, fontweight="bold")

    # Panel C: Cosine heatmap
    _setup_3d(axes[2], kind)
    axes[2].scatter(px, py, pz, c=np.array(cos), cmap="RdYlGn",
                    vmin=-1, vmax=1, s=15, alpha=0.85, edgecolors="none")
    axes[2].set_title(f"Cosine Similarity\nmean = {mean_cos:.4f}",
                      fontsize=10, fontweight="bold")

    # Panel D: Geodesic tangent comparison
    if geo_pts is not None and geo_tangents is not None:
        geo_cos = per_point_cosine(geo_tangents, geo_true)
        mean_geo_cos = float(jnp.mean(geo_cos))

        _setup_3d(axes[3], kind)
        if kind == "hyperboloid":
            gx, gy, gz = geo_pts[:, 1], geo_pts[:, 2], geo_pts[:, 0]
            gt0, gt1, gt2 = geo_true[:, 1], geo_true[:, 2], geo_true[:, 0]
            gp0, gp1, gp2 = geo_tangents[:, 1], geo_tangents[:, 2], geo_tangents[:, 0]
        else:
            gx, gy, gz = geo_pts[:, 0], geo_pts[:, 1], geo_pts[:, 2]
            gt0, gt1, gt2 = geo_true[:, 0], geo_true[:, 1], geo_true[:, 2]
            gp0, gp1, gp2 = geo_tangents[:, 0], geo_tangents[:, 1], geo_tangents[:, 2]

        axes[3].quiver(gx, gy, gz, gt0, gt1, gt2,
                       color="#60a5fa", length=0.3, normalize=False,
                       linewidth=0.8, arrow_length_ratio=0.3, alpha=0.6,
                       label="True disp.")
        axes[3].quiver(gx, gy, gz, gp0, gp1, gp2,
                       color="#a78bfa", length=0.3, normalize=False,
                       linewidth=0.8, arrow_length_ratio=0.3, alpha=0.8,
                       linestyle="dashed", label="Geodesic v₀")
        axes[3].set_title(f"Geodesic Tangents\ncos = {mean_geo_cos:.4f}",
                          fontsize=10, fontweight="bold")
        axes[3].legend(fontsize=7, loc="upper left", framealpha=0.5)
    else:
        axes[3].text(0.5, 0.5, "Computing...", ha="center", va="center",
                     fontsize=10, color="#64748b", transform=axes[3].transAxes)


# ── Main ───────────────────────────────────────────────────

def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "text.color": "#e2e8f0",
        "grid.color": "#334155",
    })

    results = []

    # ── 1. River (flat) ──
    print("═" * 50)
    print("Training River scenario...")
    m_r = EuclideanSpace(2)
    ds_r, flow_r = generate_river_data(500)
    model_r = train(m_r, ds_r, epochs=60, lr=5e-3)
    grid_r = jax.random.uniform(jax.random.PRNGKey(77), (400, 2), minval=-2.2, maxval=2.2)
    true_r = jnp.broadcast_to(flow_r * 0.5, (400, 2))
    pred_r = jax.vmap(lambda p: model_r._get_zermelo_data(p)[1])(grid_r)
    results.append(("River Flow", m_r, grid_r, true_r, pred_r, "2d", None))

    # ── 2. Vortex (flat) ──
    print("Training Vortex scenario...")
    m_v = EuclideanSpace(2)
    ds_v, _ = generate_vortex_data(600)
    model_v = train(m_v, ds_v, epochs=80, lr=5e-3)
    pts_v = ds_v.starts[:400]
    true_v = jax.vmap(m_v.log_map)(pts_v, ds_v.ends[:400])
    pred_v = jax.vmap(lambda p: model_v._get_zermelo_data(p)[1])(pts_v)
    results.append(("Vortex Flow", m_v, pts_v, true_v, pred_v, "2d", None))

    # ── 3. Hyperboloid Vortex ──
    print("Training Hyperboloid Vortex scenario...")
    m_h = Hyperboloid(intrinsic_dim=2)
    ds_h, tw_h = generate_hyperboloid_vortex(500, noise=0.0)
    model_h = train(m_h, ds_h, epochs=3000, lr=1e-3)
    grid_h = jax.vmap(m_h.random_sample, in_axes=(0, None))(
        jax.random.split(jax.random.PRNGKey(777), 300), ()
    )
    true_h = jax.vmap(tw_h)(grid_h)
    pred_h = jax.vmap(lambda p: model_h._get_zermelo_data(p)[1])(grid_h)

    # Geodesic tangent evaluation
    print("  Computing geodesic tangents (AVBD)...")
    geo_pts_h, geo_tan_h, geo_true_h = compute_geodesic_tangents(
        model_h.metric, ds_h.starts, ds_h.ends, n_eval=40
    )
    results.append(("Hyperboloid Vortex", m_h, grid_h, true_h, pred_h, "hyperboloid",
                     (geo_pts_h, geo_tan_h, geo_true_h)))

    # ── 4. Sphere Vortex ──
    print("Training Sphere Vortex scenario...")
    m_s = Sphere(radius=1.0)
    ds_s, tw_s = generate_sphere_vortex(600, noise=0.0)
    model_s = train(m_s, ds_s, epochs=100, lr=3e-3)
    grid_s = jax.vmap(m_s.random_sample, in_axes=(0, None))(
        jax.random.split(jax.random.PRNGKey(777), 300), ()
    )
    true_s = jax.vmap(tw_s)(grid_s)
    pred_s = jax.vmap(lambda p: model_s._get_zermelo_data(p)[1])(grid_s)

    print("  Computing geodesic tangents (AVBD)...")
    geo_pts_s, geo_tan_s, geo_true_s = compute_geodesic_tangents(
        model_s.metric, ds_s.starts, ds_s.ends, n_eval=40
    )
    results.append(("Sphere Vortex", m_s, grid_s, true_s, pred_s, "sphere",
                     (geo_pts_s, geo_tan_s, geo_true_s)))

    # ── Assemble figure ──
    print("═" * 50)
    print("Rendering figure...")

    fig = plt.figure(figsize=(22, 20))
    fig.suptitle("Geodesic Learning: Wind Field Recovery & Geodesic Quality",
                 fontsize=16, fontweight="bold", y=0.99, color="#f8fafc")

    outer = gridspec.GridSpec(4, 4, hspace=0.32, wspace=0.22,
                              left=0.03, right=0.97, top=0.95, bottom=0.02)

    for row, (title, manif, grid, tw, pw, kind, geo_data) in enumerate(results):
        if kind == "2d":
            axes = [fig.add_subplot(outer[row, c]) for c in range(4)]
            plot_2d_row(axes, manif, grid, tw, pw, title)
        else:
            axes = [fig.add_subplot(outer[row, c], projection="3d") for c in range(4)]
            gp, gt, gtr = geo_data if geo_data else (None, None, None)
            plot_3d_row(axes, manif, grid, tw, pw, title, kind,
                        geo_pts=gp, geo_tangents=gt, geo_true=gtr)

    # ── Summary bar at bottom ──
    cos_summary = []
    for title, _, grid, tw, pw, kind, geo_data in results:
        wcos = float(jnp.mean(per_point_cosine(tw, pw)))
        geo_cos_str = ""
        if geo_data is not None:
            _, gt, gtr = geo_data
            gcos = float(jnp.mean(per_point_cosine(gt, gtr)))
            geo_cos_str = f"  |  Geodesic cos = {gcos:.4f}"
        cos_summary.append(f"{title}: Wind cos = {wcos:.4f}{geo_cos_str}")

    summary_text = "    │    ".join(cos_summary)
    fig.text(0.5, 0.005, summary_text, ha="center", fontsize=9,
             color="#94a3b8", fontstyle="italic")

    out_path = os.path.join(os.path.dirname(__file__), "..", "geodesic_learning_results.png")
    out_path = os.path.abspath(out_path)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n✓ Saved → {out_path}")
    plt.close(fig)

    # Print summary table
    print("\n" + "═" * 60)
    print(f"{'Scenario':<25} {'Wind cos':>10} {'Geodesic cos':>14}")
    print("─" * 60)
    for title, _, grid, tw, pw, kind, geo_data in results:
        wcos = float(jnp.mean(per_point_cosine(tw, pw)))
        gcos = "—"
        if geo_data is not None:
            _, gt, gtr = geo_data
            gcos = f"{float(jnp.mean(per_point_cosine(gt, gtr))):.4f}"
        print(f"{title:<25} {wcos:>10.4f} {gcos:>14}")
    print("═" * 60)


if __name__ == "__main__":
    main()
