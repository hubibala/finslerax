#!/usr/bin/env python3
"""
Experiment: Lagrangian Randers Metric Recovery on Synthetic Arrival-Time Data
=============================================================================

Validates that HAMTools' Lagrangian geodesic approach (AVBD solver) can recover
a spatially varying Riemannian metric from arrival-time observations, establishing
a quantitative baseline against Gahtan et al. (2026) Section 5.

**Scientific question:** Can boundary-value geodesic solvers learn a Finsler
metric from arrival-time supervision, matching Eulerian eikonal baselines?

**Method:**
  1. Define a ground-truth piecewise-constant metric on a grid in [0,1]^2.
  2. Compute true arrival times from an off-boundary source via dense AVBD
     geodesic computation with a high-fidelity solver.
  3. Train a NeuralRanders metric using ArrivalTimeLoss with the AVBD solver.
  4. Evaluate recovery error at multiple observation densities.
  5. Run Riemannian (W=0) vs. Randers ablation.

**Key outputs:**
  - figs/phase1_metric_recovery_*.pdf   — True vs. recovered metric heatmaps
  - figs/phase1_density_sweep.pdf       — Error vs. observation density curve
  - figs/phase1_convergence.pdf         — Training loss convergence
  - figs/phase1_geodesic_paths.pdf      — Geodesic paths (HAMTools-only capability)
  - figs/phase1_regularization.pdf      — Jacobian regularization sweep (U-curve)
  - figs/phase1_riemannian_vs_randers.pdf — W=0 vs W ablation

**Reference:**
  Gahtan, Shpund & Bronstein (2026). Wildfire Simulation with Differentiable
  Randers-Finsler Eikonal Solvers. arXiv:2603.00035, Section 5.

**HAMTools spec references:**
  spec/MATH_SPEC.md § 1.2 (Energy Functional), § 2.1 (Geodesic Spray)
  spec/ARCH_SPEC.md § 3 (Metric Hierarchy), § 4 (Solver Interface)

Usage:
    python examples/experiment_gahtan_phase1.py [--quick]
"""

import argparse
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.zoo import Randers, Riemannian
from ham.models.learned import NeuralRanders
from ham.nn.networks import PSDMatrixField, VectorField
from ham.solvers.avbd import AVBDSolver
from ham.training.losses import ArrivalTimeLoss

# ---------------------------------------------------------------------------
# Reproducibility & Configuration
# ---------------------------------------------------------------------------

SEED = 42
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figs")


class GridMetricField(eqx.Module):
    """Per-pixel isotropic metric field on a regular grid.

    Matches Gahtan et al.'s parameterization: directly optimizable scalar
    metric values g(x) on a regular grid, bilinearly interpolated to
    arbitrary positions. Returns G(x) = g(x) * I_2.

    This eliminates the spectral bias of neural network parameterizations,
    allowing recovery of piecewise-constant metric fields.

    Args:
        grid_size: Number of grid points per axis.
        margin: Boundary margin matching the observation grid.
    """
    grid_values: jax.Array  # (grid_size, grid_size) scalar metric values
    grid_size: int = eqx.field(static=True)
    margin: float = eqx.field(static=True)

    def __init__(self, grid_size: int, margin: float = 0.05):
        self.grid_size = grid_size
        self.margin = margin
        self.grid_values = jnp.ones((grid_size, grid_size))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Bilinearly interpolate grid values and return G(x) = g(x) * I."""
        extent = 1.0 - 2 * self.margin
        gx = (x[0] - self.margin) / extent * (self.grid_size - 1)
        gy = (x[1] - self.margin) / extent * (self.grid_size - 1)
        gx = jnp.clip(gx, 0.0, self.grid_size - 1.001)
        gy = jnp.clip(gy, 0.0, self.grid_size - 1.001)
        gx0 = jnp.floor(gx).astype(jnp.int32)
        gy0 = jnp.floor(gy).astype(jnp.int32)
        gx1 = jnp.minimum(gx0 + 1, self.grid_size - 1)
        gy1 = jnp.minimum(gy0 + 1, self.grid_size - 1)
        fx = gx - gx0
        fy = gy - gy0
        v00 = self.grid_values[gy0, gx0]
        v01 = self.grid_values[gy0, gx1]
        v10 = self.grid_values[gy1, gx0]
        v11 = self.grid_values[gy1, gx1]
        val = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) \
            + v10 * (1 - fx) * fy + v11 * fx * fy
        val = jnp.maximum(val, 0.1)
        return val * jnp.eye(2)


def get_config(quick: bool = False):
    """Return experiment configuration dict.

    Args:
        quick: If True, use smaller grid and fewer iterations for smoke testing.

    All hyperparameters are documented here for reproducibility.
    Uses grid-based (per-pixel) metric parameterization by default to match
    Gahtan et al.'s approach, isolating the solver comparison from
    parameterization effects. Set param_type='neural' for neural network
    ablation.
    """
    if quick:
        return dict(
            grid_size=20,
            n_train_steps=300,
            solver_steps=20,
            solver_iters=100,
            hidden_dim=32,
            depth=2,
            lr=0.05,
            seeds=[0],
            densities=[1.0, 0.25],
            reg_lambdas=[0.0, 1e-3],
            obs_per_train_step=64,
            n_reg_points=32,
            tv_lambda=0.001,
            param_type='grid',  # 'grid' or 'neural'
        )
    return dict(
        grid_size=80,
        n_train_steps=800,
        solver_steps=30,
        solver_iters=150,
        hidden_dim=64,
        depth=3,
        lr=0.05,
        seeds=[0, 1, 2, 3, 4],
        densities=[1.0, 0.50, 0.07, 0.03],
        reg_lambdas=[0.0, 1e-4, 1e-3, 1e-2, 1e-1],
        obs_per_train_step=64,
        n_reg_points=64,
        tv_lambda=0.001,
        param_type='grid',  # 'grid' or 'neural'
    )


# ---------------------------------------------------------------------------
# Ground-Truth Metric & Arrival Times
# ---------------------------------------------------------------------------

def piecewise_metric_field(x: jnp.ndarray, boundary: float = 0.5) -> jnp.ndarray:
    """Piecewise-constant isotropic metric tensor G(x).

    G(x) = g(x) * I_2, where g(x) = 1.0 for x[0] < boundary, else 2.0.
    This mimics a medium with a sharp speed boundary (like a fuel-type change).
    The Finsler cost is F(x,v) = sqrt(g(x)) * ||v||, so speed ∝ 1/sqrt(g).

    Args:
        x: Point in R^2, shape (2,).
        boundary: x-coordinate of the metric transition. Default: 0.5.

    Returns:
        Metric tensor G(x), shape (2, 2).
    """
    g_val = jnp.where(x[0] < boundary, 1.0, 2.0)
    return g_val * jnp.eye(2)


def true_metric_scalar(x: jnp.ndarray, boundary: float = 0.5) -> float:
    """Scalar metric value g(x) for visualization."""
    return jnp.where(x[0] < boundary, 1.0, 2.0)


def make_true_metric(boundary: float = 0.5):
    """Create the ground-truth Riemannian metric for arrival time computation.

    Returns a Riemannian metric with G(x) = g(x)*I, suitable for computing
    exact geodesic distances via the AVBD solver.
    """
    manifold = EuclideanSpace(2)
    g_net = lambda x: piecewise_metric_field(x, boundary=boundary)
    return Riemannian(manifold, g_net)


def compute_true_arrival_times(source, grid_points, cfg, boundary=0.5):
    """Compute ground-truth arrival times using the SAME solver config as training.

    Uses the identical solver parameters (step_size, iterations, n_steps) and
    quadrature method (midpoint) as ArrivalTimeLoss. This ensures training
    targets are perfectly self-consistent with the predicted values, avoiding
    systematic discretization bias.

    Args:
        source: Source point, shape (2,).
        grid_points: Observation points, shape (K, 2).
        cfg: Configuration dict (provides solver_steps, solver_iters, etc.).
        boundary: x-coordinate of the metric transition.

    Returns:
        Arrival times, shape (K,).
    """
    true_metric = make_true_metric(boundary)
    # Use the SAME solver config as training for self-consistency
    gt_solver = AVBDSolver(
        step_size=0.05,
        iterations=cfg['solver_iters'],
        energy_tol=1e-6,
    )
    n_steps = cfg['solver_steps']

    def single_distance(target):
        traj = gt_solver.solve(true_metric, source, target, n_steps=n_steps)
        path = traj.xs
        # Midpoint quadrature for O(h^2) accuracy — matches ArrivalTimeLoss
        segments = jnp.diff(path, axis=0)
        midpoints = (path[:-1] + path[1:]) / 2.0
        step_costs = jax.vmap(true_metric.metric_fn)(midpoints, segments)
        return jnp.sum(step_costs)

    print("  Computing ground-truth arrival times via AVBD...")
    t0 = time.time()
    # Process in chunks to manage memory
    chunk_size = 200
    n_total = grid_points.shape[0]
    all_times = []
    for i in range(0, n_total, chunk_size):
        chunk = grid_points[i:i + chunk_size]
        times = jax.vmap(single_distance)(chunk)
        all_times.append(times)
        if (i // chunk_size) % 5 == 0:
            print(f"    Chunk {i // chunk_size + 1}/"
                  f"{(n_total + chunk_size - 1) // chunk_size}")
    result = jnp.concatenate(all_times)
    print(f"  Ground truth computed in {time.time() - t0:.1f}s")
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_grid(grid_size, margin=0.05):
    """Create a uniform grid in [margin, 1-margin]^2.

    Args:
        grid_size: Number of points per axis.
        margin: Boundary margin to avoid edge effects.

    Returns:
        Grid points, shape (grid_size^2, 2).
    """
    xs = jnp.linspace(margin, 1.0 - margin, grid_size)
    ys = jnp.linspace(margin, 1.0 - margin, grid_size)
    xx, yy = jnp.meshgrid(xs, ys)
    return jnp.stack([xx.ravel(), yy.ravel()], axis=-1)


def jacobian_regularization(metric, sample_points):
    """Squared Jacobian (Tikhonov) regularization on the metric field.

    Penalizes ||dG/dx||_F^2 to encourage spatial smoothness. This is an L2
    penalty (Tikhonov), not L1 total variation. For piecewise-constant
    recovery, L1-TV would have better inductive bias but is non-differentiable.
    We use L2 as it is smooth and compatible with gradient-based optimization.

    Args:
        metric: FinslerMetric with _get_zermelo_data or g_net.
        sample_points: Points to evaluate, shape (N, 2).

    Returns:
        Scalar penalty.
    """
    def get_metric_value(x):
        if hasattr(metric, '_get_zermelo_data'):
            H, _, _ = metric._get_zermelo_data(x)
            return H.ravel()
        elif hasattr(metric, 'g_net'):
            return metric.g_net(x).ravel()
        return jnp.eye(x.shape[0]).ravel()

    jac_fn = jax.jacfwd(get_metric_value)
    jacs = jax.vmap(jac_fn)(sample_points)  # (N, D*D, D)
    return jnp.mean(jacs ** 2)


def tv_regularization(metric):
    """Total Variation regularization on grid-based metric fields.

    Computes L1 TV: sum of absolute differences between adjacent grid values.
    Encourages piecewise-constant solutions, matching Gahtan et al.'s approach.

    Args:
        metric: Riemannian metric with GridMetricField g_net.

    Returns:
        Scalar TV penalty.
    """
    g = metric.g_net.grid_values
    dx = jnp.abs(g[:, 1:] - g[:, :-1])
    dy = jnp.abs(g[1:, :] - g[:-1, :])
    return jnp.mean(dx) + jnp.mean(dy)


def train_metric(
    key, grid_points, source, t_obs, cfg,
    reg_lambda=1e-3, density=1.0, use_wind=False,
):
    """Train a metric to recover arrival times.

    Supports two parameterization modes (cfg['param_type']):
      - 'grid': Per-pixel scalar metric field (matches Gahtan et al.).
                Uses TV regularization. No wind (Riemannian only).
      - 'neural': NeuralRanders with PSDMatrixField/VectorField.
                  Uses L2 Jacobian regularization.

    Args:
        key: JAX PRNG key.
        grid_points: Full grid, shape (N, 2).
        source: Source point, shape (2,).
        t_obs: Ground-truth arrival times for grid_points, shape (N,).
        cfg: Configuration dict.
        reg_lambda: Regularization weight (TV for grid, Jacobian for neural).
        density: Fraction of observations to use.
        use_wind: If True, learn both G and W (Randers). If False, W=0
            (Riemannian). For isotropic ground truth, W=0 is the correct
            control — Randers should not improve and may hurt.
            Ignored for param_type='grid' (always Riemannian).

    Returns:
        Tuple of (trained_metric, loss_history).
    """
    k_init, k_train = jax.random.split(key)
    param_type = cfg.get('param_type', 'grid')

    # Subsample observations
    n_total = grid_points.shape[0]
    n_obs = max(4, int(n_total * density))
    obs_idx = jax.random.choice(k_init, n_total, shape=(n_obs,), replace=False)
    obs_idx = jnp.sort(obs_idx)
    x_train = grid_points[obs_idx]
    t_train = t_obs[obs_idx]

    # Initialize metric
    manifold = EuclideanSpace(2)
    k_net, k_train = jax.random.split(k_train)

    if param_type == 'grid':
        # Grid-based per-pixel metric (matches Gahtan's parameterization)
        g_field = GridMetricField(cfg['grid_size'], margin=0.05)
        metric = Riemannian(manifold, g_field)
        use_tv = True
        tv_lambda = cfg.get('tv_lambda', 0.001)
    else:
        # Neural network metric
        k1, k2 = jax.random.split(k_net)
        h_net = PSDMatrixField(2, cfg['hidden_dim'], cfg['depth'], k1)
        w_net = VectorField(2, cfg['hidden_dim'], cfg['depth'], k2,
                            use_fourier=False)
        metric = Randers(manifold, h_net, w_net, epsilon=1e-5,
                         use_wind=use_wind)
        use_tv = False
        tv_lambda = 0.0

    # Solver and loss
    solver = AVBDSolver(
        step_size=0.05,
        iterations=cfg['solver_iters'],
        energy_tol=1e-6,
    )
    arrival_loss = ArrivalTimeLoss(
        solver=solver,
        solver_steps=cfg['solver_steps'],
    )

    # Optimizer with cosine decay
    schedule = optax.cosine_decay_schedule(cfg['lr'], cfg['n_train_steps'])
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(eqx.filter(metric, eqx.is_array))

    # Fixed regularization sample points for neural mode
    n_reg = cfg['n_reg_points']
    reg_points = None
    if not use_tv and reg_lambda > 0:
        k_reg, k_train = jax.random.split(k_train)
        reg_idx = jax.random.choice(k_reg, n_obs,
                                     shape=(min(n_reg, n_obs),), replace=False)
        reg_points = x_train[reg_idx]
        if reg_points.shape[0] < n_reg:
            pad = jnp.tile(reg_points[:1], (n_reg - reg_points.shape[0], 1))
            reg_points = jnp.concatenate([reg_points, pad])

    if use_tv:
        @eqx.filter_jit
        def train_step(metric, opt_state, batch_x, batch_t):
            def loss_fn(m):
                l_arrival = arrival_loss(m, source, batch_x, batch_t)
                l_tv = tv_regularization(m) * tv_lambda if tv_lambda > 0 else 0.0
                return l_arrival + l_tv, l_arrival
            (total_loss, arrival_only), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )(metric)
            updates, new_opt_state = optimizer.update(grads, opt_state, metric)
            new_metric = eqx.apply_updates(metric, updates)
            return new_metric, new_opt_state, total_loss, arrival_only
    else:
        @eqx.filter_jit
        def train_step(metric, opt_state, batch_x, batch_t):
            def loss_fn(m):
                l_arrival = arrival_loss(m, source, batch_x, batch_t)
                l_reg = jacobian_regularization(m, reg_points) if reg_lambda > 0 else 0.0
                return l_arrival + reg_lambda * l_reg, l_arrival
            (total_loss, arrival_only), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )(metric)
            updates, new_opt_state = optimizer.update(grads, opt_state, metric)
            new_metric = eqx.apply_updates(metric, updates)
            return new_metric, new_opt_state, total_loss, arrival_only

    # Training loop
    loss_history = []
    batch_size = min(cfg['obs_per_train_step'], n_obs)
    param_str = f"[{param_type}]"
    wind_str = "Randers" if (use_wind and param_type != 'grid') else "Riemannian"
    reg_str = f"TV λ={tv_lambda:.0e}" if use_tv else f"Jac λ={reg_lambda:.0e}"
    print(f"  Training {param_str} [{wind_str}]: {n_obs} obs ({density*100:.0f}%), "
          f"{reg_str}, batch={batch_size}")

    for step in range(cfg['n_train_steps']):
        k_train, k_batch = jax.random.split(k_train)
        batch_idx = jax.random.choice(k_batch, n_obs, shape=(batch_size,),
                                       replace=False)
        bx = x_train[batch_idx]
        bt = t_train[batch_idx]

        metric, opt_state, loss, arrival_only = train_step(
            metric, opt_state, bx, bt
        )

        # Project grid values to stay positive (grid mode only)
        if use_tv and step % 50 == 0:
            new_gv = jnp.maximum(metric.g_net.grid_values, 0.1)
            metric = eqx.tree_at(lambda m: m.g_net.grid_values, metric, new_gv)

        loss_val = float(loss)
        loss_history.append(loss_val)

        if step % 50 == 0 or step == cfg['n_train_steps'] - 1:
            print(f"    Step {step:4d}/{cfg['n_train_steps']}: "
                  f"loss={loss_val:.6f}  arrival={float(arrival_only):.6f}")

    return metric, loss_history


def evaluate_recovery(metric, grid_points, true_metric_fn, grid_size):
    """Compute relative metric recovery error.

    Args:
        metric: Trained metric.
        grid_points: Evaluation grid, shape (N, 2).
        true_metric_fn: Function mapping x -> scalar metric value.
        grid_size: Grid side length for reshaping.

    Returns:
        Dict with error statistics and fields for plotting.
    """
    g_true = jax.vmap(true_metric_fn)(grid_points)

    def learned_scalar(x):
        if hasattr(metric, '_get_zermelo_data'):
            H, _, _ = metric._get_zermelo_data(x)
        elif hasattr(metric, 'g_net'):
            H = metric.g_net(x)
        else:
            H = jnp.eye(2)
        # Isotropic: trace/dim gives scalar metric value
        return jnp.trace(H) / 2.0

    g_learned = jax.vmap(learned_scalar)(grid_points)
    rel_error = jnp.sqrt(jnp.mean((g_learned - g_true)**2)) / jnp.mean(g_true)

    return dict(
        rel_error=float(rel_error),
        g_true=np.array(g_true.reshape(grid_size, grid_size)),
        g_learned=np.array(g_learned.reshape(grid_size, grid_size)),
        g_error=np.array(
            jnp.abs(g_learned - g_true).reshape(grid_size, grid_size)),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def plot_metric_recovery(eval_result, density, fig_dir, suffix=""):
    """Plot true vs. recovered metric heatmaps (3-panel)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    vmin = min(eval_result['g_true'].min(), eval_result['g_learned'].min())
    vmax = max(eval_result['g_true'].max(), eval_result['g_learned'].max())

    im0 = axes[0].imshow(eval_result['g_true'], origin='lower',
                         cmap='viridis', vmin=vmin, vmax=vmax,
                         extent=[0, 1, 0, 1])
    axes[0].set_title('True metric $g(x)$')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(eval_result['g_learned'], origin='lower',
                         cmap='viridis', vmin=vmin, vmax=vmax,
                         extent=[0, 1, 0, 1])
    axes[1].set_title(f'Learned metric (density={density*100:.0f}%)')
    axes[1].set_xlabel('$x_1$')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(eval_result['g_error'], origin='lower',
                         cmap='Reds', extent=[0, 1, 0, 1])
    axes[2].set_title(f'|Error| (rel. = {eval_result["rel_error"]*100:.1f}%)')
    axes[2].set_xlabel('$x_1$')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle('Metric Recovery: True vs. Learned', fontsize=14, y=1.02)
    plt.tight_layout()

    fname = os.path.join(fig_dir, f'phase1_metric_recovery{suffix}')
    fig.savefig(fname + '.pdf')
    fig.savefig(fname + '.png')
    plt.close(fig)
    print(f"  Saved: {fname}.pdf")


def plot_convergence(loss_histories, labels, fig_dir):
    """Plot training loss convergence curves (semi-log)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for hist, label in zip(loss_histories, labels):
        ax.semilogy(hist, label=label, linewidth=1.5)

    ax.set_xlabel('Training step')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Convergence')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(fig_dir, 'phase1_convergence')
    fig.savefig(fname + '.pdf')
    fig.savefig(fname + '.png')
    plt.close(fig)
    print(f"  Saved: {fname}.pdf")


def plot_density_sweep(densities, errors_mean, errors_std, fig_dir):
    """Plot error vs. observation density with Gahtan baselines.

    Note: Gahtan baselines are single-run values from paper Section 5;
    no error bars were reported in the original paper.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    dens_pct = [d * 100 for d in densities]
    ax.errorbar(dens_pct, [e * 100 for e in errors_mean],
                yerr=[e * 100 for e in errors_std],
                fmt='o-', color='#2196F3', linewidth=2, capsize=5, capthick=1.5,
                label='HAMTools (Lagrangian)', markersize=8, zorder=3)

    # Gahtan baselines (single-run, no error bars reported)
    gahtan_dens = [100, 7]
    gahtan_err = [5.6, 21.2]
    ax.scatter(gahtan_dens, gahtan_err, marker='D', s=100, color='#FF5722',
               zorder=4, label='Gahtan et al. (Eikonal, single run)',
               edgecolors='black')

    ax.set_xlabel('Observation density (%)')
    ax.set_ylabel('Relative recovery error (%)')
    ax.set_title('Metric Recovery Error vs. Observation Density')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 105)
    ax.set_ylim(0, None)
    plt.tight_layout()

    fname = os.path.join(fig_dir, 'phase1_density_sweep')
    fig.savefig(fname + '.pdf')
    fig.savefig(fname + '.png')
    plt.close(fig)
    print(f"  Saved: {fname}.pdf")


def plot_regularization_sweep(reg_lambdas, errors_mean, errors_std, fig_dir):
    """Plot U-curve: error vs. Jacobian regularization strength."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_vals = [max(lam, 1e-5) for lam in reg_lambdas]

    ax.errorbar(x_vals, [e * 100 for e in errors_mean],
                yerr=[e * 100 for e in errors_std],
                fmt='s-', color='#4CAF50', linewidth=2, capsize=5, capthick=1.5,
                markersize=8)

    ax.set_xscale('log')
    ax.set_xlabel('Jacobian regularization $\\lambda$')
    ax.set_ylabel('Relative recovery error (%)')
    ax.set_title('Regularization Sweep (L2 Jacobian Penalty)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = os.path.join(fig_dir, 'phase1_regularization')
    fig.savefig(fname + '.pdf')
    fig.savefig(fname + '.png')
    plt.close(fig)
    print(f"  Saved: {fname}.pdf")


def plot_geodesic_paths(metric, source, targets, grid_points, grid_size,
                        true_metric_fn, solver, solver_steps, fig_dir):
    """Plot geodesic paths overlaid on the metric field.

    This visualization is impossible with the eikonal approach, which only
    produces arrival time fields without explicit paths. It demonstrates
    a key advantage of the Lagrangian formulation.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    g_true = np.array(jax.vmap(true_metric_fn)(grid_points).reshape(
        grid_size, grid_size))
    ax.imshow(g_true, origin='lower', cmap='bone', alpha=0.4,
              extent=[0, 1, 0, 1])

    cmap = plt.cm.plasma
    t_max = 0.0

    @eqx.filter_jit
    def compute_path(src, tgt):
        return solver.solve(metric, src, tgt, n_steps=solver_steps)

    paths = []
    for target in targets:
        traj = compute_path(source, target)
        path = np.array(traj.xs)
        segments = np.diff(path, axis=0)
        seg_lengths = np.sqrt(np.sum(segments**2, axis=1))
        cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
        paths.append((path, cum_length))
        t_max = max(t_max, cum_length[-1])

    for path, cum_length in paths:
        for i in range(len(path) - 1):
            color = cmap(cum_length[i] / max(t_max, 1e-6))
            ax.plot(path[i:i+2, 0], path[i:i+2, 1], color=color, linewidth=1.8)

    ax.plot(float(source[0]), float(source[1]), 'r*', markersize=15,
            markeredgecolor='black', markeredgewidth=0.5, zorder=5,
            label='Source')

    for t in targets:
        ax.plot(float(t[0]), float(t[1]), 'o', color='white', markersize=5,
                markeredgecolor='black', markeredgewidth=0.5, zorder=4)

    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5,
               label='Metric boundary')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, t_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Arrival time (arc length)')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('Geodesic Paths from Source\n(Lagrangian — HAMTools exclusive)')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.tight_layout()

    fname = os.path.join(fig_dir, 'phase1_geodesic_paths')
    fig.savefig(fname + '.pdf')
    fig.savefig(fname + '.png')
    plt.close(fig)
    print(f"  Saved: {fname}.pdf")


def plot_arrival_time_comparison(metric, source, grid_points, grid_size,
                                  t_true, solver, solver_steps, fig_dir):
    """Plot predicted vs. true arrival time fields side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    t_true_grid = np.array(t_true.reshape(grid_size, grid_size))

    n_eval = min(200, grid_points.shape[0])
    key = jax.random.PRNGKey(99)
    eval_idx = jax.random.choice(key, grid_points.shape[0],
                                  shape=(n_eval,), replace=False)

    @eqx.filter_jit
    def single_dist(x_target):
        traj = solver.solve(metric, source, x_target, n_steps=solver_steps)
        path = traj.xs
        segments = jnp.diff(path, axis=0)
        midpoints = (path[:-1] + path[1:]) / 2.0
        step_costs = jax.vmap(metric.metric_fn)(midpoints, segments)
        return jnp.sum(step_costs)

    t_pred_sample = jax.vmap(single_dist)(grid_points[eval_idx])
    t_obs_sample = t_true[eval_idx]

    vmin = min(t_true_grid.min(), 0)
    vmax = t_true_grid.max()

    im0 = axes[0].imshow(t_true_grid, origin='lower', cmap='inferno',
                          extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[0].plot(float(source[0]), float(source[1]), 'w*', markersize=12)
    axes[0].set_title('True arrival time $T(x)$')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    pts_eval = np.array(grid_points[eval_idx])
    t_p = np.array(t_pred_sample)
    sc = axes[1].scatter(pts_eval[:, 0], pts_eval[:, 1], c=t_p,
                          cmap='inferno', s=15, vmin=vmin, vmax=vmax)
    axes[1].plot(float(source[0]), float(source[1]), 'w*', markersize=12)
    axes[1].set_title('Predicted arrival time (AVBD)')
    axes[1].set_xlabel('$x_1$')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_aspect('equal')
    plt.colorbar(sc, ax=axes[1], shrink=0.8)

    t_obs_np = np.array(t_obs_sample)
    axes[2].scatter(t_obs_np, t_p, alpha=0.5, s=15, color='#2196F3')
    max_val = max(t_obs_np.max(), t_p.max())
    axes[2].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect')
    axes[2].set_xlabel('True arrival time')
    axes[2].set_ylabel('Predicted arrival time')
    corr = np.corrcoef(t_obs_np, t_p)[0, 1]
    axes[2].set_title(f'Correlation: {corr:.3f}')
    axes[2].legend()
    axes[2].set_aspect('equal')

    fig.suptitle('Arrival Time: True vs. Predicted', fontsize=14, y=1.02)
    plt.tight_layout()

    fname = os.path.join(fig_dir, 'phase1_arrival_times')
    fig.savefig(fname + '.pdf')
    fig.savefig(fname + '.png')
    plt.close(fig)
    print(f"  Saved: {fname}.pdf")


def plot_riemannian_vs_randers(riem_errors, randers_errors, densities, fig_dir):
    """Plot Riemannian (W=0) vs. Randers ablation comparison.

    For isotropic ground truth, Riemannian (W=0) should perform at least as
    well as Randers. If Randers performs worse, the extra wind parameters are
    absorbing noise — expected behaviour confirming correct experimental design.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    dens_pct = [d * 100 for d in densities]
    x_pos = np.arange(len(densities))
    width = 0.35

    ax.bar(x_pos - width / 2,
           [e * 100 for e in riem_errors],
           width=width, color='#2196F3', label='Riemannian (W=0)', alpha=0.8)
    ax.bar(x_pos + width / 2,
           [e * 100 for e in randers_errors],
           width=width, color='#FF9800', label='Randers (W learned)', alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{d:.0f}%' for d in dens_pct])
    ax.set_xlabel('Observation density')
    ax.set_ylabel('Relative recovery error (%)')
    ax.set_title('Riemannian vs. Randers Ablation\n'
                 '(Isotropic ground truth — W should not help)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    fname = os.path.join(fig_dir, 'phase1_riemannian_vs_randers')
    fig.savefig(fname + '.pdf')
    fig.savefig(fname + '.png')
    plt.close(fig)
    print(f"  Saved: {fname}.pdf")


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_experiment(cfg):
    """Execute the full Phase 1 experiment.

    Steps:
        1. Generate ground-truth metric and arrival times.
        2. Run density sweep (Riemannian, W=0).
        3. Run Randers ablation at selected densities.
        4. Run Jacobian regularization sweep at 100% density.
        5. Generate all visualization figures.
        6. Print summary table.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    setup_style()
    print("=" * 70)
    print("Phase 1: Synthetic Isotropic Metric Recovery")
    print("=" * 70)

    # ----- Data Generation -----
    print("\n[1] Generating ground-truth data...")
    grid_size = cfg['grid_size']
    grid = make_grid(grid_size)
    # Source placed in the g=1 region, away from boundary at x=0.5
    source = jnp.array([0.35, 0.5])
    t_true = compute_true_arrival_times(source, grid, cfg, boundary=0.5)
    print(f"  Grid: {grid_size}x{grid_size} = {grid.shape[0]} points")
    print(f"  Source: {source} (in g=1 region, off boundary)")
    print(f"  Arrival time range: "
          f"[{float(t_true.min()):.3f}, {float(t_true.max()):.3f}]")

    # ----- Density Sweep (Riemannian, W=0) -----
    print(f"\n[2] Density sweep (Riemannian, W=0): {cfg['densities']}")
    density_results = {}
    all_loss_histories = []
    all_loss_labels = []

    for density in cfg['densities']:
        errors_for_density = []
        for seed in cfg['seeds']:
            print(f"\n  --- Density={density*100:.0f}%, Seed={seed} ---")
            key = jax.random.PRNGKey(seed)
            t0 = time.time()
            metric, loss_hist = train_metric(
                key, grid, source, t_true, cfg,
                reg_lambda=0.0, density=density, use_wind=False,
            )
            dt_train = time.time() - t0
            print(f"  Training time: {dt_train:.1f}s")

            result = evaluate_recovery(metric, grid, true_metric_scalar,
                                        grid_size)
            errors_for_density.append(result['rel_error'])
            print(f"  Relative error: {result['rel_error']*100:.1f}%")

            if seed == cfg['seeds'][0]:
                all_loss_histories.append(loss_hist)
                all_loss_labels.append(f"density={density*100:.0f}%")
                if density in (cfg['densities'][0], cfg['densities'][-1]):
                    sfx = f"_{density*100:.0f}pct"
                    plot_metric_recovery(result, density, FIG_DIR, suffix=sfx)

        errors_arr = np.array(errors_for_density)
        density_results[density] = {
            'mean': float(errors_arr.mean()),
            'std': float(errors_arr.std()),
            'all': errors_for_density,
        }

    dens_list = list(density_results.keys())
    errs_mean = [density_results[d]['mean'] for d in dens_list]
    errs_std = [density_results[d]['std'] for d in dens_list]
    plot_density_sweep(dens_list, errs_mean, errs_std, FIG_DIR)
    plot_convergence(all_loss_histories, all_loss_labels, FIG_DIR)

    # ----- Randers Ablation (neural mode only) -----
    randers_results = {}
    ablation_densities = [cfg['densities'][0], cfg['densities'][-1]]
    if cfg.get('param_type', 'grid') == 'neural':
        print(f"\n[3] Randers ablation (W learned) at selected densities")
        for density in ablation_densities:
            errors_randers = []
            for seed in cfg['seeds']:
                key = jax.random.PRNGKey(seed)
                metric_r, _ = train_metric(
                    key, grid, source, t_true, cfg,
                    reg_lambda=1e-3, density=density, use_wind=True,
                )
                result_r = evaluate_recovery(metric_r, grid, true_metric_scalar,
                                              grid_size)
                errors_randers.append(result_r['rel_error'])
            errs_r = np.array(errors_randers)
            randers_results[density] = {
                'mean': float(errs_r.mean()),
                'std': float(errs_r.std()),
            }
            print(f"  Randers at {density*100:.0f}%: "
                  f"{errs_r.mean()*100:.1f}% +/- {errs_r.std()*100:.1f}%")

        riem_errs = [density_results[d]['mean'] for d in ablation_densities]
        rand_errs = [randers_results[d]['mean'] for d in ablation_densities]
        plot_riemannian_vs_randers(riem_errs, rand_errs, ablation_densities,
                                   FIG_DIR)
    else:
        print(f"\n[3] Randers ablation skipped (grid param_type)")

    # ----- Regularization Sweep -----
    print(f"\n[4] TV/Jacobian regularization sweep: {cfg['reg_lambdas']}")
    reg_results = {}
    for lam in cfg['reg_lambdas']:
        errors_for_lam = []
        for seed in cfg['seeds']:
            key = jax.random.PRNGKey(seed)
            # For grid mode, reg_lambda controls TV; for neural, Jacobian
            cfg_sweep = {**cfg, 'tv_lambda': lam}
            metric, _ = train_metric(
                key, grid, source, t_true, cfg_sweep,
                reg_lambda=lam, density=1.0, use_wind=False,
            )
            result = evaluate_recovery(metric, grid, true_metric_scalar,
                                        grid_size)
            errors_for_lam.append(result['rel_error'])
        errs = np.array(errors_for_lam)
        reg_results[lam] = {'mean': float(errs.mean()),
                             'std': float(errs.std())}
        print(f"  lambda={lam:.0e}: error="
              f"{errs.mean()*100:.1f}% +/- {errs.std()*100:.1f}%")

    reg_lams = list(reg_results.keys())
    reg_mean = [reg_results[l]['mean'] for l in reg_lams]
    reg_std = [reg_results[l]['std'] for l in reg_lams]
    plot_regularization_sweep(reg_lams, reg_mean, reg_std, FIG_DIR)

    # ----- Geodesic Path Visualization -----
    print("\n[5] Generating geodesic path visualization...")
    key = jax.random.PRNGKey(0)
    metric_best, _ = train_metric(
        key, grid, source, t_true, cfg,
        reg_lambda=1e-3, density=1.0, use_wind=False,
    )
    solver = AVBDSolver(step_size=0.05, iterations=cfg['solver_iters'])

    n_targets = 24
    angles = np.linspace(0, 2 * np.pi, n_targets, endpoint=False)
    radii = [0.25, 0.38]
    targets = []
    for r in radii:
        for theta in angles:
            tx = float(source[0]) + r * np.cos(theta)
            ty = float(source[1]) + r * np.sin(theta)
            if 0.05 < tx < 0.95 and 0.05 < ty < 0.95:
                targets.append(jnp.array([tx, ty]))

    plot_geodesic_paths(metric_best, source, targets, grid,
                        grid_size, true_metric_scalar, solver,
                        cfg['solver_steps'], FIG_DIR)

    # ----- Arrival Time Comparison -----
    print("\n[6] Generating arrival time comparison...")
    plot_arrival_time_comparison(metric_best, source, grid, grid_size,
                                 t_true, solver, cfg['solver_steps'], FIG_DIR)

    # ----- Summary Table -----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n--- Density Sweep ({cfg.get('param_type', 'grid')} param) ---")
    print(f"{'Density':>10} | {'Error (ours)':>20} | "
          f"{'Gahtan (single run)':>20}")
    print("-" * 58)
    gahtan = {1.0: 5.6, 0.07: 21.2}
    for d in dens_list:
        ours = (f"{density_results[d]['mean']*100:.1f}% "
                f"+/- {density_results[d]['std']*100:.1f}%")
        gahtan_val = f"{gahtan[d]}%" if d in gahtan else "—"
        print(f"{d*100:>9.0f}% | {ours:>20} | {gahtan_val:>20}")

    if randers_results:
        print(f"\n--- Riemannian vs. Randers Ablation ---")
        for d in ablation_densities:
            print(f"  {d*100:.0f}%: "
                  f"Riemannian={density_results[d]['mean']*100:.1f}%, "
                  f"Randers={randers_results[d]['mean']*100:.1f}%")

    print(f"\n--- Regularization Sweep ---")
    print(f"{'lambda':>10} | {'Error':>20}")
    print("-" * 35)
    for lam in reg_lams:
        err_str = (f"{reg_results[lam]['mean']*100:.1f}% "
                   f"+/- {reg_results[lam]['std']*100:.1f}%")
        print(f"{lam:>10.0e} | {err_str:>20}")

    print(f"\nFigures saved to: {os.path.abspath(FIG_DIR)}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Synthetic Metric Recovery Experiment"
    )
    parser.add_argument('--quick', action='store_true',
                        help='Run with reduced grid/iterations for smoke testing')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'gpu', 'tpu'],
                        help='JAX device to use (default: cpu).')
    args = parser.parse_args()

    from ham.utils import configure_device
    configure_device(args.device)

    cfg = get_config(quick=args.quick)
    run_experiment(cfg)


if __name__ == '__main__':
    main()
