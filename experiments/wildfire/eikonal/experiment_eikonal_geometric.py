#!/usr/bin/env python3
"""
Phase W3: Geometric Analysis of Learned Wildfire Metrics
=========================================================

Produces HAMTools-exclusive visualizations using a trained
CovariateConditionedRanders (or CovariateMeshRanders) metric:

  1. **Fire corridor geodesics** — geodesic fan from ignition point, colored by
     predicted arrival time.  Impossible with Gahtan's eikonal approach.
  2. **Jacobi divergence** — rate at which neighboring geodesics separate,
     indicating unstable fire spread directions.
  3. **Curvature anomaly map** — Finsler flag curvature sampled on a grid,
     identifying terrain/fuel features that abruptly change fire dynamics.

**Usage**::

    # Synthetic smoke-test (no dataset required):
    python examples/experiment_wildfire_geometric.py --synthetic

    # With a trained flat-grid metric checkpoint:
    python examples/experiment_eikonal_geometric.py \
        --checkpoint results/phaseEikonal1/metric_seed0.eqx \\
        --data_root /data/sim2real_fire --scene 0014_00426

**HAMTools spec references:**
    spec/MATH_SPEC.md §§ 1–2 (geodesic spray), § 3 (parallel transport)
    spec/ARCH_SPEC.md § 4.2 (AVBDSolver)
"""

import argparse
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ham.data.wildfire import WildfireScenario
from ham.geometry.curvature import flag_curvature_sample

# config.update("jax_enable_x64", True)
from ham.geometry.manifolds import EuclideanSpace
from ham.models.wildfire import CovariateConditionedRanders
from ham.solvers.avbd import AVBDSolver

FIG_DIR = "results/phaseEikonal2/figs"


# ---------------------------------------------------------------------------
# Geodesic fan
# ---------------------------------------------------------------------------


def shoot_geodesic_fan(
    metric, solver, source_world, domain_size, n_directions=64, n_steps=50
):
    """Shoot n_directions geodesics from source_world in uniformly-spaced angles.

    Args:
        metric:        Bound FinslerMetric.
        solver:        AVBDSolver.
        source_world:  (2,) world-coord source.
        domain_size:   Scalar — approximate domain extent; geodesics reach
                       source_world ± 0.4 * domain_size.
        n_directions:  Number of angular directions.
        n_steps:       AVBD path discretization.

    Returns:
        paths: (n_directions, n_steps+1, 2) array of path vertices.
        arc_lengths: (n_directions,) predicted arrival times.
    """
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_directions, endpoint=False)
    radius = 0.4 * domain_size
    targets = source_world + radius * jnp.stack(
        [jnp.cos(angles), jnp.sin(angles)], axis=-1
    )

    @eqx.filter_jit
    def all_paths_fn(ts):
        """vmap over all target directions in a single compiled call."""

        def one(target):
            traj = solver.solve(metric, source_world, target, n_steps=n_steps)
            path = traj.xs  # (n_steps+1, 2)
            segs = jnp.diff(path, axis=0)
            mids = (path[:-1] + path[1:]) / 2.0
            costs = jax.vmap(metric.metric_fn)(mids, segs)
            return path, jnp.sum(costs)

        return jax.vmap(one)(ts)

    paths_jax, arcs_jax = all_paths_fn(targets)
    return np.array(paths_jax), np.array(arcs_jax)


# ---------------------------------------------------------------------------
# Jacobi divergence
# ---------------------------------------------------------------------------


def compute_jacobi_divergence(paths):
    """Estimate geodesic bundle divergence over time.

    For each time step t, computes the mean distance between adjacent
    geodesics.  A rising curve indicates unstable fire spread (open terrain);
    a falling or flat curve indicates channelling (valley / firebreak).

    Args:
        paths: (D, T+1, 2) array of path vertices, D directions, T+1 points.

    Returns:
        divergence: (T+1,) array of mean inter-geodesic distances.
    """
    D, T1, _ = paths.shape
    # Adjacent pairs: (i, i+1 mod D)
    diffs = np.linalg.norm(paths - np.roll(paths, shift=1, axis=0), axis=-1)  # (D, T+1)
    return diffs.mean(axis=0)  # (T+1,)


# ---------------------------------------------------------------------------
# Curvature field
# ---------------------------------------------------------------------------


def compute_curvature_field(metric, grid_xy, seed=0):
    """Sample scalar Finsler flag curvature on a set of 2D points.

    Uses `flag_curvature_sample` from `ham.geometry.curvature` with independent
    random keys per point.  High-curvature regions correspond to abrupt changes
    in the fire metric (fuel boundaries, wind shear).

    Args:
        metric:   Bound CovariateConditionedRanders (2D, flat).
        grid_xy:  (N, 2) sample points in world coordinates.
        seed:     Integer PRNG seed for flag direction sampling.

    Returns:
        kappa: (N,) scalar curvature values.  May contain NaN near metric
               singularities or for near-degenerate metrics in early training.
    """
    base_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(base_key, grid_xy.shape[0])

    results = []
    for i in range(grid_xy.shape[0]):
        try:
            k = flag_curvature_sample(metric, jnp.asarray(grid_xy[i]), keys[i])
            results.append(float(k))
        except Exception:
            results.append(float("nan"))
    return np.array(results)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _setup_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_fire_corridors(
    paths, arc_lengths, arrival_field, domain_shape, source_world, output_dir, suffix=""
):
    """Plot geodesic paths over arrival time heatmap background.

    Args:
        paths:          (D, T+1, 2) path vertices in world coords.
        arc_lengths:    (D,) predicted arrival times per path.
        arrival_field:  (H, W) ground-truth arrival time array (may be None).
        domain_shape:   (H, W) of the raster domain.
        source_world:   (2,) ignition point in world coords.
        output_dir:     Output directory for figures.
        suffix:         String appended to filename.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    H, W = domain_shape
    if arrival_field is not None:
        arr = np.where(np.isinf(arrival_field), np.nan, arrival_field)
        im = ax.imshow(
            arr,
            origin="lower",
            cmap="YlOrRd",
            alpha=0.55,
            extent=[0, W, 0, H],
            aspect="auto",
        )
        plt.colorbar(im, ax=ax, shrink=0.75, label="GT arrival time")

    # Color each path by its arc length (predicted arrival time)
    t_min = arc_lengths.min()
    t_max = arc_lengths.max() + 1e-9
    cmap = plt.cm.plasma
    for path, arc in zip(paths, arc_lengths):
        c = cmap((arc - t_min) / (t_max - t_min))
        ax.plot(path[:, 0], path[:, 1], color=c, linewidth=1.2, alpha=0.8)

    ax.plot(
        float(source_world[0]),
        float(source_world[1]),
        "r*",
        markersize=14,
        markeredgecolor="black",
        markeredgewidth=0.5,
        zorder=5,
        label="Ignition",
    )
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(
        "Fire Corridor Geodesics\n(HAMTools exclusive — no eikonal equivalent)"
    )
    ax.legend(loc="lower right")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t_min, t_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.75, label="Predicted arrival time")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"phaseEikonal2_corridors{suffix}.{ext}"))
    plt.close(fig)
    print(f"  Saved: phaseEikonal2_corridors{suffix}.pdf/png")


def plot_jacobi_divergence(divergence, output_dir, suffix=""):
    """Plot geodesic bundle divergence over time.

    Args:
        divergence: (T+1,) mean inter-geodesic distance per time step.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    t_axis = np.linspace(0.0, 1.0, len(divergence))
    ax.plot(t_axis, divergence, linewidth=2.0, color="#E53935")
    peak_t = t_axis[np.argmax(divergence)]
    ax.axvline(
        peak_t,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"Peak spread at t={peak_t:.2f}",
    )
    ax.fill_between(t_axis, divergence, alpha=0.15, color="#E53935")
    ax.set_xlabel("Normalized time along geodesic")
    ax.set_ylabel("Mean inter-geodesic distance")
    ax.set_title(
        "Jacobi Divergence (Fire Spread Stability)\n"
        "Rising = unstable spread,  Flat/falling = channelled"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"phaseEikonal2_jacobi{suffix}.{ext}"))
    plt.close(fig)
    print(f"  Saved: phaseEikonal2_jacobi{suffix}.pdf/png")


def plot_curvature_field(kappa, grid_xy, domain_shape, output_dir, suffix=""):
    """Plot curvature values as a 2D scatter / heatmap.

    Args:
        kappa:        (N,) scalar curvature values.
        grid_xy:      (N, 2) sample positions.
        domain_shape: (H, W) for axis limits.
        output_dir:   Output directory.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    H, W = domain_shape
    valid = ~np.isnan(kappa)
    if valid.sum() == 0:
        print("  WARNING: all curvature values are NaN — skipping figure.")
        plt.close(fig)
        return

    sc = ax.scatter(
        grid_xy[valid, 0],
        grid_xy[valid, 1],
        c=kappa[valid],
        cmap="RdBu_r",
        s=12,
        alpha=0.85,
    )
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Flag curvature κ")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(
        "Finsler Flag Curvature Anomaly Map\n"
        "High |κ| → abrupt metric change (fuel boundary / wind shear)"
    )
    ax.set_aspect("equal")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"phaseEikonal2_curvature{suffix}.{ext}"))
    plt.close(fig)
    print(f"  Saved: phaseEikonal2_curvature{suffix}.pdf/png")


# ---------------------------------------------------------------------------
# Synthetic scenario
# ---------------------------------------------------------------------------


def _make_synthetic_scenario():
    """Build a 20x20 synthetic WildfireScenario for smoke-testing."""
    H, W = 20, 20
    spacing = 30.0
    rng = np.random.default_rng(42)
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    elev = 100.0 + 30.0 * np.sin(np.pi * rows / H) * np.cos(np.pi * cols / W)
    slope = np.abs(np.gradient(elev, spacing)[0]) / spacing
    aspect = np.arctan2(np.gradient(elev, spacing)[1], np.gradient(elev, spacing)[0])
    canopy = rng.uniform(0, 1, (H, W))
    fuel_codes = np.ones((H, W), dtype=np.int32) * 5

    ign_row, ign_col = 10, 10
    dists = np.sqrt((rows - ign_row) ** 2 + (cols - ign_col) ** 2)
    t_hours = dists * 0.03
    t_max = t_hours.max()
    t_norm = t_hours / t_max
    burned_mask = np.ones((H, W), dtype=bool)
    t_hours_inf = t_hours.copy()

    n_obs = 50
    rng2 = np.random.default_rng(0)
    flat_idx = rng2.choice(H * W, n_obs, replace=False)
    obs_pixels = np.stack(np.unravel_index(flat_idx, (H, W)), axis=-1)
    obs_times = t_norm.ravel()[flat_idx]

    return WildfireScenario(
        scene_id="synthetic",
        event_id="synthetic_fire_0",
        ignition_pixel=np.array([ign_row, ign_col], dtype=float),
        ignition_world=np.array([ign_col * spacing, ign_row * spacing]),
        arrival_times=t_norm,
        arrival_times_hours=t_hours_inf,
        obs_pixels=obs_pixels,
        obs_arrival_times=obs_times,
        elev_raster=elev,
        slope_raster=slope,
        aspect_raster=aspect,
        canopy_raster=canopy,
        fuel_code_raster=fuel_codes,
        weather_vec=np.array([20.0, 0.4, 0.5, 0.866]),
        pixel_spacing_m=spacing,
        origin_xy=np.zeros(2),
        burned_mask=burned_mask,
    )


def _make_and_train_synthetic_metric(scenario, n_epochs=5):
    """Quickly train a flat-grid metric on the synthetic scenario.

    Returns a bound CovariateConditionedRanders ready for geometric analysis.
    """
    import optax

    from ham.training.losses import ArrivalTimeLoss

    manifold = EuclideanSpace(2)
    key = jax.random.PRNGKey(0)
    # Keep metric UNBOUND during training — bind_scene is called inside the loss
    # so that the gradient tree stays consistent (no int32 rasters in grad tree).
    metric = CovariateConditionedRanders(
        manifold, key, hidden_dim=64, fuel_emb_dim=4, use_wind=True
    )

    solver = AVBDSolver(step_size=0.05, iterations=50, energy_tol=1e-6, parallel=True)
    loss_fn = ArrivalTimeLoss(solver=solver, solver_steps=15)

    H, W = scenario.elev_raster.shape
    source_world = jnp.array(
        [
            scenario.ignition_pixel[1] * scenario.pixel_spacing_m,
            scenario.ignition_pixel[0] * scenario.pixel_spacing_m,
        ]
    )
    x_obs_world = jnp.asarray(
        np.stack(
            [
                scenario.obs_pixels[:, 1] * scenario.pixel_spacing_m,
                scenario.obs_pixels[:, 0] * scenario.pixel_spacing_m,
            ],
            axis=-1,
        )
    )
    t_obs = jnp.asarray(scenario.obs_arrival_times)

    # Pre-cache scene arrays as JAX arrays (used inside loss closure)
    j_elev = jnp.asarray(scenario.elev_raster)
    j_slope = jnp.asarray(scenario.slope_raster)
    j_aspect = jnp.asarray(scenario.aspect_raster)
    j_canopy = jnp.asarray(scenario.canopy_raster)
    j_fuel = jnp.asarray(scenario.fuel_code_raster)
    j_wx = jnp.asarray(scenario.weather_vec)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(metric, eqx.is_inexact_array))

    @eqx.filter_jit
    def compute_grads(m, bx, bt):
        def _loss(mm):
            bound = mm.bind_scene(
                j_elev,
                j_slope,
                j_aspect,
                j_canopy,
                j_fuel,
                j_wx,
                scenario.pixel_spacing_m,
                jnp.zeros(2),
            )
            bound = bound.precompute_metric_field()  # required after CNN refactor
            return loss_fn(bound, source_world, bx, bt)

        return eqx.filter_value_and_grad(_loss)(m)

    for epoch in range(n_epochs):
        key_e = jax.random.PRNGKey(epoch)
        idx = jax.random.choice(key_e, x_obs_world.shape[0], (32,), replace=False)
        loss_val, grads = compute_grads(metric, x_obs_world[idx], t_obs[idx])
        updates, opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_inexact_array),
            opt_state,
            eqx.filter(metric, eqx.is_inexact_array),
        )
        metric = eqx.apply_updates(metric, updates)
        print(f"  Epoch {epoch + 1}/{n_epochs}: loss={float(loss_val):.4f}")

    # Bind once after training and precompute the field for geometric analysis
    bound_metric = metric.bind_scene(
        elev=jnp.asarray(scenario.elev_raster),
        slope=jnp.asarray(scenario.slope_raster),
        aspect=jnp.asarray(scenario.aspect_raster),
        canopy=jnp.asarray(scenario.canopy_raster),
        fuel_codes=jnp.asarray(scenario.fuel_code_raster),
        weather_vec=jnp.asarray(scenario.weather_vec),
        pixel_spacing_m=scenario.pixel_spacing_m,
        origin_xy=jnp.zeros(2),
    ).precompute_metric_field()  # metric_fn requires this to be set

    return bound_metric, solver, source_world, (H, W)


# ---------------------------------------------------------------------------
# Main analysis runner
# ---------------------------------------------------------------------------


def run_geometric_analysis(
    metric,
    solver,
    source_world,
    scenario,
    domain_shape,
    output_dir,
    suffix="",
    n_directions=64,
    n_steps=50,
    compute_curvature=True,
):
    """Run all three geometric analyses for a given scenario + trained metric.

    Args:
        metric:          Bound FinslerMetric (flat 2D).
        solver:          AVBDSolver.
        source_world:    (2,) ignition in world coords.
        scenario:        WildfireScenario.
        domain_shape:    (H, W).
        output_dir:      Output directory.
        suffix:          Figure filename suffix.
        n_directions:    Number of geodesic fan rays.
        n_steps:         AVBD discretization steps per path.
        compute_curvature: If True, compute curvature field (slow for large grids).
    """
    H, W = domain_shape
    domain_size = max(H, W) * scenario.pixel_spacing_m

    print(f"\n  [Eikonal2.1] Shooting geodesic fan ({n_directions} directions)...")
    t0 = time.time()
    paths, arcs = shoot_geodesic_fan(
        metric,
        solver,
        source_world,
        domain_size,
        n_directions=n_directions,
        n_steps=n_steps,
    )
    print(
        f"  Done in {time.time() - t0:.1f}s. Arc length range: "
        f"[{arcs.min():.3f}, {arcs.max():.3f}]"
    )
    # Convert world coords to pixel coords for display
    paths_px = (
        paths / scenario.pixel_spacing_m
    )  # (D, T+1, 2) — (x,y) / spacing = (col, row)
    source_px = source_world / scenario.pixel_spacing_m
    plot_fire_corridors(
        paths_px,
        arcs,
        scenario.arrival_times,
        domain_shape,
        source_px,
        output_dir,
        suffix=suffix,
    )

    print("\n  [Eikonal2.2] Computing Jacobi divergence...")
    div = compute_jacobi_divergence(paths_px)
    plot_jacobi_divergence(div, output_dir, suffix=suffix)
    peak_t = np.argmax(div) / len(div)
    print(
        f"  Peak spread at normalized t={peak_t:.2f} — "
        + (
            "DIVERGING (open terrain)"
            if peak_t > 0.5
            else "CHANNELLED (terrain focusing)"
        )
    )

    if compute_curvature:
        print("\n  [Eikonal2.3] Sampling flag curvature...")
        # Sparse 8x8 grid for speed
        n_curv = 8
        xs = np.linspace(source_px[0] - 0.35 * W, source_px[0] + 0.35 * W, n_curv)
        ys = np.linspace(source_px[1] - 0.35 * H, source_px[1] + 0.35 * H, n_curv)
        gx, gy = np.meshgrid(xs, ys)
        # Convert pixel coords back to world coords for metric_fn
        grid_world = np.stack(
            [
                gx.ravel() * scenario.pixel_spacing_m,
                gy.ravel() * scenario.pixel_spacing_m,
            ],
            axis=-1,
        )
        t0 = time.time()
        kappa = compute_curvature_field(metric, grid_world)
        print(
            f"  Done in {time.time() - t0:.1f}s. kappa range: "
            f"[{np.nanmin(kappa):.3f}, {np.nanmax(kappa):.3f}]"
        )
        grid_px = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        plot_curvature_field(kappa, grid_px, domain_shape, output_dir, suffix=suffix)
    else:
        print("  [Eikonal2.3] Curvature skipped (--no_curvature flag).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Phase W3: Geometric analysis of trained wildfire metrics"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run on synthetic 20x20 scenario (no dataset)",
    )
    parser.add_argument(
        "--output_dir",
        default="results/phaseEikonal2",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--n_directions",
        type=int,
        default=32,
        help="Number of geodesic fan directions (default 32)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=5,
        help="Training epochs for synthetic mode (default 5)",
    )
    parser.add_argument(
        "--no_curvature", action="store_true", help="Skip curvature computation (slow)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu", "tpu"],
        help="JAX device to use (default: cpu).",
    )
    args = parser.parse_args()

    from ham.utils import configure_device

    configure_device(args.device)

    fig_dir = os.path.join(args.output_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 65)
    print("Phase Eikonal 2: Wildfire Geometric Analysis")
    print("=" * 65)

    if args.synthetic:
        print("\n[Synthetic mode] Building 20×20 test scenario...")
        scenario = _make_synthetic_scenario()

        print(f"\nTraining flat-grid metric ({args.n_epochs} epochs)...")
        metric, solver, source_world, domain_shape = _make_and_train_synthetic_metric(
            scenario, n_epochs=args.n_epochs
        )

        run_geometric_analysis(
            metric,
            solver,
            source_world,
            scenario,
            domain_shape,
            output_dir=fig_dir,
            suffix="_synthetic",
            n_directions=args.n_directions,
            n_steps=15,
            compute_curvature=not args.no_curvature,
        )

        print(f"\nAll figures saved to: {os.path.abspath(fig_dir)}")
    else:
        print(
            "\nFor real-data analysis, provide a trained metric checkpoint "
            "and implement the load_checkpoint() call.\n"
            "For now, use --synthetic to verify the analysis pipeline."
        )
        return

    print("\n" + "=" * 65)
    print("Phase Eikonal 2 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
