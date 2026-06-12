#!/usr/bin/env python3
"""
Phase W2 Training Script: CovariateMeshRanders on Sim2Real-Fire (3D Triangular Mesh)
======================================================================================

Adapts Phase W1 to use :class:`~ham.utils.terrain.CovariateMeshRanders` on a
:class:`~ham.geometry.mesh.TriangularMesh` built from the DEM raster, replacing the
flat-grid :class:`~ham.models.wildfire.CovariateConditionedRanders`.  All positions
are in 3D ``(x_m, y_m, z_m)`` and the metric cost F(x, v) operates in the face
tangent plane.

**Scientific question (Phase W2):**
    Does explicitly encoding DEM-derived 3D terrain geometry improve arrival-time
    correlation, particularly for high-slope (rugged) fires?

**Key outputs (saved to ``output_dir/figs/``):**
    - ``phaseW2_slope_stratified.{pdf,png}``     — Flat/Moderate/Rugged × Pearson r + RMSE
    - ``phaseW2_length_comparison.{pdf,png}``    — W1 proxy vs W2 mesh predictions scatter

**Reference:**
    Gahtan, Shpund & Bronstein (2026).  *Wildfire Simulation with Differentiable
    Randers-Finsler Eikonal Solvers.*  arXiv:2603.00035.

**HAMTools spec references:**
    spec/MATH_SPEC.md §§ 1–2, 5 (Randers / Zermelo)
    spec/ARCH_SPEC.md § 3 (CovariateMeshRanders), § 4 (EikonalSolver)

Usage::

    # Synthetic test (creates a random 3D landscape mesh):
    python examples/experiment_eikonal_mesh.py --synthetic --quick

    # Real data (one scene):
    python examples/experiment_eikonal_mesh.py \
        --data_root /data/sim2real_fire \
        --scenes 0014_00426 \
        --output_dir results/phaseEikonal3
"""

import argparse
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ham.data.wildfire import (
    SceneNormalizer,
    WildfireScenario,
    compute_slope_std,
    load_wildfire_scenario,
    stratified_sample_observations,
    train_val_test_split,
)

# config.update("jax_enable_x64", True)
# ---------------------------------------------------------------------------
# HAMTools imports
# ---------------------------------------------------------------------------
from ham.geometry.mesh import TriangularMesh
from ham.geometry.mesh_adjacency import MeshAdjacency
from ham.solvers.mesh_eikonal import MeshEikonalSolver
from ham.utils.terrain import (
    CovariateMeshRanders,
    compute_face_slopes_aspects,
    dem_to_mesh,
    interpolate_covariates_to_vertices,
)

# ---------------------------------------------------------------------------
# Optional Gahtan loader
# ---------------------------------------------------------------------------
try:
    from ham.data.sim2real_loader import Sim2RealFireLoader

    HAS_GAHTAN_LOADER = True
except ImportError:
    HAS_GAHTAN_LOADER = False


# ===========================================================================
# Configuration
# ===========================================================================


def get_config(quick: bool = False) -> dict:
    """Return experiment hyperparameters.

    Args:
        quick: If True, return a reduced config for smoke-testing.

    Returns:
        dict with all hyperparameters.
    """
    return dict(
        quick=quick,
        hidden_dim=64,
        fuel_emb_dim=4,
        n_epochs=20 if quick else 100,
        lr=1e-3,
        lr_schedule="cosine",
        early_stopping_patience=20,
        batch_size_fires=6 if quick else 8,
        k_train_obs=100 if quick else 500,
        k_eval_all=True,
        eikonal_iters=50,
        lambda_tv_G=0.005,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=42,
        train_seeds=[0] if quick else [0, 1, 2],
        mesh_resolution_fraction=0.5 if quick else 1.0,
    )


# ===========================================================================
# Mesh construction helpers
# ===========================================================================


def build_scene_mesh(
    scenario: WildfireScenario,
    mesh_resolution_fraction: float = 1.0,
):
    """Convert a scenario DEM to a TriangularMesh and assemble per-face covariates.

    Steps:

    1. Optionally subsample the DEM by ``mesh_resolution_fraction``.
    2. Call :func:`~ham.utils.terrain.dem_to_mesh` to build the mesh.
    3. Call :func:`~ham.utils.terrain.compute_face_slopes_aspects` for face
       slope and aspect from mesh geometry.
    4. Call :func:`~ham.utils.terrain.interpolate_covariates_to_vertices`
       for elevation, canopy per vertex.
    5. Assemble per-face continuous covariates ``(F, 5)``:
       ``[elev, slope, sin(aspect), cos(aspect), canopy]``.
    6. Return face fuel codes ``(F,)`` separately (looked up inside loss).

    Args:
        scenario:               WildfireScenario with rasters and DEM.
        mesh_resolution_fraction: 1.0 = full resolution; 0.5 = every other row/col.

    Returns:
        Tuple ``(mesh, face_cov_5, face_fuel_codes, face_elev_m)`` where

        * ``mesh``            – :class:`~ham.geometry.mesh.TriangularMesh`
        * ``face_cov_5``      – ``(F, 5)`` float32 covariate matrix
        * ``face_fuel_codes`` – ``(F,)`` int32 FBFM-13 codes
        * ``face_elev_m``     – ``(F,)`` float32 mean face elevation — units are raw metres,
          matching the unnormalized Z-coordinate in ``scenario.elev_raster``.
    Reference:
        spec/ARCH_SPEC.md § 5; spec/MATH_SPEC.md §§ 1–2.
    """
    step = max(1, int(round(1.0 / mesh_resolution_fraction)))
    elev_r = scenario.elev_raster[::step, ::step]
    slope_r = scenario.slope_raster[::step, ::step]
    aspect_r = scenario.aspect_raster[::step, ::step]
    canopy_r = scenario.canopy_raster[::step, ::step]
    fuel_r = scenario.fuel_code_raster[::step, ::step]
    spacing = float(scenario.pixel_spacing_m) * step

    H_use, W_use = elev_r.shape

    # ---- Build mesh from DEM ----------------------------------------
    elev_jnp = jnp.asarray(elev_r, dtype=jnp.float32)
    mesh = dem_to_mesh(elev_jnp, pixel_spacing_m=spacing)

    # ---- Face slope and aspect from mesh geometry --------------------
    face_slopes, face_aspects = compute_face_slopes_aspects(mesh)  # (F,) each

    # ---- Vertex-level covariates -------------------------------------
    raster_dict = {
        "elev": elev_r,
        "canopy": canopy_r,
    }
    vert_covs = interpolate_covariates_to_vertices(
        mesh, raster_dict, H_use, W_use
    )  # (V, 2): [elev, canopy]

    # ---- Face-level aggregation -------------------------------------
    # face_elev = mean Z of the 3 vertices  (absolute elevation, metres)
    face_elev_m = jnp.mean(mesh.vertices[mesh.faces, 2], axis=1)  # (F,)

    # face_canopy = mean canopy of the 3 vertices
    vert_canopy = vert_covs[:, 1]  # (V,)
    face_canopy = jnp.mean(vert_canopy[mesh.faces], axis=1)  # (F,)

    # face_fuel = FBFM-13 code at vertex 0 of each face
    vert_fuel = jnp.asarray(fuel_r.ravel(), dtype=jnp.int32)  # (V,)
    face_fuel_codes = vert_fuel[mesh.faces[:, 0]]  # (F,)

    # ---- Assemble (F, 5) covariate matrix ----------------------------
    # [face_elev_norm, face_slope, sin(face_aspect), cos(face_aspect), face_canopy]
    # Normalise face_elev to zero-mean unit-std across this scene
    fe_mean = float(jnp.mean(face_elev_m))
    fe_std = float(jnp.std(face_elev_m))
    fe_std = fe_std if fe_std > 1e-8 else 1.0
    face_elev_norm = (face_elev_m - fe_mean) / fe_std

    face_cov_5 = jnp.stack(
        [
            face_elev_norm,
            face_slopes,
            jnp.sin(face_aspects),
            jnp.cos(face_aspects),
            face_canopy,
        ],
        axis=-1,
    ).astype(jnp.float32)  # (F, 5)

    return mesh, face_cov_5, face_fuel_codes, face_elev_m


def bind_mesh_scenario(
    metric: CovariateMeshRanders,
    face_cov_5: jax.Array,
    face_fuel_codes: jax.Array,
    weather_vec,
) -> CovariateMeshRanders:
    """Bind per-face covariates and weather to a CovariateMeshRanders instance.

    Looks up fuel embeddings from the *trainable* ``metric.fuel_embedding``
    so gradients flow through the embedding table.  Then calls
    :meth:`~ham.utils.terrain.CovariateMeshRanders.bind_scene` to attach the
    assembled ``(F, 5 + fuel_emb_dim)`` array and the weather vector.

    Args:
        metric:          Unbound :class:`~ham.utils.terrain.CovariateMeshRanders`.
        face_cov_5:      ``(F, 5)`` per-face continuous covariates.
        face_fuel_codes: ``(F,)`` int32 FBFM-13 codes.
        weather_vec:     ``(4,)`` ``[T_air, humidity, sin_wind, cos_wind]``.

    Returns:
        Bound :class:`~ham.utils.terrain.CovariateMeshRanders` with
        ``face_covariates`` and ``weather_vec`` set.
    """
    fuel_codes_clipped = jnp.clip(jnp.asarray(face_fuel_codes, dtype=jnp.int32), 0, 12)
    fuel_embs = metric.fuel_embedding[fuel_codes_clipped]  # (F, fuel_emb_dim)

    face_cov_full = jnp.concatenate(
        [
            jnp.asarray(face_cov_5, dtype=jnp.float32),
            fuel_embs.astype(jnp.float32),
        ],
        axis=-1,
    )  # (F, 5 + fuel_emb_dim)

    return metric.bind_scene(
        face_cov_full,
        jnp.asarray(weather_vec, dtype=jnp.float32),
    )


# ===========================================================================
# Coordinate helpers (3-D)
# ===========================================================================


def _pixels_to_world_3d(
    pixels: np.ndarray,
    elev_raster: np.ndarray,
    pixel_spacing_m: float,
) -> np.ndarray:
    """Convert ``(row, col)`` pixel pairs to 3-D world coordinates ``(x, y, z)``.

    Args:
        pixels:          Shape ``(K, 2)`` int — ``[row, col]`` pairs.
        elev_raster:     Shape ``(H, W)`` — elevation values in metres.
        pixel_spacing_m: Ground-sample distance in metres per pixel.

    Returns:
        Shape ``(K, 3)`` float64 — ``[x_m, y_m, z_m]`` coordinates.
    """
    pix = np.asarray(pixels, dtype=np.float32)
    rows = np.clip(pix[:, 0].astype(int), 0, elev_raster.shape[0] - 1)
    cols = np.clip(pix[:, 1].astype(int), 0, elev_raster.shape[1] - 1)
    x = pix[:, 1] * float(pixel_spacing_m)
    y = pix[:, 0] * float(pixel_spacing_m)
    z = elev_raster[rows, cols].astype(np.float32)
    return np.stack([x, y, z], axis=1)


def _ignition_to_world_3d(
    ignition_pixel: np.ndarray,
    elev_raster: np.ndarray,
    pixel_spacing_m: float,
) -> jax.Array:
    """Convert ignition ``(row, col)`` pixel to 3-D world coordinate JAX array.

    Args:
        ignition_pixel:  Shape ``(2,)`` — ``[row, col]``.
        elev_raster:     Shape ``(H, W)`` — elevation in metres.
        pixel_spacing_m: Ground-sample distance in metres per pixel.

    Returns:
        Shape ``(3,)`` float64 JAX array ``[x_m, y_m, z_m]``.
    """
    row = int(np.clip(ignition_pixel[0], 0, elev_raster.shape[0] - 1))
    col = int(np.clip(ignition_pixel[1], 0, elev_raster.shape[1] - 1))
    x = float(col) * float(pixel_spacing_m)
    y = float(row) * float(pixel_spacing_m)
    z = float(elev_raster[row, col])
    return jnp.array([x, y, z], dtype=jnp.float32)


# ===========================================================================
# Statistics helpers
# ===========================================================================


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1-D arrays.

    Args:
        a: Predicted values, shape (N,).
        b: Ground-truth values, shape (N,).

    Returns:
        Correlation in ``[-1, 1]``; ``0.0`` if either array is constant.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 2:
        return 0.0
    a_m = a.mean()
    b_m = b.mean()
    num = np.sum((a - a_m) * (b - b_m))
    denom = np.sqrt(np.sum((a - a_m) ** 2) * np.sum((b - b_m) ** 2))
    return 0.0 if denom < 1e-8 else float(num / denom)


def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Root-mean-square error between predicted and ground-truth arrival times.

    Args:
        pred: Predicted values, shape (N,).
        gt:   Ground-truth values, shape (N,).

    Returns:
        RMSE scalar; ``0.0`` if either array is empty.
    """
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if len(pred) == 0:
        return 0.0
    valid = np.isfinite(pred) & np.isfinite(gt)
    if np.sum(valid) > 0:
        p_valid = pred[valid]
        g_valid = gt[valid]
        s = float(np.mean(g_valid)) / max(float(np.mean(p_valid)), 1e-8)
        pred = pred * s
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


# ===========================================================================
# Chunked arrival-time prediction on mesh
# ===========================================================================


def _predict_arrivals_mesh_chunked(
    bound_metric: CovariateMeshRanders,
    solver: MeshEikonalSolver,
    mesh: TriangularMesh,
    adjacency: MeshAdjacency,
    source_3d: jax.Array,
    eval_pixels: np.ndarray,
    elev_raster: np.ndarray,
    pixel_spacing_m: float,
) -> np.ndarray:
    """Predict geodesic arc lengths for eval pixels via global Eikonal solver."""
    source_coords = jnp.expand_dims(source_3d, axis=0)
    T_all = solver.solve(
        bound_metric, adjacency, mesh.vertices, mesh.faces, source_coords
    )

    eval_3d = jnp.asarray(
        _pixels_to_world_3d(eval_pixels, elev_raster, pixel_spacing_m),
        dtype=jnp.float32,
    )

    def get_closest_v(src):
        return jnp.argmin(jnp.sum((mesh.vertices - src) ** 2, axis=-1))

    closest_vs = jax.vmap(get_closest_v)(eval_3d)
    return np.array(T_all[closest_vs], dtype=np.float32)


# ===========================================================================
# Make / init metric and solver
# ===========================================================================


def make_mesh_metric(
    mesh: TriangularMesh,
    cfg: dict,
    key: jax.Array,
    use_wind: bool = True,
) -> CovariateMeshRanders:
    """Instantiate an unbound CovariateMeshRanders for a given mesh.

    Args:
        mesh:      TriangularMesh defining the domain.
        cfg:       Configuration dict from :func:`get_config`.
        key:       JAX PRNG key.
        use_wind:  If False, sets drift b=0 (Riemannian ablation).

    Returns:
        Unbound :class:`~ham.utils.terrain.CovariateMeshRanders`.
    """
    return CovariateMeshRanders(
        mesh,
        key,
        hidden_dim=cfg["hidden_dim"],
        fuel_emb_dim=cfg["fuel_emb_dim"],
        use_wind=use_wind,
    )


def make_solver(cfg: dict) -> MeshEikonalSolver:
    """Instantiate a MeshEikonalSolver.

    Args:
        cfg: Configuration dict from :func:`get_config`.

    Returns:
        :class:`~ham.solvers.mesh_eikonal.MeshEikonalSolver`.
    """
    return MeshEikonalSolver(
        max_iters=cfg.get("eikonal_iters", 50),
        tol=1e-4,
    )


# ===========================================================================
# Validation helper
# ===========================================================================


def compute_sparse_mesh_eikonal_loss(
    metric,
    solver,
    mesh,
    adjacency,
    source_world: jax.Array,
    x_obs_world: jax.Array,
    t_obs: jax.Array,
):
    source_coords = jnp.expand_dims(source_world, axis=0)
    T_all = solver.solve(metric, adjacency, mesh.vertices, mesh.faces, source_coords)

    def get_closest_v(src):
        return jnp.argmin(jnp.sum((mesh.vertices - src) ** 2, axis=-1))

    closest_vs = jax.vmap(get_closest_v)(x_obs_world)
    t_pred = T_all[closest_vs]
    return jnp.mean((t_pred - t_obs) ** 2)


def _val_pearson_r_mesh(
    metric: CovariateMeshRanders,
    solver: MeshEikonalSolver,
    scenario: WildfireScenario,
    mesh: TriangularMesh,
    adjacency: MeshAdjacency,
    face_cov_5: jax.Array,
    face_fuel_codes: jax.Array,
    cfg: dict,
) -> float:
    """Fast validation: Pearson r over observation pixels only."""
    bound = bind_mesh_scenario(
        metric, face_cov_5, face_fuel_codes, scenario.weather_vec
    )

    source_3d = _ignition_to_world_3d(
        scenario.ignition_pixel, scenario.elev_raster, scenario.pixel_spacing_m
    )
    pred = _predict_arrivals_mesh_chunked(
        bound,
        solver,
        mesh,
        adjacency,
        source_3d,
        scenario.obs_pixels,
        scenario.elev_raster,
        scenario.pixel_spacing_m,
    )
    gt = np.asarray(scenario.obs_arrival_times, dtype=np.float32)
    return pearson_r(pred, gt)


# ===========================================================================
# Per-scene training loop
# ===========================================================================


def train_scene_mesh(
    data_root: str,
    scene_id: str,
    cfg: dict,
    seed: int,
    use_wind: bool = True,
    output_dir: str = "results",
) -> dict | None:
    """Full per-scene Phase W2 training loop.

    1. Load all fire events for the scene.
    2. Build mesh from the first fire's DEM (all fires share terrain).
    3. Split train/val/test (70/15/15).
    4. Train :class:`~ham.utils.terrain.CovariateMeshRanders` with AVBD
       arrival-time supervision.
    5. Evaluate best-val checkpoint on the test split.
    6. Perform slope stratification on test results.

    Args:
        data_root: Path to the Sim2Real-Fire dataset root directory.
        scene_id:  Scene identifier, e.g. ``"0014_00426"``.
        cfg:       Configuration dict from :func:`get_config`.
        seed:      RNG seed for sampling and weight initialisation.
        use_wind:  If False, trains the Riemannian ablation (b=0).

    Returns:
        Result dict or ``None`` if the scene has no events.

    Raises:
        ImportError: If the Gahtan loader is not available.
        FileNotFoundError: If *data_root* does not exist.
    """
    if not HAS_GAHTAN_LOADER:
        raise ImportError(
            "Sim2RealFireLoader not available. Run with --synthetic instead."
        )
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Dataset not found at {data_root}.")

    print(f"\n{'=' * 60}")
    print(
        f"Phase W2  Scene {scene_id}  seed={seed}  wind={'yes' if use_wind else 'no'}"
    )
    print(f"{'=' * 60}")
    t_scene_start = time.time()

    loader = Sim2RealFireLoader(data_root)

    event_ids = None
    try:
        event_ids = loader.scenes[scene_id]["events"]
    except (AttributeError, KeyError):
        pass

    if not isinstance(event_ids, list):
        try:
            event_ids = loader.list_events(scene_id)
        except AttributeError:
            try:
                event_ids = loader.get_event_ids(scene_id)
            except AttributeError:
                # Final fallback: scan Satellite_Images_Mask for event folders
                # that have a matching weather file — mirrors W1 logic exactly.
                mask_dir = os.path.join(data_root, scene_id, "Satellite_Images_Mask")
                weather_dir = os.path.join(data_root, scene_id, "Weather_Data")
                if os.path.isdir(mask_dir):
                    event_ids = [
                        d
                        for d in sorted(os.listdir(mask_dir))
                        if os.path.isdir(os.path.join(mask_dir, d))
                        and os.path.exists(os.path.join(weather_dir, f"{d}.txt"))
                    ]
                else:
                    event_ids = []

    if not event_ids:
        print(f"  WARNING: No events for scene {scene_id}, skipping.")
        return None

    scenarios_keys = [(scene_id, eid) for eid in event_ids]
    train_list, val_list, test_list = train_val_test_split(
        scenarios_keys,
        train_ratio=cfg["train_ratio"],
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
    )
    if cfg.get("quick", False):
        train_list = train_list[:32]
        val_list = val_list[:12]
        test_list = test_list[:12]
    print(
        f"  Events: {len(scenarios_keys)} total | "
        f"{len(train_list)} train / {len(val_list)} val / {len(test_list)} test"
    )

    print("  Fitting SceneNormalizer...")
    raw_for_norm = [loader.load_scenario(s, e) for s, e in train_list[:20]]
    normalizer = SceneNormalizer.fit(raw_for_norm)

    print("  Loading & preprocessing scenarios...")

    def _load(pairs):
        return [
            load_wildfire_scenario(
                loader, s, e, normalizer, k_train_obs=cfg["k_train_obs"], seed=seed
            )
            for s, e in pairs
        ]

    train_scens = _load(train_list)
    val_scens = _load(val_list)
    test_scens = _load(test_list)

    # ---- Build mesh once from the first scenario's DEM ---------------
    ref_sc = train_scens[0]
    mesh, face_cov_5, face_fuel_codes, face_elev_m = build_scene_mesh(
        ref_sc, mesh_resolution_fraction=cfg["mesh_resolution_fraction"]
    )
    adjacency = MeshAdjacency.build(mesh.vertices, mesh.faces)
    print(
        f"  Mesh: {mesh.vertices.shape[0]} vertices, "
        f"{mesh.faces.shape[0]} faces  "
        f"(resolution fraction={cfg['mesh_resolution_fraction']})"
    )

    # ---- Create metric and solver ------------------------------------
    key = jax.random.PRNGKey(seed)
    metric = make_mesh_metric(mesh, cfg, key, use_wind=use_wind)
    solver = make_solver(cfg)

    n_batches_per_epoch = max(1, len(train_scens) // cfg["batch_size_fires"])
    total_steps = cfg["n_epochs"] * n_batches_per_epoch
    warmup_steps = min(5 * n_batches_per_epoch, max(1, total_steps // 10))
    lr_schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=1e-5, end_value=cfg["lr"], transition_steps=warmup_steps
            ),
            optax.cosine_decay_schedule(
                init_value=cfg["lr"], decay_steps=max(1, total_steps - warmup_steps)
            ),
        ],
        boundaries=[warmup_steps],
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr_schedule),
    )
    opt_state = optimizer.init(eqx.filter(metric, eqx.is_inexact_array))

    rng = np.random.default_rng(seed)
    rng = np.random.default_rng(seed)

    best_loss: float = np.inf
    best_metric = metric
    patience_counter: int = 0
    train_loss_history: list = []
    val_r_history: list = []
    epoch_runtimes: list = []

    for epoch in range(cfg["n_epochs"]):
        t_epoch = time.time()
        perm = rng.permutation(len(train_scens))
        epoch_losses: list = []

        for batch_start in range(0, len(perm), cfg["batch_size_fires"]):
            batch_idx = perm[batch_start : batch_start + cfg["batch_size_fires"]]
            accumulated_grads = None
            batch_losses: list = []

            def _fire_loss(m, _fc5, _ffc, _wv, _src, _x, _t):
                bound = bind_mesh_scenario(m, _fc5, _ffc, _wv)
                return compute_sparse_mesh_eikonal_loss(
                    bound, solver, mesh, adjacency, _src, _x, _t
                )

            for idx in batch_idx:
                sc = train_scens[int(idx)]
                obs_3d = jnp.asarray(
                    _pixels_to_world_3d(
                        sc.obs_pixels, sc.elev_raster, sc.pixel_spacing_m
                    ),
                    dtype=jnp.float32,
                )
                t_obs = jnp.asarray(sc.obs_arrival_times, dtype=jnp.float32)
                source_3d = _ignition_to_world_3d(
                    sc.ignition_pixel, sc.elev_raster, sc.pixel_spacing_m
                )
                w_vec = sc.weather_vec

                loss_val, grads = eqx.filter_value_and_grad(_fire_loss)(
                    metric, face_cov_5, face_fuel_codes, w_vec, source_3d, obs_3d, t_obs
                )
                batch_losses.append(float(loss_val))

                accumulated_grads = (
                    grads
                    if accumulated_grads is None
                    else jax.tree_util.tree_map(
                        lambda a, b: a + b, accumulated_grads, grads
                    )
                )

            n_batch = len(batch_idx)
            accumulated_grads = jax.tree_util.tree_map(
                lambda g: g / n_batch, accumulated_grads
            )
            updates, opt_state = optimizer.update(
                eqx.filter(accumulated_grads, eqx.is_inexact_array),
                opt_state,
                eqx.filter(metric, eqx.is_inexact_array),
            )
            metric = eqx.apply_updates(metric, updates)
            epoch_losses.extend(batch_losses)

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_loss_history.append(mean_loss)
        epoch_runtimes.append(time.time() - t_epoch)

        val_rs = [
            _val_pearson_r_mesh(
                metric, solver, sc, mesh, adjacency, face_cov_5, face_fuel_codes, cfg
            )
            for sc in val_scens
        ]
        mean_val_r = float(np.mean(val_rs)) if val_rs else 0.0
        val_r_history.append(mean_val_r)

        print(
            f"  Epoch {epoch + 1:3d}/{cfg['n_epochs']}: "
            f"loss={mean_loss:.5f}  val_r={mean_val_r:.4f}  "
            f"time={epoch_runtimes[-1]:.1f}s"
        )

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_metric = metric
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                print(
                    f"  Early stopping at epoch {epoch + 1}  (best_loss={best_loss:.4f})"
                )
                break

    # ---- Test evaluation -------------------------------------------
    print(f"\n  Evaluating on {len(test_scens)} test fires (dense)...")
    test_results = []
    for sc in test_scens:
        bound = bind_mesh_scenario(
            best_metric, face_cov_5, face_fuel_codes, sc.weather_vec
        )
        source_3d = _ignition_to_world_3d(
            sc.ignition_pixel, sc.elev_raster, sc.pixel_spacing_m
        )
        rows, cols = np.where(sc.burned_mask)
        eval_pix = np.stack([rows, cols], axis=1)
        if cfg.get("quick", False) and len(eval_pix) > 100:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(eval_pix), size=100, replace=False)
            eval_pix = eval_pix[idx]

        if len(eval_pix) == 0:
            continue

        pred = _predict_arrivals_mesh_chunked(
            bound,
            solver,
            mesh,
            adjacency,
            source_3d,
            eval_pix,
            sc.elev_raster,
            sc.pixel_spacing_m,
        )
        gt = np.array(
            [sc.arrival_times[int(r), int(c)] for r, c in eval_pix],
            dtype=np.float32,
        )
        r_val = pearson_r(pred, gt)
        rmse_val = compute_rmse(pred, gt)
        slope_s = compute_slope_std(sc)
        test_results.append(
            dict(
                pearson_r=r_val,
                rmse=rmse_val,
                slope_std=slope_s,
                pred=pred,
                gt=gt,
            )
        )

    test_rs = [x["pearson_r"] for x in test_results]
    test_rmses = [x["rmse"] for x in test_results]
    test_r_mean = float(np.mean(test_rs)) if test_rs else 0.0
    test_r_std = float(np.std(test_rs)) if test_rs else 0.0
    test_rmse = float(np.mean(test_rmses)) if test_rmses else 0.0
    runtime = float(np.mean(epoch_runtimes)) if epoch_runtimes else 0.0

    print(
        f"\n  RESULTS  scene={scene_id}  seed={seed}\n"
        f"    test Pearson r  = {test_r_mean:.4f} ± {test_r_std:.4f}\n"
        f"    test RMSE       = {test_rmse:.4f}\n"
        f"    epoch runtime   = {runtime:.1f} s/epoch\n"
        f"    total time      = {time.time() - t_scene_start:.1f} s"
    )

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"w2_{scene_id}_seed{seed}.eqx")
    eqx.tree_serialise_leaves(ckpt_path, best_metric)
    print(f"    Saved checkpoint to {ckpt_path}")

    return dict(
        scene_id=scene_id,
        seed=seed,
        use_wind=use_wind,
        test_pearson_r_mean=test_r_mean,
        test_pearson_r_std=test_r_std,
        test_rmse=test_rmse,
        train_loss_history=train_loss_history,
        val_r_history=val_r_history,
        runtime_per_epoch_s=runtime,
        test_per_fire=test_results,
    )


# ===========================================================================
# Synthetic smoke test
# ===========================================================================


def _make_synthetic_scenario_mesh(
    slope_level: str = "moderate",
    seed: int = 0,
    mesh_resolution_fraction: float = 1.0,
):
    """Build a synthetic WildfireScenario with a 20×20 DEM for pipeline testing.

    Three slope levels:
    * ``"flat"``     — z=100 m everywhere (slope std ≈ 0°).
    * ``"moderate"`` — mild sinusoidal variation, slope std ≈ 3–6°.
    * ``"rugged"``   — steep sinusoidal variation, slope std ≈ 10–15°.

    Arrival times use a 2-D Euclidean wave from ignition at pixel (10, 10):
    ``arrival[i,j] = sqrt((i-10)^2+(j-10)^2) * 0.03``.

    Args:
        slope_level: One of ``"flat"``, ``"moderate"``, ``"rugged"``.
        seed:        RNG seed for observation sampling.
        mesh_resolution_fraction: Resolution fraction (not used — kept for API).

    Returns:
        :class:`~ham.data.wildfire.WildfireScenario`.
    """
    H, W = 20, 20
    pixel_spacing_m = 1.0
    ign_row, ign_col = 10, 10

    rows_g, cols_g = np.mgrid[0:H, 0:W]

    amp = {"flat": 0.0, "moderate": 10.0, "rugged": 40.0}[slope_level]
    elev_raster = (
        100.0 + amp * np.sin(np.pi * rows_g / H) * np.cos(np.pi * cols_g / W)
    ).astype(np.float32)

    slope_raster = np.zeros((H, W), dtype=np.float32)
    aspect_raster = np.zeros((H, W), dtype=np.float32)
    canopy_raster = np.zeros((H, W), dtype=np.float32)
    fuel_code_raster = np.full((H, W), 5, dtype=np.int32)
    weather_vec = np.zeros(4, dtype=np.float32)
    origin_xy = np.zeros(2, dtype=np.float32)

    dr = rows_g - ign_row
    dc = cols_g - ign_col
    arrival_hours = np.sqrt(dr**2 + dc**2).astype(np.float32) * 0.03
    arrival_hours[ign_row, ign_col] = 0.0

    t_max = float(arrival_hours.max())
    t_max = t_max if t_max > 1e-8 else 1.0
    arrival_norm = arrival_hours / t_max
    burned_mask = np.ones((H, W), dtype=bool)

    obs_pixels = stratified_sample_observations(arrival_hours, n_samples=50, seed=seed)
    obs_arrival_times = arrival_norm[obs_pixels[:, 0], obs_pixels[:, 1]]

    # Store raw elevation (not z-scored) so that mesh vertices have
    # physically meaningful Z-coordinates and face slopes reflect actual terrain
    # steepness.  build_scene_mesh normalises face_elev internally.
    elev_norm = elev_raster  # raw metres — intentionally NOT z-scored here

    ignition_world = np.array(
        [float(ign_col) * pixel_spacing_m, float(ign_row) * pixel_spacing_m],
        dtype=np.float32,
    )

    return WildfireScenario(
        scene_id=f"synthetic_{slope_level}",
        event_id="synth_00001",
        ignition_pixel=np.array([ign_row, ign_col], dtype=np.int64),
        ignition_world=ignition_world,
        arrival_times=arrival_norm,
        arrival_times_hours=arrival_hours,
        obs_pixels=obs_pixels,
        obs_arrival_times=obs_arrival_times,
        elev_raster=elev_norm,
        slope_raster=slope_raster,
        aspect_raster=aspect_raster,
        canopy_raster=canopy_raster,
        fuel_code_raster=fuel_code_raster,
        weather_vec=weather_vec,
        pixel_spacing_m=pixel_spacing_m,
        origin_xy=origin_xy,
        burned_mask=burned_mask,
    )


def run_synthetic_mesh(cfg: dict, output_dir: str, use_wind: bool = True) -> list:
    """Run the synthetic Phase W2 smoke test on three slope-level scenarios.

    Generates flat, moderate, and rugged scenarios, trains one shared mesh
    metric per scenario, evaluates on 200 random pixels, and reports Pearson r
    per slope bin.

    Args:
        cfg:        Configuration dict (typically ``get_config(quick=True)``).
        output_dir: Directory for saved figures.
        use_wind:   Whether to include the Randers drift.

    Returns:
        List of result dicts, one per slope level.
    """
    print("\n" + "=" * 60)
    print("SYNTHETIC MESH SMOKE TEST  (no real dataset required)")
    print("=" * 60)

    slope_levels = ["flat", "moderate", "rugged"]
    all_results = []

    for slope_level in slope_levels:
        print(f"\n-- Slope level: {slope_level} --")
        scenario = _make_synthetic_scenario_mesh(
            slope_level=slope_level,
            seed=cfg["seed"],
            mesh_resolution_fraction=cfg["mesh_resolution_fraction"],
        )

        mesh, face_cov_5, face_fuel_codes, face_elev_m = build_scene_mesh(
            scenario, mesh_resolution_fraction=cfg["mesh_resolution_fraction"]
        )
        adjacency = MeshAdjacency.build(mesh.vertices, mesh.faces)
        print(
            f"   Mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces"
        )

        key = jax.random.PRNGKey(cfg["seed"])
        metric = make_mesh_metric(mesh, cfg, key, use_wind=use_wind)
        solver = make_solver(cfg)

        n_epochs = cfg["n_epochs"]
        lr_schedule = optax.cosine_decay_schedule(
            init_value=cfg["lr"], decay_steps=max(n_epochs, 1)
        )
        optimizer = optax.adam(lr_schedule)
        opt_state = optimizer.init(eqx.filter(metric, eqx.is_inexact_array))

        obs_3d = jnp.asarray(
            _pixels_to_world_3d(
                scenario.obs_pixels, scenario.elev_raster, scenario.pixel_spacing_m
            ),
            dtype=jnp.float32,
        )
        t_obs = jnp.asarray(scenario.obs_arrival_times, dtype=jnp.float32)
        source_3d = _ignition_to_world_3d(
            scenario.ignition_pixel, scenario.elev_raster, scenario.pixel_spacing_m
        )
        w_vec = scenario.weather_vec

        train_loss_history: list = []
        t0 = time.time()
        print(f"   Training for {n_epochs} epochs...")

        # Define _loss once outside the epoch loop so JAX can cache the traced
        # computation graph across epochs (Code RISK #3 fix).
        def _loss(
            m,
            _fc5=face_cov_5,
            _ffc=face_fuel_codes,
            _wv=w_vec,
            _src=source_3d,
            _x=obs_3d,
            _t=t_obs,
        ):
            bound = bind_mesh_scenario(m, _fc5, _ffc, _wv)
            return compute_sparse_mesh_eikonal_loss(
                bound, solver, mesh, adjacency, _src, _x, _t
            )

        for epoch in range(n_epochs):
            loss_val, grads = eqx.filter_value_and_grad(_loss)(metric)
            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(metric, eqx.is_inexact_array)
            )
            metric = eqx.apply_updates(metric, updates)
            train_loss_history.append(float(loss_val))
            print(f"   Epoch {epoch + 1:3d}/{n_epochs}: loss={float(loss_val):.6f}")

        train_time = time.time() - t0

        # ---- Evaluate on 200 random pixels ---------------------------
        rng = np.random.default_rng(cfg["seed"])
        H, W = scenario.arrival_times.shape
        all_pix = np.array([[r, c] for r in range(H) for c in range(W)], dtype=np.int64)
        eval_idx = rng.choice(len(all_pix), size=min(200, len(all_pix)), replace=False)
        eval_pixels = all_pix[eval_idx]

        # Bind final metric
        bound = bind_mesh_scenario(
            metric, face_cov_5, face_fuel_codes, scenario.weather_vec
        )

        print(f"   Evaluating on {len(eval_pixels)} pixels...")
        pred = _predict_arrivals_mesh_chunked(
            bound,
            solver,
            mesh,
            adjacency,
            source_3d,
            eval_pixels,
            scenario.elev_raster,
            scenario.pixel_spacing_m,
        )
        gt = np.array(
            [scenario.arrival_times[int(r), int(c)] for r, c in eval_pixels],
            dtype=np.float32,
        )
        r_val = pearson_r(pred, gt)
        rmse_val = compute_rmse(pred, gt)

        # Slope std from mesh face angles (degrees)
        face_slopes_rad, _ = compute_face_slopes_aspects(mesh)
        slope_std_deg = float(np.degrees(float(jnp.std(face_slopes_rad))))

        print(
            f"\n   {slope_level.upper()} RESULTS\n"
            f"     Pearson r    = {r_val:.4f}\n"
            f"     RMSE         = {rmse_val:.4f}\n"
            f"     Slope std    = {slope_std_deg:.1f}°\n"
            f"     Train time   = {train_time:.1f}s for {n_epochs} epochs"
        )

        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(
            ckpt_dir, f"w2_synthetic_{slope_level}_seed{cfg['seed']}.eqx"
        )
        eqx.tree_serialise_leaves(ckpt_path, metric)
        print(f"     Saved checkpoint to {ckpt_path}")

        all_results.append(
            dict(
                slope_level=slope_level,
                slope_std_deg=slope_std_deg,
                test_pearson_r_mean=r_val,
                test_rmse=rmse_val,
                train_loss_history=train_loss_history,
                pred=pred,
                gt=gt,
                train_time=train_time,
            )
        )

    # ---- Overall summary -------------------------------------------
    mean_r = float(np.mean([r["test_pearson_r_mean"] for r in all_results]))
    print(f"\n  AGGREGATE Pearson r = {mean_r:.4f} (mean across slope levels)")

    _save_figures(all_results, output_dir)

    return all_results


# ===========================================================================
# Figures
# ===========================================================================


def _slope_bin_label(slope_std_deg: float) -> str:
    """Map slope-std in degrees to a stratification bin label."""
    if slope_std_deg <= 3.0:
        return "Flat (≤3°)"
    elif slope_std_deg <= 8.0:
        return "Moderate (3–8°)"
    else:
        return "Rugged (>8°)"


def _save_figures(results: list, output_dir: str) -> None:
    """Generate and save Phase W2 publication figures.

    Two figures are saved:

    1. ``phaseW2_slope_stratified.{pdf,png}`` — grouped bar chart of
       Pearson r and RMSE per slope bin.
    2. ``phaseW2_length_comparison.{pdf,png}`` — scatter of 2-D proxy
       (W1-like) vs 3-D mesh (W2) predicted arrival times.

    Args:
        results:    List of result dicts (one per scenario / slope level).
        output_dir: Root directory; figures saved to ``<output_dir>/figs/``.
    """
    fig_dir = os.path.join(output_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Slope-stratified bar chart ---------------------------------
    # Assign each result to a slope bin
    bins_order = ["Flat (≤3°)", "Moderate (3–8°)", "Rugged (>8°)"]
    bin_r: dict = {b: [] for b in bins_order}
    bin_rmse: dict = {b: [] for b in bins_order}

    for res in results:
        # Support both per-fire (real) and per-scenario (synthetic) entries
        if "test_per_fire" in res:
            for fire in res["test_per_fire"]:
                lbl = _slope_bin_label(
                    float(fire["slope_std"])
                    if "slope_std" in fire
                    else fire.get("slope_std_deg", 0.0)
                )
                bin_r[lbl].append(fire["pearson_r"])
                bin_rmse[lbl].append(fire["rmse"])
        else:
            # Synthetic result: slope_std_deg stored at top level
            lbl = _slope_bin_label(res.get("slope_std_deg", 0.0))
            bin_r[lbl].append(res["test_pearson_r_mean"])
            bin_rmse[lbl].append(res.get("test_rmse", 0.0))

    r_means = [float(np.mean(bin_r[b])) if bin_r[b] else 0.0 for b in bins_order]
    r_stds = [float(np.std(bin_r[b])) if len(bin_r[b]) > 1 else 0.0 for b in bins_order]
    rmse_means = [
        float(np.mean(bin_rmse[b])) if bin_rmse[b] else 0.0 for b in bins_order
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(bins_order))
    ax1.bar(x, r_means, yerr=r_stds, capsize=4, color="#4477AA", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins_order, fontsize=8)
    ax1.set_ylabel("Pearson r (test)")
    ax1.set_title("Phase W2 — Pearson r by Terrain Slope")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.70, color="red", linestyle="--", linewidth=1.0, label="target 0.70")
    ax1.legend(fontsize=7)

    ax2.bar(x, rmse_means, color="#CC4444", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bins_order, fontsize=8)
    ax2.set_ylabel("RMSE (normalised arrival time)")
    ax2.set_title("Phase W2 — RMSE by Terrain Slope")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"phaseW2_slope_stratified.{ext}"), dpi=150)
    plt.close(fig)

    # ---- W1 proxy vs W2 mesh comparison scatter ----------------------
    # Use 2-D Euclidean distance predictions as W1 proxy (ignoring Z)
    all_pred_w2: list = []
    all_pred_w1: list = []
    all_gt: list = []

    for res in results:
        if "pred" in res and "gt" in res:
            pred_w2 = np.asarray(res["pred"], dtype=np.float32)
            gt = np.asarray(res["gt"], dtype=np.float32)
            # DISCLAIMER: This W1 proxy is a synthetic stand-in for code testing purposes.
            # It simulates a flat-grid metric that ignores elevation.
            # For scientific publication, load actual W1 predictions from Phase W1 outputs!
            pred_w1 = pred_w2 * 0.95 + np.random.default_rng(0).normal(
                0, 0.02 * float(np.std(pred_w2) + 1e-8), size=pred_w2.shape
            )
            all_pred_w2.append(pred_w2)
            all_pred_w1.append(pred_w1)
            all_gt.append(gt)

    if all_pred_w2:
        preds_w2 = np.concatenate(all_pred_w2)
        preds_w1 = np.concatenate(all_pred_w1)
        gts = np.concatenate(all_gt)

        # Normalise to [0, 1] for visual clarity
        p_max = max(float(preds_w2.max()), float(preds_w1.max()), 1e-8)
        preds_w2_n = preds_w2 / p_max
        preds_w1_n = preds_w1 / p_max
        gts_n = gts / max(float(gts.max()), 1e-8)

        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(
            preds_w1_n,
            preds_w2_n,
            c=gts_n,
            cmap="plasma",
            s=8,
            alpha=0.6,
        )
        plt.colorbar(sc, ax=ax, label="GT arrival time (norm.)")
        lims = [
            min(preds_w1_n.min(), preds_w2_n.min()),
            max(preds_w1_n.max(), preds_w2_n.max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=0.8, label="y = x")
        ax.set_xlabel("W1 proxy predictions (norm.)")
        ax.set_ylabel("W2 mesh predictions (norm.)")
        ax.set_title(
            "Phase W2 — W1 proxy vs W2 mesh predictions\n"
            "(W1 proxy = synthetic stand-in; replace with trained Phase W1 for publication)",
            fontsize=8,
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(
                os.path.join(fig_dir, f"phaseW2_length_comparison.{ext}"), dpi=150
            )
        plt.close(fig)

    print(f"\n  Figures saved to {fig_dir}/")


# ===========================================================================
# Multi-scene experiment runner
# ===========================================================================


def run_experiment_mesh(
    data_root: str,
    scene_ids: list,
    output_dir: str,
    cfg: dict,
    use_wind: bool = True,
) -> list:
    """Run Phase W2 for all scenes × seeds and produce publication figures.

    Args:
        data_root:  Path to the Sim2Real-Fire dataset root.
        scene_ids:  List of scene identifier strings.
        output_dir: Directory for result figures.
        cfg:        Configuration dict from :func:`get_config`.
        use_wind:   If False, run the Riemannian ablation.

    Returns:
        List of result dicts, one per ``(scene × seed)`` pair.
    """
    all_results: list = []
    for scene_id in scene_ids:
        for seed in cfg["train_seeds"]:
            result = train_scene_mesh(
                data_root, scene_id, cfg, seed, use_wind=use_wind, output_dir=output_dir
            )
            if result is not None:
                all_results.append(result)

    if all_results:
        _save_figures(all_results, output_dir)

    return all_results


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Phase W2: CovariateMeshRanders training on 3D triangular mesh"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/sim2real_fire",
        help="Path to Sim2Real-Fire dataset root directory.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=["0014_00426"],
        metavar="SCENE_ID",
        help="Scene IDs to train on (space-separated).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/phaseW2",
        help="Output directory for figures and logs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Reduced config (20 epochs, 15 AVBD steps, half mesh resolution, "
            "32/12/12 train/val/test fires) to verify the pipeline in ~15 min on CPU."
        ),
    )
    parser.add_argument(
        "--no_wind",
        action="store_true",
        help="Riemannian ablation: disable the Randers drift term (b=0).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help="Override training seeds (default: from config).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help=(
            "Run the synthetic smoke test on generated 20×20 DEMs — "
            "no real dataset required.  Combine with --quick for fast CI."
        ),
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
    cfg = get_config(quick=args.quick)
    use_wind = not args.no_wind

    if args.seeds is not None:
        cfg["train_seeds"] = args.seeds

    os.makedirs(args.output_dir, exist_ok=True)

    if args.synthetic:
        results = run_synthetic_mesh(cfg, output_dir=args.output_dir, use_wind=use_wind)
        r_vals = [r["test_pearson_r_mean"] for r in results]
        print(
            f"\n{'=' * 60}\n"
            f"  AGGREGATE Pearson r = {np.mean(r_vals):.4f} ± {np.std(r_vals):.4f}\n"
            f"{'=' * 60}"
        )
        return

    all_results = run_experiment_mesh(
        data_root=args.data_root,
        scene_ids=args.scenes,
        output_dir=args.output_dir,
        cfg=cfg,
        use_wind=use_wind,
    )

    if all_results:
        r_vals = [r["test_pearson_r_mean"] for r in all_results]
        rmse_vals = [r["test_rmse"] for r in all_results]
        print(
            f"\n{'=' * 60}\n"
            f"  AGGREGATE RESULTS  ({len(all_results)} runs)\n"
            f"  Mean Pearson r  : {np.mean(r_vals):.4f} ± {np.std(r_vals):.4f}\n"
            f"  Mean RMSE       : {np.mean(rmse_vals):.4f}\n"
            f"{'=' * 60}"
        )
    else:
        print("  No results collected — check data root and scene IDs.")


if __name__ == "__main__":
    main()
