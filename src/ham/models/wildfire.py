"""Wildfire spread Finsler metric conditioned on terrain and weather covariates.

Provides :class:`CovariateConditionedRanders`, a Randers-type Finsler metric
whose geometry is a function of terrain (elevation, slope, aspect, canopy, fuel
type) and weather (temperature, humidity, wind direction).  Scene rasters are
"baked in" via :meth:`~CovariateConditionedRanders.bind_scene`, leaving the
model's MLP weights as the only trainable parameters.

Standalone helpers :func:`project_spd` and :func:`project_b_norm` project raw
network outputs to valid Randers data (SPD metric tensor + causality-bounded
drift) without calling :func:`jnp.linalg.eigh`, so they are safe under
``vmap`` and ``grad``.

Mathematical reference: spec/MATH_SPEC.md §§ 1–2 (Randers), § 5 (Zermelo).
Architecture reference: spec/ARCH_SPEC.md § 3 (LearnedFinsler).

See Also:
    ham.geometry.zoo.randers : Analytical Randers metric.
    ham.models.learned       : NeuralRanders (position-conditioned variant).
"""

from ham.utils.config import DEFAULT_JNP_DTYPE, DEFAULT_NP_DTYPE
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional

from ham.geometry.metric import FinslerMetric, AsymmetricMetric
from ham.geometry.manifold import Manifold
from ham.utils.math import GRAD_EPS, NORM_EPS, PSD_EPS

__all__ = [
    "project_spd",
    "project_b_norm",
    "LocalTerrainCNN",
    "CovariateConditionedRanders",
]


# ---------------------------------------------------------------------------
# Differentiable projection helpers
# ---------------------------------------------------------------------------

def project_spd(mat: jax.Array, eps_min: float, eps_max: float) -> jax.Array:
    """Project a 2×2 symmetric matrix to SPD via eigenvalue clamping.

    Computes eigenvalues from the trace/discriminant formula (no
    ``jnp.linalg.eigh``) so the function is stable under ``vmap`` and
    ``grad``.  The eigenvector rotation angle is recovered analytically.

    Args:
        mat:     (2, 2) symmetric matrix.
        eps_min: Minimum allowed eigenvalue.
        eps_max: Maximum allowed eigenvalue.

    Returns:
        (2, 2) SPD matrix with eigenvalues clamped to ``[eps_min, eps_max]``.

    Reference:
        spec/MATH_SPEC.md § 5 (Randers causality).
    """
    g11 = mat[0, 0]
    g12 = mat[0, 1]
    g22 = mat[1, 1]

    trace = g11 + g22
    disc = jnp.sqrt((g11 - g22) ** 2 + 4.0 * g12 ** 2 + 1e-8)  # 1e-8 prevents O(1e4) grad near isotropic point
    lam_max = (trace + disc) * 0.5
    lam_min = (trace - disc) * 0.5

    lam_max_c = jnp.clip(lam_max, eps_min, eps_max)
    lam_min_c = jnp.clip(lam_min, eps_min, eps_max)

    theta = 0.5 * jnp.arctan2(2.0 * g12, g11 - g22 + 1e-8)
    c = jnp.cos(theta)
    s = jnp.sin(theta)

    g11_new = lam_max_c * c ** 2 + lam_min_c * s ** 2
    g12_new = (lam_max_c - lam_min_c) * c * s
    g22_new = lam_max_c * s ** 2 + lam_min_c * c ** 2

    return jnp.stack([jnp.stack([g11_new, g12_new]),
                      jnp.stack([g12_new, g22_new])])


def project_b_norm(b: jax.Array, G_mat: jax.Array, max_norm: float = 0.9) -> jax.Array:
    """Project a drift vector so ``||b||_{G^{-1}} < max_norm``.

    The G^{-1}-norm is computed analytically for 2×2 ``G`` without matrix
    inversion, keeping the operation differentiable everywhere.

    Args:
        b:        (2,) drift vector.
        G_mat:    (2, 2) SPD matrix.
        max_norm: Target supremum for the G^{-1}-norm. Default: 0.9.

    Returns:
        (2,) scaled drift vector satisfying the causality constraint.

    Reference:
        spec/MATH_SPEC.md § 5 (Zermelo causality, ``||W||_H < 1``).
    """
    g11 = G_mat[0, 0]
    g12 = G_mat[0, 1]
    g22 = G_mat[1, 1]
    det_G = g11 * g22 - g12 ** 2
    det_G = jnp.maximum(det_G, 1e-8)

    # ||b||_{G^{-1}}^2 = b^T G^{-1} b
    #   = (b1^2 * g22 - 2*b1*b2*g12 + b2^2 * g11) / det(G)
    norm_sq = (b[0] ** 2 * g22
               - 2.0 * b[0] * b[1] * g12
               + b[1] ** 2 * g11) / det_G
    norm = jnp.sqrt(jnp.maximum(norm_sq, GRAD_EPS))

    scale = jnp.minimum(1.0, max_norm / (norm + NORM_EPS))
    return b * scale


# ---------------------------------------------------------------------------
# LocalTerrainCNN
# ---------------------------------------------------------------------------

class LocalTerrainCNN(eqx.Module):
    """Fully-convolutional terrain encoder mapping raster stacks to local metric params.

    Three 3×3 conv layers (same-padding) give a 7×7-pixel receptive field at
    each output location, matching the CNN encoder architecture of Gahtan et al.
    (2026) Section 6.  A final 1×1 conv projects the concatenated CNN features
    and per-pixel fuel embeddings to a 5-element raw-parameter vector.

    Args:
        fuel_emb_dim: Dimension of the fuel-type embedding (channels added after CNN).
        n_channels:   Number of channels in each conv layer.  Default: 64.
        key:          JAX PRNG key for weight initialisation.

    Reference:
        Gahtan, Shpund & Bronstein (2026). arXiv:2603.00035, Section 6.
    """

    conv1: eqx.nn.Conv2d   # 5 → n_channels, kernel 3
    conv2: eqx.nn.Conv2d   # n_channels → n_channels, kernel 3
    conv3: eqx.nn.Conv2d   # n_channels → n_channels, kernel 3
    head:  eqx.nn.Conv2d   # (n_channels + fuel_emb_dim + weather_dim) → 5, kernel 1
    n_channels: int = eqx.field(static=True)
    fuel_emb_dim: int = eqx.field(static=True)
    weather_dim: int = eqx.field(static=True)

    def __init__(self, fuel_emb_dim: int, n_channels: int, key: jax.Array, weather_dim: int = 4):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.n_channels = int(n_channels)
        self.fuel_emb_dim = int(fuel_emb_dim)
        self.weather_dim = int(weather_dim)
        self.conv1 = eqx.nn.Conv2d(5, n_channels, 3, padding=1, dtype=DEFAULT_JNP_DTYPE, key=k1)
        self.conv2 = eqx.nn.Conv2d(n_channels, n_channels, 3, padding=1, dtype=DEFAULT_JNP_DTYPE, key=k2)
        self.conv3 = eqx.nn.Conv2d(n_channels, n_channels, 3, padding=1, dtype=DEFAULT_JNP_DTYPE, key=k3)
        self.head  = eqx.nn.Conv2d(n_channels + fuel_emb_dim + weather_dim, 5, 1, dtype=DEFAULT_JNP_DTYPE, key=k4)

    def __call__(
        self,
        raster_stack: jax.Array,
        fuel_field: jax.Array,
        weather_vec: jax.Array,
    ) -> jax.Array:
        """Encode terrain rasters to per-pixel local raw metric parameters.

        Args:
            raster_stack: ``(5, H, W)`` float64 — stacked terrain channels
                          ``[elev, slope, sin(aspect), cos(aspect), canopy]``.
            fuel_field:   ``(fuel_emb_dim, H, W)`` float64 — learned fuel-type
                          embeddings broadcast over the spatial grid.
            weather_vec:  ``(weather_dim,)`` float64 — per-fire weather features
                          ``[T_air, humidity, sin_wind, cos_wind]``, broadcast
                          spatially so the head can learn terrain-weather interactions.

        Returns:
            ``(H, W, 5)`` float64 raw local metric parameters ``[g11, g12, g22, b1, b2]``.
        """
        x = jax.nn.relu(self.conv1(raster_stack))          # (n_channels, H, W)
        x = jax.nn.relu(self.conv2(x))                     # (n_channels, H, W)
        x = jax.nn.relu(self.conv3(x))                     # (n_channels, H, W)
        H, W = raster_stack.shape[1], raster_stack.shape[2]
        weather_spatial = jnp.broadcast_to(
            weather_vec[:, None, None], (self.weather_dim, H, W)
        )                                                    # (weather_dim, H, W)
        combined = jnp.concatenate(
            [x, fuel_field, weather_spatial], axis=0
        )                                                    # (n_channels + fuel_emb_dim + weather_dim, H, W)
        raw_local = self.head(combined)                    # (5, H, W)
        return raw_local.transpose(1, 2, 0)                # (H, W, 5)


# ---------------------------------------------------------------------------
# CovariateConditionedRanders
# ---------------------------------------------------------------------------

class CovariateConditionedRanders(AsymmetricMetric):
    """Randers-type Finsler metric conditioned on terrain and weather covariates.

    The metric tensor ``G(x)`` and drift ``b(x)`` are produced by two pathways:

    * ``global_mlp``: maps weather covariates ``(4,)`` → raw 5-vector.
    * ``local_cnn``:  a fully-convolutional encoder maps terrain rasters
      ``(5, H, W)`` and fuel embeddings ``(fuel_emb_dim, H, W)`` → per-pixel
      raw 5-vector field ``(H, W, 5)`` with a 7×7-pixel receptive field.

    The local field is precomputed once per training step via
    :meth:`precompute_metric_field` and stored as ``metric_field``.  At each
    query point the local contribution is read out by bilinear interpolation
    (differentiable w.r.t. position, preserving the Euler-Lagrange spatial
    gradient for AVBD geodesic computation).

    The raw 5-vectors are summed and split as ``[g11, g12, g22, b1, b2]``,
    then projected to valid Randers data via :func:`project_spd` and
    :func:`project_b_norm`.  Scene rasters are baked in via
    :meth:`bind_scene` and the metric field is populated via
    :meth:`precompute_metric_field` inside the differentiable training step.

    The cost function follows the Zermelo navigation formula (identical in
    structure to :class:`ham.geometry.zoo.Randers`).

    Args:
        manifold:     Topological domain. Should be 2-dimensional.
        key:          JAX PRNG key for network initialisation.
        hidden_dim:   Width of hidden layers in the global (weather) MLP.
            Default: 128.
        fuel_emb_dim: Dimension of the FBFM-13 fuel-type embedding. Default: 4.
        cnn_channels: Number of channels in each conv layer of the terrain CNN.
            Default: 64.  Total trainable parameters ≈ 95K at these defaults.
        eps_G:        Minimum eigenvalue of G. Default: 0.1.
        max_G:        Maximum eigenvalue of G. Default: 10.0.
        max_b_norm:   G^{-1}-norm bound for the drift. Default: 0.9.
        use_wind:     If False, sets ``b = 0`` (pure Riemannian). Default: True.

    Reference:
        spec/MATH_SPEC.md §§ 1–2, 5; spec/ARCH_SPEC.md § 3.
        Gahtan, Shpund & Bronstein (2026). arXiv:2603.00035, Section 6.

    See Also:
        precompute_metric_field : Must be called inside the training loss before
            :meth:`metric_fn` is invoked.
        bind_scene : Attach raster data to the model before calling
            :meth:`precompute_metric_field`.
    """

    global_mlp: eqx.nn.MLP       # weather covariates (4,) → raw 5 params
    local_cnn: LocalTerrainCNN   # fully-conv terrain encoder: (5,H,W) → (H,W,5)
    fuel_embedding: jax.Array    # (13, fuel_emb_dim) — FBFM13 type embeddings

    # Per-step precomputed local metric parameter field; set by precompute_metric_field().
    # None in the unbound model; (H, W, 5) float64 after precomputation.
    metric_field: Optional[jax.Array]      # (H, W, 5) float64 local raw params

    # Baked-in scene covariates (None until bind_scene is called)
    elev_raster: Optional[jax.Array]       # (H, W) float64
    slope_raster: Optional[jax.Array]      # (H, W) float64
    aspect_raster: Optional[jax.Array]     # (H, W) float64
    canopy_raster: Optional[jax.Array]     # (H, W) float64
    fuel_code_raster: Optional[jax.Array]  # (H, W) int32
    weather_vec: Optional[jax.Array]       # (4,) [T_air, humidity, sin_wind, cos_wind]
    # Stored as a regular JAX leaf (not static) so eqx.tree_at can update it in
    # bind_scene. Inside metric computations it is wrapped in stop_gradient to
    # prevent unintended gradient flow through the grid resolution.
    pixel_spacing_m: jax.Array             # scalar float64, metres per pixel
    scene_origin_xy: Optional[jax.Array]   # (2,) world coords of pixel (0,0) in metres

    # Configuration constants (static — known at JIT compile time)
    eps_G: float = eqx.field(static=True)
    max_G: float = eqx.field(static=True)
    max_b_norm: float = eqx.field(static=True)
    use_wind: bool = eqx.field(static=True)
    fuel_emb_dim: int = eqx.field(static=True)
    cnn_channels: int = eqx.field(static=True)

    def __init__(
        self,
        manifold: Manifold,
        key: jax.Array,
        hidden_dim: int = 128,
        fuel_emb_dim: int = 4,
        cnn_channels: int = 64,
        eps_G: float = 0.1,
        max_G: float = 10.0,
        max_b_norm: float = 0.9,
        use_wind: bool = True,
    ):
        """Initialise networks and embeddings.

        Args:
            manifold:     Topological domain (should be 2-D Euclidean).
            key:          JAX PRNG key; split internally for the two networks.
            hidden_dim:   Width of hidden layers in the global (weather) MLP.
            fuel_emb_dim: Dimension of the FBFM-13 fuel-type embedding.
            cnn_channels: Number of channels in each convolutional layer of the
                          local terrain CNN encoder.  Default: 64.
            eps_G:        Minimum eigenvalue of the metric tensor G.
            max_G:        Maximum eigenvalue of the metric tensor G.
            max_b_norm:   Upper bound on ``||b||_{G^{-1}}``.
            use_wind:     Include Randers drift (directional asymmetry).
        """
        self.manifold = manifold
        self.eps_G = float(eps_G)
        self.max_G = float(max_G)
        self.max_b_norm = float(max_b_norm)
        self.use_wind = bool(use_wind)
        self.fuel_emb_dim = int(fuel_emb_dim)
        self.cnn_channels = int(cnn_channels)

        k1, k2 = jax.random.split(key, 2)
        self.global_mlp = eqx.nn.MLP(
            in_size=4,
            out_size=5,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.tanh,
            key=k1,
        )
        self.local_cnn = LocalTerrainCNN(
            fuel_emb_dim=fuel_emb_dim,
            n_channels=cnn_channels,
            key=k2,
            weather_dim=4,  # [T_air, humidity, sin_wind, cos_wind]
        )
        self.fuel_embedding = jnp.zeros((13, fuel_emb_dim), dtype=DEFAULT_JNP_DTYPE)

        # metric_field is None until precompute_metric_field() is called
        self.metric_field = None

        # Rasters unset until bind_scene
        self.elev_raster = None
        self.slope_raster = None
        self.aspect_raster = None
        self.canopy_raster = None
        self.fuel_code_raster = None
        self.weather_vec = None
        self.pixel_spacing_m = jnp.asarray(1.0, dtype=DEFAULT_JNP_DTYPE)
        self.scene_origin_xy = None

    def bind_scene(
        self,
        elev: jax.Array,
        slope: jax.Array,
        aspect: jax.Array,
        canopy: jax.Array,
        fuel_codes: jax.Array,
        weather_vec: jax.Array,
        pixel_spacing_m: float,
        origin_xy: jax.Array,
    ) -> "CovariateConditionedRanders":
        """Return a new instance with scene rasters and weather baked in.

        All rasters must be (H, W) JAX arrays.  The returned model is a frozen
        pytree with scene data as constants; only the MLP weights are
        differentiable leaves.

        Args:
            elev:            (H, W) elevation raster [m].
            slope:           (H, W) slope raster [radians or degrees].
            aspect:          (H, W) aspect raster [radians].
            canopy:          (H, W) canopy cover raster [0, 1].
            fuel_codes:      (H, W) integer FBFM-13 fuel codes (0–12).
            weather_vec:     (4,) ``[T_air, humidity, sin_wind, cos_wind]``.
            pixel_spacing_m: Grid resolution in metres per pixel.
            origin_xy:       (2,) world-space position of pixel (0, 0) [m].

        Returns:
            New :class:`CovariateConditionedRanders` with scene attached.
        """
        return eqx.tree_at(
            lambda m: (
                m.elev_raster,
                m.slope_raster,
                m.aspect_raster,
                m.canopy_raster,
                m.fuel_code_raster,
                m.weather_vec,
                m.pixel_spacing_m,
                m.scene_origin_xy,
                m.metric_field,
            ),
            self,
            (
                jnp.asarray(elev, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(slope, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(aspect, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(canopy, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(fuel_codes, dtype=jnp.int32),
                jnp.asarray(weather_vec, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(pixel_spacing_m, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(origin_xy, dtype=DEFAULT_JNP_DTYPE),
                None,
            ),
            is_leaf=lambda x: x is None,
        )

    def bind_scene_rasters(
        self,
        elev: jax.Array,
        slope: jax.Array,
        aspect: jax.Array,
        canopy: jax.Array,
        fuel_codes: jax.Array,
        pixel_spacing_m: float,
        origin_xy: jax.Array,
    ) -> "CovariateConditionedRanders":
        """Return a new instance with terrain rasters baked in but no weather.

        Use this in combination with :meth:`bind_weather` for batched training
        where multiple fires share the same terrain but differ in weather:

        .. code-block:: python

            scene_metric = unbound_metric.bind_scene_rasters(elev, slope, ...)
            # Inside vmap over fires:
            fire_metric = scene_metric.bind_weather(weather_vec)

        Args:
            elev, slope, aspect, canopy, fuel_codes: Scene rasters.
            pixel_spacing_m: Grid resolution in metres per pixel.
            origin_xy: (2,) world-space position of pixel (0, 0) [m].

        Returns:
            New instance with terrain rasters set; ``weather_vec`` stays None.
        """
        return eqx.tree_at(
            lambda m: (
                m.elev_raster,
                m.slope_raster,
                m.aspect_raster,
                m.canopy_raster,
                m.fuel_code_raster,
                m.pixel_spacing_m,
                m.scene_origin_xy,
                m.metric_field,
            ),
            self,
            (
                jnp.asarray(elev, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(slope, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(aspect, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(canopy, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(fuel_codes, dtype=jnp.int32),
                jnp.asarray(pixel_spacing_m, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(origin_xy, dtype=DEFAULT_JNP_DTYPE),
                None,
            ),
            is_leaf=lambda x: x is None,
        )

    def bind_weather(self, weather_vec: jax.Array) -> "CovariateConditionedRanders":
        """Return a new instance with only the weather vector updated.

        Designed for use inside ``jax.vmap`` over fire batches.  Terrain rasters
        must already be baked in via :meth:`bind_scene_rasters` (called once
        per scene before the vmap) so that only the 4-element weather vector
        differs across fires in the batch.

        Args:
            weather_vec: (4,) ``[T_air, humidity, sin_wind, cos_wind]``.

        Returns:
            New instance with ``weather_vec`` updated.
        """
        return eqx.tree_at(
            lambda m: m.weather_vec,
            self,
            jnp.asarray(weather_vec, dtype=DEFAULT_JNP_DTYPE),
            is_leaf=lambda x: x is None,
        )

    def precompute_metric_field(self) -> "CovariateConditionedRanders":
        """Run the terrain CNN over the full scene rasters and cache the result.

        Must be called inside the differentiable forward pass (e.g., inside
        ``fire_loss(m)`` before passing ``m`` to the solver) so that gradients
        flow back through the CNN weights.  The returned model shares all
        weights with ``self`` but has ``metric_field`` set to a ``(H, W, 5)``
        array of raw local metric parameters ready for bilinear interpolation.

        Returns:
            New :class:`CovariateConditionedRanders` with ``metric_field`` set.

        Note:
            Calling this outside a gradient context (e.g., at bind time) is
            valid for inference but will prevent gradients from flowing to the
            CNN weights during training.
        """
        # stop_gradient on all rasters: they are frozen scene data, not trainable
        # parameters.  Without this guard, backpropagation through the CNN would
        # accumulate non-zero gradients w.r.t. the raster leaves, and any call to
        # eqx.apply_updates on a terrain-bound model would corrupt the raster values.
        raster_stack = jnp.stack([
            jax.lax.stop_gradient(self.elev_raster),
            jax.lax.stop_gradient(self.slope_raster),
            jnp.sin(jax.lax.stop_gradient(self.aspect_raster)),
            jnp.cos(jax.lax.stop_gradient(self.aspect_raster)),
            jax.lax.stop_gradient(self.canopy_raster),
        ], axis=0).astype(DEFAULT_JNP_DTYPE)  # (5, H, W)

        # Per-pixel fuel embeddings: (H, W, fuel_emb_dim) → (fuel_emb_dim, H, W)
        fuel_codes_clipped = jnp.clip(jax.lax.stop_gradient(self.fuel_code_raster), 0, 12)
        fuel_field = self.fuel_embedding[fuel_codes_clipped]  # (H, W, fuel_emb_dim)
        fuel_field = fuel_field.transpose(2, 0, 1).astype(DEFAULT_JNP_DTYPE)

        if self.weather_vec is None:
            raise ValueError(
                "precompute_metric_field() requires weather to be bound first. "
                "Call bind_weather() or bind_scene() before calling this method."
            )
        # stop_gradient on weather_vec: it is scene input data, not a trainable
        # parameter.  Gradient of loss w.r.t. CNN weights still flows correctly
        # because weather is a constant conditioning input from the CNN's perspective.
        weather = jax.lax.stop_gradient(self.weather_vec).astype(DEFAULT_JNP_DTYPE)

        field = self.local_cnn(raster_stack, fuel_field, weather)  # (H, W, 5)
        return eqx.tree_at(
            lambda m: m.metric_field, self, field, is_leaf=lambda x: x is None
        )

    def _bilinear_interp(self, raster: jax.Array, x_world: jax.Array) -> jax.Array:
        """Bilinear interpolation of a (H, W) raster at a world-coordinate point.

        Args:
            raster:  (H, W) float64 array.
            x_world: (2,) ``[x_m, y_m]`` world-coordinate position.

        Returns:
            Scalar interpolated value.
        """
        spacing = jax.lax.stop_gradient(self.pixel_spacing_m)
        origin  = jax.lax.stop_gradient(self.scene_origin_xy)
        px = (x_world[0] - origin[0]) / spacing
        py = (x_world[1] - origin[1]) / spacing
        H, W = raster.shape
        px = jnp.clip(px, 0.0, W - 1.001)
        py = jnp.clip(py, 0.0, H - 1.001)
        x0 = jnp.floor(px).astype(jnp.int32)
        y0 = jnp.floor(py).astype(jnp.int32)
        x1 = jnp.minimum(x0 + 1, W - 1)
        y1 = jnp.minimum(y0 + 1, H - 1)
        fx = px - x0
        fy = py - y0
        v00 = raster[y0, x0]
        v01 = raster[y0, x1]
        v10 = raster[y1, x0]
        v11 = raster[y1, x1]
        return (v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy)

    def _bilinear_interp_field(
        self, field: jax.Array, x_world: jax.Array
    ) -> jax.Array:
        """Bilinear interpolation of a ``(H, W, C)`` field at a world-coordinate point.

        This is the differentiable lookup used in :meth:`_get_params`.  Gradients
        flow through ``x_world`` (fractional pixel sub-position) so that the
        geodesic Euler-Lagrange equations see the correct spatial metric gradient.

        Args:
            field:   ``(H, W, C)`` float64 array (e.g. ``metric_field``).
            x_world: ``(2,)`` ``[x_m, y_m]`` world-coordinate position.

        Returns:
            ``(C,)`` interpolated feature vector.
        """
        spacing = jax.lax.stop_gradient(self.pixel_spacing_m)
        origin  = jax.lax.stop_gradient(self.scene_origin_xy)
        px = (x_world[0] - origin[0]) / spacing
        py = (x_world[1] - origin[1]) / spacing
        H, W, _ = field.shape
        px = jnp.clip(px, 0.0, W - 1.001)
        py = jnp.clip(py, 0.0, H - 1.001)
        x0 = jnp.floor(px).astype(jnp.int32)
        y0 = jnp.floor(py).astype(jnp.int32)
        x1 = jnp.minimum(x0 + 1, W - 1)
        y1 = jnp.minimum(y0 + 1, H - 1)
        fx = px - x0
        fy = py - y0
        return (field[y0, x0] * (1.0 - fx) * (1.0 - fy)
                + field[y0, x1] * fx * (1.0 - fy)
                + field[y1, x0] * (1.0 - fx) * fy
                + field[y1, x1] * fx * fy)

    def _get_params(self, x_world: jax.Array) -> tuple:
        """Compute projected SPD metric G and drift b at a world-coordinate point.

        Bilinearly interpolates the precomputed local metric field for the
        terrain contribution, then adds the global (weather) MLP output and
        projects to valid Randers data.

        Args:
            x_world: (2,) world-coordinate query point.

        Returns:
            Tuple ``(G, b)`` where G is (2, 2) SPD and b is (2,).
        """
        raw_local = self._bilinear_interp_field(self.metric_field, x_world)  # (5,)
        raw_global = self.global_mlp(self.weather_vec)                        # (5,)
        raw = raw_global + raw_local                                          # (5,)

        G_raw = jnp.stack([jnp.stack([raw[0], raw[1]]),
                           jnp.stack([raw[1], raw[2]])])
        G = project_spd(G_raw, self.eps_G, self.max_G)

        # use_wind is static — Python if is safe inside JIT
        if self.use_wind:
            b = project_b_norm(raw[3:5], G, self.max_b_norm)
        else:
            b = jnp.zeros(2, dtype=G.dtype)
        return G, b

    def zermelo_data(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the Zermelo navigation triple ``(H, W, lambda)`` at position x.

        Implements the :class:`~ham.geometry.metric.AsymmetricMetric` interface
        so that downstream consumers (losses, VAE) can use
        ``isinstance(metric, AsymmetricMetric)`` instead of fragile
        ``hasattr`` duck-typing.

        Args:
            x: (2,) world-coordinate position.

        Returns:
            H:      (2, 2) contravariant SPD metric tensor (the Riemannian sea).
            W:      (2,) contravariant drift vector (the wind).
            lambda: Causality scalar ``1 - ||W||^2_{H^{-1}}``.
        """
        G, b = self._get_params(x)
        det_G = jnp.maximum(G[0, 0] * G[1, 1] - G[0, 1] ** 2, 1e-8)
        
        # Ginv is the inverse of the covariant G. In Zermelo formulation, H = G^{-1}.
        Ginv11 = G[1, 1] / det_G
        Ginv22 = G[0, 0] / det_G
        Ginv12 = -G[0, 1] / det_G
        
        H = jnp.stack([jnp.stack([Ginv11, Ginv12]),
                       jnp.stack([Ginv12, Ginv22])])
        
        # W = G^{-1} * b = H * b
        W1 = Ginv11 * b[0] + Ginv12 * b[1]
        W2 = Ginv12 * b[0] + Ginv22 * b[1]
        W = jnp.array([W1, W2])
        
        b_Ginv_b = b[0]*W1 + b[1]*W2
        lam = jnp.maximum(1.0 - b_Ginv_b, 1e-6)
        
        return H, W, lam

    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Randers-Zermelo cost F(x, v).

        Implements the Zermelo navigation formula:

        .. math::

            F(x, v) = \\frac{\\sqrt{\\lambda \\, v^\\top G v + (b^\\top v)^2}
                             - b^\\top v}{\\lambda}

        where :math:`\\lambda = 1 - b^\\top G^{-1} b`.

        The formula is identical in structure to
        :meth:`ham.geometry.zoo.Randers.metric_fn`, with ``G`` and ``b``
        derived from covariate-conditioned MLPs rather than position networks.

        Args:
            x: (2,) world-coordinate position.
            v: (2,) tangent vector (fire-spread direction × speed).

        Returns:
            Scalar Finsler cost F(x, v) ≥ 0. Shape: ``()``.

        Reference:
            spec/MATH_SPEC.md § 5 (Zermelo navigation formula).
        """
        v_sq_raw = jnp.sum(v ** 2)
        is_zero = v_sq_raw < GRAD_EPS
        # Use a neutral unit-norm direction (1/√2, 1/√2) scaled to √(GRAD_EPS/2) to
        # avoid the diagonal bias that adding √GRAD_EPS to both components introduces.
        v_zero_safe = jnp.array([jnp.sqrt(GRAD_EPS / 2.0), jnp.sqrt(GRAD_EPS / 2.0)])
        v_safe = jnp.where(is_zero, v_zero_safe, v)

        G, b = self._get_params(x)

        det_G = G[0, 0] * G[1, 1] - G[0, 1] ** 2
        det_G = jnp.maximum(det_G, 1e-8)

        # λ = 1 - b^T G^{-1} b
        b_Ginv_b = (b[0] ** 2 * G[1, 1]
                    - 2.0 * b[0] * b[1] * G[0, 1]
                    + b[1] ** 2 * G[0, 0]) / det_G
        lam = jnp.maximum(1.0 - b_Ginv_b, 1e-6)

        Gv = jnp.dot(G, v_safe)
        v_sq_h = jnp.dot(v_safe, Gv)
        bdotv = jnp.dot(b, v_safe)

        disc = lam * v_sq_h + bdotv ** 2
        cost = (jnp.sqrt(jnp.maximum(disc, GRAD_EPS)) - bdotv) / lam
        return jnp.where(is_zero, 0.0, cost)
