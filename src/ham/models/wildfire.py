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

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional

from ham.geometry.metric import FinslerMetric
from ham.geometry.manifold import Manifold
from ham.utils.math import GRAD_EPS, NORM_EPS, PSD_EPS

__all__ = [
    "project_spd",
    "project_b_norm",
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
    disc = jnp.sqrt((g11 - g22) ** 2 + 4.0 * g12 ** 2 + 1e-12)
    lam_max = (trace + disc) * 0.5
    lam_min = (trace - disc) * 0.5

    lam_max_c = jnp.clip(lam_max, eps_min, eps_max)
    lam_min_c = jnp.clip(lam_min, eps_min, eps_max)

    theta = 0.5 * jnp.arctan2(2.0 * g12, g11 - g22 + 1e-12)
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
# CovariateConditionedRanders
# ---------------------------------------------------------------------------

class CovariateConditionedRanders(FinslerMetric):
    """Randers-type Finsler metric conditioned on terrain and weather covariates.

    The metric tensor ``G(x)`` and drift ``b(x)`` are produced by two MLPs:

    * ``global_mlp``: maps weather covariates ``(4,)`` → raw 5-vector.
    * ``local_mlp``:  maps terrain + fuel features ``(5 + fuel_emb_dim,)``
      → raw 5-vector.

    The raw 5-vectors are summed and split as ``[g11, g12, g22, b1, b2]``,
    then projected to valid Randers data via :func:`project_spd` and
    :func:`project_b_norm`.  Scene rasters are baked in via
    :meth:`bind_scene` and bilinearly interpolated at query positions inside
    :meth:`metric_fn`.

    The cost function follows the Zermelo navigation formula (identical in
    structure to :class:`ham.geometry.zoo.Randers`).

    Args:
        manifold:     Topological domain. Should be 2-dimensional.
        key:          JAX PRNG key for network initialisation.
        hidden_dim:   Width of hidden layers. Default: 128.
        fuel_emb_dim: Dimension of the FBFM-13 fuel-type embedding. Default: 4.
        eps_G:        Minimum eigenvalue of G. Default: 0.1.
        max_G:        Maximum eigenvalue of G. Default: 10.0.
        max_b_norm:   G^{-1}-norm bound for the drift. Default: 0.9.
        use_wind:     If False, sets ``b = 0`` (pure Riemannian). Default: True.

    Reference:
        spec/MATH_SPEC.md §§ 1–2, 5; spec/ARCH_SPEC.md § 3.

    See Also:
        bind_scene : Attach raster data to the model before calling
            :meth:`metric_fn`.
    """

    global_mlp: eqx.nn.MLP       # weather covariates (4,) → raw 5 params
    local_mlp: eqx.nn.MLP        # terrain+fuel covariates (5+fuel_emb_dim,) → raw 5 params
    fuel_embedding: jax.Array    # (13, fuel_emb_dim) — FBFM13 type embeddings

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

    def __init__(
        self,
        manifold: Manifold,
        key: jax.Array,
        hidden_dim: int = 128,
        fuel_emb_dim: int = 4,
        eps_G: float = 0.1,
        max_G: float = 10.0,
        max_b_norm: float = 0.9,
        use_wind: bool = True,
    ):
        """Initialise networks and embeddings.

        Args:
            manifold:     Topological domain (should be 2-D Euclidean).
            key:          JAX PRNG key; split internally for the two MLPs.
            hidden_dim:   MLP hidden-layer width.
            fuel_emb_dim: Fuel-type embedding dimension.
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

        k1, k2 = jax.random.split(key, 2)
        self.global_mlp = eqx.nn.MLP(
            in_size=4,
            out_size=5,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.tanh,
            key=k1,
        )
        self.local_mlp = eqx.nn.MLP(
            in_size=5 + fuel_emb_dim,
            out_size=5,
            width_size=hidden_dim,
            depth=3,
            activation=jax.nn.tanh,
            key=k2,
        )
        self.fuel_embedding = jnp.zeros((13, fuel_emb_dim), dtype=jnp.float64)

        # Rasters unset until bind_scene
        self.elev_raster = None
        self.slope_raster = None
        self.aspect_raster = None
        self.canopy_raster = None
        self.fuel_code_raster = None
        self.weather_vec = None
        self.pixel_spacing_m = jnp.asarray(1.0, dtype=jnp.float64)
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
            ),
            self,
            (
                jnp.asarray(elev, dtype=jnp.float64),
                jnp.asarray(slope, dtype=jnp.float64),
                jnp.asarray(aspect, dtype=jnp.float64),
                jnp.asarray(canopy, dtype=jnp.float64),
                jnp.asarray(fuel_codes, dtype=jnp.int32),
                jnp.asarray(weather_vec, dtype=jnp.float64),
                jnp.asarray(pixel_spacing_m, dtype=jnp.float64),
                jnp.asarray(origin_xy, dtype=jnp.float64),
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
            ),
            self,
            (
                jnp.asarray(elev, dtype=jnp.float64),
                jnp.asarray(slope, dtype=jnp.float64),
                jnp.asarray(aspect, dtype=jnp.float64),
                jnp.asarray(canopy, dtype=jnp.float64),
                jnp.asarray(fuel_codes, dtype=jnp.int32),
                jnp.asarray(pixel_spacing_m, dtype=jnp.float64),
                jnp.asarray(origin_xy, dtype=jnp.float64),
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
            jnp.asarray(weather_vec, dtype=jnp.float64),
            is_leaf=lambda x: x is None,
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
        px = (x_world[0] - self.scene_origin_xy[0]) / spacing
        py = (x_world[1] - self.scene_origin_xy[1]) / spacing
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

    def _get_covariates(self, x_world: jax.Array) -> tuple:
        """Assemble local terrain and global weather feature vectors.

        Args:
            x_world: (2,) world-coordinate query point.

        Returns:
            Tuple ``(local_features, weather_vec)`` with shapes
            ``(5 + fuel_emb_dim,)`` and ``(4,)``.
        """
        elev = self._bilinear_interp(self.elev_raster, x_world)
        slope = self._bilinear_interp(self.slope_raster, x_world)
        aspect_raw = self._bilinear_interp(self.aspect_raster, x_world)
        canopy = self._bilinear_interp(self.canopy_raster, x_world)

        # Integer nearest-neighbour lookup for fuel type
        spacing = jax.lax.stop_gradient(self.pixel_spacing_m)
        px = (x_world[0] - self.scene_origin_xy[0]) / spacing
        py = (x_world[1] - self.scene_origin_xy[1]) / spacing
        H, W_ = self.fuel_code_raster.shape
        ix = jnp.clip(jnp.round(px).astype(jnp.int32), 0, W_ - 1)
        iy = jnp.clip(jnp.round(py).astype(jnp.int32), 0, H - 1)
        fuel_code = self.fuel_code_raster[iy, ix]   # scalar int32
        fuel_code = jnp.clip(fuel_code, 0, 12)
        fuel_emb = self.fuel_embedding[fuel_code]   # (fuel_emb_dim,)

        local_features = jnp.concatenate([
            jnp.stack([elev, slope, jnp.sin(aspect_raw), jnp.cos(aspect_raw), canopy]),
            fuel_emb,
        ])  # (5 + fuel_emb_dim,)
        return local_features, self.weather_vec

    def _get_params(self, x_world: jax.Array) -> tuple:
        """Compute projected SPD metric G and drift b at a world-coordinate point.

        Args:
            x_world: (2,) world-coordinate query point.

        Returns:
            Tuple ``(G, b)`` where G is (2, 2) SPD and b is (2,).
        """
        local_feat, global_feat = self._get_covariates(x_world)
        raw_global = self.global_mlp(global_feat)   # (5,)
        raw_local = self.local_mlp(local_feat)       # (5,)
        raw = raw_global + raw_local                 # (5,) combined

        G_raw = jnp.stack([jnp.stack([raw[0], raw[1]]),
                           jnp.stack([raw[1], raw[2]])])
        G = project_spd(G_raw, self.eps_G, self.max_G)

        # use_wind is static — Python if is safe inside JIT
        if self.use_wind:
            b = project_b_norm(raw[3:5], G, self.max_b_norm)
        else:
            b = jnp.zeros(2, dtype=G.dtype)
        return G, b

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
