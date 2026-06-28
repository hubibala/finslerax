"""Terrain mesh utilities for HAMTools.

Provides helpers for constructing triangular meshes from DEM rasters,
interpolating raster covariates to mesh vertices, computing face normals and
slope/aspect, and a Randers metric on a TriangularMesh conditioned on per-face
covariates.

Mathematical reference: spec/MATH_SPEC.md §§ 1–2, 5.
Architecture reference: spec/ARCH_SPEC.md § 5.
"""


import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.mesh import TriangularMesh
from ham.geometry.metric import AsymmetricMetric
from ham.models.wildfire import project_b_norm, project_spd
from ham.utils.config import DEFAULT_JNP_DTYPE
from ham.utils.math import GRAD_EPS, safe_norm

__all__ = [
    "CovariateMeshRanders",
    "compute_face_normals",
    "compute_face_slopes_aspects",
    "dem_to_mesh",
    "interpolate_covariates_to_vertices",
    "pixel_to_world_3d",
]


# ---------------------------------------------------------------------------
# DEM → mesh
# ---------------------------------------------------------------------------


def dem_to_mesh(
    elevation_raster: jnp.ndarray,
    pixel_spacing_m: float = 30.0,
) -> TriangularMesh:
    """Build a TriangularMesh from a digital elevation model raster.

    Vertices are placed at world positions ``(j * pixel_spacing_m,
    i * pixel_spacing_m, elevation_raster[i, j])`` for every pixel ``(i, j)``.
    Each grid cell ``(i, j)`` with ``i < H-1`` and ``j < W-1`` is split into
    two triangles:

    * Lower-left: ``(i*W+j, (i+1)*W+j, i*W+j+1)``
    * Upper-right: ``((i+1)*W+j+1, i*W+j+1, (i+1)*W+j)``

    Args:
        elevation_raster: (H, W) array of elevation values in metres.
        pixel_spacing_m:  Ground sampling distance in metres per pixel.
            Default: 30.0 (SRTM 1-arc-second approximate).

    Returns:
        TriangularMesh with vertices shape ``(H*W, 3)`` and faces shape
        ``(2*(H-1)*(W-1), 3)`` (int32).

    Reference:
        spec/ARCH_SPEC.md § 5 (Module Structure — terrain).
    """
    H, W = elevation_raster.shape

    # Build vertex grid: k = i*W + j → (x, y, z)
    i_idx = jnp.arange(H)
    j_idx = jnp.arange(W)
    jj, ii = jnp.meshgrid(j_idx, i_idx)  # (H, W) each
    xs = jj * pixel_spacing_m
    ys = ii * pixel_spacing_m
    zs = elevation_raster
    vertices = jnp.stack([xs, ys, zs], axis=-1).reshape(H * W, 3)

    # Build faces
    i_f = jnp.arange(H - 1)
    j_f = jnp.arange(W - 1)
    jj_f, ii_f = jnp.meshgrid(j_f, i_f)  # (H-1, W-1) each
    ii_f = ii_f.ravel()
    jj_f = jj_f.ravel()

    k00 = ii_f * W + jj_f
    k10 = (ii_f + 1) * W + jj_f
    k01 = ii_f * W + jj_f + 1
    k11 = (ii_f + 1) * W + jj_f + 1

    tri_lo = jnp.stack([k00, k10, k01], axis=-1)  # lower-left
    tri_hi = jnp.stack([k11, k01, k10], axis=-1)  # upper-right
    faces = jnp.concatenate([tri_lo, tri_hi], axis=0).astype(jnp.int32)

    return TriangularMesh(vertices.astype(DEFAULT_JNP_DTYPE), faces)


# ---------------------------------------------------------------------------
# Covariate interpolation
# ---------------------------------------------------------------------------


def interpolate_covariates_to_vertices(
    mesh: TriangularMesh,
    raster_dict: dict,
    H: int,
    W: int,
) -> jnp.ndarray:
    """Extract raster covariate values at mesh vertices (aligned pixels).

    Vertices produced by :func:`dem_to_mesh` are aligned with raster pixels,
    so the ``(i, j)`` pixel value is simply read at vertex ``k = i*W + j``.

    Args:
        mesh:        TriangularMesh with ``H*W`` vertices.
        raster_dict: Dict mapping covariate name → (H, W) array.
        H:           Raster height (number of rows).
        W:           Raster width (number of columns).

    Returns:
        Array of shape ``(H*W, num_covariates)`` with covariates stacked in
        ``raster_dict.keys()`` order.

    Reference:
        spec/ARCH_SPEC.md § 5 (terrain utilities).
    """
    if mesh.vertices.shape[0] != H * W:
        raise ValueError(
            f"Mesh has {mesh.vertices.shape[0]} vertices but H*W={H * W}. "
            "Ensure the mesh was created from the same raster dimensions."
        )
    channels = []
    for _name, raster in raster_dict.items():
        flat = jnp.asarray(raster).reshape(H * W)
        channels.append(flat)
    return jnp.stack(channels, axis=-1)  # (H*W, num_covariates)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def pixel_to_world_3d(
    i_row,
    j_col,
    elevation,
    pixel_spacing_m: float = 30.0,
) -> jnp.ndarray:
    """Convert (row, col, elevation) to 3D world coordinates in metres.

    Args:
        i_row:           Pixel row index (y direction).
        j_col:           Pixel column index (x direction).
        elevation:       Elevation value in metres.
        pixel_spacing_m: Ground sampling distance in metres. Default: 30.0.

    Returns:
        (3,) array ``[x, y, z]`` in metres.
    """
    return jnp.array([j_col * pixel_spacing_m, i_row * pixel_spacing_m, elevation])


# ---------------------------------------------------------------------------
# Face normals / slope / aspect
# ---------------------------------------------------------------------------


def compute_face_normals(mesh: TriangularMesh) -> jnp.ndarray:
    """Compute per-face unit outward normals.

    Args:
        mesh: TriangularMesh with triangles of shape (F, 3, N).

    Returns:
        Array of shape (F, 3) of unit normals.

    Reference:
        spec/MATH_SPEC.md § 5 (terrain geometry).
    """
    v0 = mesh.triangles[:, 0, :]  # (F, 3)
    v1 = mesh.triangles[:, 1, :]
    v2 = mesh.triangles[:, 2, :]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = jnp.cross(e1, e2)  # (F, 3)
    norms = safe_norm(normals, axis=-1, keepdims=True)
    return normals / norms


def compute_face_slopes_aspects(
    mesh: TriangularMesh,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute slope and aspect angles for each mesh face.

    * slope  = arccos(|n_z|) — deviation from horizontal (radians),
      in [0, π/2] regardless of face winding orientation.
    * aspect = atan2(n_y, n_x) w.r.t. the upward-oriented normal — gives
      the **uphill** azimuth direction (not the GIS downhill convention).

    Args:
        mesh: TriangularMesh.

    Returns:
        Tuple ``(slopes, aspects)`` each of shape ``(F,)``.

    Reference:
        spec/MATH_SPEC.md § 5 (terrain geometry).
    """
    normals = compute_face_normals(mesh)
    # Use |n_z| so slope ∈ [0, π/2] regardless of face winding orientation.
    # A flat surface always gives slope = 0; aspect is taken from the
    # upward-oriented normal.
    n_z_abs = jnp.abs(normals[:, 2])
    # Flip sign of x/y components when n_z < 0 to get the upward normal.
    sign = jnp.sign(normals[:, 2] + 1e-12)  # +1 if n_z >= 0, else -1
    slopes = jnp.arccos(jnp.clip(n_z_abs, 0.0, 1.0))
    aspects = jnp.arctan2(sign * normals[:, 1], sign * normals[:, 0])
    return slopes, aspects


# ---------------------------------------------------------------------------
# CovariateMeshRanders
# ---------------------------------------------------------------------------


class CovariateMeshRanders(AsymmetricMetric):
    """Randers metric on a TriangularMesh with per-face covariate conditioning.

    Uses the same global/local MLP encoder architecture as
    :class:`~ham.models.wildfire.CovariateConditionedRanders` but operates on
    3D mesh faces.  Face Randers parameters ``(G_f, b_f)`` are computed from
    per-face covariate vectors via the shared encoder.

    The metric operates in ambient 3D.  Given a query point ``x``, the
    nearest face is identified via softmax-weighted face proximity
    (:meth:`~ham.geometry.mesh.TriangularMesh.get_face_weights`).  The
    velocity ``v`` is projected onto the face tangent plane, then expressed in
    the face's local 2D orthonormal frame to apply the 2×2 Randers cost.

    .. note::

        **Phase W2 approximation**: The 2×2 metric tensor ``G_f`` is applied
        in isotropic form ``0.5*(G_f[0,0]+G_f[1,1]) * I_3`` lifted to 3D.
        The full anisotropic pullback ``G_3d = sum_ij G_f[i,j] u_i ⊗ u_j``
        (where ``{u1, u2}`` is the face orthonormal frame) is Phase W2b future
        work.

    Attributes:
        global_mlp:       Weather baseline encoder (4,) → (5,).
        local_mlp:        Terrain/fuel encoder (5+fuel_emb_dim,) → (5,).
        fuel_embedding:   (13, fuel_emb_dim) trainable embeddings.
        face_covariates:  (F, 5+fuel_emb_dim) per-face local covariates
            (precomputed by caller).
        weather_vec:      (4,) global weather for this fire event.
        eps_G:            Minimum eigenvalue of G.
        max_G:            Maximum eigenvalue of G.
        max_b_norm:       G^{-1}-norm bound for drift.
        use_wind:         If False, b = 0 (Riemannian mode).
        fuel_emb_dim:     Fuel-type embedding dimension.

    Reference:
        spec/MATH_SPEC.md §§ 1–2, 5; spec/ARCH_SPEC.md § 3.
    """

    global_mlp: eqx.nn.MLP
    local_mlp: eqx.nn.MLP
    fuel_embedding: jax.Array
    face_covariates: jax.Array  # (F, 5+fuel_emb_dim) precomputed
    weather_vec: jax.Array  # (4,)
    eps_G: float = eqx.field(static=True)
    max_G: float = eqx.field(static=True)
    max_b_norm: float = eqx.field(static=True)
    use_wind: bool = eqx.field(static=True)
    fuel_emb_dim: int = eqx.field(static=True)

    def __init__(
        self,
        manifold: TriangularMesh,
        key: jax.Array,
        hidden_dim: int = 128,
        fuel_emb_dim: int = 4,
        eps_G: float = 0.1,
        max_G: float = 10.0,
        max_b_norm: float = 0.9,
        use_wind: bool = True,
    ):
        """Initialise networks, embeddings, and placeholder scene data.

        Args:
            manifold:     TriangularMesh that defines the spatial domain.
            key:          JAX PRNG key; split internally for the two MLPs.
            hidden_dim:   MLP hidden-layer width. Default: 128.
            fuel_emb_dim: Fuel-type embedding dimension. Default: 4.
            eps_G:        Minimum eigenvalue of G. Default: 0.1.
            max_G:        Maximum eigenvalue of G. Default: 10.0.
            max_b_norm:   Upper bound on ``||b||_{G^{-1}}``. Default: 0.9.
            use_wind:     Include Randers drift. Default: True.
        """
        super().__init__(manifold)
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
        self.fuel_embedding = jnp.zeros((13, fuel_emb_dim), dtype=DEFAULT_JNP_DTYPE)

        # Placeholder scene data — replaced by bind_scene before use
        self.face_covariates = jnp.zeros((1, 5 + fuel_emb_dim), dtype=DEFAULT_JNP_DTYPE)
        self.weather_vec = jnp.zeros((4,), dtype=DEFAULT_JNP_DTYPE)

    def bind_scene(
        self,
        face_cov_array: jax.Array,
        weather_vec_array: jax.Array,
    ) -> "CovariateMeshRanders":
        """Return a new instance with per-face covariates and weather baked in.

        Args:
            face_cov_array:   (F, 5+fuel_emb_dim) precomputed per-face
                covariate vectors.
            weather_vec_array: (4,) global weather vector.

        Returns:
            New :class:`CovariateMeshRanders` with scene data attached.
        """
        return eqx.tree_at(
            lambda m: (m.face_covariates, m.weather_vec),
            self,
            (
                jnp.asarray(face_cov_array, dtype=DEFAULT_JNP_DTYPE),
                jnp.asarray(weather_vec_array, dtype=DEFAULT_JNP_DTYPE),
            ),
        )

    def _get_face_params(self, face_idx: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute the Randers parameters (G_f, b_f) for a single face.

        Args:
            face_idx: Scalar integer face index.

        Returns:
            Tuple ``(G_f, b_f)`` where ``G_f`` is (2, 2) SPD and ``b_f``
            is (2,).
        """
        local_feat = jax.lax.stop_gradient(self.face_covariates)[
            face_idx
        ]  # (5+fuel_emb_dim,)
        raw_global = self.global_mlp(jax.lax.stop_gradient(self.weather_vec))
        raw_local = self.local_mlp(local_feat)
        raw = raw_global + raw_local

        G_raw = jnp.array([[raw[0], raw[1]], [raw[1], raw[2]]])
        G = project_spd(G_raw, self.eps_G, self.max_G)

        if self.use_wind:
            b = project_b_norm(raw[3:5], G, self.max_b_norm)
        else:
            b = jnp.zeros(2, dtype=raw.dtype)
        return G, b

    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Randers-Zermelo cost F(x, v) on a 3D triangular mesh face.

        Steps:

        1. Identify the nearest face via softmax face weights (argmax).
        2. Retrieve ``(G_f, b_f)`` from the covariate-conditioned MLPs.
        3. Build a face-local orthonormal frame ``{u1, u2}`` via Gram-Schmidt
           on the face edge vectors.
        4. Project ``v`` onto the tangent plane to get ``v_tan``.
        5. Apply the Zermelo navigation formula.

        .. note::

            Gradient w.r.t. ``x`` does not propagate through face selection
            (``jnp.argmax`` is non-differentiable).  For differentiable face
            blending, replace argmax with softmax-weighted interpolation of
            per-face parameters (Phase W2b).

        .. note::

            Phase W2 approximation: ``G_f`` is applied as an isotropic
            scalar ``g_iso = 0.5*(G_f[0,0]+G_f[1,1])`` in 3D.  The drift
            ``b_f`` is lifted to 3D via the face frame.  Full anisotropic
            pullback is Phase W2b future work.

        Args:
            x: (3,) position on (or near) the mesh surface.
            v: (3,) tangent velocity vector.

        Returns:
            Scalar Finsler cost F(x, v) ≥ 0. Shape: ``()``.

        Reference:
            spec/MATH_SPEC.md § 5 (Zermelo navigation formula).
        """
        v_sq_raw = jnp.sum(v**2)
        is_zero = v_sq_raw < GRAD_EPS
        v_safe = jnp.where(is_zero, v + jnp.sqrt(GRAD_EPS), v)

        # Identify nearest face
        weights = self.manifold.get_face_weights(x)  # (F,)
        face_idx = jnp.argmax(weights)

        # Face geometry
        tri = self.manifold.triangles[face_idx]  # (3, 3)
        v0_tri, v1_tri, v2_tri = tri[0], tri[1], tri[2]

        # Orthonormal face frame via Gram-Schmidt
        e1_raw = v1_tri - v0_tri
        e2_raw = v2_tri - v0_tri
        norm_e1 = safe_norm(e1_raw)
        u1 = e1_raw / norm_e1

        e2_orth = e2_raw - jnp.dot(e2_raw, u1) * u1
        norm_e2 = safe_norm(e2_orth)
        u2 = e2_orth / norm_e2

        # Face normal and tangent projection
        normal = jnp.cross(u1, u2)
        normal = normal / safe_norm(normal)
        v_tan = v_safe - jnp.dot(v_safe, normal) * normal

        # Randers parameters for this face
        G_f, b_f = self._get_face_params(face_idx)

        # Phase W2 isotropic approximation: scalar speed in 3D
        g_iso = 0.5 * (G_f[0, 0] + G_f[1, 1])

        # Lift b to 3D via face frame
        b_3d = b_f[0] * u1 + b_f[1] * u2

        # Zermelo navigation formula in 3D with isotropic G = g_iso * I
        # b_Ginv_b = ||b_3d||^2 / g_iso
        b_norm_sq = jnp.sum(b_3d**2)
        b_Ginv_b = b_norm_sq / jnp.maximum(g_iso, 1e-8)
        lam = jnp.maximum(1.0 - b_Ginv_b, 1e-6)

        v_sq_h = g_iso * jnp.sum(v_tan**2)
        bdotv = jnp.dot(b_3d, v_tan)

        disc = lam * v_sq_h + bdotv**2
        cost = (jnp.sqrt(jnp.maximum(disc, GRAD_EPS)) - bdotv) / lam
        return jnp.where(is_zero, 0.0, cost)

    def zermelo_data(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return Zermelo triple ``(H_3d, W_3d, lambda)`` at position x.

        Implements the :class:`~ham.geometry.metric.AsymmetricMetric` interface.
        Returns the Phase-W2 isotropic approximation: H = g_iso * I_3,
        W = b lifted to 3D via the face frame.

        Args:
            x: (3,) position on (or near) the mesh surface.

        Returns:
            H_3d:   (3, 3) isotropic Zermelo sea metric ``g_iso * I_3`` (the
                    same tensor ``metric_fn`` uses under ``sqrt(v^T G v)``).
            W_3d:   (3,) Zermelo wind *vector* ``W = H^{-1} b = b / g_iso``,
                    lifted to the face tangent plane.
            lambda: Causality scalar ``1 - ||W||^2_H = 1 - ||b||^2 / g_iso``.

        Note:
            The wind returned is the Zermelo wind *vector* ``b / g_iso``, not
            the drift one-form ``b`` itself.  The eikonal/zoo reconstruction
            ``F = (sqrt(lam v^T H v + (v^T H W)^2) - v^T H W)/lam`` reproduces
            ``metric_fn`` exactly only with ``W = H^{-1} b``; returning the raw
            ``b`` made the eikonal-solved metric disagree with ``metric_fn``
            (review finding MATH-ZD1).
        """
        weights = self.manifold.get_face_weights(x)
        face_idx = jnp.argmax(weights)

        tri = self.manifold.triangles[face_idx]
        v0_tri, v1_tri, v2_tri = tri[0], tri[1], tri[2]
        e1_raw = v1_tri - v0_tri
        e2_raw = v2_tri - v0_tri
        norm_e1 = safe_norm(e1_raw)
        u1 = e1_raw / norm_e1
        e2_orth = e2_raw - jnp.dot(e2_raw, u1) * u1
        norm_e2 = safe_norm(e2_orth)
        u2 = e2_orth / norm_e2

        G_f, b_f = self._get_face_params(face_idx)
        g_iso = jnp.maximum(0.5 * (G_f[0, 0] + G_f[1, 1]), 1e-8)
        b_3d = b_f[0] * u1 + b_f[1] * u2

        H_3d = g_iso * jnp.eye(3, dtype=G_f.dtype)
        # Zermelo wind vector W = H^{-1} b = b / g_iso (sea is isotropic g_iso*I).
        W_3d = b_3d / g_iso
        b_norm_sq = jnp.sum(b_3d**2)
        b_Ginv_b = b_norm_sq / g_iso
        lam = jnp.maximum(1.0 - b_Ginv_b, 1e-6)
        return H_3d, W_3d, lam
