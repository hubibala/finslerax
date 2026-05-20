"""Tests for ham.utils.terrain."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ham.utils.terrain import (
    dem_to_mesh,
    interpolate_covariates_to_vertices,
    pixel_to_world_3d,
    compute_face_normals,
    compute_face_slopes_aspects,
    CovariateMeshRanders,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _flat_dem(H=5, W=5, elev=0.0):
    """Return a flat (H, W) DEM at constant elevation."""
    return jnp.full((H, W), elev, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# pixel_to_world_3d
# ---------------------------------------------------------------------------

def test_pixel_to_world_3d():
    """Basic coordinate conversion at known pixel positions."""
    result = pixel_to_world_3d(2, 3, 50.0, pixel_spacing_m=30.0)
    # x = 3*30=90, y = 2*30=60, z=50
    np.testing.assert_allclose(result, jnp.array([90.0, 60.0, 50.0]), atol=1e-5)


def test_pixel_to_world_3d_origin():
    """Pixel (0,0) maps to (0,0,elev)."""
    result = pixel_to_world_3d(0, 0, 123.0)
    np.testing.assert_allclose(result, jnp.array([0.0, 0.0, 123.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# dem_to_mesh
# ---------------------------------------------------------------------------

def test_dem_to_mesh_shape():
    """5×5 DEM → 25 vertices, 32 faces."""
    dem = _flat_dem(H=5, W=5)
    mesh = dem_to_mesh(dem)
    assert mesh.vertices.shape == (25, 3), mesh.vertices.shape
    assert mesh.faces.shape == (32, 3), mesh.faces.shape


def test_dem_to_mesh_vertex_coords():
    """Corner vertices should have the correct (x, y, z) world positions."""
    H, W = 5, 5
    spacing = 30.0
    elev = jnp.arange(H * W, dtype=jnp.float32).reshape(H, W)
    mesh = dem_to_mesh(elev, pixel_spacing_m=spacing)

    # Vertex k = i*W + j
    def expected_vertex(i, j):
        return jnp.array([j * spacing, i * spacing, elev[i, j]])

    # top-left (0,0)
    np.testing.assert_allclose(mesh.vertices[0], expected_vertex(0, 0), rtol=1e-5)
    # top-right (0, W-1)
    np.testing.assert_allclose(mesh.vertices[W - 1], expected_vertex(0, W - 1), rtol=1e-5)
    # bottom-left (H-1, 0)
    np.testing.assert_allclose(mesh.vertices[(H - 1) * W], expected_vertex(H - 1, 0), rtol=1e-5)
    # bottom-right (H-1, W-1)
    np.testing.assert_allclose(mesh.vertices[H * W - 1], expected_vertex(H - 1, W - 1), rtol=1e-5)


def test_dem_to_mesh_face_dtype():
    """Face indices must be int32."""
    dem = _flat_dem()
    mesh = dem_to_mesh(dem)
    assert mesh.faces.dtype == jnp.int32


# ---------------------------------------------------------------------------
# compute_face_normals
# ---------------------------------------------------------------------------

def test_face_normals_flat():
    """A perfectly flat DEM → all face normals should point straight up (0,0,1)."""
    dem = _flat_dem(H=4, W=4, elev=0.0)
    mesh = dem_to_mesh(dem)
    normals = compute_face_normals(mesh)
    # All normals should be (0, 0, 1) or (0, 0, -1); cross product orientation
    # depends on winding. Check |n_z| ≈ 1.
    np.testing.assert_allclose(jnp.abs(normals[:, 2]), jnp.ones(normals.shape[0]), atol=1e-5)
    np.testing.assert_allclose(normals[:, 0], jnp.zeros(normals.shape[0]), atol=1e-5)
    np.testing.assert_allclose(normals[:, 1], jnp.zeros(normals.shape[0]), atol=1e-5)


def test_face_normals_unit_length():
    """Normals should be unit vectors."""
    dem = jnp.array(
        [[0.0, 0.0, 0.0],
         [0.0, 5.0, 0.0],
         [0.0, 0.0, 10.0]], dtype=jnp.float32
    )
    mesh = dem_to_mesh(dem, pixel_spacing_m=10.0)
    normals = compute_face_normals(mesh)
    norms = jnp.sqrt(jnp.sum(normals ** 2, axis=-1))
    np.testing.assert_allclose(norms, jnp.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# compute_face_slopes_aspects
# ---------------------------------------------------------------------------

def test_face_slopes_flat():
    """Flat DEM → slope = 0 for all faces."""
    dem = _flat_dem(H=4, W=4, elev=0.0)
    mesh = dem_to_mesh(dem)
    slopes, _ = compute_face_slopes_aspects(mesh)
    np.testing.assert_allclose(slopes, jnp.zeros_like(slopes), atol=1e-5)


# ---------------------------------------------------------------------------
# interpolate_covariates_to_vertices
# ---------------------------------------------------------------------------

def test_interpolate_covariates():
    """Covariate values should be correctly mapped to vertex indices."""
    H, W = 3, 4
    dem = _flat_dem(H=H, W=W)
    mesh = dem_to_mesh(dem)

    # Raster where pixel (i,j) = i*W + j (identical to vertex index)
    raster = jnp.arange(H * W, dtype=jnp.float32).reshape(H, W)
    raster_dict = {"r": raster}
    cov = interpolate_covariates_to_vertices(mesh, raster_dict, H, W)

    assert cov.shape == (H * W, 1)
    # vertex k should have value k
    expected = jnp.arange(H * W, dtype=jnp.float32)[:, None]
    np.testing.assert_allclose(cov, expected, atol=1e-5)


def test_interpolate_covariates_multi_channel():
    """Multiple rasters produce correct channel stacking."""
    H, W = 3, 3
    dem = _flat_dem(H=H, W=W)
    mesh = dem_to_mesh(dem)

    r1 = jnp.ones((H, W), dtype=jnp.float32)
    r2 = 2.0 * jnp.ones((H, W), dtype=jnp.float32)
    cov = interpolate_covariates_to_vertices(mesh, {"a": r1, "b": r2}, H, W)
    assert cov.shape == (H * W, 2)
    np.testing.assert_allclose(cov[:, 0], jnp.ones(H * W), atol=1e-5)
    np.testing.assert_allclose(cov[:, 1], 2.0 * jnp.ones(H * W), atol=1e-5)


# ---------------------------------------------------------------------------
# CovariateMeshRanders
# ---------------------------------------------------------------------------

def _make_metric(use_wind=True, H=4, W=4):
    """Build a CovariateMeshRanders on a small flat mesh with random scene."""
    dem = _flat_dem(H=H, W=W, elev=0.0)
    mesh = dem_to_mesh(dem, pixel_spacing_m=10.0)
    key = jax.random.PRNGKey(42)
    metric = CovariateMeshRanders(mesh, key, hidden_dim=16, fuel_emb_dim=4, use_wind=use_wind)

    F = mesh.faces.shape[0]
    feat_dim = 5 + metric.fuel_emb_dim
    face_cov = jnp.zeros((F, feat_dim), dtype=jnp.float32)
    weather = jnp.zeros((4,), dtype=jnp.float32)
    metric = metric.bind_scene(face_cov, weather)
    return metric, mesh


def test_covariate_mesh_randers_positive():
    """metric_fn should return a positive value for nonzero velocity."""
    metric, mesh = _make_metric()
    # Query point near centre of mesh
    x = jnp.array([15.0, 15.0, 0.0], dtype=jnp.float32)
    v = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    cost = metric.metric_fn(x, v)
    assert float(cost) > 0.0, f"Expected positive cost, got {cost}"


def test_covariate_mesh_randers_zero_velocity():
    """metric_fn should return 0 for the zero velocity."""
    metric, mesh = _make_metric()
    x = jnp.array([15.0, 15.0, 0.0], dtype=jnp.float32)
    v = jnp.zeros(3, dtype=jnp.float32)
    cost = metric.metric_fn(x, v)
    assert float(cost) == 0.0, f"Expected 0 cost for zero velocity, got {cost}"


def test_covariate_mesh_randers_homogeneous():
    """F(x, lambda*v) = lambda * F(x, v) for lambda > 0."""
    metric, mesh = _make_metric()
    x = jnp.array([15.0, 15.0, 0.0], dtype=jnp.float32)
    v = jnp.array([1.0, 1.0, 0.0], dtype=jnp.float32)
    lam = 3.7

    f1 = metric.metric_fn(x, v)
    f_lam = metric.metric_fn(x, lam * v)

    np.testing.assert_allclose(float(f_lam), lam * float(f1), rtol=1e-4,
                                err_msg=f"Homogeneity failed: F(x,λv)={f_lam} vs λF(x,v)={lam*f1}")


def test_covariate_mesh_randers_no_wind():
    """With use_wind=False the metric should still return a positive cost."""
    metric, mesh = _make_metric(use_wind=False)
    x = jnp.array([15.0, 15.0, 0.0], dtype=jnp.float32)
    v = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    cost = metric.metric_fn(x, v)
    assert float(cost) > 0.0


def test_covariate_mesh_randers_bind_scene_immutable():
    """bind_scene should return a new instance, not mutate the original."""
    dem = _flat_dem(H=3, W=3)
    mesh = dem_to_mesh(dem)
    key = jax.random.PRNGKey(0)
    metric = CovariateMeshRanders(mesh, key, hidden_dim=8, fuel_emb_dim=4)

    F = mesh.faces.shape[0]
    feat_dim = 5 + metric.fuel_emb_dim
    face_cov = jnp.ones((F, feat_dim), dtype=jnp.float32)
    weather = jnp.ones((4,), dtype=jnp.float32)

    metric2 = metric.bind_scene(face_cov, weather)

    # Original should still have zeros
    np.testing.assert_allclose(metric.weather_vec, jnp.zeros(4), atol=1e-5)
    # New instance should have the updated values
    np.testing.assert_allclose(metric2.weather_vec, jnp.ones(4), atol=1e-5)
