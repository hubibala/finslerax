"""Tests for CovariateConditionedRanders wildfire Finsler metric.

Covers:
  - project_spd: eigenvalue clamping correctness
  - project_b_norm: G^{-1}-norm bound after projection
  - LocalTerrainCNN: output shape, dtype, differentiability
  - precompute_metric_field: field shape, grad flow through CNN weights
  - metric_fn: positivity, 1-homogeneity, Riemannian limit, JIT/vmap, gradients
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# config.update("jax_enable_x64", True)
from ham.geometry.manifolds import EuclideanSpace
from ham.models.wildfire import (
    CovariateConditionedRanders,
    LocalTerrainCNN,
    project_b_norm,
    project_spd,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_H, _W = 10, 10
_PIXEL_SPACING = 30.0
_ORIGIN = jnp.zeros(2)
_KEY = jax.random.PRNGKey(0)


def _make_scene(key=None):
    """Return a dict of minimal 10×10 scene arrays."""
    rng = jax.random.PRNGKey(42) if key is None else key
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    return dict(
        elev=jax.random.uniform(k1, (_H, _W)) * 500.0,
        slope=jax.random.uniform(k2, (_H, _W)) * 0.5,
        aspect=jax.random.uniform(k3, (_H, _W)) * 2.0 * jnp.pi,
        canopy=jax.random.uniform(k4, (_H, _W)),
        fuel_codes=jnp.ones((_H, _W), dtype=jnp.int32) * 3,
        weather_vec=jnp.array([20.0, 0.4, 0.5, 0.866]),
        pixel_spacing_m=_PIXEL_SPACING,
        origin_xy=_ORIGIN,
    )


def _make_model(use_wind=True, key=None):
    """Return a freshly initialised CovariateConditionedRanders.

    Uses small ``cnn_channels=8`` so each test runs in seconds rather than
    waiting for 64-channel conv layers.
    """
    k = _KEY if key is None else key
    manifold = EuclideanSpace(2)
    return CovariateConditionedRanders(
        manifold, k, hidden_dim=16, fuel_emb_dim=4, cnn_channels=8,
        eps_G=0.1, max_G=10.0, max_b_norm=0.9, use_wind=use_wind,
    )


def _bound_model(use_wind=True):
    """Return a model with scene data attached and metric_field precomputed."""
    model = _make_model(use_wind=use_wind)
    scene = _make_scene()
    bound = model.bind_scene(**scene)
    return bound.precompute_metric_field()


# ---------------------------------------------------------------------------
# project_spd
# ---------------------------------------------------------------------------

class TestProjectSPD:
    """project_spd: output eigenvalues must lie in [eps_min, eps_max]."""

    def _eigs(self, mat):
        """Return eigenvalues of a 2×2 symmetric matrix (numpy)."""
        return np.linalg.eigvalsh(np.array(mat))

    @pytest.mark.parametrize("eps_min,eps_max", [(0.1, 10.0), (1e-3, 5.0)])
    def test_eigenvalues_in_range(self, eps_min, eps_max):
        key = jax.random.PRNGKey(7)
        # Generate 20 random symmetric matrices
        raw = jax.random.normal(key, (20, 2, 2))
        mats = 0.5 * (raw + raw.transpose(0, 2, 1))  # symmetrize

        for mat in mats:
            out = project_spd(mat, eps_min, eps_max)
            eigs = self._eigs(out)
            assert np.all(eigs >= eps_min - 1e-5), f"min eigenvalue {eigs.min()} < {eps_min}"
            assert np.all(eigs <= eps_max + 1e-5), f"max eigenvalue {eigs.max()} > {eps_max}"

    def test_already_spd_unchanged(self):
        """Identity-like matrix should be unchanged up to the discriminant epsilon (~1e-8)."""
        mat = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        out = project_spd(mat, 0.1, 10.0)
        np.testing.assert_allclose(np.array(out), np.array(mat), atol=1e-5)

    def test_vmap_compatible(self):
        """project_spd must run under vmap without errors."""
        key = jax.random.PRNGKey(13)
        raw = jax.random.normal(key, (8, 2, 2))
        mats = 0.5 * (raw + raw.transpose(0, 2, 1))
        batched = jax.vmap(lambda m: project_spd(m, 0.1, 10.0))(mats)
        assert batched.shape == (8, 2, 2)

    def test_grad_compatible(self):
        """project_spd must be differentiable (no NaN grad)."""
        mat = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        grad = jax.grad(lambda m: jnp.sum(project_spd(m, 0.1, 10.0)))(mat)
        assert not jnp.any(jnp.isnan(grad))


# ---------------------------------------------------------------------------
# project_b_norm
# ---------------------------------------------------------------------------

class TestProjectBNorm:
    """project_b_norm: result must satisfy the G^{-1}-norm constraint."""

    def _ginv_norm(self, b, G):
        """||b||_{G^{-1}} via explicit inversion (numpy)."""
        G_np = np.array(G)
        b_np = np.array(b)
        Ginv = np.linalg.inv(G_np)
        return float(np.sqrt(b_np @ Ginv @ b_np))

    @pytest.mark.parametrize("max_norm", [0.9, 0.5, 0.99])
    def test_norm_within_bound(self, max_norm):
        key = jax.random.PRNGKey(3)
        G = project_spd(jnp.array([[2.0, 0.3], [0.3, 1.5]]), 0.1, 10.0)
        for _ in range(10):
            key, k = jax.random.split(key)
            b_raw = jax.random.normal(k, (2,)) * 3.0
            b = project_b_norm(b_raw, G, max_norm)
            norm = self._ginv_norm(b, G)
            assert norm < max_norm + 1e-5, f"G^{{-1}}-norm {norm} exceeds max {max_norm}"

    def test_already_within_bound_unchanged(self):
        """A drift vector already below max_norm should not be scaled down."""
        G = jnp.eye(2)
        b_small = jnp.array([0.1, 0.1])  # ||b|| ≈ 0.141 < 0.9
        b_out = project_b_norm(b_small, G, 0.9)
        np.testing.assert_allclose(np.array(b_out), np.array(b_small), atol=1e-9)

    def test_grad_compatible(self):
        """project_b_norm must be differentiable (no NaN grad)."""
        G = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        b = jnp.array([1.0, -1.0])
        grad = jax.grad(lambda bv: jnp.sum(project_b_norm(bv, G, 0.9)))(b)
        assert not jnp.any(jnp.isnan(grad))


# ---------------------------------------------------------------------------
# Shared scene fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bound_model():
    return _bound_model(use_wind=True)


@pytest.fixture(scope="module")
def riemannian_model():
    return _bound_model(use_wind=False)


# ---------------------------------------------------------------------------
# metric_fn: positivity
# ---------------------------------------------------------------------------

def test_metric_fn_positive(bound_model):
    """F(x, v) > 0 for all nonzero v."""
    key = jax.random.PRNGKey(1)
    x = jnp.array([150.0, 150.0])  # centre of 10×10 @ 30m

    vs = jax.random.normal(key, (20, 2))
    vs = vs / jnp.linalg.norm(vs, axis=-1, keepdims=True)  # unit vectors

    for v in vs:
        cost = bound_model.metric_fn(x, v)
        assert float(cost) > 0.0, f"Expected F > 0, got {float(cost)}"


# ---------------------------------------------------------------------------
# metric_fn: 1-homogeneity
# ---------------------------------------------------------------------------

def test_metric_fn_homogeneous(bound_model):
    """F(x, λv) = λ F(x, v) for λ > 0."""
    x = jnp.array([150.0, 150.0])
    v = jnp.array([1.0, 0.5])

    for lam in [0.5, 1.0, 2.0, 5.0]:
        f_v = bound_model.metric_fn(x, v)
        f_lv = bound_model.metric_fn(x, lam * v)
        np.testing.assert_allclose(
            float(f_lv), lam * float(f_v),
            rtol=1e-6,
            err_msg=f"Homogeneity violated at λ={lam}",
        )


# ---------------------------------------------------------------------------
# metric_fn: Riemannian limit (use_wind=False)
# ---------------------------------------------------------------------------

def test_metric_fn_riemannian_limit(riemannian_model):
    """With use_wind=False the metric is symmetric: F(x,v) = F(x,-v)."""
    x = jnp.array([150.0, 150.0])
    key = jax.random.PRNGKey(99)
    vs = jax.random.normal(key, (10, 2))

    for v in vs:
        f_pos = riemannian_model.metric_fn(x, v)
        f_neg = riemannian_model.metric_fn(x, -v)
        np.testing.assert_allclose(
            float(f_pos), float(f_neg), rtol=1e-6,
            err_msg="Riemannian limit should be symmetric in v",
        )


def test_metric_fn_riemannian_matches_sqrt_Gv(riemannian_model):
    """With use_wind=False, F(x,v)^2 == v^T G v (up to rounding)."""
    x = jnp.array([90.0, 90.0])
    v = jnp.array([1.0, 0.0])

    # Extract G directly via _get_params
    G, b = riemannian_model._get_params(x)
    expected = jnp.sqrt(jnp.dot(v, jnp.dot(G, v)))
    actual = riemannian_model.metric_fn(x, v)
    np.testing.assert_allclose(float(actual), float(expected), rtol=1e-5)


# ---------------------------------------------------------------------------
# bind_scene: JIT and vmap
# ---------------------------------------------------------------------------

def test_bind_scene_jit(bound_model):
    """metric_fn is jit-compilable and vmappable after bind_scene."""
    import equinox as eqx

    x = jnp.array([150.0, 150.0])
    v = jnp.array([1.0, 0.5])

    # Use eqx.filter_jit — the standard equinox pattern for module methods
    f_jit = eqx.filter_jit(lambda xi, vi: bound_model.metric_fn(xi, vi))
    result = f_jit(x, v)
    assert jnp.isfinite(result), "JIT result should be finite"
    assert float(result) > 0.0

    # vmap over a batch of positions and directions
    xs = jnp.stack([x, x + 30.0, x - 30.0])
    vs = jnp.stack([v, v * 2.0, v * 0.5])
    f_vmap = jax.vmap(lambda xi, vi: bound_model.metric_fn(xi, vi))
    results = f_vmap(xs, vs)
    assert results.shape == (3,)
    assert jnp.all(jnp.isfinite(results))
    assert jnp.all(results > 0.0)


# ---------------------------------------------------------------------------
# Gradients: no NaN
# ---------------------------------------------------------------------------

def test_gradients_wrt_v(bound_model):
    """jax.grad of metric_fn w.r.t. v must not produce NaN."""
    x = jnp.array([150.0, 150.0])
    v = jnp.array([1.0, 0.5])

    grad_v = jax.grad(bound_model.metric_fn, argnums=1)(x, v)
    assert not jnp.any(jnp.isnan(grad_v)), f"NaN gradient w.r.t. v: {grad_v}"


def test_gradients_wrt_x(bound_model):
    """jax.grad of metric_fn w.r.t. x must not produce NaN."""
    x = jnp.array([150.0, 150.0])
    v = jnp.array([1.0, 0.5])

    grad_x = jax.grad(bound_model.metric_fn, argnums=0)(x, v)
    assert not jnp.any(jnp.isnan(grad_x)), f"NaN gradient w.r.t. x: {grad_x}"


def test_gradients_near_zero_v(bound_model):
    """grad w.r.t. v must be finite even for near-zero v."""
    x = jnp.array([150.0, 150.0])
    v_tiny = jnp.array([1e-7, 1e-7])

    grad_v = jax.grad(bound_model.metric_fn, argnums=1)(x, v_tiny)
    assert jnp.all(jnp.isfinite(grad_v)), f"Non-finite gradient near v=0: {grad_v}"


def test_fuel_embedding_gradient():
    """Gradient of metric_fn w.r.t. fuel_embedding must be finite (no NaN/Inf).

    The fuel embedding flows through precompute_metric_field → metric_field →
    bilinear interpolation → metric_fn.  We include the recomputation in the
    loss so that gradients actually reach fuel_embedding.
    """
    import equinox as eqx

    # Start from a bound model with metric_field precomputed
    model = _bound_model(use_wind=True)
    x = jnp.array([150.0, 150.0])
    v = jnp.array([1.0, 0.5])

    def loss(emb):
        # Update embedding then recompute the metric field so gradients flow
        m2 = eqx.tree_at(lambda m: m.fuel_embedding, model, emb)
        m2 = m2.precompute_metric_field()
        return m2.metric_fn(x, v)

    grad_emb = jax.grad(loss)(model.fuel_embedding)
    assert not jnp.any(jnp.isnan(grad_emb)), "NaN in fuel_embedding gradient"
    assert not jnp.any(jnp.isinf(grad_emb)), "Inf in fuel_embedding gradient"
    assert grad_emb.shape == model.fuel_embedding.shape


# ---------------------------------------------------------------------------
# LocalTerrainCNN and precompute_metric_field
# ---------------------------------------------------------------------------

class TestLocalTerrainCNN:
    """Basic shape, dtype, and differentiability tests for LocalTerrainCNN."""

    def _make_cnn(self):
        return LocalTerrainCNN(fuel_emb_dim=4, n_channels=8, key=jax.random.PRNGKey(0))

    def test_output_shape(self):
        cnn = self._make_cnn()
        raster = jnp.ones((5, 10, 10))
        fuel = jnp.ones((4, 10, 10))
        weather = jnp.zeros(4)
        out = cnn(raster, fuel, weather)
        assert out.shape == (10, 10, 5), f"Expected (10,10,5), got {out.shape}"

    def test_output_dtype(self):
        cnn = self._make_cnn()
        raster = jnp.ones((5, 10, 10))
        fuel = jnp.ones((4, 10, 10))
        weather = jnp.zeros(4)
        out = cnn(raster, fuel, weather)
        assert out.dtype == jnp.float32, f"Expected float32, got {out.dtype}"

    def test_output_finite(self):
        cnn = self._make_cnn()
        raster = jnp.ones((5, 10, 10))
        fuel = jnp.zeros((4, 10, 10))
        weather = jnp.zeros(4)
        out = cnn(raster, fuel, weather)
        assert jnp.all(jnp.isfinite(out)), "CNN output contains non-finite values"

    def test_grad_through_weights(self):
        cnn = self._make_cnn()
        raster = jnp.ones((5, 10, 10))
        fuel = jnp.ones((4, 10, 10))
        weather = jnp.array([20.0, 0.4, 0.5, 0.866])

        import equinox as eqx
        def loss_fn(c):
            return jnp.sum(c(raster, fuel, weather))

        grads = eqx.filter_grad(loss_fn)(cnn)
        # At least conv1 weight should have a nonzero gradient
        assert not jnp.any(jnp.isnan(grads.conv1.weight)), "NaN grad in conv1"
        assert not jnp.all(grads.conv1.weight == 0), "All-zero grad in conv1 (unexpected)"


class TestPrecomputeMetricField:
    """Tests for precompute_metric_field()."""

    def test_field_shape(self):
        model = _make_model()
        scene = _make_scene()
        bound = model.bind_scene(**scene)
        precomputed = bound.precompute_metric_field()
        assert precomputed.metric_field is not None
        assert precomputed.metric_field.shape == (_H, _W, 5), (
            f"Expected ({_H},{_W},5), got {precomputed.metric_field.shape}"
        )

    def test_field_dtype(self):
        model = _make_model()
        scene = _make_scene()
        bound = model.bind_scene(**scene).precompute_metric_field()
        assert bound.metric_field.dtype == jnp.float32

    def test_field_finite(self):
        model = _make_model()
        scene = _make_scene()
        bound = model.bind_scene(**scene).precompute_metric_field()
        assert jnp.all(jnp.isfinite(bound.metric_field)), "metric_field has non-finite values"

    def test_grad_flows_to_cnn_weights(self):
        """Gradient of metric_fn w.r.t. CNN weights must be nonzero."""
        import equinox as eqx

        x = jnp.array([150.0, 150.0])
        v = jnp.array([1.0, 0.5])
        scene = _make_scene()

        def loss_fn(m):
            bound = m.bind_scene(**scene).precompute_metric_field()
            return bound.metric_fn(x, v)

        model = _make_model()
        grads = eqx.filter_grad(loss_fn)(model)

        # CNN weights should receive gradients
        assert not jnp.any(jnp.isnan(grads.local_cnn.conv1.weight)), "NaN grad in local_cnn.conv1"
        # At random init the gradient should be nonzero
        assert not jnp.all(grads.local_cnn.conv1.weight == 0), "Zero grad in local_cnn.conv1"

    def test_bind_scene_resets_field(self):
        """bind_scene must set metric_field to None (stale field invalidated)."""
        model = _make_model()
        scene = _make_scene()
        bound = model.bind_scene(**scene).precompute_metric_field()
        assert bound.metric_field is not None

        # Re-binding should invalidate the field
        rebound = bound.bind_scene(**scene)
        assert rebound.metric_field is None


class TestRasterStopGradient:
    """Regression test: rasters must not receive gradient updates via the CNN.

    When the model is terrain-bound and eqx.filter_value_and_grad is called,
    gradients for float64 raster leaves would be computed and applied by
    eqx.apply_updates unless stop_gradient guards are in place inside
    precompute_metric_field.  This test verifies those guards are active.
    """

    def test_raster_grad_is_zero(self):
        """d(loss)/d(elev_raster) must be zero after precompute_metric_field."""
        import equinox as eqx

        model = _make_model()
        scene = _make_scene()

        def loss_fn(m):
            bound = m.bind_scene(**scene).precompute_metric_field()
            x = jnp.array([150.0, 150.0])
            v = jnp.array([1.0, 0.5])
            return bound.metric_fn(x, v)

        # Differentiating w.r.t. the unbound model: rasters are NOT leaves here.
        # Bind them explicitly to mimic the terrain-bound training scenario.
        terrain_bound = model.bind_scene(**scene)

        def loss_terrain(m):
            m2 = m.precompute_metric_field()
            x = jnp.array([150.0, 150.0])
            v = jnp.array([1.0, 0.5])
            return m2.metric_fn(x, v)

        grads = eqx.filter_grad(loss_terrain)(terrain_bound)

        # All raster gradients must be exactly zero (stop_gradient enforced)
        for attr in ("elev_raster", "slope_raster", "aspect_raster", "canopy_raster"):
            g = getattr(grads, attr)
            if g is not None:
                assert jnp.all(g == 0), (
                    f"{attr} received non-zero gradient: max|g|={float(jnp.max(jnp.abs(g)))}"
                )
