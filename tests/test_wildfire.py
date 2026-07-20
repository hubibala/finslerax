"""Fast unit gates for experiments/wildfire (the W0/W1/W4 ladder in miniature).

The heavy science gates live in the stage scripts
(``experiments.wildfire.run_stage_a_bridge`` etc.); these tests pin the
conventions that were empirically hard-won:

* solver frame is pixel (row, col); arrival hours; wind velocity downwind-cheaper;
* IoU@50 is absolute (thresholds at 0.5 x GT duration — a scale error hurts);
* the affine gauge: alpha=0 loss is scale-invariant, alpha=1 is not;
* gradients through the eikonal solver match finite differences;
* recalibrate() recovers an injected (s, c) exactly;
* encoder orientation: coupled wind along +x speeds up +col spread;
* ingest fuel-code remap and wind sign conventions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.wildfire.ingest import wind_model_coords
from experiments.wildfire.medium import (
    GridZermelo,
    dense_arrival_loss,
    direction_coverage,
    fire_metrics,
    front_normals,
    iou_at_50_hours,
    odd_coverage,
    solve_arrival_encoder,
    solve_arrival_grid,
)
from experiments.wildfire.recover import recalibrate

SHAPE = (32, 32)
ONES = jnp.ones(SHAPE)
ZEROS = jnp.zeros(SHAPE)
ISO = GridZermelo(jnp.stack([ONES, ZEROS, ONES]), jnp.zeros((2, *SHAPE)))
SRC = jnp.array([16.0, 16.0])


def _euclid():
    I, J = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    return np.hypot(I - 16, J - 16)


class TestForward:
    def test_isotropic_matches_euclidean(self):
        T = np.asarray(solve_arrival_grid(ISO, SRC, SHAPE))
        D = _euclid()
        ring = D > 4
        err = np.abs(T - D)[ring] / D[ring]
        # 32^2 keeps more near-source pixels in the ring than the stage-A 48^2
        # gate; the <1% bar at scale lives in run_stage_a_bridge (W0).
        assert err.mean() < 0.015

    def test_wind_downwind_cheaper(self):
        m = GridZermelo(jnp.stack([ONES, ZEROS, ONES]),
                        jnp.stack([ZEROS, 0.5 * ONES]))  # wind along +col
        T = np.asarray(solve_arrival_grid(m, SRC, SHAPE))
        assert T[16, 28] < T[16, 4]


class TestMetrics:
    def test_iou50_bills_the_scale(self):
        """A pure speed-scale error must hurt IoU@50 (absolute convention)."""
        T = np.asarray(solve_arrival_grid(ISO, SRC, SHAPE))
        burned = T <= np.quantile(T, 0.6)
        gt = np.where(burned, T, np.inf)
        assert iou_at_50_hours(T, gt, burned) == pytest.approx(1.0)
        assert iou_at_50_hours(2.0 * T, gt, burned) < 0.6

    def test_corr_is_affine_invariant(self):
        T = np.asarray(solve_arrival_grid(ISO, SRC, SHAPE))
        burned = T <= np.quantile(T, 0.6)
        gt = np.where(burned, T, np.inf)
        m = fire_metrics(3.0 * T + 1.0, gt, burned)
        assert m["corr"] == pytest.approx(1.0, abs=1e-5)


class TestDenseLoss:
    def _fields(self):
        T = solve_arrival_grid(ISO, SRC, SHAPE)
        burned = T <= jnp.quantile(T, 0.6)
        return T, jnp.where(burned, T, jnp.inf), burned

    def test_alpha0_scale_invariant_alpha1_not(self):
        T, T_obs, burned = self._fields()
        l0_id = float(dense_arrival_loss(T, T_obs, burned, 0.0))
        l0_sc = float(dense_arrival_loss(2.0 * T, T_obs, burned, 0.0))
        assert abs(l0_id - l0_sc) < 1e-5  # affine gauge discarded
        l1_id = float(dense_arrival_loss(T, T_obs, burned, 1.0))
        l1_sc = float(dense_arrival_loss(2.0 * T, T_obs, burned, 1.0))
        assert l1_sc > l1_id + 0.01  # absolute gauge billed

    def test_gradient_matches_fd(self):
        shape = (16, 16)
        ones, zeros = jnp.ones(shape), jnp.zeros(shape)
        src = jnp.array([8.0, 8.0])
        T_obs = solve_arrival_grid(
            GridZermelo(jnp.stack([1.3 * ones, zeros, 1.3 * ones]),
                        jnp.stack([zeros, 0.2 * ones])), src, shape)
        burned = jnp.ones(shape, dtype=bool)

        def loss(w):
            m = GridZermelo(jnp.stack([ones, zeros, ones]),
                            jnp.stack([zeros, w * ones]))
            return dense_arrival_loss(solve_arrival_grid(m, src, shape),
                                      T_obs, burned, 1.0)

        g_ad = float(jax.grad(loss)(jnp.asarray(0.1)))
        eps = 3e-3
        g_fd = (float(loss(jnp.asarray(0.1 + eps))) - float(loss(jnp.asarray(0.1 - eps)))) / (2 * eps)
        assert abs(g_ad - g_fd) / max(abs(g_fd), 1e-6) < 0.1


class TestRecalibrate:
    def test_injected_gauge_recovered(self):
        H_grid = jnp.stack([ONES, ZEROS, ONES])
        W_grid = jnp.stack([ZEROS, 0.4 * ONES])
        s0, c0 = 1.5, 0.5
        T_obs = s0 * np.asarray(
            solve_arrival_grid(GridZermelo(H_grid, c0 * W_grid), SRC, SHAPE))
        obs = T_obs <= np.quantile(T_obs, 0.3)

        def solve_c(c):
            return np.asarray(
                solve_arrival_grid(GridZermelo(H_grid, c * W_grid), SRC, SHAPE))

        r = recalibrate(solve_c, T_obs, obs)
        assert abs(r.s - s0) < 0.05 * s0
        assert abs(r.c - c0) < 0.1


class TestCoverage:
    def test_single_fire_has_no_coverage_two_crossed_do(self):
        T1 = np.asarray(solve_arrival_grid(ISO, jnp.array([16.0, 2.0]), SHAPE))
        T2 = np.asarray(solve_arrival_grid(ISO, jnp.array([2.0, 16.0]), SHAPE))
        b = np.ones(SHAPE, dtype=bool)
        n1 = front_normals(T1, b)
        n2 = front_normals(T2, b)
        cov1 = direction_coverage([n1])
        cov2 = direction_coverage([n1, n2])
        assert np.nanmedian(cov1) < 0.05  # one direction per pixel: rank 1
        # Fronts are orthogonal on the Thales circle over the two sources
        # (NOT midway between them, where they are antiparallel and the
        # structure tensor — blind to +-n — still sees rank 1).
        assert cov2[19, 9] > 0.4
        assert cov2[9, 9] < 0.2  # midpoint: antiparallel fronts, no coverage

    def test_block_pooling_shape(self):
        n = front_normals(np.asarray(solve_arrival_grid(ISO, SRC, SHAPE)),
                          np.ones(SHAPE, dtype=bool))
        cov = direction_coverage([n], block=8)
        assert cov.shape == (4, 4)

    def test_odd_coverage_needs_signed_diversity(self):
        """The odd ledger: +-pairs identify B; single-signed sets do not."""

        def const_fields(dirs):
            out = []
            for d in dirs:
                f = np.full((8, 8, 2), np.nan)
                f[..., 0], f[..., 1] = d
                out.append(f)
            return out

        def val(dirs):
            return odd_coverage(const_fields(dirs), block=8)[0, 0]

        # Single-signed orthogonal fronts: the even channel absorbs B.
        assert val([(1, 0), (0, 1)]) == pytest.approx(0.0, abs=1e-9)
        # +-x pairs only: B_y stays free -> smallest singular value 0.
        assert val([(1, 0), (-1, 0)]) == pytest.approx(0.0, abs=1e-9)
        # +-x and +-y: B fully identified (even with g12 unknown).
        assert val([(1, 0), (-1, 0), (0, 1), (0, -1)]) > 0.5
        # 4 generic directions: 4 equations < 5 unknowns -> unidentifiable.
        assert val([(1, 0), (0.6, 0.8), (-0.8, 0.6), (-0.6, -0.8)]) == pytest.approx(
            0.0, abs=1e-9)


class TestEncoderOrientation:
    def test_coupled_wind_speeds_up_plus_col(self):
        """End-to-end orientation: measured_wind +x (east/col) must make the
        encoder's arrival field asymmetric toward larger col indices."""
        import equinox as eqx

        from ham.geometry.manifolds import EuclideanSpace
        from ham.models.wildfire import CovariateConditionedRanders

        H = W = 24
        key = jax.random.PRNGKey(0)
        model = CovariateConditionedRanders(
            EuclideanSpace(2), key, hidden_dim=8, fuel_emb_dim=2,
            cnn_channels=4, use_wind="coupled")
        model = eqx.tree_at(lambda m: m.wind_coupling, model, jnp.asarray(0.6))
        bound = model.bind_scene_rasters(
            elev=jnp.zeros((H, W)), slope=jnp.zeros((H, W)),
            aspect=jnp.zeros((H, W)), canopy=jnp.zeros((H, W)),
            fuel_codes=jnp.zeros((H, W), dtype=jnp.int32),
            pixel_spacing_m=1.0, origin_xy=jnp.zeros(2),
        ).bind_weather(jnp.zeros(4), measured_wind=jnp.array([1.0, 0.0]))
        fm = bound.precompute_metric_field()
        T = np.asarray(solve_arrival_encoder(fm, jnp.array([12.0, 12.0]), (H, W), 1.0))
        assert T.shape == (H, W)
        assert T[12, 20] < T[12, 4], "downwind (+col) must arrive earlier"
        # cross-wind direction stays symmetric
        np.testing.assert_allclose(T[4, 12], T[20, 12], rtol=0.15)


class TestIngestConventions:
    def test_wind_model_coords_sign(self):
        # Wind FROM north (d=0): (wx, wy) = (0, speed); blows southward =
        # +row in model coords -> (0, +speed).
        np.testing.assert_allclose(wind_model_coords(0.0, 5.0), [0.0, 5.0])
        # Wind FROM east (d=90): (wx, wy) = (speed, 0); blows westward =
        # -col -> (-speed, 0).
        np.testing.assert_allclose(wind_model_coords(5.0, 0.0), [-5.0, 0.0])

    def test_fuel_code_remap(self):
        raw = np.array([[1.0, 13.0], [91.0, 99.0]])
        mapped = np.where(raw >= 90, 13, np.clip(raw - 1, 0, 12)).astype(np.int32)
        np.testing.assert_array_equal(mapped, [[0, 12], [13, 13]])

    def test_fit_normalizer_fits_weather_stats(self):
        # Regression (Stage W-C, scene 0004 collapse): SceneNormalizer.fit
        # silently skips the already-reduced (4,) weather vectors, leaving
        # identity weather stats and feeding raw temps/winds to the encoder.
        # fit_normalizer must fit the weather moments itself.
        from experiments.wildfire.ingest import FireRecord, SceneBundle, fit_normalizer

        ones = np.ones((4, 4))
        w4s = [np.array([60.0, 40.0, -2.0, -10.0]),
               np.array([70.0, 55.0, 3.0, 4.0])]
        fires = [
            FireRecord(event_id=f"e{i}", ignition_rc=np.array([1.0, 1.0]),
                       T_obs=ones, burned=ones.astype(bool),
                       weather4_raw=w4, measured_wind=w4[2:])
            for i, w4 in enumerate(w4s)
        ]
        b = SceneBundle(scene_id="t", elev=ones, slope=ones, aspect=ones,
                        canopy=ones, fuel_codes=ones.astype(np.int32),
                        fires=fires, downsample=1)
        norm = fit_normalizer([b])
        w = np.stack(w4s)
        np.testing.assert_allclose(norm.weather_mean, w.mean(axis=0))
        np.testing.assert_allclose(norm.weather_std, w.std(axis=0))
        # z-scoring must actually centre the fires' weather
        z = np.stack([norm.normalize_weather(w4) for w4 in w4s])
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
