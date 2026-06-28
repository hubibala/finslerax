"""Unit tests for ham.data.wildfire — no real dataset required."""

import numpy as np
import pytest
from _precision import assert_default_dtype

from ham.data.wildfire import (
    SceneNormalizer,
    _weather_to_4vec,
    extract_arrival_times,
    find_ignition_point,
    iou_at_50,
    stratified_sample_observations,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_raw(seed: int) -> dict:
    """Synthetic raw scenario dict for normalizer tests (no file I/O)."""
    rng = np.random.default_rng(seed)
    H, W = 20, 20
    return {
        "topography": {
            "elevation": rng.uniform(100, 3000, (H, W)).astype(np.float32),
            "slope":     rng.uniform(0, 45,   (H, W)).astype(np.float32),
            "aspect":    rng.uniform(0, 360,  (H, W)).astype(np.float32),
        },
        "fuel": {
            "canopy_cover": rng.uniform(0, 100, (H, W)).astype(np.float32),
            "fbfm13":       rng.integers(1, 13, (H, W)).astype(np.int32),
        },
        # Gahtan format: (T, 5) = [temp, humidity, wind_speed, sin, cos]
        "weather": rng.uniform(
            [60, 20, 0, -1, -1], [100, 80, 30, 1, 1], (10, 5)
        ).astype(np.float32),
        "masks": np.zeros((3, H, W), dtype=bool),
    }


# ---------------------------------------------------------------------------
# extract_arrival_times
# ---------------------------------------------------------------------------

class TestExtractArrivalTimes:
    def test_basic_known_pixels(self):
        """Known masks produce known arrival times."""
        masks = np.zeros((4, 3, 3), dtype=bool)
        masks[0, 0, 0] = True   # burns at t=0
        masks[2, 1, 1] = True   # burns at t=2
        # (2, 2) never burns

        arr = extract_arrival_times(masks)

        assert arr.shape == (3, 3)
        assert arr[0, 0] == pytest.approx(0.0)
        assert arr[1, 1] == pytest.approx(2.0)
        assert np.isinf(arr[2, 2])

    def test_first_burn_wins(self):
        """Pixel burned in multiple frames keeps the earliest arrival time."""
        masks = np.zeros((3, 2, 2), dtype=bool)
        masks[0, 0, 0] = True
        masks[1, 0, 0] = True   # re-burns — should be ignored
        masks[1, 0, 1] = True   # new burn at t=1

        arr = extract_arrival_times(masks)

        assert arr[0, 0] == pytest.approx(0.0)
        assert arr[0, 1] == pytest.approx(1.0)
        assert np.isinf(arr[1, 0])
        assert np.isinf(arr[1, 1])

    def test_all_burned_at_t0(self):
        """All pixels burning at t=0 yields a zero-valued arrival array."""
        masks = np.ones((3, 4, 4), dtype=bool)
        arr = extract_arrival_times(masks)
        assert np.all(arr == 0.0)

    def test_none_burned(self):
        """No burned pixels → all inf."""
        masks = np.zeros((5, 3, 3), dtype=bool)
        arr = extract_arrival_times(masks)
        assert np.all(np.isinf(arr))

    def test_dtype(self):
        masks = np.ones((2, 3, 3), dtype=np.uint8)
        arr = extract_arrival_times(masks)
        assert_default_dtype(arr)


# ---------------------------------------------------------------------------
# find_ignition_point
# ---------------------------------------------------------------------------

class TestFindIgnitionPoint:
    def test_centroid_correctness(self):
        """Centroid of symmetric first-frame burns is exact."""
        masks = np.zeros((5, 10, 10), dtype=bool)
        # Burns at (2,2), (2,4), (4,2), (4,4) → centroid (3, 3)
        for r, c in [(2, 2), (2, 4), (4, 2), (4, 4)]:
            masks[0, r, c] = True

        pt = find_ignition_point(masks)

        assert pt.shape == (2,)
        assert_default_dtype(pt)
        np.testing.assert_allclose(pt, [3.0, 3.0])

    def test_fallback_to_frames_0_2(self):
        """Falls back to frames 0–2 when frame 0 is empty."""
        masks = np.zeros((5, 6, 6), dtype=bool)
        masks[1, 3, 3] = True   # frame 0 empty; frame 1 has one pixel

        pt = find_ignition_point(masks)

        np.testing.assert_allclose(pt, [3.0, 3.0])

    def test_single_pixel(self):
        """Single burned pixel → centroid equals that pixel."""
        masks = np.zeros((2, 5, 5), dtype=bool)
        masks[0, 2, 4] = True

        pt = find_ignition_point(masks)

        np.testing.assert_allclose(pt, [2.0, 4.0])

    def test_returns_shape_2(self):
        masks = np.zeros((1, 10, 10), dtype=bool)
        pt = find_ignition_point(masks)
        assert pt.shape == (2,)


# ---------------------------------------------------------------------------
# stratified_sample_observations
# ---------------------------------------------------------------------------

class TestStratifiedSampleObservations:
    def test_output_shape(self):
        """Returns exactly n_samples rows with valid indices."""
        H, W = 30, 30
        arrival = np.full((H, W), np.inf)
        for i in range(15):
            for j in range(15):
                arrival[i, j] = float(i + j)

        idx = stratified_sample_observations(arrival, 100, seed=7)

        assert idx.shape == (100, 2)
        assert idx.dtype in (np.int64, np.int32, int)

    def test_no_out_of_bounds(self):
        H, W = 20, 20
        arrival = np.full((H, W), np.inf)
        rng = np.random.default_rng(1)
        pixels = rng.choice(H * W, 200, replace=False)
        flat = np.full(H * W, np.inf)
        flat[pixels] = np.linspace(0, 50, 200)
        arrival = flat.reshape(H, W)

        idx = stratified_sample_observations(arrival, 80, seed=3)

        assert np.all(idx[:, 0] >= 0) and np.all(idx[:, 0] < H)
        assert np.all(idx[:, 1] >= 0) and np.all(idx[:, 1] < W)

    def test_all_deciles_represented(self):
        """All 10 equal-width time buckets have at least one sample."""
        H, W = 50, 50
        rng = np.random.default_rng(0)
        flat = np.full(H * W, np.inf)
        pixels = rng.choice(H * W, 1000, replace=False)
        flat[pixels] = np.linspace(0.0, 100.0, 1000)
        arrival = flat.reshape(H, W)

        idx = stratified_sample_observations(arrival, 100, seed=0)

        assert idx.shape[0] == 100
        t_vals = arrival[idx[:, 0], idx[:, 1]]
        edges = np.linspace(0.0, 100.0, 11)
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            in_bucket = (t_vals >= lo) & (t_vals <= hi)
            assert in_bucket.any(), f"Decile bucket {i} is empty in sample"

    def test_empty_returns_zero_rows(self):
        """Returns (0, 2) when no pixels are burned."""
        arrival = np.full((10, 10), np.inf)
        idx = stratified_sample_observations(arrival, 50, seed=0)
        assert idx.shape == (0, 2)

    def test_returned_pixels_are_burned(self):
        """All returned pixels have finite arrival times."""
        H, W = 20, 20
        arrival = np.full((H, W), np.inf)
        arrival[:10, :10] = np.arange(100).reshape(10, 10).astype(float)

        idx = stratified_sample_observations(arrival, 60, seed=9)

        for r, c in idx:
            assert np.isfinite(arrival[r, c]), f"Sampled unburned pixel ({r},{c})"


# ---------------------------------------------------------------------------
# iou_at_50
# ---------------------------------------------------------------------------

class TestIouAt50:
    def test_perfect_prediction(self):
        """Identical pred and gt → IoU = 1.0."""
        H, W = 10, 10
        gt = np.linspace(0.0, 1.0, H * W).reshape(H, W)
        burned = np.ones((H, W), dtype=bool)

        assert iou_at_50(gt, gt, burned) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Pred has nothing ≤ 0.5, gt has everything ≤ 0.5 → IoU = 0.0."""
        H, W = 10, 10
        gt   = np.zeros((H, W))          # all ≤ 0.5 → in gt_perim
        pred = np.ones((H, W))           # all > 0.5 → not in pred_perim
        burned = np.ones((H, W), dtype=bool)

        assert iou_at_50(pred, gt, burned) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap → strictly between 0 and 1."""
        H, W = 10, 10
        gt   = np.zeros((H, W))          # all in gt_perim
        pred = np.zeros((H, W))
        pred[:5, :] = 1.0               # only bottom half in pred_perim
        burned = np.ones((H, W), dtype=bool)

        iou = iou_at_50(pred, gt, burned)
        assert 0.0 < iou < 1.0

    def test_burned_mask_restricts(self):
        """Pixels outside burned_mask are ignored even if pred/gt differ."""
        H, W = 4, 4
        gt   = np.zeros((H, W))
        pred = np.zeros((H, W))
        burned = np.zeros((H, W), dtype=bool)  # nothing is burned

        assert iou_at_50(pred, gt, burned) == pytest.approx(0.0)

    def test_range(self):
        """Result is always in [0, 1]."""
        rng = np.random.default_rng(42)
        H, W = 20, 20
        for _ in range(10):
            pred   = rng.uniform(0, 1, (H, W))
            gt     = rng.uniform(0, 1, (H, W))
            burned = rng.integers(0, 2, (H, W)).astype(bool)
            val = iou_at_50(pred, gt, burned)
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# SceneNormalizer
# ---------------------------------------------------------------------------

class TestSceneNormalizer:
    def test_fit_and_normalize_spatial(self):
        """After fitting and normalizing, mean ≈ 0 and std ≈ 1 per channel."""
        scenarios = [_make_synthetic_raw(i) for i in range(30)]
        normalizer = SceneNormalizer.fit(scenarios)

        elev_all, slope_all, canopy_all = [], [], []
        for raw in scenarios:
            e = raw["topography"]["elevation"]
            s = raw["topography"]["slope"]
            c = raw["fuel"]["canopy_cover"]
            en, sn, cn = normalizer.normalize_spatial(e, s, c)
            elev_all.append(en.ravel())
            slope_all.append(sn.ravel())
            canopy_all.append(cn.ravel())

        for name, arr in [
            ("elevation", np.concatenate(elev_all)),
            ("slope",     np.concatenate(slope_all)),
            ("canopy",    np.concatenate(canopy_all)),
        ]:
            np.testing.assert_allclose(
                arr.mean(), 0.0, atol=0.05,
                err_msg=f"{name} normalised mean not ≈ 0"
            )
            np.testing.assert_allclose(
                arr.std(), 1.0, atol=0.05,
                err_msg=f"{name} normalised std not ≈ 1"
            )

    def test_weather_normalization_mean(self):
        """Normalized weather vectors from training data have mean ≈ 0."""
        scenarios = [_make_synthetic_raw(i) for i in range(30)]
        normalizer = SceneNormalizer.fit(scenarios)

        weather_vecs = []
        for raw in scenarios:
            w4 = _weather_to_4vec(np.asarray(raw["weather"]))
            weather_vecs.append(normalizer.normalize_weather(w4))

        W = np.stack(weather_vecs)   # (30, 4)
        # T_air (col 0) and humidity (col 1) should be approximately centred
        np.testing.assert_allclose(W[:, 0].mean(), 0.0, atol=0.5)
        np.testing.assert_allclose(W[:, 1].mean(), 0.0, atol=0.5)

    def test_fit_empty_scenarios(self):
        """Fitting on empty list uses safe fallback (no crash, std=1)."""
        norm = SceneNormalizer.fit([])
        assert norm.elev_std == pytest.approx(1.0)
        assert norm.slope_std == pytest.approx(1.0)
        assert norm.canopy_std == pytest.approx(1.0)
        np.testing.assert_array_equal(norm.weather_std, np.ones(4))

    def test_fit_missing_keys(self):
        """Scenarios with missing rasters don't crash SceneNormalizer.fit."""
        raw = {"topography": {}, "fuel": {}, "weather": None, "masks": np.zeros((1, 5, 5))}
        norm = SceneNormalizer.fit([raw])
        assert norm.elev_std == pytest.approx(1.0)

    def test_normalize_weather_shape(self):
        norm = SceneNormalizer.fit([_make_synthetic_raw(0)])
        w = norm.normalize_weather(np.array([75.0, 50.0, 0.5, 0.866], dtype=np.float32))
        assert w.shape == (4,)
        assert_default_dtype(w)
