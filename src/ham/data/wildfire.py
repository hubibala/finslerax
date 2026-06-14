"""
Wildfire Data Loading and Processing Utilities
==============================================

Provides dataset loaders, normalizers, and spatial/temporal preprocessing methods for
the Sim2Real-Fire dataset (Gahtan et al., 2026).
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from ham.utils.config import DEFAULT_NP_DTYPE


class WildfireScenario:
    """Data class storing all spatial, temporal, and covariate attributes of a wildfire event."""

    def __init__(
        self,
        scene_id: str,
        event_id: str,
        ignition_pixel: np.ndarray,
        ignition_world: np.ndarray,
        arrival_times: np.ndarray,
        arrival_times_hours: np.ndarray,
        obs_pixels: np.ndarray,
        obs_arrival_times: np.ndarray,
        elev_raster: np.ndarray,
        slope_raster: np.ndarray,
        aspect_raster: np.ndarray,
        canopy_raster: np.ndarray,
        fuel_code_raster: np.ndarray,
        weather_vec: np.ndarray,
        pixel_spacing_m: float,
        origin_xy: np.ndarray,
        burned_mask: np.ndarray,
        val_pixels: np.ndarray | None = None,
        val_arrival_times: np.ndarray | None = None,
    ):
        self.scene_id = scene_id
        self.event_id = event_id
        self.ignition_pixel = ignition_pixel
        self.ignition_world = ignition_world
        self.arrival_times = arrival_times
        self.arrival_times_hours = arrival_times_hours
        self.obs_pixels = obs_pixels
        self.obs_arrival_times = obs_arrival_times
        self.elev_raster = elev_raster
        self.slope_raster = slope_raster
        self.aspect_raster = aspect_raster
        self.canopy_raster = canopy_raster
        self.fuel_code_raster = fuel_code_raster
        self.weather_vec = weather_vec
        self.pixel_spacing_m = pixel_spacing_m
        self.origin_xy = origin_xy
        self.burned_mask = burned_mask
        self.val_pixels = (
            val_pixels if val_pixels is not None else np.zeros((0, 2), dtype=np.int64)
        )
        self.val_arrival_times = (
            val_arrival_times
            if val_arrival_times is not None
            else np.zeros((0,), dtype=DEFAULT_NP_DTYPE)
        )


def extract_arrival_times(masks: np.ndarray) -> np.ndarray:
    """Convert binary mask sequence to pixel-level arrival times.

    Args:
        masks: Shape (T, H, W) boolean/integer array, where masks[t, r, c]
            is True if pixel (r, c) has burned by or at time step t.

    Returns:
        arrival_times: Shape (H, W) float64 array representing the first time step
            at which each pixel burned. Never-burned pixels are set to np.inf.
    """
    masks = np.asarray(masks)
    any_burned = np.any(masks, axis=0)
    first_burn = np.argmax(masks, axis=0).astype(DEFAULT_NP_DTYPE)
    first_burn[~any_burned] = np.inf
    return first_burn


def find_ignition_point(masks: np.ndarray) -> np.ndarray:
    """Determine the ignition source pixel (row, col) as the centroid of the initial burn region.

    Args:
        masks: Shape (T, H, W) boolean array.

    Returns:
        Shape (2,) float64 array representing the row and column coordinates.
    """
    masks = np.asarray(masks)
    # Check frame 0 first
    r, c = np.where(masks[0])
    if len(r) > 0:
        return np.array([np.mean(r), np.mean(c)], dtype=DEFAULT_NP_DTYPE)

    # Fallback to cumulative frames 0-2 (up to index 3)
    T = masks.shape[0]
    limit = min(3, T)
    r, c = np.where(np.any(masks[:limit], axis=0))
    if len(r) > 0:
        return np.array([np.mean(r), np.mean(c)], dtype=DEFAULT_NP_DTYPE)

    # Fallback to the entire mask sequence
    r, c = np.where(np.any(masks, axis=0))
    if len(r) > 0:
        return np.array([np.mean(r), np.mean(c)], dtype=DEFAULT_NP_DTYPE)

    # Ultimate fallback: center of the spatial grid
    H, W = masks.shape[1], masks.shape[2]
    return np.array([H / 2.0, W / 2.0], dtype=DEFAULT_NP_DTYPE)


def stratified_sample_observations(
    arrival: np.ndarray, n_samples: int, seed=None
) -> np.ndarray:
    """Sample exactly n_samples pixel coordinates stratified across 10 equal-width time bins.

    Args:
        arrival: Shape (H, W) arrival times array (finite values for burned pixels, inf otherwise).
        n_samples: Number of coordinates to sample.
        seed: Random seed for reproducibility.

    Returns:
        Shape (n_samples, 2) int array of sampled pixel coordinates.
        Returns shape (0, 2) if no pixels are burned.
    """
    rng = np.random.default_rng(seed)
    r, c = np.where(np.isfinite(arrival))
    total_candidates = len(r)
    if total_candidates == 0 or n_samples <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    coords = np.stack([r, c], axis=1)
    times = arrival[r, c]

    if total_candidates <= n_samples:
        # Sample with replacement to get exactly n_samples
        idx = rng.choice(total_candidates, size=n_samples, replace=True)
        return coords[idx]

    t_min, t_max = np.min(times), np.max(times)
    if np.abs(t_max - t_min) < 1e-9:
        indices = rng.choice(total_candidates, size=n_samples, replace=False)
        return coords[indices]

    # Partition arrival times into 10 equal-width bins
    bin_idx = np.minimum(
        9, np.floor((times - t_min) / (t_max - t_min) * 10).astype(int)
    )
    bin_coords = [coords[bin_idx == b] for b in range(10)]
    capacities = np.array([len(b) for b in bin_coords])

    # Iteratively allocate sample quotas per bin ensuring decile representation
    targets = np.zeros(10, dtype=int)
    active = np.ones(10, dtype=bool)

    remaining = n_samples
    while remaining > 0 and np.any(active):
        n_active = np.sum(active)
        share = remaining // n_active
        if share == 0:
            # Distribute remaining samples 1-by-1 to the first active bins
            active_indices = np.where(active)[0]
            for idx in active_indices[:remaining]:
                targets[idx] += 1
            break

        for b in range(10):
            if active[b]:
                give = min(share, capacities[b] - targets[b])
                targets[b] += give
                remaining -= give
                if targets[b] == capacities[b]:
                    active[b] = False

    # Sample without replacement from each bin according to targets
    samples = []
    for b in range(10):
        if targets[b] > 0:
            candidates = bin_coords[b]
            idx = rng.choice(len(candidates), size=targets[b], replace=False)
            samples.append(candidates[idx])

    res = np.concatenate(samples, axis=0)
    rng.shuffle(res)
    return res


def iou_at_50(pred: np.ndarray, gt: np.ndarray, burned: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) of the fire perimeter at a threshold of 0.5.

    Args:
        pred: Shape (H, W) predicted arrival times normalized to [0, 1].
        gt: Shape (H, W) ground-truth arrival times normalized to [0, 1].
        burned: Shape (H, W) boolean mask restricting the evaluation pixels.

    Returns:
        Intersection-over-Union value in [0, 1].
    """
    pred_bin = (pred <= 0.5) & burned
    gt_bin = (gt <= 0.5) & burned
    intersection = np.sum(pred_bin & gt_bin)
    union = np.sum(pred_bin | gt_bin)
    if union == 0:
        return 0.0
    return float(intersection / union)


def _weather_to_4vec(weather: np.ndarray) -> np.ndarray:
    """Map raw weather time-series in Gahtan format to a time-averaged 4-vector.

    Raw columns: [temp, humidity, wind_speed, wind_dir_sin, wind_dir_cos]
    Output 4-vector: [temp, humidity, wind_speed * sin, wind_speed * cos]

    Args:
        weather: Shape (T, 5) or (5,) array.

    Returns:
        Shape (4,) float64 array.
    """
    weather = np.asarray(weather)
    if weather.ndim == 1:
        t, h, ws, s, c = weather
        return np.array([t, h, ws * s, ws * c], dtype=DEFAULT_NP_DTYPE)
    elif weather.ndim == 2:
        t = np.mean(weather[:, 0])
        h = np.mean(weather[:, 1])
        wx = np.mean(weather[:, 2] * weather[:, 3])
        wy = np.mean(weather[:, 2] * weather[:, 4])
        return np.array([t, h, wx, wy], dtype=DEFAULT_NP_DTYPE)
    else:
        raise ValueError("Weather must be a 1D or 2D array.")


class SceneNormalizer:
    """Standardizes terrain features and weather parameters using fitted spatial/temporal scales."""

    def __init__(
        self,
        elev_mean: float = 0.0,
        elev_std: float = 1.0,
        slope_mean: float = 0.0,
        slope_std: float = 1.0,
        canopy_mean: float = 0.0,
        canopy_std: float = 1.0,
        weather_mean: Optional[np.ndarray] = None,
        weather_std: Optional[np.ndarray] = None,
    ):
        self.elev_mean = float(elev_mean)
        self.elev_std = float(elev_std)
        self.slope_mean = float(slope_mean)
        self.slope_std = float(slope_std)
        self.canopy_mean = float(canopy_mean)
        self.canopy_std = float(canopy_std)
        self.weather_mean = (
            np.zeros(4)
            if weather_mean is None
            else np.asarray(weather_mean, dtype=DEFAULT_NP_DTYPE)
        )
        self.weather_std = (
            np.ones(4)
            if weather_std is None
            else np.asarray(weather_std, dtype=DEFAULT_NP_DTYPE)
        )

    @classmethod
    def fit(cls, scenarios: list):
        """Compute global means and standard deviations across a list of scenarios."""
        if not scenarios:
            return cls()

        elev_vals = []
        slope_vals = []
        canopy_vals = []
        weather_vecs = []

        for sc in scenarios:
            # Extract elevation
            e = None
            if isinstance(sc, dict):
                e = sc.get("topography", {}).get("elevation")
            else:
                e = getattr(sc, "elev_raster", None)
            if e is not None:
                elev_vals.append(np.asarray(e).ravel())

            # Extract slope
            s = None
            if isinstance(sc, dict):
                s = sc.get("topography", {}).get("slope")
            else:
                s = getattr(sc, "slope_raster", None)
            if s is not None:
                slope_vals.append(np.asarray(s).ravel())

            # Extract canopy
            c = None
            if isinstance(sc, dict):
                c = sc.get("fuel", {}).get("canopy_cover")
            else:
                c = getattr(sc, "canopy_raster", None)
            if c is not None:
                canopy_vals.append(np.asarray(c).ravel())

            # Extract weather
            w = None
            if isinstance(sc, dict):
                w = sc.get("weather")
            else:
                w = getattr(sc, "weather_vec", None)
            if w is not None:
                try:
                    w4 = _weather_to_4vec(w)
                    weather_vecs.append(w4)
                except Exception:
                    pass

        # Compute spatial statistics
        if elev_vals:
            all_elev = np.concatenate(elev_vals)
            elev_mean = np.mean(all_elev)
            elev_std = np.std(all_elev)
            if elev_std < 1e-8:
                elev_std = 1.0
        else:
            elev_mean, elev_std = 0.0, 1.0

        if slope_vals:
            all_slope = np.concatenate(slope_vals)
            slope_mean = np.mean(all_slope)
            slope_std = np.std(all_slope)
            if slope_std < 1e-8:
                slope_std = 1.0
        else:
            slope_mean, slope_std = 0.0, 1.0

        if canopy_vals:
            all_canopy = np.concatenate(canopy_vals)
            canopy_mean = np.mean(all_canopy)
            canopy_std = np.std(all_canopy)
            if canopy_std < 1e-8:
                canopy_std = 1.0
        else:
            canopy_mean, canopy_std = 0.0, 1.0

        # Compute weather statistics
        if weather_vecs:
            all_w = np.stack(weather_vecs)
            weather_mean = np.mean(all_w, axis=0)
            weather_std = np.std(all_w, axis=0)
            weather_std = np.where(weather_std < 1e-8, 1.0, weather_std)
        else:
            weather_mean, weather_std = np.zeros(4), np.ones(4)

        return cls(
            elev_mean,
            elev_std,
            slope_mean,
            slope_std,
            canopy_mean,
            canopy_std,
            weather_mean,
            weather_std,
        )

    def normalize_spatial(
        self, elev: np.ndarray, slope: np.ndarray, canopy: np.ndarray
    ):
        """Apply fitted standardization to spatial covariates."""
        elev_n = (np.asarray(elev) - self.elev_mean) / self.elev_std
        slope_n = (np.asarray(slope) - self.slope_mean) / self.slope_std
        canopy_n = (np.asarray(canopy) - self.canopy_mean) / self.canopy_std
        return elev_n, slope_n, canopy_n

    def normalize_weather(self, weather: np.ndarray):
        """Apply fitted standardization to time-averaged weather 4-vector."""
        return (np.asarray(weather) - self.weather_mean) / self.weather_std


def load_wildfire_scenario(
    loader,
    scene_id: str,
    event_id: str,
    normalizer: SceneNormalizer,
    k_train_obs: int,
    seed: Optional[int] = None,
) -> WildfireScenario:
    """Load and preprocess a single fire scenario from a dataset loader.

    Args:
        loader: Sim2RealFireLoader instance.
        scene_id: Scene ID string.
        event_id: Event ID string.
        normalizer: SceneNormalizer to standardize covariates.
        k_train_obs: Number of observation pixels to sample.
        seed: Random seed for reproducibility.

    Returns:
        Preprocessed WildfireScenario instance.
    """
    raw = loader.load_scenario(scene_id, event_id)

    # Core coordinates and dimensions
    masks = np.asarray(raw["masks"])
    arrival_times_hours = extract_arrival_times(masks)
    burned_mask = np.isfinite(arrival_times_hours)

    # Standardize time to [0, 1] relative to the maximum observed arrival time
    if np.any(burned_mask):
        t_max = float(arrival_times_hours[burned_mask].max())
    else:
        t_max = 1.0
    if t_max < 1e-8:
        t_max = 1.0
    arrival_times = arrival_times_hours / t_max

    # Ignition coordinates
    ignition_pixel = find_ignition_point(masks)
    pixel_spacing_m = 30.0  # Sim2Real-Fire dataset default resolution is 30m
    ignition_world = np.array(
        [
            float(ignition_pixel[1]) * pixel_spacing_m,
            float(ignition_pixel[0]) * pixel_spacing_m,
        ],
        dtype=DEFAULT_NP_DTYPE,
    )

    # Stratified observation sampling (training targets)
    obs_pixels = stratified_sample_observations(
        arrival_times_hours, n_samples=k_train_obs, seed=seed
    )
    if len(obs_pixels) > 0:
        obs_arrival_times = arrival_times[obs_pixels[:, 0], obs_pixels[:, 1]]
    else:
        obs_arrival_times = np.zeros((0,), dtype=DEFAULT_NP_DTYPE)

    # Held-out validation pixels: draw a fresh stratified sample with a different
    # seed so val_pixels are independent of the training observation set.
    _val_seed = (seed + 9973) if seed is not None else 9973  # deterministic offset
    val_pixels = stratified_sample_observations(
        arrival_times_hours, n_samples=min(100, 2 * k_train_obs), seed=_val_seed
    )
    if len(val_pixels) > 0:
        val_arrival_times = arrival_times[val_pixels[:, 0], val_pixels[:, 1]]
    else:
        val_arrival_times = np.zeros((0,), dtype=DEFAULT_NP_DTYPE)

    # Standardize covariates
    elev_raster, slope_raster, canopy_raster = normalizer.normalize_spatial(
        raw["topography"]["elevation"],
        raw["topography"]["slope"],
        raw["fuel"]["canopy_cover"],
    )

    aspect_raster = np.asarray(raw["topography"]["aspect"], dtype=DEFAULT_NP_DTYPE)
    fuel_code_raster = np.asarray(raw["fuel"]["fbfm13"], dtype=np.int32)
    origin_xy = np.zeros(2, dtype=DEFAULT_NP_DTYPE)

    # Standardize weather
    weather_vec = normalizer.normalize_weather(_weather_to_4vec(raw["weather"]))

    return WildfireScenario(
        scene_id=scene_id,
        event_id=event_id,
        ignition_pixel=ignition_pixel,
        ignition_world=ignition_world,
        arrival_times=arrival_times,
        arrival_times_hours=arrival_times_hours,
        obs_pixels=obs_pixels,
        obs_arrival_times=obs_arrival_times,
        elev_raster=elev_raster,
        slope_raster=slope_raster,
        aspect_raster=aspect_raster,
        canopy_raster=canopy_raster,
        fuel_code_raster=fuel_code_raster,
        weather_vec=weather_vec,
        pixel_spacing_m=pixel_spacing_m,
        origin_xy=origin_xy,
        burned_mask=burned_mask,
        val_pixels=val_pixels,
        val_arrival_times=val_arrival_times,
    )


def train_val_test_split(
    scenarios_keys,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Shuffle and split scenario keys into training, validation, and testing sets.

    Args:
        scenarios_keys: List of scene/event pairs.
        train_ratio: Proportion of data allocated for training.
        val_ratio: Proportion of data allocated for validation.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (train_keys, val_keys, test_keys).
    """
    rng = np.random.default_rng(seed)
    keys = list(scenarios_keys)
    shuffled_idx = rng.permutation(len(keys))

    n_train = round(len(keys) * train_ratio)
    n_val = round(len(keys) * val_ratio)

    train_idx = shuffled_idx[:n_train]
    val_idx = shuffled_idx[n_train : n_train + n_val]
    test_idx = shuffled_idx[n_train + n_val :]

    train = [keys[i] for i in train_idx]
    val = [keys[i] for i in val_idx]
    test = [keys[i] for i in test_idx]

    return train, val, test


def compute_slope_std(scenario: WildfireScenario) -> float:
    """Calculate the standard deviation of slope values within the scenario.

    Args:
        scenario: WildfireScenario instance.

    Returns:
        Standard deviation scalar.
    """
    return float(np.std(scenario.slope_raster))
