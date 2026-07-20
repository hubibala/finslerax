"""Sim2Real-Fire ingestion for the cross-scene benchmark.

Wraps :class:`ham.data.sim2real_loader.Sim2RealFireLoader` and produces
scene bundles ready for the covariate encoder, with three deliberate
departures from ``ham.data.wildfire.load_wildfire_scenario``:

* **arrival times stay in HOURS** — the absolute scene scale is the object of
  study, so it is never normalised away at ingest;
* **rasters are downsampled** (block mean / block min) for CPU training;
* **the raw mean wind velocity is kept** alongside the standardised encoder
  weather vector, because the ``coupled`` drift mode needs the physical
  direction that z-scoring destroys.

Wind convention (empirically validated by the Stage W-C calibration
diagnostic): the weather files carry the meteorological FROM-direction, so the
loader's ``(wx, wy) = speed * (sin d, cos d)`` is the FROM-vector in
(east, north). Rasters are row-0-north, and the encoder's world frame is
``x = col`` (east), ``y = row`` (south), hence the wind (blow) velocity in
model coordinates is ``(-wx, +wy)``. A learned scalar coupling absorbs any
residual global sign error, but not a rotation — hence the diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ham.data.sim2real_loader import Sim2RealFireLoader
from ham.data.wildfire import (
    SceneNormalizer,
    _weather_to_4vec,
    extract_arrival_times,
    find_ignition_point,
)

PIXEL_SPACING_M = 30.0  # Sim2Real-Fire native resolution


@dataclass
class FireRecord:
    event_id: str
    ignition_rc: np.ndarray  # (2,) downsampled pixel coords
    T_obs: np.ndarray  # (H, W) hours, inf outside burned
    burned: np.ndarray  # (H, W) bool
    weather4_raw: np.ndarray  # (4,) [T_air, humidity, wind_x, wind_y] raw
    measured_wind: np.ndarray  # (2,) blow velocity, model (x, y) coords


@dataclass
class SceneBundle:
    scene_id: str
    elev: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray
    canopy: np.ndarray
    fuel_codes: np.ndarray  # int32
    fires: list  # list[FireRecord]
    downsample: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.elev.shape

    @property
    def pixel_spacing_m(self) -> float:
        return PIXEL_SPACING_M * self.downsample


def _block_reduce(a: np.ndarray, d: int, how: str) -> np.ndarray:
    """Downsample (H, W) by factor ``d`` with block mean/min (crop remainder)."""
    if d == 1:
        return a
    H, W = a.shape
    a = a[: H // d * d, : W // d * d].reshape(H // d, d, W // d, d)
    if how == "mean":
        return a.mean(axis=(1, 3))
    if how == "min":
        return a.min(axis=(1, 3))
    if how == "nearest":
        return a[:, 0, :, 0]
    raise ValueError(how)


def _resample_nearest(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample to ``shape``. Needed because some scenes'
    satellite masks have a different native resolution than their terrain
    rasters (e.g. scene 0012)."""
    H2, W2 = shape
    H1, W1 = a.shape
    ri = np.minimum((np.arange(H2) * H1 / H2).astype(int), H1 - 1)
    ci = np.minimum((np.arange(W2) * W1 / W2).astype(int), W1 - 1)
    return a[np.ix_(ri, ci)]


def wind_model_coords(wx: float, wy: float) -> np.ndarray:
    """Blow velocity in model (x=col/east, y=row/south) coords from the
    loader's FROM-vector components ``(wx, wy) = speed * (sin d, cos d)``."""
    return np.array([-wx, +wy])


def load_scene(
    data_root: str,
    scene_id: str,
    n_fires: int,
    downsample: int = 0,
    seed: int = 0,
    min_burned_px: int = 60,
    min_duration_h: float = 6.0,
    target_dim: int = 148,
) -> SceneBundle:
    """Load one scene with ``n_fires`` quality-filtered, subsampled fire events.

    Events are shuffled deterministically by ``seed`` and consumed until
    ``n_fires`` pass the quality filter (enough burned pixels at the working
    resolution, a non-trivial duration, finite weather). ``downsample = 0``
    picks the factor automatically so the longest raster side is ≤
    ``target_dim`` (scene sizes vary 100–430 px natively).
    """
    loader = Sim2RealFireLoader(data_root)
    events = list(loader.scenes[scene_id]["events"])
    rng = np.random.default_rng(seed)
    rng.shuffle(events)

    first = loader.load_scenario(scene_id, events[0])
    topo, fuel = first["topography"], first["fuel"]
    if downsample == 0:
        longest = max(np.asarray(topo["elevation"]).shape)
        downsample = max(1, int(np.ceil(longest / target_dim)))
    d = downsample
    elev = _block_reduce(np.asarray(topo["elevation"], float), d, "mean")
    slope = _block_reduce(np.asarray(topo["slope"], float), d, "mean")
    aspect = _block_reduce(np.asarray(topo["aspect"], float), d, "mean")
    canopy = _block_reduce(np.asarray(fuel["canopy_cover"], float), d, "mean")
    fuel_raw = _block_reduce(np.asarray(fuel["fbfm13"], float), d, "nearest")
    # FBFM13: burnable classes 1-13 -> embedding rows 0-12; non-burnable land
    # cover (raw 91-99: water/urban/rock/agriculture) -> reserved row 13.
    fuel_codes = np.where(
        fuel_raw >= 90, 13, np.clip(fuel_raw - 1, 0, 12)
    ).astype(np.int32)

    fires: list[FireRecord] = []
    for ev in events:
        if len(fires) >= n_fires:
            break
        raw = loader.load_scenario(scene_id, ev)
        masks = np.asarray(raw["masks"])
        T_full = extract_arrival_times(masks)  # hours (frame index)
        # Masks may have a different native resolution than the rasters:
        # resample onto the working raster grid.
        T_obs = _resample_nearest(T_full, elev.shape)
        burned = np.isfinite(T_obs)
        if burned.sum() < min_burned_px:
            continue
        dur = float(T_obs[burned].max())
        if dur < min_duration_h:
            continue
        w = np.asarray(raw["weather"])
        if w.ndim != 2 or len(w) == 0 or not np.all(np.isfinite(w)):
            continue
        w4 = _weather_to_4vec(w)
        ign_mask = np.asarray(find_ignition_point(masks))
        ign = ign_mask * np.array(
            [elev.shape[0] / masks.shape[1], elev.shape[1] / masks.shape[2]]
        )
        fires.append(
            FireRecord(
                event_id=ev,
                ignition_rc=ign,
                T_obs=T_obs,
                burned=burned,
                weather4_raw=w4,
                measured_wind=wind_model_coords(w4[2], w4[3]),
            )
        )
    if len(fires) < n_fires:
        print(f"  note: {scene_id} yielded only {len(fires)}/{n_fires} usable fires")
    return SceneBundle(
        scene_id=scene_id,
        elev=elev,
        slope=slope,
        aspect=aspect,
        canopy=canopy,
        fuel_codes=fuel_codes,
        fires=fires,
        downsample=d,
    )


def fit_normalizer(bundles: list) -> SceneNormalizer:
    """Fit covariate statistics on the TRAINING scenes only (no test leakage)."""
    scenarios = []
    for b in bundles:
        for f in b.fires:
            scenarios.append(
                {
                    "topography": {"elevation": b.elev, "slope": b.slope},
                    "fuel": {"canopy_cover": b.canopy},
                    "weather": f.weather4_raw,
                }
            )
    norm = SceneNormalizer.fit(scenarios)
    # SceneNormalizer.fit's weather extractor expects the raw 5-column series
    # and silently skips already-reduced 4-vectors, which would leave identity
    # weather stats (raw temps ~60s saturating the encoder) — fit them here.
    w = np.stack([f.weather4_raw for b in bundles for f in b.fires])
    std = w.std(axis=0)
    norm.weather_mean = w.mean(axis=0)
    norm.weather_std = np.where(std < 1e-8, 1.0, std)
    return norm


def bind_scene_model(model, bundle: SceneBundle, normalizer: SceneNormalizer):
    """Bake a bundle's (normalised) rasters into the encoder; weather per fire.

    The solver frame is PIXELS (``pixel_spacing_m = 1``): with G eigenvalues
    clamped to [0.1, 10] the expressible slowness is 0.3–3.2 h/px — the right
    range for fires crossing ~100 px in ≤ 72 h. Solving in metres would demand
    slownesses ~1e-2 h/m, far outside the projection bounds.
    """
    import jax.numpy as jnp

    elev_n, slope_n, canopy_n = normalizer.normalize_spatial(
        bundle.elev, bundle.slope, bundle.canopy
    )
    return model.bind_scene_rasters(
        elev=jnp.asarray(elev_n),
        slope=jnp.asarray(slope_n),
        aspect=jnp.asarray(bundle.aspect),
        canopy=jnp.asarray(canopy_n),
        fuel_codes=jnp.asarray(bundle.fuel_codes),
        pixel_spacing_m=1.0,
        origin_xy=jnp.zeros(2),
    )


def bind_fire_model(scene_model, fire: FireRecord, normalizer: SceneNormalizer):
    """Attach one fire's standardised weather + raw wind to a scene-bound model."""
    import jax.numpy as jnp

    w4n = normalizer.normalize_weather(fire.weather4_raw)
    return scene_model.bind_weather(
        jnp.asarray(w4n), measured_wind=jnp.asarray(fire.measured_wind)
    )
