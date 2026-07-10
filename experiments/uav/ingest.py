"""Ingest — PX4 ULog → per-flight segment table, and the bulk download driver.

The pipeline is four pure-pandas stages, each independently testable, with the
only heavy dependency (``pyulog``, for the actual ``.ulg`` binary parse) isolated
in :func:`parse_ulog` and imported lazily. Everything downstream of it — the
frame conversion, alignment, segmentation, energy integral, and the §4 quality
gates — runs on plain DataFrames, so :func:`simulate_raw_flight` can drive the
whole chain offline and deterministically (that is what the U0/U1 tests do, and
how the frozen fixtures are generated).

    parse_ulog / simulate_raw_flight   →  RawFlight   (async PX4 topics, NED)
    align_flight                       →  track       (uniform grid, ENU + power)
    segment_track                      →  segments    (the medium.py schema)
    apply_filters                      →  segments, survivorship census

**Frame and sign conventions — the load-bearing part.** PX4
``vehicle_local_position`` is **NED** (x=North, y=East, z=**Down**); the segment
table is **ENU-flavoured** so that ``dz > 0`` is a climb and the horizontal axes
match the wind topic (``windspeed_north``/``windspeed_east``):

    dx = ΔEast  = Δy_ned     dy = ΔNorth = Δx_ned     dz = ΔUp = −Δz_ned
    wind_x = windspeed_east  wind_y = windspeed_north

Getting this transform wrong silently inverts the ledger sign — U0 pins it with
a round-trip. Energy is the honest ``∫V·I dt`` (trapezoid on the aligned grid),
never throttle; length is densified (``Σ|Δpos|`` over the fine samples, ≥ chord).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .medium import validate_table

# numpy renamed trapz → trapezoid in 2.0 (trapz deprecated); support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

#: PX4 uORB topics we read, and the field(s) we pull from each. Field lookups
#: are best-effort (firmware revisions rename ``voltage_v`` ↔ ``voltage_filtered_v``),
#: resolved by :func:`_first_present`.
TOPIC_FIELDS = {
    "vehicle_local_position": ("x", "y", "z", "vx", "vy", "vz"),
    "battery_status": ("voltage_v|voltage_filtered_v", "current_a|current_filtered_a"),
    "wind": ("windspeed_north|wind_north", "windspeed_east|wind_east"),
    "vehicle_air_data": ("baro_alt_meter",),
}


# ===========================================================================
# Raw flight container
# ===========================================================================
@dataclass
class RawFlight:
    """Asynchronous PX4 topics (each a DataFrame with a ``t`` column in seconds).

    ``topics`` maps topic name → DataFrame; the flight-level scalars carry the
    metadata the science is per-log conditioned on (airframe class) or gated by
    (battery capacity, for the ∫I dt sanity check).
    """

    topics: dict
    log_id: str
    airframe: str = "unknown"
    vehicle_uuid: str = "unknown"  # sys_uuid — the physical vehicle (for same-vehicle fleets)
    capacity_ah: float | None = None  # pack capacity [A·h] for the U0 balance gate


@dataclass(frozen=True)
class IngestConfig:
    """Knobs for the align→segment→filter pipeline (the §4 quality gates)."""

    grid_dt: float = 0.1        # alignment grid spacing [s]
    seg_dt: float = 4.0         # nominal segment duration [s]
    cruise_speed: tuple = (2.0, 25.0)  # keep segments whose mean speed is in-band [m/s]
    accel_cap: float = 4.0      # reject segments with peak |a| above this [m/s²]
    agl_floor: float = 5.0      # reject segments below this altitude AGL [m]
    trim_s: float = 3.0         # drop this much off each end of the flight [s]
    min_duration: float = 60.0  # skip whole flights shorter than this [s]


#: Shared default config (frozen ⇒ safe as a module-level singleton default arg).
DEFAULT_INGEST = IngestConfig()


# ===========================================================================
# Stage 0 — the binary parse (the only pyulog-dependent function)
# ===========================================================================
def _first_present(df_cols, spec: str):
    """Resolve a ``a|b|c`` field spec to the first column actually present."""
    for name in spec.split("|"):
        if name in df_cols:
            return name
    return None


def parse_ulog(path, *, log_id: str | None = None) -> RawFlight:
    """Parse a ``.ulg`` file into a :class:`RawFlight` (lazy ``pyulog`` import).

    Extracts the :data:`TOPIC_FIELDS` topics, renames each resolved field to its
    canonical name, and converts timestamps (µs) to seconds. Airframe class is
    taken from ``SYS_AUTOSTART`` and pack capacity from a ``BAT*_CAPACITY`` param
    when present. Missing optional topics (``wind``, ``vehicle_air_data``) are
    simply absent from ``topics``.
    """
    try:
        from pyulog import ULog
    except ImportError as exc:  # pragma: no cover - exercised only with pyulog absent
        raise ImportError(
            "parse_ulog needs pyulog: pip install 'hamtools[uav]'"
        ) from exc

    path = Path(path)
    ulog = ULog(str(path))
    params = ulog.initial_parameters
    canonical = {"x", "y", "z", "vx", "vy", "vz", "voltage_v", "current_a",
                 "windspeed_north", "windspeed_east", "baro_alt_meter"}
    topics: dict = {}
    for name, specs in TOPIC_FIELDS.items():
        try:
            data = ulog.get_dataset(name).data
        except (KeyError, IndexError, ValueError):
            continue
        cols = {"t": np.asarray(data["timestamp"], float) * 1e-6}
        for spec in specs:
            resolved = _first_present(data.keys(), spec)
            if resolved is None:
                continue
            key = spec.split("|")[0]
            cols[key if key in canonical else resolved] = np.asarray(data[resolved], float)
        topics[name] = pd.DataFrame(cols)

    cap = next((params[k] for k in params if k.startswith("BAT") and k.endswith("CAPACITY")), None)
    return RawFlight(
        topics=topics,
        log_id=log_id or path.stem,
        airframe=str(params.get("SYS_AUTOSTART", "unknown")),
        vehicle_uuid=str(ulog.msg_info_dict.get("sys_uuid", "unknown")),
        capacity_ah=None if cap is None else float(cap) / 1000.0,  # mAh → Ah
    )


# ===========================================================================
# Stage 1 — align asynchronous topics onto a uniform grid, NED → ENU
# ===========================================================================
def align_flight(raw: RawFlight, cfg: IngestConfig = DEFAULT_INGEST) -> pd.DataFrame:
    """Resample topics onto a uniform time grid; emit an ENU track with power.

    Columns: ``t`` [s]; ``e, n, u`` position [m] (up so ``u`` is AGL over a
    ground launch); ``ve, vn, vu`` velocity [m/s]; ``voltage`` [V], ``current``
    [A], ``power`` [W]; ``wind_e, wind_n`` [m/s] (0 where the topic is absent).
    """
    lp = raw.topics["vehicle_local_position"].sort_values("t")
    t0, t1 = float(lp["t"].iloc[0]), float(lp["t"].iloc[-1])
    grid = np.arange(t0, t1, cfg.grid_dt)

    def resample(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
        if df is None or col not in df:
            return np.full(grid.shape, default)
        d = df.dropna(subset=[col]).sort_values("t")
        return np.interp(grid, d["t"].to_numpy(float), d[col].to_numpy(float))

    bat = raw.topics.get("battery_status")
    wind = raw.topics.get("wind")

    # NED → ENU: e=y, n=x, u=−z (position and velocity alike).
    track = pd.DataFrame({
        "t": grid,
        "e": resample(lp, "y"), "n": resample(lp, "x"), "u": -resample(lp, "z"),
        "ve": resample(lp, "vy"), "vn": resample(lp, "vx"), "vu": -resample(lp, "vz"),
        "voltage": resample(bat, "voltage_v"),
        "current": resample(bat, "current_a"),
        "wind_e": resample(wind, "windspeed_east"),
        "wind_n": resample(wind, "windspeed_north"),
    })
    # AGL referenced to the flight's lowest observed point (a ground proxy robust
    # to an EKF that initializes mid-air, unlike using the first sample).
    track["u"] = track["u"] - float(track["u"].min())
    track["power"] = track["voltage"] * track["current"]
    return track


def valid_current(track: pd.DataFrame) -> bool:
    """True if the battery current looks like a real discharge measurement.

    PX4 reports ``current_a = -1`` (constant) when there is no power module, and
    a stuck/zero signal is equally unusable. The energy ledger is meaningless on
    such logs — a large fraction of the public corpus — so they are skipped, not
    fit. Real discharge current is positive and varies as the throttle changes.
    """
    c = track["current"].to_numpy(float)
    return bool(np.median(c) > 0.1 and np.ptp(c) > 0.1)


def energy_balance(track: pd.DataFrame) -> dict:
    """Whole-flight ledgers for the U0 sanity gate: charge [A·h] and energy [J]."""
    t = track["t"].to_numpy(float)
    charge_c = float(_trapz(track["current"].to_numpy(float), t))  # coulombs
    energy_j = float(_trapz(track["power"].to_numpy(float), t))
    return {"charge_ah": charge_c / 3600.0, "energy_j": energy_j,
            "duration_s": float(t[-1] - t[0])}


# ===========================================================================
# Stage 2 — segmentation (the medium.py schema, densified geometry)
# ===========================================================================
def segment_track(
    track: pd.DataFrame, raw: RawFlight, cfg: IngestConfig = DEFAULT_INGEST
) -> pd.DataFrame:
    """Slice the aligned track into ~``seg_dt`` windows → the segment table.

    Length is the densified path ``Σ|Δpos|`` over the fine grid samples (≥ chord,
    the tunneling guard); energy is ``∫V·I dt`` (trapezoid) over the window; wind
    columns are the segment-mean EKF wind, kept for the Stage-C cross-check.
    Trims ``trim_s`` off each end (takeoff/landing transients).
    """
    t = track["t"].to_numpy(float)
    t0, t1 = t[0] + cfg.trim_s, t[-1] - cfg.trim_s
    pos = track[["e", "n", "u"]].to_numpy(float)
    vel = track[["ve", "vn", "vu"]].to_numpy(float)
    power = track["power"].to_numpy(float)
    windc = track[["wind_e", "wind_n"]].to_numpy(float)

    rows = []
    # Include the final partial window so the segmentation covers [t0, t1] with
    # no dropped tail (∫ over the segments = ∫ over the flight, no silent gap).
    edges = np.append(np.arange(t0, t1, cfg.seg_dt), t1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t <= b)
        if m.sum() < 3:
            continue
        p, v, ts = pos[m], vel[m], t[m]
        disp = p[-1] - p[0]
        length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        dt = float(ts[-1] - ts[0])
        # 3-D speed: the cruise band excludes hover (~0) and acro (fast); a climb
        # is moving, so gating on total speed keeps the ledger-bearing climbs.
        speed = float(np.mean(np.linalg.norm(v, axis=1)))
        accel = np.linalg.norm(np.diff(v, axis=0), axis=1) / np.diff(ts)
        rows.append({
            "log_id": raw.log_id, "airframe": raw.airframe,
            "vehicle_uuid": raw.vehicle_uuid,
            "t0": float(a), "dt": dt,
            "dx": float(disp[0]), "dy": float(disp[1]), "dz": float(disp[2]),
            "length": length, "energy": float(_trapz(power[m], ts)),
            "mid_x": float(np.mean(p[:, 0])), "mid_y": float(np.mean(p[:, 1])),
            "mid_z": float(np.mean(p[:, 2])),
            "wind_x": float(np.mean(windc[m, 0])), "wind_y": float(np.mean(windc[m, 1])),
            "speed": speed, "accel_max": float(np.max(accel)) if accel.size else 0.0,
            "agl_min": float(np.min(p[:, 2])),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Stage 3 — quality gates (§4), with an honest survivorship census
# ===========================================================================
def apply_filters(
    segments: pd.DataFrame, cfg: IngestConfig = DEFAULT_INGEST
) -> tuple[pd.DataFrame, dict]:
    """Apply the cruise-band / accel-cap / AGL-floor gates; report survivorship.

    Returns the kept segments and a census dict (fraction surviving each gate and
    overall) — survivorship is reported, never hidden (§6).
    """
    n = len(segments)
    if n == 0:
        return segments, {"n_in": 0, "n_out": 0, "kept_frac": float("nan")}
    lo, hi = cfg.cruise_speed
    g_speed = segments["speed"].between(lo, hi)
    g_accel = segments["accel_max"] <= cfg.accel_cap
    g_agl = segments["agl_min"] >= cfg.agl_floor
    keep = g_speed & g_accel & g_agl
    census = {
        "n_in": int(n), "n_out": int(keep.sum()),
        "kept_frac": float(keep.mean()),
        "pass_cruise": float(g_speed.mean()),
        "pass_accel": float(g_accel.mean()),
        "pass_agl": float(g_agl.mean()),
    }
    return segments[keep].reset_index(drop=True), census


def ingest_flight(
    raw: RawFlight, cfg: IngestConfig = DEFAULT_INGEST
) -> tuple[pd.DataFrame, dict]:
    """Full pipeline on a parsed flight → (filtered segment table, census).

    Flights shorter than ``min_duration``, missing the position or battery topic
    (bench logs, no power module), or with an invalid current signal are dropped
    whole (empty table, ``census['skipped']`` set). The returned table satisfies
    the medium.py schema contract.
    """
    if "vehicle_local_position" not in raw.topics or "battery_status" not in raw.topics:
        return pd.DataFrame(), {"skipped": "missing_topic"}
    track = align_flight(raw, cfg)
    bal = energy_balance(track)
    if bal["duration_s"] < cfg.min_duration:
        return pd.DataFrame(), {"skipped": "too_short", **bal}
    if not valid_current(track):
        return pd.DataFrame(), {"skipped": "invalid_current", **bal}
    segments = segment_track(track, raw, cfg)
    kept, census = apply_filters(segments, cfg)
    census.update(bal)
    if raw.capacity_ah is not None:
        census["charge_within_capacity"] = bal["charge_ah"] <= raw.capacity_ah * 1.05
    if len(kept):
        validate_table(kept)
    return kept, census


def ingest_logs(
    paths, cfg: IngestConfig = DEFAULT_INGEST, *, parser: Callable = parse_ulog
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest many ``.ulg`` paths → (pooled segment table, per-log census table).

    ``parser`` is injectable so tests can drive real segments through a synthetic
    raw generator without touching pyulog. Failed parses are recorded in the
    census (``error`` column), never fatal — junk logs are surveyed, not hidden.
    """
    tables, census_rows = [], []
    for p in paths:
        try:
            raw = parser(p)
            seg, census = ingest_flight(raw, cfg)
        except Exception as exc:  # a bad log must not kill the batch
            census_rows.append({"log_id": str(p), "error": f"{type(exc).__name__}: {exc}"})
            continue
        census_rows.append({"log_id": raw.log_id, **census})
        if len(seg):
            tables.append(seg)
    pooled = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return pooled, pd.DataFrame(census_rows)


# ===========================================================================
# Parquet IO (graceful without pyarrow) and the bulk download driver
# ===========================================================================
def write_segments(df: pd.DataFrame, path) -> None:
    """Persist a segment table as parquet (the corpus format)."""
    try:
        df.to_parquet(path, index=False)
    except (ImportError, ValueError) as exc:  # pragma: no cover
        raise ImportError(
            "writing parquet needs pyarrow: pip install 'hamtools[uav]'"
        ) from exc


def read_segments(path) -> pd.DataFrame:
    """Load a segment table written by :func:`write_segments`."""
    return pd.read_parquet(path)


def load_corpus(
    source, cfg: IngestConfig = DEFAULT_INGEST, *, parser: Callable = parse_ulog
) -> pd.DataFrame:
    """Load a real segment-table corpus from a directory of parquet or ``.ulg`` files.

    This is the single seam that turns every stage script from synthetic to real:
    point it at a directory of preprocessed segment tables (``*.parquet``, the
    corpus format) or raw logs (``*.ulg``, parsed and ingested on the fly) and it
    returns the same pooled table the synthetic generator produces. Parquet is
    preferred when present (already filtered); ``.ulg`` files are run through the
    full :func:`ingest_logs` pipeline. Raises a clear error if the directory holds
    neither — so a mis-set path fails loudly rather than silently going synthetic.
    """
    source = Path(source)
    parquet = sorted(source.glob("*.parquet"))
    if parquet:
        pooled = pd.concat([read_segments(p) for p in parquet], ignore_index=True)
        validate_table(pooled)
        return pooled
    ulg = sorted(source.glob("*.ulg"))
    if ulg:
        pooled, _ = ingest_logs(ulg, cfg, parser=parser)
        return pooled
    raise FileNotFoundError(f"no .parquet or .ulg files under {source}")


def download_logs(
    out_dir,
    *,
    max_logs: int = 500,
    airframe_types: tuple = ("Quadrotor",),
    min_duration_s: float = 300.0,
    session=None,
    base_url: str = "https://review.px4.io",
) -> list:
    """Bulk-download public PX4 Flight Review logs (CC-BY) matching filters.

    Best-effort driver against the Flight Review API: pulls the public log index,
    keeps rotorcraft flights over ``min_duration_s``, and downloads up to
    ``max_logs`` ``.ulg`` files into ``out_dir``. Network-touching, so it is never
    exercised by the test suite; ``session`` is injectable for offline use.
    Returns the list of downloaded paths.
    """
    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()

    index = session.get(f"{base_url}/dbinfo", timeout=30).json()
    rows = index["data"] if isinstance(index, dict) else index
    picked = [
        r for r in rows
        if r.get("mav_type") in airframe_types
        and float(r.get("duration_s", r.get("duration", 0)) or 0) >= min_duration_s
    ][:max_logs]

    paths = []
    for r in picked:
        log_id = r["log_id"]
        dest = out_dir / f"{log_id}.ulg"
        if not dest.exists():
            resp = session.get(f"{base_url}/download", params={"log": log_id}, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            time.sleep(0.2)  # be polite to the public service
        paths.append(dest)
    return paths


# ===========================================================================
# Synthetic raw generator — the offline U0/U1 driver and fixture source
# ===========================================================================
@dataclass
class RawFlightSpec:
    """A known ENU trajectory + battery draw to synthesize PX4-style raw topics."""

    waypoints: np.ndarray            # (W, 3) ENU [m], first is the launch point
    speed: float = 10.0              # cruise speed along horizontal legs [m/s]
    climb_speed: float | None = None  # speed on vertical-dominant legs (slower; default = speed)
    hover_current: float = 20.0      # baseline pack current [A]
    climb_current_gain: float = 4.0  # extra amps per m/s of climb
    idle_frac: float = 0.2           # current floor as a fraction of hover (no regen)
    voltage: float = 22.0            # ~6S nominal [V]
    wind: tuple = (0.0, 0.0)         # (east, north) EKF wind [m/s]
    rate_hz: float = 50.0            # position/velocity topic rate
    bat_hz: float = 10.0             # battery topic rate
    noise: float = 0.0               # position noise [m]
    field_generators: dict = field(default_factory=dict)


def simulate_raw_flight(
    spec: RawFlightSpec, *, log_id: str = "sim", airframe: str = "quad_a", seed: int = 0
) -> RawFlight:
    """Synthesize PX4-convention raw topics (NED, async) from a known ENU spec.

    Flies straight legs between ``waypoints`` at ``speed``; battery current rises
    with climb rate so that the whole-flight ∫V·I dt and the up/down asymmetry
    are analytically known. Position/velocity are emitted in **NED** exactly as
    ``vehicle_local_position`` would, so a correct :func:`align_flight` must undo
    the transform — the U0 round-trip. Returns a :class:`RawFlight`.
    """
    rng = np.random.default_rng(seed)
    wp = np.asarray(spec.waypoints, float)
    climb_speed = spec.speed if spec.climb_speed is None else spec.climb_speed
    # Dense ENU trajectory: constant speed along each leg (slower on climbs).
    times, enu, venu = [0.0], [wp[0].copy()], []
    for a, b in zip(wp[:-1], wp[1:]):
        d = b - a
        vertical = abs(d[2]) > np.linalg.norm(d[:2])
        leg_t = float(np.linalg.norm(d) / (climb_speed if vertical else spec.speed))
        n = max(2, int(leg_t * spec.rate_hz))
        vel = d / leg_t
        for k in range(1, n + 1):
            times.append(times[-1] + leg_t / n)
            enu.append(a + d * k / n)
            venu.append(vel)
    venu.append(venu[-1] if venu else np.zeros(3))
    t = np.asarray(times)
    enu = np.asarray(enu)
    venu = np.asarray(venu)
    if spec.noise:
        enu = enu + spec.noise * rng.standard_normal(enu.shape)

    # ENU → NED for the raw topics: x_ned=n, y_ned=e, z_ned=−u.
    ned = np.column_stack([enu[:, 1], enu[:, 0], -enu[:, 2]])
    vned = np.column_stack([venu[:, 1], venu[:, 0], -venu[:, 2]])
    lp = pd.DataFrame({
        "t": t, "x": ned[:, 0], "y": ned[:, 1], "z": ned[:, 2],
        "vx": vned[:, 0], "vy": vned[:, 1], "vz": vned[:, 2],
    })

    # Battery on its own (slower) clock; current tracks climb rate but never goes
    # negative — a multirotor draws (reduced) power descending, it does not regen.
    tb = np.arange(t[0], t[-1], 1.0 / spec.bat_hz)
    climb = np.interp(tb, t, venu[:, 2])
    current = np.maximum(
        spec.hover_current + spec.climb_current_gain * climb,
        spec.idle_frac * spec.hover_current,
    )
    bat = pd.DataFrame({"t": tb, "voltage_v": np.full(tb.shape, spec.voltage),
                        "current_a": current})

    topics = {"vehicle_local_position": lp, "battery_status": bat}
    if spec.wind != (0.0, 0.0):
        topics["wind"] = pd.DataFrame({
            "t": np.array([t[0], t[-1]]),
            "windspeed_east": np.full(2, spec.wind[0]),
            "windspeed_north": np.full(2, spec.wind[1]),
        })
    return RawFlight(topics=topics, log_id=log_id, airframe=airframe)
