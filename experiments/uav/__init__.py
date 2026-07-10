"""HAM UAV Energy Ledger — the exact-drift gauge on real multirotor flight data.

The real-data instance of the gauge/identifiability program
(``spec/uav_energy_gauge_PLAN.md``): a multirotor's segment energy splits by
parity under segment reversal into an even nuisance bracket (hover, drag,
speed profile) and an odd drift 1-form — the gravity ledger ``k·dz`` plus the
horizontal wind coupling ``b·dxy``. Trajectory shapes cannot determine ``k``
(projective invariance); the energy ledger determines it by convex least
squares, validated against the autopilot's own EKF wind estimate.

Modules: ``medium`` (the cost model and its parity structure), ``ingest``
(ULog → segment table, download driver, offline raw generator), ``synthetic``
(the U2 bridge: fleets with exact ground truth), ``estimate`` (per-log and
Schur-pooled ridge LSQ, negative control), ``evaluate`` (directed-energy R²,
ledger consistency, wind cross-check, balance diagnostics).
"""

from .estimate import (
    FleetFit,
    LogFit,
    SpatialWind,
    fit_log,
    fit_per_log,
    fit_pooled,
    fleet_predictor,
    implied_wind,
    implied_wind_field,
    k_series,
    negative_control,
)
from .evaluate import (
    climb_costs_more,
    cosine,
    directed_energy_r2,
    direction_balance,
    even_odd_leakage,
    fleet_wind_cosine,
    k_consistency,
    ledger_conditioned,
    r2,
    split_segments,
    two_bin_k,
    wind_field_cosine,
)
from .ingest import (
    IngestConfig,
    RawFlight,
    RawFlightSpec,
    align_flight,
    apply_filters,
    download_logs,
    energy_balance,
    ingest_flight,
    ingest_logs,
    load_corpus,
    parse_ulog,
    read_segments,
    segment_track,
    simulate_raw_flight,
    valid_current,
    write_segments,
)
from .medium import (
    EVEN_NAMES,
    ODD_COLUMNS,
    REQUIRED_COLUMNS,
    even_features,
    odd_features,
    rbf_wind_features,
    reverse_segments,
    validate_table,
)
from .synthetic import FleetTruth, PowerModel, make_vortex, synthesize_fleet

__all__ = [
    "EVEN_NAMES",
    "ODD_COLUMNS",
    "REQUIRED_COLUMNS",
    "FleetFit",
    "FleetTruth",
    "IngestConfig",
    "LogFit",
    "PowerModel",
    "RawFlight",
    "RawFlightSpec",
    "SpatialWind",
    "align_flight",
    "apply_filters",
    "climb_costs_more",
    "cosine",
    "directed_energy_r2",
    "direction_balance",
    "download_logs",
    "energy_balance",
    "even_features",
    "even_odd_leakage",
    "fit_log",
    "fit_per_log",
    "fit_pooled",
    "fleet_predictor",
    "fleet_wind_cosine",
    "implied_wind",
    "implied_wind_field",
    "ingest_flight",
    "ingest_logs",
    "k_consistency",
    "k_series",
    "ledger_conditioned",
    "load_corpus",
    "make_vortex",
    "negative_control",
    "odd_features",
    "parse_ulog",
    "r2",
    "rbf_wind_features",
    "read_segments",
    "reverse_segments",
    "segment_track",
    "simulate_raw_flight",
    "split_segments",
    "synthesize_fleet",
    "two_bin_k",
    "valid_current",
    "validate_table",
    "wind_field_cosine",
    "write_segments",
]
