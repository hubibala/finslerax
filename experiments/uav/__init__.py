"""HAM UAV Energy Ledger — the exact-drift gauge on real multirotor flight data.

The real-data instance of the gauge/identifiability program
(``spec/uav_energy_gauge_PLAN.md``): a multirotor's segment energy splits by
parity under segment reversal into an even nuisance bracket (hover, drag,
speed profile) and an odd drift 1-form — the gravity ledger ``k·dz`` plus the
horizontal wind coupling ``b·dxy``. Trajectory shapes cannot determine ``k``
(projective invariance); the energy ledger determines it by convex least
squares, validated against the autopilot's own EKF wind estimate.

Modules: ``medium`` (the cost model and its parity structure), ``synthetic``
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
    cosine,
    directed_energy_r2,
    direction_balance,
    even_odd_leakage,
    fleet_wind_cosine,
    k_consistency,
    r2,
    split_segments,
    wind_field_cosine,
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
    "LogFit",
    "PowerModel",
    "SpatialWind",
    "cosine",
    "directed_energy_r2",
    "direction_balance",
    "even_features",
    "even_odd_leakage",
    "fit_log",
    "fit_per_log",
    "fit_pooled",
    "fleet_predictor",
    "fleet_wind_cosine",
    "implied_wind",
    "implied_wind_field",
    "k_consistency",
    "k_series",
    "make_vortex",
    "negative_control",
    "odd_features",
    "r2",
    "rbf_wind_features",
    "reverse_segments",
    "split_segments",
    "synthesize_fleet",
    "validate_table",
    "wind_field_cosine",
]
