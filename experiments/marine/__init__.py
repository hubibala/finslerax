"""HAM Marine Navigation — time-dependent Zermelo routing for autonomous vehicles.

A dimension-agnostic frame (``Medium`` / ``Vehicle`` / ``Constraint`` / planners /
evaluation) for time-optimal path planning through ocean currents, instantiated
for a 3D buoyancy-driven underwater glider.

The physics, the framing correction (drifters give the current *directly*; the
novelty is *time-dependent* planning), and the scientific claims are documented
in ``README.md``.
"""

from .constraints import (
    Constraint,
    constraint_penalty,
    depth_envelope,
    glide_angle_limit,
    seafloor_clearance,
)
from .forecast import DecayingForecast, PerfectForecast, PersistenceForecast
from .medium import FrozenMedium, OceanMedium, build_snapshot_metric, randers_cost
from .mpc import MPCResult, run_mpc
from .planners import StationaryPlanner, TimeLiftedPlanner
from .vehicle import Glider

__all__ = [
    "Constraint",
    "DecayingForecast",
    "FrozenMedium",
    "Glider",
    "MPCResult",
    "OceanMedium",
    "PerfectForecast",
    "PersistenceForecast",
    "StationaryPlanner",
    "TimeLiftedPlanner",
    "build_snapshot_metric",
    "constraint_penalty",
    "depth_envelope",
    "glide_angle_limit",
    "randers_cost",
    "run_mpc",
    "seafloor_clearance",
]
