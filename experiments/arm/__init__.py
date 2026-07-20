"""HAM Robot-Arm Geodesics — energy-optimal, asymmetric-cost motion planning in C-space.

A differentiable geodesic-BVP study:
geodesics on configuration-space Riemannian/Finsler metrics with gravity-aware
asymmetry, obstacles folded into the metric, exact ALM task constraints, and a
metric/obstacle field learnable from demonstrations.

The whole experiment depends only on the four protocols in ``interfaces.py``
(:class:`Robot`, :class:`Scene`, :class:`DistanceField`, :class:`DemoSource`), so
the synthetic study here becomes a real-robot study by swapping providers.
"""

from .constraints import (
    constraint_penalty,
    max_constraint_violation,
    upright_constraint,
    waypoint_constraint,
)
from .evaluate import (
    path_metrics,
    polyline_deviation,
    shape_identifiability,
    spray_geodesic,
)
from .fields import (
    BoundedWind,
    HodgeWind,
    MLPDistance,
    PotentialWind,
    ScaledDistance,
    StreamWind,
    VortexWind,
)
from .interfaces import DemoSource, DistanceField, Robot, Scene
from .learn import (
    LOSS_ARMS,
    fit_field,
    recover_form_lsq,
    recover_potential_lsq,
    segment_asymmetry,
    segment_costs,
)
from .medium import ArmMetric, angle_manifold, build_arm_metric
from .planners import AVBDPlanner, EikonalPlanner
from .providers import (
    AnalyticDistance,
    CircleScene,
    GroundTruthDemos,
    PlanarArm,
)

__all__ = [
    "LOSS_ARMS",
    "AVBDPlanner",
    "AnalyticDistance",
    "ArmMetric",
    "BoundedWind",
    "CircleScene",
    "DemoSource",
    "DistanceField",
    "EikonalPlanner",
    "GroundTruthDemos",
    "HodgeWind",
    "MLPDistance",
    "PlanarArm",
    "PotentialWind",
    "Robot",
    "ScaledDistance",
    "Scene",
    "StreamWind",
    "VortexWind",
    "angle_manifold",
    "build_arm_metric",
    "constraint_penalty",
    "fit_field",
    "max_constraint_violation",
    "path_metrics",
    "polyline_deviation",
    "recover_form_lsq",
    "recover_potential_lsq",
    "segment_asymmetry",
    "segment_costs",
    "shape_identifiability",
    "spray_geodesic",
    "upright_constraint",
    "waypoint_constraint",
]
