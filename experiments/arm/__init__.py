"""HAM Robot-Arm Geodesics — energy-optimal, asymmetric-cost motion planning in C-space.

A differentiable geodesic-BVP study (the AVBD twin of ``experiments/marine``):
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
from .evaluate import path_metrics, spray_geodesic
from .fields import MLPDistance, ScaledDistance
from .interfaces import DemoSource, DistanceField, Robot, Scene
from .medium import ArmMetric, angle_manifold, build_arm_metric
from .planners import AVBDPlanner, EikonalPlanner
from .providers import (
    AnalyticDistance,
    CircleScene,
    GroundTruthDemos,
    PlanarArm,
)

__all__ = [
    "AVBDPlanner",
    "AnalyticDistance",
    "ArmMetric",
    "CircleScene",
    "DemoSource",
    "DistanceField",
    "EikonalPlanner",
    "GroundTruthDemos",
    "MLPDistance",
    "PlanarArm",
    "Robot",
    "ScaledDistance",
    "Scene",
    "angle_manifold",
    "build_arm_metric",
    "constraint_penalty",
    "max_constraint_violation",
    "path_metrics",
    "spray_geodesic",
    "upright_constraint",
    "waypoint_constraint",
]
