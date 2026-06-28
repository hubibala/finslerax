"""HAM Robot-Arm Geodesics — energy-optimal, asymmetric-cost motion planning in C-space.

A differentiable geodesic-BVP study (the AVBD twin of ``experiments/marine``):
geodesics on configuration-space Riemannian/Finsler metrics with gravity-aware
asymmetry, obstacles folded into the metric, exact ALM task constraints, and a
metric/obstacle field learnable from demonstrations.

The whole experiment depends only on the four protocols in ``interfaces.py``
(:class:`Robot`, :class:`Scene`, :class:`DistanceField`, :class:`DemoSource`), so
the synthetic study here becomes a real-robot study by swapping providers.
"""

from .interfaces import DemoSource, DistanceField, Robot, Scene
from .providers import (
    AnalyticDistance,
    CircleScene,
    GroundTruthDemos,
    PlanarArm,
)

__all__ = [
    "AnalyticDistance",
    "CircleScene",
    "DemoSource",
    "DistanceField",
    "GroundTruthDemos",
    "PlanarArm",
    "Robot",
    "Scene",
]
