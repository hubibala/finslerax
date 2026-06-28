"""Providers implementing the arm interfaces — synthetic (session 1) and real (stub)."""

from .synthetic import (
    AnalyticDistance,
    CircleScene,
    GroundTruthDemos,
    PlanarArm,
)

__all__ = [
    "AnalyticDistance",
    "CircleScene",
    "GroundTruthDemos",
    "PlanarArm",
]
