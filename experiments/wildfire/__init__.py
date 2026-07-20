"""HAM Wildfire — cross-scene generalization of Randers fire-spread models.

Companion experiment to Gahtan, Shpund & Bronstein (arXiv:2603.00035):
decomposes their cross-scene IoU collapse with the HAM identifiability
dictionary (scene gauge, wind parity, direction coverage) and repairs it via
few-shot (s, c) recalibration and measured-wind coupling. See ``README.md``
for conventions, data prerequisites, and the W-A..W-D stage scripts.
"""

from .medium import (
    GridZermelo,
    direction_coverage,
    fire_metrics,
    godunov_to_grid,
    grid_to_godunov,
    iou_at_50_hours,
    odd_coverage,
    solve_arrival_encoder,
    solve_arrival_grid,
)
from .recover import Recalibration, fit_free_field, recalibrate
from .synthetic import make_scene, simulate_fires

__all__ = [
    "GridZermelo",
    "Recalibration",
    "direction_coverage",
    "fire_metrics",
    "fit_free_field",
    "godunov_to_grid",
    "grid_to_godunov",
    "iou_at_50_hours",
    "make_scene",
    "odd_coverage",
    "recalibrate",
    "simulate_fires",
    "solve_arrival_encoder",
    "solve_arrival_grid",
]
