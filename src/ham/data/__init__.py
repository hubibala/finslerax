"""
HAM Data Package
================

Data loading, preprocessing, and normalization utilities for neural/geometric simulations.
"""

from .sim2real_loader import Sim2RealFireLoader
from .wildfire import (
    SceneNormalizer,
    WildfireScenario,
    compute_slope_std,
    extract_arrival_times,
    find_ignition_point,
    iou_at_50,
    load_wildfire_scenario,
    stratified_sample_observations,
    train_val_test_split,
)

__all__ = [
    "SceneNormalizer",
    "Sim2RealFireLoader",
    "WildfireScenario",
    "compute_slope_std",
    "extract_arrival_times",
    "find_ignition_point",
    "iou_at_50",
    "load_wildfire_scenario",
    "stratified_sample_observations",
    "train_val_test_split",
]
