"""
HAM Data Package
================

Data loading, preprocessing, and normalization utilities for neural/geometric simulations.
"""

from .wildfire import (
    WildfireScenario,
    extract_arrival_times,
    find_ignition_point,
    stratified_sample_observations,
    iou_at_50,
    SceneNormalizer,
    load_wildfire_scenario,
    train_val_test_split,
    compute_slope_std,
)
from .sim2real_loader import Sim2RealFireLoader

__all__ = [
    "WildfireScenario",
    "extract_arrival_times",
    "find_ignition_point",
    "stratified_sample_observations",
    "iou_at_50",
    "SceneNormalizer",
    "load_wildfire_scenario",
    "train_val_test_split",
    "compute_slope_std",
    "Sim2RealFireLoader",
]
