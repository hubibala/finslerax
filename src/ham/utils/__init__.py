"""Utility modules for the HAM library.

Provides numerical math primitives and device configuration.
"""

from .device import configure_device, get_device
from .math import (
    GRAD_EPS,
    NORM_EPS,
    PSD_EPS,
    TAYLOR_EPS,
    WIND_STIFFNESS,
    causal_wind_scale,
    safe_norm,
)

__all__ = [
    "GRAD_EPS",
    "NORM_EPS",
    "PSD_EPS",
    "TAYLOR_EPS",
    "WIND_STIFFNESS",
    "causal_wind_scale",
    "configure_device",
    "get_device",
    "safe_norm",
]
