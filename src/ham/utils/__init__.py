"""Utility modules for the HAM library.

Provides numerical math primitives, data downloaders, and device configuration.
"""

from .math import safe_norm, GRAD_EPS, NORM_EPS, PSD_EPS, TAYLOR_EPS
from .download_data import download_benchmark_data
from .download_weinreb import download_weinreb, process_weinreb
from .device import get_device, configure_device

__all__ = [
    "safe_norm",
    "GRAD_EPS", "NORM_EPS", "PSD_EPS", "TAYLOR_EPS",
    "download_benchmark_data",
    "download_weinreb", "process_weinreb",
    "get_device", "configure_device",
]
