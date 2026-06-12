"""Utility modules for the HAM library.

Provides numerical math primitives, data downloaders, and device configuration.
"""

from .device import configure_device, get_device
from .download_data import download_benchmark_data
from .download_weinreb import download_weinreb, process_weinreb
from .math import GRAD_EPS, NORM_EPS, PSD_EPS, TAYLOR_EPS, safe_norm

__all__ = [
    "GRAD_EPS",
    "NORM_EPS",
    "PSD_EPS",
    "TAYLOR_EPS",
    "configure_device",
    "download_benchmark_data",
    "download_weinreb",
    "get_device",
    "process_weinreb",
    "safe_norm",
]
