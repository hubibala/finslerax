"""Utility modules for the HAM library.

Provides numerical math primitives, data downloaders, and device configuration.

The single-cell data loaders (:func:`download_benchmark_data`,
:func:`download_weinreb`, :func:`process_weinreb`) depend on the optional
``[bio]`` extra (anndata, scanpy, scvelo, pandas). They are imported lazily so
that ``import ham`` works without the bio stack installed; accessing them
without the extra raises a clear :class:`ImportError`.
"""

import importlib
from typing import TYPE_CHECKING

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

# name -> submodule providing it; imported on first access (PEP 562).
_LAZY_BIO = {
    "download_benchmark_data": "download_data",
    "download_weinreb": "download_weinreb",
    "process_weinreb": "download_weinreb",
}


def __getattr__(name: str):
    submodule = _LAZY_BIO.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        mod = importlib.import_module(f".{submodule}", __name__)
    except ImportError as exc:
        raise ImportError(
            f"ham.utils.{name} requires the optional 'bio' dependencies "
            f"(anndata, scanpy, scvelo, pandas). Install with: "
            f"pip install hamtools[bio]"
        ) from exc
    return getattr(mod, name)


if TYPE_CHECKING:  # help static analysers / IDEs resolve the lazy names
    from .download_data import download_benchmark_data
    from .download_weinreb import download_weinreb, process_weinreb

__all__ = [
    "GRAD_EPS",
    "NORM_EPS",
    "PSD_EPS",
    "TAYLOR_EPS",
    "WIND_STIFFNESS",
    "causal_wind_scale",
    "configure_device",
    "download_benchmark_data",
    "download_weinreb",
    "get_device",
    "process_weinreb",
    "safe_norm",
]
