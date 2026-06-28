"""Precision-aware helpers for the dual-mode (float32/float64) test suite.

The suite runs under both JAX precisions — ``JAX_ENABLE_X64=0`` and ``=1`` (see
``ham.utils.config`` and the CI matrix). These helpers let a single assertion or
tolerance adapt to the active precision instead of hardcoding ``float32``.
"""

import numpy as np

from ham.utils.config import default_dtype, x64_enabled

__all__ = ["assert_default_dtype", "tol", "x64_enabled"]


def assert_default_dtype(x):
    """Assert ``x`` carries the configured default float dtype.

    Works for both JAX and NumPy arrays (JAX dtypes are NumPy dtypes). Replaces
    hardcoded ``x.dtype == jnp.float32`` checks so the assertion holds in both
    precision modes.
    """
    expected = np.dtype(default_dtype())
    actual = np.dtype(x.dtype)
    assert actual == expected, f"expected default float dtype {expected}, got {actual}"


def tol(atol32=1e-4, rtol32=1e-4, atol64=1e-9, rtol64=1e-9):
    """Return a precision-scaled ``(atol, rtol)`` pair.

    Loose under float32, tight under float64 — so the same comparison both
    passes in float32 *and* verifies the stronger float64 guarantee under x64.
    Defaults are sensible; pass overrides for a specific call site.
    """
    return (atol64, rtol64) if x64_enabled() else (atol32, rtol32)
