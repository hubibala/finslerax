"""Global numerical-precision configuration for the HAM library.

This module is the **single source of truth** for the floating-point precision
used throughout HAM (core, experiments, examples, tests). HAM follows the
JAX-native convention: precision is governed by JAX's own ``jax_enable_x64``
flag, and HAM merely *reads* it.

Selecting precision (all standard JAX mechanisms — pick one):

    # 1. Environment variable (recommended). JAX reads it during ``import jax``,
    #    before any array exists, so there are NO import-ordering pitfalls:
    $ JAX_ENABLE_X64=1 python script.py

    # 2. Programmatically, at startup, before the first array / jit:
    import jax
    jax.config.update("jax_enable_x64", True)

Default is ``float32`` (GPU/TPU throughput). ``float64`` is a deliberate opt-in
for stiff / ill-conditioned solves (long AVBD geodesics, fine eikonal grids,
curvature/transport cancellation, tight VJP checks).

Precision is decided at the *data-construction boundary*: solvers are
dtype-following (they allocate with ``dtype=x.dtype``), so building input data
at the configured precision propagates automatically.
"""

from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "DEFAULT_JNP_DTYPE",
    "DEFAULT_NP_DTYPE",
    "EpsKind",
    "default_dtype",
    "default_np_dtype",
    "eps",
    "x64_enabled",
]


# ---------------------------------------------------------------------------
# Public precision accessors (read JAX's live flag — the single source of truth)
# ---------------------------------------------------------------------------


def x64_enabled() -> bool:
    """Whether 64-bit precision is active.

    Reads JAX's live ``jax_enable_x64`` flag, so it reflects however precision
    was enabled (``JAX_ENABLE_X64`` env var, ``jax.config.update``, CLI flag, or
    a test fixture).
    """
    return bool(jax.config.jax_enable_x64)


def default_dtype() -> jnp.dtype:
    """Default JAX float dtype for the active precision.

    ``float64`` when x64 is enabled, else ``float32``. Implemented via
    :func:`jax.dtypes.canonicalize_dtype`, the canonical JAX precision map.
    """
    return jnp.dtype(jax.dtypes.canonicalize_dtype(jnp.float64))


def default_np_dtype() -> np.dtype:
    """Default NumPy float dtype for the active precision.

    ``float64`` when x64 is enabled, else ``float32`` (via
    :func:`jax.dtypes.canonicalize_dtype`).
    """
    return np.dtype(jax.dtypes.canonicalize_dtype(np.float64))


# Backwards-compatible snapshots of the startup precision. The accessors above
# are authoritative (they read the live flag); these constants capture the value
# at import time for the common case where precision is fixed at startup, and
# are the canonical names imported across the codebase.
DEFAULT_JNP_DTYPE: jnp.dtype = default_dtype()
DEFAULT_NP_DTYPE: np.dtype = default_np_dtype()


# ---------------------------------------------------------------------------
# Precision-scaled numerical floors
# ---------------------------------------------------------------------------
#
# Stability epsilons must scale with the working precision.
#
# The float32 branch reproduces the legacy constants *exactly* so that
# default-mode behaviour is unchanged. The float64 branch is derived from the
# double-precision machine epsilon ``finfo(float64).eps ≈ 2.22e-16``:
#
#   - "grad" / "psd" / "norm" are additive floors -> scale ~ machine epsilon.
#   - "taylor" is a cancellation crossover (naive sin(x)/x etc. loses precision
#     near x ~ sqrt(eps)) -> scale ~ sqrt(machine epsilon).
#
# See spec/MATH_SPEC.md § 6 for the role of each constant.


class EpsKind(str, Enum):
    """Role of a precision-scaled numerical floor (see :func:`eps`).

    Subclasses ``str`` so members compare/serialise as their value (e.g.
    ``EpsKind.GRAD == "grad"``), which keeps call sites readable and lets
    string values be coerced via ``EpsKind(value)``.
    """

    GRAD = "grad"  # :data:`ham.utils.math.GRAD_EPS` — sqrt backward guard.
    NORM = "norm"  # ``NORM_EPS`` — forward zero-vector threshold.
    PSD = "psd"  # ``PSD_EPS`` — positive-definite regularisation floor.
    TAYLOR = "taylor"  # ``TAYLOR_EPS`` — Taylor-series cancellation crossover.


_EPS_FLOAT32: dict[EpsKind, float] = {
    EpsKind.GRAD: 1e-6,
    EpsKind.NORM: 1e-8,
    EpsKind.PSD: 1e-4,
    EpsKind.TAYLOR: 1e-4,
}

_EPS_FLOAT64: dict[EpsKind, float] = {
    EpsKind.GRAD: 1e-13,  # sqrt backward guard; well above finfo(f64).tiny.
    EpsKind.NORM: 1e-12,  # forward zero-vector threshold.
    EpsKind.PSD: 1e-10,  # positive-definite regularisation floor.
    EpsKind.TAYLOR: 1e-8,  # ~ sqrt(finfo(f64).eps); Taylor-series crossover.
}


def eps(kind: EpsKind) -> float:
    """Precision-scaled numerical floor for the active working precision.

    Args:
        kind: Which floor to return, as an :class:`EpsKind` member (a plain
            string with the same value is also accepted and coerced).

    Returns:
        The float32 historical value when x64 is off, or the tightened
        float64 value when x64 is on.

    Raises:
        ValueError: If ``kind`` is not a recognised :class:`EpsKind`.
    """
    table = _EPS_FLOAT64 if x64_enabled() else _EPS_FLOAT32
    return table[EpsKind(kind)]
