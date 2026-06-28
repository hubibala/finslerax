"""Canonical numerical stability primitives for HAMTools.

This module centralises all epsilon constants and gradient-safe numerical
operations used throughout the library. Every constant corresponds to a specific
stability role described in spec/MATH_SPEC.md § 6 and should be imported rather
than re-defining magic numbers in downstream modules.
"""

import jax.numpy as jnp

from ham.utils.config import EpsKind as _EpsKind
from ham.utils.config import eps as _eps

__all__ = [
    "GRAD_EPS",
    "NORM_EPS",
    "PSD_EPS",
    "TAYLOR_EPS",
    "WIND_STIFFNESS",
    "causal_wind_scale",
    "safe_norm",
    "safe_norm_additive",
]

# ---------------------------------------------------------------------------
# Canonical numerical constants (P2: consolidate magic numbers)
# ---------------------------------------------------------------------------

# These floors scale with the working precision (see ham.utils.config.eps):
# the float32 values below are the legacy defaults; under HAM_X64 they
# tighten to float64-appropriate values. They are module-level constants because
# precision is fixed at import time.

GRAD_EPS = _eps(_EpsKind.GRAD)
"""Guard for ``jnp.sqrt`` backward pass at zero.

Used inside :func:`safe_norm`. Ref: ``spec/MATH_SPEC.md § 6.1``.
Precision-scaled: ``1e-6`` (float32) / ``1e-13`` (float64). For float32,
consider a larger eps if a squared-sum underflows.
"""

NORM_EPS = _eps(_EpsKind.NORM)
"""Threshold for deciding whether a vector is numerically zero.

Used in forward-pass branching (e.g., ``norm < NORM_EPS``), NOT in
backward-pass guards. Precision-scaled: ``1e-8`` (float32) / ``1e-12`` (float64).
"""

PSD_EPS = _eps(_EpsKind.PSD)
"""Canonical positive-definite regularisation floor.

All modules that regularise metric matrices (G -> G + eps*I) should import
this constant rather than hardcoding magic numbers. Precision-scaled:
``1e-4`` (float32) / ``1e-10`` (float64).
"""

TAYLOR_EPS = _eps(_EpsKind.TAYLOR)
"""Threshold for switching to Taylor expansions.

When a quantity (e.g. sin(theta)/theta) is below this threshold,
implementations should switch to a Taylor series to avoid catastrophic
cancellation. Precision-scaled: ``1e-4`` (float32) / ``1e-8`` (float64,
``~ sqrt(finfo(float64).eps)``).
"""

WIND_STIFFNESS = 30.0
"""Default sharpness of the Zermelo causal wind clamp.

Sets the width (``~1/WIND_STIFFNESS``) of the transition shell in which
:func:`causal_wind_scale` bends a wind toward the causal boundary. Larger
values approach the hard clamp ``min(||W||, max_speed)``; smaller values smear
the transition over a wider band. Ref: ``spec/MATH_SPEC.md § 5``.
"""

# ---------------------------------------------------------------------------
# Numerical Primitives
# ---------------------------------------------------------------------------


def safe_norm(x, axis=-1, keepdims=False, eps=GRAD_EPS):
    """Compute the L2 norm safely, avoiding NaN gradients at zero.

    Uses a ``max``-clamping strategy: ``sqrt(max(sum(x²), eps))``.
    This ensures that the derivative 1/(2*sqrt(·)) remains finite at the origin.

    Note:
        Returns ``sqrt(eps)`` (not 0) when ``x = 0``. This is intended for
        backward-pass stability. For forward-pass zero-checks, use
        ``NORM_EPS``.

    Args:
        x: Input array of arbitrary shape.
        axis: Axis along which to compute the norm. Defaults to -1.
        keepdims: Whether to keep the reduced axis. Defaults to False.
        eps: Guard epsilon. Defaults to :const:`GRAD_EPS` (1e-6).

    Returns:
        Array of L2 norms with the same dtype as x.
    """
    sq = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.sqrt(jnp.maximum(sq, eps))


def causal_wind_scale(norm, max_speed, stiffness=WIND_STIFFNESS):
    r"""Smooth, identity-preserving causal clamp factor for Zermelo winds.

    Returns a multiplicative scale ``s`` such that ``s * norm`` is a
    :math:`C^\infty` smooth under-approximation of ``min(norm, max_speed)``,
    guaranteeing ``s * norm < max_speed`` *strictly*. This enforces the Zermelo
    strong-convexity bound ``||W||_H < 1`` (with ``max_speed = 1 - epsilon``;
    see ``spec/MATH_SPEC.md § 5``) without distorting physically-valid winds.

    Construction — the temperature-controlled smooth minimum::

        phi(r) = r - softplus(stiffness * (r - max_speed)) / stiffness

    which is monotone (``phi'(r) = 1 - sigmoid(stiffness*(r - max_speed))`` lies
    in ``(0, 1)``) and satisfies ``sup_r phi(r) = max_speed``, approached from
    below. The returned scale is ``phi(norm) / norm``.

    Why not ``max_speed * tanh(norm) / norm`` (the historical squash)?
        ``tanh`` has slope ``max_speed < 1`` at the origin, so it bends *every*
        wind — e.g. a requested ``||W|| = 0.5`` silently becomes ``~0.46``.
        This clamp is instead the identity to within
        ``~exp(-stiffness * (max_speed - norm)) / stiffness`` whenever
        ``norm < max_speed``; bending is confined to a thin shell of width
        ``~1/stiffness`` around the causal boundary, where it is unavoidable.

    Args:
        norm: Non-negative wind magnitude ``||W||_H``. Use a gradient-safe norm
            (e.g. :func:`safe_norm`) so the backward pass is finite at zero.
        max_speed: Causal supremum, typically ``1 - epsilon``.
        stiffness: Transition sharpness. Larger → thinner shell, closer to the
            hard clamp. Defaults to :const:`WIND_STIFFNESS`.

    Returns:
        Scale ``s`` (same shape as ``norm``) with ``s * norm < max_speed``.
    """
    # softplus(z) = log(1 + e^z), evaluated stably via logaddexp.
    phi = norm - jnp.logaddexp(0.0, stiffness * (norm - max_speed)) / stiffness
    # Guard the ~exp(-stiffness * max_speed) undershoot of phi at norm -> 0
    # (would otherwise flip the wind sign by a negligible amount).
    phi = jnp.maximum(phi, 0.0)
    # Floor the divisor with ``maximum`` (not ``+ eps``): an additive guard would
    # bias the scale by ~eps/norm even for valid winds, breaking exact identity;
    # ``maximum`` only engages at norm ~ 0 (where phi ~ 0, so the ratio -> 0).
    return phi / jnp.maximum(norm, GRAD_EPS)


def safe_norm_additive(x, axis=-1, keepdims=False, eps=GRAD_EPS):
    """Compute the L2 norm using additive regularisation.

    Uses the pattern specified in ``spec/MATH_SPEC.md § 6.1``:
    ``sqrt(sum(x²) + eps²)``. This is C^infinity smooth and preferable for
    higher-order autodiff (e.g., Berwald connection), but introduces a
    small bias everywhere.

    Args:
        x: Input array of arbitrary shape.
        axis: Axis along which to compute the norm. Defaults to -1.
        keepdims: Whether to keep the reduced axis. Defaults to False.
        eps: Regularisation parameter (epsilon). Defaults to :const:`GRAD_EPS`.

    Returns:
        Array of regularised L2 norms.
    """
    sq = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.sqrt(sq + eps**2)
