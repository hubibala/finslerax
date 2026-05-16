"""Canonical numerical stability primitives for HAMTools.

This module centralises all epsilon constants and gradient-safe numerical
operations used throughout the library. Every constant corresponds to a specific
stability role described in spec/MATH_SPEC.md § 6 and should be imported rather
than re-defining magic numbers in downstream modules.
"""

import jax.numpy as jnp

__all__ = [
    "GRAD_EPS",
    "NORM_EPS",
    "PSD_EPS",
    "TAYLOR_EPS",
    "safe_norm",
    "safe_norm_additive",
]

# ---------------------------------------------------------------------------
# Canonical numerical constants (P2: consolidate magic numbers)
# ---------------------------------------------------------------------------

GRAD_EPS = 1e-12
"""Guard for ``jnp.sqrt`` backward pass at zero. 

Used inside :func:`safe_norm`. Ref: ``spec/MATH_SPEC.md § 6.1``.
Chosen for float64 safety; for float32 consider larger eps if squared-sum
underflows.
"""

NORM_EPS = 1e-8
"""Threshold for deciding whether a vector is numerically zero.

Used in forward-pass branching (e.g., ``norm < NORM_EPS``), NOT in 
backward-pass guards.
"""

PSD_EPS = 1e-4
"""Canonical positive-definite regularisation floor.

All modules that regularise metric matrices (G -> G + eps*I) should import 
this constant rather than hardcoding magic numbers.
"""

TAYLOR_EPS = 1e-6
"""Threshold for switching to Taylor expansions.

When a quantity (e.g. sin(theta)/theta) is below this threshold, 
implementations should switch to a Taylor series to avoid catastrophic 
cancellation.
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
        eps: Guard epsilon. Defaults to :const:`GRAD_EPS` (1e-12).

    Returns:
        Array of L2 norms with the same dtype as x.
    """
    sq = jnp.sum(x ** 2, axis=axis, keepdims=keepdims)
    return jnp.sqrt(jnp.maximum(sq, eps))

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
    sq = jnp.sum(x ** 2, axis=axis, keepdims=keepdims)
    return jnp.sqrt(sq + eps**2)