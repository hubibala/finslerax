import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Canonical numerical constants (P2: consolidate magic numbers)
# ---------------------------------------------------------------------------
GRAD_EPS = 1e-12     # Epsilon for gradient-safe operations (sqrt, norm)
NORM_EPS = 1e-8      # Epsilon for norm-based comparisons/thresholds
PSD_EPS = 1e-4       # Epsilon for positive-definite regularisation
TAYLOR_EPS = 1e-6    # Threshold for switching to Taylor expansions

# ---------------------------------------------------------------------------
# Canonical safe_norm — gradient-safe L2 norm
# ---------------------------------------------------------------------------

def safe_norm(x, axis=-1, keepdims=False, eps=GRAD_EPS):
    """Computes norm(x) safely, avoiding NaN gradients at x=0.

    Uses the ``sqrt(max(sum(x²), eps))`` pattern so that both the forward
    *and* backward passes are well-defined everywhere. This is the single
    canonical implementation that should be used throughout the codebase.
    """
    sq = jnp.sum(x ** 2, axis=axis, keepdims=keepdims)
    return jnp.sqrt(jnp.maximum(sq, eps))