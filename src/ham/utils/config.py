"""
Global configuration for the HAM library.

This module defines standard numerical types used throughout the library
to ensure consistency and allow easy toggling between performance and precision.
"""

import jax.numpy as jnp
import numpy as np

# Global configuration for the default numeric precision
# Set DEFAULT_JNP_DTYPE to jnp.float32 for standard GPU performance,
# or jnp.float64 for maximal numerical stability in geometric solvers.
DEFAULT_JNP_DTYPE: jnp.dtype = jnp.float32
DEFAULT_NP_DTYPE: np.dtype = np.float32
