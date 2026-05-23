import jax.numpy as jnp
import numpy as np

# Global configuration for the default numeric precision
# Set this to jnp.float32 for standard GPU performance,
# or jnp.float64 for maximal numerical stability in geometric solvers.

DEFAULT_JNP_DTYPE = jnp.float32
DEFAULT_NP_DTYPE = np.float32

