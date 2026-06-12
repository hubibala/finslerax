__version__ = "1.1.0"

from jax import config

from . import bio, geometry, solvers
from .bio import GeometricVAE
from .geometry import FinslerMetric, Manifold
from .geometry.mesh import TriangularMesh
from .geometry.surfaces import Sphere
from .geometry.zoo import DiscreteRanders, Euclidean, Randers, Riemannian
from .nn import PSDMatrixField, VectorField
from .solvers import AVBDSolver, ExponentialMap
from .utils import safe_norm
from .vis import (
    generate_icosphere,
    plot_indicatrices,
    plot_sphere,
    plot_trajectory,
    plot_vector_field,
    setup_3d_plot,
)


def enable_x64():
    import warnings

    warnings.warn(
        "enable_x64 is deprecated. HAM now uses a configurable DTYPE defaulting to float32.", stacklevel=2
    )
    # config.update("jax_enable_x64", True)
