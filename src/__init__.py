__version__ = "0.1.0"

from jax import config
from . import geometry
from . import solvers
from . import bio

from .geometry import Manifold, FinslerMetric
from .geometry.surfaces import Sphere
from .geometry.zoo import Euclidean, Riemannian, Randers, DiscreteRanders
from .geometry.mesh import TriangularMesh
from .solvers import AVBDSolver, ExponentialMap
from .bio import GeometricVAE
from .nn import VectorField, PSDMatrixField
from .vis import setup_3d_plot, plot_sphere, plot_trajectory, generate_icosphere, plot_indicatrices, plot_vector_field
from .utils import safe_norm

def enable_x64():
    config.update("jax_enable_x64", True)