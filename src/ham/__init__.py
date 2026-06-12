"""
HAM — Differentiable Finsler Geometry in JAX
=============================================

A library for learnable Finsler metrics, geodesic solvers, and parallel
transport built on JAX and Equinox.

Subpackages
-----------
geometry : Manifolds, metrics (Euclidean/Riemannian/Randers), transport
models   : Neural and pullback metric implementations
solvers  : Geodesic BVP (AVBD) and IVP (ExponentialMap) solvers
training : Multi-phase training pipeline and modular losses
bio      : Single-cell biology wrappers (GeometricVAE, BioDataset)
nn       : Neural network building blocks (VectorField, PSDMatrixField)
"""

__version__ = "1.1.0"

# Core geometry
from ham.geometry import (
    DiscreteRanders,
    Euclidean,
    EuclideanSpace,
    Hyperboloid,
    Paraboloid,
    Randers,
    Riemannian,
    Sphere,
    Torus,
)
from ham.geometry.manifold import Manifold
from ham.geometry.metric import FinslerMetric
from ham.geometry.transport import BerwaldConnection

# Solvers
from ham.solvers.avbd import AVBDSolver, Trajectory
from ham.solvers.geodesic import ExponentialMap

__all__ = [
    # Solvers
    "AVBDSolver",
    "BerwaldConnection",
    "DiscreteRanders",
    "Euclidean",
    "EuclideanSpace",
    "ExponentialMap",
    "FinslerMetric",
    "Hyperboloid",
    # Geometry
    "Manifold",
    "Paraboloid",
    "Randers",
    "Riemannian",
    "Sphere",
    "Torus",
    "Trajectory",
    "__version__",
]
