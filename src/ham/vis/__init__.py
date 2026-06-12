from .vis import setup_3d_plot, plot_vector_field, plot_sphere, plot_trajectory, plot_indicatrices, generate_icosphere
from .hyperbolic import plot_poincare_disk
from .isosurface import differentiable_marching_cubes, compute_analytical_normals

__all__ = [
    "setup_3d_plot",
    "plot_vector_field",
    "plot_sphere",
    "plot_trajectory",
    "plot_indicatrices",
    "generate_icosphere",
    "plot_poincare_disk",
    "differentiable_marching_cubes",
    "compute_analytical_normals",
]