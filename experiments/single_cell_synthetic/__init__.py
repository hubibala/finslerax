"""HAM Synthetic Single-Cell Waddington Geometry.

A ground-truth precursor to the real Weinreb Randers-metric experiment: a
branching Waddington landscape with a *tunable* non-conservative flux (the ``κ``
axis), simulated destructively into scRNA-seq-realistic counts with clonal
barcodes and noisy RNA velocity, on which HAM's learned asymmetric Randers
metric pipeline is run and scored against known truth.

Framing, claims (H1–H4), staged study (A–D), baselines, and honest caveats are
documented in ``README.md`` and ``spec/single_cell_synthetic_PLAN.md``.
"""

from .dataset import SingleCellDataset
from .drift import HodgeField, SparseVFC, helmholtz_hodge_rbf
from .generator import GeneratorConfig, RandomDecoder, generate
from .landscape import Landscape, least_action_path, om_action
from .metric import (
    build_randers,
    build_true_metric,
    conformal_sea,
    navigable_wind_scale,
    pullback_sea,
)
from .solvers import RandersFlowMatching, exact_geodesic, train_rfm

__all__ = [
    "GeneratorConfig",
    "HodgeField",
    "Landscape",
    "RandersFlowMatching",
    "RandomDecoder",
    "SingleCellDataset",
    "SparseVFC",
    "build_randers",
    "build_true_metric",
    "conformal_sea",
    "exact_geodesic",
    "generate",
    "helmholtz_hodge_rbf",
    "least_action_path",
    "navigable_wind_scale",
    "om_action",
    "pullback_sea",
    "train_rfm",
]
