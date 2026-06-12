import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.metric import AsymmetricMetric

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# =============================================================================
# EXPERIMENT RESULT CONTAINER
# =============================================================================


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    name: str
    category: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, base_dir: str = None):
        """Save results to disk."""
        if base_dir is None:
            base_dir = RESULTS_DIR

        exp_dir = os.path.join(base_dir, self.category, self.name)
        os.makedirs(exp_dir, exist_ok=True)

        summary = {
            "name": self.name,
            "category": self.category,
            "success": self.success,
            "metrics": {
                k: float(v) if isinstance(v, (np.floating, float, jax.Array)) else v
                for k, v in self.metrics.items()
            },
            "metadata": self.metadata,
            "runtime_seconds": self.runtime_seconds,
            "timestamp": self.timestamp,
        }
        with open(os.path.join(exp_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if self.arrays:
            np.savez_compressed(os.path.join(exp_dir, "arrays.npz"), **self.arrays)

        return exp_dir

    @classmethod
    def load(cls, exp_dir: str) -> "ExperimentResult":
        """Load results from disk."""
        with open(os.path.join(exp_dir, "summary.json")) as f:
            summary = json.load(f)

        arrays = {}
        arrays_path = os.path.join(exp_dir, "arrays.npz")
        if os.path.exists(arrays_path):
            with np.load(arrays_path) as data:
                arrays = dict(data)

        return cls(
            name=summary["name"],
            category=summary["category"],
            success=summary["success"],
            metrics=summary["metrics"],
            arrays=arrays,
            metadata=summary.get("metadata", {}),
            runtime_seconds=summary.get("runtime_seconds", 0),
            timestamp=summary.get("timestamp", ""),
        )


# =============================================================================
# EXPERIMENT BASE CLASS
# =============================================================================


class Experiment(ABC):
    """Base class for all experiments."""

    name: str = "base_experiment"
    category: str = "uncategorized"
    description: str = "Base experiment class"

    def __init__(self):
        self.result = None

    def setup(self) -> None:
        """Setup experiment (override if needed)."""
        pass

    @abstractmethod
    def run(self) -> ExperimentResult:
        """Run the experiment (must override)."""
        raise NotImplementedError

    @abstractmethod
    def visualize(self, save_path: Optional[str] = None) -> matplotlib.figure.Figure:
        """Create visualization (must override)."""
        raise NotImplementedError

    def execute(
        self, save: bool = True, visualize: bool = True, verbose: bool = True
    ) -> ExperimentResult:
        """Full execution: setup, run, save, visualize."""
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Experiment: {self.name}")
            print(f"Category: {self.category}")
            print(f"{'=' * 60}")

        self.setup()

        start_time = time.time()
        self.result = self.run()
        self.result.runtime_seconds = time.time() - start_time

        if verbose:
            print(f"\nCompleted in {self.result.runtime_seconds:.2f}s")
            print(f"Success: {self.result.success}")

            if self.result.metrics:
                print("\nMetrics:")
                for k, v in self.result.metrics.items():
                    if isinstance(v, float) or isinstance(v, jax.Array):
                        print(f"  {k}: {float(v):.6f}")
                    else:
                        print(f"  {k}: {v}")

        if save:
            exp_dir = self.result.save()
            if verbose:
                print(f"\nResults saved to: {exp_dir}")

        if visualize:
            fig_dir = os.path.join(FIGURES_DIR, self.category)
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, f"{self.name}.pdf")
            try:
                fig = self.visualize(save_path=fig_path)
                if fig is not None:
                    plt.close(fig)
                    if verbose:
                        print(f"Figure saved to: {fig_path}")
            except Exception as e:
                print(f"Visualization failed: {e}")
                import traceback

                traceback.print_exc()

        return self.result


# =============================================================================
# SYNTHETIC METRIC
# =============================================================================


def _bilinear_interp_point(grid: jax.Array, pt_pixel: jax.Array) -> jax.Array:
    """Interpolates a grid (C, H, W) at a continuous pixel coordinate (row, col) = pt_pixel."""
    C, H, W = grid.shape
    py, px = pt_pixel[0], pt_pixel[1]  # row, col
    px = jnp.clip(px, 0.0, W - 1.001)
    py = jnp.clip(py, 0.0, H - 1.001)
    x0 = jnp.floor(px).astype(jnp.int32)
    y0 = jnp.floor(py).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, W - 1)
    y1 = jnp.minimum(y0 + 1, H - 1)
    fx = px - x0
    fy = py - y0

    return (
        grid[:, y0, x0] * (1.0 - fx) * (1.0 - fy)
        + grid[:, y0, x1] * fx * (1.0 - fy)
        + grid[:, y1, x0] * (1.0 - fx) * fy
        + grid[:, y1, x1] * fx * fy
    )


class SyntheticZermeloMetric(AsymmetricMetric):
    """
    A continuous Randers metric defined by grid parameters for H and W.
    It expects spatial coordinates to be within [0, H-1] x [0, W-1].
    """

    H_grid: jax.Array  # Shape: (3, H, W) -> (h11, h12, h22)
    W_grid: jax.Array  # Shape: (2, H, W) -> (w1, w2)
    lam_grid: jax.Array  # Shape: (1, H, W) -> scalar speed lambda

    def __init__(
        self, H_grid: jax.Array, W_grid: jax.Array, lam_grid: Optional[jax.Array] = None
    ):
        super().__init__(manifold=EuclideanSpace(2))
        self.H_grid = H_grid
        self.W_grid = W_grid
        if lam_grid is None:
            self.lam_grid = jnp.ones((1, H_grid.shape[1], H_grid.shape[2]))
        else:
            self.lam_grid = lam_grid

    def zermelo_data(self, z: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Interpolates the metric at the spatial coordinate z.
        z is expected to be [row, col] continuous coordinates.
        """
        h_vals = _bilinear_interp_point(self.H_grid, z)
        w_vals = _bilinear_interp_point(self.W_grid, z)
        lam_val = _bilinear_interp_point(self.lam_grid, z)

        H_mat = jnp.array([[h_vals[0], h_vals[1]], [h_vals[1], h_vals[2]]])
        return H_mat, w_vals, lam_val[0]

    def metric_fn(self, z: jax.Array, v: jax.Array) -> jax.Array:
        H, W, lam = self.zermelo_data(z)

        # Zermelo/Randers energy functional
        # F(z, v) = sqrt( (lam*v + W)^T H (lam*v + W) ) / lam - W^T H v
        # Wait, the standard Zermelo navigation with speed lambda:
        # F = sqrt( v^T H v / lambda + (W^T H v)^2 / lambda^2 ) - W^T H v / lambda
        # Let's align with the AVBD solver which expects standard Riemannian + drift if Asymmetric.
        # Following HAM's `spec/MATH_SPEC.md` or typical Randers formulation:
        v_h_v = jnp.dot(v, jnp.dot(H, v))
        W_h_W = jnp.dot(W, jnp.dot(H, W))
        W_h_v = jnp.dot(W, jnp.dot(H, v))

        lam_safe = jnp.maximum(lam, 1e-4)
        alpha_sq = v_h_v / lam_safe

        # To make it equivalent to Gahtan's G and B:
        # F(v) = sqrt( v^T G v ) + B^T v
        B = -jnp.dot(H, W) / lam_safe
        HW = jnp.dot(H, W)
        G = (H + jnp.outer(HW, HW) / lam_safe) / lam_safe

        v_G_v = jnp.dot(v, jnp.dot(G, v))
        B_dot_v = jnp.dot(B, v)

        return jnp.sqrt(jnp.maximum(v_G_v, 1e-8)) + B_dot_v


# =============================================================================
# GROUND TRUTH COMPUTATION
# =============================================================================


def euclidean_distance_field(M: int, N: int, source_i: int, source_j: int) -> jax.Array:
    """Compute exact Euclidean distance from source point."""
    I, J = jnp.meshgrid(jnp.arange(M), jnp.arange(N), indexing="ij")
    return jnp.sqrt((I - source_i) ** 2 + (J - source_j) ** 2)


def anisotropic_distance_field(
    M: int,
    N: int,
    source_i: int,
    source_j: int,
    lambda1: float,
    lambda2: float,
    theta: float = 0.0,
) -> jax.Array:
    """Compute exact distance for anisotropic metric."""
    Y, X = jnp.meshgrid(
        jnp.arange(M) - source_i, jnp.arange(N) - source_j, indexing="ij"
    )

    c, s = jnp.cos(theta), jnp.sin(theta)
    X_rot = c * X + s * Y
    Y_rot = -s * X + c * Y

    return jnp.sqrt(lambda1 * X_rot**2 + lambda2 * Y_rot**2)


def create_metric_from_eigenvalues(
    M: int, N: int, lambda1: float, lambda2: float, theta: float = 0.0
) -> jax.Array:
    """Create metric tensor G = R^T diag(λ1, λ2) R."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    g11 = lambda1 * c**2 + lambda2 * s**2
    g12 = (lambda1 - lambda2) * c * s
    g22 = lambda1 * s**2 + lambda2 * c**2

    G = jnp.zeros((3, M, N))
    G = G.at[0].set(g11)
    G = G.at[1].set(g12)
    G = G.at[2].set(g22)
    return G


# =============================================================================
# ERROR METRICS
# =============================================================================


def compute_errors(
    T_computed: jax.Array, T_exact: jax.Array, mask: Optional[jax.Array] = None
) -> Dict[str, float]:
    """Compute error metrics between computed and exact solutions."""
    if mask is None:
        mask = jnp.ones_like(T_computed, dtype=bool)

    valid = mask & jnp.isfinite(T_computed) & jnp.isfinite(T_exact)

    def calc_errors():
        diff = jnp.where(valid, T_computed - T_exact, 0.0)
        T_ex = jnp.where(valid, T_exact, 0.0)

        l1 = jnp.mean(jnp.abs(diff), where=valid)
        l2 = jnp.sqrt(jnp.mean(diff**2, where=valid))
        linf = jnp.max(jnp.abs(diff), where=valid, initial=0.0)

        T_scale = jnp.max(jnp.abs(T_ex), where=valid, initial=1e-10)
        rel_l2 = l2 / T_scale
        rel_linf = linf / T_scale
        return {
            "l1": float(l1),
            "l2": float(l2),
            "linf": float(linf),
            "rel_l2": float(rel_l2),
            "rel_linf": float(rel_linf),
        }

    def nan_errors():
        return {
            "l1": float("nan"),
            "l2": float("nan"),
            "linf": float("nan"),
            "rel_l2": float("nan"),
            "rel_linf": float("nan"),
        }

    if bool(jnp.sum(valid) > 0):
        return calc_errors()
    else:
        return nan_errors()


def compute_convergence_rate(
    h_values: List[float], errors: List[float]
) -> Tuple[float, float]:
    """Compute convergence rate via log-log linear regression."""
    log_h = np.log(np.array(h_values))
    log_e = np.log(np.array(errors))

    A = np.vstack([log_h, np.ones_like(log_h)]).T
    result = np.linalg.lstsq(A, log_e, rcond=None)
    rate, const = result[0]

    residuals = log_e - (rate * log_h + const)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return float(rate), float(r_squared)


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================


def plot_arrival_time(
    T: np.ndarray,
    ax: plt.Axes = None,
    title: str = "Arrival Time",
    contour_levels: int = 20,
    cmap: str = "viridis",
) -> plt.Axes:
    """Plot arrival time field with contours."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    T_np = np.asarray(T)
    T_np = np.where(np.isinf(T_np), np.nan, T_np)

    im = ax.imshow(T_np, origin="upper", cmap=cmap)
    plt.colorbar(im, ax=ax, label="T(x)")

    if np.nanmax(T_np) > np.nanmin(T_np):
        levels = np.linspace(np.nanmin(T_np), np.nanmax(T_np), contour_levels)
        ax.contour(T_np, levels=levels, colors="white", linewidths=0.5, alpha=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    return ax


def plot_error_map(
    error: np.ndarray,
    ax: plt.Axes = None,
    title: str = "Error",
    cmap: str = "RdBu_r",
    symmetric: bool = True,
) -> plt.Axes:
    """Plot error field."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    err_np = np.asarray(error)
    vmax = np.nanmax(np.abs(err_np)) if symmetric else np.nanmax(err_np)
    vmin = -vmax if symmetric else np.nanmin(err_np)

    im = ax.imshow(err_np, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Error")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    return ax


def plot_metric_ellipses(
    G: np.ndarray, ax: plt.Axes, step: int = 10, scale: float = 0.4, color: str = "blue"
) -> None:
    """Overlay metric tensor ellipses."""
    G_np = np.asarray(G)
    if G_np.ndim == 2:
        return
    M, N = G_np.shape[1], G_np.shape[2]

    for i in range(step // 2, M, step):
        for j in range(step // 2, N, step):
            g = G_np[:, i, j]
            G_mat = np.array([[g[0], g[1]], [g[1], g[2]]])
            try:
                G_inv = np.linalg.inv(G_mat)
                vals, vecs = np.linalg.eigh(G_inv)
                r = np.sqrt(np.maximum(vals, 1e-10))
                rx, ry = scale * r[1], scale * r[0]
                major_vec = vecs[:, 1] if r[1] >= r[0] else vecs[:, 0]
                angle = np.degrees(np.arctan2(major_vec[1], major_vec[0]))
                ellipse = Ellipse(
                    (j, i),
                    width=2 * rx,
                    height=2 * ry,
                    angle=angle,
                    fill=False,
                    edgecolor=color,
                    linewidth=0.5,
                    alpha=0.6,
                )
                ax.add_patch(ellipse)
            except:
                pass


def plot_drift_field(
    W: np.ndarray, ax: plt.Axes, step: int = 10, color: str = "red", scale: float = None
) -> None:
    """Overlay drift vectors (W_grid)."""
    W_np = np.asarray(W)
    M, N = W_np.shape[1], W_np.shape[2]
    X, Y = np.meshgrid(np.arange(N), np.arange(M))
    ax.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        W_np[1, ::step, ::step],
        W_np[0, ::step, ::step],
        color=color,
        alpha=0.7,
        scale=scale,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_interior_mask(
    M: int,
    N: int,
    boundary_width: int = 3,
    source_mask: Optional[jax.Array] = None,
    source_radius: int = 3,
) -> jax.Array:
    """Create mask for interior points."""
    mask = jnp.zeros((M, N), dtype=bool)
    mask = mask.at[boundary_width:-boundary_width, boundary_width:-boundary_width].set(
        True
    )

    if source_mask is not None:
        I, J = jnp.meshgrid(jnp.arange(M), jnp.arange(N), indexing="ij")

        # Function to check distance to ANY source
        def dist_to_sources(i, j):
            dists = jnp.where(
                source_mask, jnp.maximum(jnp.abs(I - i), jnp.abs(J - j)), 1000
            )
            return jnp.min(dists)

        dist_grid = jax.vmap(jax.vmap(dist_to_sources))(I, J)
        mask = jnp.where(dist_grid <= source_radius, False, mask)

    return mask


def create_sparse_observation_mask(
    M: int,
    N: int,
    fraction: float,
    source_mask: Optional[jax.Array] = None,
    seed: int = 42,
) -> jax.Array:
    """Create sparse random observation mask."""
    key = jax.random.PRNGKey(seed)

    interior = get_interior_mask(M, N, 3, source_mask)
    probs = jax.random.uniform(key, shape=(M, N))

    # We want exactly `fraction` of the interior points, or approximately.
    # To do it exactly in JAX:
    valid_scores = jnp.where(interior, probs, -1.0)
    n_valid = jnp.sum(interior)
    n_obs = jnp.maximum(1, jnp.round(n_valid * fraction).astype(jnp.int32))

    # Threshold at the (N - n_obs)-th smallest valid score
    flat_scores = valid_scores.flatten()
    sorted_scores = jnp.sort(flat_scores)
    threshold = sorted_scores[-(n_obs + 1)]  # Values greater than this are selected

    obs_mask = interior & (valid_scores > threshold)
    return obs_mask


# =============================================================================
# EXPERIMENT REGISTRY
# =============================================================================

EXPERIMENT_REGISTRY: Dict[str, type] = {}


def register_experiment(cls):
    """Decorator to register an experiment class."""
    key = f"{cls.category}_{cls.name}"
    EXPERIMENT_REGISTRY[key] = cls
    return cls
