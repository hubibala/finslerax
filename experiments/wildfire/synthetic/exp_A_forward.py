import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import time

from experiments.wildfire.synthetic.experiment_base import (
    Experiment, ExperimentResult, register_experiment,
    euclidean_distance_field, anisotropic_distance_field, create_metric_from_eigenvalues,
    compute_errors, compute_convergence_rate,
    plot_arrival_time, plot_error_map, plot_metric_ellipses, plot_drift_field,
    SyntheticZermeloMetric
)
from experiments.wildfire.synthetic.metric_recovery import eikonal_to_zermelo
from ham.solvers.eikonal import EikonalSolver


def get_solver():
    return EikonalSolver(max_iters=100, tol=1e-6)


# =============================================================================
# A1: ISOTROPIC CONVERGENCE
# =============================================================================
@register_experiment
class A1_IsotropicConvergence(Experiment):
    """Verify O(h) convergence on isotropic problem."""

    name = "A1_isotropic_convergence"
    category = "A_forward_solver"
    description = "Verify O(h) convergence rate for isotropic eikonal equation"

    def __init__(self, grid_sizes: List[int] = None):
        super().__init__()
        self.grid_sizes = grid_sizes or [25, 50, 100, 200, 400]

    def run(self) -> ExperimentResult:
        h_values, errors_l2, errors_linf, rel_errors = [], [], [], []

        # Fixed physical domain [0, 1] x [0, 1]
        L = 1.0
        solver = EikonalSolver(max_iters=50, tol=1e-8)

        for N in self.grid_sizes:
            print(f"  Grid size: {N}x{N}")
            h = L / max(1, N - 1)

            G = jnp.zeros((3, N, N))
            G = G.at[0].set(1.0)
            G = G.at[2].set(1.0)
            B = jnp.zeros((2, N, N))

            source_i, source_j = N // 2, N // 2
            # Source physical coordinates
            source_x = source_i * h
            source_y = source_j * h
            source_coords = jnp.array([source_x, source_y])

            H, W = eikonal_to_zermelo(G, B)
            metric = SyntheticZermeloMetric(H, W)

            T_physical, X, Y = solver.solve(metric, source_coords, 
                                            grid_extent=(0, L, 0, L), 
                                            grid_shape=(N, N))

            T_exact = jnp.sqrt((X - source_x) ** 2 + (Y - source_y) ** 2)

            interior_mask = jnp.zeros((N, N), dtype=bool)
            interior_mask = interior_mask.at[2:-2, 2:-2].set(True)
            
            # Mask out 5x5 area around source
            I, J = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
            near_source = (jnp.abs(I - source_i) <= 2) & (jnp.abs(J - source_j) <= 2)
            interior_mask = interior_mask & (~near_source)

            diff = jnp.where(interior_mask, T_physical - T_exact, 0.0)

            l2 = jnp.sqrt(jnp.mean(diff ** 2, where=interior_mask))
            linf = jnp.max(jnp.abs(diff), where=interior_mask, initial=0.0)
            max_exact = jnp.max(T_exact, where=interior_mask, initial=1e-5)
            rel_l2 = l2 / max_exact

            h_values.append(float(h))
            errors_l2.append(float(l2))
            errors_linf.append(float(linf))
            rel_errors.append(float(rel_l2))
            print(f"    h={h:.4f}, L2={l2:.6f}, Linf={linf:.6f}, rel={rel_l2:.6f}")

        rate_l2, r2_l2 = compute_convergence_rate(h_values, errors_l2)
        rate_linf, r2_linf = compute_convergence_rate(h_values, errors_linf)
        print(f"\n  L2 rate: {rate_l2:.2f}, Linf rate: {rate_linf:.2f}")

        self._data = {'h': h_values, 'e_l2': errors_l2, 'e_linf': errors_linf, 'rel': rel_errors}

        return ExperimentResult(
            name=self.name, category=self.category,
            success=(0.65 <= rate_l2 <= 1.5),
            metrics={'rate_l2': rate_l2, 'rate_linf': rate_linf, 'r2_l2': r2_l2,
                     'r2_linf': r2_linf, 'max_rel_error': max(rel_errors)},
            arrays={'h_values': np.array(h_values), 'errors_l2': np.array(errors_l2),
                    'errors_linf': np.array(errors_linf), 'rel_errors': np.array(rel_errors)},
            metadata={'grid_sizes': self.grid_sizes}
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        h, e_l2, e_linf = self._data['h'], self._data['e_l2'], self._data['e_linf']

        ax.loglog(h, e_l2, 'o-', label=f'L² (rate={self.result.metrics["rate_l2"]:.2f})', markersize=8)
        ax.loglog(h, e_linf, 's-', label=f'L∞ (rate={self.result.metrics["rate_linf"]:.2f})', markersize=8)

        h_ref = np.array([h[0], h[-1]])
        ax.loglog(h_ref, e_l2[0] * (h_ref / h[0]), 'k--', alpha=0.5, label='O(h)')

        ax.set_xlabel('Grid spacing h')
        ax.set_ylabel('Error (physical units)')
        ax.set_title('A1: Isotropic Convergence Study')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A2: UNIFORM ANISOTROPIC
# =============================================================================
@register_experiment
class A2_UniformAnisotropic(Experiment):
    """Verify handling of uniform anisotropic metric."""
    
    name = "A2_uniform_anisotropic"
    category = "A_forward_solver"
    description = "Verify elliptical wavefronts with diagonal anisotropic metric"
    
    def __init__(self, N: int = 200, a: float = 2.0, b: float = 0.5):
        super().__init__()
        self.N, self.a, self.b = N, a, b
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(self.a**2)
        G = G.at[2].set(self.b**2)
        B = jnp.zeros((2, N, N))
        
        source_i, source_j = N // 2, N // 2
        source_coords = jnp.array([source_i, source_j], dtype=jnp.float32)
        
        H, W = eikonal_to_zermelo(G, B)
        metric = SyntheticZermeloMetric(H, W)
        solver = get_solver()
        
        T, _, _ = solver.solve(metric, source_coords, grid_extent=(0, N-1, 0, N-1), grid_shape=(N, N))
        T_exact = anisotropic_distance_field(N, N, source_i, source_j, self.a**2, self.b**2, 0.0)
        
        interior_mask = jnp.zeros((N, N), dtype=bool)
        interior_mask = interior_mask.at[5:-5, 5:-5].set(True)
        I, J = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
        near_source = (jnp.abs(I - source_i) <= 3) & (jnp.abs(J - source_j) <= 3)
        interior_mask = interior_mask & (~near_source)
        
        errs = compute_errors(T, T_exact, interior_mask)
        print(f"  Relative L2 error: {errs['rel_l2']:.6f}")
        
        self.T, self.T_exact, self.G = T, T_exact, G
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=errs['rel_l2'] < 0.02,
            metrics=errs,
            arrays={'T': np.array(T), 'T_exact': np.array(T_exact)},
            metadata={'N': N, 'a': self.a, 'b': self.b}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Computed T')
        plot_metric_ellipses(np.array(self.G), axes[0], step=20, scale=3.0)
        plot_arrival_time(self.T_exact, ax=axes[1], title='Exact T')
        plot_error_map(self.T - self.T_exact, ax=axes[2], 
                       title=f'Error (rel L2={self.result.metrics["rel_l2"]:.4f})')
        fig.suptitle(f'A2: Uniform Anisotropic (a={self.a}, b={self.b})', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A3: ROTATED ANISOTROPIC  
# =============================================================================
@register_experiment
class A3_RotatedAnisotropic(Experiment):
    """Verify handling of rotated anisotropic metric with nonzero g12."""
    
    name = "A3_rotated_anisotropic"
    category = "A_forward_solver"
    description = "Verify correct handling of rotated metric (nonzero g12)"
    
    def __init__(self, N: int = 200, lambda1: float = 4.0, lambda2: float = 0.25, theta_deg: float = 45.0):
        super().__init__()
        self.N, self.lambda1, self.lambda2 = N, lambda1, lambda2
        self.theta = np.deg2rad(theta_deg)
        self.theta_deg = theta_deg
        
    def run(self) -> ExperimentResult:
        N = self.N
        G = create_metric_from_eigenvalues(N, N, self.lambda1, self.lambda2, self.theta)
        B = jnp.zeros((2, N, N))
        
        source_i, source_j = N // 2, N // 2
        source_coords = jnp.array([source_i, source_j], dtype=jnp.float32)
        
        H, W = eikonal_to_zermelo(G, B)
        metric = SyntheticZermeloMetric(H, W)
        solver = get_solver()
        
        T, _, _ = solver.solve(metric, source_coords, grid_extent=(0, N-1, 0, N-1), grid_shape=(N, N))
        T_exact = anisotropic_distance_field(N, N, source_i, source_j, self.lambda1, self.lambda2, self.theta)
        
        interior_mask = jnp.zeros((N, N), dtype=bool)
        interior_mask = interior_mask.at[5:-5, 5:-5].set(True)
        I, J = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
        near_source = (jnp.abs(I - source_i) <= 3) & (jnp.abs(J - source_j) <= 3)
        interior_mask = interior_mask & (~near_source)
        
        errs = compute_errors(T, T_exact, interior_mask)
        print(f"  Relative L2 error: {errs['rel_l2']:.6f}")
        print(f"  g12 at center: {G[1, N//2, N//2]:.4f}")
        
        self.T, self.T_exact, self.G = T, T_exact, G
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=errs['rel_l2'] < 0.02,
            metrics=errs,
            arrays={'T': np.array(T), 'T_exact': np.array(T_exact)},
            metadata={'theta_deg': self.theta_deg, 'lambda1': self.lambda1, 'lambda2': self.lambda2}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Computed T')
        plot_metric_ellipses(np.array(self.G), axes[0], step=20, scale=3.0)
        plot_arrival_time(self.T_exact, ax=axes[1], title='Exact T')
        plot_error_map(self.T - self.T_exact, ax=axes[2],
                       title=f'Error (rel L2={self.result.metrics["rel_l2"]:.4f})')
        fig.suptitle(f'A3: Rotated Anisotropic (θ={self.theta_deg}°)', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A4: CONSTANT DRIFT
# =============================================================================
@register_experiment
class A4_ConstantDrift(Experiment):
    """Verify constant drift produces asymmetric wavefronts."""
    
    name = "A4_constant_drift"
    category = "A_forward_solver"
    description = "Verify drift field produces asymmetric wavefronts (Randers)"
    
    def __init__(self, N: int = 200, drift_magnitude: float = 0.3):
        super().__init__()
        self.N, self.drift_magnitude = N, drift_magnitude
        
    def run(self) -> ExperimentResult:
        N, b = self.N, self.drift_magnitude
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0)
        G = G.at[2].set(1.0)
        B = jnp.zeros((2, N, N))
        B = B.at[0].set(b)
        
        source_i, source_j = N // 2, N // 2
        source_coords = jnp.array([source_i, source_j], dtype=jnp.float32)
        
        H, W = eikonal_to_zermelo(G, B)
        metric = SyntheticZermeloMetric(H, W)
        solver = get_solver()
        
        T, _, _ = solver.solve(metric, source_coords, grid_extent=(0, N-1, 0, N-1), grid_shape=(N, N))
        
        # Check asymmetry
        T_left = float(T[source_i, source_j - 30])
        T_right = float(T[source_i, source_j + 30])
        asymmetry = (T_left - T_right) / (T_left + T_right)
        
        print(f"  T at x=-30: {T_left:.4f}, T at x=+30: {T_right:.4f}")
        print(f"  Asymmetry: {asymmetry:.4f}")
        
        self.T, self.B = T, B
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=asymmetry > 0.1,
            metrics={'T_left': T_left, 'T_right': T_right, 'asymmetry': asymmetry},
            arrays={'T': np.array(T)},
            metadata={'drift_magnitude': b}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Arrival Time with Drift')
        plot_drift_field(np.array(self.B), axes[0], step=15, scale=20)
        
        N = self.N
        profile = np.array(self.T[N//2, :])
        x = np.arange(N) - N//2
        axes[1].plot(x, profile)
        axes[1].axvline(0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('x offset')
        axes[1].set_ylabel('Arrival time')
        axes[1].set_title(f'Profile (b={self.drift_magnitude})')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('A4: Constant Drift Field', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A5: COMBINED ANISOTROPIC + DRIFT
# =============================================================================
@register_experiment
class A5_CombinedAnisotropicDrift(Experiment):
    """Full Randers-Finsler: anisotropic metric + drift field."""

    name = "A5_combined_anisotropic_drift"
    category = "A_forward_solver"
    description = "Verify full Randers-Finsler equation"

    def __init__(self, N: int = 100, N_fine: int = 400):
        super().__init__()
        self.N, self.N_fine = N, N_fine

    def run(self) -> ExperimentResult:
        N, N_fine = self.N, self.N_fine
        lambda1, lambda2, theta = 2.0, 0.5, np.deg2rad(30)
        drift_x, drift_y = 0.2, 0.1

        # Coarse solution
        G = create_metric_from_eigenvalues(N, N, lambda1, lambda2, theta)
        B = jnp.zeros((2, N, N))
        B = B.at[0].set(drift_x)
        B = B.at[1].set(drift_y)
        source_coords = jnp.array([N // 2, N // 2], dtype=jnp.float32)

        H, W = eikonal_to_zermelo(G, B)
        metric = SyntheticZermeloMetric(H, W)
        solver = get_solver()
        
        T, _, _ = solver.solve(metric, source_coords, grid_extent=(0, N-1, 0, N-1), grid_shape=(N, N))

        # Fine solution
        G_fine = create_metric_from_eigenvalues(N_fine, N_fine, lambda1, lambda2, theta)
        B_fine = jnp.zeros((2, N_fine, N_fine))
        B_fine = B_fine.at[0].set(drift_x)
        B_fine = B_fine.at[1].set(drift_y)
        source_coords_fine = jnp.array([N_fine // 2, N_fine // 2], dtype=jnp.float32)

        H_fine, W_fine = eikonal_to_zermelo(G_fine, B_fine)
        metric_fine = SyntheticZermeloMetric(H_fine, W_fine)
        
        T_fine, _, _ = solver.solve(metric_fine, source_coords_fine, grid_extent=(0, N_fine-1, 0, N_fine-1), grid_shape=(N_fine, N_fine))

        scale = N_fine // N
        T_fine_rescaled = T_fine[::scale, ::scale] / scale

        interior = jnp.zeros((N, N), dtype=bool)
        interior = interior.at[5:-5, 5:-5].set(True)
        I, J = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
        near_source = (jnp.abs(I - N//2) <= 3) & (jnp.abs(J - N//2) <= 3)
        interior = interior & (~near_source)

        diff = jnp.where(interior, T - T_fine_rescaled, 0.0)
        rel_l2 = float(jnp.sqrt(jnp.mean(diff ** 2, where=interior)) / jnp.mean(jnp.abs(T), where=interior))
        print(f"  Relative L2 vs fine: {rel_l2:.6f}")

        self.T, self.T_fine_rescaled = T, T_fine_rescaled
        self.G, self.B = G, B

        return ExperimentResult(
            name=self.name, category=self.category,
            success=rel_l2 < 0.10,
            metrics={'rel_l2': rel_l2},
            arrays={'T': np.array(T), 'T_fine': np.array(T_fine_rescaled)},
            metadata={'N': N, 'N_fine': N_fine}
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Coarse')
        plot_metric_ellipses(np.array(self.G), axes[0], step=10, scale=1.5)
        plot_drift_field(np.array(self.B), axes[0], step=10, scale=30)
        plot_arrival_time(self.T_fine_rescaled, ax=axes[1], title='Fine (rescaled)')
        plot_error_map(self.T - self.T_fine_rescaled, ax=axes[2],
                       title=f'Error (rel L2={self.result.metrics["rel_l2"]:.4f})')
        fig.suptitle('A5: Anisotropic + Drift', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A6: PIECEWISE METRIC
# =============================================================================
@register_experiment
class A6_PiecewiseMetric(Experiment):
    """Test wavefront refraction at metric discontinuity."""
    
    name = "A6_piecewise_metric"
    category = "A_forward_solver"
    description = "Verify wavefront refraction at interface"
    
    def __init__(self, N: int = 200):
        super().__init__()
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0)
        G = G.at[2].set(1.0)
        G = G.at[0, :, N//2:].set(4.0)  # slow region
        G = G.at[2, :, N//2:].set(4.0)
        
        B = jnp.zeros((2, N, N))
        
        source_i, source_j = N // 2, N // 4
        source_coords = jnp.array([source_i, source_j], dtype=jnp.float32)
        
        H, W = eikonal_to_zermelo(G, B)
        metric = SyntheticZermeloMetric(H, W)
        solver = get_solver()
        
        T, _, _ = solver.solve(metric, source_coords, grid_extent=(0, N-1, 0, N-1), grid_shape=(N, N))
        
        T_at_interface = float(T[N//2, N//2])
        T_past = float(T[N//2, N//2 + 10])
        
        speed_in_slow = (T_past - T_at_interface) / 10
        print(f"  Expected speed in slow region: 2.0, got: {speed_in_slow:.2f}")
        
        self.T, self.G = T, G
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=1.5 < speed_in_slow < 2.5,
            metrics={'speed_in_slow': speed_in_slow},
            arrays={'T': np.array(T), 'G11': np.array(G[0])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Arrival Time')
        axes[0].axvline(self.N//2, color='r', linestyle='--', linewidth=2, label='Interface')
        axes[0].legend()
        
        im = axes[1].imshow(np.array(self.G[0]), origin='upper', cmap='RdYlBu_r')
        plt.colorbar(im, ax=axes[1], label='g₁₁')
        axes[1].axvline(self.N//2, color='k', linestyle='--', linewidth=2)
        axes[1].set_title('Metric g₁₁')
        
        fig.suptitle('A6: Piecewise Metric', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A7: SPATIALLY VARYING METRIC
# =============================================================================
@register_experiment
class A7_SpatiallyVaryingMetric(Experiment):
    """Test smoothly varying metric field."""

    name = "A7_spatially_varying_metric"
    category = "A_forward_solver"
    description = "Verify solver handles smooth metric variations"

    def __init__(self, N: int = 200):
        super().__init__()
        self.N = N

    def run(self) -> ExperimentResult:
        N = self.N

        j_coords = jnp.arange(N, dtype=jnp.float32)
        variation = 1.0 + 0.5 * jnp.sin(2 * np.pi * j_coords / N)
        variation_grid = jnp.tile(variation, (N, 1))

        G = jnp.zeros((3, N, N))
        G = G.at[0].set(variation_grid)
        G = G.at[2].set(variation_grid)

        B = jnp.zeros((2, N, N))

        source_coords = jnp.array([N // 2, N // 2], dtype=jnp.float32)

        H, W = eikonal_to_zermelo(G, B)
        metric = SyntheticZermeloMetric(H, W)
        solver = get_solver()

        T, _, _ = solver.solve(metric, source_coords, grid_extent=(0, N-1, 0, N-1), grid_shape=(N, N))

        T_np = np.array(T)
        grad_i = np.zeros_like(T_np)
        grad_j = np.zeros_like(T_np)
        grad_i[1:-1, :] = (T_np[2:, :] - T_np[:-2, :]) / 2
        grad_j[:, 1:-1] = (T_np[:, 2:] - T_np[:, :-2]) / 2

        g11 = np.array(G[0])
        g22 = np.array(G[2])
        a11 = 1.0 / g11
        a22 = 1.0 / g22
        grad_norm_G = np.sqrt(a11 * grad_i ** 2 + a22 * grad_j ** 2)

        interior = np.zeros((N, N), dtype=bool)
        interior[10:-10, 10:-10] = True
        interior[N // 2 - 5:N // 2 + 6, N // 2 - 5:N // 2 + 6] = False

        eikonal_residual = np.abs(grad_norm_G[interior] - 1.0)
        mean_residual = float(np.mean(eikonal_residual))
        max_residual = float(np.max(eikonal_residual))

        print(f"  Eikonal residual: mean={mean_residual:.6f}, max={max_residual:.6f}")

        self.T, self.G = T, G
        self.grad_norm_G = grad_norm_G

        return ExperimentResult(
            name=self.name, category=self.category,
            success=mean_residual < 0.05,
            metrics={'mean_residual': mean_residual, 'max_residual': max_residual},
            arrays={'T': np.array(T), 'G11': np.array(G[0]),
                    'grad_norm_G': grad_norm_G},
            metadata={'N': N}
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plot_arrival_time(self.T, ax=axes[0], title='Arrival Time')

        im = axes[1].imshow(np.array(self.G[0]), origin='upper', cmap='viridis')
        plt.colorbar(im, ax=axes[1], label='g₁₁')
        axes[1].set_title('Spatially Varying g₁₁')

        residual = np.abs(self.grad_norm_G - 1.0)
        im = axes[2].imshow(residual, origin='upper', cmap='hot', vmin=0, vmax=0.2)
        plt.colorbar(im, ax=axes[2], label='||∇T||_G - 1|')
        axes[2].set_title(f'Eikonal Residual (mean={self.result.metrics["mean_residual"]:.4f})')

        fig.suptitle('A7: Spatially Varying Metric', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A8: MULTI-SOURCE
# =============================================================================
@register_experiment
class A8_MultiSource(Experiment):
    """Verify Voronoi-like structure with multiple sources."""
    
    name = "A8_multi_source"
    category = "A_forward_solver"
    description = "Verify correct multi-source handling"
    
    def __init__(self, N: int = 200):
        super().__init__()
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0)
        G = G.at[2].set(1.0)
        B = jnp.zeros((2, N, N))
        
        sources = [(N//4, N//4), (N//4, 3*N//4), (3*N//4, N//2)]
        source_coords = jnp.array(sources, dtype=jnp.float32)
        
        H, W = eikonal_to_zermelo(G, B)
        metric = SyntheticZermeloMetric(H, W)
        solver = get_solver()
        
        T, _, _ = solver.solve(metric, source_coords, grid_extent=(0, N-1, 0, N-1), grid_shape=(N, N))
        
        T_exact = jnp.full((N, N), jnp.inf)
        for si, sj in sources:
            T_exact = jnp.minimum(T_exact, euclidean_distance_field(N, N, si, sj))
        
        interior = jnp.zeros((N, N), dtype=bool)
        interior = interior.at[3:-3, 3:-3].set(True)
        I, J = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
        
        for si, sj in sources:
            near = (jnp.abs(I - si) <= 2) & (jnp.abs(J - sj) <= 2)
            interior = interior & (~near)
        
        errs = compute_errors(T, T_exact, interior)
        
        self.T, self.T_exact, self.sources = T, T_exact, sources
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=errs['rel_l2'] < 0.02,
            metrics=errs,
            arrays={'T': np.array(T), 'T_exact': np.array(T_exact)},
            metadata={'sources': sources}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for ax, T_arr, title in [(axes[0], self.T, 'Computed'), (axes[1], self.T_exact, 'Exact')]:
            plot_arrival_time(T_arr, ax=ax, title=title)
            for si, sj in self.sources:
                ax.plot(sj, si, 'r*', markersize=15)
        
        plot_error_map(self.T - self.T_exact, ax=axes[2],
                       title=f'Error (rel L2={self.result.metrics["rel_l2"]:.4f})')
        fig.suptitle('A8: Multi-Source', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# A9: ITERATION COMPLEXITY
# =============================================================================
@register_experiment
class A9_IterationComplexity(Experiment):
    """Verify O(1) iterations, O(N²) total work."""

    name = "A9_iteration_complexity"
    category = "A_forward_solver"
    description = "Verify iteration count is O(1)"

    def __init__(self, grid_sizes: List[int] = None):
        super().__init__()
        self.grid_sizes = grid_sizes or [50, 100, 200, 400]

    def run(self) -> ExperimentResult:
        times = []

        for N in self.grid_sizes:
            print(f"  Grid: {N}x{N}")

            G = jnp.zeros((3, N, N))
            G = G.at[0].set(1.0)
            G = G.at[2].set(1.0)
            B = jnp.zeros((2, N, N))
            source_coords = jnp.array([[N // 2, N // 2]], dtype=jnp.float32)
            
            H, W = eikonal_to_zermelo(G, B)
            metric = SyntheticZermeloMetric(H, W)
            solver = get_solver()

            # Warmup JIT
            T_warmup, _, _ = solver.solve(metric, source_coords, grid_extent=(0, N-1, 0, N-1), grid_shape=(N, N))
            jax.block_until_ready(T_warmup)

            start = time.time()
            T_timed, _, _ = solver.solve(metric, source_coords, grid_extent=(0, N-1, 0, N-1), grid_shape=(N, N))
            jax.block_until_ready(T_timed)
            elapsed = time.time() - start

            times.append(elapsed)
            print(f"    Time: {elapsed:.3f}s")

        # Check O(N²) scaling for time
        N_arr = np.array(self.grid_sizes, dtype=float)
        t_arr = np.array(times)
        log_N = np.log(N_arr)
        log_t = np.log(t_arr)
        alpha, _ = np.polyfit(log_N, log_t, 1)

        self._data = {'sizes': self.grid_sizes, 'times': times}

        return ExperimentResult(
            name=self.name, category=self.category,
            success=(1 < alpha < 3.0),
            metrics={'time_scaling': alpha},
            arrays={'grid_sizes': np.array(self.grid_sizes),
                    'times': np.array(times)},
            metadata={'grid_sizes': self.grid_sizes}
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        N = np.array(self._data['sizes'])

        axes.loglog(N, self._data['times'], 'o-', markersize=10, label='Measured')
        axes.loglog(N, self._data['times'][0] * (N / N[0]) ** 2, 'k--', alpha=0.5,
                       label=f'O(N²), measured α={self.result.metrics["time_scaling"]:.2f}')
        axes.set_xlabel('Grid size N')
        axes.set_ylabel('Time (s)')
        axes.set_title('Timing')
        axes.legend()
        axes.grid(True, alpha=0.3)

        fig.suptitle('A9: Complexity Analysis', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig

# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    A1_IsotropicConvergence,
    A2_UniformAnisotropic,
    A3_RotatedAnisotropic,
    A4_ConstantDrift,
    A5_CombinedAnisotropicDrift,
    A6_PiecewiseMetric,
    A7_SpatiallyVaryingMetric,
    A8_MultiSource,
    A9_IterationComplexity,
]

def run_all(save=True, visualize=True):
    """Run all Category A experiments."""
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)
    
    print("\n" + "="*60)
    print("CATEGORY A SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results

if __name__ == "__main__":
    run_all()
