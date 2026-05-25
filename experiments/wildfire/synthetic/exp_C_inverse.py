import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx
from typing import Dict, List, Optional, Tuple

from experiments.wildfire.synthetic.experiment_base import (
    Experiment, ExperimentResult, register_experiment,
    get_interior_mask, create_sparse_observation_mask,
    plot_arrival_time, plot_error_map, SyntheticZermeloMetric
)
from ham.solvers.eikonal import EikonalSolver
from experiments.wildfire.synthetic.metric_recovery import (
    MetricRecoveryOptimizer, eikonal_to_zermelo, zermelo_to_eikonal
)

def evaluate_recovery(G_true, G_rec, interior_mask):
    """Compute relative error for G11 and G22."""
    g11_true = G_true[0][interior_mask]
    g11_rec = G_rec[0][interior_mask]
    g22_true = G_true[2][interior_mask]
    g22_rec = G_rec[2][interior_mask]
    
    err_g11 = float(jnp.sqrt(jnp.mean((g11_true - g11_rec)**2)) / jnp.mean(g11_true))
    err_g22 = float(jnp.sqrt(jnp.mean((g22_true - g22_rec)**2)) / jnp.mean(g22_true))
    return err_g11, err_g22

def evaluate_drift(B_true, B_rec, interior_mask):
    """Compute relative error for drift B."""
    b1_true, b2_true = B_true[0][interior_mask], B_true[1][interior_mask]
    b1_rec, b2_rec = B_rec[0][interior_mask], B_rec[1][interior_mask]
    
    err_b1 = float(jnp.sqrt(jnp.mean((b1_true - b1_rec)**2)))
    err_b2 = float(jnp.sqrt(jnp.mean((b2_true - b2_rec)**2)))
    return err_b1, err_b2

# =============================================================================
# C1: ISOTROPIC RECOVERY (FULL OBS)
# =============================================================================
@register_experiment
class C1_IsotropicFull(Experiment):
    """Recover isotropic metric from full observations (Dual Solver Comparison)."""
    
    name = "C1_isotropic_recovery_full"
    category = "C_inverse_problem"
    description = "Recover isotropic metric from full observations"
    
    def __init__(self, N: int = 40, n_iter: int = 250):
        super().__init__()
        self.N, self.n_iter = N, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N//2:].set(2.0)
        G_true = G_true.at[2, :, N//2:].set(2.0)
        
        B_true = jnp.zeros((2, N, N))
        
        source_coords = jnp.array([[N//2, N//4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//4].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        obs_mask = get_interior_mask(N, N, 3, source_mask)
        
        # Eikonal Optimizer
        print("\n  Training Eikonal Optimizer...")
        opt_eik = MetricRecoveryOptimizer(N, N, solver_type='eikonal', lambda_H=0.005, constrain_isotropic=True)
        metrics_eik = opt_eik.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
        G_rec_eik, _ = opt_eik.get_G_B()
        
        # AVBD Optimizer
        print("  Training AVBD Optimizer...")
        opt_avbd = MetricRecoveryOptimizer(N, N, solver_type='avbd', lambda_H=0.005, constrain_isotropic=True)
        metrics_avbd = opt_avbd.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
        G_rec_avbd, _ = opt_avbd.get_G_B()
        
        interior = get_interior_mask(N, N, 5, source_mask)
        err_eik, _ = evaluate_recovery(G_true, G_rec_eik, interior)
        err_avbd, _ = evaluate_recovery(G_true, G_rec_avbd, interior)
        
        print(f"  Eikonal Error: {err_eik:.4f}")
        print(f"  AVBD Error: {err_avbd:.4f}")
        
        self.G_true = G_true
        self.G_rec_eik, self.G_rec_avbd = G_rec_eik, G_rec_avbd
        self.opt_eik, self.opt_avbd = opt_eik, opt_avbd
        self.T_obs = T_obs
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=err_eik < 0.2 and err_avbd < 0.3,
            metrics={'err_eikonal': err_eik, 'err_avbd': err_avbd, **metrics_eik},
            arrays={'G_true': np.array(G_true), 'G_eik': np.array(G_rec_eik), 'G_avbd': np.array(G_rec_avbd),
                   'loss_eik': np.array(opt_eik.history['loss']), 'loss_avbd': np.array(opt_avbd.history['loss'])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        
        vmin, vmax = float(self.G_true[0].min()), float(self.G_true[0].max())
        
        axes[0,0].imshow(self.G_true[0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0,0].set_title('True g₁₁')
        
        axes[0,1].imshow(self.G_rec_eik[0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0,1].set_title(f'Eikonal Recovered (err={self.result.metrics["err_eikonal"]:.2%})')
        
        axes[0,2].imshow(self.G_rec_avbd[0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0,2].set_title(f'AVBD Recovered (err={self.result.metrics["err_avbd"]:.2%})')
        
        axes[0,3].plot(self.opt_eik.history['loss'], label='Eikonal')
        axes[0,3].plot(self.opt_avbd.history['loss'], label='AVBD')
        axes[0,3].set_yscale('log')
        axes[0,3].legend()
        axes[0,3].set_title('Loss History')
        
        plot_arrival_time(self.T_obs, ax=axes[1,0], title='Observed T')
        
        err_eik = self.G_rec_eik[0] - self.G_true[0]
        plot_error_map(err_eik, ax=axes[1,1], title='Eikonal Error Map')
        
        err_avbd = self.G_rec_avbd[0] - self.G_true[0]
        plot_error_map(err_avbd, ax=axes[1,2], title='AVBD Error Map')
        
        j = self.N // 2
        axes[1,3].plot(self.G_true[0, j, :], 'k-', label='True', lw=2)
        axes[1,3].plot(self.G_rec_eik[0, j, :], 'b--', label='Eikonal', lw=2)
        axes[1,3].plot(self.G_rec_avbd[0, j, :], 'r-.', label='AVBD', lw=2)
        axes[1,3].set_title('Cross-section')
        axes[1,3].legend()
        
        fig.suptitle('C1: Isotropic Recovery (Full Observations)', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C2: ISOTROPIC RECOVERY (SPARSE OBS)
# =============================================================================
@register_experiment
class C2_IsotropicSparse(Experiment):
    """Recover isotropic metric from sparse observations."""
    
    name = "C2_isotropic_recovery_sparse"
    category = "C_inverse_problem"
    description = "Recover isotropic metric from 10% observations"
    
    def __init__(self, N: int = 40, obs_fraction: float = 0.1, n_iter: int = 300):
        super().__init__()
        self.N, self.obs_fraction, self.n_iter = N, obs_fraction, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        j_coords = jnp.arange(N, dtype=jnp.float32)
        base = 1.0 + 0.8 * jax.nn.sigmoid((j_coords - N/2) / 5.0)
        
        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(jnp.tile(base, (N, 1)))
        G_true = G_true.at[2].set(G_true[0])
        
        B_true = jnp.zeros((2, N, N))
        
        source_coords = jnp.array([[N//2, N//4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//4].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        obs_mask = create_sparse_observation_mask(N, N, self.obs_fraction, source_mask, seed=46)
        print(f"  Using {obs_mask.sum()} observations ({self.obs_fraction*100:.1f}%)")
        
        print("\n  Training Eikonal Optimizer...")
        opt_eik = MetricRecoveryOptimizer(N, N, solver_type='eikonal', lambda_H=0.01, constrain_isotropic=True)
        opt_eik.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
        G_rec_eik, _ = opt_eik.get_G_B()
        
        print("  Training AVBD Optimizer...")
        opt_avbd = MetricRecoveryOptimizer(N, N, solver_type='avbd', lambda_H=0.01, constrain_isotropic=True)
        opt_avbd.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
        G_rec_avbd, _ = opt_avbd.get_G_B()
        
        interior = get_interior_mask(N, N, 5, source_mask)
        err_eik, _ = evaluate_recovery(G_true, G_rec_eik, interior)
        err_avbd, _ = evaluate_recovery(G_true, G_rec_avbd, interior)
        
        print(f"  Eikonal Error: {err_eik:.4f}")
        print(f"  AVBD Error: {err_avbd:.4f}")
        
        self.G_true = G_true
        self.G_rec_eik, self.G_rec_avbd = G_rec_eik, G_rec_avbd
        self.obs_mask = obs_mask
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=err_eik < 0.2 and err_avbd < 0.3,
            metrics={'err_eikonal': err_eik, 'err_avbd': err_avbd},
            arrays={'G_true': np.array(G_true), 'G_eik': np.array(G_rec_eik), 'G_avbd': np.array(G_rec_avbd)},
            metadata={'N': N, 'obs_fraction': self.obs_fraction}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        
        vmin, vmax = float(self.G_true[0].min()), float(self.G_true[0].max())
        
        axes[0].imshow(self.G_true[0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        oy, ox = jnp.where(self.obs_mask)
        axes[0].scatter(ox, oy, c='r', s=2, alpha=0.5)
        axes[0].set_title('True g₁₁ + Obs')
        
        axes[1].imshow(self.G_rec_eik[0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Eikonal Recovered (err={self.result.metrics["err_eikonal"]:.2%})')
        
        axes[2].imshow(self.G_rec_avbd[0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2].set_title(f'AVBD Recovered (err={self.result.metrics["err_avbd"]:.2%})')
        
        j = self.N // 2
        axes[3].plot(self.G_true[0, j, :], 'k-', label='True', lw=2)
        axes[3].plot(self.G_rec_eik[0, j, :], 'b--', label='Eikonal', lw=2)
        axes[3].plot(self.G_rec_avbd[0, j, :], 'r-.', label='AVBD', lw=2)
        axes[3].set_title('Cross-section')
        axes[3].legend()
        
        fig.suptitle(f'C2: Sparse Recovery ({self.obs_fraction*100:.0f}% obs)', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C3: DIAGONAL ANISOTROPIC RECOVERY
# =============================================================================
@register_experiment
class C3_DiagonalAnisotropic(Experiment):
    """Recover diagonal anisotropic metric (g11 ≠ g22)."""
    
    name = "C3_diagonal_anisotropic"
    category = "C_inverse_problem"
    description = "Recover diagonal anisotropic metric"
    
    def __init__(self, N: int = 40, n_iter: int = 300):
        super().__init__()
        self.N, self.n_iter = N, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.5)  # g11
        G_true = G_true.at[2].set(0.5)  # g22
        G_true = G_true.at[0, :, N//2:].set(0.5)
        G_true = G_true.at[2, :, N//2:].set(1.5)
        
        B_true = jnp.zeros((2, N, N))
        
        source_coords = jnp.array([[N//2, N//2]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=47)
        
        print("\n  Training Eikonal Optimizer...")
        opt_eik = MetricRecoveryOptimizer(N, N, solver_type='eikonal', lambda_H=0.01, constrain_isotropic=False)
        # Keep g12=0 by nullifying it in the mask? Actually, opt_eik optimizes H. 
        # By default it will recover off-diagonals unless constrained. Let's let it recover all.
        opt_eik.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
        G_rec_eik, _ = opt_eik.get_G_B()
        
        print("  Training AVBD Optimizer...")
        opt_avbd = MetricRecoveryOptimizer(N, N, solver_type='avbd', lambda_H=0.01, constrain_isotropic=False)
        opt_avbd.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
        G_rec_avbd, _ = opt_avbd.get_G_B()
        
        interior = get_interior_mask(N, N, 5, source_mask)
        eik_g11, eik_g22 = evaluate_recovery(G_true, G_rec_eik, interior)
        avbd_g11, avbd_g22 = evaluate_recovery(G_true, G_rec_avbd, interior)
        
        print(f"  Eikonal Error: g11={eik_g11:.4f}, g22={eik_g22:.4f}")
        print(f"  AVBD Error: g11={avbd_g11:.4f}, g22={avbd_g22:.4f}")
        
        self.G_true = G_true
        self.G_rec_eik, self.G_rec_avbd = G_rec_eik, G_rec_avbd
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=eik_g11 < 0.25 and avbd_g11 < 0.35,
            metrics={'eik_g11': eik_g11, 'eik_g22': eik_g22, 'avbd_g11': avbd_g11, 'avbd_g22': avbd_g22},
            arrays={'G_true': np.array(G_true), 'G_eik': np.array(G_rec_eik), 'G_avbd': np.array(G_rec_avbd)},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for row, (c, name) in enumerate([(0, 'g₁₁'), (2, 'g₂₂')]):
            vmin = float(min(self.G_true[c].min(), self.G_rec_eik[c].min()))
            vmax = float(max(self.G_true[c].max(), self.G_rec_eik[c].max()))
            
            axes[row, 0].imshow(self.G_true[c], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'True {name}')
            
            axes[row, 1].imshow(self.G_rec_eik[c], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'Eikonal {name}')
            
            axes[row, 2].imshow(self.G_rec_avbd[c], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[row, 2].set_title(f'AVBD {name}')
            
        fig.suptitle('C3: Diagonal Anisotropic Recovery', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C5: DRIFT RECOVERY
# =============================================================================
@register_experiment
class C5_DriftRecovery(Experiment):
    """Recover drift field with known metric."""
    
    name = "C5_drift_recovery"
    category = "C_inverse_problem"
    description = "Recover drift B with fixed G"
    
    def __init__(self, N: int = 40, n_iter: int = 250):
        super().__init__()
        self.N, self.n_iter = N, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        
        B_true = jnp.zeros((2, N, N))
        B_true = B_true.at[0].set(0.2)
        B_true = B_true.at[1].set(0.1)
        
        source_coords = jnp.array([[N//2, N//2]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        obs_mask = create_sparse_observation_mask(N, N, 0.15, source_mask, seed=48)
        
        print("\n  Training Eikonal Optimizer...")
        opt_eik = MetricRecoveryOptimizer(N, N, solver_type='eikonal', recover_H=False, recover_W=True, lambda_W=0.01)
        opt_eik.model = eqx.tree_at(lambda m: m.H_grid, opt_eik.model, H_true)
        opt_eik.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
        _, B_rec_eik = opt_eik.get_G_B()
        
        print("  Training AVBD Optimizer...")
        opt_avbd = MetricRecoveryOptimizer(N, N, solver_type='avbd', recover_H=False, recover_W=True, lambda_W=0.01)
        opt_avbd.model = eqx.tree_at(lambda m: m.H_grid, opt_avbd.model, H_true)
        opt_avbd.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
        _, B_rec_avbd = opt_avbd.get_G_B()
        
        interior = get_interior_mask(N, N, 5, source_mask)
        eik_b1, eik_b2 = evaluate_drift(B_true, B_rec_eik, interior)
        avbd_b1, avbd_b2 = evaluate_drift(B_true, B_rec_avbd, interior)
        
        print(f"  Eikonal Error: b1={eik_b1:.4f}, b2={eik_b2:.4f}")
        print(f"  AVBD Error: b1={avbd_b1:.4f}, b2={avbd_b2:.4f}")
        
        self.B_true = B_true
        self.B_rec_eik, self.B_rec_avbd = B_rec_eik, B_rec_avbd
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=eik_b1 < 0.15 and avbd_b1 < 0.15,
            metrics={'eik_b1': eik_b1, 'eik_b2': eik_b2, 'avbd_b1': avbd_b1, 'avbd_b2': avbd_b2},
            arrays={'B_true': np.array(B_true), 'B_eik': np.array(B_rec_eik), 'B_avbd': np.array(B_rec_avbd)},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for row, (c, name) in enumerate([(0, 'b₁'), (1, 'b₂')]):
            vmin = float(min(self.B_true[c].min(), self.B_rec_eik[c].min()))
            vmax = float(max(self.B_true[c].max(), self.B_rec_eik[c].max()))
            
            axes[row, 0].imshow(self.B_true[c], origin='upper', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'True {name}')
            
            axes[row, 1].imshow(self.B_rec_eik[c], origin='upper', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'Eikonal {name}')
            
            axes[row, 2].imshow(self.B_rec_avbd[c], origin='upper', cmap='RdBu_r', vmin=vmin, vmax=vmax)
            axes[row, 2].set_title(f'AVBD {name}')
            
        fig.suptitle('C5: Drift Recovery', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C7: REGULARIZATION ABLATION
# =============================================================================
@register_experiment
class C7_RegularizationAblation(Experiment):
    """Show TV regularization is necessary for sparse observations."""
    
    name = "C7_regularization_ablation"
    category = "C_inverse_problem"
    description = "Recovery error vs regularization strength (Eikonal only for speed)"
    
    def __init__(self, N: int = 40, n_iter: int = 200):
        super().__init__()
        self.N, self.n_iter = N, n_iter
        self.lambda_values = [0, 0.005, 0.01, 0.05, 0.1]
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N//2:].set(2.0)
        G_true = G_true.at[2, :, N//2:].set(2.0)
        
        B_true = jnp.zeros((2, N, N))
        
        source_coords = jnp.array([[N//2, N//4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//4].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        obs_mask = create_sparse_observation_mask(N, N, 0.05, source_mask, seed=47)
        
        results = []
        for lam in self.lambda_values:
            print(f"\n  λ = {lam}")
            opt = MetricRecoveryOptimizer(N, N, solver_type='eikonal', lambda_H=lam, constrain_isotropic=True)
            opt.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
            
            G_rec, _ = opt.get_G_B()
            interior = get_interior_mask(N, N, 5, source_mask)
            err_g11, _ = evaluate_recovery(G_true, G_rec, interior)
            
            results.append({'lambda': lam, 'error': err_g11})
            print(f"    Error: {err_g11:.4f}")
            
        best = min(results, key=lambda x: x['error'])
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'best_lambda': best['lambda'], 'best_error': best['error'], 
                    'no_reg_error': results[0]['error']},
            arrays={'lambdas': np.array([r['lambda'] for r in results]),
                   'errors': np.array([r['error'] for r in results])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        lambdas = np.array([r['lambda'] for r in self.results])
        errors = np.array([r['error'] for r in self.results])
        
        lambdas_plot = np.where(lambdas == 0, 1e-3, lambdas)
        ax.semilogx(lambdas_plot, errors, 'o-', markersize=10, lw=2)
        
        best_idx = np.argmin(errors)
        ax.scatter([lambdas_plot[best_idx]], [errors[best_idx]], c='red', s=200, zorder=5, 
                   label=f'Best: λ={lambdas[best_idx]:.3f}')
        
        ax.set_xlabel('Regularization λ')
        ax.set_ylabel('Relative Error')
        ax.set_title('C7: Regularization Ablation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C8: OBSERVATION DENSITY
# =============================================================================
@register_experiment
class C8_ObservationDensity(Experiment):
    """Recovery error vs observation density."""
    
    name = "C8_observation_density"
    category = "C_inverse_problem"
    description = "Recovery error vs observation density (Eikonal only)"
    
    def __init__(self, N: int = 40, n_iter: int = 200):
        super().__init__()
        self.N, self.n_iter = N, n_iter
        self.obs_fractions = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N//2:].set(2.0)
        G_true = G_true.at[2, :, N//2:].set(2.0)
        B_true = jnp.zeros((2, N, N))
        
        source_coords = jnp.array([[N//2, N//4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//4].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        results = []
        for frac in self.obs_fractions:
            print(f"\n  Fraction: {frac*100:.0f}%")
            
            if frac >= 1.0:
                obs_mask = get_interior_mask(N, N, 3, source_mask)
            else:
                obs_mask = create_sparse_observation_mask(N, N, frac, source_mask, seed=48)
            
            lambda_H = 0.01 if frac < 0.5 else 0.001
            opt = MetricRecoveryOptimizer(N, N, solver_type='eikonal', lambda_H=lambda_H, constrain_isotropic=True)
            opt.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
            
            G_rec, _ = opt.get_G_B()
            interior = get_interior_mask(N, N, 5, source_mask)
            err_g11, _ = evaluate_recovery(G_true, G_rec, interior)
            
            results.append({'fraction': frac, 'error': err_g11})
            print(f"    Error: {err_g11:.4f}")
            
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'min_error': min(r['error'] for r in results)},
            arrays={'fractions': np.array([r['fraction'] for r in results]),
                   'errors': np.array([r['error'] for r in results])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        fracs = np.array([r['fraction'] for r in self.results]) * 100
        errors = np.array([r['error'] for r in self.results])
        
        ax.plot(fracs, errors, 'o-', markersize=10, lw=2)
        ax.set_xlabel('Observation Density (%)')
        ax.set_ylabel('Relative Error')
        ax.set_title('C8: Observation Density Study')
        ax.grid(True, alpha=0.3)
        
        for f, e in zip(fracs, errors):
            ax.annotate(f'{e:.2%}', xy=(f, e), xytext=(5, 5), textcoords='offset points', fontsize=9)
            
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C9: NOISE ROBUSTNESS
# =============================================================================
@register_experiment
class C9_NoiseRobustness(Experiment):
    """Test recovery under measurement noise."""
    
    name = "C9_noise_robustness"
    category = "C_inverse_problem"
    description = "Recovery error vs noise level (Eikonal only)"
    
    def __init__(self, N: int = 40, n_iter: int = 200):
        super().__init__()
        self.N, self.n_iter = N, n_iter
        self.noise_levels = [0, 0.02, 0.05, 0.1, 0.2]
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N//2:].set(2.0)
        G_true = G_true.at[2, :, N//2:].set(2.0)
        B_true = jnp.zeros((2, N, N))
        
        source_coords = jnp.array([[N//2, N//4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//4].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_clean, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=49)
        
        results = []
        np.random.seed(50)
        
        for noise in self.noise_levels:
            print(f"\n  Noise: {noise*100:.0f}%")
            
            T_obs = T_clean
            if noise > 0:
                std = float(jnp.std(T_clean))
                noise_tensor = noise * std * jnp.array(np.random.randn(N, N))
                T_obs = T_clean + noise_tensor
            
            opt = MetricRecoveryOptimizer(N, N, solver_type='eikonal', lambda_H=0.02, constrain_isotropic=True)
            opt.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
            
            G_rec, _ = opt.get_G_B()
            interior = get_interior_mask(N, N, 5, source_mask)
            err_g11, _ = evaluate_recovery(G_true, G_rec, interior)
            
            results.append({'noise': noise, 'error': err_g11})
            print(f"    Error: {err_g11:.4f}")
            
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'clean_error': results[0]['error'], 'noisy_error': results[-1]['error']},
            arrays={'noise_levels': np.array([r['noise'] for r in results]),
                   'errors': np.array([r['error'] for r in results])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        noise = np.array([r['noise'] for r in self.results]) * 100
        errors = np.array([r['error'] for r in self.results])
        
        ax.plot(noise, errors, 'o-', markersize=10, lw=2)
        ax.set_xlabel('Noise Level (%)')
        ax.set_ylabel('Relative Error')
        ax.set_title('C9: Noise Robustness')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# C10: MULTIPLE SOURCES
# =============================================================================
@register_experiment
class C10_MultipleSources(Experiment):
    """Test if multiple sources improve recovery."""
    
    name = "C10_multiple_sources"
    category = "C_inverse_problem"
    description = "Recovery with multiple ignition points (Eikonal only)"
    
    def __init__(self, N: int = 40, n_iter: int = 200):
        super().__init__()
        self.N, self.n_iter = N, n_iter
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N//2:].set(2.0)
        G_true = G_true.at[2, :, N//2:].set(2.0)
        B_true = jnp.zeros((2, N, N))
        
        results = []
        
        for n_sources, sources in [(1, [[N//2, N//4]]), 
                                   (2, [[N//4, N//4], [3*N//4, N//4]]),
                                   (3, [[N//4, N//4], [3*N//4, N//4], [N//2, 3*N//4]])]:
            print(f"\n  {n_sources} source(s)")
            
            source_coords = jnp.array(sources, dtype=jnp.float32)
            source_mask = jnp.zeros((N, N), dtype=bool)
            for s in sources:
                source_mask = source_mask.at[s[0], s[1]].set(True)
                
            H_true, W_true = eikonal_to_zermelo(G_true, B_true)
            metric_true = SyntheticZermeloMetric(H_true, W_true)
            solver = EikonalSolver(max_iters=50, tol=1e-5)
            T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
            
            obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=51)
            
            opt = MetricRecoveryOptimizer(N, N, solver_type='eikonal', lambda_H=0.02, constrain_isotropic=True)
            opt.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
            
            G_rec, _ = opt.get_G_B()
            interior = get_interior_mask(N, N, 5, source_mask)
            err_g11, _ = evaluate_recovery(G_true, G_rec, interior)
            
            results.append({'n_sources': n_sources, 'error': err_g11})
            print(f"    Error: {err_g11:.4f}")
            
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'error_1source': results[0]['error'], 'error_3sources': results[-1]['error']},
            arrays={'n_sources': np.array([r['n_sources'] for r in results]),
                   'errors': np.array([r['error'] for r in results])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        n = [r['n_sources'] for r in self.results]
        errors = [r['error'] for r in self.results]
        
        ax.bar(n, errors, color='steelblue', edgecolor='black')
        ax.set_xlabel('Number of Sources')
        ax.set_ylabel('Relative Error')
        ax.set_title('C10: Multiple Sources')
        ax.set_xticks(n)
        
        for i, (ni, ei) in enumerate(zip(n, errors)):
            ax.annotate(f'{ei:.2%}', xy=(ni, ei), ha='center', va='bottom')
            
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    C1_IsotropicFull,
    C2_IsotropicSparse,
    C3_DiagonalAnisotropic,
    C5_DriftRecovery,
    C7_RegularizationAblation,
    C8_ObservationDensity,
    C9_NoiseRobustness,
    C10_MultipleSources,
]

def run_all(save=True, visualize=True):
    """Run all Category C experiments."""
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)
    
    print("\n" + "="*60)
    print("CATEGORY C SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results

if __name__ == "__main__":
    run_all()
