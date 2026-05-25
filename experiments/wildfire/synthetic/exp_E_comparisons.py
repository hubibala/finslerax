import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import time

from experiments.wildfire.synthetic.experiment_base import (
    Experiment, ExperimentResult, register_experiment,
    get_interior_mask, create_sparse_observation_mask,
    plot_arrival_time, SyntheticZermeloMetric
)
from ham.solvers.eikonal import EikonalSolver
from ham.solvers.avbd import AVBDSolver
from experiments.wildfire.synthetic.metric_recovery import (
    MetricRecoveryOptimizer, eikonal_to_zermelo
)

# =============================================================================
# E1: SOLVER FORWARD RUNTIME
# =============================================================================
@register_experiment
class E1_SolverForwardRuntime(Experiment):
    """Compare Fast Sweeping (Eikonal) vs AVBD forward runtime."""
    
    name = "E1_solver_forward_runtime"
    category = "E_comparisons"
    description = "Runtime comparison: Fast Sweeping vs AVBD"
    
    def __init__(self, grid_sizes: List[int] = None):
        super().__init__()
        self.grid_sizes = grid_sizes or [20, 40, 60]  # Kept small because AVBD is slow for large N
        
    def run(self) -> ExperimentResult:
        results = []
        
        for N in self.grid_sizes:
            print(f"  Grid: {N}x{N}")
            
            G = jnp.zeros((3, N, N))
            G = G.at[0].set(1.0)
            G = G.at[2].set(1.0)
            B = jnp.zeros((2, N, N))
            
            source_coords = jnp.array([[N//2, N//2]], dtype=jnp.float32)
            
            H, W = eikonal_to_zermelo(G, B)
            metric = SyntheticZermeloMetric(H, W)
            
            # Eikonal (Fast Sweeping)
            solver_eik = EikonalSolver(max_iters=50, tol=1e-5)
            
            # Warmup Eikonal
            _, _, _ = solver_eik.solve(metric, source_coords, (0, N-1, 0, N-1), (N, N))
            
            start = time.time()
            T_eik, _, _ = solver_eik.solve(metric, source_coords, (0, N-1, 0, N-1), (N, N))
            # Block until JAX finishes execution
            T_eik.block_until_ready()
            time_eik = time.time() - start
            
            # AVBD (Geodesic BVP)
            solver_avbd = AVBDSolver()
            obs_coords = jnp.array([[N//4, N//4]], dtype=jnp.float32)
            
            # Warmup AVBD (evaluating 1 path)
            _ = solver_avbd.solve(metric, source_coords[0], obs_coords)
            
            start = time.time()
            # AVBD solves for specific points. To compare, we compute paths to N points.
            test_points = jnp.array([[i, j] for i in range(1, N, N//10 + 1) for j in range(1, N, N//10 + 1)], dtype=jnp.float32)
            vmap_solve = jax.jit(jax.vmap(lambda obs: solver_avbd.solve(metric, source_coords[0], jnp.array([obs]))))
            
            paths_avbd = vmap_solve(test_points)
            paths_avbd[0].block_until_ready()
            time_avbd = time.time() - start
            
            # Normalize to estimate full field evaluation time for AVBD
            time_avbd_est = time_avbd * ((N*N) / len(test_points))
            
            results.append({
                'N': N,
                'time_eik': time_eik,
                'time_avbd': time_avbd_est,
                'speedup': time_avbd_est / time_eik if time_eik > 0 else 0
            })
            
            print(f"    Eikonal: {time_eik:.4f}s, AVBD (est): {time_avbd_est:.4f}s, Speedup: {results[-1]['speedup']:.1f}x")
            
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'avg_speedup': np.mean([r['speedup'] for r in results])},
            arrays={'grid_sizes': np.array([r['N'] for r in results]),
                   'time_eik': np.array([r['time_eik'] for r in results]),
                   'time_avbd': np.array([r['time_avbd'] for r in results])},
            metadata={'grid_sizes': self.grid_sizes}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        N = np.array([r['N'] for r in self.results])
        t_eik = np.array([r['time_eik'] for r in self.results])
        t_avbd = np.array([r['time_avbd'] for r in self.results])
        
        axes[0].loglog(N, t_eik, 'o-', label='Eikonal (Fast Sweeping)', markersize=10)
        axes[0].loglog(N, t_avbd, 's-', label='AVBD (Geodesic BVP - est)', markersize=10)
        axes[0].set_xlabel('Grid size N')
        axes[0].set_ylabel('Time (s)')
        axes[0].set_title('Forward Runtime Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        speedup = t_avbd / t_eik
        axes[1].bar(range(len(N)), speedup, tick_label=[str(n) for n in N])
        axes[1].set_xlabel('Grid size N')
        axes[1].set_ylabel('Speedup (Eikonal / AVBD)')
        axes[1].set_title('Speedup of Eikonal Solver')
        
        fig.suptitle('E1: Eikonal vs AVBD Runtime', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# E3: REGULARIZATION STRATEGIES
# =============================================================================
@register_experiment
class E3_RegularizationStrategies(Experiment):
    """Compare TV vs Tikhonov vs no regularization (Eikonal)."""
    
    name = "E3_regularization_strategies"
    category = "E_comparisons"
    description = "Compare different regularization approaches"
    
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
        
        source_coords = jnp.array([[N//2, N//4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//4].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        obs_mask = create_sparse_observation_mask(N, N, 0.05, source_mask, seed=57)
        
        results = {}
        
        for reg_type in ['none', 'tv', 'tikhonov']:
            print(f"\n  Regularization: {reg_type}")
            
            lam = 0.01 if reg_type != 'none' else 0.0
            
            opt = MetricRecoveryOptimizer(N, N, solver_type='eikonal', lambda_H=lam, 
                                          constrain_isotropic=True, reg_type=reg_type)
            opt.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=False)
            
            G_rec, _ = opt.get_G_B()
            interior = get_interior_mask(N, N, 5, source_mask)
            
            g11_true = G_true[0][interior]
            g11_rec = G_rec[0][interior]
            rel_err = float(jnp.sqrt(jnp.mean((g11_true - g11_rec)**2)) / jnp.mean(g11_true))
            
            results[reg_type] = {'error': rel_err, 'G': np.array(G_rec)}
            print(f"    Error: {rel_err:.4f}")
            
        self.results = results
        self.G_true = G_true
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={f'error_{k}': v['error'] for k, v in results.items()},
            arrays={f'G_{k}': v['G'] for k, v in results.items()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        
        vmin, vmax = float(self.G_true[0].min()), float(self.G_true[0].max())
        
        axes[0].imshow(self.G_true[0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('True g₁₁')
        
        for ax, (name, res) in zip(axes[1:], self.results.items()):
            ax.imshow(res['G'][0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'{name.upper()} (err={res["error"]:.2%})')
        
        fig.suptitle('E3: Regularization Comparison', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# E4: OPTIMIZATION METHODS
# =============================================================================
@register_experiment
class E4_OptimizationMethods(Experiment):
    """Compare SGD vs Adam for metric recovery."""
    
    name = "E4_optimization_methods"
    category = "E_comparisons"
    description = "Compare optimization algorithms"
    
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
        
        source_coords = jnp.array([[N//2, N//4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//4].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=58)
        
        results = {}
        
        for method in ['sgd', 'adam']:
            print(f"\n  Method: {method}")
            
            lr = 0.05 if method == 'sgd' else 0.01
            
            opt = MetricRecoveryOptimizer(N, N, solver_type='eikonal', lambda_H=0.01, 
                                          constrain_isotropic=True, optimizer_type=method)
            opt.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=lr, verbose=False)
            
            G_rec, _ = opt.get_G_B()
            interior = get_interior_mask(N, N, 5, source_mask)
            
            g11_true = G_true[0][interior]
            g11_rec = G_rec[0][interior]
            rel_err = float(jnp.sqrt(jnp.mean((g11_true - g11_rec)**2)) / jnp.mean(g11_true))
            
            results[method] = {'error': rel_err, 'losses': opt.history['loss'], 'G': np.array(G_rec)}
            print(f"    Final error: {rel_err:.4f}")
            
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={f'error_{k}': v['error'] for k, v in results.items()},
            arrays={f'losses_{k}': np.array(v['losses']) for k, v in results.items()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for method, res in self.results.items():
            axes[0].semilogy(res['losses'], label=f'{method.upper()} (err={res["error"]:.2%})')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        methods = list(self.results.keys())
        errors = [self.results[m]['error'] for m in methods]
        axes[1].bar(methods, errors, color=['steelblue', 'coral'])
        axes[1].set_ylabel('Relative Error')
        axes[1].set_title('Final Error')
        
        fig.suptitle('E4: Optimization Methods', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# E5: SCALABILITY STUDY
# =============================================================================
@register_experiment
class E5_Scalability(Experiment):
    """Full pipeline scalability in JAX: forward + backward."""
    
    name = "E5_scalability"
    category = "E_comparisons"
    description = "Scalability of full forward+backward pipeline"
    
    def __init__(self, grid_sizes: List[int] = None):
        super().__init__()
        self.grid_sizes = grid_sizes or [20, 40, 80, 160]
        
    def run(self) -> ExperimentResult:
        results = []
        
        for N in self.grid_sizes:
            print(f"  Grid: {N}x{N}")
            
            G_true = jnp.zeros((3, N, N))
            G_true = G_true.at[0].set(1.0)
            G_true = G_true.at[2].set(1.0)
            B_true = jnp.zeros((2, N, N))
            
            source_coords = jnp.array([[N//2, N//2]], dtype=jnp.float32)
            source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
            
            opt = MetricRecoveryOptimizer(N, N, solver_type='eikonal')
            
            # Create synthetic target observation
            H_true, W_true = eikonal_to_zermelo(G_true, B_true)
            metric_true = SyntheticZermeloMetric(H_true, W_true)
            solver = EikonalSolver(max_iters=50, tol=1e-5)
            T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
            
            obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=59)
            
            # Forward pass timing (already compiled from fit, but we can time one step)
            # We'll just run .fit for 5 steps and measure average time per step
            start = time.time()
            opt.fit(source_coords, T_obs, obs_mask, n_iter=5, lr=0.01, verbose=False)
            time_per_step = (time.time() - start) / 5.0
            
            results.append({
                'N': N,
                'time_per_step': time_per_step
            })
            
            print(f"    Avg Time/Step (Fwd+Bwd+Update): {time_per_step:.4f}s")
            
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'avg_step_time': np.mean([r['time_per_step'] for r in results])},
            arrays={'grid_sizes': np.array([r['N'] for r in results]),
                   'time_per_step': np.array([r['time_per_step'] for r in results])},
            metadata={'grid_sizes': self.grid_sizes}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        N = np.array([r['N'] for r in self.results])
        t_step = np.array([r['time_per_step'] for r in self.results])
        
        ax.loglog(N, t_step, 'o-', label='Forward + Backward + Update', markersize=10)
        ax.loglog(N, t_step[0] * (N/N[0])**2, 'k--', alpha=0.5, label='O(N²)')
        ax.set_xlabel('Grid size N')
        ax.set_ylabel('Time (s)')
        ax.set_title('E5: Scalability Study (JAX)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    E1_SolverForwardRuntime,
    E3_RegularizationStrategies,
    E4_OptimizationMethods,
    E5_Scalability,
]

def run_all(save=True, visualize=True):
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)
    
    print("\n" + "="*60)
    print("CATEGORY E SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results

if __name__ == "__main__":
    run_all()
