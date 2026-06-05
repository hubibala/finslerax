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
            obs_coords = jnp.array([N//4, N//4], dtype=jnp.float32)
            
            # Warmup AVBD (evaluating 1 path)
            _ = solver_avbd.solve(metric, source_coords[0], obs_coords)
            
            # AVBD solves for specific points. To compare, we compute paths to N points.
            test_points = jnp.array([[i, j] for i in range(1, N, N//10 + 1) for j in range(1, N, N//10 + 1)], dtype=jnp.float32)
            vmap_solve = jax.jit(jax.vmap(lambda obs: solver_avbd.solve(metric, source_coords[0], obs)))
            
            # Warmup vmap_solve
            _ = vmap_solve(test_points[:1])
            
            start = time.time()
            paths_avbd = vmap_solve(test_points)
            paths_avbd.xs.block_until_ready()
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
        
        for solver_name in ['eikonal', 'avbd']:
            for reg_type in ['none', 'tv', 'tikhonov']:
                key = f"{solver_name}_{reg_type}"
                print(f"\n  Solver: {solver_name}, Regularization: {reg_type}")
                
                lam = 0.01 if reg_type != 'none' else 0.0
                
                opt = MetricRecoveryOptimizer(N, N, solver_type=solver_name, lambda_H=lam, 
                                              constrain_isotropic=True, reg_type=reg_type)
                opt.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
                
                G_rec, B_rec = opt.get_G_B()
                
                # Validate learned metric via Eikonal solver
                H_rec, W_rec = eikonal_to_zermelo(G_rec, B_rec)
                metric_rec = SyntheticZermeloMetric(H_rec, W_rec)
                T_pred, _, _ = solver.solve(metric_rec, source_coords, (0, N-1, 0, N-1), (N, N))
                
                valid = jnp.isfinite(T_pred)
                t_err = float(jnp.sqrt(jnp.mean((T_obs[valid] - T_pred[valid])**2)))
                
                results[key] = {'error': t_err, 'G': np.array(G_rec)}
                print(f"    T Error: {t_err:.4f}")
                
        self.results = results
        self.G_true = G_true
        self.T_obs = T_obs
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={f'error_{k}': v['error'] for k, v in results.items()},
            arrays={f'G_{k}': v['G'] for k, v in results.items()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        
        vmin, vmax = float(self.G_true[0].min()), float(self.G_true[0].max())
        
        axes[0, 0].imshow(self.G_true[0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('True g₁₁')
        axes[1, 0].axis('off')
        
        for i, reg_type in enumerate(['none', 'tv', 'tikhonov']):
            res_eik = self.results[f"eikonal_{reg_type}"]
            axes[0, i+1].imshow(res_eik['G'][0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0, i+1].set_title(f'Eikonal {reg_type.upper()} (T err={res_eik["error"]:.3f})')
            
            res_avbd = self.results[f"avbd_{reg_type}"]
            axes[1, i+1].imshow(res_avbd['G'][0], origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1, i+1].set_title(f'AVBD {reg_type.upper()} (T err={res_avbd["error"]:.3f})')
        
        fig.suptitle('E3: Regularization Comparison (Dual Solver)', fontsize=14)
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
        
        for solver_name in ['eikonal', 'avbd']:
            for method in ['sgd', 'adam']:
                key = f"{solver_name}_{method}"
                print(f"\n  Solver: {solver_name}, Method: {method}")
                
                lr = 0.05 if method == 'sgd' else 0.01
                
                opt = MetricRecoveryOptimizer(N, N, solver_type=solver_name, lambda_H=0.01, 
                                              constrain_isotropic=True, optimizer_type=method)
                opt.fit(source_coords, T_obs, obs_mask, n_iter=self.n_iter, lr=lr, verbose=True)
                
                G_rec, B_rec = opt.get_G_B()
                
                # Validate learned metric via Eikonal solver
                H_rec, W_rec = eikonal_to_zermelo(G_rec, B_rec)
                metric_rec = SyntheticZermeloMetric(H_rec, W_rec)
                T_pred, _, _ = solver.solve(metric_rec, source_coords, (0, N-1, 0, N-1), (N, N))
                
                valid = jnp.isfinite(T_pred)
                t_err = float(jnp.sqrt(jnp.mean((T_obs[valid] - T_pred[valid])**2)))
                
                results[key] = {'error': t_err, 'losses': opt.history['loss'], 'G': np.array(G_rec)}
                print(f"    Final T error: {t_err:.4f}")
                
        self.results = results
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={f'error_{k}': v['error'] for k, v in results.items()},
            arrays={f'losses_{k}': np.array(v['losses']) for k, v in results.items()},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = {'eikonal_sgd': 'blue', 'eikonal_adam': 'cyan', 
                  'avbd_sgd': 'red', 'avbd_adam': 'orange'}
        
        for key, res in self.results.items():
            axes[0].semilogy(res['losses'], label=f'{key} (err={res["error"]:.3f})', color=colors[key])
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        methods = list(self.results.keys())
        errors = [self.results[m]['error'] for m in methods]
        axes[1].bar(methods, errors, color=[colors[m] for m in methods])
        axes[1].set_ylabel('T Error')
        axes[1].set_title('Final Validation Error')
        plt.xticks(rotation=45)
        
        fig.suptitle('E4: Optimization Methods Comparison (Dual Solver)', fontsize=14)
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
        self.M_sizes = [10, 50, 100, 200, 500]
        
    def run(self) -> ExperimentResult:
        results_eik = []
        results_avbd = []
        
        print("\n  Eikonal Scaling (fixed M=50, varying N):")
        for N in self.grid_sizes:
            G_true = jnp.zeros((3, N, N))
            G_true = G_true.at[0].set(1.0)
            G_true = G_true.at[2].set(1.0)
            B_true = jnp.zeros((2, N, N))
            
            source_coords = jnp.array([[N//2, N//2]], dtype=jnp.float32)
            source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
            
            opt = MetricRecoveryOptimizer(N, N, solver_type='eikonal')
            
            H_true, W_true = eikonal_to_zermelo(G_true, B_true)
            metric_true = SyntheticZermeloMetric(H_true, W_true)
            solver = EikonalSolver(max_iters=50, tol=1e-5)
            T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
            
            obs_mask = create_sparse_observation_mask(N, N, 50.0/(N*N), source_mask, seed=59)
            
            # Warmup
            opt.fit(source_coords, T_obs, obs_mask, n_iter=1, lr=0.01, verbose=False)
            start = time.time()
            opt.fit(source_coords, T_obs, obs_mask, n_iter=5, lr=0.01, verbose=False)
            time_per_step = (time.time() - start) / 5.0
            
            results_eik.append({'N': N, 'time_per_step': time_per_step})
            print(f"    N={N}: {time_per_step:.4f}s / step")
            
        print("\n  AVBD Scaling (fixed N=100, varying M):")
        N = 100
        G_true = jnp.zeros((3, N, N)).at[0].set(1.0).at[2].set(1.0)
        B_true = jnp.zeros((2, N, N))
        source_coords = jnp.array([[N//2, N//2]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
            
        for M in self.M_sizes:
            opt = MetricRecoveryOptimizer(N, N, solver_type='avbd')
            obs_mask = create_sparse_observation_mask(N, N, M/(N*N), source_mask, seed=60)
            
            # Warmup
            opt.fit(source_coords, T_obs, obs_mask, n_iter=1, lr=0.01, verbose=False)
            start = time.time()
            opt.fit(source_coords, T_obs, obs_mask, n_iter=5, lr=0.01, verbose=False)
            time_per_step = (time.time() - start) / 5.0
            
            results_avbd.append({'M': M, 'time_per_step': time_per_step})
            print(f"    M={M}: {time_per_step:.4f}s / step")
            
        self.results_eik = results_eik
        self.results_avbd = results_avbd
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'eik_base_time': results_eik[0]['time_per_step']},
            arrays={'N_sizes': np.array(self.grid_sizes), 'M_sizes': np.array(self.M_sizes)},
            metadata={'grid_sizes': self.grid_sizes}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        N = np.array([r['N'] for r in self.results_eik])
        t_eik = np.array([r['time_per_step'] for r in self.results_eik])
        
        axes[0].loglog(N, t_eik, 'o-', label='Eikonal Step Time', markersize=8)
        axes[0].loglog(N, t_eik[0] * (N/N[0])**2, 'k--', alpha=0.5, label='O(N²)')
        axes[0].set_xlabel('Grid size N')
        axes[0].set_ylabel('Time (s)')
        axes[0].set_title('Eikonal Scaling with Grid Size N (fixed M=50)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        M = np.array([r['M'] for r in self.results_avbd])
        t_avbd = np.array([r['time_per_step'] for r in self.results_avbd])
        
        axes[1].loglog(M, t_avbd, 's-', color='red', label='AVBD Step Time', markersize=8)
        axes[1].loglog(M, t_avbd[0] * (M/M[0]), 'k--', alpha=0.5, label='O(M)')
        axes[1].set_xlabel('Number of observations M')
        axes[1].set_ylabel('Time (s)')
        axes[1].set_title('AVBD Scaling with Obs M (fixed N=100)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('E5: Scalability Study (Forward + Backward)', fontsize=14)
        plt.tight_layout()
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
