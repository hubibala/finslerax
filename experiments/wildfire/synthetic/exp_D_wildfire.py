import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from experiments.wildfire.synthetic.experiment_base import (
    Experiment, ExperimentResult, register_experiment,
    get_interior_mask, create_sparse_observation_mask,
    plot_arrival_time, plot_error_map, plot_drift_field,
    SyntheticZermeloMetric
)

from ham.solvers.eikonal import EikonalSolver
from experiments.wildfire.synthetic.metric_recovery import (
    MetricRecoveryOptimizer, eikonal_to_zermelo
)

# =============================================================================
# D1: TERRAIN-DRIVEN PROPAGATION
# =============================================================================
@register_experiment
class D1_TerrainDriven(Experiment):
    """Fire spread influenced by terrain slope."""
    
    name = "D1_terrain_driven"
    category = "D_synthetic_wildfire"
    description = "Terrain slope affects fire spread speed"
    
    def __init__(self, N: int = 150):
        super().__init__()
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        i_coords = jnp.arange(N, dtype=jnp.float32)
        j_coords = jnp.arange(N, dtype=jnp.float32)
        I, J = jnp.meshgrid(i_coords, j_coords, indexing='ij')
        
        elevation = 50 * jnp.exp(-((I - N/2)**2 + (J - N/2)**2) / (2 * (N/4)**2))
        
        grad_i = jnp.zeros_like(elevation)
        grad_j = jnp.zeros_like(elevation)
        grad_i = grad_i.at[1:-1, :].set((elevation[2:, :] - elevation[:-2, :]) / 2)
        grad_j = grad_j.at[:, 1:-1].set((elevation[:, 2:] - elevation[:, :-2]) / 2)
        
        slope_mag = jnp.sqrt(grad_i**2 + grad_j**2)
        speed_factor = 1.0 / (1.0 + 0.5 * slope_mag)
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0 / speed_factor**2)
        G = G.at[2].set(1.0 / speed_factor**2)
        
        B = jnp.zeros((2, N, N))
        
        source_coords = jnp.array([[N//4, N//4]], dtype=jnp.float32)
        
        H, W = eikonal_to_zermelo(G, B)
        metric = SyntheticZermeloMetric(H, W)
        
        solver = EikonalSolver(max_iters=100, tol=1e-4)
        T, _, _ = solver.solve(metric, source_coords, (0, N-1, 0, N-1), (N, N))
        
        self.T, self.G, self.elevation = T, G, elevation
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'T_max': float(T.max()), 'slope_max': float(slope_mag.max())},
            arrays={'T': np.array(T), 'elevation': np.array(elevation)},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im = axes[0].imshow(self.elevation, origin='upper', cmap='terrain')
        plt.colorbar(im, ax=axes[0], label='Elevation')
        axes[0].set_title('Terrain')
        
        plot_arrival_time(self.T, ax=axes[1], title='Fire Arrival Time')
        
        im = axes[2].imshow(self.G[0], origin='upper', cmap='hot')
        plt.colorbar(im, ax=axes[2], label='g₁₁')
        axes[2].set_title('Metric (from slope)')
        
        fig.suptitle('D1: Terrain-Driven Fire Spread', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# D2: WIND-DRIVEN PROPAGATION
# =============================================================================
@register_experiment
class D2_WindDriven(Experiment):
    """Fire spread influenced by wind (drift field)."""
    
    name = "D2_wind_driven"
    category = "D_synthetic_wildfire"
    description = "Wind creates asymmetric fire spread"
    
    def __init__(self, N: int = 150, wind_speed: float = 0.25, wind_dir_deg: float = 45):
        super().__init__()
        self.N = N
        self.wind_speed = wind_speed
        self.wind_dir = np.deg2rad(wind_dir_deg)
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0)
        G = G.at[2].set(1.0)
        
        B = jnp.zeros((2, N, N))
        B = B.at[0].set(self.wind_speed * np.cos(self.wind_dir))
        B = B.at[1].set(self.wind_speed * np.sin(self.wind_dir))
        
        source_coords = jnp.array([[N//2, N//2]], dtype=jnp.float32)
        
        H, W = eikonal_to_zermelo(G, B)
        metric_wind = SyntheticZermeloMetric(H, W)
        
        H_nowind, W_nowind = eikonal_to_zermelo(G, jnp.zeros_like(B))
        metric_nowind = SyntheticZermeloMetric(H_nowind, W_nowind)
        
        solver = EikonalSolver(max_iters=100, tol=1e-4)
        T, _, _ = solver.solve(metric_wind, source_coords, (0, N-1, 0, N-1), (N, N))
        T_nowind, _, _ = solver.solve(metric_nowind, source_coords, (0, N-1, 0, N-1), (N, N))
        
        self.T, self.T_nowind, self.B = T, T_nowind, B
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'T_max': float(T.max()), 'T_nowind_max': float(T_nowind.max())},
            arrays={'T': np.array(T), 'T_nowind': np.array(T_nowind)},
            metadata={'wind_speed': self.wind_speed, 'wind_dir_deg': np.rad2deg(self.wind_dir)}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        plot_arrival_time(self.T_nowind, ax=axes[0], title='No Wind')
        
        plot_arrival_time(self.T, ax=axes[1], title='With Wind')
        plot_drift_field(self.B, axes[1], step=15, scale=15, color='white')
        
        diff = self.T - self.T_nowind
        plot_error_map(diff, ax=axes[2], title='Difference (wind effect)')
        
        fig.suptitle('D2: Wind-Driven Fire Spread', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# D3: FUEL HETEROGENEITY
# =============================================================================
@register_experiment
class D3_FuelHeterogeneity(Experiment):
    """Varying fuel loads create speed variations."""
    
    name = "D3_fuel_heterogeneity"
    category = "D_synthetic_wildfire"
    description = "Fuel load variations affect spread rate"
    
    def __init__(self, N: int = 150):
        super().__init__()
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        np.random.seed(52)
        fuel_np = np.ones((N, N))
        
        for _ in range(20):
            ci, cj = np.random.randint(10, N-10, 2)
            radius = np.random.randint(10, 30)
            intensity = np.random.uniform(0.5, 2.0)
            
            i_coords = np.arange(N)
            j_coords = np.arange(N)
            I, J = np.meshgrid(i_coords, j_coords, indexing='ij')
            
            dist = np.sqrt((I - ci)**2 + (J - cj)**2)
            fuel_np[dist < radius] = intensity
            
        from scipy.ndimage import gaussian_filter
        fuel_smooth = gaussian_filter(fuel_np, sigma=2)
        fuel = jnp.array(fuel_smooth)
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0 / (fuel**2 + 0.1))
        G = G.at[2].set(G[0])
        
        B = jnp.zeros((2, N, N))
        source_coords = jnp.array([[N//4, N//4]], dtype=jnp.float32)
        
        H, W = eikonal_to_zermelo(G, B)
        metric = SyntheticZermeloMetric(H, W)
        
        solver = EikonalSolver(max_iters=100, tol=1e-4)
        T, _, _ = solver.solve(metric, source_coords, (0, N-1, 0, N-1), (N, N))
        
        self.T, self.fuel = T, fuel
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'T_max': float(T.max()), 'fuel_range': (float(fuel.min()), float(fuel.max()))},
            arrays={'T': np.array(T), 'fuel': np.array(fuel)},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im = axes[0].imshow(self.fuel, origin='upper', cmap='YlOrRd')
        plt.colorbar(im, ax=axes[0], label='Fuel Load')
        axes[0].set_title('Fuel Distribution')
        
        plot_arrival_time(self.T, ax=axes[1], title='Fire Arrival Time')
        
        fig.suptitle('D3: Fuel Heterogeneity', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# D4: COMBINED TERRAIN + WIND
# =============================================================================
@register_experiment
class D4_CombinedScenario(Experiment):
    """Full realistic scenario with terrain and wind."""
    
    name = "D4_combined_scenario"
    category = "D_synthetic_wildfire"
    description = "Combined terrain, fuel, and wind effects"
    
    def __init__(self, N: int = 200):
        super().__init__()
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        i_coords = jnp.arange(N, dtype=jnp.float32)
        j_coords = jnp.arange(N, dtype=jnp.float32)
        I, J = jnp.meshgrid(i_coords, j_coords, indexing='ij')
        
        elevation = (30 * jnp.sin(2 * np.pi * I / N) * jnp.cos(np.pi * J / N) + 
                    20 * jnp.exp(-((I - N/2)**2 + (J - N/3)**2) / (2 * (N/6)**2)))
        
        grad_i = jnp.zeros_like(elevation)
        grad_j = jnp.zeros_like(elevation)
        grad_i = grad_i.at[1:-1, :].set((elevation[2:, :] - elevation[:-2, :]) / 2)
        grad_j = grad_j.at[:, 1:-1].set((elevation[:, 2:] - elevation[:, :-2]) / 2)
        slope_mag = jnp.sqrt(grad_i**2 + grad_j**2)
        
        np.random.seed(53)
        fuel_np = np.ones((N, N))
        for _ in range(15):
            ci, cj = np.random.randint(20, N-20, 2)
            radius = np.random.randint(15, 40)
            intensity = np.random.uniform(0.6, 1.5)
            I_np, J_np = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
            dist = np.sqrt((I_np - ci)**2 + (J_np - cj)**2)
            fuel_np[dist < radius] = intensity
            
        fuel = jnp.array(fuel_np)
        speed = fuel / (1.0 + 0.3 * slope_mag)
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0 / (speed**2 + 0.1))
        G = G.at[2].set(G[0])
        
        B = jnp.zeros((2, N, N))
        B = B.at[0].set(0.15)
        B = B.at[1].set(0.1)
        
        source_coords = jnp.array([[N//4, N//4]], dtype=jnp.float32)
        
        H, W = eikonal_to_zermelo(G, B)
        metric = SyntheticZermeloMetric(H, W)
        
        solver = EikonalSolver(max_iters=100, tol=1e-4)
        T, _, _ = solver.solve(metric, source_coords, (0, N-1, 0, N-1), (N, N))
        
        self.T, self.G, self.B = T, G, B
        self.elevation, self.fuel = elevation, fuel
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'T_max': float(T.max())},
            arrays={'T': np.array(T), 'elevation': np.array(elevation), 'fuel': np.array(fuel)},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        im = axes[0,0].imshow(self.elevation, origin='upper', cmap='terrain')
        plt.colorbar(im, ax=axes[0,0], label='Elevation')
        axes[0,0].set_title('Terrain')
        
        im = axes[0,1].imshow(self.fuel, origin='upper', cmap='YlOrRd')
        plt.colorbar(im, ax=axes[0,1], label='Fuel')
        axes[0,1].set_title('Fuel')
        
        plot_arrival_time(self.T, ax=axes[1,0], title='Arrival Time')
        plot_drift_field(self.B, axes[1,0], step=20, scale=20, color='white')
        
        im = axes[1,1].imshow(self.G[0], origin='upper', cmap='hot')
        plt.colorbar(im, ax=axes[1,1], label='g₁₁')
        axes[1,1].set_title('Effective Metric')
        
        fig.suptitle('D4: Combined Wildfire Scenario', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# D5: FIRE LINE RECONSTRUCTION
# =============================================================================
@register_experiment
class D5_FireLineReconstruction(Experiment):
    """Reconstruct fire evolution from sparse observations (Dual Solver Comparison)."""
    
    name = "D5_fire_line_reconstruction"
    category = "D_synthetic_wildfire"
    description = "Reconstruct fire lines from sparse arrival time obs"
    
    def __init__(self, N: int = 50, n_iter: int = 300):
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
        B_true = B_true.at[0].set(0.1)
        
        source_coords = jnp.array([[N//2, N//4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//4].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_true, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=54)
        
        print("\n  Training Eikonal Optimizer...")
        opt_eik = MetricRecoveryOptimizer(N, N, solver_type='eikonal', recover_H=True, recover_W=True, 
                                          lambda_H=0.01, lambda_W=0.01, constrain_isotropic=True)
        opt_eik.fit(source_coords, T_true, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
        H_rec_eik, W_rec_eik = opt_eik.model.H_grid, opt_eik.model.W_grid
        
        print("  Training AVBD Optimizer...")
        opt_avbd = MetricRecoveryOptimizer(N, N, solver_type='avbd', recover_H=True, recover_W=True, 
                                           lambda_H=0.01, lambda_W=0.01, constrain_isotropic=True)
        opt_avbd.fit(source_coords, T_true, obs_mask, n_iter=self.n_iter, lr=0.05, verbose=True)
        H_rec_avbd, W_rec_avbd = opt_avbd.model.H_grid, opt_avbd.model.W_grid
        
        # Predict full fields using Eikonal Solver
        metric_eik = SyntheticZermeloMetric(H_rec_eik, W_rec_eik)
        metric_avbd = SyntheticZermeloMetric(H_rec_avbd, W_rec_avbd)
        T_pred_eik, _, _ = solver.solve(metric_eik, source_coords, (0, N-1, 0, N-1), (N, N))
        T_pred_avbd, _, _ = solver.solve(metric_avbd, source_coords, (0, N-1, 0, N-1), (N, N))
        
        valid = jnp.isfinite(T_pred_eik)
        err_eik = float(jnp.abs(T_true[valid] - T_pred_eik[valid]).mean())
        err_avbd = float(jnp.abs(T_true[valid] - T_pred_avbd[valid]).mean())
        
        print(f"  Eikonal T Error: {err_eik:.4f}")
        print(f"  AVBD T Error: {err_avbd:.4f}")
        
        self.T_true, self.T_pred_eik, self.T_pred_avbd, self.obs_mask = T_true, T_pred_eik, T_pred_avbd, obs_mask
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=err_eik < 5.0 and err_avbd < 5.0,
            metrics={'err_eikonal': err_eik, 'err_avbd': err_avbd},
            arrays={'T_true': np.array(T_true), 'T_eik': np.array(T_pred_eik), 'T_avbd': np.array(T_pred_avbd)},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        plot_arrival_time(self.T_true, ax=axes[0], title='True Arrival Time')
        oy, ox = jnp.where(self.obs_mask)
        axes[0].scatter(ox, oy, c='red', s=5, alpha=0.5)
        
        plot_arrival_time(self.T_pred_eik, ax=axes[1], title=f'Eikonal Reconstruction (MAE={self.result.metrics["err_eikonal"]:.2f})')
        plot_arrival_time(self.T_pred_avbd, ax=axes[2], title=f'AVBD Reconstruction (MAE={self.result.metrics["err_avbd"]:.2f})')
        
        fig.suptitle('D5: Fire Line Reconstruction', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# D6: PARAMETER SENSITIVITY
# =============================================================================
@register_experiment
class D6_ParameterSensitivity(Experiment):
    """How do errors in G/B affect arrival time predictions?"""
    
    name = "D6_parameter_sensitivity"
    category = "D_synthetic_wildfire"
    description = "Sensitivity of predictions to parameter errors"
    
    def __init__(self, N: int = 100):
        super().__init__()
        self.N = N
        self.perturbation_levels = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N//2:].set(2.0)
        G_true = G_true.at[2, :, N//2:].set(2.0)
        
        B_true = jnp.zeros((2, N, N))
        B_true = B_true.at[0].set(0.15)
        
        source_coords = jnp.array([[N//2, N//2]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        
        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_true, _, _ = solver.solve(metric_true, source_coords, (0, N-1, 0, N-1), (N, N))
        
        results = []
        np.random.seed(55)
        
        for pert in self.perturbation_levels:
            G_pert = G_true * (1 + pert * jnp.array(np.random.randn(3, N, N)))
            G_pert = G_pert.at[0].set(jnp.clip(G_pert[0], a_min=0.1))
            G_pert = G_pert.at[2].set(jnp.clip(G_pert[2], a_min=0.1))
            
            H_pert, W_pert = eikonal_to_zermelo(G_pert, B_true)
            metric_pert = SyntheticZermeloMetric(H_pert, W_pert)
            T_pert, _, _ = solver.solve(metric_pert, source_coords, (0, N-1, 0, N-1), (N, N))
            
            interior = get_interior_mask(N, N, 5, source_mask)
            rel_err = float(jnp.sqrt(jnp.mean((T_true[interior] - T_pert[interior])**2)) / jnp.abs(T_true[interior]).max())
            
            results.append({'perturbation': pert, 'T_error': rel_err})
            print(f"  Pert={pert*100:.0f}%: T error={rel_err:.4f}")
            
        self.results = results
        self.T_true = T_true
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'error_at_10pct': results[2]['T_error'], 'error_at_30pct': results[4]['T_error']},
            arrays={'perturbations': np.array([r['perturbation'] for r in results]),
                   'errors': np.array([r['T_error'] for r in results])},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pert = np.array([r['perturbation'] for r in self.results]) * 100
        err = np.array([r['T_error'] for r in self.results]) * 100
        
        ax.plot(pert, err, 'o-', markersize=10, lw=2)
        ax.set_xlabel('Parameter Perturbation (%)')
        ax.set_ylabel('Arrival Time Error (%)')
        ax.set_title('D6: Parameter Sensitivity Analysis')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    D1_TerrainDriven,
    D2_WindDriven,
    D3_FuelHeterogeneity,
    D4_CombinedScenario,
    D5_FireLineReconstruction,
    D6_ParameterSensitivity,
]

def run_all(save=True, visualize=True):
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)
    
    print("\n" + "="*60)
    print("CATEGORY D SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results

if __name__ == "__main__":
    run_all()
