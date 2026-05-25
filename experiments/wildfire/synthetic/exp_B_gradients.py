import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from experiments.wildfire.synthetic.experiment_base import (
    Experiment, ExperimentResult, register_experiment,
    create_metric_from_eigenvalues, plot_arrival_time, plot_error_map
)
from ham.solvers.eikonal import _fast_sweeping_solve
import functools

@functools.partial(jax.jit, static_argnames=('hx', 'hy', 'max_iters', 'tol'))
def compute_loss_jitted(G_in, B_in, source_mask, dL_dT, hx, hy, max_iters, tol):
    T = _fast_sweeping_solve(G_in, B_in, source_mask, hx, hy, max_iters, tol)
    return jnp.sum(T * dL_dT)

val_and_grad_jitted = jax.jit(
    jax.value_and_grad(compute_loss_jitted, argnums=(0, 1)),
    static_argnames=('hx', 'hy', 'max_iters', 'tol')
)

def backward_pass(G, B, source_mask, dL_dT, hx=1.0, hy=1.0, max_iters=50, tol=1e-4):
    """Wrapper to get analytical gradients."""
    loss, grads = val_and_grad_jitted(G, B, source_mask, dL_dT, hx, hy, max_iters, tol)
    return grads[0], grads[1]

def finite_difference_gradient(G, B, source_mask, dL_dT, param, indices, eps=1e-5, hx=1.0, hy=1.0, max_iters=50, tol=1e-4):
    """Compute gradient via central finite differences."""
    c, i, j = indices
    
    if param == 'G':
        G_plus = G.at[c, i, j].add(eps)
        G_minus = G.at[c, i, j].add(-eps)
        L_plus = compute_loss_jitted(G_plus, B, source_mask, dL_dT, hx, hy, max_iters, tol)
        L_minus = compute_loss_jitted(G_minus, B, source_mask, dL_dT, hx, hy, max_iters, tol)
    else:
        B_plus = B.at[c, i, j].add(eps)
        B_minus = B.at[c, i, j].add(-eps)
        L_plus = compute_loss_jitted(G, B_plus, source_mask, dL_dT, hx, hy, max_iters, tol)
        L_minus = compute_loss_jitted(G, B_minus, source_mask, dL_dT, hx, hy, max_iters, tol)
        
    return float((L_plus - L_minus) / (2 * eps))


# =============================================================================
# B1: FD VS IMPLICIT (ISOTROPIC)
# =============================================================================
@register_experiment
class B1_FD_Isotropic(Experiment):
    """Verify gradients on isotropic problem via finite differences."""
    
    name = "B1_fd_vs_implicit_isotropic"
    category = "B_gradient_verification"
    description = "Verify backward pass gradients against FD (isotropic)"
    
    def __init__(self, N: int = 50, n_test_points: int = 20, eps: float = 1e-4):
        super().__init__()
        self.N, self.n_test_points, self.eps = N, n_test_points, eps
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0)
        G = G.at[2].set(1.0)
        B = jnp.zeros((2, N, N))
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        
        T = _fast_sweeping_solve(G, B, source_mask, 1.0, 1.0, 50, 1e-4)
        
        target_i, target_j = N//2 + 15, N//2 + 10
        dL_dT = jnp.zeros_like(T).at[target_i, target_j].set(1.0)
        
        dL_dG, dL_dB = backward_pass(G, B, source_mask, dL_dT)
        
        np.random.seed(42)
        test_points = []
        while len(test_points) < self.n_test_points:
            i, j = np.random.randint(5, N-5, 2)
            if abs(i - N//2) > 3 or abs(j - N//2) > 3:
                test_points.append((i, j))
        
        results_G, results_B = [], []
        
        for i, j in test_points:
            for c in [0, 2]:
                fd = finite_difference_gradient(G, B, source_mask, dL_dT, 'G', (c, i, j), self.eps)
                impl = float(dL_dG[c, i, j])
                rel_err = abs(fd - impl) / abs(fd) if abs(fd) > 1e-10 else abs(impl)
                results_G.append({'fd': fd, 'impl': impl, 'rel_err': rel_err})
            
            for c in [0, 1]:
                fd = finite_difference_gradient(G, B, source_mask, dL_dT, 'B', (c, i, j), self.eps)
                impl = float(dL_dB[c, i, j])
                rel_err = abs(fd - impl) / abs(fd) if abs(fd) > 1e-10 else abs(impl)
                results_B.append({'fd': fd, 'impl': impl, 'rel_err': rel_err})
        
        G_errors = [r['rel_err'] for r in results_G if r['rel_err'] < 10]
        B_errors = [r['rel_err'] for r in results_B if r['rel_err'] < 10]
        
        pct_good_G = sum(1 for e in G_errors if e < 0.1) / len(G_errors) * 100 if G_errors else 0
        pct_good_B = sum(1 for e in B_errors if e < 0.1) / len(B_errors) * 100 if B_errors else 0
        
        print(f"  G: {pct_good_G:.1f}% < 10% err, B: {pct_good_B:.1f}% < 10% err")
        
        self.results_G, self.results_B = results_G, results_B
        self.T, self.dL_dG, self.dL_dB = T, dL_dG, dL_dB
        self.target = (target_i, target_j)
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=(pct_good_G > 80 and pct_good_B > 80),
            metrics={'pct_good_G': pct_good_G, 'pct_good_B': pct_good_B,
                    'mean_err_G': float(np.mean(G_errors)), 'mean_err_B': float(np.mean(B_errors))},
            arrays={'G_fd': np.array([r['fd'] for r in results_G]),
                   'G_impl': np.array([r['impl'] for r in results_G]),
                   'B_fd': np.array([r['fd'] for r in results_B]),
                   'B_impl': np.array([r['impl'] for r in results_B])},
            metadata={'N': N, 'n_test': self.n_test_points}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for ax, results, name in [(axes[0,0], self.results_G, 'G'), (axes[0,1], self.results_B, 'B')]:
            fd = np.array([r['fd'] for r in results])
            impl = np.array([r['impl'] for r in results])
            ax.scatter(fd, impl, alpha=0.6)
            lim = max(abs(fd).max(), abs(impl).max()) * 1.1
            if lim > 0:
                ax.plot([-lim, lim], [-lim, lim], 'r--')
            ax.set_xlabel('Finite Difference')
            ax.set_ylabel('Implicit')
            ax.set_title(f'{name} gradients')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        B_mag = np.array(jnp.sqrt(self.dL_dB[0]**2 + self.dL_dB[1]**2))
        G_mag = np.array(jnp.sqrt(self.dL_dG[0]**2 + self.dL_dG[2]**2))
        
        for ax, mag, name in [(axes[1,0], B_mag, 'B'), (axes[1,1], G_mag, 'G')]:
            im = ax.imshow(mag, origin='upper', cmap='hot')
            plt.colorbar(im, ax=ax)
            ax.plot(self.target[1], self.target[0], 'c*', markersize=15)
            ax.plot(self.N//2, self.N//2, 'g*', markersize=15)
            ax.set_title(f'|∇_{name} L|')
        
        fig.suptitle('B1: Gradient Verification (Isotropic)', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B2: FD VS IMPLICIT (ANISOTROPIC)
# =============================================================================
@register_experiment
class B2_FD_Anisotropic(Experiment):
    """Verify gradients for anisotropic metric (all 3 components)."""
    
    name = "B2_fd_vs_implicit_anisotropic"
    category = "B_gradient_verification"
    description = "Verify gradients for anisotropic metric"
    
    def __init__(self, N: int = 50, n_test_points: int = 15, eps: float = 1e-4):
        super().__init__()
        self.N, self.n_test_points, self.eps = N, n_test_points, eps
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = create_metric_from_eigenvalues(N, N, 2.0, 0.5, np.deg2rad(30))
        B = jnp.zeros((2, N, N))
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        
        T = _fast_sweeping_solve(G, B, source_mask, 1.0, 1.0, 50, 1e-4)
        
        dL_dT = jnp.zeros_like(T).at[N//2 + 12, N//2 + 8].set(1.0)
        
        dL_dG, _ = backward_pass(G, B, source_mask, dL_dT)
        
        np.random.seed(43)
        test_points = []
        while len(test_points) < self.n_test_points:
            i, j = np.random.randint(5, N-5, 2)
            if abs(i - N//2) > 3 or abs(j - N//2) > 3:
                test_points.append((i, j))
        
        results = {0: [], 1: [], 2: []}
        for i, j in test_points:
            for c in range(3):
                fd = finite_difference_gradient(G, B, source_mask, dL_dT, 'G', (c, i, j), self.eps)
                impl = float(dL_dG[c, i, j])
                rel_err = abs(fd - impl) / abs(fd) if abs(fd) > 1e-10 else abs(impl)
                results[c].append({'fd': fd, 'impl': impl, 'rel_err': rel_err})
        
        metrics = {}
        for c, name in [(0, 'g11'), (1, 'g12'), (2, 'g22')]:
            errs = [r['rel_err'] for r in results[c] if r['rel_err'] < 10]
            if errs:
                metrics[f'pct_good_{name}'] = sum(1 for e in errs if e < 0.1) / len(errs) * 100
                print(f"  {name}: {metrics[f'pct_good_{name}']:.1f}% < 10% err")
        
        self.results = results
        pct_good = {c: 100 * sum(1 for r in results[c] if r['rel_err'] < 0.1) / len(results[c])
                    for c in [0, 1, 2]}
        return ExperimentResult(
            name=self.name, category=self.category,
            success=all(pct > 70 for pct in pct_good.values()),
            metrics={**{f'pct_good_{n}': pct_good[c] for c, n in [(0,'g11'),(1,'g12'),(2,'g22')]},
                     **{f'mean_err_{n}': float(np.mean([r['rel_err'] for r in results[c]]))
                        for c, n in [(0,'g11'),(1,'g12'),(2,'g22')]}},
            arrays={**{f'{n}_fd': np.array([r['fd'] for r in results[c]])
                       for c, n in [(0,'g11'),(1,'g12'),(2,'g22')]},
                    **{f'{n}_impl': np.array([r['impl'] for r in results[c]])
                       for c, n in [(0,'g11'),(1,'g12'),(2,'g22')]}},
            metadata={'N': self.N, 'n_test_points': self.n_test_points}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (c, name) in zip(axes, [(0, 'g₁₁'), (1, 'g₁₂'), (2, 'g₂₂')]):
            fd = np.array([r['fd'] for r in self.results[c]])
            impl = np.array([r['impl'] for r in self.results[c]])
            ax.scatter(fd, impl, alpha=0.6, s=50)
            lim = max(abs(fd).max(), abs(impl).max()) * 1.1
            if lim > 0:
                ax.plot([-lim, lim], [-lim, lim], 'r--')
            ax.set_xlabel('FD')
            ax.set_ylabel('Implicit')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        fig.suptitle('B2: Anisotropic Gradient Verification', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B3: FD VS IMPLICIT (DRIFT)
# =============================================================================
@register_experiment
class B3_FD_Drift(Experiment):
    """Verify B gradients with nonzero drift."""
    
    name = "B3_fd_vs_implicit_drift"
    category = "B_gradient_verification"
    description = "Verify drift gradients"
    
    def __init__(self, N: int = 50, n_test_points: int = 15, eps: float = 1e-4):
        super().__init__()
        self.N, self.n_test_points, self.eps = N, n_test_points, eps
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0)
        G = G.at[2].set(1.0)
        B = jnp.zeros((2, N, N))
        B = B.at[0].set(0.15)
        B = B.at[1].set(0.08)
        
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        
        T = _fast_sweeping_solve(G, B, source_mask, 1.0, 1.0, 50, 1e-4)
        
        dL_dT = jnp.zeros_like(T).at[N//2 + 10, N//2 + 10].set(1.0)
        
        _, dL_dB = backward_pass(G, B, source_mask, dL_dT)
        
        np.random.seed(44)
        test_points = []
        while len(test_points) < self.n_test_points:
            i, j = np.random.randint(5, N-5, 2)
            if abs(i - N//2) > 3 or abs(j - N//2) > 3:
                test_points.append((i, j))
        
        results = {0: [], 1: []}
        for i, j in test_points:
            for c in range(2):
                fd = finite_difference_gradient(G, B, source_mask, dL_dT, 'B', (c, i, j), self.eps)
                impl = float(dL_dB[c, i, j])
                rel_err = abs(fd - impl) / abs(fd) if abs(fd) > 1e-10 else abs(impl)
                results[c].append({'fd': fd, 'impl': impl, 'rel_err': rel_err})
        
        metrics = {}
        for c, name in [(0, 'b1'), (1, 'b2')]:
            errs = [r['rel_err'] for r in results[c] if r['rel_err'] < 10]
            if errs:
                metrics[f'pct_good_{name}'] = sum(1 for e in errs if e < 0.1) / len(errs) * 100
        
        self.results = results
        pct_good = {c: 100 * sum(1 for r in results[c] if r['rel_err'] < 0.1) / len(results[c])
                    for c in [0, 1]}
        return ExperimentResult(
            name=self.name, category=self.category,
            success=all(pct > 70 for pct in pct_good.values()),
            metrics={**{f'pct_good_{n}': pct_good[c] for c, n in [(0, 'b1'), (1, 'b2')]},
                     **{f'mean_err_{n}': float(np.mean([r['rel_err'] for r in results[c]]))
                        for c, n in [(0, 'b1'), (1, 'b2')]}},
            arrays={**{f'{n}_fd': np.array([r['fd'] for r in results[c]])
                       for c, n in [(0, 'b1'), (1, 'b2')]},
                    **{f'{n}_impl': np.array([r['impl'] for r in results[c]])
                       for c, n in [(0, 'b1'), (1, 'b2')]}},
            metadata={'N': self.N, 'n_test_points': self.n_test_points}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (c, name) in zip(axes, [(0, 'b₁'), (1, 'b₂')]):
            fd = np.array([r['fd'] for r in self.results[c]])
            impl = np.array([r['impl'] for r in self.results[c]])
            ax.scatter(fd, impl, alpha=0.6, s=50)
            lim = max(abs(fd).max(), abs(impl).max()) * 1.1
            if lim > 0:
                ax.plot([-lim, lim], [-lim, lim], 'r--')
            ax.set_xlabel('FD')
            ax.set_ylabel('Implicit')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        fig.suptitle('B3: Drift Gradient Verification', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B4: RING LOSS PATTERN
# =============================================================================
@register_experiment
class B4_RingLoss(Experiment):
    """Visualize adjoint propagation pattern for ring loss."""
    
    name = "B4_ring_loss_pattern"
    category = "B_gradient_verification"
    description = "Adjoint propagation with ring loss"
    
    def __init__(self, N: int = 100):
        super().__init__()
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0)
        G = G.at[2].set(1.0)
        B = jnp.zeros((2, N, N))
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        
        T = _fast_sweeping_solve(G, B, source_mask, 1.0, 1.0, 50, 1e-4)
        
        dL_dT = jnp.zeros_like(T)
        radius = 25
        for theta in np.linspace(0, 2*np.pi, 32, endpoint=False):
            i = int(N//2 + radius * np.sin(theta))
            j = int(N//2 + radius * np.cos(theta))
            if 0 <= i < N and 0 <= j < N:
                dL_dT = dL_dT.at[i, j].set(1.0)
        
        dL_dG, dL_dB = backward_pass(G, B, source_mask, dL_dT)
        
        G_mag = jnp.sqrt(dL_dG[0]**2 + dL_dG[2]**2)
        B_mag = jnp.sqrt(dL_dB[0]**2 + dL_dB[1]**2)
        
        self.T, self.dL_dT, self.G_mag, self.B_mag = T, dL_dT, G_mag, B_mag
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'G_max': float(G_mag.max()), 'B_max': float(B_mag.max())},
            arrays={'T': np.array(T), 'G_mag': np.array(G_mag), 'B_mag': np.array(B_mag)},
            metadata={'N': N, 'radius': radius}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        
        plot_arrival_time(self.T, ax=axes[0], title='T')
        
        axes[1].imshow(np.array(self.dL_dT), origin='upper', cmap='Reds')
        axes[1].set_title('dL/dT (ring)')
        
        im = axes[2].imshow(np.array(self.G_mag), origin='upper', cmap='hot')
        plt.colorbar(im, ax=axes[2])
        axes[2].set_title('|∇_G L|')
        
        im = axes[3].imshow(np.array(self.B_mag), origin='upper', cmap='hot')
        plt.colorbar(im, ax=axes[3])
        axes[3].set_title('|∇_B L|')
        
        fig.suptitle('B4: Ring Loss Adjoint Pattern', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B5: GRADIENT MAGNITUDE DECAY
# =============================================================================
@register_experiment
class B5_GradientDecay(Experiment):
    """Analyze gradient magnitude decay with distance from target."""
    
    name = "B5_gradient_decay"
    category = "B_gradient_verification"
    description = "Gradient magnitude vs distance from target"
    
    def __init__(self, N: int = 100):
        super().__init__()
        self.N = N
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0)
        G = G.at[2].set(1.0)
        B = jnp.zeros((2, N, N))
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        
        target_i, target_j = N//2 + 20, N//2 + 20
        dL_dT = jnp.zeros((N, N)).at[target_i, target_j].set(1.0)
        
        dL_dG, _ = backward_pass(G, B, source_mask, dL_dT)
        
        G_mag = np.array(jnp.sqrt(dL_dG[0]**2 + dL_dG[2]**2))
        
        I, J = np.meshgrid(np.arange(N) - target_i, np.arange(N) - target_j, indexing='ij')
        dist = np.sqrt(I**2 + J**2)
        
        distances, magnitudes = [], []
        for d in range(1, 40):
            mask = (dist >= d - 0.5) & (dist < d + 0.5)
            if mask.sum() > 0:
                distances.append(d)
                magnitudes.append(float(G_mag[mask].mean()))
        
        self.distances, self.magnitudes = distances, magnitudes
        self.G_mag = G_mag
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=True,
            metrics={'max_grad': float(G_mag.max())},
            arrays={'distances': np.array(distances), 'magnitudes': np.array(magnitudes)},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im = axes[0].imshow(self.G_mag, origin='upper', cmap='hot')
        plt.colorbar(im, ax=axes[0])
        axes[0].set_title('|∇_G L|')
        
        axes[1].semilogy(self.distances, self.magnitudes, 'o-')
        axes[1].set_xlabel('Distance from target')
        axes[1].set_ylabel('Mean |∇_G L|')
        axes[1].set_title('Gradient Decay')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('B5: Gradient Magnitude Decay', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B8: RANDOM DIRECTIONS
# =============================================================================
@register_experiment
class B8_RandomDirections(Experiment):
    """Test gradient agreement along random directions."""
    
    name = "B8_random_directions"
    category = "B_gradient_verification"
    description = "Verify gradient correct a.e. via random directions"
    
    def __init__(self, N: int = 50, n_directions: int = 100):
        super().__init__()
        self.N, self.n_directions = N, n_directions
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0)
        G = G.at[2].set(1.0)
        B = jnp.zeros((2, N, N))
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        
        T = _fast_sweeping_solve(G, B, source_mask, 1.0, 1.0, 50, 1e-4)
        
        dL_dT = jnp.zeros_like(T).at[N//2 + 10, N//2 + 10].set(1.0)
        
        _, dL_dB = backward_pass(G, B, source_mask, dL_dT)
        
        eps = 1e-4
        np.random.seed(45)
        
        errors = []
        for _ in range(self.n_directions):
            dB_np = np.zeros((2, N, N))
            dB_np[:, 5:-5, 5:-5] = np.random.randn(2, N-10, N-10)
            dB = jnp.array(dB_np)
            dB = dB / jnp.linalg.norm(dB)
            
            impl_dir = float(jnp.sum(dL_dB * dB))
            
            B_plus = B + eps * dB
            B_minus = B - eps * dB
            
            T_plus = _fast_sweeping_solve(G, B_plus, source_mask, 1.0, 1.0, 50, 1e-4)
            T_minus = _fast_sweeping_solve(G, B_minus, source_mask, 1.0, 1.0, 50, 1e-4)
            
            fd_dir = float(jnp.sum((T_plus - T_minus) * dL_dT) / (2 * eps))
            
            rel_err = abs(fd_dir - impl_dir) / abs(fd_dir) if abs(fd_dir) > 1e-10 else abs(impl_dir)
            errors.append(rel_err)
        
        errors = np.array(errors)
        pct_good = (errors < 0.1).sum() / len(errors) * 100
        print(f"  {pct_good:.1f}% directions < 10% error")
        
        self.errors = errors
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=pct_good > 10,
            metrics={'pct_good': pct_good, 'median_err': float(np.median(errors))},
            arrays={'errors': errors},
            metadata={'N': N, 'n_directions': self.n_directions}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(self.errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(0.1, color='r', linestyle='--', label='10%')
        axes[0].set_xlabel('Relative Error')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'{self.result.metrics["pct_good"]:.1f}% < 10%')
        axes[0].legend()
        
        sorted_err = np.sort(self.errors)
        cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
        axes[1].plot(sorted_err, cdf)
        axes[1].axvline(0.1, color='r', linestyle='--')
        axes[1].axhline(0.9, color='g', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Error')
        axes[1].set_ylabel('CDF')
        axes[1].set_title('CDF')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle('B8: Random Direction Verification', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# B9: GRADIENT UNDER PERTURBATION
# =============================================================================
@register_experiment
class B9_GradientStability(Experiment):
    """Test gradient stability under small parameter perturbations."""
    
    name = "B9_gradient_stability"
    category = "B_gradient_verification"
    description = "Gradient stability under perturbations"
    
    def __init__(self, N: int = 50, n_perturbations: int = 20):
        super().__init__()
        self.N, self.n_perturbations = N, n_perturbations
        
    def run(self) -> ExperimentResult:
        N = self.N
        
        G = jnp.zeros((3, N, N))
        G = G.at[0].set(1.0)
        G = G.at[2].set(1.0)
        B = jnp.zeros((2, N, N))
        source_mask = jnp.zeros((N, N), dtype=bool).at[N//2, N//2].set(True)
        
        dL_dT = jnp.zeros((N, N)).at[N//2 + 10, N//2 + 10].set(1.0)
        
        dL_dG_base, _ = backward_pass(G, B, source_mask, dL_dT)
        
        np.random.seed(46)
        variations = []
        
        for _ in range(self.n_perturbations):
            G_pert = G + 0.01 * jnp.array(np.random.randn(3, N, N))
            G_pert = G_pert.at[0].set(jnp.clip(G_pert[0], a_min=0.5))
            G_pert = G_pert.at[2].set(jnp.clip(G_pert[2], a_min=0.5))
            
            dL_dG_pert, _ = backward_pass(G_pert, B, source_mask, dL_dT)
            
            rel_diff = float(jnp.mean(jnp.abs(dL_dG_pert - dL_dG_base)) / jnp.mean(jnp.abs(dL_dG_base)))
            variations.append(rel_diff)
        
        variations = np.array(variations)
        
        return ExperimentResult(
            name=self.name, category=self.category,
            success=np.mean(variations) < 0.5,
            metrics={'mean_variation': float(np.mean(variations)), 
                    'max_variation': float(np.max(variations))},
            arrays={'variations': variations},
            metadata={'N': N}
        )
    
    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(self.result.arrays['variations'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(self.result.arrays['variations']), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(self.result.arrays["variations"]):.3f}')
        ax.set_xlabel('Relative Gradient Variation')
        ax.set_ylabel('Count')
        ax.set_title('B9: Gradient Stability')
        ax.legend()
        
        fig.suptitle('B9: Gradient Under Perturbation', fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
        return fig


# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    B1_FD_Isotropic,
    B2_FD_Anisotropic,
    B3_FD_Drift,
    B4_RingLoss,
    B5_GradientDecay,
    B8_RandomDirections,
    B9_GradientStability,
]

def run_all(save=True, visualize=True):
    """Run all Category B experiments."""
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)
    
    print("\n" + "="*60)
    print("CATEGORY B SUMMARY")
    print("="*60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results

if __name__ == "__main__":
    run_all()
