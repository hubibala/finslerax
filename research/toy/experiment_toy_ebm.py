"""
Toy 2D experiment for EBM-Finsler Geodesics.
Generates a 2D Gaussian Mixture / Bifurcation, trains the EBM with CD, 
and solves geodesics on the learned energy landscape.
"""
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt

from ham.nn.ebm import ScalarEnergyField
from ham.models.learned import EnergyBasedRanders
from ham.geometry.manifold import Manifold
from ham.geometry.manifolds.euclidean_space import EuclideanSpace
from ham.solvers.avbd import AVBDSolver
from ham.training.losses_ebm import ContrastiveDivergenceLoss
from ham.training.pipeline import HAMPipeline, TrainingPhase

def generate_bifurcation_data(n_samples=1000):
    t = np.random.uniform(0, 1, n_samples)
    branch = np.random.choice([1, -1], n_samples)
    
    x = 2 * t
    y = branch * 2 * t
    
    noise = np.random.normal(0, 0.2, (n_samples, 2))
    data = np.stack([x, y], axis=1) + noise
    return data.astype(np.float32)

class ToyModel(eqx.Module):
    ebm: ScalarEnergyField
    metric: EnergyBasedRanders
    solver: AVBDSolver
    manifold: Manifold
    
    def __init__(self, key):
        self.manifold = EuclideanSpace(2)
        # Using softplus in MLP, no fourier features to keep energy smooth globally
        self.ebm = ScalarEnergyField(2, 64, 3, key, use_fourier=False)
        self.metric = EnergyBasedRanders(self.manifold, self.ebm, wind_scale=1.0, beta=100)
        self.solver = AVBDSolver(step_size=0.5, iterations=100, tol=1e-6, implicit_diff=True)

class DummyDataset:
    def __init__(self, X):
        self.X = X
        self.V = np.zeros_like(X)
        self.labels = None
        self.lineage_pairs = None

def main():
    key = jax.random.PRNGKey(42)
    data = generate_bifurcation_data(2000)
    dataset = DummyDataset(data)
    
    model = ToyModel(key)
    
    # Only train the EBM weights
    def filter_spec(m):
        return jax.tree_util.tree_map(lambda leaf: eqx.is_inexact_array(leaf), m)
        
    phase = TrainingPhase(
        name="EBM_CD",
        epochs=150,
        optimizer=optax.adam(1e-3),
        losses=[ContrastiveDivergenceLoss(sgld_steps=20, sgld_step_size=0.02, sgld_noise_scale=0.05, l2_reg_weight=0.005)],
        filter_spec=filter_spec,
        requires_pairs=False
    )
    
    pipeline = HAMPipeline(model)
    print("Training EBM...")
    trained_model = pipeline.fit(dataset, [phase], batch_size=256, seed=42)
    
    # Ensure the metric uses the trained EBM
    trained_model = eqx.tree_at(lambda m: m.metric.ebm, trained_model, trained_model.ebm)
    
    print("Computing Landscape...")
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 3, 50), np.linspace(-3, 3, 50))
    grid_pts = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    
    energies = jax.vmap(trained_model.ebm)(grid_pts)
    energies = energies.reshape(50, 50)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, energies, levels=50, cmap='viridis')
    plt.colorbar(label='Energy')
    
    # Visualize the wind field
    def get_wind(x):
        _, w_safe, _ = trained_model.metric.zermelo_data(x)
        return w_safe
    wind_fn = jax.vmap(get_wind)
    W = wind_fn(grid_pts)
    W_x = W[:, 0].reshape(grid_x.shape)
    W_y = W[:, 1].reshape(grid_y.shape)
    
    stride = 5
    plt.quiver(grid_x[::stride, ::stride], grid_y[::stride, ::stride], 
               W_x[::stride, ::stride], W_y[::stride, ::stride], 
               color='white', alpha=0.5, scale=100)
    
    plt.scatter(dataset.X[:, 0], dataset.X[:, 1], c='white', s=5, alpha=0.5, label='Data')
    
    pt_root = jnp.array([0.0, 0.0])
    pt_b1 = jnp.array([2.0, 2.0])
    pt_b2 = jnp.array([2.0, -2.0])
    
    print("Solving geodesics...")
    traj_fwd = trained_model.solver.solve(trained_model.metric, pt_root, pt_b1, n_steps=10)
    traj_bwd = trained_model.solver.solve(trained_model.metric, pt_b1, pt_root, n_steps=10)
    
    # Tricky geodesic between branches
    traj_tricky = trained_model.solver.solve(trained_model.metric, pt_b1, pt_b2, n_steps=20)
    
    plt.plot(traj_fwd.xs[:, 0], traj_fwd.xs[:, 1], 'r-', linewidth=2, label='Fwd (Root->B1)')
    plt.plot(traj_bwd.xs[:, 0], traj_bwd.xs[:, 1], 'g--', linewidth=2, label='Bwd (B1->Root)')
    plt.plot(traj_tricky.xs[:, 0], traj_tricky.xs[:, 1], 'm-', linewidth=2, label='Tricky (B1->B2)')
    
    print(f"Energy Fwd (Downhill): {traj_fwd.energy:.4f}")
    print(f"Energy Bwd (Uphill):   {traj_bwd.energy:.4f}")
    print(f"Energy Tricky (B1->B2): {traj_tricky.energy:.4f}")
    print(f"Midpoint of tricky path: {traj_tricky.xs[15]}")
    
    os.makedirs("results_toy", exist_ok=True)
    plt.legend()
    plt.savefig("results_toy/ebm_landscape.png")
    print("Saved to results_toy/ebm_landscape.png")

if __name__ == "__main__":
    main()
