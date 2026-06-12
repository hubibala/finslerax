import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import anndata

from ham.geometry.manifolds import EuclideanSpace
from ham.nn.ebm import PseudotimePotential
from ham.models.learned import PseudotimeRanders
from ham.solvers.avbd import AVBDSolver
from ham.training.losses_ebm import MSELoss
from ham.training.pipeline import HAMPipeline, TrainingPhase

# ------------------------------------------------------------------------------
# 1. Dataset Wrapper
# ------------------------------------------------------------------------------
class DiffusionDataset:
    def __init__(self, X_diff, dpt):
        self.X = jnp.array(X_diff)
        self.dpt = jnp.array(dpt)
        # HAMPipeline expects X, V, and labels
        self.V = jnp.zeros_like(self.X)
        self.labels = self.dpt
        self.length = len(X_diff)
        self.lineage_pairs = None
        
    def __len__(self):
        return self.length
        
    def get_batch(self, key, batch_size):
        idx = jax.random.choice(key, self.length, shape=(batch_size,))
        return (self.X[idx], self.dpt[idx])

# ------------------------------------------------------------------------------
# 2. Main Script
# ------------------------------------------------------------------------------
def main():
    print("Loading Diffusion data...")
    adata = anndata.read_h5ad("data/weinreb_diffusion.h5ad")
    
    # Extract 10D diffusion maps and 1D DPT
    X_diff = adata.obsm['X_diffmap']
    dpt = adata.obs['dpt_pseudotime'].values
    
    dataset = DiffusionDataset(X_diff, dpt)
    print(f"Loaded {len(dataset)} cells.")
    
    # --------------------------------------------------------------------------
    # Phase 1: Train Pseudotime Potential
    # --------------------------------------------------------------------------
    print("\n[Phase 1] Training Pseudotime Potential...")
    key = jax.random.PRNGKey(42)
    model = PseudotimePotential(dim=10, hidden_dim=64, depth=3, key=key)
    
    pipeline = HAMPipeline(model)
    
    phase = TrainingPhase(
        name="Train_Potential",
        epochs=100,
        optimizer=optax.adam(1e-3),
        losses=[MSELoss(weight=1.0)],
        filter_spec=lambda m: jax.tree_util.tree_map(eqx.is_array, m)
    )
    
    trained_model = pipeline.fit(dataset, [phase], batch_size=256)
    
    # --------------------------------------------------------------------------
    # Phase 2: Compute Geodesic
    # --------------------------------------------------------------------------
    print("\n[Phase 2] Computing Geodesic...")
    manifold = EuclideanSpace(10)
    # Wind scale < 1.0 ensures ||W|| < 1 constraint is nicely scaled
    metric = PseudotimeRanders(manifold, trained_model, wind_scale=0.5)
    
    # We will pick a root cell and a mature cell
    # Root cell is day 2 (we can just use the exact iroot used for DPT)
    iroot = adata.uns.get('iroot', 0)
    
    # Find a mature cell (e.g. day 6 with max pseudotime)
    day6_mask = adata.obs['time_point'] == 6
    if not np.any(day6_mask):
        day6_mask = np.ones(len(adata), dtype=bool)
    mature_idx = np.where(day6_mask)[0][np.argmax(dpt[day6_mask])]
    
    z_start = dataset.X[iroot]
    z_end = dataset.X[mature_idx]
    
    solver = AVBDSolver(step_size=0.1, grad_clip=1.0, iterations=500, implicit_diff=True)
    
    traj = solver.solve(metric, z_start, z_end, n_steps=30)
    
    # Save the trajectory
    os.makedirs("results_weinreb_diffusion", exist_ok=True)
    np.save("results_weinreb_diffusion/trajectory_10d.npy", traj.xs)
    print("Saved 10D geodesic to results_weinreb_diffusion/trajectory_10d.npy")
    
if __name__ == "__main__":
    main()
