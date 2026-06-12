"""Evaluation metrics for biological trajectory inference.

Provides functions to compute Directionality Score (symmetry breaking)
and Lineage Alignment Score (accuracy of intermediate states).
"""

import jax
import jax.numpy as jnp

def compute_directionality_score(model, z_start, z_end, n_steps=10):
    """Ratio of backward to forward energy along the geodesic.
    
    A higher score (> 1) means the metric successfully breaks symmetry, 
    penalizing travel against the expected biological flow (target -> source) 
    compared to the natural flow (source -> target).
    
    Args:
        model: Model containing `.solver` and `.metric`.
        z_start: Source point, shape (D,).
        z_end: Target point, shape (D,).
        n_steps: Segments in the BVP solver.
        
    Returns:
        Scalar score F(B->A) / F(A->B).
    """
    # Solve forward
    traj_fwd = model.solver.solve(model.metric, z_start, z_end, n_steps=n_steps)
    # Solve backward
    traj_bwd = model.solver.solve(model.metric, z_end, z_start, n_steps=n_steps)
    
    return traj_bwd.energy / (traj_fwd.energy + 1e-8)

def compute_lineage_alignment(z_pred_midpoint, z_obs_midpoint):
    """Distance between predicted and observed intermediate states.
    
    For a triple (day2, day4, day6), we compute the BVP from day2 to day6 
    with 2 segments. The middle vertex is the predicted day4 state.
    
    Args:
        z_pred_midpoint: Predicted middle state from AVBD, shape (D,).
        z_obs_midpoint: Empirically observed middle state, shape (D,).
        
    Returns:
        MSE distance.
    """
    return jnp.mean((z_pred_midpoint - z_obs_midpoint)**2)
