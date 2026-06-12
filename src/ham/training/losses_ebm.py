"""Loss components for training Energy-Based Models (EBMs).

Provides Contrastive Divergence (CD) loss using Stochastic Gradient 
Langevin Dynamics (SGLD) to sample from the energy landscape.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from .losses import LossComponent

def sgld_step(x, ebm_model, step_size, noise_scale, key):
    """A single step of Stochastic Gradient Langevin Dynamics.
    
    x_{t+1} = x_t + step_size * \nabla \log p(x_t) + noise_scale * N(0, I)
    Since p(x) \propto \exp(-E(x)), \nabla \log p(x) = -\nabla E(x).
    """
    grad_fn = jax.grad(ebm_model)
    score = -grad_fn(x)
    # Clip the score to prevent SGLD from diverging, especially in high dimensions 
    # where the structural confinement gradient can be massive.
    score_norm = jnp.linalg.norm(score)
    score_clipped = jnp.where(score_norm > 10.0, score * (10.0 / (score_norm + 1e-8)), score)
    
    noise = jax.random.normal(key, x.shape)
    x_next = x + step_size * score_clipped + noise_scale * noise
    return x_next

def sgld_sample_single(x_init, ebm_model, num_steps, step_size, noise_scale, key):
    """Runs SGLD for a specified number of steps on a single point."""
    def body_fn(i, state):
        x, k = state
        k, subk = jax.random.split(k)
        x_next = sgld_step(x, ebm_model, step_size, noise_scale, subk)
        return x_next, k
    
    x_final, _ = jax.lax.fori_loop(0, num_steps, body_fn, (x_init, key))
    return x_final

class ContrastiveDivergenceLoss(LossComponent):
    """Contrastive Divergence (CD) Loss for Energy-Based Models.
    
    Uses Stochastic Gradient Langevin Dynamics (SGLD) to sample from the model
    distribution and trains the EBM to minimize the energy of data samples
    while maximizing the energy of generated samples.
    """
    sgld_steps: int = eqx.field(static=True)
    sgld_step_size: float = eqx.field(static=True)
    sgld_noise_scale: float = eqx.field(static=True)
    l2_reg_weight: float = eqx.field(static=True)
    
    def __init__(self, weight: float = 1.0, name: str = "CD", 
                 sgld_steps: int = 50, sgld_step_size: float = 1e-2, 
                 sgld_noise_scale: float = 1e-2, l2_reg_weight: float = 0.001):
        super().__init__(weight, name)
        self.sgld_steps = int(sgld_steps)
        self.sgld_step_size = float(sgld_step_size)
        self.sgld_noise_scale = float(sgld_noise_scale)
        self.l2_reg_weight = float(l2_reg_weight)
        
    def __call__(self, model: eqx.Module, batch: tuple, key: jax.Array) -> jnp.ndarray:
        # In HAMPipeline, vmap is applied over the batch dimension,
        # so x_real is a single data point of shape (D,).
        x_real = batch[0]
        
        ebm_model = getattr(model, "ebm", model)
        
        k1, k2 = jax.random.split(key)
        
        # Initialize SGLD from data with slight noise
        x_init = x_real + 0.05 * jax.random.normal(k1, x_real.shape)
        
        # Sample x_fake via SGLD
        x_fake = sgld_sample_single(
            x_init, ebm_model, self.sgld_steps, 
            self.sgld_step_size, self.sgld_noise_scale, k2
        )
        
        # Stop gradient on the generated sample so we don't backprop through the sampling loop
        x_fake = jax.lax.stop_gradient(x_fake)
        
        # Evaluate energies
        e_real = ebm_model(x_real)
        e_fake = ebm_model(x_fake)
        
        # Contrastive Divergence objective: E(data) - E(model)
        loss = e_real - e_fake
        
        # Optional: L2 regularization on energies to prevent them from drifting to infinity
        if self.l2_reg_weight > 0.0:
            reg = e_real**2 + e_fake**2
            loss = loss + self.l2_reg_weight * reg
            
        return loss * self.weight

class DenoisingScoreMatchingLoss(LossComponent):
    """Denoising Score Matching (DSM) Loss for Energy-Based Models.
    
    Perturbs data with Gaussian noise and trains the energy gradient (score)
    to point back to the clean data point. Much more stable in high dimensions than CD.
    """
    sigma: float = eqx.field(static=True)
    
    def __init__(self, weight: float = 1.0, name: str = "DSM", sigma: float = 0.1):
        super().__init__(weight, name)
        self.sigma = float(sigma)
        
    def __call__(self, model: eqx.Module, batch: tuple, key: jax.Array) -> jnp.ndarray:
        # In HAMPipeline, vmap is applied over the batch dimension,
        # so x_real is a single data point of shape (D,).
        x_real = batch[0]
        ebm_model = getattr(model, "ebm", model)
        
        # Perturb data
        noise = jax.random.normal(key, x_real.shape) * self.sigma
        x_pert = x_real + noise
        
        # Target score points back to x_real
        target_score = -noise / (self.sigma ** 2)
        
        # Predicted score is -\nabla E(x_pert)
        def energy_fn(x):
            return ebm_model(x)
            
        predicted_score = -jax.grad(energy_fn)(x_pert)
        
        # L2 distance between predicted and target scores
        loss = jnp.mean((predicted_score - target_score) ** 2)
        
        return loss * self.weight

class MSELoss(LossComponent):
    """Mean Squared Error Loss for Pseudotime Potential training.
    
    Expects batch[0] = x_diffmap, batch[1] = target_dpt.
    """
    def __init__(self, weight: float = 1.0, name: str = "MSE"):
        super().__init__(weight, name)
        
    def __call__(self, model: eqx.Module, batch: tuple, key: jax.Array) -> jnp.ndarray:
        x_diffmap = batch[0]
        # In HAMPipeline, batch is (X, V, labels). We stored DPT in labels.
        target_dpt = batch[2]
        
        pred_dpt = model(x_diffmap)
        
        loss = jnp.mean((pred_dpt - target_dpt) ** 2)
        return loss * self.weight
