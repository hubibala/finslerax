"""Non-parametric Energy Functions based on Kernel Density Estimation."""

import jax
import jax.numpy as jnp
import equinox as eqx

class GaussianKDEEnergy(eqx.Module):
    """Energy landscape defined by a Gaussian Kernel Density Estimate.
    
    Provides a smooth, analytical energy field based on fixed data centers.
    E(x) = -log( 1/N \sum_{i=1}^N exp( -||x - c_i||^2 / (2\sigma^2) ) )
    
    This acts as a non-parametric EBM to test geometric properties independent
    of neural network training stability.
    """
    centers: jax.Array  # Shape: (N, D)
    sigma: float
    
    def __init__(self, centers: jax.Array, sigma: float):
        """Initializes the KDE energy landscape.
        
        Args:
            centers: Fixed dataset to serve as KDE centers, shape (N, D).
            sigma: Bandwidth for the Gaussian kernels.
        """
        self.centers = jnp.asarray(centers)
        self.sigma = float(sigma)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the KDE energy at point x.
        
        Args:
            x: A single point in the state space, shape (D,).
            
        Returns:
            Scalar energy value E(x).
        """
        # Squared distances from x to all centers: ||x - c_i||^2
        # Shape: (N,)
        sq_dists = jnp.sum((self.centers - x) ** 2, axis=-1)
        
        # log-sum-exp trick for numerical stability
        # log(sum(exp(-sq_dists / (2 * sigma^2))))
        # We drop the 1/N and standardizing constants since AVBD solver only cares about \nabla E(x).
        # But for absolute scalar values, we might want to keep it well-behaved.
        
        log_density = jax.nn.logsumexp(-sq_dists / (2.0 * self.sigma ** 2))
        
        # E(x) = -log p(x)
        return -log_density
