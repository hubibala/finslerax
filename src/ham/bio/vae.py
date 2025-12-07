import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Any

from ham.geometry.metric import FinslerMetric
from ham.solvers.geodesic import ExponentialMap

class GeometricVAE(eqx.Module):
    """
    A VAE that JOINTLY learns the Encoder, Decoder, and the Finsler Metric.
    """
    encoder_net: eqx.Module
    decoder_net: eqx.Module
    
    # Learnable Metric
    metric: FinslerMetric 
    
    # Static Configuration
    solver: ExponentialMap = eqx.field(static=True)
    data_dim: int = eqx.field(static=True)    # <--- ADDED
    latent_dim: int = eqx.field(static=True)  # <--- ADDED

    def __init__(self, 
                 data_dim: int, 
                 latent_dim: int, 
                 metric: FinslerMetric, 
                 key: jax.random.PRNGKey):
        
        self.data_dim = data_dim      # <--- ASSIGNED
        self.latent_dim = latent_dim  # <--- ASSIGNED
        
        k1, k2 = jax.random.split(key)
        
        # Encoder: Data -> (z, v, logvar)
        self.encoder_net = eqx.nn.MLP(in_size=data_dim, out_size=latent_dim * 3, width_size=64, depth=3, key=k1)
        
        # Decoder: z -> Data
        self.decoder_net = eqx.nn.MLP(in_size=latent_dim, out_size=data_dim, width_size=64, depth=3, key=k2)
        
        self.metric = metric
        # Fast solver for training loop (few steps)
        self.solver = ExponentialMap(step_size=0.1, max_steps=10)

    def encode(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        out = self.encoder_net(x)
        size = out.shape[0] // 3
        
        mu = out[:size]
        v_pred = out[size:2*size]
        logvar = out[2*size:]
        
        # Reparameterization
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        z_sample = mu + eps * std
        
        # Manifold Projection
        z_proj = self.metric.manifold.project(z_sample)
        v_proj = self.metric.manifold.to_tangent(z_proj, v_pred)
        
        return z_proj, v_proj, logvar

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder_net(z)

    def predict_trajectory(self, x: jnp.ndarray, key: jax.random.PRNGKey, steps: int = 5):
        """
        Forecasting: What will this cell look like in the future?
        """
        z, v, _ = self.encode(x, key)
        
        # Create a temporary solver for the rollout duration
        dt = 1.0 / steps
        rollout_solver = ExponentialMap(step_size=dt, max_steps=steps)
        
        # Integrate Geodesic: z(t) using the LEARNED metric
        traj_z, _ = rollout_solver.trace(self.metric, z, v)
        
        # Decode trajectory: x(t)
        traj_x = jax.vmap(self.decode)(traj_z)
        
        return traj_x

    def loss_fn(self, x: jnp.ndarray, key: jax.random.PRNGKey):
        z, v, logvar = self.encode(x, key)
        x_rec = self.decode(z)
        
        # 1. Reconstruction Loss
        recon_loss = jnp.mean((x - x_rec)**2)
        
        # 2. KL Divergence
        kl_loss = -0.5 * jnp.sum(1 + logvar - jnp.square(z) - jnp.exp(logvar))
        
        # 3. Action Loss
        action_loss = self.metric.energy(z, v)
        
        # 4. Metric Regularization
        if hasattr(self.metric, 'h_net'):
            H_x = self.metric.h_net(z)
            I = jnp.eye(H_x.shape[0])
            metric_reg = jnp.mean((H_x - I)**2)
        else:
            metric_reg = 0.0

        total_loss = recon_loss + 1e-4 * kl_loss + 1e-3 * action_loss + 1.0 * metric_reg
        
        return total_loss, (recon_loss, action_loss, metric_reg)