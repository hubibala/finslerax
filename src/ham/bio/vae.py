import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Any

from ham.geometry.metric import FinslerMetric

def safe_norm(x, axis=-1, keepdims=False, eps=1e-12):
    n = jnp.linalg.norm(x, axis=axis, keepdims=keepdims)
    return jnp.maximum(n, eps)

def normalize(x, axis=-1, eps=1e-12):
    return x / safe_norm(x, axis=axis, keepdims=True, eps=eps)

class PowerSpherical(eqx.Module):
    """
    Power Spherical distribution with corrected broadcasting for JAX.
    """
    mean: jnp.ndarray  
    conc: jnp.ndarray  

    def __init__(self, mean: jnp.ndarray, conc: jnp.ndarray):
        self.mean = normalize(mean, axis=-1)
        self.conc = conc

    def sample(self, key: jax.random.PRNGKey, shape: Tuple[int] = ()) -> jnp.ndarray:
        d = self.mean.shape[-1]
        k1, k2 = jax.random.split(key)
        
        # 1. Sample 't' (marginal)
        alpha = (d - 1) / 2.0 + self.conc
        beta = (d - 1) / 2.0
        z = jax.random.beta(k1, alpha, beta, shape=shape)
        t = 2.0 * z - 1.0  
        
        # 2. Sample 'v' (conditional)
        v_raw = jax.random.normal(k2, shape + (d - 1,))
        v = normalize(v_raw, axis=-1)
        
        # 3. Construct 'y' in canonical frame
        factor = jnp.sqrt(jnp.maximum(1.0 - t**2, 1e-12))
        y = jnp.concatenate([t[..., None], factor[..., None] * v], axis=-1)
        
        # 4. Rotate 'y' to align with 'self.mean'
        e0 = jnp.zeros_like(self.mean)
        e0 = e0.at[0].set(1.0)
        u = e0 - self.mean
        u_norm_sq = jnp.sum(u**2)
        
        def reflect(y_in):
            dot = jnp.sum(u * y_in, axis=-1, keepdims=True)
            scale = 2.0 * (dot / (u_norm_sq + 1e-12))
            return y_in - scale * u
            
        sample = jax.lax.cond(
            u_norm_sq < 1e-12,
            lambda _: y,
            lambda _: reflect(y),
            operand=None
        )
        return sample

    def kl_divergence_uniform(self) -> jnp.ndarray:
        return self.conc

class GeometricVAE(eqx.Module):
    """
    Relativistic VAE with Zermelo Control Dynamics.
    """
    encoder_net: eqx.Module
    decoder_net: eqx.Module
    metric: FinslerMetric 
    
    data_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True) 

    def __init__(self, data_dim, latent_dim, metric, key):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.metric = metric
        
        k1, k2 = jax.random.split(key)
        self.encoder_net = eqx.nn.MLP(data_dim, latent_dim + 1, 128, 3, activation=jax.nn.gelu, key=k1)
        self.decoder_net = eqx.nn.MLP(latent_dim, data_dim, 128, 3, activation=jax.nn.gelu, key=k2)

    def _get_dist(self, x):
        out = self.encoder_net(x)
        mu_raw = out[:self.latent_dim]
        kappa_raw = out[self.latent_dim:]
        mu = normalize(mu_raw) 
        kappa = jax.nn.softplus(kappa_raw)[0] + 1.0
        return PowerSpherical(mu, kappa)

    def encode(self, x, key):
        dist = self._get_dist(x)
        return dist.sample(key)

    def decode(self, z):
        return self.decoder_net(z)

    def project_control(self, x, v_rna):
        """
        Projects RNA velocity (Control Action) into latent space.
        Returns u_lat (Control Vector).
        """
        def mean_fn(x_in):
            out = self.encoder_net(x_in)
            mu_raw = out[:self.latent_dim]
            return normalize(mu_raw)

        z_mean, u_lat = jax.jvp(mean_fn, (x,), (v_rna,))
        return z_mean, u_lat

    def loss_fn(self, x, v_rna, key):
        # 1. VAE Pass
        dist = self._get_dist(x)
        z_sample = dist.sample(key)
        x_rec = self.decode(z_sample)
        
        recon_loss = jnp.mean((x - x_rec)**2)
        kl_loss = dist.kl_divergence_uniform()
        
        # 2. Control Dynamics (Zermelo Navigation)
        z_mean, u_lat = self.project_control(x, v_rna)
        
        if hasattr(self.metric, '_get_zermelo_data'):
            _, W, _ = self.metric._get_zermelo_data(z_mean)
        else:
            W = jnp.zeros_like(u_lat)

        # Resultant Trajectory: dot_z = u + W
        dot_z = u_lat + W
        
        # 3. Zermelo Alignment (Symmetry Breaker)
        # We maximize the alignment between Wind W and Control u
        w_dir = normalize(W)
        v_dir = normalize(u_lat)
        # Loss is negative dot product (minimize to align)
        align_loss = -jnp.dot(w_dir, v_dir)

        # 4. Geodesic Spray Loss
        # Minimize the fictitious force on the resultant path
        spray_vec = self.metric.spray(z_mean, dot_z)
        spray_norm = self.metric.inner_product(z_mean, dot_z, spray_vec, spray_vec)
        
        # Weights
        total_loss = (1.0 * recon_loss + 
                      1e-4 * kl_loss + 
                      0.1 * align_loss + 
                      1e-3 * spray_norm)
        
        # EXPOSE ALL STATS
        return total_loss, (recon_loss, kl_loss, spray_norm, align_loss)