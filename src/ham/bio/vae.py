import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Any

from ham.geometry.metric import FinslerMetric
from ham.geometry.surfaces import Hyperboloid

class WrappedNormal(eqx.Module):
    """
    Wrapped Normal distribution on a Manifold.
    Defined by a mean mu and a diagonal scale sigma in the tangent space at the origin/pole.
    """
    mean: jnp.ndarray
    scale: jnp.ndarray
    manifold: Any

    def __init__(self, mean: jnp.ndarray, scale: jnp.ndarray, manifold: Any):
        self.mean = manifold.project(mean)
        self.scale = scale
        self.manifold = manifold

    def sample(self, key: jax.random.PRNGKey, shape: Tuple[int] = ()) -> jnp.ndarray:
        d = self.manifold.intrinsic_dim
        v_flat = jax.random.normal(key, shape + (d,)) * self.scale
        
        origin = jnp.zeros_like(self.mean)
        if isinstance(self.manifold, Hyperboloid):
            origin = origin.at[..., 0].set(1.0)
            v_origin = jnp.concatenate([jnp.zeros(shape + (1,)), v_flat], axis=-1)
        else:
            # Sphere or Euclidean flat space fallback
            radius = getattr(self.manifold, "radius", 1.0)
            origin = origin.at[..., -1].set(radius)
            v_origin = jnp.concatenate([v_flat, jnp.zeros(shape + (1,))], axis=-1)
            
        v_at_mean = self.manifold.parallel_transport(origin, self.mean, v_origin)
        z = self.manifold.exp_map(self.mean, v_at_mean)
        return z

    def kl_divergence_std_normal(self) -> jnp.ndarray:
        kl = -jnp.log(self.scale + 1e-6) + (self.scale**2) / 2.0 - 0.5
        return jnp.sum(kl, axis=-1)

class GeometricVAE(eqx.Module):
    """
    Geometric VAE with Zermelo Control Dynamics.
    """
    encoder_net: eqx.Module
    decoder_net: eqx.Module
    metric: FinslerMetric 
    manifold: Any
    
    data_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True) 

    def __init__(self, data_dim, latent_dim, metric, key):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.metric = metric
        self.manifold = metric.manifold
        
        k1, k2 = jax.random.split(key)
        
        d_amb = self.manifold.ambient_dim
        d_int = self.manifold.intrinsic_dim
        
        self.encoder_net = eqx.nn.MLP(data_dim, d_amb + d_int, 128, 3, activation=jax.nn.gelu, key=k1)
        self.decoder_net = eqx.nn.MLP(d_amb, data_dim, 128, 3, activation=jax.nn.gelu, key=k2)

    def _get_dist(self, x):
        out = self.encoder_net(x)
        d_amb = self.manifold.ambient_dim
        
        mu_raw = out[:d_amb]
        log_scale = out[d_amb:]
        
        mu = self.manifold.project(mu_raw)
        scale = jax.nn.softplus(log_scale) + 1e-4
        
        return WrappedNormal(mu, scale, self.manifold)

    def encode(self, x, key):
        dist = self._get_dist(x)
        return dist.sample(key)

    def decode(self, z):
        return self.decoder_net(z)

    def project_control(self, x, v_rna):
        """
        Projects RNA velocity (Control Action) into latent space.
        Uses JVP to map dx -> dz.
        """
        def mean_fn(x_in):
            dist = self._get_dist(x_in)
            return dist.mean

        z_mean, u_lat = jax.jvp(mean_fn, (x,), (v_rna,))
        
        # u_lat is in the ambient space. Ensure it's tangent to z_mean.
        u_lat = self.manifold.to_tangent(z_mean, u_lat)
        return z_mean, u_lat

