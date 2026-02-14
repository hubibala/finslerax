import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Any

from ham.geometry.metric import FinslerMetric
from ham.geometry.surfaces import Hyperboloid

class WrappedHyperbolicNormal(eqx.Module):
    """
    Wrapped Normal distribution on the Hyperboloid.
    Defined by a mean mu (on H^n) and a diagonal scale sigma (in tangent space).
    """
    mean: jnp.ndarray  # Shape: (..., D+1)
    scale: jnp.ndarray # Shape: (..., D) - Diagonal covariance in tangent space
    manifold: Hyperboloid

    def __init__(self, mean: jnp.ndarray, scale: jnp.ndarray, manifold: Hyperboloid):
        self.mean = manifold.project(mean)
        self.scale = scale
        self.manifold = manifold

    def sample(self, key: jax.random.PRNGKey, shape: Tuple[int] = ()) -> jnp.ndarray:
        # 1. Sample v ~ N(0, scale) in the tangent space of the ORIGIN
        # Origin O = (1, 0, ..., 0)
        # Tangent at O is just (0, v1, v2, ...) ~ R^D
        d = self.manifold.intrinsic_dim
        v_flat = jax.random.normal(key, shape + (d,)) * self.scale
        
        # Embed v into ambient space at Origin: (0, v_flat)
        v_origin = jnp.concatenate([jnp.zeros(shape + (1,)), v_flat], axis=-1)
        
        # 2. Transport v from Origin to Mean
        # We use parallel transport along the geodesic from Origin to self.mean
        origin = jnp.zeros_like(self.mean)
        origin = origin.at[..., 0].set(1.0)
        
        v_at_mean = self.manifold.parallel_transport(origin, self.mean, v_origin)
        
        # 3. Apply Exponential Map at Mean
        z = self.manifold.exp_map(self.mean, v_at_mean)
        return z

    def kl_divergence_std_normal(self) -> jnp.ndarray:
        """
        Computes KL(q(z)||p(z)) where p(z) is standard Wrapped Normal at Origin.
        Approximation: KL is computed between the tangent space Gaussians.
        """
        # KL between N(0, scale) and N(0, 1)
        # sum(log(1/sigma) + (sigma^2 + 0^2)/(2*1) - 0.5)
        # = sum(-log(sigma) + sigma^2/2 - 0.5)
        
        kl = -jnp.log(self.scale + 1e-6) + (self.scale**2) / 2.0 - 0.5
        return jnp.sum(kl, axis=-1)

class GeometricVAE(eqx.Module):
    """
    Hyperbolic VAE with Zermelo Control Dynamics.
    """
    encoder_net: eqx.Module
    decoder_net: eqx.Module
    metric: FinslerMetric 
    manifold: Hyperboloid
    
    data_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True) 

    def __init__(self, data_dim, latent_dim, metric, key):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.metric = metric
        self.manifold = Hyperboloid(intrinsic_dim=latent_dim)
        
        k1, k2 = jax.random.split(key)
        
        # Encoder: Outputs (D+1) for mean + (D) for scale
        # We need ambient dim for the mean because we project it.
        self.encoder_net = eqx.nn.MLP(data_dim, (latent_dim + 1) + latent_dim, 128, 3, activation=jax.nn.gelu, key=k1)
        
        # Decoder: Takes (D+1) [hyperboloid point] -> Data
        self.decoder_net = eqx.nn.MLP(latent_dim + 1, data_dim, 128, 3, activation=jax.nn.gelu, key=k2)

    def _get_dist(self, x):
        out = self.encoder_net(x)
        
        # Split output
        mu_raw = out[:self.latent_dim + 1]
        log_scale = out[self.latent_dim + 1:]
        
        # 1. Project mean to Hyperboloid
        mu = self.manifold.project(mu_raw)
        
        # 2. Scale is strictly positive
        scale = jax.nn.softplus(log_scale) + 1e-4
        
        return WrappedHyperbolicNormal(mu, scale, self.manifold)

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

    def loss_fn(self, x, v_rna, key):
        # 1. VAE Pass
        dist = self._get_dist(x)
        z_sample = dist.sample(key)
        x_rec = self.decode(z_sample)
        
        recon_loss = jnp.mean((x - x_rec)**2)
        kl_loss = jnp.mean(dist.kl_divergence_std_normal())
        
        # 2. Control Dynamics (Zermelo Navigation)
        z_mean, u_lat = self.project_control(x, v_rna)
        
        if hasattr(self.metric, '_get_zermelo_data'):
            _, W, _ = self.metric._get_zermelo_data(z_mean)
        else:
            W = jnp.zeros_like(u_lat)

        # Resultant Trajectory: dot_z = u + W
        dot_z = u_lat + W
        
        # 3. Zermelo Alignment (Symmetry Breaker)
        # Minimize angle between Wind and Velocity
        # Note: We must use Minkowski inner product for alignment? 
        # Actually, for "direction", cosine similarity in the ambient embedding 
        # is usually sufficient and more stable for optimization.
        # But let's use the proper Minkowski norm for normalization.
        
        norm_w = self.manifold._minkowski_norm(W)
        norm_v = self.manifold._minkowski_norm(u_lat)
        
        # Avoid div by zero
        w_dir = W / jnp.maximum(norm_w, 1e-6)[..., None]
        v_dir = u_lat / jnp.maximum(norm_v, 1e-6)[..., None]
        
        # Alignment = -<W_dir, V_dir>_L
        align_loss = -self.manifold._minkowski_dot(w_dir, v_dir)

        # 4. Geodesic Spray Loss
        spray_vec = self.metric.spray(z_mean, dot_z)
        # Norm of the spray vector (acceleration)
        spray_norm = self.metric.inner_product(z_mean, dot_z, spray_vec, spray_vec)
        
        # Weights
        total_loss = (1.0 * recon_loss + 
                      1e-4 * kl_loss + 
                      0.1 * align_loss + 
                      1e-3 * spray_norm)
        
        return total_loss, (recon_loss, kl_loss, spray_norm, align_loss)