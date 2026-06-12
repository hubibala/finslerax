import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Any, Optional

from ham.geometry.metric import AsymmetricMetric, FinslerMetric
from ham.geometry import Hyperboloid

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

    def _prior_origin(self) -> jnp.ndarray:
        """Canonical origin of the prior (the base point of the standard wrapped normal).

        Mirrors the per-manifold convention used by :meth:`sample`:
        the zero vector for flat Euclidean space, the time-axis pole
        ``e_0`` for the hyperboloid, and ``radius * e_{-1}`` for the sphere.
        """
        origin = jnp.zeros_like(self.mean)
        if self.manifold.ambient_dim == self.manifold.intrinsic_dim:
            return origin
        elif isinstance(self.manifold, Hyperboloid):
            return origin.at[..., 0].set(1.0)
        else:
            radius = getattr(self.manifold, "radius", 1.0)
            return origin.at[..., -1].set(radius)

    def sample(self, key: jax.random.PRNGKey, shape: Tuple[int] = ()) -> jnp.ndarray:
        d = self.manifold.intrinsic_dim
        v_flat = jax.random.normal(key, shape + (d,)) * self.scale

        origin = self._prior_origin()
        if self.manifold.ambient_dim == self.manifold.intrinsic_dim:
            v_origin = v_flat
        elif isinstance(self.manifold, Hyperboloid):
            v_origin = jnp.concatenate([jnp.zeros(shape + (1,)), v_flat], axis=-1)
        else:
            # Sphere or other fallback
            v_origin = jnp.concatenate([v_flat, jnp.zeros(shape + (1,))], axis=-1)

        v_at_mean = self.manifold.parallel_transport(origin, self.mean, v_origin)
        z = self.manifold.exp_map(self.mean, v_at_mean)
        return z

    def kl_divergence_std_normal(self) -> jnp.ndarray:
        r"""KL divergence to the standard wrapped normal prior at the origin.

        For a wrapped normal :math:`q = \mathrm{exp}_\mu(\mathrm{PT}_{o\to\mu}(u))`
        with :math:`u \sim \mathcal N(0, \sigma^2)` in the tangent space at the
        origin, the divergence to the prior :math:`p` (origin mean, unit scale)
        decomposes into a *scale* term and a *location* term:

        .. math::
            \mathrm{KL}(q\,\|\,p)
            = \underbrace{\sum_i \big(\tfrac{1}{2}\sigma_i^2 - \log\sigma_i - \tfrac12\big)}_{\text{scale}}
            + \underbrace{\tfrac12\, d_M(o, \mu)^2}_{\text{location}} .

        The location term :math:`\tfrac12 d_M(o,\mu)^2` is the manifold
        generalisation of the Euclidean :math:`\tfrac12\|\mu\|^2`; it is
        computed as :math:`\tfrac12\|\log_o \mu\|^2` in the tangent metric at
        the origin (Minkowski norm on the hyperboloid).  Omitting it leaves the
        posterior *mean* entirely unregularised, which collapses the KL's role
        as a prior on latent location.

        Reference: Nagano et al., *A Wrapped Normal Distribution on Hyperbolic
        Space for Gradient-Based Learning*, ICML 2019.
        """
        scale_kl = jnp.sum(
            -jnp.log(self.scale + 1e-6) + (self.scale**2) / 2.0 - 0.5, axis=-1
        )

        origin = self._prior_origin()
        v0 = self.manifold.log_map(origin, self.mean)
        if hasattr(self.manifold, "_minkowski_dot"):
            # Geodesic distance² in the Minkowski tangent metric (hyperboloid).
            d_sq = self.manifold._minkowski_dot(v0, v0)
        else:
            d_sq = jnp.sum(v0**2, axis=-1)
        location_kl = 0.5 * jnp.maximum(d_sq, 0.0)

        return scale_kl + location_kl

class GeometricVAE(eqx.Module):
    """
    Geometric VAE with Zermelo Control Dynamics.
    """
    encoder_net: eqx.Module
    decoder_net: eqx.Module
    metric: FinslerMetric 
    manifold: Any
    classifier_head: Optional[eqx.nn.Linear] = None  # add this field
    
    data_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True) 
    solver: Any = eqx.field(default=None)

    def __init__(self, data_dim, latent_dim, metric, key, solver=None, encoder_net=None, decoder_net=None, classifier_head=None):
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.metric = metric
        self.manifold = metric.manifold
        self.solver = solver
        self.classifier_head = classifier_head
        
        k1, k2 = jax.random.split(key)
        
        d_amb = self.manifold.ambient_dim
        d_int = self.manifold.intrinsic_dim
        
        if encoder_net is not None:
            self.encoder_net = encoder_net
        else:
            self.encoder_net = eqx.nn.MLP(data_dim, d_amb + d_int, 128, 3, activation=jax.nn.gelu, key=k1)
            
        if decoder_net is not None:
            self.decoder_net = decoder_net
        else:
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

    def loss_fn(self, x, v_rna, key):
        """Monolithic loss function for backwards compatibility with training scripts."""
        dist = self._get_dist(x)
        z = dist.sample(key)
        x_rec = self.decode(z)
        
        recon_loss = jnp.mean((x - x_rec)**2)
        kl_loss = jnp.mean(dist.kl_divergence_std_normal())
        
        z_mean, u_lat = self.project_control(x, v_rna)
        
        if isinstance(self.metric, AsymmetricMetric):
            _, W, _ = self.metric.zermelo_data(z_mean)
        else:
            W = jnp.zeros_like(u_lat)
            
        norm_w = self.manifold._minkowski_norm(W)
        norm_v = self.manifold._minkowski_norm(u_lat)
        
        w_dir = W / jnp.maximum(norm_w, 1e-6)[..., None]
        v_dir = u_lat / jnp.maximum(norm_v, 1e-6)[..., None]
        align_loss = -self.manifold._minkowski_dot(w_dir, v_dir)
        
        dot_z = u_lat + W
        spray_vec = self.metric.spray(z_mean, dot_z)
        spray_loss = self.metric.inner_product(z_mean, dot_z, spray_vec, spray_vec)
        
        total_loss = recon_loss + 1e-4 * kl_loss + 0.1 * align_loss + 1.0 * spray_loss
        
        return total_loss, (recon_loss, kl_loss, spray_loss, align_loss)


