import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Any

class LossComponent(eqx.Module):
    """Base class for modular loss components."""
    weight: float
    name: str

    def __init__(self, weight: float = 1.0, name: str = "Loss"):
        self.weight = weight
        self.name = name

    def __call__(self, model: eqx.Module, batch: Tuple[Any, ...], key: jax.random.PRNGKey) -> jnp.ndarray:
        raise NotImplementedError("Loss component must implement __call__")

class ReconstructionLoss(LossComponent):
    """MSE reconstruction loss for the VAE."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "Recon")

    def __call__(self, model, batch, key):
        x = batch[0]
        dist = model._get_dist(x)
        z_sample = dist.sample(key)
        x_rec = model.decode(z_sample)
        return jnp.mean((x - x_rec)**2) * self.weight

class KLDivergenceLoss(LossComponent):
    """KL Divergence loss for the wrapped normal distribution."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "KL")

    def __call__(self, model, batch, key):
        x = batch[0]
        dist = model._get_dist(x)
        return jnp.mean(dist.kl_divergence_std_normal()) * self.weight

class ZermeloAlignmentLoss(LossComponent):
    """Aligns the latent velocity with the learned Wind field W."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "Z_Align")

    def __call__(self, model, batch, key):
        x, v_rna = batch[0], batch[1]
        z_mean, u_lat = model.project_control(x, v_rna)
        
        if hasattr(model.metric, '_get_zermelo_data'):
            _, W, _ = model.metric._get_zermelo_data(z_mean)
        else:
            W = jnp.zeros_like(u_lat)

        norm_w = model.manifold._minkowski_norm(W)
        norm_v = model.manifold._minkowski_norm(u_lat)
        
        w_dir = W / jnp.maximum(norm_w, 1e-6)[..., None]
        v_dir = u_lat / jnp.maximum(norm_v, 1e-6)[..., None]
        
        return -model.manifold._minkowski_dot(w_dir, v_dir) * self.weight

class GeodesicSprayLoss(LossComponent):
    """Penalizes acceleration (spray vector norm) to encourage geodesic trajectories."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "Spray")

    def __call__(self, model, batch, key):
        x, v_rna = batch[0], batch[1]
        z_mean, u_lat = model.project_control(x, v_rna)
        
        if hasattr(model.metric, '_get_zermelo_data'):
            _, W, _ = model.metric._get_zermelo_data(z_mean)
        else:
            W = jnp.zeros_like(u_lat)

        dot_z = u_lat + W
        spray_vec = model.metric.spray(z_mean, dot_z)
        spray_norm = model.metric.inner_product(z_mean, dot_z, spray_vec, spray_vec)
        
        return spray_norm * self.weight

class ContrastiveAlignmentLoss(LossComponent):
    """Aligns the Wind field with the log map between parent and child points in latent space."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "Cont_Align")

    def __call__(self, model, batch, key):
        parent_x, child_x = batch[0], batch[1]
        
        k1, k2 = jax.random.split(key)
        parent_z = model.encode(parent_x, k1)
        child_z = model.encode(child_x, k2)
        
        _, W_out, _ = model.metric._get_zermelo_data(parent_z)
        v_tan = model.manifold.log_map(parent_z, child_z)
        
        align_score = -model.manifold._minkowski_dot(W_out, v_tan)
        return align_score * self.weight

class MetricAnchorLoss(LossComponent):
    """Anchors the metric tensor H(x) to Identity to prevent degenerate solutions."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "H_Reg")

    def __call__(self, model, batch, key):
        parent_x = batch[0]
        parent_z = model.encode(parent_x, key)
        
        H_out, _, _ = model.metric._get_zermelo_data(parent_z)
        dim = H_out.shape[-1]
        I = jnp.eye(dim)
        
        return jnp.mean((H_out - I)**2) * self.weight

class MetricSmoothnessLoss(LossComponent):
    """Jacobian penalty on W to encourage smooth vector fields."""
    def __init__(self, weight: float = 0.1):
        super().__init__(weight, "Smoothness")

    def __call__(self, model, batch, key):
        parent_x = batch[0]
        parent_z = model.encode(parent_x, key)
        
        def get_w_single(pt):
            _, W_out, _ = model.metric._get_zermelo_data(pt)
            return W_out
        
        jac = jax.jacfwd(get_w_single)(parent_z)
        
        return jnp.mean(jac**2) * self.weight
