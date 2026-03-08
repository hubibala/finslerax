import jax
import jax.numpy as jnp
import equinox as eqx
from ..geometry.metric import FinslerMetric
from ..geometry.manifold import Manifold
from ..geometry.zoo import Randers
from ..nn.networks import VectorField, PSDMatrixField
from ..utils.math import safe_norm

class NeuralRanders(Randers, eqx.Module):
    """
    A Learnable Randers Metric parameterized by Neural Networks.
    Inherits from Randers to reuse the correct Zermelo cost logic.
    """
    dim: int = eqx.field(static=True)

    def __init__(self, manifold: Manifold, key: jax.Array, 
                 hidden_dim: int = 32, depth: int = 2,
                 use_fourier: bool = True):
        k1, k2 = jax.random.split(key)
        self.dim = manifold.ambient_dim
        
        h_net = PSDMatrixField(self.dim, hidden_dim, depth, k1)
        w_net = VectorField(self.dim, hidden_dim, depth, k2, 
                            use_fourier=use_fourier, fourier_scale=3.0)
                            
        super().__init__(manifold, h_net, w_net, epsilon=1e-5)

class PullbackRanders(Randers, eqx.Module):
    """
    A Randers Metric where H(z) is strictly defined by the Decoder's 
    geometry (Pullback Metric), and only the Wind W(z) is learned.
    """
    decoder: eqx.Module # Pass the frozen decoder here
    dim: int = eqx.field(static=True)

    def __init__(self, manifold, decoder, key, hidden_dim=32):
        self.dim = manifold.ambient_dim
        self.decoder = decoder
        
        # We NO LONGER need a PSDMatrixField! 
        # We only need the VectorField for the Wind.
        w_net = VectorField(self.dim, hidden_dim, 2, key)
        
        # Pass a dummy h_net to the parent class, we will override it in _get_zermelo_data
        super().__init__(manifold, h_net=lambda x: x, w_net=w_net, epsilon=1e-5)

    def _get_zermelo_data(self, z: jax.Array):
        # 1. The Exact Pullback Metric H(z) = J^T J
        J = jax.jacfwd(self.decoder)(z)
        H = jnp.dot(J.T, J)
        
        # Add a tiny diagonal for numerical stability
        H = H + 1e-4 * jnp.eye(self.dim)
        
        # 2. The Learned Wind W(z)
        W_raw = self.w_net(z)
        W_raw = self.manifold.to_tangent(z, W_raw)
        
        # 3. The Causality Squasher (Same as before)
        w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))
        w_norm = jnp.sqrt(jnp.maximum(w_norm_sq, 1e-8))
        max_speed = 1.0 - self.epsilon
        squash_factor = (max_speed * jnp.tanh(w_norm)) / (w_norm + 1e-8)
        W_safe = W_raw * squash_factor
        
        # 4. Conformal factor
        safe_w_norm_sq = jnp.dot(W_safe, jnp.dot(H, W_safe))
        lambda_factor = 1.0 - safe_w_norm_sq
        
        return H, W_safe, lambda_factor