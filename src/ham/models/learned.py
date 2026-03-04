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