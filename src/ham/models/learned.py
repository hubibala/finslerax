import jax
import jax.numpy as jnp
import equinox as eqx
from ..geometry.metric import FinslerMetric
from ..geometry.manifold import Manifold
from ..nn.networks import VectorField, PSDMatrixField
from ..utils.math import safe_norm

class NeuralRanders(FinslerMetric, eqx.Module):
    """
    A Learnable Randers Metric parameterized by Neural Networks.
    
    Structure:
        F(x, v) = Zermelo(v, W(x), H(x))
        
    Components:
        H(x): Riemannian Base Metric (The "Sea"). Learned by PSDMatrixField.
        W(x): Vector Field (The "Wind"). Learned by VectorField.
        
    Constraints:
        Enforces convexity (|W|_H < 1) via smooth tanh gating.
    """
    h_net: PSDMatrixField
    w_net: VectorField
    manifold: Manifold
    epsilon: float = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, manifold: Manifold, key: jax.Array, 
                 hidden_dim: int = 32, depth: int = 2,
                 use_fourier: bool = True):
        self.manifold = manifold
        self.epsilon = 1e-5
        k1, k2 = jax.random.split(key)
        self.dim = manifold.ambient_dim
        
        # Metric Tensor H(x)
        # Low-frequency bias (terrain usually changes slowly)
        self.h_net = PSDMatrixField(self.dim, hidden_dim, depth, k1)
        
        # Wind Field W(x)
        # High-frequency bias (turbulence/vortices) via Fourier Features
        # Scale=3.0 allows capturing wave numbers up to ~18, covering complex flows.
        self.w_net = VectorField(self.dim, hidden_dim, depth, k2, 
                                 use_fourier=use_fourier, fourier_scale=3.0)

    def _get_zermelo_data(self, x: jnp.ndarray):
        """
        Retrieves H and W at point x, ensuring mathematical validity.
        """
        # 1. Base Metric H(x)
        H_raw = self.h_net(x)                     # network output
        # Make symmetric + positive definite
        H = 0.5 * (H_raw + H_raw.T)
        diag = jnp.diag(H)
        diag_safe = jnp.maximum(diag, 0.01)
        H = H.at[jnp.diag_indices_from(H)].set(diag_safe)
        H = H + 0.005 * jnp.eye(H.shape[-1])      # isotropic safety
        
        # 2. Raw Wind W_raw(x)
        W_raw = self.w_net(x)
        
        # Ensure W is tangent to the manifold (project it)
        # This is critical so the wind doesn't blow "off" the surface
        W_raw = self.manifold.to_tangent(x, W_raw)
        
        # 3. Convexity Gating (The "Speed of Light" limit)
        # Calculate norm of W in metric H: |W|^2 = W^T H W
        w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))
        w_norm = jnp.sqrt(w_norm_sq + 1e-8)
        
        # Soft Gating: Scale W so |W| is always < 1.0 (minus epsilon)
        # tanh(x)/x is a smooth function approx 1 near 0, decaying for large x.
        scale = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
        
        W = W_raw * scale
        
        # 4. Compute Lambda = 1 - |W|^2
        # Used for the Finsler cost calculation
        lam = 1.0 - (w_norm * scale)**2
        
        return H, W, lam

    def norm(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        The Learned Cost Function (Zermelo-Randers), completely safeguarded against v=0.
        """
        # 1. STRICT SAFEGUARD: Ensure v is never mathematically zero
        # We use keepdims=False for is_zero to easily mask the scalar output later
        v_mag = safe_norm(v, axis=-1)
        is_zero = v_mag < 1e-7
        
        # Expand is_zero for the vector addition
        v_safe = jnp.where(is_zero[..., None], v + 1e-7, v)
        
        H, W, lam = self._get_zermelo_data(x)
        
        # 2. Standard Zermelo-Randers Formula using v_safe
        # (Using jnp.sum and matmul is safer for batched inputs than jnp.dot)
        Hv = jnp.matmul(H, v_safe)
        HW = jnp.matmul(H, W)
        
        v_sq_h = jnp.sum(v_safe * Hv, axis=-1)
        W_dot_v = jnp.sum(v_safe * HW, axis=-1)
        
        discriminant = lam * v_sq_h + W_dot_v**2
        
        # The +1e-9 is still a great safety net for floating point jitter
        sqrt_D = jnp.sqrt(jnp.maximum(discriminant, 0.0) + 1e-9)
        
        F_safe = (sqrt_D - W_dot_v) / lam
        
        # 3. THE MAGIC SHIELD: 
        # If the cell was a clone (v=0), force the output to exactly 0.0.
        # JAX tracks this branch and will cleanly backpropagate a gradient of 0.0!
        return jnp.where(is_zero, 0.0, F_safe)

    # If your solver specifically calls `metric_fn`, alias it so nothing breaks:
    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return self.norm(x, v)
        
    def energy(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * self.norm(x, v)**2