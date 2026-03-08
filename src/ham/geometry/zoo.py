import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable

from ham.geometry.metric import FinslerMetric
from ham.geometry.manifold import Manifold
from ham.geometry.mesh import TriangularMesh
from ham.utils.math import safe_norm

class Euclidean(FinslerMetric):
    """Standard Euclidean metric: F(x, v) = |v|."""
    # No extra fields needed, inherits 'manifold' from FinslerMetric
    
    def __repr__(self) -> str:
        return f"Euclidean(manifold={self.manifold})"

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return safe_norm(v)

class Riemannian(FinslerMetric):
    """General Riemannian metric: F(x, v) = sqrt( v^T G(x) v )."""
    g_net: Callable[[jnp.ndarray], jnp.ndarray]

    def __init__(self, manifold: Manifold, g_net: Callable[[jnp.ndarray], jnp.ndarray]):
        super().__init__(manifold)
        self.g_net = g_net

    def __repr__(self) -> str:
        return f"Riemannian(manifold={self.manifold})"

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        G_x = self.g_net(x)
        G_x = 0.5 * (G_x + G_x.T)
        quad = jnp.dot(v, jnp.dot(G_x, v))
        return jnp.sqrt(jnp.maximum(quad, 1e-12))

class Randers(FinslerMetric):
    """
    Rigorous Randers Metric (Zermelo Navigation).
    """
    h_net: Callable[[jnp.ndarray], jnp.ndarray]
    w_net: Callable[[jnp.ndarray], jnp.ndarray]
    epsilon: float = eqx.field(static=True)

    def __init__(self, 
                 manifold: Manifold, 
                 h_net: Callable[[jnp.ndarray], jnp.ndarray],
                 w_net: Callable[[jnp.ndarray], jnp.ndarray],
                 epsilon: float = 1e-5):
        super().__init__(manifold)
        self.h_net = h_net
        self.w_net = w_net
        self.epsilon = epsilon

    def __repr__(self) -> str:
        return f"Randers(manifold={self.manifold}, epsilon={self.epsilon})"

    def _get_zermelo_data(self, z: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Returns the Riemannian tensor H, the Wind vector W, and the Lambda scalar.
        Enforces the strict Zermelo causality constraint ||W||_H < 1.
        """
        # 1. Get Riemannian metric H(z) from the h_net (PSD matrix field)
        H = self.h_net(z)
        H = 0.5 * (H + H.T)  # Symmetrize
        
        # Ensure H is positive definite
        diag = jnp.diag(H)
        diag_safe = jnp.maximum(diag, 0.01)
        H = H.at[jnp.diag_indices_from(H)].set(diag_safe)
        H = H + 0.005 * jnp.eye(H.shape[-1])
        
        # 2. Get raw wind W(z) from the w_net (vector field)
        W_raw = self.w_net(z)
        # Project to tangent space to avoid exploding components
        W_raw = self.manifold.to_tangent(z, W_raw)
        
        # 3. THE CAUSALITY SQUASHER
        # Calculate the norm of the wind in the local Riemannian metric: sqrt(W^T * H * W)
        w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))
        w_norm = jnp.sqrt(jnp.maximum(w_norm_sq, 1e-8))
        
        # Squash the norm so it is strictly < 0.95 (leaving a 0.05 safety margin)
        max_speed = 1.0 - self.epsilon
        squash_factor = (max_speed * jnp.tanh(w_norm)) / (w_norm + 1e-8)
        
        # Scale the wind vector
        W_safe = W_raw * squash_factor
        
        # 4. Compute the conformal factor Lambda (standard Randers formula)
        # Lambda = 1 - ||W||_H^2 (Since we squashed it, this is guaranteed to be > 0)
        safe_w_norm_sq = jnp.dot(W_safe, jnp.dot(H, W_safe))
        lambda_factor = 1.0 - safe_w_norm_sq
        
        return H, W_safe, lambda_factor

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        v_mag = safe_norm(v, axis=-1)
        is_zero = v_mag < 1e-7
        v_safe = jnp.where(is_zero[..., None], v + 1e-7, v)
        
        H, W, lam = self._get_zermelo_data(x)
        
        Hv = jnp.matmul(H, v_safe)
        HW = jnp.matmul(H, W)
        
        v_sq_h = jnp.sum(v_safe * Hv, axis=-1)
        W_dot_v = jnp.sum(v_safe * HW, axis=-1)
        
        discriminant = lam * v_sq_h + W_dot_v**2
        cost = (jnp.sqrt(jnp.maximum(discriminant, 0.0) + 1e-9) - W_dot_v) / lam
        return jnp.where(is_zero, 0.0, cost)

    def norm(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return self.metric_fn(x, v)

class DiscreteRanders(FinslerMetric):
    """Anisotropic Mesh Metric (Wind per face)."""
    face_winds: jnp.ndarray
    epsilon: float = eqx.field(static=True)

    def __init__(self, mesh: TriangularMesh, face_winds: jnp.ndarray, epsilon: float = 1e-5):
        super().__init__(mesh)
        self.face_winds = face_winds
        self.epsilon = epsilon

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Use differentiable weights to allow gradients to flow between faces
        weights = self.manifold.get_face_weights(x)
        W_raw = jnp.dot(weights, self.face_winds)
        w_norm = jnp.linalg.norm(W_raw)
        scale = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
        W = W_raw * scale
        lam = 1.0 - (w_norm * scale)**2
        
        v_sq = jnp.dot(v, v)
        W_dot_v = jnp.dot(W, v)
        discriminant = lam * v_sq + W_dot_v**2
        cost = (jnp.sqrt(jnp.maximum(discriminant, 1e-8)) - W_dot_v) / lam
        return cost