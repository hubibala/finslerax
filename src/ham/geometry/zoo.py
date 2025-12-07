
import jax
import jax.numpy as jnp
from typing import Callable, Optional

from ham.geometry.metric import FinslerMetric
from ham.geometry.manifold import Manifold
from ham.geometry.mesh import TriangularMesh
from ham.utils.math import safe_norm

class Euclidean(FinslerMetric):
    """Standard Euclidean metric: F(x, v) = |v|."""
    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return safe_norm(v)

class Riemannian(FinslerMetric):
    """General Riemannian metric: F(x, v) = sqrt( v^T G(x) v )."""
    def __init__(self, manifold: Manifold, g_net: Callable[[jnp.ndarray], jnp.ndarray]):
        super().__init__(manifold)
        self.g_net = g_net

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        G_x = self.g_net(x)
        G_x = 0.5 * (G_x + G_x.T)
        quad = jnp.dot(v, jnp.dot(G_x, v))
        return jnp.sqrt(jnp.maximum(quad, 1e-12))

class Randers(FinslerMetric):
    """
    Rigorous Randers Metric (Zermelo Navigation).
    F(x, v) = ( sqrt( lambda |v|_h^2 + <W, v>_h^2 ) - <W, v>_h ) / lambda
    """
    def __init__(self, 
                 manifold: Manifold, 
                 h_net: Callable[[jnp.ndarray], jnp.ndarray],
                 w_net: Callable[[jnp.ndarray], jnp.ndarray],
                 epsilon: float = 1e-5):
        super().__init__(manifold)
        self.h_net = h_net
        self.w_net = w_net
        self.epsilon = epsilon

    def _get_zermelo_data(self, x: jnp.ndarray):
        H = self.h_net(x)
        H = 0.5 * (H + H.T) 
        
        W_raw = self.w_net(x)
        
        # Norm in metric H
        w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))
        w_norm = jnp.sqrt(w_norm_sq + 1e-8)
        
        # Enforce convexity: ||W|| < 1
        scale = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
        W = W_raw * scale
        
        # Lambda = 1 - ||W||^2
        w_final_sq = w_norm_sq * (scale**2)
        lam = 1.0 - w_final_sq
        
        return H, W, lam

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        H, W, lam = self._get_zermelo_data(x)
        
        v_sq_h = jnp.dot(v, jnp.dot(H, v))
        W_dot_v = jnp.dot(v, jnp.dot(H, W))
        
        discriminant = lam * v_sq_h + W_dot_v**2
        cost = (jnp.sqrt(jnp.maximum(discriminant, 1e-12)) - W_dot_v) / lam
        return cost

class DiscreteRanders(FinslerMetric):
    """
    Anisotropic Mesh Metric (Wind per face).
    F(x, v) = Zermelo(v, W_face)
    """
    def __init__(self, mesh: TriangularMesh, face_winds: jnp.ndarray, epsilon: float = 1e-5):
        super().__init__(mesh)
        self.face_winds = face_winds
        self.epsilon = epsilon
        if not isinstance(mesh, TriangularMesh): raise ValueError("Requires TriangularMesh.")

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        face_idx = self.manifold.get_face_index(x)
        W_raw = self.face_winds[face_idx]
        
        # Stabilize Wind
        w_norm = jnp.linalg.norm(W_raw)
        scale = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
        W = W_raw * scale
        lam = 1.0 - (w_norm * scale)**2
        
        v_sq = jnp.dot(v, v)
        W_dot_v = jnp.dot(W, v)
        
        discriminant = lam * v_sq + W_dot_v**2
        cost = (jnp.sqrt(jnp.maximum(discriminant, 1e-8)) - W_dot_v) / lam
        return cost