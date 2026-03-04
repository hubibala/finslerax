import jax
import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod
from typing import Tuple
from ..utils.math import safe_norm

@jax.custom_jvp
def _safe_norm_ratio_jvp(x, y):
    """
    Computes ||x|| / ||y|| safely.
    If ||y|| is 0, outputs 1.0 (since x conceptually approaches y).
    """
    nx = jnp.linalg.norm(x, axis=-1, keepdims=True)
    ny = jnp.linalg.norm(y, axis=-1, keepdims=True)
    return jnp.where(ny < 1e-12, 1.0, nx / jnp.maximum(ny, 1e-12))

@_safe_norm_ratio_jvp.defjvp
def _safe_norm_ratio_jvp_def(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    
    nx = jnp.linalg.norm(x, axis=-1, keepdims=True)
    ny = jnp.linalg.norm(y, axis=-1, keepdims=True)
    
    is_zero = ny < 1e-12
    nx_safe = jnp.where(is_zero, 1.0, nx)
    ny_safe = jnp.where(is_zero, 1.0, ny)
    
    # Primal out
    primal_out = jnp.where(is_zero, 1.0, nx / ny_safe)
    
    # JVP
    # d( ||x|| / ||y|| ) = ( ||y|| * d(||x||) - ||x|| * d(||y||) ) / ||y||^2
    # d(||x||) = x . x_dot / ||x||
    dnx = jnp.where(nx < 1e-12, 0.0, jnp.sum(x * x_dot, axis=-1, keepdims=True) / nx_safe)
    dny = jnp.where(is_zero, 0.0, jnp.sum(y * y_dot, axis=-1, keepdims=True) / ny_safe)
    
    # Quot rule
    tangent_out = jnp.where(is_zero, 0.0, (ny_safe * dnx - nx * dny) / (ny_safe**2))
    
    return primal_out, tangent_out

class Manifold(eqx.Module, ABC):
    """
    Abstract base class for the topological domain.
    
    This class defines the domain M and its constraints (e.g., sticking to a sphere).
    Crucially, it does *not* define distance or geodesics; those are the 
    responsibility of the `FinslerMetric` class.
    
    Reference: ARCH_SPEC.md, Section 2.1
    """

    @property
    @abstractmethod
    def ambient_dim(self) -> int:
        """
        The dimension of the ambient embedding space (N).
        For a sphere S^2 embedded in R^3, this is 3.
        """
        pass
    
    @property
    @abstractmethod
    def intrinsic_dim(self) -> int:
        """
        The intrinsic dimension of the manifold (D).
        For a sphere S^2, this is 2.
        """
        pass

    @abstractmethod
    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Projects a point from ambient space back onto the manifold.
        
        Args:
            x: Point in ambient space R^N.
            
        Returns:
            x_proj: Point on the manifold M.
        """
        pass

    @abstractmethod
    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Projects an ambient vector v onto the tangent space T_x M.
        
        Args:
            x: Base point on the manifold M.
            v: Vector in ambient space R^N.
            
        Returns:
            v_proj: Vector in T_x M.
        """
        pass

    @abstractmethod
    def retract(self, x: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
        """
        Retraction: maps a tangent vector delta ∈ T_x M back to a point on the manifold.
        
        Should satisfy:
        retract(x, 0) = x
        D(retract)(x,0) [·] = Id  (first-order approximation of exponential map)
        
        Common simple choices:
        - Project(x + delta)   (projected retraction — cheap but not always great)
        - Exponential map approx (Euler, Cayley, etc.)
        - Closed-form for symmetric spaces (sphere, Stiefel, etc.)
        """
        pass

    def log_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the tangent vector (velocity) v in T_x M pointing from x to y,
        such that retract(x, v) = y (or exp_map(x, v) = y).
        
        The default implementation provides a mathematically rigorous first-order 
        approximation by extracting the tangent component of the secant vector.
        """
        v = self.to_tangent(x, y - x)
        
        # Scaling correction: The straight-line ambient distance is a much better
        # approximation of intrinsic distance than the length of the projected secant.
        # This strictly prevents the solver from taking topological "shortcuts" through 
        # the interior of curved objects (like the hole of a Torus), where purely normal 
        # secants artificially project to zero length and trick the optimizer.
        
        # We use a custom JVP to compute ||y - x|| / ||v|| safely.
        # If y approaches x, v approaches y - x, and the ratio goes cleanly to 1.
        scale = _safe_norm_ratio_jvp(y - x, v)
        
        return v * scale
    
    @abstractmethod
    def random_sample(self, key: jax.Array, shape: Tuple[int, ...]) -> jnp.ndarray:
        """
        Returns random points on the manifold.
        
        Args:
            key: JAX PRNG key.
            shape: Batch shape (e.g., (B,)). The final dimension 
                   will be `ambient_dim` automatically.
            
        Returns:
            samples: Array of shape (*shape, ambient_dim).
        """
        pass