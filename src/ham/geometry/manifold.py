import jax
import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod
from typing import Tuple

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