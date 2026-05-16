"""
Abstract Manifold base class — the topological domain for HAMTools.

This module defines :class:`Manifold`, the abstract base class that
specifies the domain M and its constraints (projection, tangent spaces,
retraction). The manifold does *not* define distance or geodesics;
those are the responsibility of :class:`~ham.geometry.metric.FinslerMetric`.

Subclasses must implement :meth:`project`, :meth:`to_tangent`,
:meth:`retract`, and :meth:`random_sample`. Optionally override
:meth:`exp_map` and :meth:`log_map` with closed-form expressions
for better accuracy (see :mod:`ham.geometry.manifolds`).

Architecture reference: spec/ARCH_SPEC.md § 2.1.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod
from ham.utils.math import safe_norm, GRAD_EPS

@jax.custom_jvp
def _safe_norm_ratio(x, y):
    """
    Computes ||x|| / ||y|| safely.
    If ||y|| is 0, outputs 1.0 (since x conceptually approaches y).
    
    A custom JVP is registered because the naive quotient ||x||/||y|| has an 
    undefined derivative when ||y|| = 0. The custom rule clamps the tangent 
    to zero in that regime, ensuring clean AD through `log_map` (see below). 
    The derivative follows the quotient rule: 
    d(||x||/||y||) = (||y|| * d||x|| - ||x|| * d||y||) / ||y||^2.
    
    Reference:
    See `spec/MATH_SPEC.md` § 6 for the general numerical stability strategy. 
    This JVP guard complements the epsilon-regularization of F.
    """
    nx = safe_norm(x, axis=-1, keepdims=True)
    ny = safe_norm(y, axis=-1, keepdims=True)
    
    # safe_norm bottoms out at sqrt(GRAD_EPS). We use a 1.5x multiplier to reliably detect 
    # zero-vectors while absorbing floating-point fuzziness around the mathematical floor.
    zero_threshold = jnp.sqrt(GRAD_EPS) * 1.5
    is_zero = ny < zero_threshold
    return jnp.where(is_zero, 1.0, nx / ny)

@_safe_norm_ratio.defjvp
def _safe_norm_ratio_jvp(primals, tangents):
    x, y = primals
    x_dot, y_dot = tangents
    
    nx = safe_norm(x, axis=-1, keepdims=True)
    ny = safe_norm(y, axis=-1, keepdims=True)
    
    # safe_norm bottoms out at sqrt(GRAD_EPS). We use a 1.5x multiplier to reliably detect 
    # zero-vectors while absorbing floating-point fuzziness around the mathematical floor.
    zero_threshold = jnp.sqrt(GRAD_EPS) * 1.5
    is_zero = ny < zero_threshold
    ny_safe = jnp.where(is_zero, 1.0, ny)
    
    # Primal out
    primal_out = jnp.where(is_zero, 1.0, nx / ny_safe)
    
    # JVP
    # d( ||x|| / ||y|| ) = ( ||y|| * d(||x||) - ||x|| * d(||y||) ) / ||y||^2
    # d(||x||) = x . x_dot / ||x||
    dnx = jnp.where(nx < zero_threshold, 0.0, jnp.sum(x * x_dot, axis=-1, keepdims=True) / nx)
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
    
    Inherits from `eqx.Module`, making all subclasses valid JAX PyTrees composable 
    with `jax.jit`, `jax.vmap`, and `jax.grad`.
    
    See `ham.geometry.manifolds` for concrete implementations (Sphere, Torus, 
    Hyperboloid, etc.) and `examples/demo_trajectories.py` for usage.
    
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
    def project(self, x: jax.Array) -> jax.Array:
        """
        Projects a point from ambient space back onto the manifold.
        
        Args:
            x: Point in ambient space R^N.
            
        Returns:
            x_proj: Point on the manifold M. Shape: same as input `x`.
        """
        pass

    @abstractmethod
    def to_tangent(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Projects an ambient vector v onto the tangent space T_x M.
        
        Args:
            x: Base point on the manifold M.
            v: Vector in ambient space R^N.
            
        Returns:
            v_proj: Vector in T_x M. Shape: same as input `v`.
        """
        pass

    @abstractmethod
    def retract(self, x: jax.Array, delta: jax.Array) -> jax.Array:
        """
        Retraction: maps a tangent vector delta ∈ T_x M back to a point on the manifold.
        
        Should satisfy:
        retract(x, 0) = x
        D(retract)(x,0) [·] = Id  (first-order approximation of exponential map)
        
        Common simple choices:
        - Project(x + delta)   (projected retraction — cheap but not always great)
        - Exponential map approx (Euler, Cayley, etc.)
        - Closed-form for symmetric spaces (sphere, Stiefel, etc.)

        Args:
            x: Base point on the manifold M. Shape: `(D,)` or `(N,)` in ambient coordinates.
            delta: Tangent vector in T_x M. Shape: same as `x`.

        Returns:
            Point on M. Shape: same as `x`.
        """
        pass

    def exp_map(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Exponential map Exp_x(v): follow the geodesic from x with velocity v.

        The default implementation delegates to :meth:`retract`, which is a
        first-order approximation. Subclasses (e.g., Sphere, Hyperboloid)
        should override with the closed-form exponential when available.

        Args:
            x: Base point on M. Shape: `(D,)` or `(N,)` in ambient coordinates.
            v: Tangent vector in T_x M. Shape: same as `x`.

        Returns:
            Point on M reached by following the geodesic. Shape: same as `x`.
        """
        return self.retract(x, v)

    def log_map(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """
        Computes the tangent vector (velocity) v in T_x M pointing from x to y,
        approximately satisfying `exp_map(x, v) ≈ y`. Accuracy depends on the 
        subclass implementation of `exp_map`/`retract`.
        
        The default implementation provides a first-order approximation by extracting 
        the tangent component of the secant vector.
        
        Note:
            The projected secant \\Pi_{T_xM}(y - x) can have shorter ambient length than 
            y - x on highly curved manifolds, causing the solver to exploit interior 
            'shortcuts'. The scaling factor ||y - x|| / ||\\Pi_{T_xM}(y - x)|| corrects 
            for this, preserving the ambient chord length as a proxy for intrinsic distance.
            
        Args:
            x: Source point on M. Shape: `(D,)` or `(N,)` in ambient coordinates.
            y: Target point on M. Shape: same as `x`.
            
        Returns:
            Tangent vector v ∈ T_x M such that exp_map(x, v) ≈ y. Shape: same as `x`.
        """
        v = self.to_tangent(x, y - x)
        
        # We use a custom JVP to compute ||y - x|| / ||v|| safely.
        # If y approaches x, v approaches y - x, and the ratio goes cleanly to 1.
        # We clamp the scale factor to prevent numerical instability for near-normal displacements.
        scale = jnp.clip(_safe_norm_ratio(y - x, v), 0.0, 1e4)
        
        return v * scale
    
    @abstractmethod
    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """
        Returns random points on the manifold.
        
        Args:
            key: JAX PRNG key.
            shape: Batch shape (e.g., (B,)). The final dimension 
                   will be `ambient_dim` automatically.
            
        Returns:
            samples: Array of shape (*shape, ambient_dim).
            
        Raises:
            Behavior on invalid `key` or negative `shape` entries is subclass-defined.
        """
        pass