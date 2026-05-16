"""
Berwald parallel-transport integrator.

Implements the Berwald connection (spec/MATH_SPEC.md § 3) for transporting
tangent vectors along geodesics on Finsler manifolds. The connection
coefficients are derived by differentiating the geodesic spray twice
w.r.t. velocity.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod

from ham.geometry.metric import FinslerMetric

class Connection(eqx.Module):
    """
    Base class for geometric connections.
    Inherits from eqx.Module to ensure valid JAX PyTree behavior.
    """
    metric: FinslerMetric
        
    @abstractmethod
    def christoffel_symbols(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Compute the connection coefficients at (x, v).

        Args:
            x: Position on the manifold, shape (D,).
            v: Tangent vector at x, shape (D,).

        Returns:
            Connection coefficients Γ^i_{jk}, shape (D, D, D).
        """
        pass
        
    @abstractmethod
    def parallel_transport(self, path_x: jax.Array, path_v: jax.Array, vec_start: jax.Array) -> jax.Array:
        """
        Transport a vector along a discrete path.

        Args:
            path_x: Positions along the curve, shape (T, D).
            path_v: Velocities along the curve, shape (T, D).
            vec_start: Initial tangent vector to transport, shape (D,).

        Returns:
            Transported vectors at each point, shape (T, D).
        """
        pass

class BerwaldConnection(Connection):
    r"""
    Berwald Connection induced by a Finsler spray.
    
    The Berwald connection coefficients are defined as:
        ^B\Gamma^i_{jk} = \partial^2 G^i / \partial v^j \partial v^k
    
    Computationally, the coefficients are the Hessian of the `metric.spray` 
    function with respect to the velocity argument. See `spec/MATH_SPEC.md § 3.1`.
    """
    
    def christoffel_symbols(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Berwald connection coefficients via Hessian of the spray.

        Computes $^B\Gamma^i_{jk}(x,v) = \partial^2 G^i / \partial v^j \partial v^k$
        using two nested `jax.jacfwd` calls on `metric.spray`.

        Args:
            x: Position, shape (D,).
            v: Tangent vector, shape (D,).

        Returns:
            Coefficients tensor, shape (D, D, D).
            
        Note:
            This differentiates through the linear solver in `metric.spray`. It assumes 
            the Hessian of the energy is reasonably conditioned (regularised).
        """
        # G^i(x, v) is defined via self.metric.spray(x, v)
        # We need the Hessian w.r.t. the velocity argument (argnums=1)
        # Using jacfwd twice on arg 1 gives a tensor of shape (D, D, D)
        jacobian_v = jax.jacfwd(self.metric.spray, argnums=1)
        hessian_v = jax.jacfwd(jacobian_v, argnums=1)
        return hessian_v(x, v)

    def parallel_transport(self, path_x: jax.Array, path_v: jax.Array, vec_start: jax.Array) -> jax.Array:
        r"""
        Parallel transports a vector 'vec_start' along a trajectory (path_x, path_v).
        
        Equation: $$\frac{dX^i}{dt} + \,^B\Gamma^i_{jk}(\gamma, \dot\gamma)\,\dot\gamma^j X^k = 0$$

        Args:
            path_x: Discrete positions along the curve, shape (T, D).
            path_v: Velocities at each position, shape (T, D).
            vec_start: Initial tangent vector, shape (D,).

        Returns:
            Transported vectors aligned with path_x, shape (T, D).
            The returned array has the same leading dimension as `path_x`. Entry `i` 
            is the transported vector at `path_x[i]`, computed from integrating over 
            the segment ending at that point. In particular, `result[0] == vec_start`.
            
        Note:
            Assumes the curve is parameterised over [0, 1] with uniform spacing.
            Implementations may be vmapped externally for batched paths.
        """
        if path_x.shape[0] < 2:
            return jnp.broadcast_to(vec_start, path_x.shape)
            
        dt = 1.0 / (len(path_x) - 1)
        
        def transport_ode(carry_vec, inputs):
            x, x_next, v = inputs
            
            # (D, D, D)
            gamma = self.christoffel_symbols(x, v)
            
            # dX^i_dt = - Gamma^i_jk v^j X^k
            dx = -jnp.einsum('ijk,j,k->i', gamma, v, carry_vec)
            
            new_vec = carry_vec + dx * dt
            
            # Project onto tangent space at the NEXT point to prevent drift bias
            new_vec = self.metric.manifold.to_tangent(x_next, new_vec)
            return new_vec, new_vec

        # Run exactly T-1 steps
        _, transported_vecs = jax.lax.scan(
            transport_ode, 
            vec_start, 
            (path_x[:-1], path_x[1:], path_v[:-1])
        )
        
        result = jnp.concatenate([vec_start[None, :], transported_vecs], axis=0)
        return result

