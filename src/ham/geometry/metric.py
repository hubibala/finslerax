import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod

from ham.geometry.manifold import Manifold

class FinslerMetric(eqx.Module):
    """
    The abstract base class for all Finsler metrics.
    Inheriting from eqx.Module ensures all subclasses are valid JAX PyTrees.
    """
    manifold: Manifold

    @abstractmethod
    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        The fundamental Finsler cost function F(x, v).
        Must be 1-homogeneous in v.
        """
        pass

    def energy(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * self.metric_fn(x, v)**2

    def inner_product(self, x: jnp.ndarray, v: jnp.ndarray, 
                      w1: jnp.ndarray, w2: jnp.ndarray) -> jnp.ndarray:
        # Hessian of E w.r.t v, evaluated at (x, v)
        g_fn = jax.hessian(self.energy, argnums=1)
        g_x_v = g_fn(x, v)
        return jnp.dot(w1, jnp.dot(g_x_v, w2))

    def spray(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Geodesic Spray G(x, v).
        Solves: Hess_v(E) * (-2G) = Grad_x(E) - Jac_x(Grad_v(E)) * v
        """
        grad_v_fn = jax.grad(self.energy, argnums=1)
        
        # We need grad_x(E) and Jac_x(grad_v E) * v
        # Using JVP for the mixed term is efficient
        grad_x = jax.grad(self.energy, argnums=0)(x, v)
        
        def d_dv_fixed_v(pos):
            return grad_v_fn(pos, v)
            
        _, mixed_term = jax.jvp(d_dv_fixed_v, (x,), (v,))
        rhs = grad_x - mixed_term
        
        hess_v = jax.hessian(self.energy, argnums=1)(x, v)
        
        # Solve for acceleration
        # Regularize hessian slightly to avoid singular matrices
        acc = jnp.linalg.solve(hess_v + 1e-12 * jnp.eye(x.shape[0]), rhs)
        return -0.5 * acc

    def geod_acceleration(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return -2.0 * self.spray(x, v)
