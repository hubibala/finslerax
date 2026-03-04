import jax
import jax.numpy as jnp
from ham.geometry.metric import FinslerMetric

class Connection:
    """Base class for geometric connections."""
    def __init__(self, metric: FinslerMetric):
        self.metric = metric
        
    def christoffel_symbols(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
        
    def parallel_transport(self, path_x: jnp.ndarray, path_v: jnp.ndarray, vec_start: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

class BerwaldConnection(Connection):
    """
    Berwald Connection induced by a Finsler spray.
    
    The Berwald connection coefficients are defined as:
        Gamma^i_jk = \\partial^2 G^i / \\partial v^j \\partial v^k
    """
    
    def christoffel_symbols(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # G^i(x, v) is defined via self.metric.spray(x, v)
        # We need the Hessian w.r.t. the velocity argument (argnums=1)
        # Using jacfwd twice on arg 1 gives a tensor of shape (D, D, D)
        jacobian_v = jax.jacfwd(self.metric.spray, argnums=1)
        hessian_v = jax.jacfwd(jacobian_v, argnums=1)
        return hessian_v(x, v)

    def parallel_transport(self, path_x: jnp.ndarray, path_v: jnp.ndarray, vec_start: jnp.ndarray) -> jnp.ndarray:
        """
        Parallel transports a vector 'vec_start' along a trajectory (path_x, path_v).
        
        Equation: dX_dt + Gamma^i_jk(x, v) * v^j * X^k = 0
        """
        def transport_ode(carry_vec, inputs):
            x, v = inputs
            
            # (D, D, D)
            gamma = self.christoffel_symbols(x, v)
            
            # dX^i_dt = - Gamma^i_jk v^j X^k
            dx = -jnp.einsum('ijk,j,k->i', gamma, v, carry_vec)
            
            dt = 1.0 / len(path_x)
            new_vec = carry_vec + dx * dt
            
            new_vec = self.metric.manifold.to_tangent(x, new_vec)
            return new_vec, new_vec

        _, transported_vecs = jax.lax.scan(
            transport_ode, 
            vec_start, 
            (path_x, path_v)
        )
        
        result = jnp.concatenate([vec_start[None, :], transported_vecs[:-1]], axis=0)
        return result

# Wrapper for backward compatibility
def berwald_transport(metric: FinslerMetric, 
                      path_x: jnp.ndarray, 
                      path_v: jnp.ndarray, 
                      vec_start: jnp.ndarray) -> jnp.ndarray:
    return BerwaldConnection(metric).parallel_transport(path_x, path_v, vec_start)
