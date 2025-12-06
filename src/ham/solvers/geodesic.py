import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from ham.geometry.metric import FinslerMetric

class GeodesicState(NamedTuple):
    x: jnp.ndarray
    v: jnp.ndarray
    t: float

class ExponentialMap:
    """
    Solves the Initial Value Problem (IVP) for Geodesics.
    Given (x0, v0), computes x(t).
    
    Equivalent to the Riemannian Exponential Map: Exp_x(v).
    """
    def __init__(self, step_size: float = 0.01, max_steps: int = 200):
        self.step_size = step_size
        self.max_steps = max_steps  # Increased default for better manifold adherence

    def _step_rk4(self, metric: FinslerMetric, state: GeodesicState, dt: float) -> GeodesicState:
        """Standard Runge-Kutta 4 integration step for the Spray ODE."""
        x, v = state.x, state.v
        
        def dynamics(y):
            curr_x, curr_v = y[:x.shape[0]], y[x.shape[0]:]
            dx = curr_v
            dv = metric.geod_acceleration(curr_x, curr_v) # -2G
            return jnp.concatenate([dx, dv])

        y0 = jnp.concatenate([x, v])
        
        k1 = dynamics(y0)
        k2 = dynamics(y0 + 0.5 * dt * k1)
        k3 = dynamics(y0 + 0.5 * dt * k2)
        k4 = dynamics(y0 + dt * k3)
        
        y_next = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        x_next = y_next[:x.shape[0]]
        v_next = y_next[x.shape[0]:]
        
        # Enforce manifold constraints
        x_proj = metric.manifold.project(x_next)
        v_proj = metric.manifold.to_tangent(x_proj, v_next)
        
        return GeodesicState(x_proj, v_proj, state.t + dt)

    def shoot(self, metric: FinslerMetric, x0: jnp.ndarray, v0: jnp.ndarray) -> jnp.ndarray:
        """Computes the endpoint Exp_x0(v0) assuming t=1."""
        n_steps = self.max_steps
        dt = 1.0 / n_steps
        
        init_state = GeodesicState(x0, v0, 0.0)
        
        def body_fn(i, s):
            return self._step_rk4(metric, s, dt)
            
        final_state = jax.lax.fori_loop(0, n_steps, body_fn, init_state)
        return final_state.x

    def trace(self, metric: FinslerMetric, x0: jnp.ndarray, v0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Returns the full trajectory (xs, vs)."""
        n_steps = self.max_steps
        dt = 1.0 / n_steps
        
        def step_fn(s, _):
            s_new = self._step_rk4(metric, s, dt)
            return s_new, (s_new.x, s_new.v)
            
        init_state = GeodesicState(x0, v0, 0.0)
        _, (xs, vs) = jax.lax.scan(step_fn, init_state, None, length=n_steps)
        
        full_xs = jnp.concatenate([x0[None, :], xs], axis=0)
        full_vs = jnp.concatenate([v0[None, :], vs], axis=0)
        return full_xs, full_vs