"""Initial Value Problem solver for Finsler geodesics.

Implements the Finslerian exponential map by integrating the Spray ODE
ddot{x}^i + 2G^i(x, dot{x}) = 0 (see spec/MATH_SPEC.md § 2.1) via a standard
Runge-Kutta 4 scheme with manifold projection.

Classes:
    GeodesicState: Integration state (position, velocity).
    ExponentialMap: RK4-based IVP geodesic solver.

See also: spec/ARCH_SPEC.md § 4.4.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple
from ham.geometry.metric import FinslerMetric
from ham.utils.math import safe_norm, GRAD_EPS

__all__ = ["GeodesicState", "ExponentialMap"]

class GeodesicState(NamedTuple):
    """State of the geodesic ODE integrator.

    Attributes:
        x: Position on the manifold, shape (D,).
        v: Velocity (tangent vector) at x, shape (D,).
    """
    x: jax.Array
    v: jax.Array

class ExponentialMap(eqx.Module):
    """Solves the Initial Value Problem (IVP) for Geodesics via RK4.

    Solves the Spray ODE for geodesics:
        ddot{x}^i + 2G^i(x, dot{x}) = 0
    where G^i are the geodesic spray coefficients (see spec/MATH_SPEC.md § 2.1).
    Integration is performed via RK4 with manifold projection at each stage
    to counteract numerical drift and maintain 4th-order accuracy.

    Attributes:
        max_steps: Number of integration steps per trajectory.
        max_velocity: Magnitude limit for velocity vectors.
        max_accel: Magnitude limit for acceleration vectors.
    """
    max_steps: int = eqx.field(static=True)
    max_velocity: float = eqx.field(static=True)
    max_accel: float = eqx.field(static=True)

    def __init__(
        self, 
        step_size: float = 0.01,
        max_steps: int = 0, 
        max_velocity: float = 1e6, 
        max_accel: float = 1e6
    ):
        """
        Args:
            step_size: Nominal timestep for integration. Used to derive 
                max_steps if max_steps is 0. Default: 0.01.
            max_steps: Number of RK4 steps. If > 0, overrides step_size. 
                Default: 0.
            max_velocity: Magnitude limit for velocity vectors. Default: 1e6.
            max_accel: Magnitude limit for acceleration vectors. Default: 1e6.
        """
        if max_steps > 0:
            self.max_steps = max_steps
        else:
            # Derive steps from step_size assuming t_max=1.0
            self.max_steps = int(jnp.ceil(1.0 / step_size))
            
        self.max_velocity = max_velocity
        self.max_accel = max_accel

    def _step_rk4(self, metric: FinslerMetric, state: GeodesicState, dt: float) -> GeodesicState:
        """Standard Runge-Kutta 4 integration step for the Spray ODE.

        Numerical safeguards:
            - Velocity is clipped to max_velocity to prevent ODE blow-up.
            - Acceleration is clipped to max_accel.
            - Position is projected to the manifold at each RK4 stage to 
              maintain integration fidelity on curved submanifolds.

        Args:
            metric: The FinslerMetric defining the geometry.
            state: Current GeodesicState (position, velocity).
            dt: Integration timestep.

        Returns:
            Updated GeodesicState after one RK4 step.
        """
        x, v = state.x, state.v
        dim = x.shape[0]
        
        def dynamics(y):
            # Pack (x, v) into a single state vector y = [x; v] for standard ODE form
            curr_x, curr_v = y[:dim], y[dim:]
            
            curr_v_norm = safe_norm(curr_v, eps=GRAD_EPS)
            # Explicit maximum speed limit to prevent the ODE from exploding
            v_scale = jnp.minimum(1.0, self.max_velocity / (curr_v_norm + GRAD_EPS))
            safe_v = curr_v * v_scale
            
            dx = safe_v
            # acceleration: ddot{x}^i = -2G^i(x, v)
            dv = metric.geod_acceleration(curr_x, safe_v) 
            
            # Clip acceleration as well for stability
            dv_norm = safe_norm(dv, eps=GRAD_EPS)
            dv_scale = jnp.minimum(1.0, self.max_accel / (dv_norm + GRAD_EPS))
            safe_dv = dv * dv_scale
            
            return jnp.concatenate([dx, safe_dv])

        y0 = jnp.concatenate([x, v])
        
        k1 = dynamics(y0)
        k2 = dynamics(y0 + 0.5 * dt * k1)
        k3 = dynamics(y0 + 0.5 * dt * k2)
        k4 = dynamics(y0 + dt * k3)
        
        y_next = y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        x_next = y_next[:dim]
        v_next = y_next[dim:]
        
        # Enforce manifold constraints on final result
        x_proj = metric.manifold.project(x_next)
        v_proj = metric.manifold.to_tangent(x_proj, v_next)
        
        return GeodesicState(x_proj, v_proj)

    def shoot(
        self, 
        metric: FinslerMetric, 
        x0: jax.Array, 
        v0: jax.Array, 
        t_max: float = 1.0
    ) -> jax.Array:
        """Computes the Finslerian exponential map Exp_{x0}(t_max * v0).

        Use when only the endpoint is needed (memory-efficient via jax.lax.fori_loop).
        Mathematically equivalent to integrating the Spray ODE from x0 with 
        velocity v0 for t_max time units.

        Args:
            metric: The FinslerMetric defining the geometry.
            x0: Initial position on the manifold, shape (D,).
            v0: Initial tangent velocity at x0, shape (D,).
            t_max: Total integration time. Default: 1.0.

        Returns:
            Final position on the manifold, shape (D,).
        """
        n_steps = self.max_steps
        dt = t_max / n_steps
        init_state = GeodesicState(x0, v0)
        
        def body_fn(i, s):
            return self._step_rk4(metric, s, dt)
            
        final_state = jax.lax.fori_loop(0, n_steps, body_fn, init_state)
        return final_state.x

    def trace(
        self, 
        metric: FinslerMetric, 
        x0: jax.Array, 
        v0: jax.Array, 
        t_max: float = 1.0
    ) -> tuple[jax.Array, jax.Array]:
        """Returns the full geodesic trajectory (positions and velocities) up to t_max.

        Use when the full trajectory is needed for visualization or analysis.
        The returned arrays include the initial conditions at index 0.

        Args:
            metric: The FinslerMetric defining the geometry.
            x0: Initial position on the manifold, shape (D,).
            v0: Initial tangent velocity at x0, shape (D,).
            t_max: Total integration time. Default: 1.0.

        Returns:
            Tuple (xs, vs) where:
                xs: Positions along the geodesic, shape (max_steps + 1, D).
                vs: Velocities along the geodesic, shape (max_steps + 1, D).
        """
        n_steps = self.max_steps
        dt = t_max / n_steps
        
        def step_fn(s, _):
            s_new = self._step_rk4(metric, s, dt)
            return s_new, (s_new.x, s_new.v)
            
        init_state = GeodesicState(x0, v0)
        _, (xs, vs) = jax.lax.scan(step_fn, init_state, None, length=n_steps)
        
        full_xs = jnp.concatenate([x0[None, :], xs], axis=0)
        full_vs = jnp.concatenate([v0[None, :], vs], axis=0)
        return full_xs, full_vs