import jax
import jax.numpy as jnp
from typing import NamedTuple, List, Callable, Optional, Tuple
from functools import partial
from ..geometry.metric import FinslerMetric

class Trajectory(NamedTuple):
    xs: jnp.ndarray
    vs: jnp.ndarray
    energy: jnp.ndarray
    constraint_violation: jnp.ndarray

class SolverState(NamedTuple):
    path: jnp.ndarray          # Inner points (T-2, D)
    lambdas: jnp.ndarray       # Lagrange multipliers (T-2, C)
    stiffness: jnp.ndarray     # Penalty parameters (T-2, C)
    prev_path: jnp.ndarray     # For momentum
    step: int
    max_violation: float
    prev_energy: float         # Energy from previous iteration
    curr_energy: float         # Energy from current iteration

class AVBDSolver:
    """
    Augmented Vertex Block Descent (AVBD) Solver.
    
    Minimizes Action S[x] while adaptively hardening constraints.
    
    Ref: 'Augmented Vertex Block Descent' (SIGGRAPH 2025)
    """
    def __init__(self, step_size: float = 0.01, beta: float = 10.0, 
                 iterations: int = 100, tol: float = 1e-6,
                 momentum: float = 0.9, energy_tol: float = 1e-5):
        self.step_size = step_size
        self.beta = beta          # Constraint hardening rate
        self.iterations = iterations 
        self.tol = tol
        self.momentum = momentum  # Nesterov momentum (0 = none)
        self.energy_tol = energy_tol

    def solve(self, metric: FinslerMetric, 
              p_start: jnp.ndarray, p_end: jnp.ndarray, 
              n_steps: int = 20,
              constraints: Optional[List[Callable[[jnp.ndarray], float]]] = None) -> Trajectory:
        
        # 0. Constraint Setup
        if constraints is None:
            constraints = []

        num_constraints = len(constraints)
        
        # 1. Initialization (Linear + Noise)
        t = jnp.linspace(0, 1, n_steps + 1)[:, None]
        linear_path = (1 - t) * p_start + t * p_end
        
        # Initialize Dual Variables (Lambdas, Stiffness)
        n_inner = n_steps - 1
        init_lambdas = jnp.zeros((n_inner, num_constraints)) if num_constraints > 0 else jnp.empty((n_inner, 0))
        init_stiffness = jnp.ones((n_inner, num_constraints)) * 1.0 if num_constraints > 0 else jnp.empty((n_inner, 0))

        # Initial projection + noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=linear_path.shape) * 1e-5
        path_guess = jax.vmap(metric.manifold.project)(linear_path + noise)
        init_inner = path_guess[1:-1]

        # Initial energy (pure, no penalties)
        init_full = jnp.concatenate([p_start[None, :], init_inner, p_end[None, :]], axis=0)
        init_vel = init_full[1:] - init_full[:-1]
        init_energy = jnp.sum(jax.vmap(metric.energy)(init_full[:-1], init_vel))

        state = SolverState(
            path=init_inner,
            lambdas=init_lambdas,
            stiffness=init_stiffness,
            prev_path=init_inner,
            step=0,
            max_violation=1.0,
            prev_energy=jnp.inf,
            curr_energy=init_energy
        )

        # 2. Local Augmented Lagrangian (for block updates)
        def local_aug_lagrangian(full_path: jnp.ndarray, idx: int, lams: jnp.ndarray, k_vals: jnp.ndarray) -> float:
            n_full = full_path.shape[0]
            
            x_prev = full_path[idx]           # always valid
            x      = full_path[idx + 1]       # the point we're updating
            
            # Indices
            idx_next = idx + 2
            
            # Masks (static-friendly)
            has_next = idx_next < n_full
            
            # Incoming velocity always exists
            v_in = x - x_prev
            
            # Outgoing velocity: use mask
            x_next = jnp.where(has_next, full_path[idx_next], p_end)
            v_out  = x_next - x
            
            # Local energy: always include incoming; conditionally add outgoing
            J_local = metric.energy(x_prev, v_in)
            J_local = J_local + jnp.where(has_next, metric.energy(x, v_out), 0.0)
            
            # Penalty (unchanged)
            penalty = 0.0
            if num_constraints > 0:           # this if is ok: num_constraints is Python int, static
                val = jnp.stack([c(x) for c in constraints])
                penalty = jnp.sum(lams * val + 0.5 * k_vals * (val ** 2))
            
            return J_local + penalty

        # 3. Primal Block Update (per vertex)
        def block_primal_update(state: SolverState, order: jnp.ndarray) -> jnp.ndarray:
            def update_vertex(carry_path: jnp.ndarray, idx: int) -> Tuple[jnp.ndarray, None]:
                full_path = jnp.concatenate([p_start[None, :], carry_path, p_end[None, :]], axis=0)
                
                # Current point
                x = carry_path[idx]
                
                # Nesterov lookahead in ambient (or skip if momentum high causes issues)
                y_ambient = x + self.momentum * (x - state.prev_path[idx])
                y = metric.manifold.project(y_ambient)  # Project lookahead to manifold
                
                # Local grad at y (ambient grad of local energy + penalty)
                local_grad_ambient = jax.grad(lambda y: local_aug_lagrangian(
                    full_path.at[idx + 1].set(y), idx, state.lambdas[idx], state.stiffness[idx]
                ))(y)
                
                # Project to tangent space
                tang_grad = metric.manifold.to_tangent(y, local_grad_ambient)
                
                # Intrinsic step
                tang_delta = -self.step_size * tang_grad
                
                # Retract to new point on manifold
                new_x = metric.manifold.retract(y, tang_delta)
                
                return carry_path.at[idx].set(new_x), None

            new_path, _ = jax.lax.scan(update_vertex, state.path, order)
            return new_path

        # 4. Single AVBD Update Step
        @jax.jit
        def update_step(s: SolverState) -> SolverState:
            key = jax.random.PRNGKey(s.step)
            order = jax.random.permutation(key, jnp.arange(n_inner))
            
            # A. Primal: Block descent
            p_new = block_primal_update(s, order)
            
            # B. Dual Update
            if num_constraints > 0:
                def get_c(pt):
                    return jnp.stack([c(pt) for c in constraints])
                
                c_vals = jax.vmap(get_c)(p_new)
                k_new = s.stiffness + self.beta * jnp.abs(c_vals)
                l_new = s.lambdas + k_new * c_vals
                max_viol = jnp.max(jnp.abs(c_vals))
            else:
                l_new, k_new = s.lambdas, s.stiffness
                max_viol = 0.0
            
            # C. Compute current pure energy
            full_path = jnp.concatenate([p_start[None, :], p_new, p_end[None, :]], axis=0)
            velocities = full_path[1:] - full_path[:-1]
            curr_energy = jnp.sum(jax.vmap(metric.energy)(full_path[:-1], velocities))
            
            return SolverState(
                path=p_new,
                lambdas=l_new,
                stiffness=k_new,
                prev_path=s.path,
                step=s.step + 1,
                max_violation=max_viol,
                prev_energy=s.curr_energy,
                curr_energy=curr_energy
            )

        # 5. Run with Early Stopping
        def cond_fun(s: SolverState) -> bool:
            high_violation = s.max_violation > self.tol
            energy_changing = jnp.abs(s.curr_energy - s.prev_energy) > self.energy_tol
            not_max_iter = s.step < self.iterations
            return (high_violation | energy_changing) & not_max_iter

        final_state = jax.lax.while_loop(cond_fun, update_step, state)

        # 6. Assemble final trajectory
        full_path = jnp.concatenate([p_start[None, :], final_state.path, p_end[None, :]], axis=0)
        velocities = full_path[1:] - full_path[:-1]
        pure_energy = jnp.sum(jax.vmap(metric.energy)(full_path[:-1], velocities))

        return Trajectory(
            xs=full_path, 
            vs=velocities, 
            energy=pure_energy,
            constraint_violation=final_state.max_violation
        )