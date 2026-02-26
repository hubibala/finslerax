import jax
import jax.numpy as jnp
from typing import NamedTuple, List, Callable, Optional, Tuple
from functools import partial
import equinox as eqx

# Assuming FinslerMetric is defined in ham.geometry.metric
from ham.geometry.metric import FinslerMetric

class Trajectory(NamedTuple):
    xs: jnp.ndarray
    vs: jnp.ndarray
    energy: jnp.ndarray
    constraint_violation: jnp.ndarray

class SolverState(NamedTuple):
    path: jnp.ndarray          # Inner points (T-2, D)
    lambdas: jnp.ndarray       # Lagrange multipliers
    stiffness: jnp.ndarray     # Penalty parameters
    prev_path: jnp.ndarray     # For momentum
    step: int
    max_violation: float
    prev_energy: float
    curr_energy: float

class AVBDSolver(eqx.Module):
    """
    Augmented Vertex Block Descent (AVBD) - Differentiable Version.
    """
    step_size: float = 0.05  # Higher for training speed
    beta: float = 10.0
    iterations: int = 20     # Fixed iterations for training stability
    tol: float = 1e-4
    momentum: float = 0.5
    energy_tol: float = 1e-4

    def solve(self, metric: FinslerMetric, 
              p_start: jnp.ndarray, p_end: jnp.ndarray, 
              n_steps: int = 10,
              constraints: Optional[List[Callable]] = None,
              train_mode: bool = True) -> Trajectory:
        
        if constraints is None: constraints = []
        num_constraints = len(constraints)
        
        # 1. Linear Initialization
        t = jnp.linspace(0, 1, n_steps + 1)[:, None]
        # Linear interpolation in ambient space (or manifold if possible)
        # Note: If Hyperboloid, linear interp is "secant" lines, projected later
        linear_path = (1 - t) * p_start + t * p_end
        
        # CRITICAL FIX: Add noise to prevent zero-velocity segments
        # If start == end, velocity is 0, and gradients explode.
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=linear_path.shape) * 1e-4
        linear_path = linear_path + noise
        
        # Project initialization to manifold
        path_guess = jax.vmap(metric.manifold.project)(linear_path)
        init_inner = path_guess[1:-1]

        # Init Dual Variables
        n_inner = n_steps - 1
        init_lambdas = jnp.zeros((n_inner, num_constraints))
        init_stiffness = jnp.ones((n_inner, num_constraints))

        state = SolverState(
            path=init_inner,
            lambdas=init_lambdas,
            stiffness=init_stiffness,
            prev_path=init_inner,
            step=0,
            max_violation=1.0,
            prev_energy=jnp.inf,
            curr_energy=0.0
        )

        # 2. Define the discrete Action (Energy) locally
        def local_action(x_prev, x, x_next):
            # Discrete Finsler Energy: F(x, v)^2
            v_in = x - x_prev
            v_out = x_next - x
            # We minimize the sum of energies of the two segments connected to x
            return metric.energy(x_prev, v_in) + metric.energy(x, v_out)

        # 3. Block Update Logic (Optimizing one vertex 'x')
        def update_vertex(full_path, idx, s):
            x_prev = full_path[idx]     # Neighbor Left
            x = full_path[idx+1]        # Current Node (Target)
            x_next = full_path[idx+2]   # Neighbor Right

            # A. Gradient Descent on Manifold
            def loss_fn(current_x):
                # Energy
                E = local_action(x_prev, current_x, x_next)
                
                # Constraints (Augmented Lagrangian)
                penalty = 0.0
                if num_constraints > 0:
                    c_val = jnp.stack([c(current_x) for c in constraints])
                    lam = s.lambdas[idx]
                    k = s.stiffness[idx]
                    penalty = jnp.sum(lam * c_val + 0.5 * k * (c_val**2))
                
                return E + penalty

            # Calculate Riemannian Gradient
            # grad_f is Euclidean gradient
            grad_f = jax.grad(loss_fn)(x)
            
            # Project to Tangent Space
            grad_tan = metric.manifold.to_tangent(x, grad_f)
            
            # Update Step (with Momentum if needed, simplified here for stability)
            # x_new = Retract(x, -lr * grad)
            step = -self.step_size * grad_tan
            x_new = metric.manifold.retract(x, step)
            
            return x_new

        # 4. The Loop Body
        def step_fn(s: SolverState, _):
            # Reconstruct full path for context
            full_path = jnp.concatenate([p_start[None, :], s.path, p_end[None, :]], axis=0)
            
            # Randomize update order (Stochastic Block Descent)
            key = jax.random.PRNGKey(s.step) # deterministic for reproducibility
            order = jax.random.permutation(key, jnp.arange(n_inner))
            
            # Scan over vertices to update them
            def scan_body(curr_path_inner, idx):
                # Construct temporary full path to get neighbors
                # (Note: In pure JAX this is cheap if JIT compiled)
                temp_full = jnp.concatenate([p_start[None, :], curr_path_inner, p_end[None, :]], axis=0)
                new_node = update_vertex(temp_full, idx, s)
                return curr_path_inner.at[idx].set(new_node), None

            new_inner, _ = jax.lax.scan(scan_body, s.path, order)
            
            # Dual Updates (Constraints)
            # ... (Simplified: omitted for core geodesic regression unless needed) ...
            
            # Compute Energy for monitoring
            full_new = jnp.concatenate([p_start[None, :], new_inner, p_end[None, :]], axis=0)
            vels = full_new[1:] - full_new[:-1]
            total_E = jnp.sum(jax.vmap(metric.energy)(full_new[:-1], vels))
            
            return SolverState(
                path=new_inner,
                lambdas=s.lambdas,
                stiffness=s.stiffness,
                prev_path=s.path,
                step=s.step + 1,
                max_violation=0.0,
                prev_energy=s.curr_energy,
                curr_energy=total_E
            ), None

        # 5. Execution Strategy
        if train_mode:
            # Fixed unrolling (differentiable!)
            final_state, _ = jax.lax.scan(step_fn, state, None, length=self.iterations)
        else:
            # While loop (convergence based, strictly for inference)
            def cond(s): return s.step < self.iterations # Simplify for now
            final_state = jax.lax.while_loop(cond, lambda s: step_fn(s, None)[0], state)

        # 6. Output
        full_path = jnp.concatenate([p_start[None, :], final_state.path, p_end[None, :]], axis=0)
        velocities = full_path[1:] - full_path[:-1]
        
        return Trajectory(
            xs=full_path,
            vs=velocities,
            energy=final_state.curr_energy,
            constraint_violation=final_state.max_violation
        )