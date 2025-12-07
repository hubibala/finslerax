import jax
import jax.numpy as jnp
from typing import NamedTuple, List, Callable, Optional
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
    step: int
    max_violation: float

class AVBDSolver:
    """
    Augmented Vertex Block Descent (AVBD) Solver.
    
    Minimizes Action S[x] while adaptively hardening constraints.
    
    Ref: 'Augmented Vertex Block Descent' (SIGGRAPH 2025)
    """
    def __init__(self, step_size: float = 0.01, beta: float = 10.0, 
                 iterations: int = 100, tol: float = 1e-6):
        self.step_size = step_size
        self.beta = beta          # Constraint hardening rate (from C++ beta=1.0)
        self.iterations = iterations 
        self.tol = tol

    def solve(self, metric: FinslerMetric, 
              p_start: jnp.ndarray, p_end: jnp.ndarray, 
              n_steps: int = 20,
              constraints: Optional[List[Callable[[jnp.ndarray], float]]] = None) -> Trajectory:
        
        # 0. Constraint Setup with Safe Math
        if constraints is None:
            # Default to manifold projection using SAFE distance
            def projection_constraint(x):
                proj = metric.manifold.project(x)
                # Use squared distance to strictly avoid sqrt(0) singularity
                # C(x) = 0.5 * ||x - P(x)||^2
                return 0.5 * jnp.sum((x - proj)**2)
            
            constraints = [projection_constraint]

        num_constraints = len(constraints)
        
        # 1. Initialization (Linear + Noise)
        t = jnp.linspace(0, 1, n_steps + 1)[:, None]
        linear_path = (1 - t) * p_start + t * p_end
        
        # Initialize Dual Variables (Lambdas, Stiffness)
        n_inner = n_steps - 1
        init_lambdas = jnp.zeros((n_inner, max(1, num_constraints)))
        
        # Initialize stiffness conservatively (start soft, harden later)
        init_stiffness = jnp.ones((n_inner, max(1, num_constraints))) * 1.0

        # Initial projection to start near valid manifold
        # Using a small noise to break symmetry for Zermelo problems
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=linear_path.shape) * 1e-5
        path_guess = jax.vmap(metric.manifold.project)(linear_path + noise)
        init_inner = path_guess[1:-1]

        state = SolverState(
            path=init_inner,
            lambdas=init_lambdas,
            stiffness=init_stiffness,
            step=0,
            max_violation=1.0
        )

        # 2. The Augmented Lagrangian (Primal Objective)
        def augmented_lagrangian(inner_path, lams, k_vals):
            full_path = jnp.concatenate([p_start[None, :], inner_path, p_end[None, :]], axis=0)
            xs = full_path[:-1]
            vs = full_path[1:] - full_path[:-1]
            
            # Action (Physics)
            energies = jax.vmap(metric.energy)(xs, vs)
            J = jnp.sum(energies)
            
            # Constraints (Dual)
            if num_constraints > 0:
                def point_cost(x, lam, k):
                    cost = 0.0
                    for i, c_fn in enumerate(constraints):
                        val = c_fn(x)
                        # ALM: lambda*C + 0.5*k*C^2
                        cost += lam[i] * val + 0.5 * k[i] * (val**2)
                    return cost
                
                penalty = jnp.sum(jax.vmap(point_cost)(inner_path, lams, k_vals))
                return J + penalty
            return J

        grad_fn = jax.value_and_grad(augmented_lagrangian, argnums=0)

        # 3. The AVBD Update Step (Interleaved Primal/Dual)
        @jax.jit
        def update_step(i, s: SolverState):
            # --- A. Primal Update (Physics) ---
            # Approximating the Newton step with Gradient Descent
            # (Newton is expensive for general neural metrics, GD is robust)
            loss, grads = grad_fn(s.path, s.lambdas, s.stiffness)
            p_new = s.path - self.step_size * grads
            
            # --- B. Dual Update (Constraint Hardening) ---

            if num_constraints > 0:
                def get_c(pt):
                    return jnp.stack([c(pt) for c in constraints])
                
                # Evaluate constraints at NEW positions
                c_vals = jax.vmap(get_c)(p_new)
                
                # 1. Update Stiffness: k <- k + beta * |C|
                # This corresponds to "UpdateLinkState" in C++
                k_new = s.stiffness + self.beta * jnp.abs(c_vals)
                
                # 2. Update Lambdas: lambda <- lambda + k_new * C
                # Using the UPDATED stiffness is a key trick from the paper
                l_new = s.lambdas + k_new * c_vals
                
                max_viol = jnp.max(jnp.abs(c_vals))
            else:
                l_new, k_new = s.lambdas, s.stiffness
                max_viol = 0.0
                
            return SolverState(p_new, l_new, k_new, i, max_viol)

        # 4. Run Optimization
        # We loop 'iterations' times. Each iter does 1 physics step + 1 constraint hardening.
        final_state = jax.lax.fori_loop(0, self.iterations, update_step, state)

        # 5. Assemble Trajectory
        full_path = jnp.concatenate([p_start[None, :], final_state.path, p_end[None, :]], axis=0)
        velocities = full_path[1:] - full_path[:-1]
        
        # Compute final pure energy (without penalties) for reporting
        pure_energy = jnp.sum(jax.vmap(metric.energy)(full_path[:-1], velocities))

        return Trajectory(
            xs=full_path, 
            vs=velocities, 
            energy=pure_energy,
            constraint_violation=final_state.max_violation
        )