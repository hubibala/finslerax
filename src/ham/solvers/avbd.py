"""Augmented Vertex Block Descent (AVBD) boundary-value geodesic solver.

This module implements a differentiable BVP solver that finds energy-minimizing 
geodesics between two points. It operates by optimizing a discretized path of 
vertices using randomized Gauss-Seidel sweeps (Block Descent) on the manifold.

The solver supports arbitrary Finsler metrics and equality constraints via 
an Augmented Lagrangian Method (ALM).

Classes:
    Trajectory: Data container for solver results.
    SolverState: Internal state tracker for iterative optimization.
    AVBDSolver: The core BVP solver implementation.

See also: spec/ARCH_SPEC.md § 4.2.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple, List, Callable, Optional, Union
from functools import partial

from ham.geometry.metric import FinslerMetric
from ham.utils.math import safe_norm, GRAD_EPS

__all__ = ["Trajectory", "AVBDSolver"]

class Trajectory(NamedTuple):
    """Result of a geodesic solver.

    Attributes:
        xs: Discretized path points, shape (T+1, D) where T = n_steps.
        vs: Velocity vectors between consecutive points, shape (T, D).
        energy: Total discrete path energy (sum of segment energies), scalar.
        constraint_violation: Maximum constraint violation across inner vertices.
    """
    xs: jax.Array
    vs: jax.Array
    energy: jax.Array
    constraint_violation: jax.Array

class SolverState(NamedTuple):
    """Internal state for the iterative AVBD solver.

    Attributes:
        path: Inner path vertices (excluding boundaries), shape (T-1, D).
        lambdas: Lagrange multipliers for constraints, shape (T-1, C).
        stiffness: Penalty parameters for constraints, shape (T-1, C).
        step: Current iteration counter.
        max_violation: Maximum constraint violation in the current iteration.
        prev_energy: Total path energy from the previous iteration.
        curr_energy: Total path energy from the current iteration.
    """
    path: jax.Array
    lambdas: jax.Array
    stiffness: jax.Array
    step: int
    max_violation: jax.Array
    prev_energy: jax.Array
    curr_energy: jax.Array

class AVBDSolver(eqx.Module):
    """Augmented Vertex Block Descent (AVBD) Geodesic Solver.

    Finds the energy-minimizing path between two boundary points by directly 
    optimizing discretized path vertices. The solver is fully JAX-compatible 
    and differentiable with respect to metric parameters.

    Algorithm:
        Discretizes the path into T+1 vertices. Minimizes the discrete action
        L = sum(E(x_i, v_i)) + Penalty via Gauss-Seidel sweeps. Constraints 
        are enforced using the Augmented Lagrangian Method (ALM).

    Attributes:
        step_size: Learning rate for vertex gradient descent.
        beta: Growth factor for penalty stiffness (ALM).
        iterations: Maximum number of full path sweeps.
        grad_clip: Maximum norm for vertex gradients to ensure stability.
        energy_tol: Tolerance for early stopping (relative energy change).
    """
    step_size: float = 0.05
    beta: float = 1.2
    iterations: int = 50
    grad_clip: float = 10.0
    energy_tol: float = 1e-6
    tol: float = 1e-4  # For backward compatibility

    def solve(
        self, 
        metric: FinslerMetric, 
        p_start: jax.Array, 
        p_end: jax.Array, 
        n_steps: int = 10,
        constraints: Optional[List[Callable[[jax.Array], jax.Array]]] = None,
        train_mode: bool = True,
        key: Optional[jax.Array] = None
    ) -> Trajectory:
        """Finds the energy-minimizing geodesic between two points.

        Args:
            metric: FinslerMetric instance defining the geometry.
            p_start: Start point on the manifold, shape (D,).
            p_end: End point on the manifold, shape (D,).
            n_steps: Number of discrete path segments. Default: 10.
            constraints: Optional list of scalar functions c(x) = 0. 
                Each must map (D,) -> scalar.
            train_mode: If True, uses jax.lax.scan (differentiable). 
                If False, uses jax.lax.while_loop (supports early stopping).
            key: PRNG key for stochastic vertex permutation. If None, 
                a deterministic key is derived from p_start and p_end.

        Returns:
            A Trajectory containing the optimized path and statistics.
        """
        # 1. Initialization and RNG handling
        if key is None:
            # Deterministic but data-dependent fallback for vmap compatibility
            seed_val = jnp.sum(p_start + p_end).astype(jnp.int32)
            key = jax.random.fold_in(jax.random.PRNGKey(0), seed_val)
        
        k_init, k_sweep = jax.random.split(key)
        
        # Standardize constraints
        actual_constraints = constraints if constraints is not None else []
        num_constraints = len(actual_constraints)
        
        # Linear initialization in ambient space, then projected
        t = jnp.linspace(0, 1, n_steps + 1)[:, None]
        linear_path = (1 - t) * p_start + t * p_end
        
        # Perturb slightly to avoid zero-velocity singularities at initialization
        noise = jax.random.normal(k_init, shape=linear_path.shape) * 1e-4
        path_guess = jax.vmap(metric.manifold.project)(linear_path + noise)
        init_inner = path_guess[1:-1]

        # Initialize ALM variables
        n_inner = n_steps - 1
        init_lambdas = jnp.zeros((n_inner, num_constraints))
        init_stiffness = jnp.ones((n_inner, num_constraints))

        state = SolverState(
            path=init_inner,
            lambdas=init_lambdas,
            stiffness=init_stiffness,
            step=0,
            max_violation=jnp.array(0.0),
            prev_energy=jnp.inf,
            curr_energy=jnp.array(0.0)
        )

        def get_local_energy(x_prev, x, x_next):
            # Sum of energies of the two adjacent segments
            v_in = metric.manifold.log_map(x_prev, x)
            v_out = metric.manifold.log_map(x, x_next)
            return metric.energy(x_prev, v_in) + metric.energy(x, v_out)

        def update_vertex(full_path, v_idx, s: SolverState):
            """Optimizes vertex full_path[v_idx]."""
            x_prev = full_path[v_idx - 1]
            x = full_path[v_idx]
            x_next = full_path[v_idx + 1]

            def loss_fn(curr_x):
                E = get_local_energy(x_prev, curr_x, x_next)
                
                # Augmented Lagrangian Penalty
                penalty = 0.0
                if num_constraints > 0:
                    c_vals = jnp.stack([c(curr_x) for c in actual_constraints])
                    lam = s.lambdas[v_idx - 1]
                    mu = s.stiffness[v_idx - 1]
                    penalty = jnp.sum(lam * c_vals + 0.5 * mu * (c_vals**2))
                
                return E + penalty

            # Projected Riemannian Gradient Descent
            grad_euc = jax.grad(loss_fn)(x)
            grad_tan = metric.manifold.to_tangent(x, grad_euc)
            
            # Clip for stability
            gnorm = safe_norm(grad_tan, eps=GRAD_EPS)
            scale = jnp.minimum(1.0, self.grad_clip / (gnorm + GRAD_EPS))
            step = -self.step_size * grad_tan * scale
            
            return metric.manifold.retract(x, step)

        def step_fn(s: SolverState, _):
            # A. Vertex Sweep (Gauss-Seidel)
            full_path = jnp.concatenate([p_start[None, :], s.path, p_end[None, :]], axis=0)
            
            # Randomized order for stochastic block descent
            sweep_key = jax.random.fold_in(k_sweep, s.step)
            v_indices = jax.random.permutation(sweep_key, jnp.arange(1, n_steps))

            def sweep_body(curr_p, v_idx):
                new_v = update_vertex(curr_p, v_idx, s)
                return curr_p.at[v_idx].set(new_v), None

            new_full, _ = jax.lax.scan(sweep_body, full_path, v_indices)
            new_inner = new_full[1:-1]

            # B. Dual Updates (ALM)
            max_v = jnp.array(0.0)
            new_lambdas = s.lambdas
            new_stiffness = s.stiffness

            if num_constraints > 0:
                def get_c_all(x):
                    return jnp.stack([c(x) for c in actual_constraints])
                
                all_c = jax.vmap(get_c_all)(new_inner) # (n_inner, num_constraints)
                new_lambdas = s.lambdas + s.stiffness * all_c
                new_stiffness = jnp.minimum(s.stiffness * self.beta, 1e6)
                max_v = jnp.max(jnp.abs(all_c))

            # C. Energy Monitoring
            vels = jax.vmap(metric.manifold.log_map)(new_full[:-1], new_full[1:])
            total_E = jnp.sum(jax.vmap(metric.energy)(new_full[:-1], vels))

            return SolverState(
                path=new_inner,
                lambdas=new_lambdas,
                stiffness=new_stiffness,
                step=s.step + 1,
                max_violation=max_v,
                prev_energy=s.curr_energy,
                curr_energy=total_E
            ), None

        # Execution
        if train_mode:
            final_state, _ = jax.lax.scan(step_fn, state, None, length=self.iterations)
        else:
            def cond(s):
                iter_not_done = s.step < self.iterations
                energy_not_conv = jnp.abs(s.curr_energy - s.prev_energy) > self.energy_tol
                return iter_not_done & energy_not_conv
            final_state = jax.lax.while_loop(cond, lambda s: step_fn(s, None)[0], state)

        # Final Trajectory Reassembly
        full_xs = jnp.concatenate([p_start[None, :], final_state.path, p_end[None, :]], axis=0)
        full_vs = jax.vmap(metric.manifold.log_map)(full_xs[:-1], full_xs[1:])
        
        return Trajectory(
            xs=full_xs,
            vs=full_vs,
            energy=final_state.curr_energy,
            constraint_violation=final_state.max_violation
        )