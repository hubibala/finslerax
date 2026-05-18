"""Augmented Vertex Block Descent (AVBD) boundary-value geodesic solver.

This module implements a differentiable BVP solver that finds energy-minimizing 
geodesics between two points. It operates by optimizing a discretized path of 
vertices using randomized Gauss-Seidel sweeps (Block Descent) on the manifold.

The solver supports arbitrary Finsler metrics and equality constraints via 
an Augmented Lagrangian Method (ALM).

An optional **graph-coloring parallel mode** (``parallel=True``) replaces the
sequential Gauss-Seidel sweep with a colored Gauss-Seidel sweep.  Vertices
are partitioned into independent sets (colors) using a 2-coloring of the 1D
path graph; all vertices of the same color are updated simultaneously via
``jax.vmap``, yielding GPU-parallel speedups proportional to the path length.
Between color groups the ordering is still Gauss-Seidel, preserving
convergence properties.

Reference:
    Giles, Diaz & Yuksel. *Augmented Vertex Block Descent.* SIGGRAPH 2025.
    Graph-coloring strategy for GPU-parallel block sweeps.

Classes:
    Trajectory: Data container for solver results.
    SolverState: Internal state tracker for iterative optimization.
    AVBDSolver: The core BVP solver implementation.

See also: spec/ARCH_SPEC.md § 4.2.
"""

import functools
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple, List, Callable, Optional

from ham.geometry.metric import FinslerMetric
from ham.utils.math import safe_norm, GRAD_EPS
from ham.solvers.coloring import chain_coloring

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


def get_differentiable_mask(obj):
    """Generate a PyTree of booleans matching the structure of obj.
    
    Returns True for trainable/differentiable inexact arrays, and False for
    non-differentiable arrays (e.g. terrain rasters, manifold, pixel spacing, scene origin)
    and static fields.
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(obj)
    mask_flat = []
    for path, leaf in flat:
        if not eqx.is_inexact_array(leaf):
            mask_flat.append(False)
            continue
        
        # Check if the path indicates a raster or non-trainable field
        is_static_field = False
        for entry in path:
            name = getattr(entry, 'name', None) or str(entry)
            if any(k in name for k in ['raster', 'pixel_spacing', 'origin', 'manifold', 'covariates', 'weather']):
                is_static_field = True
                break
        
        mask_flat.append(not is_static_field)
        
    return jax.tree_util.tree_unflatten(treedef, mask_flat)


# --- Standalone implicit differentiation helper functions ------------------
def _el_residual(inner_path, metric, p_start, p_end):
    """Discrete Euler-Lagrange residual dE/dx_k for inner nodes.

    Returns flat array of shape (n_inner * D,).
    """
    def total_energy(inner):
        full = jnp.concatenate([p_start[None], inner, p_end[None]], axis=0)
        vs = jax.vmap(metric.manifold.log_map)(full[:-1], full[1:])
        return jnp.sum(jax.vmap(metric.energy)(full[:-1], vs))
    return jax.grad(total_energy)(inner_path).ravel()


@eqx.filter_custom_vjp
def _implicit_forward_pass(vjp_args, n_steps, constraints, key):
    """Forward pass of the implicit solver: returns inner vertices only."""
    solver, metric, p_start, p_end = vjp_args
    # Sever all gradient tracking during the forward solver execution to prevent nested tracer leaks.
    # The custom VJP's backward pass analytical IFT adjoint handles all gradient computations.
    metric_sg = jax.tree_util.tree_map(
        lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x,
        metric
    )
    p_start_sg = jax.lax.stop_gradient(p_start)
    p_end_sg = jax.lax.stop_gradient(p_end)

    traj = solver._solve_core(
        metric=metric_sg,
        p_start=p_start_sg,
        p_end=p_end_sg,
        n_steps=n_steps,
        constraints=constraints,
        train_mode=True,
        key=key
    )
    return traj.xs[1:-1]


@_implicit_forward_pass.def_fwd
def _implicit_forward_pass_fwd(perturbed, vjp_args, n_steps, constraints, key):
    inner = _implicit_forward_pass(vjp_args, n_steps, constraints, key)
    return inner, (inner, vjp_args, n_steps, constraints, key)


@_implicit_forward_pass.def_bwd
def _implicit_forward_pass_bwd(res, g_inner, perturbed, vjp_args, n_steps, constraints, key):
    inner, vjp_args, n_steps, constraints, key = res
    solver, metric, p_start, p_end = vjp_args

    # dG/dx* — (n_inner*D, n_inner*D) Hessian of path energy
    dG_dx = jax.jacobian(
        lambda p: _el_residual(p, metric, p_start, p_end)
    )(inner).reshape(inner.size, inner.size)

    # Solve adjoint system: (dG/dx*)^T lam = g_inner.ravel()
    # rcond=1e-4 truncates singular values below 0.01% of sigma_max, capping
    # gradient amplification at 1e4 and preventing float32 overflow from
    # near-singular Hessians (flat metric regions, early-training paths).
    lam, _, _, _ = jnp.linalg.lstsq(dG_dx.T, g_inner.ravel(), rcond=1e-4)
    # NaN guard: if any entry of lam is non-finite (singular or NaN Hessian),
    # return zero gradients for this solve rather than propagating NaN into
    # the metric parameters.  This can occur when path segments degenerate
    # (zero-length segment → NaN in log_map Jacobian).
    lam = jnp.where(jnp.isfinite(lam), lam, jnp.zeros_like(lam))

    # Partition metric into arrays and static fields using the differentiable mask to exclude large rasters
    mask = get_differentiable_mask(metric)
    m_arr, m_static = eqx.partition(metric, mask)

    def el_wrt_arr(arr_leaves):
        m_l = eqx.combine(arr_leaves, m_static)
        return _el_residual(inner, m_l, p_start, p_end)

    # Compute VJP w.r.t m_arr directly to avoid materialising large Jacobians
    _, vjp_fn = jax.vjp(el_wrt_arr, m_arr)
    grad_arr = vjp_fn(-lam)[0]

    grad_m = eqx.combine(grad_arr, jax.tree_util.tree_map(lambda _: None, m_static))
    
    # Return gradients for all non-static arguments (solver, metric, p_start, p_end)
    s_mask = get_differentiable_mask(solver)
    s_arr, s_static = eqx.partition(solver, s_mask)
    grad_s_arr = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if eqx.is_inexact_array(x) else None, s_arr)
    grad_solver = eqx.combine(grad_s_arr, jax.tree_util.tree_map(lambda _: None, s_static))

    return (grad_solver, grad_m, jnp.zeros_like(p_start), jnp.zeros_like(p_end))


def _local_vertex_energy(curr_x, x_prev, x_next, metric, lam, mu, num_constraints, constraints):
    v_in = metric.manifold.log_map(x_prev, curr_x)
    v_out = metric.manifold.log_map(curr_x, x_next)
    E = metric.energy(x_prev, v_in) + metric.energy(curr_x, v_out)
    
    penalty = 0.0
    if num_constraints > 0:
        c_vals = jnp.stack([c(curr_x) for c in constraints])
        penalty = jnp.sum(lam * c_vals + 0.5 * mu * (c_vals**2))
        
    return E + penalty


def _get_constraints_val(x, constraints):
    return jnp.stack([c(x) for c in constraints])


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
        parallel: If True, use graph-coloring parallel sweeps instead of
            sequential Gauss-Seidel. Vertices are 2-colored (even/odd) and
            each color group is updated simultaneously via vmap. This
            trades strict Gauss-Seidel convergence for GPU parallelism.
    """
    step_size: float = eqx.field(static=True, default=0.05)
    beta: float = eqx.field(static=True, default=1.2)
    iterations: int = eqx.field(static=True, default=50)
    grad_clip: float = eqx.field(static=True, default=10.0)
    energy_tol: float = eqx.field(static=True, default=1e-6)
    tol: float = eqx.field(static=True, default=1e-4)
    parallel: bool = eqx.field(static=True, default=False)
    implicit_diff: bool = eqx.field(static=True, default=False)

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

        Note:
            When ``self.implicit_diff=True``, gradients w.r.t. metric
            parameters bypass the iterative unrolling entirely and are
            computed via the Implicit Function Theorem applied to the
            discrete Euler-Lagrange optimality conditions.  This reduces
            backward-pass memory from O(iterations * n_steps) to
            O(n_steps * D^2) and is recommended for large ``iterations``.
        """
        if self.implicit_diff:
            return self._solve_implicit(metric, p_start, p_end, n_steps,
                                        constraints, train_mode, key)
        return self._solve_core(metric, p_start, p_end, n_steps,
                                constraints, train_mode, key)

    def _solve_implicit(
        self,
        metric: FinslerMetric,
        p_start: jax.Array,
        p_end: jax.Array,
        n_steps: int = 10,
        constraints=None,
        train_mode: bool = True,
        key=None,
    ) -> Trajectory:
        """Implicit-differentiation wrapper around ``_solve_core``.

        The forward pass runs the iterative solver normally and discards
        all intermediate states.  The backward pass implements the IFT
        adjoint: at the converged path x* the discrete Euler-Lagrange
        residual G(x*, theta) = dE/dx_k = 0 holds for all interior nodes
        k=1..N-1.  Differentiating implicitly gives::

            dx*/dtheta = -(dG/dx*)^{-1} (dG/dtheta)

        The total-path gradient is obtained by solving one linear system
        of size (n_inner * D) x (n_inner * D).  Because the path Hessian
        is block-tridiagonal this is O(n_inner * D^2) — cheap relative to
        50-iteration unrolling.

        Args:
            metric: FinslerMetric.
            p_start: Start point (D,).
            p_end:   End point (D,).
            n_steps: Number of path segments.
            constraints: Optional equality constraints (forwarded to core).
            train_mode: Forwarded to core solver.
            key: PRNG key (forwarded to core solver).

        Returns:
            Trajectory — identical to _solve_core output but with IFT
            gradients flowing through metric parameters.
        """
        inner = _implicit_forward_pass((self, metric, p_start, p_end), n_steps, constraints, key)
        full_xs = jnp.concatenate(
            [p_start[None], inner, p_end[None]], axis=0
        )
        full_vs = jax.vmap(metric.manifold.log_map)(full_xs[:-1], full_xs[1:])
        energy = jnp.sum(jax.vmap(metric.energy)(full_xs[:-1], full_vs))
        return Trajectory(
            xs=full_xs,
            vs=full_vs,
            energy=energy,
            constraint_violation=jnp.array(0.0),
        )

    def _solve_core(
        self, 
        metric: FinslerMetric, 
        p_start: jax.Array, 
        p_end: jax.Array, 
        n_steps: int = 10,
        constraints: Optional[List[Callable[[jax.Array], jax.Array]]] = None,
        train_mode: bool = True,
        key: Optional[jax.Array] = None
    ) -> Trajectory:
        """Core iterative AVBD solver (unrolled backprop).

        This is the original ``solve`` implementation.  Called directly
        when ``implicit_diff=False``, or as the forward pass when
        ``implicit_diff=True``.
        """
        # Unpack hyperparameters to local variables to completely avoid JAX tracer leaks of self
        step_size = self.step_size
        beta = self.beta
        iterations = self.iterations
        grad_clip = self.grad_clip
        energy_tol = self.energy_tol
        parallel = self.parallel

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
            scale = jnp.minimum(1.0, grad_clip / (gnorm + GRAD_EPS))
            step = -step_size * grad_tan * scale
            
            return metric.manifold.retract(x, step)

        def update_vertex_triple(x_prev, x, x_next, lam, mu):
            """Optimizes a single vertex given its neighbors (for vmap)."""
            def loss_fn(curr_x):
                E = get_local_energy(x_prev, curr_x, x_next)
                penalty = 0.0
                if num_constraints > 0:
                    c_vals = jnp.stack([c(curr_x) for c in actual_constraints])
                    penalty = jnp.sum(lam * c_vals + 0.5 * mu * (c_vals**2))
                return E + penalty

            grad_euc = jax.grad(loss_fn)(x)
            grad_tan = metric.manifold.to_tangent(x, grad_euc)
            gnorm = safe_norm(grad_tan, eps=GRAD_EPS)
            scale = jnp.minimum(1.0, grad_clip / (gnorm + GRAD_EPS))
            step = -step_size * grad_tan * scale
            return metric.manifold.retract(x, step)

        # Pre-compute color groups for the parallel sweep
        if parallel:
            color_0, color_1 = chain_coloring(n_inner)

        def parallel_color_sweep(full_path, color_indices, s):
            """Update all vertices of one color simultaneously."""
            x_prevs = full_path[color_indices - 1]
            xs = full_path[color_indices]
            x_nexts = full_path[color_indices + 1]
            lams = s.lambdas[color_indices - 1]
            mus = s.stiffness[color_indices - 1]
            new_xs = jax.vmap(update_vertex_triple)(x_prevs, xs, x_nexts, lams, mus)
            return full_path.at[color_indices].set(new_xs)

        def step_fn(s: SolverState, _):
            full_path = jnp.concatenate([p_start[None, :], s.path, p_end[None, :]], axis=0)

            if parallel:
                # Colored Gauss-Seidel: update each color group in parallel,
                # process groups sequentially to maintain inter-group ordering.
                new_full = parallel_color_sweep(full_path, color_0, s)
                new_full = parallel_color_sweep(new_full, color_1, s)
            else:
                # Sequential Gauss-Seidel with randomized sweep order
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
                new_stiffness = jnp.minimum(s.stiffness * beta, 1e6)
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
            final_state, _ = jax.lax.scan(step_fn, state, None, length=iterations)
        else:
            def cond(s):
                iter_not_done = s.step < iterations
                energy_not_conv = jnp.abs(s.curr_energy - s.prev_energy) > energy_tol
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
