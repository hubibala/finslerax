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

from typing import Callable, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.metric import FinslerMetric
from ham.solvers.coloring import chain_coloring
from ham.utils.math import GRAD_EPS, PSD_EPS, safe_norm

__all__ = ["AVBDSolver", "Trajectory"]


class Trajectory(NamedTuple):
    """Result of a geodesic solver.

    Attributes:
        xs: Discretized path points, shape (T+1, D) where T = n_steps.
        vs: Velocity vectors between consecutive points, shape (T, D).
        energy: Total discrete path energy (sum of segment energies), scalar.
        constraint_violation: Maximum constraint violation across inner vertices.
        lambdas: Lagrange multipliers for constraints at convergence, shape (T-1, C).
        stiffness: Penalty parameters for constraints at convergence, shape (T-1, C).
    """

    xs: jax.Array
    vs: jax.Array
    energy: jax.Array
    constraint_violation: jax.Array
    lambdas: jax.Array
    stiffness: jax.Array


class SolverState(NamedTuple):
    """Internal state for the iterative AVBD solver.

    Attributes:
        path: Inner path vertices (excluding boundaries), shape (T-1, D).
        lambdas: Lagrange multipliers for constraints, shape (T-1, C).
        stiffness: Penalty parameters for constraints, shape (T-1, C).
        prev_c: Constraint values from the previous iteration, shape (T-1, C).
            Used by the ALM schedule to grow stiffness only on stagnation.
        step: Current iteration counter.
        max_violation: Maximum constraint violation in the current iteration.
        prev_energy: Total path energy from the previous iteration.
        curr_energy: Total path energy from the current iteration.
    """

    path: jax.Array
    lambdas: jax.Array
    stiffness: jax.Array
    prev_c: jax.Array
    step: int
    max_violation: jax.Array
    prev_energy: jax.Array
    curr_energy: jax.Array


# Fallback name-substring heuristic for modules that do not declare
# ``nondiff_fields`` explicitly. Prefer the explicit declaration: this list is
# fragile (matches substrings anywhere in the field path) and exists only for
# backward compatibility with metrics written before the protocol existed.
_DEFAULT_NONDIFF_KEYS = (
    "raster",
    "pixel_spacing",
    "origin",
    "manifold",
    "covariates",
    "weather",
)


def get_differentiable_mask(obj):
    """Generate a PyTree of booleans matching the structure of obj.

    Returns True for trainable/differentiable inexact arrays, and False for
    non-differentiable arrays (e.g. terrain rasters, manifold, pixel spacing,
    scene origin) and static fields.

    Modules can opt out of the name heuristic by declaring an explicit
    ``nondiff_fields`` attribute (an iterable of field-name substrings, e.g.
    ``nondiff_fields: ClassVar[tuple[str, ...]] = ("elevation_map",)``); when
    present it replaces :data:`_DEFAULT_NONDIFF_KEYS` entirely.
    """
    nondiff_keys = tuple(getattr(obj, "nondiff_fields", _DEFAULT_NONDIFF_KEYS))
    flat, treedef = jax.tree_util.tree_flatten_with_path(obj)
    mask_flat = []
    for path, leaf in flat:
        if not eqx.is_inexact_array(leaf):
            mask_flat.append(False)
            continue

        # Check if the path indicates a raster or non-trainable field
        is_static_field = False
        for entry in path:
            name = getattr(entry, "name", None) or str(entry)
            if any(k in name for k in nondiff_keys):
                is_static_field = True
                break

        mask_flat.append(not is_static_field)

    return jax.tree_util.tree_unflatten(treedef, mask_flat)


# --- Standalone implicit differentiation helper functions ------------------
def _el_residual(inner_path, metric, p_start, p_end, constraints, lambdas, stiffness):
    """Discrete Euler-Lagrange residual of the Lagrangian for inner nodes.

    Calculates the Euclidean gradient of the total path Lagrangian (energy + constraints)
    and projects it onto the manifold tangent space to satisfy $G(x^*) = 0$.

    Returns flat array of shape (n_inner * D,).
    """

    def total_lagrangian(inner):
        full = jnp.concatenate([p_start[None], inner, p_end[None]], axis=0)
        vs = jax.vmap(metric.manifold.log_map)(full[:-1], full[1:])
        energy = jnp.sum(jax.vmap(metric.energy)(full[:-1], vs))

        penalty = 0.0
        if constraints is not None and len(constraints) > 0:
            # Evaluate constraints across all inner vertices -> shape: (n_inner, C)
            c_vals = jnp.stack([jax.vmap(c)(inner) for c in constraints], axis=-1)
            penalty = jnp.sum(lambdas * c_vals + 0.5 * stiffness * (c_vals**2))

        return energy + penalty

    # Differentiate wrt inner vertices
    grad_euc = jax.grad(total_lagrangian)(inner_path)

    # Project the Euclidean residual onto the tangent space
    grad_tan = jax.vmap(metric.manifold.to_tangent)(inner_path, grad_euc)
    return grad_tan.ravel()


@eqx.filter_custom_vjp
def _implicit_forward_pass(vjp_args, n_steps, constraints, key):
    """Forward pass of the implicit solver."""
    solver, metric, p_start, p_end = vjp_args
    # Sever all gradient tracking during the forward solver execution to prevent nested tracer leaks.
    # The custom VJP's backward pass analytical IFT adjoint handles all gradient computations.
    metric_sg = jax.tree_util.tree_map(
        lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x, metric
    )
    p_start_sg = jax.lax.stop_gradient(p_start)
    p_end_sg = jax.lax.stop_gradient(p_end)

    traj = solver._solve_core(
        metric=metric_sg,
        p_start=p_start_sg,
        p_end=p_end_sg,
        n_steps=n_steps,
        constraints=constraints,
        train_mode=False,  # O(1) memory mapping, relies entirely on analytical adjoint
        key=key,
    )
    return traj.xs[1:-1], traj.lambdas, traj.stiffness


@_implicit_forward_pass.def_fwd
def _implicit_forward_pass_fwd(perturbed, vjp_args, n_steps, constraints, key):
    inner, lambdas, stiffness = _implicit_forward_pass(
        vjp_args, n_steps, constraints, key
    )
    return (inner, lambdas, stiffness), (
        inner,
        lambdas,
        stiffness,
        vjp_args,
        n_steps,
        constraints,
        key,
    )


@_implicit_forward_pass.def_bwd
def _implicit_forward_pass_bwd(
    res, g_out, perturbed, vjp_args, n_steps, constraints, key
):
    inner, lambdas, stiffness, vjp_args, _n_steps, constraints, _key = res
    solver, metric, p_start, p_end = vjp_args
    g_inner, _g_lambdas, _g_stiffness = g_out

    # Project outgoing loss gradients onto the tangent space so they map correctly to the Hessian
    g_inner_tan = jax.vmap(metric.manifold.to_tangent)(inner, g_inner)

    # -----------------------------------------------------------------------
    # Block-Tridiagonal Thomas Algorithm for the IFT adjoint system
    # -----------------------------------------------------------------------
    # The path Jacobian J = dG/dx* is block-tridiagonal with D×D blocks.
    # For N interior vertices and ambient dimension D, the dense solve is
    # O(N²D²) memory and O(N³D³) flops.  The Thomas algorithm reduces this
    # to O(N·D²) memory and O(N·D³) flops — a factor of N improvement.
    #
    # Block structure of J:
    #   Row k: A[k]·x[k-1] + B[k]·x[k] + C[k]·x[k+1]
    # where A[k] = dG_k/dx_{k-1}, B[k] = dG_k/dx_k, C[k] = dG_k/dx_{k+1}.
    # The adjoint (cotangent) system solved below is J^T·λ = g.
    #
    # Full path (N+2 points, first=p_start, last=p_end):
    #   full_ext[0]   = p_start  (boundary, fixed)
    #   full_ext[1..N] = inner[0..N-1]
    #   full_ext[N+1] = p_end    (boundary, fixed)
    #
    # For each inner vertex k (0-indexed), the local energy E_k depends only
    # on full_ext[k], full_ext[k+1], full_ext[k+2].

    n_inner = inner.shape[0]
    D = inner.shape[1]
    full_ext = jnp.concatenate([p_start[None], inner, p_end[None]], axis=0)

    # Compute all three D×D block Jacobians for every inner vertex via vmap.
    # vmap over k = 0 .. n_inner-1; dynamic indexing into full_ext is safe.
    ks = jnp.arange(n_inner)

    def get_ABC(k, lam_k, mu_k):
        xp = full_ext[k]
        xc = full_ext[k + 1]
        xn = full_ext[k + 2]

        def G_k(xp_, xc_, xn_):
            def L_loc(xc__):
                v_in = metric.manifold.log_map(xp_, xc__)
                v_out = metric.manifold.log_map(xc__, xn_)
                E = metric.energy(xp_, v_in) + metric.energy(xc__, v_out)
                # The converged residual is of the full Lagrangian, so the
                # IFT Jacobian must include the ALM penalty curvature.
                if constraints is not None and len(constraints) > 0:
                    c_vals = jnp.stack([c(xc__) for c in constraints])
                    E = E + jnp.sum(lam_k * c_vals + 0.5 * mu_k * c_vals**2)
                return E

            grad_euc = jax.grad(L_loc)(xc_)
            # Project to the tangent space so we differentiate the same map
            # whose root the solver found (see `_el_residual`).
            return metric.manifold.to_tangent(xc_, grad_euc)

        A_k = jax.jacobian(lambda xp_: G_k(xp_, xc, xn))(xp)  # dG_k/dx_{k-1}
        B_k = jax.jacobian(lambda xc_: G_k(xp, xc_, xn))(xc)  # dG_k/dx_k
        C_k = jax.jacobian(lambda xn_: G_k(xp, xc, xn_))(xn)  # dG_k/dx_{k+1}
        return A_k, B_k, C_k

    A_blocks, B_blocks, C_blocks = jax.vmap(get_ABC)(
        ks, lambdas, stiffness
    )  # each (N, D, D)

    # The cotangent solve needs J^T lam = g (not J lam = g). J^T is block-
    # tridiagonal with row-k blocks (C[k-1]^T, B[k]^T, A[k+1]^T). For the
    # symmetric case (Euclidean manifold, energy Hessian) this coincides with
    # (A, B, C), but with tangent projection or penalty terms J is generally
    # non-symmetric.
    AT = jnp.swapaxes(A_blocks, -1, -2)
    BT = jnp.swapaxes(B_blocks, -1, -2)
    CT = jnp.swapaxes(C_blocks, -1, -2)

    # Tikhonov regularisation on each diagonal block — canonical PSD floor.
    # Adds eps * I to each diagonal block before all solves, bounding the
    # condition number.
    _btd_reg = PSD_EPS
    g_rhs = g_inner_tan  # shape (N, D) — right-hand side of adjoint system

    # Forward sweep (Thomas forward elimination): left to right, k = 1 .. N-1.
    # Carry = (B'[k-1], g'[k-1]).  Scan input at step k = (sub[k], diag[k],
    # super[k-1], g[k]) of the transposed system.
    B0_reg = BT[0] + _btd_reg * jnp.eye(D, dtype=BT.dtype)

    def fwd_step(carry, inputs):
        B_prev, g_prev = carry
        A_k, B_k, C_prev, g_k = inputs
        # w = A_k @ inv(B_prev)  solved as  B_prev^T @ w^T = A_k^T
        w = jax.scipy.linalg.solve(B_prev.T, A_k.T, lower=False).T  # (D, D)
        B_new = B_k + _btd_reg * jnp.eye(D, dtype=B_k.dtype) - w @ C_prev
        g_new = g_k - w @ g_prev
        return (B_new, g_new), (B_new, g_new)

    if n_inner > 1:
        fwd_inputs = (
            CT[:-1],  # sub'[1..N-1]   = C[0..N-2]^T,  (N-1, D, D)
            BT[1:],  # diag'[1..N-1]  = B[1..N-1]^T
            AT[1:],  # super'[0..N-2] = A[1..N-1]^T
            g_rhs[1:],  # g[1..N-1],  (N-1, D)
        )
        (_, _), (B_prime_rest, g_prime_rest) = jax.lax.scan(
            fwd_step, (B0_reg, g_rhs[0]), fwd_inputs
        )
        B_prime = jnp.concatenate([B0_reg[None], B_prime_rest], axis=0)  # (N, D, D)
        g_prime = jnp.concatenate([g_rhs[0][None], g_prime_rest], axis=0)  # (N, D)
    else:
        B_prime = B0_reg[None]
        g_prime = g_rhs

    # Backward substitution: right to left, k = N-2 .. 0.
    # Solve for lam_last, then scan backwards.
    lam_last = jax.scipy.linalg.solve(B_prime[-1], g_prime[-1])

    def bwd_step(lam_next, inputs):
        B_k, g_k, C_k = inputs
        rhs = g_k - C_k @ lam_next
        lam_k = jax.scipy.linalg.solve(B_k, rhs)
        return lam_k, lam_k

    if n_inner > 1:
        bwd_inputs = (
            B_prime[:-1][::-1],  # B'[N-2..0]  reversed (N-1, D, D)
            g_prime[:-1][::-1],  # g'[N-2..0]  reversed (N-1, D)
            AT[1:][::-1],  # super'[N-2..0] = A[N-1..1]^T reversed (N-1, D, D)
        )
        _, lam_rest_rev = jax.lax.scan(bwd_step, lam_last, bwd_inputs)
        # lam_rest_rev is (N-1, D) in reversed order [lam_{N-2}, ..., lam_0]
        lam = jnp.concatenate([lam_rest_rev[::-1], lam_last[None]], axis=0)  # (N, D)
    else:
        lam = lam_last[None]

    # NaN guard: zero out any row that blew up (degenerate triangle / flat metric).
    lam = jnp.where(jnp.isfinite(lam), lam, jnp.zeros_like(lam))
    lam_flat = lam.ravel()  # (N*D,)

    # Partition metric into arrays and static fields using the differentiable mask
    mask = get_differentiable_mask(metric)
    m_arr, m_static = eqx.partition(metric, mask)

    def el_wrt_all(arr_leaves, p_s, p_e):
        m_l = eqx.combine(arr_leaves, m_static)
        return _el_residual(inner, m_l, p_s, p_e, constraints, lambdas, stiffness)

    # Compute VJP w.r.t dynamic metric parameters and boundary points simultaneously
    _, vjp_fn = jax.vjp(el_wrt_all, m_arr, p_start, p_end)
    grad_arr, grad_p_start, grad_p_end = vjp_fn(-lam_flat)[0:3]

    grad_m = eqx.combine(grad_arr, jax.tree_util.tree_map(lambda _: None, m_static))

    # Return gradients for all non-static arguments (solver, metric, p_start, p_end)
    s_mask = get_differentiable_mask(solver)
    s_arr, s_static = eqx.partition(solver, s_mask)
    grad_s_arr = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x) if eqx.is_inexact_array(x) else None, s_arr
    )
    grad_solver = eqx.combine(
        grad_s_arr, jax.tree_util.tree_map(lambda _: None, s_static)
    )

    return (grad_solver, grad_m, grad_p_start, grad_p_end)


class AVBDSolver(eqx.Module):
    """Augmented Vertex Block Descent (AVBD) Geodesic Solver.

    Finds the energy-minimizing path between two boundary points by directly
    optimizing discretized path vertices. The solver is fully JAX-compatible
    and differentiable with respect to metric parameters.

    Algorithm:
        Discretizes the path into T+1 vertices. Minimizes the discrete action
        L = sum(E(x_i, v_i)) + Penalty via Gauss-Seidel sweeps. Constraints
        are enforced using the Augmented Lagrangian Method (ALM).

    Note:
        The dual variables are updated after *every* primal sweep (a
        primal-dual variant rather than textbook ALM, which converges the
        primal subproblem between dual updates). Dual updates are gated on
        ``|c| > tol`` — once a constraint is satisfied to tolerance its
        multiplier and stiffness freeze — and the stiffness grows by ``beta``
        (capped at 1e6) only where the violation also failed to shrink by at
        least 4x since the previous sweep (the standard ALM stagnation
        schedule). Setting ``tol`` below the primal accuracy floor (single
        clipped GD sweeps, float32) together with a large ``beta`` can still
        destabilize the duals.

    Attributes:
        step_size: Learning rate for vertex gradient descent.
        beta: Growth factor for penalty stiffness (ALM).
        iterations: Maximum number of full path sweeps.
        grad_clip: Maximum norm for vertex gradients to ensure stability.
        energy_tol: Tolerance for early stopping (relative energy change).
        tol: Tolerance for maximum constraint violation.
        parallel: If True, use graph-coloring parallel sweeps instead of
            sequential Gauss-Seidel. Vertices are 2-colored (even/odd) and
            each color group is updated simultaneously via vmap. This
            trades strict Gauss-Seidel convergence for GPU parallelism.
        implicit_diff: Enable O(1) memory analytical adjoints during backprop.
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
        constraints: Optional[list[Callable[[jax.Array], jax.Array]]] = None,
        train_mode: bool = True,
        key: Optional[jax.Array] = None,
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
        if self.implicit_diff:
            return self._solve_implicit(
                metric, p_start, p_end, n_steps, constraints, train_mode, key
            )
        return self._solve_core(
            metric, p_start, p_end, n_steps, constraints, train_mode, key
        )

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
        """Implicit-differentiation wrapper around ``_solve_core``."""
        inner, lambdas, stiffness = _implicit_forward_pass(
            (self, metric, p_start, p_end), n_steps, constraints, key
        )
        full_xs = jnp.concatenate([p_start[None], inner, p_end[None]], axis=0)
        full_vs = jax.vmap(metric.manifold.log_map)(full_xs[:-1], full_xs[1:])
        energy = jnp.sum(jax.vmap(metric.energy)(full_xs[:-1], full_vs))

        # Determine violation via evaluated constraints
        violation = jnp.array(0.0)
        if constraints is not None and len(constraints) > 0:
            c_vals = jnp.stack([jax.vmap(c)(inner) for c in constraints], axis=-1)
            violation = jnp.max(jnp.abs(c_vals))

        return Trajectory(
            xs=full_xs,
            vs=full_vs,
            energy=energy,
            constraint_violation=violation,
            lambdas=lambdas,
            stiffness=stiffness,
        )

    def _solve_core(
        self,
        metric: FinslerMetric,
        p_start: jax.Array,
        p_end: jax.Array,
        n_steps: int = 10,
        constraints: Optional[list[Callable[[jax.Array], jax.Array]]] = None,
        train_mode: bool = True,
        key: Optional[jax.Array] = None,
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
        tol = self.tol
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
        # inf sentinel: the first iteration never counts as stagnation.
        init_prev_c = jnp.full((n_inner, num_constraints), jnp.inf)

        state = SolverState(
            path=init_inner,
            lambdas=init_lambdas,
            stiffness=init_stiffness,
            prev_c=init_prev_c,
            step=0,
            max_violation=jnp.array(0.0),
            prev_energy=jnp.array(jnp.inf),
            curr_energy=jnp.array(0.0),
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
            color_0_raw, color_1_raw = chain_coloring(n_inner)
            # Offset by 1 since full_path includes p_start at index 0
            color_0 = color_0_raw + 1
            color_1 = color_1_raw + 1

        def parallel_color_sweep(full_path, color_indices, s):
            """Update all vertices of one color simultaneously."""
            x_prevs = full_path[color_indices - 1]
            xs = full_path[color_indices]
            x_nexts = full_path[color_indices + 1]
            # Map back to 0-indexed inner arrays for dual variables
            lams = s.lambdas[color_indices - 1]
            mus = s.stiffness[color_indices - 1]
            new_xs = jax.vmap(update_vertex_triple)(x_prevs, xs, x_nexts, lams, mus)
            return full_path.at[color_indices].set(new_xs)

        def step_fn(s: SolverState, _):
            full_path = jnp.concatenate(
                [p_start[None, :], s.path, p_end[None, :]], axis=0
            )

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
            new_prev_c = s.prev_c

            if num_constraints > 0:

                def get_c_all(x):
                    return jnp.stack([c(x) for c in actual_constraints])

                all_c = jax.vmap(get_c_all)(new_inner)  # (n_inner, num_constraints)
                # Freeze duals where the constraint is already satisfied to
                # tolerance: once |c| sits at the primal accuracy floor,
                # further lambda accumulation and stiffness growth destabilize
                # the bounded-step sweeps (the solve diverges *after* having
                # converged).
                needs_work = jnp.abs(all_c) > tol
                new_lambdas = s.lambdas + jnp.where(
                    needs_work, s.stiffness * all_c, 0.0
                )
                # Standard ALM schedule: grow the penalty only where the
                # violation also failed to shrink sufficiently since the last
                # sweep. Unconditional geometric growth races the stiffness to
                # the cap, which the clipped primal steps cannot track.
                stalled = needs_work & (jnp.abs(all_c) > 0.25 * jnp.abs(s.prev_c))
                new_stiffness = jnp.where(
                    stalled, jnp.minimum(s.stiffness * beta, 1e6), s.stiffness
                )
                max_v = jnp.max(jnp.abs(all_c))
                new_prev_c = all_c

            # C. Energy Monitoring
            vels = jax.vmap(metric.manifold.log_map)(new_full[:-1], new_full[1:])
            total_E = jnp.sum(jax.vmap(metric.energy)(new_full[:-1], vels))

            return SolverState(
                path=new_inner,
                lambdas=new_lambdas,
                stiffness=new_stiffness,
                prev_c=new_prev_c,
                step=s.step + 1,
                max_violation=max_v,
                prev_energy=s.curr_energy,
                curr_energy=total_E,
            ), None

        # Execution
        if train_mode:
            final_state, _ = jax.lax.scan(step_fn, state, None, length=iterations)
        else:

            def cond(s):
                iter_not_done = s.step < iterations
                energy_not_conv = jnp.abs(s.curr_energy - s.prev_energy) > energy_tol
                constraint_not_met = s.max_violation > tol
                return iter_not_done & (energy_not_conv | constraint_not_met)

            final_state = jax.lax.while_loop(cond, lambda s: step_fn(s, None)[0], state)

        # Final Trajectory Reassembly
        full_xs = jnp.concatenate(
            [p_start[None, :], final_state.path, p_end[None, :]], axis=0
        )
        full_vs = jax.vmap(metric.manifold.log_map)(full_xs[:-1], full_xs[1:])

        return Trajectory(
            xs=full_xs,
            vs=full_vs,
            energy=final_state.curr_energy,
            constraint_violation=final_state.max_violation,
            lambdas=final_state.lambdas,
            stiffness=final_state.stiffness,
        )
