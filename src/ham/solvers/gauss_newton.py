"""Global block-tridiagonal Gauss-Newton geodesic solver.

Where :class:`~ham.solvers.avbd.AVBDSolver` relaxes the discrete path with
*local* Gauss-Seidel gradient steps — whose convergence suffers the classic
O(N^2) critical slowing down of a 1-D Laplacian relaxation (see
``spec/AVBD_LATENT_FINDINGS_2026-06-14.md``) — this solver takes a *global*
second-order step over the whole path at once.

The discrete path energy E(x_1, ..., x_{N-1}) with fixed endpoints has a Hessian
that is **block-tridiagonal** with D x D blocks, because each segment energy
couples only neighbouring vertices.  A damped Newton (Levenberg-Marquardt) step

    (H + mu I) dx = -g

is solved in O(N D^3) by the block-Thomas algorithm — the *same* structure the
implicit adjoint in :mod:`ham.solvers.avbd` already uses for the backward pass.
Because the step is global, low-frequency path deformations are resolved in a
number of iterations that is **independent of N** (≈ tens), eliminating the
critical-slowing wall.

Trade-offs and scope:

* Designed for the latent-space use case — a :class:`~ham.geometry.manifolds.EuclideanSpace`
  (or any manifold whose ``log_map`` is the projected secant).  Curved-manifold
  retraction is handled by ``manifold.project`` but the block-tridiagonal Hessian
  assumes the ambient parameterisation; for strongly curved intrinsic manifolds
  prefer :class:`AVBDSolver`.
* Newton is **locally** convergent: from a cold straight-line guess that dives
  into a stiff, high-cost void it can stall.  Pair it with numerical continuation
  (:func:`ham.solvers.continuation.solve_continuation`) — anneal the metric and/or
  refine the resolution, warm-starting via ``init_path`` — for stiff metrics.

Reference:
    Block-tridiagonal Newton for discrete geodesic BVPs; cf. discrete geodesic
    calculus (Rumpf-Wirth) and trajectory-optimisation Riccati/LQR structure.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.metric import FinslerMetric
from ham.solvers.avbd import Trajectory

__all__ = ["GaussNewtonGeodesic"]


def _local_pair_energy(metric, xprev, xc, xnext):
    """Energy of the two segments adjacent to the interior vertex ``xc``."""
    v_in = metric.manifold.log_map(xprev, xc)
    v_out = metric.manifold.log_map(xc, xnext)
    return metric.energy(xprev, v_in) + metric.energy(xc, v_out)


def _total_energy(interior, metric, p0, pN):
    full = jnp.concatenate([p0[None], interior, pN[None]], axis=0)
    vs = jax.vmap(metric.manifold.log_map)(full[:-1], full[1:])
    return jnp.sum(jax.vmap(metric.energy)(full[:-1], vs))


def _build_blocks(interior, metric, p0, pN):
    """Gradient and block-tridiagonal Hessian blocks (A sub-, B diagonal, C super-).

    For interior vertex k: B[k] = d2E/dx_k^2, A[k] = d2E/dx_k dx_{k-1},
    C[k] = d2E/dx_k dx_{k+1}.  Shapes: g (n,D); A,B,C (n,D,D).
    """
    full = jnp.concatenate([p0[None], interior, pN[None]], axis=0)
    n_inner = interior.shape[0]
    ks = jnp.arange(n_inner)

    def per_vertex(k):
        xp, xc, xn = full[k], full[k + 1], full[k + 2]
        g = jax.grad(lambda x: _local_pair_energy(metric, xp, x, xn))(xc)
        B = jax.hessian(lambda x: _local_pair_energy(metric, xp, x, xn))(xc)
        A = jax.jacobian(
            lambda xpr: jax.grad(lambda x: _local_pair_energy(metric, xpr, x, xn))(xc)
        )(xp)
        C = jax.jacobian(
            lambda xnx: jax.grad(lambda x: _local_pair_energy(metric, xp, x, xnx))(xc)
        )(xn)
        return g, A, B, C

    return jax.vmap(per_vertex)(ks)


def _block_thomas(A, B, C, rhs, mu):
    """Solve (H + mu I) x = rhs for the symmetric block-tridiagonal H.

    Diagonal blocks B[k], sub-diagonal A[k] (couples to k-1), super-diagonal C[k]
    (couples to k+1).  Returns x of shape (n, D).  Robust to n == 1.
    """
    n, D = rhs.shape
    eye = jnp.eye(D, dtype=B.dtype)
    Bd = B + mu * eye

    def fwd(carry, inp):
        Bprev, gprev = carry
        A_k, B_k, C_prev, r_k = inp
        w = jax.scipy.linalg.solve(Bprev, A_k)  # A_k @ inv(Bprev)
        Bnew = B_k - w @ C_prev
        rnew = r_k - w @ gprev
        return (Bnew, rnew), (Bnew, rnew)

    fwd_inputs = (A[1:], Bd[1:], C[:-1], rhs[1:])
    (_, _), (Bp_rest, rp_rest) = jax.lax.scan(fwd, (Bd[0], rhs[0]), fwd_inputs)
    Bp = jnp.concatenate([Bd[0][None], Bp_rest], axis=0)
    rp = jnp.concatenate([rhs[0][None], rp_rest], axis=0)

    x_last = jax.scipy.linalg.solve(Bp[-1], rp[-1])

    def bwd(x_next, inp):
        Bp_k, rp_k, C_k = inp
        x_k = jax.scipy.linalg.solve(Bp_k, rp_k - C_k @ x_next)
        return x_k, x_k

    bwd_inputs = (Bp[:-1][::-1], rp[:-1][::-1], C[:-1][::-1])
    _, x_rev = jax.lax.scan(bwd, x_last, bwd_inputs)
    x = jnp.concatenate([x_rev[::-1], x_last[None]], axis=0)
    return x


class _GNState(eqx.Module):
    interior: jax.Array
    mu: jax.Array
    energy: jax.Array
    grad_norm: jax.Array


class GaussNewtonGeodesic(eqx.Module):
    """Global LM-damped block-tridiagonal Newton geodesic solver.

    Attributes:
        iterations: Maximum number of (damped) Newton iterations.
        mu0: Initial Levenberg-Marquardt damping.
        mu_dec: Factor to shrink mu on an accepted step (< 1).
        mu_inc: Factor to grow mu on a rejected step (> 1).
        mu_min, mu_max: Damping clamp range.
        tol: Stop when the projected-gradient norm falls below this.
    """

    iterations: int = eqx.field(static=True, default=40)
    mu0: float = eqx.field(static=True, default=1e-2)
    mu_dec: float = eqx.field(static=True, default=0.4)
    mu_inc: float = eqx.field(static=True, default=4.0)
    mu_min: float = eqx.field(static=True, default=1e-9)
    mu_max: float = eqx.field(static=True, default=1e9)
    line_search: int = eqx.field(static=True, default=8)
    tol: float = eqx.field(static=True, default=1e-8)

    def solve(
        self,
        metric: FinslerMetric,
        p_start: jax.Array,
        p_end: jax.Array,
        n_steps: int = 10,
        init_path: Optional[jax.Array] = None,
        train_mode: bool = True,
    ) -> Trajectory:
        """Find the energy-minimising geodesic between two points.

        Args:
            metric: FinslerMetric defining the geometry.
            p_start, p_end: Boundary points, shape (D,).
            n_steps: Number of path segments (path has n_steps + 1 vertices).
            init_path: Optional warm-start path, shape (n_steps + 1, D). The
                boundary vertices are overwritten by p_start / p_end. Strongly
                recommended for stiff metrics (use numerical continuation).
            train_mode: If True, runs a fixed ``iterations`` ``lax.scan`` (jittable,
                differentiable via unrolling). If False, uses a ``while_loop`` with
                early stopping on ``tol``.

        Returns:
            A :class:`~ham.solvers.avbd.Trajectory`.
        """
        D = p_start.shape[0]
        n_inner = n_steps - 1

        if init_path is None:
            t = jnp.linspace(0.0, 1.0, n_steps + 1)[:, None]
            base = (1 - t) * p_start + t * p_end
        else:
            base = jnp.concatenate(
                [p_start[None], init_path[1:-1], p_end[None]], axis=0
            )
        interior = jax.vmap(metric.manifold.project)(base[1:-1])

        E0 = _total_energy(interior, metric, p_start, p_end)
        state = _GNState(
            interior=interior,
            mu=jnp.asarray(self.mu0, interior.dtype),
            energy=E0,
            grad_norm=jnp.asarray(jnp.inf, interior.dtype),
        )

        mu_dec, mu_inc = self.mu_dec, self.mu_inc
        mu_min, mu_max = self.mu_min, self.mu_max
        n_ls = self.line_search

        def step(s: _GNState) -> _GNState:
            # Assemble the global gradient + block-tridiagonal Hessian once.
            g, A, B, C = _build_blocks(s.interior, metric, p_start, p_end)
            gnorm = jnp.linalg.norm(g)

            # Bounded LM line search: try increasing damping until the *total*
            # path energy decreases (the trust-region globalisation that makes
            # the second-order step robust in stiff, non-convex voids). Blocks
            # are reused across trials, so each trial is just a block-Thomas solve.
            def trial(_j, carry):
                mu, found, best_int, best_E, succ_mu = carry
                dx = _block_thomas(A, B, C, -g, mu)
                cand = jax.vmap(metric.manifold.project)(s.interior + dx)
                E = _total_energy(cand, metric, p_start, p_end)
                take = jnp.isfinite(E) & (E < s.energy) & (~found)
                best_int = jnp.where(take, cand, best_int)
                best_E = jnp.where(take, E, best_E)
                succ_mu = jnp.where(take, mu, succ_mu)
                found = found | take
                next_mu = jnp.where(found, mu, jnp.minimum(mu * mu_inc, mu_max))
                return (next_mu, found, best_int, best_E, succ_mu)

            init = (s.mu, jnp.bool_(False), s.interior, s.energy, s.mu)
            mu_end, found, best_int, best_E, succ_mu = jax.lax.fori_loop(
                0, n_ls, trial, init
            )
            # On success shrink the damping toward a full Newton step; otherwise
            # keep the grown damping (and the unchanged iterate) for the next iter.
            new_mu = jnp.where(
                found,
                jnp.maximum(succ_mu * mu_dec, mu_min),
                jnp.minimum(mu_end, mu_max),
            )
            return _GNState(
                interior=best_int,
                mu=new_mu,
                energy=best_E,
                grad_norm=gnorm,
            )

        if train_mode:
            final = jax.lax.fori_loop(
                0, self.iterations, lambda _i, s: step(s), state
            )
        else:
            # Early-stop on the projected-gradient norm. Prime with one step so
            # grad_norm is populated, then loop until convergence or the cap.
            def cond(carry):
                s, i = carry
                return (i < self.iterations) & (s.grad_norm > self.tol)

            def body(carry):
                s, i = carry
                return step(s), i + 1

            final, _ = jax.lax.while_loop(cond, body, (step(state), 1))

        full_xs = jnp.concatenate(
            [p_start[None], final.interior, p_end[None]], axis=0
        )
        full_vs = jax.vmap(metric.manifold.log_map)(full_xs[:-1], full_xs[1:])
        return Trajectory(
            xs=full_xs,
            vs=full_vs,
            energy=final.energy,
            constraint_violation=jnp.array(0.0),
            lambdas=jnp.zeros((n_inner, 0)),
            stiffness=jnp.zeros((n_inner, 0)),
        )
