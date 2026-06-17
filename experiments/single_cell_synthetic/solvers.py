"""Geodesic solvers — exact BVP (continuation) + amortized Randers Flow Matching.

Two ways to get a Randers geodesic between latent endpoints:

* :func:`exact_geodesic` — the exact discrete BVP via HAM's
  ``solve_continuation`` (multilevel coarse→fine AVBD warm-starts, then a
  Gauss–Newton polish).  Continuation is mandatory on stiff metrics — a cold
  AVBD solve on a long latent geodesic suffers O(N²) Gauss-Seidel critical
  slowing and fixed-step divergence (memory ``avbd-long-geodesic-diagnosis``,
  ``avbd-solver-upgrade-verdict``).  This is the ground-truth path for Stage A
  and the reference Stage D amortization is measured against.

* :class:`RandersFlowMatching` + :func:`train_rfm` — a low-parameter amortized
  interpolant ``φθ(z₀,z₁,t)`` trained to **minimize the asymmetric Randers
  action** ``∫F(γ,γ̇)²dt`` over sampled endpoint pairs (PLAN Stage D / H4).  This
  generalizes Metric Flow Matching's symmetric kinetic-energy interpolant to the
  *directed* Randers action, and is the enabler for real 130k-cell scale where a
  BVP per lineage triple inside the loop is infeasible.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ham.solvers import AVBDSolver, GaussNewtonGeodesic, Trajectory, solve_continuation


# =============================================================================
# Exact BVP geodesic via continuation
# =============================================================================
def exact_geodesic(
    metric,
    z0: jax.Array,
    z1: jax.Array,
    *,
    n_steps: int = 32,
    schedule: tuple[int, ...] = (8, 16, 32),
    avbd_iters: int = 300,
    step_size: float = 0.03,
    gn_iters: int = 25,
    init_path: jax.Array | None = None,
) -> Trajectory:
    """Exact Randers geodesic via multilevel AVBD continuation + GN polish.

    Args:
        metric: A :class:`ham.geometry.zoo.Randers` (or any FinslerMetric).
        z0, z1: Endpoints, shape ``(d,)``.
        n_steps: Final path resolution (overrides last entry of ``schedule``).
        schedule: Coarse→fine vertex counts for the AVBD continuation stages.
        avbd_iters: AVBD sweeps per stage.
        step_size: AVBD vertex step size.
        gn_iters: Gauss–Newton polish iterations (N-independent convergence).
        init_path: Optional warm-start for the first (coarsest) stage.

    Returns:
        The final :class:`Trajectory` (``.xs`` path, ``.energy`` = ½ΣF² action).
    """
    schedule = (*tuple(schedule[:-1]), n_steps)
    stages = []
    for k in schedule:
        stages.append((AVBDSolver(step_size=step_size, iterations=avbd_iters, grad_clip=10.0), metric, k))
    stages.append((GaussNewtonGeodesic(iterations=gn_iters), metric, n_steps))
    return solve_continuation(stages, z0, z1, init_path=init_path)


# =============================================================================
# Amortized Randers Flow Matching (Stage D)
# =============================================================================
class RandersFlowMatching(eqx.Module):
    """Amortized geodesic interpolant ``γ(t) = lerp(z₀,z₁,t) + t(1-t)·hθ(z₀,z₁,t)``.

    The boundary bump ``t(1-t)`` pins ``γ(0)=z₀, γ(1)=z₁`` exactly for any net
    output, so only the *shape* between the endpoints is learned.  Trained to
    minimize the directed Randers action; one forward pass replaces a BVP solve.
    """

    mlp: eqx.nn.MLP
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, key, width: int = 64, depth: int = 3):
        self.dim = dim
        self.mlp = eqx.nn.MLP(
            in_size=2 * dim + 1, out_size=dim, width_size=width, depth=depth,
            activation=jax.nn.tanh, key=key,
        )

    def point(self, z0, z1, t):
        """Interpolant point ``γ(t)`` for scalar ``t``."""
        inp = jnp.concatenate([z0, z1, jnp.atleast_1d(t)])
        base = (1.0 - t) * z0 + t * z1
        return base + t * (1.0 - t) * self.mlp(inp)

    def path(self, z0, z1, n_steps: int = 32) -> jax.Array:
        """Materialize the interpolant as an ``(n_steps+1, d)`` path."""
        ts = jnp.linspace(0.0, 1.0, n_steps + 1)
        return jax.vmap(lambda t: self.point(z0, z1, t))(ts)


def _path_action(interp: RandersFlowMatching, metric, z0, z1, n_quad: int) -> jax.Array:
    """Discrete Randers action ``Σ F(γ, γ̇)²·Δt`` of the interpolant for one pair."""
    ts = jnp.linspace(0.0, 1.0, n_quad + 1)
    pts = jax.vmap(lambda t: interp.point(z0, z1, t))(ts)
    v = (pts[1:] - pts[:-1]) * n_quad  # γ̇ ≈ Δγ/Δt, Δt = 1/n_quad
    mid = 0.5 * (pts[1:] + pts[:-1])
    F = jax.vmap(metric.metric_fn)(mid, v)
    return jnp.sum(F**2) / n_quad


def train_rfm(
    metric,
    pairs: np.ndarray,
    *,
    dim: int,
    key,
    width: int = 64,
    depth: int = 3,
    steps: int = 1500,
    lr: float = 3e-3,
    batch: int = 64,
    n_quad: int = 24,
) -> tuple[RandersFlowMatching, list[float]]:
    """Train the amortized interpolant to minimize mean Randers action over pairs.

    Args:
        metric: The Randers metric whose action is minimized.
        pairs: Endpoint pairs, shape ``(P, 2, d)`` (``pairs[:,0]`` = z₀).
        dim: Latent dimension.
        key: PRNG key.
        steps, lr, batch, n_quad: optimization hyperparameters.

    Returns:
        ``(trained_interp, loss_history)``.
    """
    import optax

    pairs = jnp.asarray(pairs, jnp.float32)
    interp = RandersFlowMatching(dim, key, width, depth)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(interp, eqx.is_array))

    @eqx.filter_jit
    def step(interp, opt_state, batch_pairs):
        def loss_fn(m):
            acts = jax.vmap(lambda p: _path_action(m, metric, p[0], p[1], n_quad))(batch_pairs)
            return jnp.mean(acts)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(interp)
        updates, opt_state = opt.update(grads, opt_state)
        interp = eqx.apply_updates(interp, updates)
        return interp, opt_state, loss

    rng = np.random.default_rng(0)
    p = pairs.shape[0]
    history = []
    for _ in range(steps):
        idx = rng.choice(p, size=min(batch, p), replace=False)
        interp, opt_state, loss = step(interp, opt_state, pairs[idx])
        history.append(float(loss))
    return interp, history
