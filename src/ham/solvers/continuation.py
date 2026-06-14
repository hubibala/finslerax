"""Numerical continuation drivers for geodesic boundary-value solvers.

The discrete geodesic energy under a stiff, data-driven metric is non-convex, and
a cold straight-line guess between distant latents dives into a high-cost void
where *every* local solver (Gauss-Seidel relaxation or Newton alike) is unstable
or critically slow (see ``spec/AVBD_LATENT_FINDINGS_2026-06-14.md``). Numerical
continuation is the standard cure: solve a sequence of easier problems, each
warm-started from the previous solution, so the iterate is always inside a good
basin.

Two homotopies are useful and compose:

* **Metric annealing** — start from a gentle metric (small conformal exponent /
  bandwidth) and increase its stiffness across stages. Defuses the divergence of
  fixed-step updates in stiff voids.
* **Multilevel / coarse-to-fine** — solve at a coarse resolution (small N, cheap
  and free of O(N^2) critical slowing), then refine. The low-frequency shape is
  fixed cheaply on the coarse path and only locally polished on fine paths.

Both reduce to repeated calls to a solver's ``solve(..., init_path=...)`` with the
previous (resampled) path carried forward. Any solver exposing that signature
(``AVBDSolver``, ``GaussNewtonGeodesic``) works as a stage solver.
"""

from typing import Iterable, Optional

import jax
import jax.numpy as jnp

from ham.solvers.avbd import Trajectory

__all__ = ["resample_path", "reparametrize_arclength", "solve_continuation"]


def resample_path(path: jax.Array, n_steps: int, project=None) -> jax.Array:
    """Resample a path to ``n_steps + 1`` vertices by linear interpolation.

    Interpolates each coordinate against a uniform parameter in [0, 1]. Endpoints
    are preserved exactly. Used to transfer a coarse warm-start onto a finer grid
    (multilevel) or vice versa.

    Args:
        path: Source path, shape (M + 1, D).
        n_steps: Target number of segments; output has ``n_steps + 1`` vertices.
        project: Optional ``manifold.project`` callable applied per vertex after
            interpolation (keeps the resampled path on the manifold).

    Returns:
        Resampled path, shape (n_steps + 1, D).
    """
    m = path.shape[0]
    src_t = jnp.linspace(0.0, 1.0, m)
    dst_t = jnp.linspace(0.0, 1.0, n_steps + 1)
    out = jax.vmap(lambda col: jnp.interp(dst_t, src_t, col), in_axes=1, out_axes=1)(
        path
    )
    if project is not None:
        out = jax.vmap(project)(out)
    return out


def reparametrize_arclength(path: jax.Array, n_steps: Optional[int] = None) -> jax.Array:
    """Resample a path to uniform **arc length** (vs uniform parameter index).

    Energy-minimising geodesics are constant-*Finsler*-speed, so their vertices
    are sparser where the metric is cheap and denser where it is expensive — this
    looks like bunching and limits how well a fixed vertex budget resolves a
    curved valley.  Reparametrising to uniform Euclidean arc length redistributes
    the vertices evenly along the *same* curve (it changes only the
    parametrisation, not the path image), which is what you usually want for
    display and for downstream uniform sampling.

    Args:
        path: Source path, shape ``(M + 1, D)``.
        n_steps: Target number of segments; output has ``n_steps + 1`` vertices.
            Defaults to keeping the input vertex count.

    Returns:
        Arc-length-uniform path, shape ``(n_steps + 1, D)`` with exact endpoints.
    """
    if n_steps is None:
        n_steps = path.shape[0] - 1
    seg = jnp.linalg.norm(path[1:] - path[:-1], axis=1)
    s = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    total = s[-1]
    # Guard a degenerate (zero-length) path: fall back to index parametrisation.
    src_t = jnp.where(total > 0, s / jnp.where(total > 0, total, 1.0),
                      jnp.linspace(0.0, 1.0, path.shape[0]))
    dst_t = jnp.linspace(0.0, 1.0, n_steps + 1)
    return jax.vmap(lambda col: jnp.interp(dst_t, src_t, col), in_axes=1, out_axes=1)(
        path
    )


def solve_continuation(
    stages: Iterable,
    p_start: jax.Array,
    p_end: jax.Array,
    *,
    init_path: Optional[jax.Array] = None,
    return_history: bool = False,
):
    """Run a geodesic solve as a sequence of warm-started continuation stages.

    Each stage is a ``(solver, metric, n_steps)`` tuple. The path from one stage is
    resampled to the next stage's ``n_steps`` (via :func:`resample_path`, projected
    onto the metric's manifold) and passed as ``init_path``. Annealing and
    multilevel are both expressed by varying ``metric`` and ``n_steps`` across
    stages.

    Args:
        stages: Iterable of ``(solver, metric, n_steps)``. ``solver`` must expose
            ``solve(metric, p_start, p_end, n_steps=..., init_path=...)`` returning a
            :class:`~ham.solvers.avbd.Trajectory`.
        p_start, p_end: Boundary points, shape (D,).
        init_path: Optional warm-start for the *first* stage, shape
            ``(stages[0].n_steps + 1, D)``.
        return_history: If True, also return the list of per-stage trajectories.

    Returns:
        The final :class:`~ham.solvers.avbd.Trajectory`, or
        ``(final_traj, [stage_trajs...])`` if ``return_history``.

    Example:
        >>> stages = [
        ...     (AVBDSolver(step_size=0.05, iterations=300), make_metric(alpha=1.0), 8),
        ...     (AVBDSolver(step_size=0.05, iterations=400), make_metric(alpha=4.0), 16),
        ...     (GaussNewtonGeodesic(iterations=30),         make_metric(alpha=8.0), 32),
        ... ]
        >>> traj = solve_continuation(stages, z0, z1)
    """
    stages = list(stages)
    if not stages:
        raise ValueError("solve_continuation requires at least one stage")

    path = init_path
    history = []
    for solver, metric, n_steps in stages:
        if path is not None and path.shape[0] != n_steps + 1:
            path = resample_path(path, n_steps, project=metric.manifold.project)
        traj = solver.solve(
            metric, p_start, p_end, n_steps=n_steps, init_path=path
        )
        path = traj.xs
        history.append(traj)

    final = history[-1]
    if return_history:
        return final, history
    return final
