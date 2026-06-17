"""Baselines & controls for the asymmetry-pays headline (PLAN §9).

* :func:`symmetric_riemannian` — ``β=0`` (``use_wind=False``).  The core internal
  control: "does the asymmetry pay?"  Should tie the full drift at ``κ=0`` and
  fall behind as ``κ`` grows.
* :func:`potential_only_wind` — wind = the **gradient part only** of the
  reconstructed drift (``HodgeField.grad_part``).  The Hodge-predicted negative
  control: a gradient one-form only tilts cost, so it cannot represent the flux
  and must track the symmetric baseline at ``κ>0``.
* :func:`cellrank_fate` — a CellRank-2-style velocity-kernel Markov chain →
  absorption probabilities.  The discrete/graph fate-prediction SOTA the
  continuous Randers geodesic is benchmarked against.
* Dynamo **Least-Action-Path** — the theory check (H3) lives in
  :func:`landscape.least_action_path` (numeric min-action of the *known* ``f,D``);
  re-exported here for discoverability.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .landscape import least_action_path  # noqa: F401  (re-export, H3 baseline)
from .metric import ScaledField, build_randers


class _ZeroField(eqx.Module):
    dim: int = eqx.field(static=True)

    def __call__(self, z):
        return jnp.zeros((self.dim,), dtype=z.dtype)


class _GradPartField(eqx.Module):
    """Expose a :class:`HodgeField`'s gradient part as a standalone wind module.

    Wrapping the bound method keeps the underlying ``HodgeField`` arrays as proper
    (traced) PyTree children when the metric is passed to a jitted solver, instead
    of smuggling them in as a bound-method closure.
    """

    hodge: eqx.Module

    def __call__(self, z):
        return self.hodge.grad_part(z)


# =============================================================================
# Metric controls
# =============================================================================
def symmetric_riemannian(sea_fn, *, dim: int):
    """Symmetric-Riemannian metric (``β=0``): the "does asymmetry pay" control."""
    # use_wind=False zeroes the wind; the drift_fn is irrelevant but required.
    zero = ScaledField(_ZeroField(dim), 0.0)
    return build_randers(sea_fn, zero, dim=dim, wind_scale=0.0, use_wind=False)


def potential_only_randers(sea_fn, hodge_field, *, dim: int, points, margin: float = 0.85):
    """Randers with wind = gradient (conservative) part only (negative control).

    By the Hodge argument this can only tilt cost endpoint-to-endpoint and cannot
    bend geodesics irreversibly, so at ``κ>0`` it should fail to recover the flux
    and track the symmetric baseline (PLAN §6.2 note).
    """
    return build_randers(sea_fn, _GradPartField(hodge_field), dim=dim, points=points, margin=margin)


# =============================================================================
# CellRank-style velocity-kernel Markov fate
# =============================================================================
def cellrank_fate(
    X: np.ndarray,
    velocity: np.ndarray,
    terminal_masks: list[np.ndarray],
    *,
    k: int = 30,
    temperature: float = 0.3,
) -> np.ndarray:
    """Velocity-biased kNN Markov chain → per-cell absorption probabilities.

    For each cell, transitions to kNN neighbours are weighted by the cosine
    alignment between the edge direction and the cell's RNA velocity (the
    CellRank-2 velocity kernel), then row-normalised to a stochastic matrix.
    Terminal cells are made absorbing; absorption probabilities are obtained from
    the fundamental matrix ``B = (I - Q)⁻¹ R`` (PLAN §9 baseline row).

    Args:
        X: Embedding, shape ``(n, d)``.
        velocity: RNA velocity in the same frame, shape ``(n, d)``.
        terminal_masks: list of boolean masks (one per macrostate / fate).
        k: neighbours per cell.
        temperature: softmax temperature on the cosine alignment.

    Returns:
        Absorption probabilities, shape ``(n, n_fates)``.
    """
    X = np.asarray(X, np.float64)
    velocity = np.asarray(velocity, np.float64)
    n = X.shape[0]

    # kNN via brute-force distances (CPU-friendly at this scale).
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nbrs = np.argsort(d2, axis=1)[:, :k]  # (n, k)

    P = np.zeros((n, n))
    for i in range(n):
        edges = X[nbrs[i]] - X[i]  # (k, d)
        en = np.linalg.norm(edges, axis=1) + 1e-8
        vn = np.linalg.norm(velocity[i]) + 1e-8
        cos = (edges @ velocity[i]) / (en * vn)
        w = np.exp(cos / temperature)
        w /= w.sum()
        P[i, nbrs[i]] = w

    # Absorbing macrostates: rows of terminal cells point only to themselves.
    is_term = np.zeros(n, bool)
    fate_of_term = np.full(n, -1)
    for f, mask in enumerate(terminal_masks):
        idx = np.nonzero(mask)[0]
        is_term[idx] = True
        fate_of_term[idx] = f
    for i in np.nonzero(is_term)[0]:
        P[i] = 0.0
        P[i, i] = 1.0

    trans = np.nonzero(~is_term)[0]
    term = np.nonzero(is_term)[0]
    n_f = len(terminal_masks)

    Q = P[np.ix_(trans, trans)]
    R = P[np.ix_(trans, term)]
    # aggregate columns by fate
    R_fate = np.zeros((len(trans), n_f))
    for j, t in enumerate(term):
        R_fate[:, fate_of_term[t]] += R[:, j]

    B = np.linalg.solve(np.eye(len(trans)) - Q, R_fate)  # (|trans|, n_f)

    absorption = np.zeros((n, n_f))
    absorption[trans] = B
    for i in term:
        absorption[i, fate_of_term[i]] = 1.0
    return absorption
