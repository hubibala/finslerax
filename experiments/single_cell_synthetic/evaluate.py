"""Evaluation — recovery, directionality, lineage, fate, collapse, LAP (PLAN §8).

Nothing here optimizes; it measures recovered geometry against the synthetic
ground truth.  Every oracle-dependent metric takes explicit ground-truth arrays
or a :class:`Landscape`, so the *same* functions run on real data by simply not
calling the oracle ones (the ``SingleCellDataset.has_oracle`` gate).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from .landscape import Landscape
from .solvers import exact_geodesic


# =============================================================================
# Vector-field recovery (drift, Hodge-resolved) — H1/H2
# =============================================================================
def mean_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """Mean per-point cosine similarity between two fields ``(N, d)``."""
    A, B = np.asarray(A), np.asarray(B)
    num = np.sum(A * B, axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8
    return float(np.mean(num / den))


def global_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """Magnitude-weighted (global) cosine — robust where one field is sparse.

    The localized flux is near-zero away from the saddle, so a per-point cosine
    there is pure direction noise; the global cosine weights by magnitude and is
    the honest flux-recovery score (it is exactly 0 when the true field is 0).
    """
    A, B = np.asarray(A), np.asarray(B)
    num = np.sum(A * B)
    den = np.sqrt(np.sum(A * A)) * np.sqrt(np.sum(B * B)) + 1e-12
    return float(num / den)


def curl(field_fn, points: np.ndarray) -> np.ndarray:
    """Scalar vorticity ``(∇×f)_z`` of a differentiable field at ``points``."""
    pts = jnp.asarray(points, jnp.float32)
    jac = jax.vmap(jax.jacobian(field_fn))(pts)
    return np.asarray(jac[:, 1, 0] - jac[:, 0, 1])


def curl_recovery(pred_fn, landscape: Landscape, points: np.ndarray) -> float:
    """Pearson correlation of recovered vorticity vs the true ``-κ∇²ψ``.

    A robust, inverse-problem-free flux signature: exactly 0 at ``κ=0`` and
    growing with ``κ``.  Complements the vector flux cosine.
    """
    cp = curl(pred_fn, points)
    ct = curl(landscape.drift, points)
    if np.std(ct) < 1e-8:
        return float("nan")  # κ=0: no true vorticity to correlate against
    return float(np.corrcoef(cp, ct)[0, 1])


def drift_recovery(pred_full, true_full, pred_sol=None, true_sol=None,
                   pred_grad=None, true_grad=None) -> dict:
    """Bundle the recovery scores (full + Hodge-resolved grad/sol)."""
    out = {"full_cosine": mean_cosine(pred_full, true_full)}
    if pred_sol is not None:
        out["flux_cosine"] = global_cosine(pred_sol, true_sol)
        out["flux_mag_pred"] = float(np.linalg.norm(np.asarray(pred_sol), axis=1).mean())
        out["flux_mag_true"] = float(np.linalg.norm(np.asarray(true_sol), axis=1).mean())
    if pred_grad is not None:
        out["grad_cosine"] = global_cosine(pred_grad, true_grad)
    return out


# =============================================================================
# Directionality — H1
# =============================================================================
def geodesic_cost(metric, z0, z1, *, n_steps: int = 24, **kw) -> float:
    """Randers arc length (travel-time cost) of the geodesic ``z0 → z1``."""
    traj = exact_geodesic(metric, jnp.asarray(z0, jnp.float32), jnp.asarray(z1, jnp.float32),
                          n_steps=n_steps, **kw)
    return float(metric.arc_length(traj.xs))


def directionality_score(metric, early, late, *, n_steps: int = 24) -> float:
    """``cost(late→early) / cost(early→late)`` — should be ``>1`` (forward cheaper).

    Moving *with* the developmental flow (early→late, downhill) is cheaper than
    against it; the ratio exceeds 1 and increases with the flux ``κ`` (PLAN §8).
    """
    fwd = geodesic_cost(metric, early, late, n_steps=n_steps)
    rev = geodesic_cost(metric, late, early, n_steps=n_steps)
    return rev / (fwd + 1e-9)


# =============================================================================
# Lineage alignment — H1
# =============================================================================
def lineage_alignment(metric, triples_states: np.ndarray, *, n_steps: int = 24) -> float:
    """Mean MSE between geodesic midpoints and observed clonal mid-states.

    ``triples_states`` is ``(M, 3, d)`` of (early, mid, late) latent states.  For
    each, the early→late geodesic's midpoint vertex is compared to the observed
    mid cell (the clone's day-mid state).
    """
    triples_states = np.asarray(triples_states, np.float32)
    errs = []
    for early, mid, late in triples_states:
        traj = exact_geodesic(metric, jnp.asarray(early), jnp.asarray(late), n_steps=n_steps)
        pred_mid = np.asarray(traj.xs[n_steps // 2])
        errs.append(np.sum((pred_mid - mid) ** 2))
    return float(np.mean(errs)) if errs else float("nan")


# =============================================================================
# Fate prediction — H1
# =============================================================================
def fate_by_geodesic(metric, sources: np.ndarray, terminals: np.ndarray,
                     *, n_steps: int = 20, **solve_kw) -> tuple[np.ndarray, np.ndarray]:
    """Predict each source's fate as the nearest terminal by geodesic cost.

    Only the relative cost ordering matters, so callers may pass cheaper solver
    settings (e.g. ``schedule``, ``avbd_iters``, ``gn_iters``) via ``solve_kw``.
    Returns ``(pred_labels, margin)`` where ``margin`` is the cost gap between
    the two terminals (a confidence score for AUROC).
    """
    terminals = np.asarray(terminals, np.float32)
    preds, margins = [], []
    for s in np.asarray(sources, np.float32):
        costs = np.array([geodesic_cost(metric, s, t, n_steps=n_steps, **solve_kw)
                          for t in terminals])
        preds.append(int(np.argmin(costs)))
        margins.append(float(costs.max() - costs.min()))
    return np.array(preds), np.array(margins)


def fate_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred_labels) == np.asarray(true_labels)))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Binary AUROC via the rank statistic (no sklearn dependency)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_pos = ranks[labels == 1].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# =============================================================================
# Collapse frontier — H2
# =============================================================================
def collapse_dstar(dims, cosines, threshold: float = 0.5) -> float:
    """Smallest latent dim where recovery cosine drops below ``threshold``.

    Returns ``inf`` if recovery never collapses over the swept dims (graceful).
    """
    dims, cosines = np.asarray(dims), np.asarray(cosines)
    below = dims[cosines < threshold]
    return float(below.min()) if below.size else float("inf")


# =============================================================================
# LAP ↔ Randers correspondence — H3
# =============================================================================
def path_discrepancy(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Mean symmetric nearest-point distance between two polylines (Hausdorff-ish).

    Endpoint-aligned shape comparison for the Randers geodesic vs the true
    Onsager–Machlup minimum-action path (H3).
    """
    A, B = np.asarray(path_a), np.asarray(path_b)
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    ab = np.sqrt(d2.min(axis=1)).mean()
    ba = np.sqrt(d2.min(axis=0)).mean()
    return float(0.5 * (ab + ba))


# =============================================================================
# Solver diagnostics & cost curves — Stage A / D
# =============================================================================
def solver_diagnostics(metric, z0, z1, *, n_steps: int = 32, **kw) -> dict:
    """Energy, endpoint-aligned displacement, finiteness, and wall-clock."""
    t0 = time.perf_counter()
    traj = exact_geodesic(metric, jnp.asarray(z0, jnp.float32), jnp.asarray(z1, jnp.float32),
                          n_steps=n_steps, **kw)
    traj.xs.block_until_ready()
    wall = time.perf_counter() - t0
    straight = jnp.linspace(jnp.asarray(z0, jnp.float32), jnp.asarray(z1, jnp.float32), n_steps + 1)
    disp = float(jnp.mean(jnp.linalg.norm(traj.xs - straight, axis=1)))
    return {
        "energy": float(traj.energy),
        "arc_length": float(metric.arc_length(traj.xs)),
        "displacement": disp,
        "finite": bool(jnp.isfinite(traj.energy)),
        "wall_s": wall,
        "n_steps": n_steps,
    }
