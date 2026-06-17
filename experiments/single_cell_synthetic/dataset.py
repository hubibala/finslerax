"""The ``SingleCellDataset`` interface — the contract Stages B–D consume.

This is deliberately defined *first* (PLAN §12 "dataset interface lock"): the
synthetic :mod:`generator` emits one of these, and a future
``research/weinreb`` loader is meant to satisfy the **same** contract verbatim,
so the inference + evaluation code (`metric`, `drift`, `solvers`, `baselines`,
`evaluate`) transfers to real data with zero rewrite.

Two tiers of fields:

* **Observable** (present on real scRNA-seq too): integer ``X_counts``, the
  PCA/embedding ``X_pca``, projected ``velocity_pca``, ``clone_id``,
  ``time_point``, and (derived) lineage ``triples``.
* **Oracle** (synthetic-only ground truth, ``None`` on real data): ``true_state``,
  ``true_drift`` and its Hodge split, ``fate_label``, ``branch``, the decoder and
  its Jacobian.  Every evaluation that needs ground truth guards on these being
  present, so the *same* :mod:`evaluate` functions run on real data (skipping the
  oracle-only metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class SingleCellDataset:
    """A destructively-sampled single-cell developmental snapshot dataset.

    Shapes (``n`` = total cells, ``D_gene`` = gene dim, ``d`` = embedding dim):

    Observable
        X_counts:    ``(n, D_gene)`` int   — negative-binomial UMI counts.
        X_pca:       ``(n, d)``      float — PCA / latent embedding.
        velocity_pca:``(n, d)``      float — noisy RNA velocity in embedding space.
        clone_id:    ``(n,)``        int   — lineage barcode (``-1`` = unlabelled).
        time_point:  ``(n,)``        int   — destructive snapshot index (0..T-1).

    Oracle (synthetic only; ``None`` on real loaders)
        true_state:  ``(n, d_true)`` float — latent developmental state ``x``.
        true_drift:  ``(n, d_true)`` float — ``f(x)`` at the cell.
        true_grad / true_sol: ``(n, d_true)`` — exact Hodge split of ``true_drift``.
        fate_label:  ``(n,)``        int   — terminal valley (0/1), via the clone.
        branch:      ``(n,)``        int   — pre/post-bifurcation branch id.

    Maps
        decoder:     ``x -> rates`` (``ℝ^{d_true} -> ℝ^{D_gene}``), the true map.
        pca_mean / pca_components: the fitted PCA transform of ``log1p(norm)``
            counts, so a learned latent can be compared in a common frame.
    """

    # --- observable ---
    X_counts: np.ndarray
    X_pca: np.ndarray
    velocity_pca: np.ndarray
    clone_id: np.ndarray
    time_point: np.ndarray

    # --- lineage supervision ---
    triples: np.ndarray | None = None  # (M, 3) indices (early, mid, late)

    # --- oracle (synthetic) ---
    true_state: np.ndarray | None = None
    true_drift: np.ndarray | None = None
    true_grad: np.ndarray | None = None
    true_sol: np.ndarray | None = None
    fate_label: np.ndarray | None = None
    branch: np.ndarray | None = None
    # noisy velocity in the *true latent* frame (oracle); use for the
    # metric-isolation stages (A/C/D) where ``velocity_pca``'s embedding-frame
    # rotation would otherwise confound the drift/flux comparison against
    # ``true_drift``.  ``velocity_pca`` remains the realistic embedding-study input.
    velocity_true: np.ndarray | None = None

    # --- maps / transforms ---
    decoder: Callable[[np.ndarray], np.ndarray] | None = None
    pca_mean: np.ndarray | None = None
    pca_components: np.ndarray | None = None

    # --- free-form provenance (kappa, seed, noise level, ...) ---
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return int(self.X_counts.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.X_counts.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.X_pca.shape[1])

    @property
    def has_oracle(self) -> bool:
        """Whether synthetic ground truth is attached (gates oracle metrics)."""
        return self.true_state is not None

    def at_time(self, t: int) -> np.ndarray:
        """Boolean mask of cells observed at snapshot ``t``."""
        return self.time_point == t

    def clones(self) -> dict[int, np.ndarray]:
        """Map ``clone_id -> cell indices`` (excludes the ``-1`` unlabelled pool)."""
        out: dict[int, np.ndarray] = {}
        cid = self.clone_id
        for c in np.unique(cid):
            if c < 0:
                continue
            out[int(c)] = np.nonzero(cid == c)[0]
        return out

    def build_triples(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Derive ``(early, mid, late)`` lineage triples from clones across time.

        For each clone observed at ≥3 distinct timepoints, sample one cell from
        the earliest, a middle, and the latest snapshot.  These are the
        cross-time supervision and the fate ground truth (a clone's terminal
        valley is its fate; PLAN §5.3).  Caches into ``self.triples``.
        """
        rng = rng or np.random.default_rng(0)
        triples = []
        for _, idx in self.clones().items():
            tps = self.time_point[idx]
            uniq = np.unique(tps)
            if uniq.size < 3:
                continue
            t_lo, t_hi = uniq[0], uniq[-1]
            t_mid = uniq[uniq.size // 2]
            early = rng.choice(idx[tps == t_lo])
            mid = rng.choice(idx[tps == t_mid])
            late = rng.choice(idx[tps == t_hi])
            triples.append((early, mid, late))
        self.triples = np.array(triples, dtype=np.int64) if triples else np.empty((0, 3), np.int64)
        return self.triples
