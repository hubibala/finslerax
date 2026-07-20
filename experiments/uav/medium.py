"""The UAV segment cost model — an even nuisance bracket plus an odd drift 1-form.

A multirotor's segment energy decomposes by parity under segment *reversal*
(``Δx → −Δx`` at fixed duration, length and speed profile):

    E_seg ≈ [c₀·Δt + c₁·len + c₂·len²/Δt + s·|Δz|] + [k·Δz + b·Δxy]
            └────────────── even (nuisance) ───────┘  └─ odd: the 1-form β ─┘

* The even bracket absorbs hover, drag, the speed profile, and the *symmetric*
  half of the vertical cost (``|Δz|``: climbing and descending both cost extra
  over cruise). It is a nuisance model — fitted, never interpreted.
* The odd part is the drift 1-form. ``k·dz`` is the exact (gravity) ledger,
  ``k = mg(1/η↑ + η↓)/2``; ``b·dxy`` is the horizontal wind coupling. For
  quadratic drag the wind coupling is ``b = −2c₂·W`` *exactly* (the ``|W|²``
  remainder is even and lands in ``c₀``), so the fitted nuisance coefficient
  ``c₂`` converts the recovered form back into a wind vector in m/s.

Even and odd features are L²-orthogonal over any direction-balanced segment
set — that is what keeps the nuisance from leaking into the ledger, and
``evaluate.direction_balance`` quantifies how far a real segment set is from
balanced.

Conventions (fixed at ingest, asserted by :func:`validate_table`): SI units
(m, s, J); z-up, so ``dz > 0`` is a climb; ``length`` is the densified 3-D path
length (≥ chord); ``energy = ∫V·I dt``. One table row = one segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Columns every segment table must carry (ingest and the synthetic generator
#: both produce them). Extra columns (mid_x/y/z, wind_x/y, quality flags) ride
#: along untouched.
REQUIRED_COLUMNS = (
    "log_id",     # flight/log identifier
    "airframe",   # airframe class label
    "t0",         # segment start time within the log [s]
    "dt",         # duration [s]
    "dx", "dy", "dz",  # displacement, z-up [m]
    "length",     # densified 3-D path length [m]
    "energy",     # ∫V·I dt over the segment [J]
)

EVEN_NAMES = ("dt", "length", "length2_over_dt", "abs_dz")
ODD_COLUMNS = ("dz", "dx", "dy")  # ledger + horizontal wind channels


def validate_table(df: pd.DataFrame) -> None:
    """Assert the segment-table contract (schema, units sanity, length ≥ chord)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"segment table missing columns: {missing}")
    num = df[["dt", "dx", "dy", "dz", "length", "energy"]].to_numpy(float)
    if not np.all(np.isfinite(num)):
        raise ValueError("segment table has non-finite dt/displacement/length/energy")
    if np.any(df["dt"].to_numpy(float) <= 0):
        raise ValueError("segment table has non-positive durations")
    chord = np.sqrt((df[["dx", "dy", "dz"]].to_numpy(float) ** 2).sum(axis=1))
    if np.any(df["length"].to_numpy(float) < chord - 1e-6):
        raise ValueError("segment length shorter than its chord (broken densification)")


def even_features(df: pd.DataFrame) -> np.ndarray:
    """Even (reversal-invariant) nuisance features, shape ``(N, 4)`` = EVEN_NAMES."""
    dt = df["dt"].to_numpy(float)
    ln = df["length"].to_numpy(float)
    dz = df["dz"].to_numpy(float)
    return np.column_stack([dt, ln, ln**2 / dt, np.abs(dz)])


def odd_features(df: pd.DataFrame, channels: tuple = ("dz", "dx", "dy")) -> np.ndarray:
    """Odd (reversal-negating) drift features: displacement columns, shape ``(N, len(channels))``.

    ``("dz",)`` is the ledger alone; ``("dz", "dx", "dy")`` adds the uniform
    horizontal wind channel; ``("dx",)`` is the east-west fake potential used
    as the negative control.
    """
    bad = [c for c in channels if c not in ODD_COLUMNS]
    if bad:
        raise ValueError(f"unknown odd channels {bad}; choose from {ODD_COLUMNS}")
    return df[list(channels)].to_numpy(float)


def reverse_segments(df: pd.DataFrame) -> pd.DataFrame:
    """The feature-space reversal involution: flip displacements, keep dt/length/energy.

    This is a *modeling* operation (it defines the even/odd parity the model is
    built on), not a data transformation — a real reversed flight would have its
    own energy. Used to build direction-balanced sets and to test parity.
    """
    out = df.copy()
    out[["dx", "dy", "dz"]] = -out[["dx", "dy", "dz"]]
    return out


def rbf_wind_features(
    df: pd.DataFrame, centers: np.ndarray, width: float
) -> tuple[np.ndarray, tuple]:
    """Spatial wind atoms ``[∇φ_j(mid)·Δxy | ⋆dφ_j(mid)·Δxy]``, shape ``(N, 2K)``.

    The ``recover_form_lsq`` pattern from ``experiments/arm/learn.py`` with RBF
    atoms swapped for flight features: gradient atoms carry the exact (curl-free)
    part of the horizontal drift, rotated atoms the co-exact (divergence-free)
    part. Requires ``mid_x``/``mid_y`` columns.
    """
    mid = df[["mid_x", "mid_y"]].to_numpy(float)  # (N, 2)
    dxy = df[["dx", "dy"]].to_numpy(float)  # (N, 2)
    d = mid[:, None, :] - centers[None, :, :]  # (N, K, 2)
    phi = np.exp(-(d**2).sum(-1) / (2.0 * width**2))  # (N, K)
    g = -d / width**2 * phi[..., None]  # (N, K, 2) rows ∇φ_j(mid)
    gdot = (g * dxy[:, None, :]).sum(-1)  # (N, K)
    rdot = -g[..., 1] * dxy[:, [0]] + g[..., 0] * dxy[:, [1]]  # ⋆dφ_j · Δxy
    K = centers.shape[0]
    names = tuple(f"grad_{j}" for j in range(K)) + tuple(f"rot_{j}" for j in range(K))
    return np.hstack([gdot, rdot]), names
