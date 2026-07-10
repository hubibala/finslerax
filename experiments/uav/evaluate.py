"""Evaluation — held-out directed-energy skill, ledger consistency, wind cross-check.

The headline metric is :func:`directed_energy_r2`: on held-out segments, match
each climb to the most similar descent (same log, nearest ``(length, dt)``) and
score the predicted energy *difference* of each pair. Matching makes the even
bracket nearly cancel, so the score is dominated by the odd channel the gauge
claim is about — a model with the right nuisance but no ledger keeps only the
residual even-bracket correlation that imperfect matching leaves behind, far
below the full model (Stage B shows the gap).

The wind cross-check compares the recovered wind (via ``estimate.implied_wind``)
against the onboard EKF estimate — an instrument the fit never saw.
:func:`direction_balance` and :func:`even_odd_leakage` are the U3 diagnostics:
how far a segment set is from the direction-balanced ideal where even/odd
orthogonality is exact.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .medium import even_features, odd_features


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination (centered)."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    return 1.0 - ss_res / ss_tot


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Global cosine similarity of two (stacks of) vectors, Frobenius-normalized."""
    u, v = np.asarray(u, float).ravel(), np.asarray(v, float).ravel()
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))


def split_segments(
    df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-log stratified random segment split → (train, test).

    Every log keeps at least one train segment (a log split entirely into test
    would have no fit to predict it with), so 1-segment logs stay in train.
    """
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(df), dtype=bool)
    for _, sub in df.groupby("log_id", sort=True):
        idx = df.index.get_indexer(sub.index)
        n_test = min(max(1, round(test_frac * len(idx))), len(idx) - 1)
        if n_test > 0:
            test_mask[rng.choice(idx, size=n_test, replace=False)] = True
    return df[~test_mask], df[test_mask]


# ---------------------------------------------------------------------------
# Directed-energy skill (the Stage B success metric)
# ---------------------------------------------------------------------------
def _match_pairs(sub: pd.DataFrame, dz_min: float) -> list:
    """Greedy climb↔descent matching by nearest ``(length, dt, |dz|)`` within one log.

    ``|dz|`` is a match feature so the *even* vertical overhead cancels within
    a pair too — what remains is odd almost by construction.
    """
    up = sub[sub["dz"] > dz_min]
    dn = sub[sub["dz"] < -dz_min]
    if len(up) == 0 or len(dn) == 0:
        return []

    def feats(t: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [t["length"].to_numpy(float), t["dt"].to_numpy(float),
             np.abs(t["dz"].to_numpy(float))]
        )

    scale = feats(sub).std(axis=0) + 1e-9
    fu = feats(up) / scale
    fd = feats(dn) / scale
    used = np.zeros(len(dn), dtype=bool)
    pairs = []
    for i in range(min(len(up), len(dn))):
        d2 = ((fu[i][None, :] - fd) ** 2).sum(axis=1)
        d2[used] = np.inf
        j = int(np.argmin(d2))
        used[j] = True
        pairs.append((up.index[i], dn.index[j]))
    return pairs


def directed_energy_r2(
    df: pd.DataFrame,
    predict: Callable[[pd.DataFrame], np.ndarray],
    dz_min: float = 1.0,
) -> tuple[float, int]:
    """R² of predicted vs observed energy differences over matched climb/descent pairs.

    Returns ``(r2, n_pairs)``. Matching (per log, nearest length/dt) makes the
    even bracket nearly cancel within each pair, so the observed differences are
    dominated by ``2k·|dz|`` plus the wind term — the odd channel.
    """
    pred_all = pd.Series(np.asarray(predict(df), float), index=df.index)
    obs_diff, pred_diff = [], []
    for _, sub in df.groupby("log_id", sort=True):
        for i_up, i_dn in _match_pairs(sub, dz_min):
            obs_diff.append(df.at[i_up, "energy"] - df.at[i_dn, "energy"])
            pred_diff.append(pred_all[i_up] - pred_all[i_dn])
    if len(obs_diff) < 3:
        return float("nan"), len(obs_diff)
    return r2(np.array(obs_diff), np.array(pred_diff)), len(obs_diff)


# ---------------------------------------------------------------------------
# Ledger consistency and wind cross-check
# ---------------------------------------------------------------------------
def k_consistency(k_by_log: pd.Series) -> float:
    """Within-airframe coefficient of variation of the per-log ledger, std/mean."""
    return float(k_by_log.std(ddof=1) / k_by_log.mean())


def fleet_wind_cosine(w_hat: np.ndarray, df: pd.DataFrame) -> float:
    """Cosine between a recovered uniform wind and the fleet-mean EKF wind."""
    ekf = df[["wind_x", "wind_y"]].dropna().to_numpy(float)
    return cosine(np.asarray(w_hat, float), ekf.mean(axis=0))


def wind_field_cosine(wind_fn: Callable[[np.ndarray], np.ndarray], df: pd.DataFrame) -> float:
    """Global cosine between a recovered wind *field* (at segment midpoints) and EKF wind."""
    sub = df.dropna(subset=["wind_x", "wind_y"])
    mids = sub[["mid_x", "mid_y"]].to_numpy(float)
    pred = np.stack([wind_fn(m) for m in mids])
    return cosine(pred, sub[["wind_x", "wind_y"]].to_numpy(float))


# ---------------------------------------------------------------------------
# Stage-A pilot metrics — the crudest possible ledger read, a-priori of any fit
# ---------------------------------------------------------------------------
def _updown(df: pd.DataFrame, rate_min: float):
    """Split the *vertical-dominant* segments into climb / descent bins.

    Only segments where vertical speed exceeds horizontal (the climb-out and
    descent phases) enter the bins — a cruise leg with a slight altitude drift
    carries the horizontal drag term, which would otherwise leak into the crude
    ledger read and inflate its variance. Returns ``(vz, specific power, up, dn)``.
    """
    dt = df["dt"].to_numpy(float)
    vz = df["dz"].to_numpy(float) / dt
    vh = np.sqrt(df["dx"].to_numpy(float) ** 2 + df["dy"].to_numpy(float) ** 2) / dt
    sp = df["energy"].to_numpy(float) / dt
    vertical = np.abs(vz) > vh
    return vz, sp, (vz > rate_min) & vertical, (vz < -rate_min) & vertical


def two_bin_k(df: pd.DataFrame, rate_min: float = 0.3) -> float:
    """Crudest ledger estimate [J/m]: up/down specific-power gap over climb-rate gap.

    ``(⟨E/dt⟩_climb − ⟨E/dt⟩_descent) / (⟨vz⟩_climb − ⟨vz⟩_descent)`` — the Stage-A
    two-bin read (§4 U-A2). Biased by the symmetric |dz| overhead when climb/
    descent rates are lopsided, but monotone in the true ``k`` and seedless; its
    within-airframe CV is the pilot's consistency gate.
    """
    vz, sp, up, dn = _updown(df, rate_min)
    if up.sum() < 2 or dn.sum() < 2:
        return float("nan")
    return float((sp[up].mean() - sp[dn].mean()) / (vz[up].mean() - vz[dn].mean()))


def ledger_conditioned(
    df: pd.DataFrame, *, min_dz_std: float = 1.5, min_updown: int = 3, rate_min: float = 1.0
) -> pd.Series:
    """Per-log boolean: does the flight climb/descend enough to *identify* ``k``?

    A flight flown at constant altitude has no vertical dynamic range, so its
    ledger coefficient is unidentifiable and the per-log fit blows up (± hundreds
    of J/m on real data). This gate — a minimum altitude spread and enough
    climb *and* descent segments above ``rate_min`` — is the honest precondition
    for reading ``k`` from a log, not a fit-quality cherry-pick: it is decidable
    before the fit, from the geometry alone.
    """
    out = {}
    for lid, g in df.groupby("log_id", sort=True):
        vz = (g["dz"].to_numpy(float) / g["dt"].to_numpy(float))
        out[str(lid)] = bool(
            float(g["dz"].std()) > min_dz_std
            and (vz > rate_min).sum() >= min_updown
            and (vz < -rate_min).sum() >= min_updown
        )
    return pd.Series(out, name="conditioned")


def climb_costs_more(df: pd.DataFrame, rate_min: float = 0.3) -> bool | None:
    """True if climb specific-power exceeds descent (the physically required sign).

    The Stage-A go/no-go signal (§4 U-A1): the up/down asymmetry must have the
    correct sign in ≥ 90% of logs or the energy channel is too dirty to proceed.
    ``None`` when a log lacks enough climb/descent segments to judge.
    """
    _, sp, up, dn = _updown(df, rate_min)
    if up.sum() < 2 or dn.sum() < 2:
        return None
    return bool(sp[up].mean() > sp[dn].mean())


# ---------------------------------------------------------------------------
# U3 diagnostics — how direction-balanced is the segment set?
# ---------------------------------------------------------------------------
def direction_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Per-log displacement imbalance ``‖Σ Δ‖ / Σ ‖Δ‖`` ∈ [0, 1] per channel.

    Columns ``vertical`` (the ledger's channel — near 0 for any flight that
    lands where it took off in altitude, i.e. every flight) and ``horizontal``
    (flags one-way ferry missions). 0 = perfectly balanced, 1 = one-way.
    """
    out = {}
    for lid, sub in df.groupby("log_id", sort=True):
        dz = sub["dz"].to_numpy(float)
        dxy = sub[["dx", "dy"]].to_numpy(float)
        out[str(lid)] = {
            "vertical": float(abs(dz.sum()) / (np.abs(dz).sum() + 1e-30)),
            "horizontal": float(
                np.linalg.norm(dxy.sum(axis=0))
                / (np.linalg.norm(dxy, axis=1).sum() + 1e-30)
            ),
        }
    return pd.DataFrame.from_dict(out, orient="index")


def even_odd_leakage(df: pd.DataFrame) -> float:
    """Max |correlation| between any even and any odd feature column.

    Exactly zero on a reversal-balanced set (parity); on real fleets this
    quantifies the residual coupling the ridge fit has to disentangle.
    """
    E = even_features(df)
    O = odd_features(df)
    E = E - E.mean(axis=0)
    O = O - O.mean(axis=0)
    denom = np.outer(np.linalg.norm(E, axis=0), np.linalg.norm(O, axis=0)) + 1e-30
    return float(np.max(np.abs(E.T @ O) / denom))
