"""Convex estimators for the segment energy model — per-log and pooled ridge LSQ.

Everything here is a linear least-squares solve on the features of
``medium.py``: no seeds, no iterations, float64 numpy (there is no
differentiable structure to justify JAX — the science is in the feature
parity, not the optimizer). Columns are standardized to unit RMS internally so
one ridge parameter is meaningful across features of wildly different scales;
coefficients are returned in raw physical units.

* :func:`fit_log` / :func:`fit_per_log` — the Stage A/B workhorse: per-log
  ``[even | odd]`` ridge fit.
* :func:`fit_pooled` — hierarchical fit: per-log even nuisance (each flight
  owns its hover/drag), shared odd drift across the fleet. The per-log blocks
  are eliminated by Schur complement, so fleets of thousands of logs stay a
  small dense solve.
* :func:`negative_control` — the falsification arm: swap the ledger ``dz`` for
  a fake east-west potential ``dx`` and check the recovered coefficient ≈ 0.
* :func:`implied_wind` — converts the fitted odd wind form back to m/s through
  the fitted drag coefficient (``W = −b/(2ĉ₂)``, exact for quadratic drag).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from typing import Callable

import numpy as np
import pandas as pd

from .medium import EVEN_NAMES, even_features, odd_features, rbf_wind_features


# ---------------------------------------------------------------------------
# Scaled ridge solve
# ---------------------------------------------------------------------------
def _column_scales(X: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.mean(X**2, axis=0))
    s[s == 0.0] = 1.0
    return s


def _ridge_solve(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge LSQ with unit-RMS column standardization; ``ridge=0`` → plain lstsq."""
    s = _column_scales(X)
    Z = X / s
    if ridge == 0.0:
        w = np.linalg.lstsq(Z, y, rcond=None)[0]
    else:
        n, p = Z.shape
        w = np.linalg.solve(Z.T @ Z + ridge * n * np.eye(p), Z.T @ y)
    return w / s


# ---------------------------------------------------------------------------
# Per-log fits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LogFit:
    """Per-log coefficients of ``E ≈ even·c + odd·θ`` (raw units)."""

    log_id: str
    even: np.ndarray            # (4,) coefficients of EVEN_NAMES
    odd: np.ndarray             # (len(odd_names),)
    odd_names: tuple
    n_segments: int
    resid_rms: float            # in-sample residual RMS [J]

    @property
    def k(self) -> float:
        """The ledger k̂ [J/m] (nan if 'dz' was not an odd channel)."""
        return float(self.odd[self.odd_names.index("dz")]) if "dz" in self.odd_names else float("nan")

    @property
    def wind_form(self) -> np.ndarray | None:
        """The horizontal odd form b̂ [J/m] if the wind channel was fitted."""
        if "dx" in self.odd_names and "dy" in self.odd_names:
            return np.array([self.odd[self.odd_names.index("dx")],
                             self.odd[self.odd_names.index("dy")]])
        return None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return even_features(df) @ self.even + odd_features(df, self.odd_names) @ self.odd


def fit_log(
    df: pd.DataFrame,
    *,
    odd: tuple = ("dz", "dx", "dy"),
    ridge: float = 1e-6,
) -> LogFit:
    """Ridge LSQ of ``[even | odd]`` on the segment energies of one log."""
    Xe, Xo = even_features(df), odd_features(df, odd)
    X = np.hstack([Xe, Xo])
    y = df["energy"].to_numpy(float)
    coef = _ridge_solve(X, y, ridge)
    resid = y - X @ coef
    log_id = str(df["log_id"].iloc[0]) if len(df) else "?"
    return LogFit(
        log_id=log_id,
        even=coef[: Xe.shape[1]],
        odd=coef[Xe.shape[1]:],
        odd_names=tuple(odd),
        n_segments=len(df),
        resid_rms=float(np.sqrt(np.mean(resid**2))) if len(df) else float("nan"),
    )


def fit_per_log(df: pd.DataFrame, **kwargs) -> dict:
    """:func:`fit_log` per ``log_id`` → ``{log_id: LogFit}``."""
    return {str(lid): fit_log(sub, **kwargs) for lid, sub in df.groupby("log_id", sort=True)}


def k_series(fits: dict) -> pd.Series:
    """Per-log ledger estimates as a Series (for consistency stats)."""
    return pd.Series({lid: f.k for lid, f in fits.items()}, name="k")


def fleet_predictor(fits: dict) -> Callable[[pd.DataFrame], np.ndarray]:
    """A ``predict(df)`` over a table spanning many logs, from per-log fits."""

    def predict(df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df))
        for lid, sub in df.groupby("log_id", sort=False):
            out[df.index.get_indexer(sub.index)] = fits[str(lid)].predict(sub)
        return out

    return predict


def negative_control(df: pd.DataFrame, *, ridge: float = 1e-6) -> pd.Series:
    """Per-log coefficient of the fake east-west potential ``k'·dx`` (should be ≈ 0)."""
    fits = fit_per_log(df, odd=("dx",), ridge=ridge)
    return pd.Series({lid: float(f.odd[0]) for lid, f in fits.items()}, name="k_fake")


# ---------------------------------------------------------------------------
# Pooled (hierarchical) fit — per-log even nuisance, shared odd drift
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpatialWind:
    """Horizontal odd form ``b(x) = Σ w_g·∇φ + w_r·⋆dφ`` from fitted RBF atoms [J/m]."""

    centers: np.ndarray
    width: float
    w_grad: np.ndarray
    w_rot: np.ndarray

    def __call__(self, xy: np.ndarray) -> np.ndarray:
        d = np.asarray(xy, float) - self.centers  # (K, 2)
        phi = np.exp(-(d**2).sum(-1) / (2.0 * self.width**2))
        g = -d / self.width**2 * phi[:, None]  # (K, 2) rows ∇φ_j
        r = np.stack([-g[:, 1], g[:, 0]], axis=-1)  # ⋆dφ_j
        return self.w_grad @ g + self.w_rot @ r


@dataclass(frozen=True)
class FleetFit:
    """Pooled fit: per-log block (even, + dz unless shared) and fleet-shared odd."""

    per_log: dict               # {log_id: (p_perlog,) coefficients}
    per_log_names: tuple
    shared: np.ndarray          # (p_shared,)
    shared_names: tuple
    spatial: SpatialWind | None
    _builders: tuple = dataclasses_field(repr=False, default=())  # (per-log, shared) df -> matrix

    @property
    def k_by_log(self) -> pd.Series | None:
        if "dz" in self.per_log_names:
            j = self.per_log_names.index("dz")
            return pd.Series({lid: c[j] for lid, c in self.per_log.items()}, name="k")
        return None

    @property
    def k(self) -> float:
        """Shared ledger (nan when k is per-log — use ``k_by_log`` then)."""
        return float(self.shared[self.shared_names.index("dz")]) if "dz" in self.shared_names else float("nan")

    @property
    def wind_form(self) -> np.ndarray | None:
        if "dx" in self.shared_names and "dy" in self.shared_names:
            return np.array([self.shared[self.shared_names.index("dx")],
                             self.shared[self.shared_names.index("dy")]])
        return None

    @property
    def c2_by_log(self) -> pd.Series:
        j = self.per_log_names.index("length2_over_dt")
        return pd.Series({lid: c[j] for lid, c in self.per_log.items()}, name="c2")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        build_p, build_s = self._builders
        out = np.empty(len(df))
        for lid, sub in df.groupby("log_id", sort=False):
            pred = build_p(sub) @ self.per_log[str(lid)] + build_s(sub) @ self.shared
            out[df.index.get_indexer(sub.index)] = pred
        return out


def fit_pooled(
    df: pd.DataFrame,
    *,
    share_k: bool = False,
    wind: str | None = "uniform",
    centers: np.ndarray | None = None,
    width: float | None = None,
    ridge: float = 1e-6,
) -> FleetFit:
    """Hierarchical ridge LSQ: per-log even nuisance, fleet-shared odd drift.

    ``share_k=False`` keeps the ledger per-log (mass/payload varies per flight)
    and shares only the wind; ``share_k=True`` pools the ledger too. ``wind``
    is ``"uniform"`` (dx, dy columns), ``"spatial"`` (RBF atoms at ``centers``
    with ``width``; needs mid_x/mid_y), or ``None``. The per-log blocks are
    eliminated with a Schur complement, so the dense solve is only
    ``p_shared × p_shared``.
    """
    perlog_names = EVEN_NAMES + (() if share_k else ("dz",))
    odd_shared: tuple = ("dz",) if share_k else ()
    if wind == "uniform":
        odd_shared = (*odd_shared, "dx", "dy")
    elif wind == "spatial":
        if centers is None or width is None:
            raise ValueError("wind='spatial' needs centers and width")
    elif wind is not None:
        raise ValueError(f"unknown wind mode {wind!r}")
    if not odd_shared and wind != "spatial":
        raise ValueError("nothing shared — use fit_per_log instead")

    def build_perlog(sub: pd.DataFrame) -> np.ndarray:
        Xe = even_features(sub)
        return Xe if share_k else np.hstack([Xe, odd_features(sub, ("dz",))])

    def build_shared(sub: pd.DataFrame) -> np.ndarray:
        parts = []
        if odd_shared:
            parts.append(odd_features(sub, odd_shared))
        if wind == "spatial":
            parts.append(rbf_wind_features(sub, centers, width)[0])
        return np.hstack(parts)

    shared_names = odd_shared
    if wind == "spatial":
        shared_names = shared_names + rbf_wind_features(df.iloc[:1], centers, width)[1]

    # Global unit-RMS scales so per-log and shared blocks share one ridge.
    sp = _column_scales(build_perlog(df))
    ss = _column_scales(build_shared(df))
    n_total = len(df)
    pp, ps = len(sp), len(ss)

    S = np.zeros((ps, ps))
    r = np.zeros(ps)
    cache = {}
    for lid, sub in df.groupby("log_id", sort=True):
        A = build_perlog(sub) / sp
        B = build_shared(sub) / ss
        y = sub["energy"].to_numpy(float)
        G = A.T @ A + ridge * len(sub) * np.eye(pp)
        Gi_At = np.linalg.solve(G, A.T)
        S += B.T @ B - (B.T @ A) @ Gi_At @ B
        r += B.T @ y - (B.T @ A) @ Gi_At @ y
        cache[str(lid)] = (Gi_At, B, y)

    theta = np.linalg.solve(S + ridge * n_total * np.eye(ps), r)
    per_log = {
        lid: (Gi_At @ (y - B @ theta)) / sp
        for lid, (Gi_At, B, y) in cache.items()
    }
    theta = theta / ss

    spatial = None
    if wind == "spatial":
        K = centers.shape[0]
        w_wind = theta[-2 * K:]
        spatial = SpatialWind(centers=np.asarray(centers, float), width=float(width),
                              w_grad=w_wind[:K], w_rot=w_wind[K:])

    return FleetFit(
        per_log=per_log,
        per_log_names=perlog_names,
        shared=theta,
        shared_names=shared_names,
        spatial=spatial,
        _builders=(build_perlog, build_shared),
    )


# ---------------------------------------------------------------------------
# Wind in physical units
# ---------------------------------------------------------------------------
def implied_wind(fit) -> np.ndarray:
    """Recovered wind vector ``Ŵ = −b̂/(2ĉ₂)`` [m/s] from a LogFit or FleetFit.

    Exact for quadratic drag; the fitted nuisance coefficient ĉ₂ supplies the
    unit conversion, which is the one place the even bracket is (mildly)
    interpreted.
    """
    b = fit.wind_form
    if b is None:
        raise ValueError("fit has no uniform wind channel")
    if isinstance(fit, FleetFit):
        c2 = float(fit.c2_by_log.median())
    else:
        c2 = float(fit.even[EVEN_NAMES.index("length2_over_dt")])
    return -b / (2.0 * c2)


def implied_wind_field(fit: FleetFit) -> Callable[[np.ndarray], np.ndarray]:
    """Spatial version: ``Ŵ(x) = −b̂(x)/(2ĉ₂)`` from a spatial pooled fit."""
    if fit.spatial is None:
        raise ValueError("fit has no spatial wind channel")
    c2 = float(fit.c2_by_log.median())
    return lambda xy: -fit.spatial(xy) / (2.0 * c2)
