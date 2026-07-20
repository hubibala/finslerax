"""Synthetic Randers fire scenes with known ground-truth (sea, wind).

Every scene is a ``GridZermelo`` built from smooth random fields with a fixed
seed, so gates can compare recovered fields against exact truth. Fires are
forward eikonal solves from random interior ignitions, optionally truncated
(partial burn) and hour-quantised (satellite realism).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

from .medium import GridZermelo, solve_arrival_grid


def _smooth_field(rng, shape, sigma, lo, hi):
    """Smooth random field in [lo, hi] via Gaussian-filtered white noise."""
    f = gaussian_filter(rng.standard_normal(shape), sigma)
    f = (f - f.min()) / max(f.max() - f.min(), 1e-9)
    return lo + (hi - lo) * f


def make_sea(shape, seed, anisotropy=True):
    """Smooth SPD sea field (3, H, W): eigenvalues in [0.5, 2.5], smooth angle.

    Sea eigenvalues are squared slownesses (h/px)^2 — a fire in this sea
    crosses a 64 px domain in roughly 40–150 h, matching Sim2Real durations.
    """
    rng = np.random.default_rng(seed)
    lam1 = _smooth_field(rng, shape, sigma=10, lo=0.5, hi=2.5)
    if anisotropy:
        lam2 = _smooth_field(rng, shape, sigma=10, lo=0.5, hi=2.5)
        theta = _smooth_field(rng, shape, sigma=14, lo=0.0, hi=np.pi)
    else:
        lam2, theta = lam1, np.zeros(shape)
    c, s = np.cos(theta), np.sin(theta)
    h11 = lam1 * c**2 + lam2 * s**2
    h12 = (lam1 - lam2) * c * s
    h22 = lam1 * s**2 + lam2 * c**2
    return jnp.asarray(np.stack([h11, h12, h22]))


def make_wind(shape, seed, H_grid, rel_mag=0.4, uniform=True):
    """Wind velocity field (2, H, W) with ``||W||_H = rel_mag`` (causal < 1).

    ``uniform=True`` gives a constant direction (the realistic per-fire case);
    otherwise a smooth rotational pattern. The magnitude is normalised
    pointwise in the local sea norm so causality holds everywhere.
    """
    rng = np.random.default_rng(seed)
    if uniform:
        ang = rng.uniform(0, 2 * np.pi)
        w = np.broadcast_to(
            np.array([np.cos(ang), np.sin(ang)])[:, None, None], (2, *shape)
        ).copy()
    else:
        w = np.stack(
            [
                _smooth_field(rng, shape, sigma=12, lo=-1, hi=1),
                _smooth_field(rng, shape, sigma=12, lo=-1, hi=1),
            ]
        )
    h11, h12, h22 = np.asarray(H_grid)
    norm_H = np.sqrt(
        np.maximum(h11 * w[0] ** 2 + 2 * h12 * w[0] * w[1] + h22 * w[1] ** 2, 1e-12)
    )
    return jnp.asarray(w * (rel_mag / norm_H))


@dataclass
class Scene:
    """A synthetic ground-truth scene."""

    H_grid: jnp.ndarray
    W_grid: jnp.ndarray
    shape: tuple[int, int]
    metric: GridZermelo = field(init=False)

    def __post_init__(self):
        self.metric = GridZermelo(self.H_grid, self.W_grid)


def make_scene(shape=(64, 64), seed=0, anisotropy=True, wind_rel_mag=0.4,
               wind_uniform=True) -> Scene:
    H_grid = make_sea(shape, seed, anisotropy=anisotropy)
    if wind_rel_mag > 0:
        W_grid = make_wind(shape, seed + 1, H_grid, wind_rel_mag, wind_uniform)
    else:
        W_grid = jnp.zeros((2, *shape))
    return Scene(H_grid, W_grid, shape)


@dataclass
class Fire:
    """One observed fire: ignition, arrival field (hours), burned mask."""

    ignition_rc: np.ndarray
    T_obs: np.ndarray  # (H, W), inf outside burned
    burned: np.ndarray  # (H, W) bool


def simulate_fires(
    scene: Scene,
    n_fires: int,
    seed: int = 0,
    burn_quantile: float = 0.55,
    quantize_hours: float = 0.0,
) -> list[Fire]:
    """Forward-solve ``n_fires`` from random interior ignitions.

    ``burn_quantile`` truncates each fire at that quantile of its arrival
    times (partial burn — fronts leave the pixel before covering the domain),
    which is also what makes per-fire direction coverage genuinely partial.
    ``quantize_hours > 0`` floors times to that resolution (satellite passes).
    """
    rng = np.random.default_rng(seed)
    H, W = scene.shape
    fires = []
    for _ in range(n_fires):
        ign = np.array(
            [rng.uniform(0.15 * H, 0.85 * H), rng.uniform(0.15 * W, 0.85 * W)]
        )
        T = np.asarray(solve_arrival_grid(scene.metric, jnp.asarray(ign), scene.shape))
        t_end = np.quantile(T[T < 1e4], burn_quantile)
        burned = T <= t_end
        T_obs = np.where(burned, T, np.inf)
        if quantize_hours > 0:
            T_obs = np.where(
                burned, np.floor(T_obs / quantize_hours) * quantize_hours, np.inf
            )
        fires.append(Fire(ign, T_obs, burned))
    return fires
