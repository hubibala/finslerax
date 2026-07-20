"""Inverse fire geometry: free-field (sea, wind) recovery and the (s, c) gauge refit.

Two estimators live here:

* :func:`fit_free_field` — the Gahtan-style inverse problem: optimise per-pixel
  ``(H_grid, W_grid)`` through the differentiable eikonal solver against one or
  more observed fires. Multi-fire is the multi-source identifiability
  mechanism; single-fire is the confounded negative control (Stage W-A gate).
* :func:`recalibrate` — the few-shot scene gauge refit (UAV Stage-D recipe):
  freeze everything, refit only a global time scale ``s`` (closed-form given
  the wind multiplier) and a wind multiplier ``c`` (1-D grid + refine), from
  the first K hours of a new fire.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ham.geometry.metric import AsymmetricMetric

from .medium import GridZermelo, dense_arrival_loss, solve_arrival_grid, tv_reg

# ---------------------------------------------------------------------------
# Free-field recovery
# ---------------------------------------------------------------------------


class FreeField(eqx.Module):
    """Trainable per-pixel Zermelo fields with causality projection."""

    H_grid: jax.Array  # (3, H, W)
    W_grid: jax.Array  # (2, H, W)

    def __init__(self, shape: tuple[int, int]):
        H, W = shape
        self.H_grid = jnp.stack(
            [jnp.ones((H, W)), jnp.zeros((H, W)), jnp.ones((H, W))]
        )
        self.W_grid = jnp.zeros((2, H, W))

    def project(self) -> FreeField:
        """Clamp sea eigen-range, enforce det > 0 and causal ``||W||_H < 0.9``."""
        h11 = jnp.clip(self.H_grid[0], 0.1, 10.0)
        h22 = jnp.clip(self.H_grid[2], 0.1, 10.0)
        h12 = jnp.clip(self.H_grid[1], -jnp.sqrt(h11 * h22) * 0.95,
                       jnp.sqrt(h11 * h22) * 0.95)
        w = self.W_grid
        norm_sq = w[0] * (h11 * w[0] + h12 * w[1]) + w[1] * (h12 * w[0] + h22 * w[1])
        scale = jnp.where(
            norm_sq > 0.81, 0.9 / jnp.maximum(jnp.sqrt(norm_sq), 1e-8), 1.0
        )
        return eqx.tree_at(
            lambda m: (m.H_grid, m.W_grid),
            self,
            (jnp.stack([h11, h12, h22]), w * scale),
        )


@dataclass
class FitResult:
    model: FreeField
    losses: list

    @property
    def metric(self) -> GridZermelo:
        return GridZermelo(self.model.H_grid, self.model.W_grid)


def fit_free_field(
    fires,
    shape: tuple[int, int],
    recover_wind: bool = True,
    lam_tv_H: float = 3e-4,
    lam_tv_W: float = 3e-4,
    coverage_weight: float = 0.0,
    coverage: np.ndarray | None = None,
    n_iter: int = 300,
    lr: float = 0.05,
    alpha: float = 1.0,
    verbose: bool = False,
) -> FitResult:
    """Recover (H_grid, W_grid) from observed fires through the eikonal solver.

    Args:
        fires: list of :class:`~experiments.wildfire.synthetic.Fire`.
        recover_wind: If False the wind stays frozen at zero (even-only fit).
        lam_tv_H / lam_tv_W: Gahtan-exact TV strengths on the two fields.
        coverage_weight: If > 0, parity-aware shrinkage — penalise
            ``(1 - coverage) * |W|^2`` so wind is pulled to zero exactly where
            no odd information exists, instead of being TV-smoothed sideways.
        coverage: (H, W) direction-coverage mask in [0, 1] (required if
            ``coverage_weight > 0``).
        alpha: dense-loss gauge blend (1 = absolute, 0 = affine).
    """
    model = FreeField(shape)
    igns = jnp.stack([jnp.asarray(f.ignition_rc) for f in fires])
    T_obs = jnp.stack([jnp.asarray(f.T_obs) for f in fires])
    burned = jnp.stack([jnp.asarray(f.burned) for f in fires])
    cov = (
        jnp.asarray(coverage) if coverage is not None else jnp.ones(shape)
    )

    filter_spec = jax.tree_util.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda m: m.H_grid, filter_spec, True)
    filter_spec = eqx.tree_at(lambda m: m.W_grid, filter_spec, recover_wind)

    optimizer = optax.adam(
        optax.exponential_decay(lr, transition_steps=400, decay_rate=0.9)
    )
    opt_state = optimizer.init(eqx.filter(model, filter_spec))

    @eqx.filter_value_and_grad
    def loss_fn(diff_model, static_model):
        m = eqx.combine(diff_model, static_model)
        metric = GridZermelo(m.H_grid, m.W_grid)

        def one(ign, T_o, b):
            T = solve_arrival_grid(metric, ign, shape)
            return dense_arrival_loss(T, T_o, b, alpha)

        data = jnp.mean(
            jnp.stack([one(igns[i], T_obs[i], burned[i]) for i in range(len(fires))])
        )
        reg = tv_reg(m.H_grid, lam_tv_H) + tv_reg(m.W_grid, lam_tv_W)
        if coverage_weight > 0:
            reg = reg + coverage_weight * jnp.sum(
                (1.0 - cov) * jnp.sum(m.W_grid**2, axis=0)
            )
        return data + reg

    @eqx.filter_jit
    def step(model, opt_state):
        diff_model, static_model = eqx.partition(model, filter_spec)
        loss, grads = loss_fn(diff_model, static_model)
        updates, opt_state = optimizer.update(grads, opt_state, diff_model)
        model = eqx.combine(eqx.apply_updates(diff_model, updates), static_model)
        return model.project(), opt_state, loss

    losses = []
    for it in range(n_iter):
        model, opt_state, loss = step(model, opt_state)
        losses.append(float(loss))
        if verbose and it % 50 == 0:
            print(f"  iter {it}: loss={losses[-1]:.5f}")
        if it > 60 and abs(losses[-1] - losses[-11]) < 1e-7 * max(abs(losses[-11]), 1e-9):
            break
    return FitResult(model, losses)


def wind_error(W_est, W_true) -> float:
    """Mean pointwise wind-velocity error |Ŵ − W| (px/h), domain average."""
    d = np.asarray(W_est) - np.asarray(W_true)
    return float(np.mean(np.hypot(d[0], d[1])))


def sea_error(H_est, H_true) -> float:
    """Mean relative Frobenius error of the sea tensor."""
    d = np.asarray(H_est) - np.asarray(H_true)
    num = np.sqrt(d[0] ** 2 + 2 * d[1] ** 2 + d[2] ** 2)
    t = np.asarray(H_true)
    den = np.sqrt(t[0] ** 2 + 2 * t[1] ** 2 + t[2] ** 2)
    return float(np.mean(num / np.maximum(den, 1e-9)))


# ---------------------------------------------------------------------------
# Few-shot (s, c) recalibration — the scene gauge refit
# ---------------------------------------------------------------------------


class ScaledWindMetric(AsymmetricMetric):
    """Wrap any AsymmetricMetric, multiplying its wind velocity by ``c``.

    For a ``lam = 1`` (Gahtan) inner metric the scalar stays 1; for an exact
    Zermelo inner metric the causality scalar is recomputed for the scaled
    wind. Used by :func:`recalibrate` so one code path serves both the grid
    fields and the covariate encoder.
    """

    inner: AsymmetricMetric
    c: jax.Array

    def __init__(self, inner: AsymmetricMetric, c):
        super().__init__(manifold=inner.manifold)
        self.inner = inner
        self.c = jnp.asarray(c)

    def zermelo_data(self, z):
        H, W, lam = self.inner.zermelo_data(z)
        Wc = self.c * W
        w_norm_sq = Wc @ H @ Wc
        lam_new = jnp.where(
            lam >= 1.0 - 1e-9, lam, jnp.clip(1.0 - w_norm_sq, 1e-6, 1.0)
        )
        return H, Wc, lam_new

    def metric_fn(self, z, v):
        H, W, lam = self.zermelo_data(z)
        HW = H @ W
        B = -HW / lam
        G = (H + jnp.outer(HW, HW) / lam) / lam
        return jnp.sqrt(jnp.maximum(v @ G @ v, 1e-12)) + B @ v


@dataclass
class Recalibration:
    s: float
    c: float
    T_pred: np.ndarray  # recalibrated arrival field s * T_c
    mse: float


def recalibrate(
    solve_c,
    T_obs: np.ndarray,
    obs_mask: np.ndarray,
    c_grid=None,
    refine_steps: int = 2,
) -> Recalibration:
    """Fit the 2-parameter scene gauge ``(s, c)`` from partial observations.

    ``solve_c(c) -> T`` must return the arrival field with the model's wind
    velocity multiplied by ``c``. The time scale is exact and closed-form given
    ``c`` (arrival times are 1-homogeneous in the slowness):
    ``s*(c) = <T_c, T_obs> / <T_c, T_c>`` on the observed pixels — the convex
    inner loop. The outer loop is a 1-D grid over ``c`` with local refinement.

    Operationally honest: ``obs_mask`` is typically the first K hours of the
    new fire (early isochrones), available in any real forecast setting.
    """
    if c_grid is None:
        c_grid = np.linspace(0.0, 2.0, 9)
    obs_mask = np.asarray(obs_mask, dtype=bool)
    t = np.asarray(T_obs)[obs_mask]

    def eval_c(c):
        T_c = np.asarray(solve_c(float(c)))
        p = T_c[obs_mask]
        s = float(np.dot(p, t) / max(np.dot(p, p), 1e-12))
        mse = float(np.mean((s * p - t) ** 2))
        return s, mse, T_c

    results = {float(c): eval_c(c) for c in c_grid}
    for _ in range(refine_steps):
        cs = sorted(results)
        best = min(cs, key=lambda c: results[c][1])
        i = cs.index(best)
        lo = cs[max(i - 1, 0)]
        hi = cs[min(i + 1, len(cs) - 1)]
        for c in (0.5 * (lo + best), 0.5 * (best + hi)):
            if float(c) not in results:
                results[float(c)] = eval_c(c)
    c_best = min(results, key=lambda c: results[c][1])
    s_best, mse, T_c = results[c_best]
    return Recalibration(s=s_best, c=c_best, T_pred=s_best * T_c, mse=mse)
