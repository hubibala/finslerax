"""Randers fire spread on a pixel grid — even sea, odd wind, and the scene gauge.

A spreading fire is a unit-speed front in a Randers geometry: the directional
slowness along a front normal ``n`` is

    1/speed(n) = sqrt(n^T G n) + B . n

with ``G`` the even channel (fuel/slope anisotropy — invariant under n -> -n)
and ``B`` the odd channel (wind drift — flips sign). Arrival times are the
*ledger* channel: ``T`` is a value function, so exact drift parts ARE visible
in it (unlike path shapes, where gradient drifts are projectively invisible).
The binding identifiability constraint is instead **parity/direction coverage**:
a fire burns each pixel once, in one direction — one scalar slowness equation
against five unknowns — so within a single fire the G/B split is confounded.
Multi-fire (multi-source) coverage is what buys the odd channel back.

Cross-scene, one more low-dimensional nuisance rides on top: the scene's
absolute speed scale ``s`` (and wind coupling ``c``) — the affine gauge that
correlation metrics discard by construction and IoU@50 bills in full.

Conventions (bound once, everywhere):

* Rasters are ``(H, W)`` row-major; **solver coordinates are pixel
  coordinates** ``(row, col)`` with unit spacing, so speeds are px/hour and
  arrival times are HOURS. :class:`GridZermelo` and :func:`solve_arrival_grid`
  work in this frame directly.
* ``CovariateConditionedRanders`` works in world coordinates with
  ``x = col * spacing`` (east) and ``y = row * spacing``;
  :func:`solve_arrival_encoder` solves in that frame and transposes back to
  ``(H, W)``.
* Metric fields are the Gahtan ``lam = 1`` parametrization: sea ``H_grid``
  (SPD, even) + wind velocity ``W_grid`` (odd, downwind cheaper), inducing the
  primal cost ``F(v) = sqrt(v^T G v) + B.v`` with ``G = H + (HW)(HW)^T`` and
  ``B = -HW`` (arXiv:2603.00035 form; exact Bao–Robles–Shen Zermelo would use
  ``lam = 1 - ||W||_H^2``).
* IoU@50 follows the paper: burned regions at 50% of the *ground-truth* fire
  duration, in absolute hours — predicted times are normalised by the GT max,
  never by their own scale, so the affine gauge stays visible in the metric.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.metric import AsymmetricMetric
from ham.solvers.eikonal import EikonalSolver
from ham.utils.config import DEFAULT_JNP_DTYPE

# ---------------------------------------------------------------------------
# Grid Randers metric (pixel coordinates)
# ---------------------------------------------------------------------------


def _bilinear(grid: jax.Array, pt_rc: jax.Array) -> jax.Array:
    """Interpolate a ``(C, H, W)`` grid at continuous ``(row, col)`` coords."""
    _, H, W = grid.shape
    pr, pc = pt_rc[0], pt_rc[1]
    pc = jnp.clip(pc, 0.0, W - 1.001)
    pr = jnp.clip(pr, 0.0, H - 1.001)
    c0 = jnp.floor(pc).astype(jnp.int32)
    r0 = jnp.floor(pr).astype(jnp.int32)
    c1 = jnp.minimum(c0 + 1, W - 1)
    r1 = jnp.minimum(r0 + 1, H - 1)
    fc = pc - c0
    fr = pr - r0
    return (
        grid[:, r0, c0] * (1 - fc) * (1 - fr)
        + grid[:, r0, c1] * fc * (1 - fr)
        + grid[:, r1, c0] * (1 - fc) * fr
        + grid[:, r1, c1] * fc * fr
    )


class GridZermelo(AsymmetricMetric):
    """Randers metric from bilinearly interpolated ``(H_grid, W_grid)`` rasters.

    ``H_grid``: (3, H, W) sea components (h11, h12, h22) in (row, col) axes —
    even channel. ``W_grid``: (2, H, W) wind velocity (w_row, w_col) — odd
    channel, downwind cheaper. Coordinates are pixel ``(row, col)``; pass
    ``grid_extent=(0, H-1, 0, W-1)`` to the eikonal solver.
    """

    H_grid: jax.Array
    W_grid: jax.Array

    def __init__(self, H_grid: jax.Array, W_grid: jax.Array):
        super().__init__(manifold=EuclideanSpace(2))
        self.H_grid = jnp.asarray(H_grid)
        self.W_grid = jnp.asarray(W_grid)

    def zermelo_data(self, z: jax.Array):
        h = _bilinear(self.H_grid, z)
        w = _bilinear(self.W_grid, z)
        H_mat = jnp.array([[h[0], h[1]], [h[1], h[2]]])
        return H_mat, w, jnp.asarray(1.0)

    def metric_fn(self, z: jax.Array, v: jax.Array) -> jax.Array:
        """Primal Randers cost F(v) = sqrt(v^T G v) + B.v (lam = 1 form)."""
        H, W, _lam = self.zermelo_data(z)
        HW = H @ W
        B = -HW
        G = H + jnp.outer(HW, HW)
        return jnp.sqrt(jnp.maximum(v @ G @ v, 1e-12)) + B @ v


def grid_to_godunov(H_grid: jax.Array, W_grid: jax.Array):
    """Vectorised ``(H, W) -> (G, B)`` Godunov fields (lam = 1)."""
    h11, h12, h22 = H_grid[0], H_grid[1], H_grid[2]
    w1, w2 = W_grid[0], W_grid[1]
    b1 = -(h11 * w1 + h12 * w2)
    b2 = -(h12 * w1 + h22 * w2)
    G = jnp.stack([h11 + b1 * b1, h12 + b1 * b2, h22 + b2 * b2])
    B = jnp.stack([b1, b2])
    return G, B


def godunov_to_grid(G: jax.Array, B: jax.Array):
    """Inverse of :func:`grid_to_godunov`: ``(G, B) -> (H_grid, W_grid)``."""
    g11, g12, g22 = G[0], G[1], G[2]
    b1, b2 = B[0], B[1]
    h11 = g11 - b1 * b1
    h12 = g12 - b1 * b2
    h22 = g22 - b2 * b2
    det = jnp.maximum(h11 * h22 - h12**2, 1e-9)
    w1 = -(h22 * b1 - h12 * b2) / det
    w2 = -(-h12 * b1 + h11 * b2) / det
    return jnp.stack([h11, h12, h22]), jnp.stack([w1, w2])


# ---------------------------------------------------------------------------
# Arrival-time solves
# ---------------------------------------------------------------------------

SOLVER = EikonalSolver(max_iters=80, tol=1e-5)


def solve_arrival_grid(
    metric: GridZermelo,
    ignition_rc: jax.Array,
    shape: tuple[int, int],
    solver: EikonalSolver = SOLVER,
) -> jax.Array:
    """Arrival-time field (hours) from ignition(s) in pixel (row, col) coords."""
    H, W = shape
    T, _, _ = solver.solve(
        metric,
        jnp.asarray(ignition_rc, dtype=DEFAULT_JNP_DTYPE),
        grid_extent=(0.0, float(H - 1), 0.0, float(W - 1)),
        grid_shape=(H, W),
    )
    return T


def solve_arrival_encoder(
    model,
    ignition_rc: jax.Array,
    shape: tuple[int, int],
    pixel_spacing_m: float,
    solver: EikonalSolver = SOLVER,
) -> jax.Array:
    """Arrival field (hours, (H, W) row-major) for a bound covariate encoder.

    The encoder's world frame is ``x = col * spacing`` / ``y = row * spacing``,
    so we solve on a grid whose axis 0 is x (shape ``(W, H)``) and transpose
    back. ``model`` must have scene + weather bound and ``metric_field``
    precomputed.
    """
    H, W = shape
    s = float(pixel_spacing_m)
    ign = jnp.asarray(ignition_rc, dtype=DEFAULT_JNP_DTYPE)
    src_xy = jnp.stack([ign[..., 1] * s, ign[..., 0] * s], axis=-1)
    T_xy, _, _ = solver.solve(
        model,
        src_xy,
        grid_extent=(0.0, (W - 1) * s, 0.0, (H - 1) * s),
        grid_shape=(W, H),
    )
    return T_xy.T


# ---------------------------------------------------------------------------
# Metrics (paper conventions, arXiv:2603.00035)
# ---------------------------------------------------------------------------


def pearson_corr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Pearson correlation between predicted and GT arrival times on ``mask``."""
    p = np.asarray(pred)[mask]
    g = np.asarray(gt)[mask]
    if p.size < 2 or p.std() < 1e-12 or g.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, g)[0, 1])


def iou_at_50_hours(
    pred_hours: np.ndarray, gt_hours: np.ndarray, burned: np.ndarray
) -> float:
    """IoU of burned regions at 50% of the GT fire duration, in absolute hours.

    Both fields are thresholded at ``0.5 * max(gt on burned)`` — the predicted
    field is deliberately NOT normalised by its own scale, so a wrong absolute
    speed scale shows up as a wrong 50%-perimeter (the paper's IoU@50).
    """
    burned = np.asarray(burned, dtype=bool)
    if not burned.any():
        return float("nan")
    tau = 0.5 * float(np.asarray(gt_hours)[burned].max())
    if tau <= 0:
        return float("nan")
    pred_bin = (np.asarray(pred_hours) <= tau) & burned
    gt_bin = (np.asarray(gt_hours) <= tau) & burned
    union = np.sum(pred_bin | gt_bin)
    return float(np.sum(pred_bin & gt_bin) / union) if union else 0.0


def rel_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """RMSE normalised by the maximum GT arrival time (paper's rel. RMSE)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    g = np.asarray(gt)[m]
    d = np.asarray(pred)[m] - g
    return float(np.sqrt(np.mean(d**2)) / max(g.max(), 1e-9))


def fire_metrics(pred_hours, gt_hours, burned) -> dict:
    """The three paper metrics for one fire, as a dict."""
    return {
        "corr": pearson_corr(pred_hours, gt_hours, burned),
        "iou50": iou_at_50_hours(pred_hours, gt_hours, burned),
        "rel_rmse": rel_rmse(pred_hours, gt_hours, burned),
    }


# ---------------------------------------------------------------------------
# Dense arrival loss (both gauges) + TV regularisation
# ---------------------------------------------------------------------------


def dense_arrival_loss(
    T_pred: jax.Array, T_obs: jax.Array, burned: jax.Array, alpha: float | jax.Array
) -> jax.Array:
    """Masked dense arrival loss; ``alpha`` blends the two gauges.

    ``alpha = 0``: 1 − Pearson (affine-invariant — learns shape, discards the
    scene scale). ``alpha = 1``: relative MSE normalised by the constant
    ``max(T_obs)`` (absolute — the gauge is billed here, monotone-equivalent
    to the paper's ½‖T − T_obs‖²). Report both; their gap IS the gauge.
    """
    mask = jnp.asarray(burned) & jnp.isfinite(T_obs)
    n = jnp.sum(mask) + 1e-8
    p = jnp.where(mask, T_pred, 0.0)
    o = jnp.where(mask, T_obs, 0.0)
    mu_p = jnp.sum(p) / n
    mu_o = jnp.sum(o) / n
    dp = jnp.where(mask, p - mu_p, 0.0)
    do = jnp.where(mask, o - mu_o, 0.0)
    l_pearson = 1.0 - jnp.sum(dp * do) / jnp.sqrt(
        jnp.sum(dp**2) * jnp.sum(do**2) + 1e-8
    )
    t_max = jnp.maximum(jnp.max(jnp.where(mask, T_obs, -jnp.inf)), 1e-6)
    l_abs = jnp.sum(jnp.where(mask, ((T_pred - T_obs) / t_max) ** 2, 0.0)) / n
    return (1.0 - alpha) * l_pearson + alpha * l_abs


def tv_reg(field: jax.Array, lam: float) -> jax.Array:
    """Gahtan-exact sum-reduced TV over a ``(C, H, W)`` field."""
    if lam <= 0:
        return jnp.asarray(0.0)
    dx = jnp.abs(field[:, :, 1:] - field[:, :, :-1])
    dy = jnp.abs(field[:, 1:, :] - field[:, :-1, :])
    return lam * (jnp.sum(dx) + jnp.sum(dy))


# ---------------------------------------------------------------------------
# Direction-coverage mask (the identifiability ledger, computable pre-fit)
# ---------------------------------------------------------------------------


def front_normals(T_obs: np.ndarray, burned: np.ndarray, smooth_sigma: float = 1.5):
    """Unit front normals ``∇T/|∇T|`` of one fire, NaN where undefined.

    The arrival field is smoothed (Gaussian, burned-region-aware) before
    differencing because satellite isochrones are hour-quantised.
    Returns ``(H, W, 2)`` array of (d/drow, d/dcol) unit normals.
    """
    from scipy.ndimage import gaussian_filter

    burned = np.asarray(burned, dtype=bool)
    T = np.where(burned, T_obs, 0.0)
    w = burned.astype(float)
    num = gaussian_filter(T * w, smooth_sigma)
    den = np.maximum(gaussian_filter(w, smooth_sigma), 1e-6)
    T_s = num / den
    gr, gc = np.gradient(T_s)
    mag = np.hypot(gr, gc)
    ok = burned & (mag > 1e-9)
    n = np.full((*T_obs.shape, 2), np.nan)
    n[ok, 0] = gr[ok] / mag[ok]
    n[ok, 1] = gc[ok] / mag[ok]
    return n


def _structure_tensor(normal_list: list[np.ndarray]) -> np.ndarray:
    """Per-pixel ``M(x) = Σ_f n_f(x) n_f(x)^T`` over fires; (H, W, 2, 2)."""
    shape = normal_list[0].shape[:2]
    M = np.zeros((*shape, 2, 2))
    for n in normal_list:
        ok = np.isfinite(n[..., 0])
        outer = np.einsum("...i,...j->...ij", np.nan_to_num(n), np.nan_to_num(n))
        M += np.where(ok[..., None, None], outer, 0.0)
    return M


def _eig_ratio(M: np.ndarray) -> np.ndarray:
    """λ_min/λ_max of a field of symmetric 2x2 matrices, in [0, 1]."""
    tr = M[..., 0, 0] + M[..., 1, 1]
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] ** 2
    disc = np.sqrt(np.maximum(tr**2 - 4 * det, 0.0))
    lam_max = 0.5 * (tr + disc)
    lam_min = 0.5 * (tr - disc)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(lam_max > 1e-9, lam_min / np.maximum(lam_max, 1e-12), 0.0)
    return np.clip(cov, 0.0, 1.0)


def direction_coverage(normal_list: list[np.ndarray], block: int = 1) -> np.ndarray:
    """Direction coverage in [0, 1] from per-fire front normals.

    ``λ_min/λ_max`` of the accumulated structure tensor — 0 where every fire
    crossed in the same (or no) direction (odd channel invisible), → 1 for
    isotropic coverage. The field analogue of the UAV ``ledger_conditioned``
    gate: computable from observations alone, before any fit.

    ``block > 1`` pools the structure tensor over ``block x block`` patches
    before the eigen-ratio (returning the coarse (H//block, W//block) map).
    This is the honest granularity: arrival-time information is transported
    along characteristics and coupled by smoothness, so identifiability is a
    patch property at the wind's correlation scale, not a per-pixel one — the
    per-pixel map is a visual, the block map is the statistic (Stage W-B).
    """
    M = _structure_tensor(normal_list)
    if block > 1:
        H, W = M.shape[:2]
        M = M[: H // block * block, : W // block * block]
        M = M.reshape(H // block, block, W // block, block, 2, 2).sum(axis=(1, 3))
    return _eig_ratio(M)


def odd_coverage(normal_list: list[np.ndarray], block: int = 8) -> np.ndarray:
    """Odd-channel (wind) identifiability per block, from front normals alone.

    Each observation direction ``n`` contributes one slowness equation
    ``s(n) = sqrt(n^T G n) + B.n``. The structure tensor (see
    :func:`direction_coverage`) is blind to the sign of ``n`` — yet the odd
    channel is identified precisely by *signed* diversity: an antiparallel
    pair separates B exactly (``s(n) - s(-n) = 2 B.n``), while single-signed
    orthogonal directions let the even channel absorb B.

    Statistic: stack, over all fires and pixels in the block, rows
    ``[q(n) | n]`` with ``q(n) = (n1^2, 2 n1 n2, n2^2)`` the even (sign-blind)
    monomials; project the even span out of the ``n`` columns and return the
    smallest singular value of the residual, normalised by sqrt(#rows) — 0
    when B is unidentifiable given an unknown G, → bounded above as signed
    direction diversity grows.
    """
    return odd_null_data(normal_list, block)[0]


def odd_null_data(normal_list: list[np.ndarray], block: int = 8):
    """Per-block odd-channel identifiability data from front normals.

    For every ``block x block`` patch, stack the slowness linearisation rows
    over all fires/pixels: ``ds_i = q(n_i) . xG + n_i . xB`` with the even
    monomials ``q(n) = (n1^2, 2 n1 n2, n2^2)``. Least-squares-project the even
    span out of the signed direction columns: ``R = N - Q @ coef``.

    Returns ``(cov, u, coef_u)``:

    * ``cov``    (Hb, Wb) — ``sigma_min(R)/sqrt(m)``, the odd coverage score;
    * ``u``      (Hb, Wb, 2) — the least-identified WIND direction (right
      singular vector of ``R`` for the smallest singular value);
    * ``coef_u`` (Hb, Wb, 3) — the even-channel compensation ``coef @ u``:
      perturbing ``B += d*u`` together with ``G_params -= 2*d*(coef@u)``
      changes every observed slowness by only ``d * R@u`` (norm
      ``d * sigma_min``), i.e. it is loss-flat exactly where ``cov = 0``.
      This is what the Stage W-B curvature gate exercises.
    """
    shape = normal_list[0].shape[:2]
    Hb, Wb = shape[0] // block, shape[1] // block
    cov = np.zeros((Hb, Wb))
    u_out = np.zeros((Hb, Wb, 2))
    coef_out = np.zeros((Hb, Wb, 3))
    for bi in range(Hb):
        for bj in range(Wb):
            rows = []
            for n in normal_list:
                patch = n[bi * block:(bi + 1) * block, bj * block:(bj + 1) * block]
                ok = np.isfinite(patch[..., 0])
                rows.append(patch[ok])
            N = np.concatenate(rows, axis=0)
            if len(N) < 5:
                continue
            Q = np.stack([N[:, 0] ** 2, 2 * N[:, 0] * N[:, 1], N[:, 1] ** 2], axis=1)
            coef, *_ = np.linalg.lstsq(Q, N, rcond=None)
            R = N - Q @ coef
            _, s, Vt = np.linalg.svd(R, full_matrices=False)
            cov[bi, bj] = s[-1] / np.sqrt(len(N))
            u_out[bi, bj] = Vt[-1]
            coef_out[bi, bj] = coef @ Vt[-1]
    return cov, u_out, coef_out
