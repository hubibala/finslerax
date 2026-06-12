"""Tests for the 3D volumetric anisotropic Eikonal solver.

Covers forward accuracy against analytic Randers/Riemannian distances
(isotropic, diagonal anisotropy, constant wind) and gradient consistency of
the implicit custom-VJP backward pass.
"""

import equinox as eqx
import jax.numpy as jnp

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.metric import AsymmetricMetric
from ham.solvers.volumetric_eikonal import VolumetricEikonalSolver, _volumetric_solve


class Iso3D(AsymmetricMetric):
    def __init__(self):
        self.manifold = EuclideanSpace(3)

    def metric_fn(self, x, v):
        return jnp.sqrt(jnp.sum(v**2))

    def zermelo_data(self, x):
        return jnp.eye(3), jnp.zeros(3), jnp.array(1.0)


class AnisoSea3D(AsymmetricMetric):
    """Sea metric diag(4, 1, 1): motion along x twice as slow."""

    def __init__(self):
        self.manifold = EuclideanSpace(3)

    def metric_fn(self, x, v):
        H, _, _ = self.zermelo_data(x)
        return jnp.sqrt(v @ H @ v)

    def zermelo_data(self, x):
        return jnp.diag(jnp.array([4.0, 1.0, 1.0])), jnp.zeros(3), jnp.array(1.0)


class Wind3D(AsymmetricMetric):
    """Identity sea with constant wind along x."""

    w: float = eqx.field(static=True)

    def __init__(self, w: float):
        self.manifold = EuclideanSpace(3)
        self.w = float(w)

    def metric_fn(self, x, v):
        H, W, lam = self.zermelo_data(x)
        Hv = H @ v
        Wv = W @ Hv
        return (jnp.sqrt(lam * (v @ Hv) + Wv**2) - Wv) / lam

    def zermelo_data(self, x):
        W = jnp.array([self.w, 0.0, 0.0])
        return jnp.eye(3), W, 1.0 - jnp.sum(W**2)


_EXT = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
_SRC = jnp.array([[0.0, 0.0, 0.0]])


def _probe(metric, n=41, d=0.5):
    # n=41 keeps the probe lattice-aligned: h=0.05 and d=0.5 is exactly
    # 10 cells, so T at the probe index is comparable to the analytic value.
    solver = VolumetricEikonalSolver(max_iters=100, tol=1e-6)
    T, Q, B = solver.solve(metric, _SRC, _EXT, (n, n, n))
    ic = n // 2
    off = int(round(d / (2.0 / (n - 1))))
    return T, Q, B, ic, off


def test_volumetric_isotropic_forward():
    d = 0.5
    T, _, _, ic, off = _probe(Iso3D(), d=d)
    assert abs(T[ic + off, ic, ic] - d) < 0.03
    assert abs(T[ic, ic + off, ic] - d) < 0.03
    assert abs(T[ic, ic, ic + off] - d) < 0.03
    # Diagonal: first-order Godunov diffusion, looser tolerance
    assert abs(T[ic + off, ic + off, ic] - d * jnp.sqrt(2.0)) < 0.08


def test_volumetric_diagonal_anisotropy():
    """Would have caught the primal-vs-dual tensor confusion (C4)."""
    d = 0.5
    T, Q, _, ic, off = _probe(AnisoSea3D(), d=d)
    assert abs(T[ic + off, ic, ic] - 2 * d) < 0.05
    assert abs(T[ic, ic + off, ic] - d) < 0.03
    # The stored Q must be the dual tensor: Q11 = 1/H11 = 0.25
    assert abs(Q[0, ic, ic, ic] - 0.25) < 1e-5


def test_volumetric_constant_wind_directional():
    """Would have caught the signed-upwinding drift handling (C5)."""
    w, d = 0.5, 0.5
    T, _, _, ic, off = _probe(Wind3D(w), d=d)
    assert abs(T[ic + off, ic, ic] - d / (1 + w)) < 0.03, (
        f"downwind T={T[ic + off, ic, ic]:.4f}, expected {d / (1 + w):.4f}"
    )
    assert abs(T[ic - off, ic, ic] - d / (1 - w)) < 0.05, (
        f"upwind T={T[ic - off, ic, ic]:.4f}, expected {d / (1 - w):.4f}"
    )
    cross = d / jnp.sqrt(1 - w * w)
    assert abs(T[ic, ic + off, ic] - cross) < 0.05
    assert abs(T[ic, ic, ic + off] - cross) < 0.05


def test_volumetric_gradients():
    """Reverse-mode gradients of the implicit solve are FD-consistent."""
    from jax.test_util import check_grads

    n = 9
    h = 2.0 / (n - 1)
    source_mask = jnp.zeros((n, n, n), dtype=bool).at[n // 2, n // 2, n // 2].set(True)

    ax = jnp.arange(n, dtype=jnp.float32)
    q11 = 1.0 + 0.1 * jnp.sin(ax)[:, None, None] * jnp.ones((n, n, n))
    q22 = 1.0 + 0.1 * jnp.cos(ax)[None, :, None] * jnp.ones((n, n, n))
    q33 = jnp.ones((n, n, n))
    q12 = 0.1 * jnp.ones((n, n, n))
    q13 = 0.05 * jnp.ones((n, n, n))
    q23 = -0.05 * jnp.ones((n, n, n))
    Q = jnp.stack([q11, q22, q33, q12, q13, q23], axis=0)

    B = jnp.stack(
        [
            0.1 * jnp.ones((n, n, n)),
            -0.1 * jnp.ones((n, n, n)),
            0.05 * jnp.ones((n, n, n)),
        ],
        axis=0,
    )

    def fwd(q_tensor, b_tensor):
        return _volumetric_solve(q_tensor, b_tensor, source_mask, h, h, h, 100, 1e-6)

    check_grads(fwd, (Q, B), order=1, modes=["rev"], eps=1e-3)
