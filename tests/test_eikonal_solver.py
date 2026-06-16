import equinox as eqx
import jax.numpy as jnp

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.metric import AsymmetricMetric
from ham.geometry.zoo import Randers
from ham.solvers.eikonal import EikonalSolver, _fast_sweeping_solve


class DummyRandersMetric(AsymmetricMetric):
    wind_scale: float = eqx.field(static=True)
    def __init__(self, dim: int, wind_scale: float = 0.0):
        self.manifold = EuclideanSpace(dim)
        self.wind_scale = float(wind_scale)

    def metric_fn(self, x, v):
        H, W, lam = self.zermelo_data(x)
        # Re-implement metric_fn just for testing completeness or rely on Zermelo alone
        return jnp.sqrt(jnp.sum(v**2))

    def zermelo_data(self, x):
        # Identity sea metric
        H = jnp.eye(2)
        # Constant wind vector
        W = jnp.array([self.wind_scale, self.wind_scale])
        w_norm_sq = jnp.dot(W, jnp.dot(H, W))
        lam = 1.0 - w_norm_sq
        return H, W, lam

def test_eikonal_solver_forward_isotropic():
    solver = EikonalSolver(max_iters=100, tol=1e-5)
    # 0 wind gives a standard Euclidean metric
    metric = DummyRandersMetric(2, wind_scale=0.0)

    nx, ny = 21, 21
    extent = (-1.0, 1.0, -1.0, 1.0)
    source = jnp.array([0.0, 0.0])

    T, X, Y = solver.solve(metric, source, extent, (nx, ny))

    # Analytical distance is sqrt(x^2 + y^2)
    T_analytical = jnp.sqrt(X**2 + Y**2)

    max_error = jnp.max(jnp.abs(T - T_analytical))
    assert max_error < 0.1, f"Max error {max_error} too large for isotropic case"

def test_fast_sweeping_gradients_anisotropic():
    # Test implicit differentiation gradients using jax.test_util.check_grads
    from jax.test_util import check_grads

    nx, ny = 11, 11
    hx, hy = 0.2, 0.2

    source_mask = jnp.zeros((nx, ny), dtype=bool).at[5, 5].set(True)

    # Create an anisotropic G and B to test the full Godunov branches
    # G needs to be positive definite. We'll use identity for simplicity but with perturbations.
    g11 = jnp.ones((nx, ny)) + 0.1 * jnp.sin(jnp.arange(nx)[:, None])
    g22 = jnp.ones((nx, ny)) + 0.1 * jnp.cos(jnp.arange(ny)[None, :])
    g12 = 0.2 * jnp.ones((nx, ny))
    G = jnp.stack([g11, g12, g22], axis=0)

    # B represents drift
    b1 = 0.1 * jnp.ones((nx, ny))
    b2 = -0.1 * jnp.ones((nx, ny))
    B = jnp.stack([b1, b2], axis=0)

    def fwd(g_tensor, b_tensor):
        return _fast_sweeping_solve(g_tensor, b_tensor, source_mask, hx, hy, 100, 1e-6)

    check_grads(fwd, (G, B), order=1, modes=['rev'], eps=1e-3)


# ---------------------------------------------------------------------------
# Directional regression tests (would have caught the axis-swap and the
# wind-sign-flip bugs fixed in 2026-06; see reviews/REPORT_SOLVERS_2026-06-12.md)
# ---------------------------------------------------------------------------


class ConstWindMetric(AsymmetricMetric):
    """Identity sea metric with a constant wind vector."""

    wx: float = eqx.field(static=True)
    wy: float = eqx.field(static=True)

    def __init__(self, wx: float = 0.0, wy: float = 0.0):
        self.manifold = EuclideanSpace(2)
        self.wx = float(wx)
        self.wy = float(wy)

    def metric_fn(self, x, v):
        H, W, lam = self.zermelo_data(x)
        Hv = H @ v
        Wv = W @ Hv
        return (jnp.sqrt(lam * (v @ Hv) + Wv**2) - Wv) / lam

    def zermelo_data(self, x):
        H = jnp.eye(2)
        W = jnp.array([self.wx, self.wy])
        return H, W, 1.0 - W @ H @ W


class AnisoSeaMetric(AsymmetricMetric):
    """Diagonal sea metric diag(4, 1): motion along x is twice as slow."""

    def __init__(self):
        self.manifold = EuclideanSpace(2)

    def metric_fn(self, x, v):
        H, _, _ = self.zermelo_data(x)
        return jnp.sqrt(v @ H @ v)

    def zermelo_data(self, x):
        return jnp.diag(jnp.array([4.0, 1.0])), jnp.zeros(2), jnp.array(1.0)


def _solve_probe(metric, n=61, d=0.5):
    solver = EikonalSolver(max_iters=200, tol=1e-6)
    T, _, _ = solver.solve(
        metric, jnp.array([0.0, 0.0]), (-1.0, 1.0, -1.0, 1.0), (n, n)
    )
    ic = n // 2
    off = int(round(d / (2.0 / (n - 1))))
    return T, ic, off


def test_eikonal_constant_wind_directional():
    """Downwind must be fast, upwind slow: T(+x)=d/(1+w), T(-x)=d/(1-w)."""
    w, d = 0.5, 0.5
    T, ic, off = _solve_probe(ConstWindMetric(wx=w), d=d)

    assert abs(T[ic + off, ic] - d / (1 + w)) < 0.03, (
        f"downwind T={T[ic + off, ic]:.4f}, expected {d / (1 + w):.4f}"
    )
    assert abs(T[ic - off, ic] - d / (1 - w)) < 0.05, (
        f"upwind T={T[ic - off, ic]:.4f}, expected {d / (1 - w):.4f}"
    )
    # Crosswind (Zermelo tacking): d / sqrt(1 - w^2)
    cross = d / jnp.sqrt(1 - w * w)
    assert abs(T[ic, ic + off] - cross) < 0.05
    assert abs(T[ic, ic - off] - cross) < 0.05


def test_eikonal_constant_wind_y_axis():
    """Same as above with the wind along y - guards the axis pairing."""
    w, d = 0.5, 0.5
    T, ic, off = _solve_probe(ConstWindMetric(wy=w), d=d)

    assert abs(T[ic, ic + off] - d / (1 + w)) < 0.03
    assert abs(T[ic, ic - off] - d / (1 - w)) < 0.05
    cross = d / jnp.sqrt(1 - w * w)
    assert abs(T[ic + off, ic] - cross) < 0.05
    assert abs(T[ic - off, ic] - cross) < 0.05


def test_eikonal_diagonal_anisotropy():
    """H = diag(4, 1) makes x-travel twice as slow: T(+-x)=2d, T(+-y)=d."""
    d = 0.5
    T, ic, off = _solve_probe(AnisoSeaMetric(), d=d)

    assert abs(T[ic + off, ic] - 2 * d) < 0.05
    assert abs(T[ic - off, ic] - 2 * d) < 0.05
    assert abs(T[ic, ic + off] - d) < 0.03
    assert abs(T[ic, ic - off] - d) < 0.03


# ---------------------------------------------------------------------------
# Production-path regression: the *real* Randers.zermelo_data (soft causal
# clamp) must reproduce the analytic Zermelo times for the *requested* wind.
# The old ``tanh`` squash silently mapped 0.5 -> 0.46, so downwind/upwind times
# would have matched d/(1±0.46) instead of d/(1±0.5); this test pins the fix.
# ---------------------------------------------------------------------------


def test_eikonal_real_randers_faithful_to_requested_wind():
    """Soft-clamped Randers reproduces analytic Zermelo for the requested wind.

    A strong wind (0.8) is used so the discrimination is unambiguous: the old
    ``tanh`` squash mapped 0.8 -> ~0.66, shifting downwind time from d/1.8 to
    d/1.66 -- a gap far larger than the first-order FSM error.
    """
    w, d = 0.8, 0.5
    metric = Randers(
        EuclideanSpace(2),
        lambda x: jnp.eye(2),
        lambda x: jnp.array([w, 0.0]),  # constant, physically valid wind
    )
    T, ic, off = _solve_probe(metric, n=81, d=d)

    downwind = float(T[ic + off, ic])
    # Faithful to the *requested* 0.8.
    assert abs(downwind - d / (1 + w)) < 0.02, downwind
    # And clearly NOT the old squashed value d/(1 + tanh(0.8)).
    w_old = float(jnp.tanh(w))  # ~0.664
    assert abs(downwind - d / (1 + w_old)) > abs(downwind - d / (1 + w)), downwind


def test_eikonal_refinement_convergence():
    """First-order FSM error must shrink as the grid is refined (consistency)."""
    metric = ConstWindMetric(wx=0.4)
    d = 0.5
    w = 0.4
    target_down = d / (1 + w)

    errs = []
    for n in (41, 81):
        T, ic, off = _solve_probe(metric, n=n, d=d)
        errs.append(abs(float(T[ic + off, ic]) - target_down))

    # Refining the grid must not increase the error (monotone consistency).
    assert errs[1] <= errs[0] + 1e-4, f"error grew under refinement: {errs}"
