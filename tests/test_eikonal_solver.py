import equinox as eqx
import jax.numpy as jnp

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.metric import AsymmetricMetric
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
