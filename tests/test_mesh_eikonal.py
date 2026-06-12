import equinox as eqx
import jax.numpy as jnp

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.mesh_adjacency import MeshAdjacency
from ham.geometry.metric import AsymmetricMetric
from ham.solvers.mesh_eikonal import MeshEikonalSolver, _fast_mesh_solve


class DummyRandersMetric(AsymmetricMetric):
    wind_scale: float = eqx.field(static=True)
    def __init__(self, dim: int, wind_scale: float = 0.0):
        self.manifold = EuclideanSpace(dim)
        self.wind_scale = float(wind_scale)

    def metric_fn(self, x, v):
        return jnp.sqrt(jnp.sum(v**2))

    def zermelo_data(self, x):
        H = jnp.eye(2)
        W = jnp.array([self.wind_scale, self.wind_scale])
        w_norm_sq = jnp.dot(W, jnp.dot(H, W))
        lam = 1.0 - w_norm_sq
        return H, W, lam

def _create_simple_mesh():
    # A simple planar mesh (a square with two triangles)
    vertices = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ], dtype=jnp.float32)
    faces = jnp.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=jnp.int32)
    return vertices, faces

def test_mesh_solver_forward():
    vertices, faces = _create_simple_mesh()
    mesh_adj = MeshAdjacency.build(vertices, faces, num_ref_points=1)

    solver = MeshEikonalSolver(max_iters=50, tol=1e-5)
    metric = DummyRandersMetric(2, wind_scale=0.0)

    source = jnp.array([[0.0, 0.0]])
    T = solver.solve(metric, mesh_adj, vertices, faces, source)

    # Distance to source [0, 0] should be roughly Euclidean
    T_analytical = jnp.sqrt(jnp.sum(vertices**2, axis=-1))

    # It won't be perfectly Euclidean because it's constrained to the edges of the mesh
    # and first-order Godunov has numerical diffusion (linear interpolation on the edge).
    max_error = jnp.max(jnp.abs(T - T_analytical))
    assert max_error < 0.35, f"Max error {max_error} too large"

def test_mesh_sweeping_gradients():
    from jax.test_util import check_grads

    vertices, faces = _create_simple_mesh()
    mesh_adj = MeshAdjacency.build(vertices, faces, num_ref_points=1)

    F = len(faces)
    V = len(vertices)

    G_faces = jnp.stack([jnp.eye(2) for _ in range(F)])
    B_faces = jnp.zeros((F, 2))

    source_mask = jnp.zeros(V, dtype=bool).at[0].set(True)

    def fwd(g_tensor, b_tensor):
        g_tensor = jnp.asarray(g_tensor)
        b_tensor = jnp.asarray(b_tensor)
        return _fast_mesh_solve(
            g_tensor, b_tensor, source_mask,
            mesh_adj.sweep_orderings, mesh_adj.vertex_adjacency,
            vertices, 50, 1e-6
        )

    check_grads(fwd, (G_faces, B_faces), order=1, modes=['rev'], eps=1e-3)


# ---------------------------------------------------------------------------
# Directional regression test (would have caught the wind-sign-flip bug fixed
# in 2026-06; see reviews/REPORT_SOLVERS_2026-06-12.md)
# ---------------------------------------------------------------------------


def _structured_grid_mesh(nm: int):
    """Regular triangulated grid over [-1, 1]^2 with nm x nm vertices."""
    import numpy as np

    xs = np.linspace(-1.0, 1.0, nm)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    verts = np.stack([XX.ravel(), YY.ravel()], axis=-1)

    def vid(i, j):
        return i * nm + j

    faces = []
    for i in range(nm - 1):
        for j in range(nm - 1):
            faces.append([vid(i, j), vid(i + 1, j), vid(i, j + 1)])
            faces.append([vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
    return (
        jnp.array(verts, dtype=jnp.float32),
        jnp.array(np.array(faces, dtype=np.int32)),
    )


class ConstWindMetric(AsymmetricMetric):
    wx: float = eqx.field(static=True)

    def __init__(self, wx: float):
        self.manifold = EuclideanSpace(2)
        self.wx = float(wx)

    def metric_fn(self, x, v):
        H, W, lam = self.zermelo_data(x)
        Hv = H @ v
        Wv = W @ Hv
        return (jnp.sqrt(lam * (v @ Hv) + Wv**2) - Wv) / lam

    def zermelo_data(self, x):
        H = jnp.eye(2)
        W = jnp.array([self.wx, 0.0])
        return H, W, 1.0 - W @ H @ W


def test_mesh_constant_wind_directional():
    """Downwind arrival must be earlier than upwind on a planar mesh."""
    import numpy as np

    nm = 21
    w, d = 0.5, 0.5
    verts, faces = _structured_grid_mesh(nm)
    mesh_adj = MeshAdjacency.build(verts, faces, num_ref_points=2)

    solver = MeshEikonalSolver(max_iters=100, tol=1e-6)
    T = solver.solve(ConstWindMetric(w), mesh_adj, verts, faces,
                     jnp.array([[0.0, 0.0]]))
    Tg = np.array(T).reshape(nm, nm)

    ic = nm // 2
    off = int(round(d / (2.0 / (nm - 1))))

    assert abs(Tg[ic + off, ic] - d / (1 + w)) < 0.03, (
        f"downwind T={Tg[ic + off, ic]:.4f}, expected {d / (1 + w):.4f}"
    )
    assert abs(Tg[ic - off, ic] - d / (1 - w)) < 0.06, (
        f"upwind T={Tg[ic - off, ic]:.4f}, expected {d / (1 - w):.4f}"
    )
    # Crosswind: d / sqrt(1 - w^2), looser tol (mesh diffusion off-axis)
    cross = d / np.sqrt(1 - w * w)
    assert abs(Tg[ic, ic + off] - cross) < 0.1
    assert abs(Tg[ic, ic - off] - cross) < 0.1
