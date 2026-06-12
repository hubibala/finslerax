"""Anisotropic arrival-time fronts: mesh + volumetric Eikonal solvers.

Showcases the two Eulerian fast-sweeping solvers on a single figure:

Top row — :class:`MeshEikonalSolver` on a curved manifold (icosphere):
    Geodesic arrival times from a point source under a Randers metric.
    Without wind the wavefronts are concentric great circles; switching on a
    zonal jet (rotation about the z-axis) skews every front downwind while
    the *same* mesh and source are used — the asymmetry is purely metric.

Bottom row — :class:`VolumetricEikonalSolver` on a dense 3D grid:
    A "slow lens" (refractive inclusion, 55% speed reduction) plus a planar
    vortex wind. The mid-plane slice shows fronts refracting around the lens
    and advecting with the vortex; the 3D panel extracts nested wavefront
    isosurfaces with the differentiable marching cubes.

Run:  python examples/demo_eikonal_fronts.py
Writes examples/visualizations/eikonal_fronts.png and opens the figure.
"""

import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ham.geometry import Randers, Sphere
from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.mesh_adjacency import MeshAdjacency
from ham.solvers import MeshEikonalSolver, VolumetricEikonalSolver
from ham.vis import differentiable_marching_cubes, generate_icosphere

OUT_DIR = Path(__file__).parent / "visualizations"
OUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# 1. Mesh Eikonal on the sphere: great circles vs. zonal jet
# =============================================================================

print("Building icosphere mesh...")
verts, faces = generate_icosphere(radius=1.0, subdivisions=4)
mesh_adj = MeshAdjacency.build(verts, faces, num_ref_points=4)
print(f"  {verts.shape[0]} vertices, {faces.shape[0]} faces")

sphere = Sphere(intrinsic_dim=2, radius=1.0)


def h_net(x):
    return jnp.eye(3)


def w_net(x):
    # Zonal jet: rigid rotation about the z-axis (max speed 0.75 at equator)
    return 0.75 * jnp.array([-x[1], x[0], 0.0])


metric_calm = Randers(sphere, h_net, w_net, use_wind=False)
metric_wind = Randers(sphere, h_net, w_net, use_wind=True)

source_pt = jnp.array([[1.0, 0.0, 0.0]])
mesh_solver = MeshEikonalSolver(max_iters=50, tol=1e-6)

print("Solving mesh Eikonal (calm)...")
t0 = time.perf_counter()
T_calm = np.array(mesh_solver.solve(metric_calm, mesh_adj, verts, faces, source_pt))
print(f"  done in {time.perf_counter() - t0:.1f}s, T range [0, {T_calm.max():.2f}]")

print("Solving mesh Eikonal (zonal jet)...")
t0 = time.perf_counter()
T_wind = np.array(mesh_solver.solve(metric_wind, mesh_adj, verts, faces, source_pt))
print(f"  done in {time.perf_counter() - t0:.1f}s, T range [0, {T_wind.max():.2f}]")


def plot_mesh_arrival(ax, T_vertex, title, n_bands=14):
    """Render the sphere mesh with banded arrival-time face colors."""
    verts_np = np.asarray(verts)
    faces_np = np.asarray(faces)
    tris = verts_np[faces_np]  # (F, 3, 3)
    T_face = T_vertex[faces_np].mean(axis=1)

    # Discrete (banded) colormap makes the wavefronts readable on 3D surfaces
    cmap = plt.get_cmap("turbo", n_bands)
    colors = cmap(T_face / T_face.max())

    # Simple Lambertian shading for depth perception
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.3, 0.85])
    light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.clip(n @ light, 0, 1)[:, None]
    colors[:, :3] *= shade

    pc = Poly3DCollection(tris, facecolors=colors, edgecolors="none")
    ax.add_collection3d(pc)

    ax.scatter(*(1.02 * np.asarray(source_pt[0])), color="white",
               edgecolor="black", s=80, zorder=10, label="source")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(-1, 1)
    ax.set_axis_off()
    ax.set_title(title)
    # Camera centered on the source so the front shapes are comparable
    ax.view_init(elev=18, azim=8)
    return cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(0.0, float(T_face.max()))
    )


# =============================================================================
# 2. Volumetric Eikonal: slow lens + planar vortex wind
# =============================================================================

LENS_CENTER = jnp.array([0.35, 0.0, 0.0])
LENS_RADIUS = 0.35  # Gaussian length scale of the inclusion


def speed(x):
    """Propagation speed: 1 in free space, 0.45 inside the lens."""
    d2 = jnp.sum((x - LENS_CENTER) ** 2)
    return 1.0 - 0.55 * jnp.exp(-d2 / LENS_RADIUS**2)


def h_net_3d(x):
    return jnp.eye(3) / speed(x) ** 2


def w_net_3d(x):
    # Planar vortex around the z-axis (counter-clockwise)
    return 0.45 * jnp.array([-x[1], x[0], 0.0])


metric_3d = Randers(EuclideanSpace(3), h_net_3d, w_net_3d)

EXTENT = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
N = 49
SOURCE_3D = jnp.array([[-0.6, 0.0, 0.0]])

print(f"Solving volumetric Eikonal on a {N}^3 grid (lens + vortex)...")
t0 = time.perf_counter()
vol_solver = VolumetricEikonalSolver(max_iters=100, tol=1e-6)
T_vol, _, _ = vol_solver.solve(metric_3d, SOURCE_3D, EXTENT, (N, N, N))
T_vol.block_until_ready()
print(f"  done in {time.perf_counter() - t0:.1f}s, T range [0, {float(T_vol.max()):.2f}]")
T_np = np.asarray(T_vol)


def plot_volume_slice(ax):
    """Mid-plane z=0: filled arrival-time contours, wind field, lens outline."""
    k = N // 2
    axis = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    Tz = T_np[:, :, k]

    cf = ax.contourf(X, Y, Tz, levels=24, cmap="turbo")
    ax.contour(X, Y, Tz, levels=12, colors="white", linewidths=0.6, alpha=0.7)

    # Wind quiver (coarse)
    q = axis[::6]
    QX, QY = np.meshgrid(q, q, indexing="ij")
    ax.quiver(QX, QY, -0.45 * QY, 0.45 * QX, color="white", alpha=0.85,
              scale=12, width=0.004)

    lens = plt.Circle(np.asarray(LENS_CENTER[:2]), LENS_RADIUS, fill=False,
                      color="black", linestyle="--", linewidth=1.5,
                      label="slow lens (0.45x speed)")
    ax.add_patch(lens)
    ax.plot(*np.asarray(SOURCE_3D[0, :2]), "o", color="white",
            markeredgecolor="black", markersize=9, label="source")

    ax.set_aspect("equal")
    ax.set_title("Volumetric Eikonal — z=0 slice\n(vortex wind + refractive lens)")
    ax.set_xlabel("x"), ax.set_ylabel("y")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    return cf


def plot_wavefront_isosurfaces(ax, levels):
    """Nested arrival-time isosurfaces via differentiable marching cubes."""
    cmap = plt.get_cmap("turbo")
    t_max = max(levels)
    for t_iso, alpha in zip(levels, (0.55, 0.40, 0.22)):
        tris, valid = differentiable_marching_cubes(T_vol, float(t_iso), EXTENT)
        tris_np = np.asarray(tris).reshape(-1, 3, 3)
        mask = np.asarray(valid).reshape(-1)
        tri_sel = tris_np[mask]
        color = cmap(0.15 + 0.8 * t_iso / t_max)
        pc = Poly3DCollection(
            tri_sel, facecolors=color, edgecolors="none", alpha=alpha
        )
        ax.add_collection3d(pc)
        print(f"  isosurface T={t_iso:.2f}: {tri_sel.shape[0]} triangles")

    # Slow-lens outline (wireframe sphere)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 18), np.linspace(0, np.pi, 10))
    cx, cy, cz = np.asarray(LENS_CENTER)
    ax.plot_wireframe(
        cx + LENS_RADIUS * np.cos(u) * np.sin(v),
        cy + LENS_RADIUS * np.sin(u) * np.sin(v),
        cz + LENS_RADIUS * np.cos(v),
        color="gray", linewidth=0.5, alpha=0.5,
    )

    ax.scatter(*np.asarray(SOURCE_3D[0]), color="black", s=50, label="source")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(-1, 1)
    ax.set_xticks([-1, 0, 1]), ax.set_yticks([-1, 0, 1]), ax.set_zticks([-1, 0, 1])
    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    ax.set_title("Volumetric Eikonal — nested wavefronts\n(marching-cubes isosurfaces)")
    ax.view_init(elev=24, azim=-118)


# =============================================================================
# 3. Compose figure
# =============================================================================

print("Rendering figure...")
fig = plt.figure(figsize=(13, 11))

ax1 = fig.add_subplot(2, 2, 1, projection="3d")
sm1 = plot_mesh_arrival(ax1, T_calm, "Mesh Eikonal on $S^2$ — calm\n(concentric great-circle fronts)")
fig.colorbar(sm1, ax=ax1, shrink=0.6, label="arrival time T")

ax2 = fig.add_subplot(2, 2, 2, projection="3d")
sm2 = plot_mesh_arrival(ax2, T_wind, "Mesh Eikonal on $S^2$ — zonal jet\n(fronts skewed downwind)")
fig.colorbar(sm2, ax=ax2, shrink=0.6, label="arrival time T")

# Wind arrows on the equator for the jet panel
theta = np.linspace(0, 2 * np.pi, 18, endpoint=False)
eq = 1.04 * np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
wind = 0.25 * np.stack([-eq[:, 1], eq[:, 0], np.zeros_like(theta)], axis=1)
ax2.quiver(eq[:, 0], eq[:, 1], eq[:, 2], wind[:, 0], wind[:, 1], wind[:, 2],
           color="black", alpha=0.7, arrow_length_ratio=0.35)

ax3 = fig.add_subplot(2, 2, 3)
cf = plot_volume_slice(ax3)
fig.colorbar(cf, ax=ax3, shrink=0.85, label="arrival time T")

ax4 = fig.add_subplot(2, 2, 4, projection="3d")
t_levels = np.quantile(T_np, [0.04, 0.14, 0.32])
plot_wavefront_isosurfaces(ax4, list(t_levels))

fig.suptitle(
    "Anisotropic Eikonal solvers: arrival-time fronts under Randers (Zermelo) metrics",
    fontsize=14,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))

out_path = OUT_DIR / "eikonal_fronts.png"
fig.savefig(out_path, dpi=160)
print(f"Saved {out_path}")
plt.show()
