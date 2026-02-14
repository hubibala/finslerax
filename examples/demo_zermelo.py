import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

from ham.geometry import Sphere, Randers, TriangularMesh
from ham.geometry.zoo import Euclidean
from ham.solvers import AVBDSolver
from ham.vis import setup_3d_plot, plot_sphere, plot_trajectory, generate_icosphere, plot_indicatrices

# --- 1. Define the Physics ---
radius = 1.0
sphere_cont = Sphere(radius)

# Wind: Strong rotation around Z-axis (Equatorial Trade Winds)
# Strength 0.8 at equator. Counter-Clockwise.
def w_net(x): 
    base = jnp.array([-x[1], x[0], 0.0])
    return 0.9 * base

def h_net(x): return jnp.eye(3)

metric_randers = Randers(sphere_cont, h_net, w_net)
metric_riem = Randers(sphere_cont, h_net, lambda x: jnp.zeros(3))

# --- 2. Mission: South -> North ---
# The solver will now perturb this slightly to break the symmetry
start = jnp.array([1.0, 0.0, 0.0])
end   = jnp.array([0.0,  0.0, 1.0])

# --- 3. Solve (Continuous) ---
solver = AVBDSolver(step_size=0.05, beta=10.0, iterations=200, tol=1e-6)

print("Solving Riemannian (Great Circle)...")
traj_riem = solver.solve(metric_riem, start, end, n_steps=40)

print("Solving Randers (Zermelo)...")
traj_rand = solver.solve(metric_randers, start, end, n_steps=40)

# Calculate Energies using VMAP (Fixes dimensions error)
batch_energy = jax.vmap(metric_randers.energy)
e_riem = batch_energy(traj_riem.xs[:-1], traj_riem.vs).sum()
e_rand = batch_energy(traj_rand.xs[:-1], traj_rand.vs).sum()

print(f"Energy Riemannian path: {e_riem:.4f}")
print(f"Energy Randers path:    {e_rand:.4f}")

# --- 4. Solve (Discrete Mesh) ---
# High-Res Icosphere (Subdivision=3 -> ~1280 faces)
verts, faces = generate_icosphere(radius=1.0, subdivisions=3)
mesh_discrete = TriangularMesh(verts, faces)
metric_mesh = Euclidean(mesh_discrete) # Isotropic benchmark

print("Solving Discrete Mesh Geodesic...")
traj_mesh = solver.solve(metric_mesh, start, end, n_steps=40)

# --- 5. Visualization ---
fig, ax = setup_3d_plot()
ax.set_title(f"Zermelo Navigation S^2\nRanders Energy: {e_rand:.2f} vs Riem: {e_riem:.2f}")

plot_sphere(ax, alpha=0.05)

# Plot Wind Vectors
theta = np.linspace(0, 2*np.pi, 20)
equator_pts = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
wind_vecs = np.array(jax.vmap(w_net)(jnp.array(equator_pts)))
ax.quiver(equator_pts[:,0], equator_pts[:,1], equator_pts[:,2], 
          wind_vecs[:,0], wind_vecs[:,1], wind_vecs[:,2], 
          length=0.2, color='blue', alpha=0.6, label='Wind')

# Plot Paths
plot_trajectory(ax, traj_riem, color='gray', linestyle='--', label='Riemannian (Great Circle)')
plot_trajectory(ax, traj_rand, color='green', linewidth=3, label='Randers (Wind Optimized)')
plot_trajectory(ax, traj_mesh, color='orange', linestyle=':', linewidth=2, label='Discrete Mesh Path')

# Plot Indicatrices
ind_pts = traj_rand.xs[::6]
plot_indicatrices(ax, metric_randers, ind_pts, scale=0.15, color='purple')

ax.legend()
plt.show()