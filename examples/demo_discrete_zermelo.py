import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

from ham.geometry import Sphere, Randers, TriangularMesh, DiscreteRanders
from ham.solvers import AVBDSolver
from ham.vis import setup_3d_plot, plot_sphere, plot_trajectory, generate_icosphere, plot_indicatrices

# --- 1. Continuous Physics ---
radius = 1.0
sphere_cont = Sphere(radius)

# Wind: Rotational around Z (Equatorial Trade Winds)
def w_net(x): 
    base = jnp.array([-x[1], x[0], 0.0])
    return 0.8 * base

def h_net(x): return jnp.eye(3)

metric_randers = Randers(sphere_cont, h_net, w_net)
metric_riem = Randers(sphere_cont, h_net, lambda x: jnp.zeros(3))

# --- 2. Mission: South -> North ---
start = jnp.array([0.0, 1.0, 0.0])
end   = jnp.array([0.0,  0.0, 1.0])

solver = AVBDSolver(step_size=0.05, beta=10.0, iterations=500, tol=1e-6)

print("Solving Riemannian (Great Circle)...")
traj_riem = solver.solve(metric_riem, start, end, n_steps=40)

print("Solving Randers (Continuous)...")
traj_rand = solver.solve(metric_randers, start, end, n_steps=40)

batch_energy = jax.vmap(metric_randers.energy)
e_riem = batch_energy(traj_riem.xs[:-1], traj_riem.vs).sum()
e_rand = batch_energy(traj_rand.xs[:-1], traj_rand.vs).sum()

print(f"Energy Riemannian path: {e_riem:.4f}")
print(f"Energy Randers path:    {e_rand:.4f}")

# --- 3. Discrete Physics ---
print("Generating Icosphere...")
verts, faces = generate_icosphere(radius=1.0, subdivisions=1)
mesh_discrete = TriangularMesh(verts, faces)

print("Sampling Wind field onto Mesh Faces...")
face_centers = jnp.mean(verts[faces], axis=1) # (F, 3)
face_winds = jax.vmap(w_net)(face_centers)

metric_discrete_randers = DiscreteRanders(mesh_discrete, face_winds)

print("Solving Discrete Finsler Geodesic...")
traj_mesh = solver.solve(metric_discrete_randers, start, end, n_steps=40)
e_mesh = batch_energy(traj_mesh.xs[:-1], traj_mesh.vs).sum()
print(f"Energy Discrete Mesh path: {e_mesh:.4f}")

# --- 4. Visualization ---
fig, ax = setup_3d_plot()
ax.set_title(f"Zermelo S^2: Discrete Matches Continuous\\nEnergy: {e_rand:.2f} (Cont) vs {e_mesh:.2f} (Disc)")

plot_sphere(ax, alpha=0.05)

# Wind Vectors
theta = np.linspace(0, 2*np.pi, 20)
equator_pts = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
wind_vecs = np.array(jax.vmap(w_net)(jnp.array(equator_pts)))
ax.quiver(equator_pts[:,0], equator_pts[:,1], equator_pts[:,2], 
          wind_vecs[:,0], wind_vecs[:,1], wind_vecs[:,2], 
          length=0.2, color='blue', alpha=0.6, label='Wind')

# Paths
plot_trajectory(ax, traj_riem, color='gray', linestyle='--', label='Riemannian (Great Circle)')
plot_trajectory(ax, traj_rand, color='green', linewidth=4, label='Randers (Continuous)')
plot_trajectory(ax, traj_mesh, color='orange', linestyle=':', linewidth=2, label='Randers (Discrete)')

# Indicatrices
ind_pts = traj_rand.xs[::6]
plot_indicatrices(ax, metric_randers, ind_pts, scale=0.15, color='purple')

ax.legend()
plt.savefig("zermelo_demo.png")
plt.show()