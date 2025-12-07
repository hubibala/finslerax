import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import jax
import jax.numpy as jnp

def setup_3d_plot():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    return fig, ax

def plot_sphere(ax, radius=1.0, alpha=0.1, color='gray'):
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = radius * np.cos(u)*np.sin(v)
    y = radius * np.sin(u)*np.sin(v)
    z = radius * np.cos(v)
    ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)

# --- FIXED FUNCTION ---
def plot_vector_field(ax, points, vectors, scale=0.2, color='blue', alpha=0.5, label=None, **kwargs):
    """
    Plots 3D quiver arrows. 
    Now accepts **kwargs to pass styling (linewidth, etc.) to ax.quiver.
    """
    p, v = np.array(points), np.array(vectors)
    if len(p) == 0: return
    
    ax.quiver(p[:,0], p[:,1], p[:,2], 
              v[:,0], v[:,1], v[:,2], 
              length=scale, color=color, alpha=alpha, label=label, **kwargs)
# ----------------------

def plot_trajectory(ax, traj, color='red', linewidth=2, linestyle='-', label=None):
    if hasattr(traj, 'xs'):
        xs = np.array(traj.xs)
    elif isinstance(traj, tuple):
        xs = np.array(traj[0])
    else:
        xs = np.array(traj)
    ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], color=color, linewidth=linewidth, linestyle=linestyle, label=label)
    ax.scatter(xs[0,0], xs[0,1], xs[0,2], color=color, s=20)

def plot_indicatrices(ax, metric, points, scale=0.15, n_theta=40, color='purple'):
    if len(points) == 0: return
    for p in points:
        n = p / (np.linalg.norm(p) + 1e-8)
        a = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([0., 1., 0.])
        e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        
        thetas = np.linspace(0, 2*np.pi, n_theta)
        us = np.outer(np.cos(thetas), e1) + np.outer(np.sin(thetas), e2)
        
        # JAX computation
        p_jax = jnp.array(p)
        us_jax = jnp.array(us)
        # Fix: Ensure metric_fn handles batching or use vmap inside here if metric doesn't
        # Safest is strictly map:
        costs = np.array([metric.metric_fn(p_jax, u) for u in us_jax])
        
        vs = us / (costs[:, None] + 1e-8)
        loop = p + scale * vs
        loop = np.vstack([loop, loop[0]])
        ax.plot(loop[:,0], loop[:,1], loop[:,2], color=color, alpha=0.8, linewidth=1.5)

def generate_icosphere(radius=1.0, subdivisions=3):
    t = (1.0 + 5.0**0.5) / 2.0
    verts = [[-1,t,0], [1,t,0], [-1,-t,0], [1,-t,0], [0,-1,t], [0,1,t], [0,-1,-t], [0,1,-t], [t,0,-1], [t,0,1], [-t,0,-1], [-t,0,1]]
    faces = [[0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11], [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8], [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9], [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1]]
    verts = np.array(verts); faces = np.array(faces)
    for _ in range(subdivisions):
        new_faces = []
        new_verts = verts.tolist()
        midpoint_cache = {}
        def get_midpoint(i1, i2):
            key = tuple(sorted((i1, i2)))
            if key in midpoint_cache: return midpoint_cache[key]
            v1, v2 = np.array(new_verts[i1]), np.array(new_verts[i2])
            new_verts.append(((v1+v2)/2.0).tolist())
            return midpoint_cache.setdefault(key, len(new_verts)-1)
        for v0, v1, v2 in faces:
            a, b, c = get_midpoint(v0, v1), get_midpoint(v1, v2), get_midpoint(v2, v0)
            new_faces.extend([[v0, a, c], [v1, b, a], [v2, c, b], [a, b, c]])
        verts = np.array(new_verts); faces = np.array(new_faces)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * radius
    return jnp.array(verts), jnp.array(faces)