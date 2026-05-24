import json
import os

def write_nb(filepath, title, cells_data):
    cells = []
    
    # Intro markdown
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "This notebook provides a detailed, mathematically rigorous walkthrough of the **Holonomic Association Model (HAM)**.\n",
            "The HAM library treats differentiable **Finsler Geometries** as learnable modules in JAX, allowing us to solve optimal control and simulation problems purely through geometric dynamics.\n\n",
            "### Core Philosophy: Metric-First Design\n",
            "In HAM, the `FinslerMetric` object is the single source of truth. We define the scalar **Energy Functional**:\n",
            "$$\\mathcal{E}[\\gamma] = \\int \\frac{1}{2} F^2(x, \\dot{x}) dt$$\n",
            "where $F(x,v)$ is the Finsler cost function. Everything else—including the Geodesic Spray (the physics engine), curvature, and Berwald parallel transport—is auto-differentiated directly from $F$ using `jax.grad` and `jax.hessian`, entirely avoiding the $O(N^3)$ computational cost of expanding Christoffel symbols manually."
        ]
    })
    
    for c_type, content in cells_data:
        if c_type == "markdown":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in content.split("\n")]
            })
        elif c_type == "code":
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in content.split("\n")]
            })
            
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filepath, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"Created {filepath}")

# ==========================================
# 1. Zermelo Notebook
# ==========================================
zermelo_cells = [
    ("markdown", "## 1. Setup and Imports\nWe begin by importing JAX and the HAM library. HAM provides geometric primitives such as topological constraints (`Sphere`) and specific metric classes (`Euclidean`, `Randers`). We also import Plotly for interactive 3D rendering."),
    ("code", "import jax.numpy as jnp\nimport jax\nimport plotly.graph_objects as go\nimport numpy as np\n\nfrom ham.geometry import Sphere, Randers, TriangularMesh\nfrom ham.geometry import Euclidean\nfrom ham.solvers import AVBDSolver\nfrom ham.vis import generate_icosphere"),
    ("markdown", "## 2. Zermelo's Navigation Problem (The Randers Metric)\nTo model asymmetric travel costs (e.g. wind, ocean currents, or chemical gradients), HAM uses the **Randers Metric**. This is derived from Zermelo's Navigation Problem, which asks for the time-optimal path across a Riemannian manifold (a \"sea\" defined by $h_{ij}(x)$) subject to a background vector field (a \"wind\" $W^i(x)$).\n\nThe resulting Finsler metric is given by the exact formula:\n$$ F(x, v) = \\frac{\\sqrt{\\lambda \\|v\\|_h^2 + \\langle W, v \\rangle_h^2} - \\langle W, v \\rangle_h}{\\lambda} $$\nwhere $\\lambda = 1 - \\|W\\|_h^2$. \n\nHAM strictly enforces the causality constraint $\\|W\\|_h < 1$ (the wind cannot be faster than the agent's maximum speed).\n\nBelow, we define an equatorial trade wind system rotating around the Z-axis."),
    ("code", "# Radius of the continuous manifold\nradius = 1.0\nsphere_cont = Sphere(radius)\n\n# Wind: Strong rotation around Z-axis\n# Strength 0.8 at equator. Counter-Clockwise.\ndef w_net(x): \n    base = jnp.array([-x[1], x[0], 0.0])\n    return 0.9 * base\n\ndef h_net(x): \n    return jnp.eye(3)\n\nmetric_randers = Randers(sphere_cont, h_net, w_net)\nmetric_riem = Randers(sphere_cont, h_net, lambda x: jnp.zeros(3))"),
    ("markdown", "## 3. Mission: South -> North\nWe set our start point at the equator and our target destination at the north pole. We aim to compute the geodesic path bridging these points."),
    ("code", "start = jnp.array([1.0, 0.0, 0.0])\nend   = jnp.array([0.0,  0.0, 1.0])"),
    ("markdown", "## 4. Solving the Boundary Value Problem (AVBD)\nTo find the geodesics, we use HAM's **Augmented Vertex Block Descent (AVBD)** solver. \nUnlike an Initial Value Problem solver (which shoots a particle and hopes it hits the target), the AVBD solver formulates a Boundary Value Problem (BVP). It optimizes discrete path vertices $x_0, \\dots, x_N$ directly to minimize the global energy $\\sum E(x_i, v_i)$ subject to topological manifold constraints.\n\n- **Riemannian**: Will yield the standard shortest distance path (a Great Circle).\n- **Randers**: Will yield the fastest path taking advantage of the wind."),
    ("code", "solver = AVBDSolver(step_size=0.05, beta=10.0, iterations=200, tol=1e-6)\n\nprint(\"Solving Riemannian (Great Circle)...\")\ntraj_riem = solver.solve(metric_riem, start, end, n_steps=40)\n\nprint(\"Solving Randers (Zermelo)...\")\ntraj_rand = solver.solve(metric_randers, start, end, n_steps=40)\n\n# Calculate Energies\nbatch_energy = jax.vmap(metric_randers.energy)\ne_riem = batch_energy(traj_riem.xs[:-1], traj_riem.vs).sum()\ne_rand = batch_energy(traj_rand.xs[:-1], traj_rand.vs).sum()\n\nprint(f\"Energy Riemannian path: {e_riem:.4f}\")\nprint(f\"Energy Randers path:    {e_rand:.4f}\")"),
    ("markdown", "## 5. Solving on a Discrete Mesh\nHAM bridges continuous mathematics with discrete geometry. Here we build an icosphere mesh (`TriangularMesh`) and solve for the discrete geodesic path natively over the facets."),
    ("code", "# High-Res Icosphere (Subdivision=3 -> ~1280 faces)\nverts, faces = generate_icosphere(radius=1.0, subdivisions=3)\nmesh_discrete = TriangularMesh(verts, faces)\nmetric_mesh = Euclidean(mesh_discrete) # Isotropic benchmark\n\nprint(\"Solving Discrete Mesh Geodesic...\")\ntraj_mesh = solver.solve(metric_mesh, start, end, n_steps=40)"),
    ("markdown", "## 6. Interactive 3D Visualization\nWe use Plotly to render an interactive 3D visualization. We plot the vector field, the continuous Riemannian path, the continuous Randers path, and the discrete mesh path. You can rotate and zoom to inspect how the Randers path veers off the shortest-distance geodesic to exploit the wind."),
    ("code", "fig = go.Figure()\n\n# Plot transparent sphere (Mesh3d)\nfig.add_trace(go.Mesh3d(\n    x=verts[:,0], y=verts[:,1], z=verts[:,2],\n    i=faces[:,0], j=faces[:,1], k=faces[:,2],\n    color='lightblue', opacity=0.3, alphahull=0, name='Manifold'\n))\n\n# Plot Wind Vectors (Cones)\ntheta = np.linspace(0, 2*np.pi, 20)\nequator_pts = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)\nwind_vecs = np.array(jax.vmap(w_net)(jnp.array(equator_pts)))\n\nfig.add_trace(go.Cone(\n    x=equator_pts[:,0], y=equator_pts[:,1], z=equator_pts[:,2],\n    u=wind_vecs[:,0], v=wind_vecs[:,1], w=wind_vecs[:,2],\n    sizemode='absolute', sizeref=0.2, colorscale='Blues', showscale=False, name='Wind'\n))\n\n# Plot Paths\nfig.add_trace(go.Scatter3d(\n    x=traj_riem.xs[:,0], y=traj_riem.xs[:,1], z=traj_riem.xs[:,2],\n    mode='lines', line=dict(color='gray', width=4, dash='dash'), name='Riemannian (Great Circle)'\n))\nfig.add_trace(go.Scatter3d(\n    x=traj_rand.xs[:,0], y=traj_rand.xs[:,1], z=traj_rand.xs[:,2],\n    mode='lines', line=dict(color='green', width=6), name='Randers (Wind Optimized)'\n))\nfig.add_trace(go.Scatter3d(\n    x=traj_mesh.xs[:,0], y=traj_mesh.xs[:,1], z=traj_mesh.xs[:,2],\n    mode='lines', line=dict(color='orange', width=4, dash='dot'), name='Discrete Mesh Path'\n))\n\nfig.update_layout(\n    title=f\"Zermelo Navigation S^2 | Energy: Randers={e_rand:.2f}, Riem={e_riem:.2f}\", \n    scene=dict(aspectmode='data')\n)\nfig.show()")
]

# ==========================================
# 2. Vortex Notebook
# ==========================================
vortex_cells = [
    ("markdown", "## 1. Setup and Imports\nThe `demo_vortex` notebook illustrates the resilience of the **Augmented Vertex Block Descent (AVBD)** solver when faced with extreme, non-linear geometric sprays. We use Plotly for interactive 3D rendering."),
    ("code", "from ham.utils.config import DEFAULT_JNP_DTYPE, DEFAULT_NP_DTYPE\nimport jax\nimport jax.numpy as jnp\nimport plotly.graph_objects as go\nimport numpy as np\nfrom jax import config\n\nfrom ham.geometry import Sphere, Randers\nfrom ham.solvers import AVBDSolver\nfrom ham.vis import generate_icosphere"),
    ("markdown", "## 2. Modeling the Vortex Field\nWe define a highly localized rotational vector field with exponential decay. The extreme gradients in this wind field test the numerical stability of the implicit dynamics underlying the AVBD optimization."),
    ("code", "def vortex_field(center, strength=1.0, decay=2.0):\n    center = center / jnp.linalg.norm(center)\n    def flow(x):\n        cos_dist = jnp.dot(x, center)\n        dist = jnp.arccos(jnp.clip(cos_dist, -1.0, 1.0))\n        v_rot = jnp.cross(center, x)\n        magnitude = strength * jnp.exp(-decay * (dist**2))\n        return magnitude * v_rot\n    return flow"),
    ("markdown", "## 3. Environment Definition\nWe construct the `Randers` metric using Zermelo's Navigation formula. Notice the incredibly high `strength=10.0` which forces extreme path curvature in order to minimize travel time through the heavy headwind."),
    ("code", "sphere = Sphere(radius=1.0)\nvortex_center = jnp.array([0.0, 1.0, 0.0])\nw_net = vortex_field(vortex_center, strength=10.0, decay=5.0)\nh_net = lambda x: jnp.eye(3)\nmetric = Randers(sphere, h_net, w_net)\n\nstart = jnp.array([1.0, 0.0, 0.0])\nend   = jnp.array([-0.99, 0.1, 0.0]) \nend   = end / jnp.linalg.norm(end)"),
    ("markdown", "## 4. Solver Stiffness Comparison (`beta` parameter)\nThe AVBD solver incorporates a `beta` penalty that controls block descent stiffness. \n- **High Beta (Stiff)**: Quickly freezes the path, behaving greedily and potentially getting trapped in local topological minima.\n- **Low Beta (Relaxed)**: Allows massive lateral relaxation along the manifold to explore topologically distinct routes around the vortex core."),
    ("code", "print(\"Solving 'Stiff' (High Beta)...\")\nsolver_stiff = AVBDSolver(step_size=0.05, beta=20.0, iterations=100)\ntraj_stiff = solver_stiff.solve(metric, start, end, n_steps=40)\n\nprint(\"Solving 'Relaxed' (Low Beta)...\")\nsolver_relaxed = AVBDSolver(step_size=0.05, beta=0.5, iterations=100) \ntraj_relaxed = solver_relaxed.solve(metric, start, end, n_steps=40)"),
    ("markdown", "## 5. Visualizing the Vortex Escape in 3D\nThe interactive visualization clearly shows the relaxed AVBD solver dynamically shifting the vertices globally to discover a much more efficient topological route (the 'D' shape) avoiding the worst of the headwind. Rotate the render to see how the red path swoops around the vortex core!"),
    ("code", "fig = go.Figure()\n\npts, faces = generate_icosphere(radius=1.0, subdivisions=3)\nwind_vecs = np.array(jax.vmap(w_net)(pts))\n\n# Plot Manifold\nfig.add_trace(go.Mesh3d(\n    x=pts[:,0], y=pts[:,1], z=pts[:,2],\n    i=faces[:,0], j=faces[:,1], k=faces[:,2],\n    color='lightblue', opacity=0.2, alphahull=0, name='Sphere'\n))\n\n# Plot Vortex Vectors\n# Subsample to keep the plot clean\nmask = np.linalg.norm(wind_vecs, axis=1) > 0.5\nfig.add_trace(go.Cone(\n    x=pts[mask,0], y=pts[mask,1], z=pts[mask,2],\n    u=wind_vecs[mask,0], v=wind_vecs[mask,1], w=wind_vecs[mask,2],\n    sizemode='absolute', sizeref=0.3, colorscale='Viridis', showscale=True, name='Vortex'\n))\n\n# Plot Paths\nfig.add_trace(go.Scatter3d(\n    x=traj_stiff.xs[:,0], y=traj_stiff.xs[:,1], z=traj_stiff.xs[:,2],\n    mode='lines+markers', line=dict(color='gray', width=4, dash='dash'), \n    marker=dict(size=3), name='Stiff Solver (beta=20)'\n))\nfig.add_trace(go.Scatter3d(\n    x=traj_relaxed.xs[:,0], y=traj_relaxed.xs[:,1], z=traj_relaxed.xs[:,2],\n    mode='lines+markers', line=dict(color='red', width=6), \n    marker=dict(size=3), name='Relaxed Solver (beta=0.5)'\n))\n\nfig.update_layout(title=\"AVBD Solver Stiffness Comparison (Rotate to explore!)\", scene=dict(aspectmode='data'))\nfig.show()")
]

# ==========================================
# 3. Learned Wind Notebook
# ==========================================
learned_cells = [
    ("markdown", "## 1. Setup and Inverse Problem Definition\nOne of the Holonomic Association Model's most powerful features is its differentiable architecture. Because the metrics and solvers (like AVBD) are fully written in JAX, we can compute gradients *through* the optimal path-finding process.\n\nThis enables us to solve **inverse geometric problems**: if we observe agents taking specific optimal paths, can we deduce the underlying vector field (wind) they are experiencing?"),
    ("code", "import jax.numpy as jnp\nimport jax\nimport optax\nimport plotly.graph_objects as go\nimport numpy as np\nfrom jax import config\n\nfrom ham.geometry import Sphere, Randers\nfrom ham.solvers import AVBDSolver\nfrom ham.vis import generate_icosphere"),
    ("markdown", "## 2. Generating Ground Truth Observations\nWe establish a ground truth wind field (a vortex) and generate 4 trajectory observations. We assume we know the start and end points of these trajectories, but we *don't* know the wind field generating the curvature."),
    ("code", "def vortex_field(x): \n    center = jnp.array([0.0, 1.0, 0.0])\n    return 0.8 * jnp.cross(center, x)\n\nsphere = Sphere(radius=1.0)\ntrue_metric = Randers(sphere, lambda x: jnp.eye(3), vortex_field)\nsolver = AVBDSolver(step_size=0.05, beta=5.0, iterations=100)\n\n# True Trajectories (Observations)\nstarts = [jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0]), \n          jnp.array([-1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, -1.0])]\nends   = [jnp.array([0.0, 0.0, -1.0]), jnp.array([1.0, 0.0, 0.0]), \n          jnp.array([0.0, 0.0, 1.0]), jnp.array([-1.0, 0.0, 0.0])]\n\ntrue_trajs = [solver.solve(true_metric, s, e, n_steps=20) for s, e in zip(starts, ends)]"),
    ("markdown", "## 3. Parameterizing the Unknown Vector Field\nWe parameterize our learned wind field with a simple 3D vector parameter `w_param`. \nOur Finsler metric function is dynamically reconstructed on every forward pass using this trainable parameter."),
    ("code", "w_param = jnp.array([0.0, 0.0, 0.0]) # Initial guess: No wind\n\ndef make_metric(w):\n    # Reconstruct the wind field from the learned vector\n    def learned_wind(x): return jnp.cross(w, x)\n    return Randers(sphere, lambda x: jnp.eye(3), learned_wind)"),
    ("markdown", "## 4. Differentiable Loss Function\nWe define an L2 loss between the vertices of our currently predicted paths and the ground truth paths. \nBecause `AVBDSolver.solve` utilizes continuous gradient descents that are differentiable themselves, we can invoke `jax.value_and_grad` on this loss function directly!"),
    ("code", "@jax.jit\ndef loss_fn(w):\n    metric = make_metric(w)\n    loss = 0.0\n    for i, (s, e) in enumerate(zip(starts, ends)):\n        pred_traj = solver.solve(metric, s, e, n_steps=20)\n        loss += jnp.mean((pred_traj.xs - true_trajs[i].xs)**2)\n    return loss\n\nloss_and_grad = jax.value_and_grad(loss_fn)"),
    ("markdown", "## 5. Gradient Descent (Learning the Wind)\nWe use the Adam optimizer from `optax` to iteratively backpropagate through the AVBD solver, adjusting the underlying Zermelo wind parameter to minimize the path discrepancies."),
    ("code", "optimizer = optax.adam(learning_rate=0.1)\nopt_state = optimizer.init(w_param)\n\nprint(\"Learning the wind field...\")\nfor step in range(30):\n    loss, grads = loss_and_grad(w_param)\n    updates, opt_state = optimizer.update(grads, opt_state)\n    w_param = optax.apply_updates(w_param, updates)\n    if step % 5 == 0:\n        print(f\"Step {step:02d} | Loss: {loss:.4f} | Learned Wind Param: {w_param}\")\n\nprint(f\"\\nTrue Wind Param: [0.0, 1.0, 0.0] * 0.8 = [0.0, 0.8, 0.0]\")\nprint(f\"Learned Param:   {w_param}\")"),
    ("markdown", "## 6. Interactive 3D Visualization\nWe use Plotly to render the true data vs the learned paths, alongside the inferred wind field. Rotate the interactive render to verify that our `w_param` vector successfully induces the expected rotational drift!"),
    ("code", "learned_metric = make_metric(w_param)\nlearned_trajs = [solver.solve(learned_metric, s, e, n_steps=20) for s, e in zip(starts, ends)]\n\nfig = go.Figure()\npts, faces = generate_icosphere(radius=1.0, subdivisions=2)\n\n# Plot sphere\nfig.add_trace(go.Mesh3d(\n    x=pts[:,0], y=pts[:,1], z=pts[:,2],\n    i=faces[:,0], j=faces[:,1], k=faces[:,2],\n    color='white', opacity=0.2, alphahull=0\n))\n\n# Plot Learned Wind\nlearned_vecs = np.array(jax.vmap(learned_metric.drift_fn)(pts))\nfig.add_trace(go.Cone(\n    x=pts[:,0], y=pts[:,1], z=pts[:,2],\n    u=learned_vecs[:,0], v=learned_vecs[:,1], w=learned_vecs[:,2],\n    sizemode='absolute', sizeref=0.3, colorscale='Oranges', showscale=False, name='Learned Wind'\n))\n\n# Plot True vs Learned Paths\nfor i, (t_true, t_learn) in enumerate(zip(true_trajs, learned_trajs)):\n    fig.add_trace(go.Scatter3d(\n        x=t_true.xs[:,0], y=t_true.xs[:,1], z=t_true.xs[:,2],\n        mode='lines', line=dict(color='black', width=6, dash='dash'), \n        name='True Data' if i==0 else showlegend=False\n    ))\n    fig.add_trace(go.Scatter3d(\n        x=t_learn.xs[:,0], y=t_learn.xs[:,1], z=t_learn.xs[:,2],\n        mode='lines', line=dict(color='red', width=4), \n        name='Learned' if i==0 else showlegend=False\n    ))\n\nfig.update_layout(title=\"Inverse Design: Recovering Zermelo Wind from Trajectories\", scene=dict(aspectmode='data'))\nfig.show()")
]

# ==========================================
# 4. Discrete Zermelo Notebook
# ==========================================
discrete_cells = [
    ("markdown", "## 1. Discrete Manifolds in HAM\nThe Holonomic Association Model doesn't strictly require continuous, analytically-defined manifolds. It fully supports real-world topologies via the `TriangularMesh` class. \nThis allows us to learn Finslerian costs and perform optimal navigation on discrete LiDAR point clouds or triangulated CAD models."),
    ("code", "import jax.numpy as jnp\nimport jax\nimport plotly.graph_objects as go\nimport numpy as np\n\nfrom ham.geometry import TriangularMesh, Randers, Euclidean\nfrom ham.solvers import AVBDSolver\nfrom ham.vis import generate_icosphere"),
    ("markdown", "## 2. Generating a Discrete Environment\nWe generate a discrete mesh representing a sphere (an icosphere). Notice we are no longer using the `Sphere` continuous topology class."),
    ("code", "print(\"Generating Discrete Mesh...\")\nverts, faces = generate_icosphere(radius=1.0, subdivisions=3)\nmesh = TriangularMesh(verts, faces)"),
    ("markdown", "## 3. Defining Discrete Wind\nOn a discrete mesh, the wind vector field $W^i(x)$ is defined globally as a function of the 3D position in the embedding space. The `DiscreteRanders` metric automatically projects these 3D vectors onto the local tangent planes of the individual faces to maintain topological validity."),
    ("code", "def w_net(x): \n    base = jnp.array([-x[1], x[0], 0.0])\n    return 0.8 * base\n\ndef h_net(x): return jnp.eye(3)\n\ndiscrete_randers = Randers(mesh, h_net, w_net)\ndiscrete_euclidean = Euclidean(mesh)"),
    ("markdown", "## 4. Discrete Trajectory Optimization\nWe define start and end points and rely on the same `AVBDSolver`. Because the metric uses the `TriangularMesh` backend, the Augmented Vertex Block Descent automatically transitions to calculating gradients over the discrete face topology, ensuring paths navigate edges safely without penetrating the surface!"),
    ("code", "start = jnp.array([1.0, 0.0, 0.0])\nend   = jnp.array([0.0, 0.0, 1.0])\n\nsolver = AVBDSolver(step_size=0.05, beta=10.0, iterations=300)\n\nprint(\"Solving Discrete Riemannian (Shortest Path)...\")\ntraj_eucl = solver.solve(discrete_euclidean, start, end, n_steps=40)\n\nprint(\"Solving Discrete Randers (Fastest Path with Wind)...\")\ntraj_rand = solver.solve(discrete_randers, start, end, n_steps=40)"),
    ("markdown", "## 5. Interactive 3D Mesh Visualization\nWe use Plotly's `Mesh3d` to render the wireframe faceted surface. This interactive view allows you to verify that the discrete solver successfully mapped the geodesics directly across the mesh edges rather than penetrating the interior of the shape. Rotate the render to see the wind flow and the topology!"),
    ("code", "fig = go.Figure()\n\n# Plot wireframe mesh using Plotly\n# To get a wireframe effect with Mesh3d, we can plot the edges or just use low opacity\nfig.add_trace(go.Mesh3d(\n    x=verts[:,0], y=verts[:,1], z=verts[:,2],\n    i=faces[:,0], j=faces[:,1], k=faces[:,2],\n    color='lightgray', opacity=0.8, alphahull=0,\n    lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.5, specular=0.2),\n    name='Discrete Mesh'\n))\n\n# Plot Trajectories\nfig.add_trace(go.Scatter3d(\n    x=traj_eucl.xs[:,0], y=traj_eucl.xs[:,1], z=traj_eucl.xs[:,2],\n    mode='lines', line=dict(color='black', width=5, dash='dash'), name='Discrete Euclidean'\n))\nfig.add_trace(go.Scatter3d(\n    x=traj_rand.xs[:,0], y=traj_rand.xs[:,1], z=traj_rand.xs[:,2],\n    mode='lines', line=dict(color='blue', width=6), name='Discrete Randers (Wind Optimized)'\n))\n\n# Wind visualization\ntheta = np.linspace(0, 2*np.pi, 20)\nequator_pts = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)\nwind_vecs = np.array(jax.vmap(w_net)(jnp.array(equator_pts)))\n\nfig.add_trace(go.Cone(\n    x=equator_pts[:,0], y=equator_pts[:,1], z=equator_pts[:,2],\n    u=wind_vecs[:,0], v=wind_vecs[:,1], w=wind_vecs[:,2],\n    sizemode='absolute', sizeref=0.3, colorscale='Blues', showscale=False, name='Wind'\n))\n\nfig.update_layout(title=\"Discrete Zermelo Navigation on a Mesh\", scene=dict(aspectmode='data'))\nfig.show()")
]

os.makedirs("examples/notebooks", exist_ok=True)
write_nb("examples/notebooks/demo_zermelo.ipynb", "Zermelo Navigation (Continuous)", zermelo_cells)
write_nb("examples/notebooks/demo_vortex.ipynb", "Vortex Field Solver Resilience", vortex_cells)
write_nb("examples/notebooks/demo_learned_wind.ipynb", "Inverse Design: Learning the Wind", learned_cells)
write_nb("examples/notebooks/demo_discrete_zermelo.ipynb", "Discrete Manifold Navigation", discrete_cells)
