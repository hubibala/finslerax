import sys
import re

with open("experiments/wildfire/eikonal/experiment_eikonal_mesh.py", "r") as f:
    text = f.read()

# Replace config variables
text = re.sub(r"avbd_n_steps=15 if quick else 30,", "eikonal_iters=50,", text)
text = re.sub(r"avbd_iters=30 if quick else 50,", "", text)

# Replace imports
text = text.replace("from ham.solvers.avbd import AVBDSolver", "from ham.solvers.mesh_eikonal import MeshEikonalSolver")
text = text.replace("from ham.training.losses import ArrivalTimeLoss", "from ham.training.losses import curriculum_alpha")

# Update make_solver
text = text.replace("def make_solver(cfg: dict) -> AVBDSolver:", "def make_solver(cfg: dict) -> MeshEikonalSolver:")
text = text.replace("AVBDSolver(", "MeshEikonalSolver(")
text = re.sub(r"step_size=0.05,.*implicit_diff=True,", "max_iters=cfg[\"eikonal_iters\"], tol=1e-4", text, flags=re.DOTALL)

# Add compute_sparse_mesh_eikonal_loss
loss_fn = """def compute_sparse_mesh_eikonal_loss(
    metric,
    solver,
    mesh,
    source_world: jax.Array,
    x_obs_world: jax.Array,
    t_obs: jax.Array,
):
    source_coords = jnp.expand_dims(source_world, axis=0)
    T_all = solver.solve(metric, mesh.adjacency, mesh.vertices, mesh.faces, source_coords)
    
    def get_closest_v(src):
        return jnp.argmin(jnp.sum((mesh.vertices - src)**2, axis=-1))
        
    closest_vs = jax.vmap(get_closest_v)(x_obs_world)
    t_pred = T_all[closest_vs]
    return jnp.mean((t_pred - t_obs)**2)
"""

# Insert loss function
text = text.replace("def _val_pearson_r_mesh(", loss_fn + "\n\ndef _val_pearson_r_mesh(")

# Update _val_pearson_r_mesh signature
text = text.replace("solver: AVBDSolver,", "solver: MeshEikonalSolver,")

# Update _predict_arrivals_mesh_chunked
predict_fn = """def _predict_arrivals_mesh_chunked(
    bound_metric,
    solver,
    mesh,
    source_3d: jax.Array,
    eval_pixels: np.ndarray,
    elev_raster: np.ndarray,
    pixel_spacing_m: float,
):
    source_coords = jnp.expand_dims(source_3d, axis=0)
    T_all = solver.solve(bound_metric, mesh.adjacency, mesh.vertices, mesh.faces, source_coords)
    
    eval_3d = _pixels_to_world_3d(eval_pixels, elev_raster, pixel_spacing_m)
    
    def get_closest_v(src):
        return jnp.argmin(jnp.sum((mesh.vertices - src)**2, axis=-1))
        
    closest_vs = jax.vmap(get_closest_v)(eval_3d)
    return np.array(T_all[closest_vs], dtype=np.float32)
"""
text = re.sub(r"def _predict_arrivals_mesh_chunked\(.*?\n\)\s*->\s*np\.ndarray:(?:\n(?:[ \t]+.*|\s*))*?\n\s+return np\.array\(all_pred, dtype=np\.float32\)", predict_fn, text, flags=re.DOTALL)

# Update _val_pearson_r_mesh body
text = text.replace("pred = _predict_arrivals_mesh_chunked(\n        bound, solver, source_3d,\n        scenario.obs_pixels, scenario.elev_raster,\n        scenario.pixel_spacing_m, cfg[\"avbd_n_steps\"],\n    )", "pred = _predict_arrivals_mesh_chunked(\n        bound, solver, mesh, source_3d,\n        scenario.obs_pixels, scenario.elev_raster,\n        scenario.pixel_spacing_m\n    )")

# Update test evaluation
text = text.replace("pred = _predict_arrivals_mesh_chunked(\n            bound, solver, source_3d, eval_pix,\n            sc.elev_raster, sc.pixel_spacing_m, cfg[\"avbd_n_steps\"]\n        )", "pred = _predict_arrivals_mesh_chunked(\n            bound, solver, mesh, source_3d, eval_pix,\n            sc.elev_raster, sc.pixel_spacing_m\n        )")

# Update training loop
text = text.replace("arrival_loss_obj = ArrivalTimeLoss(\n        solver=solver, solver_steps=cfg[\"avbd_n_steps\"]\n    )", "")
text = text.replace("def _fire_loss(m, _fc5, _ffc, _wv, _src, _x, _t):\n                bound = bind_mesh_scenario(m, _fc5, _ffc, _wv)\n                return arrival_loss_obj(bound, _src, _x, _t)", "def _fire_loss(m, _fc5, _ffc, _wv, _src, _x, _t):\n                bound = bind_mesh_scenario(m, _fc5, _ffc, _wv).precompute_metric_field()\n                return compute_sparse_mesh_eikonal_loss(bound, solver, mesh, _src, _x, _t)")

with open("experiments/wildfire/eikonal/experiment_eikonal_mesh.py", "w") as f:
    f.write(text)
