import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Any
from ham.utils.math import safe_norm
from ham.solvers.geodesic import ExponentialMap

class LossComponent(eqx.Module):
    """Base class for modular loss components."""
    weight: float
    name: str

    def __init__(self, weight: float = 1.0, name: str = "Loss"):
        self.weight = weight
        self.name = name

    def __call__(self, model: eqx.Module, batch: Tuple[Any, ...], key: jax.random.PRNGKey) -> jnp.ndarray:
        raise NotImplementedError("Loss component must implement __call__")

class ReconstructionLoss(LossComponent):
    """MSE reconstruction loss for the VAE."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "Recon")

    def __call__(self, model, batch, key):
        x = batch[0]
        dist = model._get_dist(x)
        z_sample = dist.sample(key)
        x_rec = model.decode(z_sample)
        return jnp.mean((x - x_rec)**2) * self.weight

class KLDivergenceLoss(LossComponent):
    """KL Divergence loss for the wrapped normal distribution."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "KL")

    def __call__(self, model, batch, key):
        x = batch[0]
        dist = model._get_dist(x)
        return jnp.mean(dist.kl_divergence_std_normal()) * self.weight

class ZermeloAlignmentLoss(LossComponent):
    """Aligns the latent velocity with the learned Wind field W."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "Z_Align")

    def __call__(self, model, batch, key):
        x, v_rna = batch[0], batch[1]
        z_mean, u_lat = model.project_control(x, v_rna)
        
        if hasattr(model.metric, '_get_zermelo_data'):
            _, W, _ = model.metric._get_zermelo_data(z_mean)
        else:
            W = jnp.zeros_like(u_lat)

        norm_w = model.manifold._minkowski_norm(W)
        norm_v = model.manifold._minkowski_norm(u_lat)
        
        w_dir = W / jnp.maximum(norm_w, 1e-6)[..., None]
        v_dir = u_lat / jnp.maximum(norm_v, 1e-6)[..., None]
        
        return -model.manifold._minkowski_dot(w_dir, v_dir) * self.weight

class GeodesicSprayLoss(LossComponent):
    """Penalizes acceleration (spray vector norm) to encourage geodesic trajectories."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "Spray")

    def __call__(self, model, batch, key):
        x, v_rna = batch[0], batch[1]
        z_mean, u_lat = model.project_control(x, v_rna)
        
        if hasattr(model.metric, '_get_zermelo_data'):
            _, W, _ = model.metric._get_zermelo_data(z_mean)
        else:
            W = jnp.zeros_like(u_lat)

        dot_z = u_lat + W
        spray_vec = model.metric.spray(z_mean, dot_z)
        spray_norm = model.metric.inner_product(z_mean, dot_z, spray_vec, spray_vec)
        
        return spray_norm * self.weight

class ContrastiveAlignmentLoss(LossComponent):
    """Aligns the Wind field with the log map between parent and child points in latent space."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "Cont_Align")

    def __call__(self, model, batch, key):
        parent_x, child_x = batch[0], batch[1]
        
        k1, k2 = jax.random.split(key)
        parent_z = model.encode(parent_x, k1)
        child_z = model.encode(child_x, k2)
        
        _, W_out, _ = model.metric._get_zermelo_data(parent_z)
        v_tan = model.manifold.log_map(parent_z, child_z)
        
        align_score = -model.manifold._minkowski_dot(W_out, v_tan)
        return align_score * self.weight

class MetricAnchorLoss(LossComponent):
    """Anchors the metric tensor H(x) to Identity to prevent degenerate solutions."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "H_Reg")

    def __call__(self, model, batch, key):
        parent_x = batch[0]
        parent_z = model.encode(parent_x, key)
        
        H_out, _, _ = model.metric._get_zermelo_data(parent_z)
        dim = H_out.shape[-1]
        I = jnp.eye(dim)
        
        return jnp.mean((H_out - I)**2) * self.weight

class MetricSmoothnessLoss(LossComponent):
    """Jacobian penalty on W to encourage smooth vector fields."""
    def __init__(self, weight: float = 0.1):
        super().__init__(weight, "Smoothness")

    def __call__(self, model, batch, key):
        parent_x = batch[0]
        parent_z = model.encode(parent_x, key)
        
        def get_w_single(pt):
            _, W_out, _ = model.metric._get_zermelo_data(pt)
            return W_out
        
        jac = jax.jacfwd(get_w_single)(parent_z)
        
        return jnp.mean(jac**2) * self.weight

def _solve_and_integrate_impl(model, z_start, z_end):
    """
    Solves the BVP and integrates the action.
    
    Defined at module level so JAX's tracer doesn't encounter a raw function
    object when tracing through vmap.
    """
    # 1. Solve the exact BVP using the AVBDSolver
    # AVBDSolver.solve expects (metric, p_start, p_end)
    traj_result = model.solver.solve(model.metric, z_start, z_end)
    trajectory = traj_result.xs  # Extract the points (N, latent_dim)
    
    # 2. Compute velocities via finite differences along the solved path
    N = trajectory.shape[0]
    dt = 1.0 / (N - 1)
    velocities = jnp.diff(trajectory, axis=0) / dt
    positions = trajectory[:-1]
    
    # 4. Integrate along the path
    step_energies = jax.vmap(model.metric.energy)(positions, velocities)
    return jnp.sum(step_energies) * dt

# Use eqx.filter_checkpoint instead of jax.checkpoint to properly handle
# non-array pytree leaves (e.g. activation functions in eqx.nn.MLP).
_solve_and_integrate = eqx.filter_checkpoint(_solve_and_integrate_impl)


class AVBDPathEnergyLoss(LossComponent):
    """
    Computes the Path Energy by actively solving the Boundary Value Problem (BVP).
    
    This uses the AVBDSolver to find the true geodesic under the *current* learned 
    metric between z_start and z_end. It then computes the total 
    Finsler/Randers action of that path. Minimizing this loss pulls the 
    metric so that the data trajectory becomes the path of least resistance.
    """
    solver_steps: int = eqx.field(static=True)

    def __init__(self, weight: float = 1.0, solver_steps: int = 15):
        super().__init__(weight, "AVBDEnergy")
        self.solver_steps = solver_steps

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        
        # Stochastic encoding
        k1, k2 = jax.random.split(key)
        z_start = model.encode(x_start, k1)
        z_end = model.encode(x_end, k2)
        
        # The pipeline's loss_fn is already vmapped over the batch,
        # so z_start and z_end are single samples here.
        energy = _solve_and_integrate(model, z_start, z_end)
        
        return energy * self.weight

class WindThermodynamicLoss(LossComponent):
    """
    Penalizes the magnitude of the Wind field. 
    By making wind 'expensive', we force the network to warp the Riemannian 
    tensor G(z) to explain the data trajectories, breaking the Identifiability Trap.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "WindCost")

    def __call__(self, model, batch, key):
        x = batch[0]
        # Encode to latent space (single sample — pipeline already vmaps over batch)
        z = model.encode(x, key)
        
        H_matrix, W, _ = model.metric._get_zermelo_data(z)
        # Cost is the squared magnitude of the wind in the local metric
        wind_cost = jnp.dot(W, jnp.dot(H_matrix, W))
            
        return wind_cost * self.weight

def _solve_and_align_impl(model, z_s, z_e, dx_true):
    # 1. Solve the exact boundary value problem
    traj = model.solver.solve(model.metric, z_s, z_e)
    
    # 2. Extract initial latent velocity
    # traj.vs[0] is the log_map from x_0 to x_1. 
    # Multiply by number of segments to scale it to the time domain t in [0, 1]
    N_steps = traj.xs.shape[0] - 1
    v_latent = traj.vs[0] * N_steps
    
    # 3. Pushforward through the decoder via JVP
    dec_fn = lambda z: model.decode(z)
    _, v_data_pred = jax.jvp(dec_fn, (z_s,), (v_latent,))
    
    # 4. Maximize cosine similarity
    norm_true = safe_norm(dx_true) + 1e-8
    norm_pred = safe_norm(v_data_pred) + 1e-8
    cos_sim = jnp.dot(v_data_pred, dx_true) / (norm_pred * norm_true)
    
    # We want cos_sim to be 1.0, so loss is 1 - cos_sim
    return 1.0 - cos_sim

_solve_and_align = eqx.filter_checkpoint(_solve_and_align_impl)

class JVPGeodesicAlignmentLoss(LossComponent):
    """
    Extracts the initial tangent vector of the learned geodesic, pushes it 
    forward into the data space via the decoder's JVP, and maximizes its 
    cosine similarity with the actual observed biological displacement.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "JVP_Align")

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        
        k1, k2 = jax.random.split(key)
        z_start = model.encode(x_start, k1)
        z_end = model.encode(x_end, k2)
        
        # True displacement in the high-dimensional data space
        dx_true = x_end - x_start

        loss = _solve_and_align(model, z_start, z_end, dx_true)
        return loss * self.weight

def _compute_contrast_impl(model, z_s, z_e, margin):
    # 1. Solve the forward path and get its energy
    traj = model.solver.solve(model.metric, z_s, z_e)
    E_fwd = traj.energy
    
    # 2. Re-evaluate the path in reverse
    xs_rev = traj.xs[::-1]
    
    # On curved manifolds, log_map(B, A) is not simply -log_map(A, B),
    # so we rigorously compute the reverse sequential log maps
    vs_rev = jax.vmap(model.manifold.log_map)(xs_rev[:-1], xs_rev[1:])
    
    # 3. Compute the backward energy using the metric
    E_rev = jnp.sum(jax.vmap(model.metric.energy)(xs_rev[:-1], vs_rev))
    
    # 4. Contrastive Hinge Loss: We want E_rev > E_fwd + margin
    # If E_rev is large, this term goes to 0. If E_rev is small, it applies a penalty.
    return jax.nn.relu(E_fwd - E_rev + margin)

_compute_contrast = eqx.filter_checkpoint(_compute_contrast_impl)

class TemporalContrastiveLoss(LossComponent):
    """
    Forces the action of traversing the geodesic backward to be strictly 
    greater than traversing it forward. This explicitly punishes symmetric 
    metrics and forces the network to learn the asymmetric Wind field W(z).
    """
    margin: float

    def __init__(self, weight: float = 1.0, margin: float = 1.0):
        super().__init__(weight, "TempCont")
        self.margin = margin

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        k1, k2 = jax.random.split(key)
        z_start = model.encode(x_start, k1)
        z_end = model.encode(x_end, k2)

        loss = _compute_contrast(model, z_start, z_end, self.margin)
        return loss * self.weight

class KinematicPriorLoss(LossComponent):
    margin: float

    def __init__(self, weight: float = 1.0, margin: float = 0.5):
        super().__init__(weight, "Kinematic")
        self.margin = margin

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        
        k1, k2 = jax.random.split(key)
        z_start = model.encode(x_start, k1)
        z_end = model.encode(x_end, k2)
        
        v_diff = model.manifold.log_map(z_start, z_end)
        dist = safe_norm(v_diff, axis=-1)
        
        # Only apply the penalty if the distance is LARGER than the margin.
        # Once they are within 0.5 units of each other, the loss is 0.
        loss = jax.nn.relu(dist - self.margin)**2
        
        return jnp.mean(loss) * self.weight

def _compute_wind_alignment_impl(model, z_s, z_e):
    # 1. Solve the path
    traj = model.solver.solve(model.metric, z_s, z_e, n_steps=4)
    
    # 2. Extract the Wind W(z) at every point along the path
    def get_w(z):
        _, W, _ = model.metric._get_zermelo_data(z)
        return W
    ws = jax.vmap(get_w)(traj.xs[:-1])
    
    # 3. Path velocities
    vs = traj.vs 
    
    # 4. Maximize Cosine Similarity between the Wind and the Path
    w_norm = safe_norm(ws) + 1e-6
    v_norm = safe_norm(vs) + 1e-6
    cos_sim = jnp.sum(ws * vs, axis=-1) / (w_norm * v_norm)
    
    # We want cos_sim to be 1.0, so loss is 1.0 - cos_sim
    return jnp.mean(1.0 - cos_sim)

_compute_wind_alignment = eqx.filter_checkpoint(_compute_wind_alignment_impl)

class WindAlignmentLoss(LossComponent):
    """
    Directly forces the learned Zermelo Wind W(z) to align with the 
    forward biological trajectory found by the solver. This guarantees 
    the network knows W is responsible for the directed dynamics.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "WindAlign")

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        k1, k2 = jax.random.split(key)
        z_start = model.encode(x_start, k1)
        z_end = model.encode(x_end, k2)

        loss = _compute_wind_alignment(model, z_start, z_end)
        return loss * self.weight

def _shoot_and_evaluate_impl(model, shooter, z_s, z_e, key):
    # 3. The initial momentum is the autonomous biological wind
    _, W_start, _ = model.metric._get_zermelo_data(z_s)
    
    # 3.5. Avoid NaNs from shooting with exactly 0 initial velocity.
    # W_start could be ~0 at the beginning of training. 
    noise = jax.random.normal(key, W_start.shape)
    noise = model.manifold.to_tangent(z_s, noise)
    noise = noise / (safe_norm(noise) + 1e-12) * 1e-4

    w_norm = safe_norm(W_start)
    w_norm_expanded = jnp.expand_dims(w_norm, axis=-1) if w_norm.ndim < W_start.ndim else w_norm
    W_start = jnp.where(w_norm_expanded < 1e-5, W_start + noise, W_start)
    
    # 4. Shoot the geodesic forward using your RK4!
    # This computes the full path feeling both H(z) and W(z)
    xs, vs = shooter.trace(model.metric, z_s, W_start)

    xs_future = xs[1:]
    
    # 5. Time-Agnostic Evaluation: 
    # Calculate the squared distance from every point on the trajectory to z_end.
    dists_sq = jnp.sum((xs_future - z_e)**2, axis=-1)
    
    # The loss is simply the distance at the closest point of approach.
    min_dist_sq = jnp.min(dists_sq)
    
    return min_dist_sq

def _shoot_and_evaluate(model, shooter, z_s, z_e, key):
    return eqx.filter_checkpoint(_shoot_and_evaluate_impl)(model, shooter, z_s, z_e, key)

class TimeAgnosticWindShooterLoss(LossComponent):
    """
    Shoots a particle forward using the full Geodesic Spray (RK4).
    The initial velocity is the Zermelo Wind W(z). The trajectory is evaluated 
    based on its closest approach to the target state, making it robust to 
    varying biological timesteps (delta t).
    """
    shooter: ExponentialMap

    def __init__(self, weight: float = 1.0, max_steps: int = 15): 
        super().__init__(weight, "IVP_Shooter")
        # Instantiate your RK4 solver. 
        # We keep max_steps relatively low during training for speed.
        self.shooter = ExponentialMap(step_size=1.0/max_steps, max_steps=max_steps)

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        
        # Split key for deterministic embedding and shooting noise
        k1, k2, k_shoot = jax.random.split(key, 3)
        
        # 1. Deterministic embedding onto the base manifold
        z_start = model.encode(x_start, k1)
        z_end = model.encode(x_end, k2)

        # 2. Vectorize the forward shooting across the batch
        loss = _shoot_and_evaluate(model, self.shooter, z_start, z_end, k_shoot)
        return jnp.mean(loss) * self.weight

def _shoot_and_evaluate_overdamped_impl(model, z_s, z_e):
    # Overdamped Physics: We just flow along the Wind. No acceleration/inertia.
    def step_fn(z_current, _):
        _, W, _ = model.metric._get_zermelo_data(z_current)
        z_next = z_current + W * self.dt
        return z_next, z_next
        
    _, xs = jax.lax.scan(step_fn, z_s, None, length=self.steps)
    
    # We evaluate how close the future flow gets to the biological target
    dists_sq = jnp.sum((xs - z_e)**2, axis=-1)
    return jnp.min(dists_sq)

def _shoot_and_evaluate_overdamped(model, z_s, z_e, key):
    return eqx.filter_checkpoint(_shoot_and_evaluate_overdamped_impl)(model, z_s, z_e)

class OverdampedIVPShooterLoss(LossComponent):
    """
    Simulates 'leaving the system alone' using overdamped biological physics.
    The particle flows exactly along the latent Wind field W(z).
    """
    steps: int
    dt: float
    
    def __init__(self, weight: float = 1.0, steps: int = 15, dt: float = 0.1):
        super().__init__(weight, "IVP_Shooter")
        self.steps = steps
        self.dt = dt

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        
        z_start = model.encode(x_start, key)
        z_end = model.encode(x_end, key)

        loss = _shoot_and_evaluate_overdamped(model, z_start, z_end, key)
        return jnp.mean(loss) * self.weight
