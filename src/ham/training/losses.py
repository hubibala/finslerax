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

class VelocityDirectionAlignmentLoss(LossComponent):
    """
    Computes negative cosine similarity between the true data velocity
    projected into latent space and the metric's learned drift direction (Wind vector W).
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "VelDirAlign")

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        v_true = x_end - x_start
        
        # Project data velocity to latent space
        z_start, v_lat = model.project_control(x_start, v_true)
        
        # Get metric drift (Wind vector W)
        if hasattr(model.metric, '_get_zermelo_data'):
            _, W, _ = model.metric._get_zermelo_data(z_start)
        else:
            W = jnp.zeros_like(v_lat)
            
        # Cosine similarity in latent space
        norm_w = safe_norm(W, axis=-1)
        norm_v = safe_norm(v_lat, axis=-1)
        
        w_dir = W / jnp.maximum(norm_w, 1e-8)[..., None]
        v_dir = v_lat / jnp.maximum(norm_v, 1e-8)[..., None]
        
        cos_sim = jnp.sum(w_dir * v_dir, axis=-1)
        
        return (1.0 - cos_sim) * self.weight

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
        
        if hasattr(model.metric, '_get_zermelo_data'):
            H_out, _, _ = model.metric._get_zermelo_data(parent_z)
        elif hasattr(model.metric, 'g_net'):
            H_out = model.metric.g_net(parent_z)
            H_out = 0.5 * (H_out + H_out.T)
        else:
            return 0.0

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
    traj_result = model.solver.solve(model.metric, z_start, z_end)
    trajectory = traj_result.xs  
    
    N = trajectory.shape[0]
    dt = 1.0 / (N - 1)
    velocities = jnp.diff(trajectory, axis=0) / dt
    positions = trajectory[:-1]
    
    step_energies = jax.vmap(model.metric.energy)(positions, velocities)
    return jnp.sum(step_energies) * dt

_solve_and_integrate = eqx.filter_checkpoint(_solve_and_integrate_impl)

def _solve_avbd_trajectory_impl(model, z_start, z_end, n_steps):
    """
    Finds the geodesic between z_start and z_end using the model's AVBD solver.
    """
    traj = model.solver.solve(model.metric, z_start, z_end, n_steps=n_steps)
    return traj.xs

_solve_avbd_trajectory = eqx.filter_checkpoint(_solve_avbd_trajectory_impl)

class LongTrajectoryAlignmentLoss(LossComponent):
    """
    Stable alignment loss that compares the full observed path
    against the geodesic (BVP solution) predicted by the metric.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "LongTrajAlign")

    def __call__(self, model, batch, key):
        # batch[2] is expected to be Traj_long (T, 3) for a single trajectory in the vmap
        traj_obs_data = batch[2] 
        n_points = traj_obs_data.shape[0]
        
        # 1. Encode to latent space (using a fixed key for Deterministic context if possible, 
        # or just vmap encode)
        # We use a static key to avoid stochastic 'jitter' in the regression target.
        keys = jax.random.split(key, n_points)
        z_obs = jax.vmap(model.encode)(traj_obs_data, keys)
        
        # 2. Solve the BVP between observed start and end
        z_start = z_obs[0]
        z_end = z_obs[-1]
        n_segments = n_points - 1
        
        z_geo = _solve_avbd_trajectory(model, z_start, z_end, n_segments)
        
        # 3. Penalize the difference
    # 3. Penalize the difference
        # This forces the learned geometry's 'straight lines' to be the data's paths.
        return jnp.mean((z_geo - z_obs)**2) * self.weight

class EulerLagrangeResidualLoss(LossComponent):
    """
    Physically-informed loss that penalizes violations of the Euler-Lagrange equations.
    This provides a simulation-free, mathematically rigorous method to align empirical 
    trajectories with the geodesics of the learned Randers metric.
    
    The residual R measures the degree to which an observed path deviates from the 
    extremum of the energy functional. R = d/dt (dL/dv) - dL/dz.
    """
    epsilon: float = eqx.field(static=True)

    def __init__(self, weight: float = 1.0, epsilon: float = 1e-4):
        super().__init__(weight, "EL_Residual")
        # smoothing for Finsler non-smoothness at v=0
        self.epsilon = epsilon

    def __call__(self, model: eqx.Module, batch: Tuple[Any, ...], key: jax.random.PRNGKey) -> jnp.ndarray:
        # batch[2] is expected to be the empirical trajectory Traj_long: (T, D_data)
        if len(batch) < 3 or batch[2] is None:
            return 0.0
            
        traj_obs = batch[2]
        T = traj_obs.shape[0]
        
        # 1. Map empirical trajectory to latent space
        # Use a fixed key per point to ensure the 'path' is spatially consistent for differentiation
        keys = jax.random.split(key, T)
        z_traj = jax.vmap(model.encode)(traj_obs, keys)
        
        # 2. Extract continuous state, velocity, and acceleration approximations
        # Using centered finite differences as the discrete approximation of spline derivatives
        dt = 1.0 / (T - 1)
        v_traj = jnp.gradient(z_traj, axis=0) / dt
        a_traj = jnp.gradient(v_traj, axis=0) / dt

        def compute_el_residual_sq(z, v, a):
            """Point-wise evaluation of the Euler-Lagrange residual norm squared."""
            
            # Define smoothed Lagrangian L(z, v) = 1/2 * F_eps(z, v)^2
            def L_smooth(z_pt, v_pt):
                # Retrieve Riemannian metric H and Wind W from the Randers metric
                if hasattr(model.metric, '_get_zermelo_data'):
                    H_pt, W_pt, _ = model.metric._get_zermelo_data(z_pt)
                else:
                    # Identity fallback for non-Randers/Base metrics
                    H_pt = jnp.eye(z_pt.shape[0])
                    W_pt = jnp.zeros_like(z_pt)
                
                # F_eps = sqrt(v^T H v + eps^2) - <W, v>_H
                # This formulation ensures smoothness at v=0 while capturing Randers asymmetry
                v_norm_sq = jnp.dot(v_pt, jnp.dot(H_pt, v_pt))
                v_norm_eps = jnp.sqrt(v_norm_sq + self.epsilon**2)
                
                # Wind interaction using the local metric tensor
                W_dot_v = jnp.dot(W_pt, jnp.dot(H_pt, v_pt))
                
                F = v_norm_eps - W_dot_v
                return 0.5 * (F**2)

            # dL/dv gradient function
            grad_v_fn = jax.grad(L_smooth, argnums=1)
            
            # Total time derivative components: d/dt(dL/dv) = (d2L/dv2)*a + (d2L/dzdv)*v
            # We utilize jax.jvp for efficient forward-mode Hessian-vector products
            _, hess_v_a = jax.jvp(lambda v_arg: grad_v_fn(z, v_arg), (v,), (a,))
            _, mixed_term = jax.jvp(lambda z_arg: grad_v_fn(z_arg, v), (z,), (v,))
            
            # Spatial gradient dL/dz
            grad_z = jax.grad(L_smooth, argnums=0)(z, v)
            
            # Euler-Lagrange Residual vector: R = d/dt(dL/dv) - dL/dz
            residual = hess_v_a + mixed_term - grad_z
            
            # Evaluate norm using the 'frozen' Riemannian metric tensor H
            # Evaluation magnitude is geometrically consistent with the manifold's curvature
            if hasattr(model.metric, '_get_zermelo_data'):
                H_frozen, _, _ = model.metric._get_zermelo_data(z)
            else:
                H_frozen = jnp.eye(z.shape[0])
            H_frozen = jax.lax.stop_gradient(H_frozen)
            
            # Squared norm under Riemannian metric H
            return jnp.dot(residual, jnp.dot(H_frozen, residual))

        # Vectorized evaluation over the entire trajectory time horizon
        # Completely avoids ODE solvers by operating point-wise
        el_violations = jax.vmap(compute_el_residual_sq)(z_traj, v_traj, a_traj)
        
        return jnp.mean(el_violations) * self.weight

class AVBDPathEnergyLoss(LossComponent):
    solver_steps: int = eqx.field(static=True)

    def __init__(self, weight: float = 1.0, solver_steps: int = 15):
        super().__init__(weight, "AVBDEnergy")
        self.solver_steps = solver_steps

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        k1, k2 = jax.random.split(key)
        z_start = model.encode(x_start, k1)
        z_end = model.encode(x_end, k2)
        
        energy = _solve_and_integrate(model, z_start, z_end)
        return energy * self.weight

class WindThermodynamicLoss(LossComponent):
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "WindCost")

    def __call__(self, model, batch, key):
        x = batch[0]
        z = model.encode(x, key)
        if hasattr(model.metric, '_get_zermelo_data'):
            H_matrix, W, _ = model.metric._get_zermelo_data(z)
            wind_cost = jnp.dot(W, jnp.dot(H_matrix, W))
        else:
            wind_cost = 0.0
        return wind_cost * self.weight

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
        loss = jax.nn.relu(dist - self.margin)**2
        return jnp.mean(loss) * self.weight

# =====================================================================
# THE NEW FINSLER ACTION MATCHING LOSS
# =====================================================================

class FinslerActionMatchingLoss(LossComponent):
    """
    Minimizes the Randers energy of observed biological transitions directly.
    Requires joint training so the VAE reconstruction prevents scale collapse.
    Calculates $E = F(z, v)^2$, avoiding ODE integration entirely during training.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "ActionMatching")

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        
        # 1. Stochastic encoding to latent space
        k1, k2 = jax.random.split(key)
        z_start = model.encode(x_start, k1)
        z_end = model.encode(x_end, k2)
        
        # 2. Approximate the tangent vector (observed biological flow)
        v = z_end - z_start
        
        # 3. Calculate Finsler energy using the model's underlying Randers metric
        # This implicitly utilizes Pullback H(z) and evaluates the Learned W(z)
        energy = model.metric.energy(z_start, v) 
        
        return jnp.mean(energy) * self.weight

class WindAssistedTrajectoryAlignmentLoss(LossComponent):
    """
    Aligns short rolled-out trajectories (using geodesic shooter) with observed displacements.
    Encourages the learned Wind/Metric to produce geodesics that match the observed flows.
    """
    rollout_steps: int
    dt: float

    def __init__(self, weight: float = 1.0, rollout_steps: int = 1, dt: float = 1.0):
        super().__init__(weight, "WindTrajAlign")
        # Ensure rollout_steps is at least 1 for the particle shooter
        self.rollout_steps = max(1, rollout_steps)
        self.dt = dt

    def __call__(self, model, batch, key):
        x_start, x_end = batch[0], batch[1]
        
        k1, k2 = jax.random.split(key)
        z_start = model.encode(x_start, k1)
        z_end = model.encode(x_end, k2)
        
        # v_init represents the total latent displacement to reach the end in t=1
        v_init = z_end - z_start 
        
        # Shoot using the geodesic ODE (Particle Shooter)
        # Using 1 step to avoid gradient chaos through RK4 loops, 
        # while explicitly utilizing geod_acceleration(z)
        ivp_shooter = ExponentialMap(max_steps=self.rollout_steps)
        z_pred = ivp_shooter.shoot(model.metric, z_start, v_init)
        
        # Decode rolled z and align with observed future endpoint
        x_pred = model.decode(z_pred)
        
        mse_loss = jnp.mean((x_pred - x_end)**2)
        return mse_loss * self.weight

# =====================================================================
# SOTA LOSS FUNCTIONS: FINSLERIAN FLOW MATCHING & EIKONAL ALIGNMENT
# =====================================================================

class FinslerianFlowMatchingLoss(LossComponent):
    """
    Learns the Randers wind field W_theta(z) by matching it to the empirical drift.
    Mathematically aligns the vector field with observed trajectory derivatives.
    
    Reframes the trajectory alignment as finding the optimal W that minimizes 
    the deviation of the empirical path from a wind-assisted geodesic.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "FinslerFlowMatching")

    def __call__(self, model: eqx.Module, batch: Tuple[Any, ...], key: jax.random.PRNGKey) -> jnp.ndarray:
        # batch[2] is Traj_long: (T, D_data)
        if len(batch) < 3 or batch[2] is None:
            return 0.0
            
        traj_obs = batch[2]
        T = traj_obs.shape[0]
        dt = 1.0 / (T - 1)
        
        # 1. Map empirical trajectory to latent space
        keys = jax.random.split(key, T)
        z_traj = jax.vmap(model.encode)(traj_obs, keys)
        
        # 2. Empirical velocity (tangent vector)
        v_traj = jnp.gradient(z_traj, axis=0) / dt
        
        # 3. Retrieve Randers data (Riemannian H and Wind W)
        def get_randers_data(z):
            if hasattr(model.metric, '_get_zermelo_data'):
                H, W, _ = model.metric._get_zermelo_data(z)
                return H, W
            return jnp.eye(z.shape[0]), jnp.zeros_like(z)
            
        H_traj, W_traj = jax.vmap(get_randers_data)(z_traj)
        
        # 4. Velocity-Drift Alignment
        # In Zermelo navigation, if the wind W blows in the direction of v, 
        # the cost F(z, v) is minimized. 
        # This loss encourages W to align with the normalized tangent vector.
        
        v_norm_h = jax.vmap(lambda v, H: jnp.sqrt(jnp.dot(v, jnp.dot(H, v)) + 1e-8))(v_traj, H_traj)
        v_unit = v_traj / jnp.maximum(v_norm_h, 1e-6)[..., None]
        
        # We align W with v_unit
        # Higher alignment (dot product) reduces the loss
        alignment = jax.vmap(lambda w, v, H: jnp.dot(w, jnp.dot(H, v)))(W_traj, v_unit, H_traj)
        
        # Regress towards perfect alignment (normalized W magnitude can be learned or regularized)
        loss = 1.0 - alignment
        
        return jnp.mean(loss) * self.weight
