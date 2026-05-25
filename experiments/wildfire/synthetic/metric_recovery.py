import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Dict, Optional, Tuple, Any

from ham.solvers.eikonal import EikonalSolver
from ham.solvers.avbd import AVBDSolver
from ham.training.losses import DenseArrivalTimeLoss, ArrivalTimeLoss
from experiments.wildfire.synthetic.experiment_base import SyntheticZermeloMetric

def compute_tv_regularization(H_grid: jax.Array, W_grid: Optional[jax.Array],
                              lambda_H: float, lambda_W: float) -> jax.Array:
    """Compute Total Variation (TV) regularization for H and W grids."""
    loss_reg = 0.0
    
    if lambda_H > 0:
        # H_grid shape: (3, M, N)
        dh_dx = H_grid[:, :, 1:] - H_grid[:, :, :-1]
        dh_dy = H_grid[:, 1:, :] - H_grid[:, :-1, :]
        tv_H = jnp.mean(jnp.abs(dh_dx)) + jnp.mean(jnp.abs(dh_dy))
        loss_reg += lambda_H * tv_H
        
    if W_grid is not None and lambda_W > 0:
        # W_grid shape: (2, M, N)
        dw_dx = W_grid[:, :, 1:] - W_grid[:, :, :-1]
        dw_dy = W_grid[:, 1:, :] - W_grid[:, :-1, :]
        tv_W = jnp.mean(jnp.abs(dw_dx)) + jnp.mean(jnp.abs(dw_dy))
        loss_reg += lambda_W * tv_W
        
    return loss_reg

def eikonal_to_zermelo(G: jax.Array, B: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Convert Eikonal Godunov variables (G, B) to Zermelo (H, W) assuming lam=1.
    G shape (3, ...), B shape (2, ...)
    """
    # G = (H + (HW)(HW)^T)
    # B = -HW
    # Therefore, HW = -B
    # H = G - (-B)(-B)^T = G - BB^T
    # And since HW = -B => H W = -B => W = H^-1 (-B)
    
    # H = G - B B^T
    g11, g12, g22 = G[0], G[1], G[2]
    b1, b2 = B[0], B[1]
    
    h11 = g11 - b1*b1
    h12 = g12 - b1*b2
    h22 = g22 - b2*b2
    H = jnp.stack([h11, h12, h22], axis=0)
    
    # Det H
    detH = h11 * h22 - h12**2
    detH_safe = jnp.maximum(detH, 1e-6)
    
    # H^-1
    inv_h11 = h22 / detH_safe
    inv_h12 = -h12 / detH_safe
    inv_h22 = h11 / detH_safe
    
    # W = H^-1 (-B)
    w1 = inv_h11 * (-b1) + inv_h12 * (-b2)
    w2 = inv_h12 * (-b1) + inv_h22 * (-b2)
    W = jnp.stack([w1, w2], axis=0)
    
    return H, W

def zermelo_to_eikonal(H: jax.Array, W: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Convert Zermelo (H, W) back to Eikonal (G, B) with lam=1.
    H shape (3, ...), W shape (2, ...)
    """
    h11, h12, h22 = H[0], H[1], H[2]
    w1, w2 = W[0], W[1]
    
    b1 = -(h11 * w1 + h12 * w2)
    b2 = -(h12 * w1 + h22 * w2)
    B = jnp.stack([b1, b2], axis=0)
    
    g11 = h11 + b1*b1
    g12 = h12 + b1*b2
    g22 = h22 + b2*b2
    G = jnp.stack([g11, g12, g22], axis=0)
    
    return G, B

class MetricRecoveryModel(eqx.Module):
    """Holds the parameters to be optimized."""
    H_grid: jax.Array
    W_grid: jax.Array
    
    def __init__(self, M: int, N: int, init_H: jax.Array = None, init_W: jax.Array = None):
        if init_H is None:
            # Initialize with isotropic H=I
            self.H_grid = jnp.zeros((3, M, N))
            self.H_grid = self.H_grid.at[0].set(1.0)
            self.H_grid = self.H_grid.at[2].set(1.0)
        else:
            self.H_grid = init_H
            
        if init_W is None:
            self.W_grid = jnp.zeros((2, M, N))
        else:
            self.W_grid = init_W

    def project(self, constrain_isotropic: bool) -> 'MetricRecoveryModel':
        """Projects H to be SPD and optionally isotropic."""
        H = self.H_grid
        
        # Clip diagonals to avoid blowup or collapse
        h11 = jnp.clip(H[0], 0.1, 10.0)
        h22 = jnp.clip(H[2], 0.1, 10.0)
        h12 = H[1]
        
        if constrain_isotropic:
            avg = (h11 + h22) / 2.0
            h11 = avg
            h22 = avg
            h12 = jnp.zeros_like(h12)
        else:
            # Enforce determinant > 0: h12^2 < h11 * h22
            max_h12 = jnp.sqrt(h11 * h22) * 0.95
            h12 = jnp.clip(h12, -max_h12, max_h12)
            
        H_new = jnp.stack([h11, h12, h22], axis=0)
        
        # Project W for causal bound: ||W||_H < 1
        # This is roughly equivalent to ||B||_G^-1 < 1
        W = self.W_grid
        W_norm_sq = (W[0] * (h11 * W[0] + h12 * W[1]) + 
                     W[1] * (h12 * W[0] + h22 * W[1]))
        
        mask = W_norm_sq > 0.81  # Max magnitude 0.9
        scale = jnp.where(mask, 0.9 / jnp.maximum(jnp.sqrt(W_norm_sq), 1e-8), 1.0)
        W_new = W * scale
        
        return eqx.tree_at(
            lambda m: (m.H_grid, m.W_grid),
            self,
            (H_new, W_new)
        )

class MetricRecoveryOptimizer:
    """Optimizer for synthetic metric recovery using JAX and Optax.
    Supports both 'eikonal' and 'avbd' solvers.
    """
    
    def __init__(self, M: int, N: int, solver_type: str = 'eikonal',
                 recover_H: bool = True, recover_W: bool = False,
                 constrain_isotropic: bool = True,
                 lambda_H: float = 0.01,
                 lambda_W: float = 0.01,
                 reg_type: str = 'tv',
                 optimizer_type: str = 'adam'):
        self.M = M
        self.N = N
        self.solver_type = solver_type
        self.recover_H = recover_H
        self.recover_W = recover_W
        self.constrain_isotropic = constrain_isotropic
        self.lambda_H = lambda_H
        self.lambda_W = lambda_W
        self.reg_type = reg_type
        self.optimizer_type = optimizer_type
        
        self.model = MetricRecoveryModel(M, N)
        
        if self.solver_type == 'eikonal':
            self.solver = EikonalSolver(max_iters=50, tol=1e-5)
            self.loss_fn = DenseArrivalTimeLoss(weight=1.0)
        elif self.solver_type == 'avbd':
            self.solver = AVBDSolver(step_size=0.1, iterations=40, parallel=True)
            self.loss_fn = ArrivalTimeLoss(solver=self.solver, solver_steps=20, weight=1.0)
        else:
            raise ValueError(f"Unknown solver_type: {solver_type}")
            
        self.history = {'loss': [], 'loss_data': [], 'loss_reg': []}
        
    def fit(self, source_coords: jax.Array, T_obs: jax.Array, obs_mask: jax.Array,
            n_iter: int = 500, lr: float = 0.1, verbose: bool = True, alpha: float = 0.0,
            patience: int = 20, min_delta: float = 1e-4) -> Dict:
        """Fit parameters to observed arrival times."""
        
        # We need continuous coordinates for observations if using AVBD
        M, N = self.M, self.N
        I, J = jnp.meshgrid(jnp.arange(M), jnp.arange(N), indexing='ij')
        obs_coords = jnp.stack([I[obs_mask], J[obs_mask]], axis=-1).astype(jnp.float32)
        T_obs_values = T_obs[obs_mask]
        
        if source_coords.ndim == 1:
            source_coords = source_coords[None, :]
        
        # Setup Optax
        schedule = optax.exponential_decay(init_value=lr, transition_steps=100, decay_rate=0.8)
        # Filter trainable parameters
        filter_spec = jax.tree_util.tree_map(lambda _: False, self.model)
        filter_spec = eqx.tree_at(lambda m: m.H_grid, filter_spec, self.recover_H)
        filter_spec = eqx.tree_at(lambda m: m.W_grid, filter_spec, self.recover_W)
        if self.optimizer_type == 'adam':
            optimizer = optax.adam(lr)
        else:
            optimizer = optax.sgd(lr)
            
        opt_state = optimizer.init(eqx.filter(self.model, filter_spec))

        @eqx.filter_value_and_grad(has_aux=True)
        def compute_loss(diff_model: MetricRecoveryModel, static_model: MetricRecoveryModel, current_iter: int):
            model = eqx.combine(diff_model, static_model)
            metric = SyntheticZermeloMetric(model.H_grid, model.W_grid)
            
            if self.solver_type == 'eikonal':
                T_pred, _, _ = self.solver.solve(metric, source_coords, 
                                               grid_extent=(0, M-1, 0, N-1), 
                                               grid_shape=(M, N))
                
                T_obs_masked = jnp.where(obs_mask, T_obs, jnp.nan)
                diff = T_pred[obs_mask] - T_obs_values
                loss_data = 0.5 * jnp.sum(diff**2)
            else:
                src = source_coords[0]
                
                # Curriculum alpha
                warmup = int(0.2 * n_iter)
                ramp = int(0.6 * n_iter)
                
                def get_alpha(it):
                    return jnp.where(it < warmup, 0.0, 
                           jnp.where(it > warmup + ramp, 1.0, 
                           (it - warmup) / jnp.maximum(ramp, 1)))
                           
                current_alpha = get_alpha(current_iter)
                loss_data = self.loss_fn(metric, src, obs_coords, T_obs_values, alpha=current_alpha)
                
            if self.reg_type == 'tv':
                loss_reg = compute_tv_regularization(model.H_grid, model.W_grid, 
                                                     self.lambda_H if self.recover_H else 0.0,
                                                     self.lambda_W if self.recover_W else 0.0)
            elif self.reg_type == 'tikhonov':
                # Tikhonov: penalty on magnitude of H and W
                loss_reg = 0.0
                if self.recover_H:
                    loss_reg += self.lambda_H * jnp.sum(model.H_grid**2)
                if self.recover_W:
                    loss_reg += self.lambda_W * jnp.sum(model.W_grid**2)
            else:
                loss_reg = 0.0
                
            loss = loss_data + loss_reg
            return loss, (loss_data, loss_reg)

        @eqx.filter_jit
        def make_step(model, opt_state, current_iter):
            diff_model, static_model = eqx.partition(model, filter_spec)
            (loss, (loss_data, loss_reg)), grads = compute_loss(diff_model, static_model, current_iter)
            updates, opt_state = optimizer.update(grads, opt_state, diff_model)
            diff_model = eqx.apply_updates(diff_model, updates)
            model = eqx.combine(diff_model, static_model)
            model = model.project(self.constrain_isotropic)
            return model, opt_state, loss, loss_data, loss_reg

        try:
            from tqdm.auto import tqdm
            pbar = tqdm(range(n_iter), desc=f"Training ({self.solver_type})", disable=not verbose, leave=False)
        except ImportError:
            pbar = range(n_iter)
            
        best_loss = float('inf')
        no_improvement_count = 0
        
        # Determine when alpha curriculum finishes so we don't early stop during ramp
        alpha_done_iter = 0
        if self.solver_type == 'avbd':
            warmup = int(0.2 * n_iter)
            ramp = int(0.6 * n_iter)
            alpha_done_iter = warmup + ramp
            
        for it in pbar:
            self.model, opt_state, loss, loss_data, loss_reg = make_step(self.model, opt_state, it)
            
            loss_val = float(loss)
            self.history['loss'].append(loss_val)
            self.history['loss_data'].append(float(loss_data))
            self.history['loss_reg'].append(float(loss_reg))
            
            if verbose:
                if hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(loss=f"{loss_val:.4f}", data=f"{float(loss_data):.4f}")
                elif it % 50 == 0 or it == n_iter - 1:
                    print(f"  Iter {it}: loss={loss_val:.4f} (data={float(loss_data):.4f}, reg={float(loss_reg):.4f})")
                    
            if it < alpha_done_iter:
                # Reset best loss continuously while curriculum changes loss landscape
                best_loss = loss_val
                no_improvement_count = 0
            else:
                if loss_val < best_loss - min_delta:
                    best_loss = loss_val
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
            if no_improvement_count >= patience:
                if verbose:
                    if hasattr(pbar, 'write'):
                        pbar.write(f"  Early stopping at iteration {it} (no improvement in {patience} iters)")
                    else:
                        print(f"  Early stopping at iteration {it} (no improvement in {patience} iters)")
                break
                
        return {'final_loss': self.history['loss'][-1], 
                'final_data': self.history['loss_data'][-1], 
                'final_reg': self.history['loss_reg'][-1]}
                
    def get_G_B(self) -> Tuple[jax.Array, jax.Array]:
        """Returns the recovered parameters in Eikonal G/B form."""
        return zermelo_to_eikonal(self.model.H_grid, self.model.W_grid)
