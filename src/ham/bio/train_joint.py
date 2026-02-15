import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
from typing import Tuple

from ham.bio.vae import GeometricVAE
from ham.bio.data import BioDataset

# --- Pure Update Functions (Outside Class) ---

@eqx.filter_jit
def step_manifold(model: GeometricVAE, 
                  opt_state: optax.OptState, 
                  optimizer: optax.GradientTransformation,
                  x_batch: jnp.ndarray, 
                  v_batch: jnp.ndarray, 
                  key: jax.random.PRNGKey):
    
    keys = jax.random.split(key, x_batch.shape[0])
    
    def batch_loss_fn(m):
        losses, stats = jax.vmap(m.loss_fn)(x_batch, v_batch, keys)
        return jnp.mean(losses), stats

    (loss, (recon, kl, spray, align)), grads = eqx.filter_value_and_grad(batch_loss_fn, has_aux=True)(model)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    
    return new_model, new_opt_state, loss, jnp.mean(recon), jnp.mean(kl), jnp.mean(align)

@eqx.filter_jit
def step_metric(model: GeometricVAE,
                opt_state: optax.OptState,
                optimizer: optax.GradientTransformation,
                z_parent: jnp.ndarray,
                z_child: jnp.ndarray):
    
    def contrastive_loss(m):
        # 1. Compute Tangent Direction (Log Map)
        v_tan = jax.vmap(m.manifold.log_map)(z_parent, z_child)
        
        # 2. Retrieve Fields (Vectorized)
        # We access the internal metric (NeuralRanders)
        # We need H(x) and W(x)
        def get_fields(pt):
            H, W, _ = m.metric._get_zermelo_data(pt)
            return H, W

        get_fields_vmap = jax.vmap(get_fields)
        H_batch, W_batch = get_fields_vmap(z_parent)
        
        # 3. Alignment Loss (The Objective)
        # Minimize Energy? Or just maximize alignment?
        # Simple alignment: -<W, v> (using Euclidean/Minkowski dot for direction)
        # Better: Use the actual Finsler asymmetric norm, but dot product is stable.
        # We use the Minkowski dot product native to the manifold embedding.
        align_scores = -jax.vmap(m.manifold._minkowski_dot)(W_batch, v_tan)
        loss_align = jnp.mean(align_scores)
        
        # 4. Regularization (CRITICAL for stability)
        
        # A. Anchor H(x) to Identity (Prevents H -> 0, which allows W -> inf)
        dim = H_batch.shape[-1]
        I = jnp.eye(dim)
        loss_h_reg = jnp.mean((H_batch - I)**2)
        
        # B. Smoothness (Jacobian Penalty)
        # Penalize dW/dx to prevent chaotic vector fields
        def get_w_single(pt):
            _, W, _ = m.metric._get_zermelo_data(pt)
            return W
        
        # Jacobian of W w.r.t x (Forward mode is efficient here)
        # Result shape: (Batch, Dim_out, Dim_in)
        jac_fn = jax.vmap(jax.jacfwd(get_w_single))
        jacobians = jac_fn(z_parent)
        loss_smooth = jnp.mean(jacobians**2)
        
        # Weighted Sum
        # Alignment is ~1.0. 
        # Reg should be strong enough to anchor (1.0) 
        # Smoothness can be mild (0.1)
        total_loss = loss_align + 1.0 * loss_h_reg + 0.1 * loss_smooth
        
        return total_loss

    loss, grads = eqx.filter_value_and_grad(contrastive_loss)(model)
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    
    return new_model, new_opt_state, loss

class GeometricTrainer:
    """
    Dual-Phase Trainer for HAM.
    """
    def __init__(self, 
                 model: GeometricVAE, 
                 learning_rate: float = 1e-3, 
                 seed: int = 2025):
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        self.key = jax.random.PRNGKey(seed)

    def train_step_manifold(self, x_batch, v_batch, key):
        return step_manifold(self.model, self.opt_state, self.optimizer, x_batch, v_batch, key)

    def train_step_metric(self, z_parent, z_child):
        return step_metric(self.model, self.opt_state, self.optimizer, z_parent, z_child)

    def train(self, 
              dataset: BioDataset, 
              batch_size: int = 256, 
              epochs_manifold: int = 100,
              epochs_metric: int = 50):
        
        data_x, data_v = dataset.X, dataset.V
        num_samples = data_x.shape[0]
        
        print(f"=== Phase 1: Manifold Learning ({epochs_manifold} epochs) ===")
        for epoch in range(epochs_manifold):
            self.key, subkey = jax.random.split(self.key)
            perm = jax.random.permutation(subkey, num_samples)
            epoch_loss = 0.0
            
            for step in range(num_samples // batch_size):
                idx = perm[step*batch_size : (step+1)*batch_size]
                batch_x, batch_v = data_x[idx], data_v[idx]
                step_key = jax.random.fold_in(subkey, step)
                
                self.model, self.opt_state, loss, r, k, a = self.train_step_manifold(
                    batch_x, batch_v, step_key
                )
                epoch_loss += loss

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {epoch_loss:.4f} | Recon: {r:.4f} | KL: {k:.4f}")

        if dataset.lineage_pairs is None:
            print("No lineage pairs found. Skipping Phase 2.")
            return self.model

        print(f"=== Phase 2: Metric/Wind Learning ({epochs_metric} epochs) ===")
        lineage_pairs = dataset.lineage_pairs
        num_pairs = lineage_pairs.shape[0]
        
        for epoch in range(epochs_metric):
            self.key, subkey = jax.random.split(self.key)
            perm = jax.random.permutation(subkey, num_pairs)
            epoch_align_loss = 0.0
            
            for step in range(num_pairs // batch_size):
                idx = perm[step*batch_size : (step+1)*batch_size]
                pair_indices = lineage_pairs[idx]
                
                x_parents = data_x[pair_indices[:, 0]]
                x_children = data_x[pair_indices[:, 1]]
                
                key_p, key_c = jax.random.split(subkey, 2)
                # We stop gradients on Z here to focus purely on the Metric field training
                z_parents = jax.lax.stop_gradient(jax.vmap(lambda x: self.model.encode(x, key_p))(x_parents))
                z_children = jax.lax.stop_gradient(jax.vmap(lambda x: self.model.encode(x, key_c))(x_children))
                
                self.model, self.opt_state, loss = self.train_step_metric(
                    z_parents, z_children
                )
                epoch_align_loss += loss
                
            if epoch % 5 == 0:
                print(f"Epoch {epoch} | Lineage Align Loss: {epoch_align_loss:.4f}")
                
        return self.model