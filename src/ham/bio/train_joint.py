import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import time
from typing import Tuple

from ham.bio.vae import GeometricVAE

class GeometricTrainer:
    """
    Manager for training the Relativistic VAE.
    """
    def __init__(self, 
                 model: GeometricVAE, 
                 learning_rate: float = 1e-3, 
                 seed: int = 2025):
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        self.key = jax.random.PRNGKey(seed)

    @eqx.filter_jit
    def train_step(self, 
                   model: GeometricVAE, 
                   opt_state: optax.OptState, 
                   x_batch: jnp.ndarray, 
                   v_batch: jnp.ndarray, 
                   key: jax.random.PRNGKey):
        
        keys = jax.random.split(key, x_batch.shape[0])
        
        def batch_loss_fn(m):
            # Expecting 4 stats now
            losses, stats = jax.vmap(m.loss_fn)(x_batch, v_batch, keys)
            return jnp.mean(losses), stats

        (loss, (recon, kl, spray, align)), grads = eqx.filter_value_and_grad(batch_loss_fn, has_aux=True)(model)
        
        updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        
        # Return means of all stats
        return new_model, new_opt_state, loss, jnp.mean(recon), jnp.mean(kl), jnp.mean(spray), jnp.mean(align)

    def train(self, 
              data_x: jnp.ndarray, 
              data_v: jnp.ndarray, 
              batch_size: int = 256, 
              epochs: int = 1000,
              log_interval: int = 20): # Increased frequency
        
        num_samples = data_x.shape[0]
        num_steps_per_epoch = num_samples // batch_size
        
        print(f"Starting Training: {num_samples} cells, {epochs} epochs.")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            self.key, subkey = jax.random.split(self.key)
            perm = jax.random.permutation(subkey, num_samples)
            x_shuffled = data_x[perm]
            v_shuffled = data_v[perm]
            
            epoch_loss = 0.0
            # 4 accumulators
            epoch_stats = jnp.zeros(4)
            
            for step in range(num_steps_per_epoch):
                start = step * batch_size
                end = start + batch_size
                batch_x = x_shuffled[start:end]
                batch_v = v_shuffled[start:end]
                
                step_key = jax.random.fold_in(subkey, step)
                
                # Unpack 7 values
                self.model, self.opt_state, loss, r, k, s, a = self.train_step(
                    self.model, self.opt_state, batch_x, batch_v, step_key
                )
                
                epoch_loss += loss
                epoch_stats += jnp.array([r, k, s, a])
                
            if epoch % log_interval == 0:
                avg_loss = epoch_loss / num_steps_per_epoch
                avg_stats = epoch_stats / num_steps_per_epoch
                dt = time.time() - start_time
                
                # Format: Align should start near 0.0 and decrease (become negative)
                print(f"Epoch {epoch:04d} | Loss: {avg_loss:.4f} | "
                      f"Recon: {avg_stats[0]:.4f} | KL: {avg_stats[1]:.4f} | "
                      f"Spray: {avg_stats[2]:.4f} | Align: {avg_stats[3]:.4f} | "
                      f"Time: {dt:.2f}s")
                      
        return self.model