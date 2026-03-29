import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import List, Callable, Tuple, Any
from dataclasses import dataclass
from ham.training.losses import LossComponent

@dataclass
class TrainingPhase:
    """Defines a phase in the training pipeline."""
    name: str
    epochs: int
    optimizer: optax.GradientTransformation
    losses: List[LossComponent]
    filter_spec: Callable[[eqx.Module], Any]
    requires_pairs: bool = False

class HAMPipeline:
    """Orchestrates the training of a HAM model through sequential declarative phases."""
    def __init__(self, model: eqx.Module):
        self.model = model

    def fit(self, dataset, phases: List[TrainingPhase], batch_size: int = 256, seed: int = 2025, lineage_triples: Any = None):
        key = jax.random.PRNGKey(seed)
        
        for phase in phases:
            print(f"=== Phase: {phase.name} ({phase.epochs} epochs) ===")
            
            # Setup parameter partitioning for this phase
            # True means trainable, False means frozen
            trainable_mask = phase.filter_spec(self.model)
            diff_model, static_model = eqx.partition(self.model, trainable_mask)
            
            opt_state = phase.optimizer.init(diff_model)
            
            # Define the loss function for the active parameters
            def loss_fn(diff, static, batch_data, step_key):
                full_model = eqx.combine(diff, static)
                total_loss = 0.0
                stats = {}
                for loss_comp in phase.losses:
                    loss_key, step_key = jax.random.split(step_key)
                    val = loss_comp(full_model, batch_data, loss_key)
                    total_loss += val
                    stats[loss_comp.name] = val
                return total_loss, stats
                
            # JIT compile the training step
            @eqx.filter_jit
            def train_step(diff, static, state, batch_data, step_key):
                # Vmap over batch axis
                batch_keys = jax.random.split(step_key, batch_data[0].shape[0])
                
                def batch_loss(d_m):
                    l, s = jax.vmap(loss_fn, in_axes=(None, None, 0, 0))(d_m, static, batch_data, batch_keys)
                    # Use eqx.tree_at / tree_map to average the aux stats correctly over the batch
                    return jnp.mean(l), jax.tree_util.tree_map(jnp.mean, s)
                
                (loss, stats), grads = eqx.filter_value_and_grad(batch_loss, has_aux=True)(diff)
                updates, new_state = phase.optimizer.update(grads, state, diff)
                new_diff = eqx.apply_updates(diff, updates)
                
                return new_diff, new_state, loss, stats

            # Training loop for the phase
            data_x, data_v = dataset.X, dataset.V
            num_samples = data_x.shape[0]
            
            if phase.requires_pairs and dataset.lineage_pairs is None and lineage_triples is None:
                print("Skipping phase: requires lineage pairs/triples but none found.")
                continue

            # Before the epoch loop, extract labels once
            data_x, data_v = dataset.X, dataset.V
            data_labels = dataset.labels if dataset.labels is not None \
                        else jnp.zeros(dataset.X.shape[0])  # fallback if no labels
            num_samples = data_x.shape[0]
                
            for epoch in range(phase.epochs):
                key, subkey = jax.random.split(key)
                epoch_loss = 0.0
                epoch_stats = {l.name: 0.0 for l in phase.losses}
                
                if phase.requires_pairs:
                    if lineage_triples is not None:
                        num_items = lineage_triples.shape[0]
                    else:
                        num_items = dataset.lineage_pairs.shape[0]
                    perm = jax.random.permutation(subkey, num_items)
                else:
                    num_items = num_samples
                    perm = jax.random.permutation(subkey, num_items)
                
                steps_per_epoch = max(1, num_items // batch_size)
                
                for step in range(steps_per_epoch):
                    idx = perm[step * batch_size : (step + 1) * batch_size]
                    
                    # Unpack batch data depending on what the phase requires
                    if phase.requires_pairs:
                        if lineage_triples is not None:
                            triple_idx = lineage_triples[idx]
                            i2 = triple_idx[:, 0]
                            i4 = triple_idx[:, 1]
                            i6 = triple_idx[:, 2]
                            batch_data = (data_x[i2], data_v[i2], data_labels[i2], data_x[i4], data_x[i6])
                        else:
                            pair_indices = dataset.lineage_pairs[idx]
                            batch_data = (data_x[pair_indices[:, 0]], data_x[pair_indices[:, 1]])
                            # If the dataset has long trajectories, pass them as the third element
                            if hasattr(dataset, "Traj_long"):
                                batch_data += (dataset.Traj_long[idx],)
                    else:
                         batch_data = (data_x[idx], data_v[idx], data_labels[idx])
                        
                    step_key = jax.random.fold_in(subkey, step)
                    
                    diff_model, opt_state, loss, stats = train_step(
                        diff_model, static_model, opt_state, batch_data, step_key
                    )
                    
                    epoch_loss += loss
                    for k_stat, v_stat in stats.items():
                        epoch_stats[k_stat] += v_stat
                        
                if epoch % 10 == 0 or epoch == phase.epochs - 1:
                    stat_str = " | ".join([f"{k}: {v/steps_per_epoch:.4f}" for k, v in epoch_stats.items()])
                    print(f"Epoch {epoch:03d} | Total Loss: {epoch_loss/steps_per_epoch:.4f} | {stat_str}")
            
            # Recombine the model for the next phase
            self.model = eqx.combine(diff_model, static_model)
            
        return self.model
