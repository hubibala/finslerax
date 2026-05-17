"""Multi-phase declarative training pipeline for HAM models.

Provides TrainingPhase (a declarative description of one training stage) and
HAMPipeline (an orchestrator that executes phases sequentially with per-phase
parameter freezing, modular loss composition, and lineage-triple batching).

See also:
    spec/ARCH_SPEC.md § 6.4 -- Training Pipeline.
    ham.training.losses -- Modular loss components.
    examples/weinreb_smoke_test.py -- Minimal usage example.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import List, Callable, Tuple, Any
from dataclasses import dataclass
from ham.training.losses import LossComponent

@dataclass
class TrainingPhase:
    """Declarative description of one phase in the HAMPipeline training loop.

    Each phase specifies which parameters to train (via filter_spec), which
    losses to apply, and how many epochs to run. Phases are executed
    sequentially; model state from phase k carries into phase k+1. The
    weighted outputs of all active losses are summed to produce the total loss.

    Attributes:
        name: Human-readable label printed during training.
        epochs: Number of full passes through the data for this phase.
        optimizer: An optax.GradientTransformation (e.g., optax.adam(1e-3)).
        losses: List of LossComponent instances whose outputs are summed.
        filter_spec: Callable taking the model (eqx.Module) and returning a
            PyTree of booleans with the same structure, where True marks
            trainable leaves and False marks frozen leaves. Used by eqx.partition.
        requires_pairs: If True, the phase expects lineage pair or triple data.
            If neither dataset.lineage_pairs nor lineage_triples is available,
            the phase is skipped with a printed warning. Default: False.

    Example:
        >>> phase = TrainingPhase(
        ...     name='VAE',
        ...     epochs=100,
        ...     optimizer=optax.adam(1e-3),
        ...     losses=[ReconstructionLoss(), KLDivergenceLoss(weight=1e-4)],
        ...     filter_spec=lambda m: jax.tree_util.tree_map(eqx.is_array, m),
        ... )
    """
    name: str
    epochs: int
    optimizer: optax.GradientTransformation
    losses: List[LossComponent]
    filter_spec: Callable[[eqx.Module], Any]
    requires_pairs: bool = False

class HAMPipeline:
    """Orchestrates HAM model training through sequential declarative phases.

    For each TrainingPhase, the pipeline:
    1. Partitions the model into trainable / frozen parameters via filter_spec.
    2. Initializes the optimizer for the trainable partition.
    3. Runs the training loop with vmapped mini-batch gradient descent.
    4. Recombines the model for the next phase.

    The pipeline mutates self.model in-place across phases. The return value
    of fit() and self.model after fit() are the same object.

    Attributes:
        model: The eqx.Module being trained. Updated after each phase.

    See also:
        spec/ARCH_SPEC.md § 6.4, ham.training.losses, examples/weinreb_smoke_test.py.
    """
    def __init__(self, model: eqx.Module):
        """Args:
            model: An eqx.Module to train. Must be compatible with the
                filter_spec callables and LossComponent signatures in phases.
        """
        self.model = model

    def fit(self, dataset, phases: List[TrainingPhase], batch_size: int = 256,
            seed: int = 2025, lineage_triples: Any = None):
        """Execute the training pipeline.

        Runs each TrainingPhase in sequence with mini-batch gradient descent.

        Args:
            dataset: Object with attributes X (shape (N, D)), V (shape (N, D)),
                and optionally labels (shape (N,) or None), lineage_pairs
                (shape (P, 2) or None), and Traj_long.
            phases: List of TrainingPhase objects executed in order.
            batch_size: Number of samples per mini-batch. Default: 256.
                Note: tail samples are dropped if num_items % batch_size != 0.
            seed: Random seed for reproducibility. Default: 2025.
            lineage_triples: Optional array of shape (T, 3) with index triples
                (i, j, k) into dataset.X for lineage-aware losses. When provided
                and a phase has requires_pairs=True, triples take precedence over
                dataset.lineage_pairs.

        Returns:
            The trained eqx.Module. Same object as self.model (mutated in-place).

        Note:
            Phases with requires_pairs=True are silently skipped if no lineage
            data is available; a message is printed to stdout.
            Despite the name, requires_pairs controls both pair and triple
            batching modes.

        Example:
            >>> pipeline = HAMPipeline(model)
            >>> trained = pipeline.fit(dataset, [phase1, phase2], batch_size=128)
        """
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
            if phase.requires_pairs and dataset.lineage_pairs is None and lineage_triples is None:
                print("Skipping phase: requires lineage pairs/triples but none found.")
                continue

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
