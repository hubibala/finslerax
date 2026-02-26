import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from ham.solvers.avbd import AVBDSolver

class GeodesicFlowTrainer:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        self.solver = AVBDSolver(step_size=0.1, iterations=15) # Fast settings for training

    @eqx.filter_jit
    def train_step(self, model, opt_state, z_parent, z_child):
        
        def loss_fn(m):
            # 1. Regress Geodesics (Solving the Inverse Problem)
            # "Find the path of least action from Parent to Child under current Metric"
            # Vectorized over batch
            solve_fn = jax.vmap(lambda s, e: self.solver.solve(m.metric, s, e, n_steps=8, train_mode=True))
            trajectories = solve_fn(z_parent, z_child)
            
            # 2. Action Loss (The "Principle of Least Action")
            # We want the 'cost' of the transition to be small.
            # In Randers: F(v) = |v|_M - <W, v>
            # Minimizing F encourages W to align with v.
            action_loss = jnp.mean(trajectories.energy)
            
            # 3. Regularization (Anchor)
            # Prevent the metric from shrinking to zero to cheat the energy loss.
            # We sample points along the path and force H(x) approx Identity
            def regularize_metric(x):
                M, W, _ = m.metric._get_fields(x) # Access internal fields
                dim = M.shape[-1]
                return jnp.mean((M - jnp.eye(dim))**2) + 0.1 * jnp.mean(W**2)
            
            # Sample a few points from the trajectories to regularize
            sample_pts = trajectories.xs[:, ::2, :] # Subsample
            reg_loss = jnp.mean(jax.vmap(jax.vmap(regularize_metric))(sample_pts))
            
            return action_loss + 1.0 * reg_loss

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        
        return new_model, new_opt_state, loss

    def train_phase2(self, dataset, epochs=50, batch_size=64):
        print(f"=== Phase 2: Geodesic Regression ({epochs} epochs) ===")
        
        if dataset.lineage_pairs is None:
            print("Error: No lineage pairs found!")
            return self.model
            
        pairs = dataset.lineage_pairs
        X = dataset.X
        n_pairs = pairs.shape[0]
        
        # Pre-encode all data (Phase 1 is frozen)
        # In a real rigorous setting, we might fine-tune encoder too, 
        # but freezing it is safer for stability.
        print("Pre-encoding dataset...")
        z_all = jax.vmap(lambda x: self.model.encode(x, jax.random.PRNGKey(0)))(X)
        z_all = jax.lax.stop_gradient(z_all)
        
        for epoch in range(epochs):
            perm = np.random.permutation(n_pairs)
            epoch_loss = 0
            
            for i in range(0, n_pairs, batch_size):
                idx = perm[i:i+batch_size]
                batch_pairs = pairs[idx]
                
                z_p = z_all[batch_pairs[:, 0]]
                z_c = z_all[batch_pairs[:, 1]]
                
                self.model, self.opt_state, loss = self.train_step(
                    self.model, self.opt_state, z_p, z_c
                )
                epoch_loss += loss
                
            if epoch % 5 == 0:
                print(f"Epoch {epoch} | Geodesic Action Loss: {epoch_loss:.4f}")
                
        return self.model