import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple, Tuple, Callable
from functools import partial
import optax
import equinox as eqx

# ────────────────────────────────────────────────────────────────
# Your imports (adjust paths as needed)
# ────────────────────────────────────────────────────────────────
from ham.geometry.surfaces import EuclideanSpace, Hyperboloid, Sphere
from ham.geometry.metric import FinslerMetric
from ham.models.learned import NeuralRanders
from ham.solvers.avbd import AVBDSolver
from ham.utils.math import safe_norm

# ────────────────────────────────────────────────────────────────
# Data structures & helpers
# ────────────────────────────────────────────────────────────────

class SyntheticDataset(NamedTuple):
    starts: jnp.ndarray
    ends: jnp.ndarray

def cosine_similarity(true: jnp.ndarray, pred: jnp.ndarray) -> float:
    """Mean cosine similarity between two sets of vectors."""
    norms_true = safe_norm(true, axis=-1)
    norms_pred = safe_norm(pred, axis=-1)
    dots = jnp.sum(true * pred, axis=-1)
    return float(jnp.mean(dots / (norms_true * norms_pred)))


# ────────────────────────────────────────────────────────────────
# Scenario 1: River (constant rightward flow, flat)
# ────────────────────────────────────────────────────────────────

def generate_river_data(n_samples: int = 500) -> Tuple[SyntheticDataset, str, Callable]:
    key = jax.random.PRNGKey(0)
    starts = jax.random.uniform(key, (n_samples, 2), minval=-2.0, maxval=2.0)
    
    flow_dir = jnp.array([1.0, 0.0])
    step_size = 0.5
    noise = jax.random.normal(key, (n_samples, 2)) * 0.05
    
    ends = starts + flow_dir * step_size + noise
    
    def true_wind(x):
        return flow_dir * step_size
    
    return SyntheticDataset(starts, ends), "River Flow (Rightward)", true_wind


# ────────────────────────────────────────────────────────────────
# Scenario 2: Vortex (rotational, flat)
# ────────────────────────────────────────────────────────────────

def generate_vortex_data(n_samples: int = 500) -> Tuple[SyntheticDataset, str, Callable]:
    key = jax.random.PRNGKey(42)
    starts = jax.random.uniform(key, (n_samples, 2), minval=-2.0, maxval=2.0)
    
    dt = 0.3
    c, s = jnp.cos(dt), jnp.sin(dt)
    R = jnp.array([[c, -s], [s, c]])
    
    ends = jnp.dot(starts, R.T)
    
    def true_wind(x):
        v_rot = jnp.array([-x[1], x[0]])
        return 0.3 * v_rot / (safe_norm(v_rot) + 1e-8)
    
    return SyntheticDataset(starts, ends), "Vortex Flow (Counter-Clockwise)", true_wind


# ────────────────────────────────────────────────────────────────
# Scenario 3: Noisy Vortex on Hyperboloid (curved + noise)
# ────────────────────────────────────────────────────────────────

def generate_vortex_on_hyperboloid(n_samples: int = 500, noise_level: float = 0.15) -> Tuple[SyntheticDataset, str, Callable]:
    key = jax.random.PRNGKey(123)
    manifold = Hyperboloid(radius=1.0)
    
    starts = jax.vmap(manifold.random_sample, in_axes=(0, None))(
        jax.random.split(key, n_samples), ()
    )
    
    def raw_flow(x):
        v_rot = jnp.array([-x[1], x[0], 0.0])
        magnitude = 1.5 * jnp.exp(-5.0 * (x[2]**2))
        norm = safe_norm(v_rot)
        return magnitude * v_rot / norm
    
    # Tangent-projected wind for true flow
    def true_wind(x):
        raw_v = raw_flow(x)
        return manifold.to_tangent(x, raw_v)
    
    dt = 0.2
    noise_keys = jax.random.split(key, n_samples)
    
    def compute_delta(s, noise_key):
        tang_wind = true_wind(s)
        raw_noise = jax.random.normal(noise_key, shape=(3,)) * noise_level
        tang_noise = manifold.to_tangent(s, raw_noise)
        
        tang_delta = tang_wind * dt + tang_noise
        end = manifold.retract(s, tang_delta)
        return end
    
    ends = jax.vmap(compute_delta)(starts, noise_keys)
    
    return SyntheticDataset(starts, ends), "Noisy Vortex on Hyperboloid", true_wind


# ────────────────────────────────────────────────────────────────
# Scenario 4: Noisy Vortex on Sphere (compact curved + noise)
# ────────────────────────────────────────────────────────────────

def generate_vortex_on_sphere(n_samples: int = 500, noise_level: float = 0.1) -> Tuple[SyntheticDataset, str, Callable]:
    key = jax.random.PRNGKey(456)
    manifold = Sphere(radius=1.0)
    
    starts = jax.vmap(manifold.random_sample, in_axes=(0, None))(
        jax.random.split(key, n_samples), ()
    )
    
    def raw_flow(x):
        v_rot = jnp.array([-x[1], x[0], 0.0])
        magnitude = 1.5 * jnp.exp(-5.0 * (x[2]**2))  # milder than hyperboloid
        norm = safe_norm(v_rot)
        return magnitude * v_rot / norm
    
    def true_wind(x):
        raw_v = raw_flow(x)
        return manifold.to_tangent(x, raw_v)
    
    dt = 0.15  # slightly smaller
    noise_keys = jax.random.split(key, n_samples)
    
    def compute_delta(s, noise_key):
        tang_wind = true_wind(s)
        raw_noise = jax.random.normal(noise_key, shape=(3,)) * noise_level
        tang_noise = manifold.to_tangent(s, raw_noise)
        
        tang_delta = tang_wind * dt + tang_noise
        end = manifold.retract(s, tang_delta)
        return end
    
    ends = jax.vmap(compute_delta)(starts, noise_keys)
    
    return SyntheticDataset(starts, ends), "Noisy Vortex on Sphere", true_wind


# ────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────

class GeodesicRecoveryTrainer:
    def __init__(self, model, learning_rate: float = 1e-3, debug_mode: bool = False):
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        self.solver = AVBDSolver(step_size=0.1, iterations=30)
        self.debug_mode = debug_mode

    @eqx.filter_jit
    def train_step(self, model, opt_state, z_p, z_c):
        def loss_fn(m):
            # Solve BVP for each pair
            solve_fn = jax.vmap(lambda s, e: self.solver.solve(m, s, e, n_steps=8))
            trajectories = solve_fn(z_p, z_c)
            
            energy = trajectories.energy
            xs_last = trajectories.xs[:, -1]
            valid_mask = jnp.isfinite(energy) & (energy > -1e6) & (energy < 1e6)
            
            def log_invalid_callback(energy, xs_last, mask):
                n_inv = np.sum(~mask)
                if n_inv > 0:
                    print(f"Batch invalid trajectories: {n_inv}/{energy.shape[0]}")
                    print("Invalid energies:", energy[~mask])
                    print("Invalid final points norm:", np.linalg.norm(xs_last[~mask], axis=-1))

            jax.debug.callback(log_invalid_callback, energy, xs_last, valid_mask)
            
            loss_action = jnp.mean(jnp.where(valid_mask, energy, 1e8))  # huge penalty
            
            # Reg only on valid points
            def reg(x):
                M, W, _ = m._get_zermelo_data(x)
                loss_m = jnp.mean((M - jnp.eye(M.shape[-1]))**2)
                loss_w = jnp.mean(W**2)
                return (jnp.where(jnp.all(jnp.isfinite(M)) & jnp.all(jnp.isfinite(W)), 
                                 loss_m + 0.001 * loss_w, 1e6), M, W)
            
            reg_losses, Ms, Ws = jax.vmap(reg)(z_p)
            loss_reg = jnp.mean(reg_losses)
            
            total = loss_action + 10.0 * loss_reg
            return jnp.where(jnp.isfinite(total), total, 1e12)  # escape NaN

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss



# ────────────────────────────────────────────────────────────────
# Run & Visualize
# ────────────────────────────────────────────────────────────────

def run_test(scenario_name: str, dataset_gen_fn: Callable):
    print(f"\n=== {scenario_name} ===")
    
    result = dataset_gen_fn()
    if len(result) == 3:
        dataset, desc, true_flow = result
    else:
        dataset, desc = result
        true_flow = None
    
    print(f"Generated {len(dataset.starts)} path pairs.")
    
    manifold = EuclideanSpace(2) if "River" in scenario_name or "Vortex Flow" in scenario_name else Hyperboloid(radius=1.0) if "Hyperboloid" in scenario_name else Sphere(radius=1.0)
    
    key = jax.random.PRNGKey(2025)
    model = NeuralRanders(manifold, key, hidden_dim=32)
    
    trainer = GeodesicRecoveryTrainer(model, learning_rate=0.005, debug_mode=True)
    
    # Training loop
    batch_size = 64
    losses = []
    
    for epoch in range(50):
        perm = np.random.permutation(len(dataset.starts))
        epoch_loss = 0.0
        for i in range(0, len(dataset.starts), batch_size):
            idx = perm[i:i + batch_size]
            model, trainer.opt_state, loss = trainer.train_step(
                model, trainer.opt_state, dataset.starts[idx], dataset.ends[idx]
            )
            epoch_loss += float(loss)
        losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:02d} | Loss: {epoch_loss:.4f}")
    
    # ────────────────────────────────────────────────────────────────
    # Evaluation & Visualization
    # ────────────────────────────────────────────────────────────────
    print("Evaluating learned wind field...")
    
    # Grid on manifold (dense for curved)
    key_grid = jax.random.PRNGKey(777)
    grid_pts = jax.vmap(manifold.random_sample, in_axes=(0, None))(
        jax.random.split(key_grid, 400), ()
    )
    
    def get_learned_wind(z):
        _, W, _ = model._get_zermelo_data(z)
        return W
    
    learned_W = jax.vmap(get_learned_wind)(grid_pts)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title(f"{desc}\nLearned Randers Wind Field")
    
    ax.quiver(grid_pts[:, 0], grid_pts[:, 1],
              learned_W[:, 0], learned_W[:, 1],
              color='red', pivot='mid', scale=40)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"result_{scenario_name.replace(' ', '_').lower()}.png", dpi=300)
    print(f"Saved: result_{scenario_name.replace(' ', '_').lower()}.png")
    
    # Held-out cosine sim
    key_test = jax.random.PRNGKey(999)
    test_starts = jax.vmap(manifold.random_sample, in_axes=(0, None))(
        jax.random.split(key_test, 256), ()
    )
    
    if true_flow is not None:
        test_true_W = jax.vmap(true_flow)(test_starts)
        test_pred_W = jax.vmap(get_learned_wind)(test_starts)
        cos_sim_test = cosine_similarity(test_true_W, test_pred_W)
        print(f"Held-out cosine similarity: {cos_sim_test:.4f}")
    else:
        print("No ground-truth flow provided for this scenario.")


# ────────────────────────────────────────────────────────────────
# Run all
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_test("River Flow", 
             partial(generate_river_data, n_samples=600))
    run_test("Vortex Flow", 
             partial(generate_vortex_data, n_samples=600))
    run_test("Noisy Vortex on Sphere", 
             partial(generate_vortex_on_sphere, n_samples=600, noise_level=0.0))