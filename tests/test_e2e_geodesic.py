"""
End-to-End Biological Inference Test (The Waddington Acid Test)
================================================================

Validates the HAM pipeline under the Initial Value Problem (IVP) paradigm:
  1. Base topology is Euclidean; complex geometry is learned via Pullback.
  2. The system is trained via Forward Shooting (leaving it alone to evolve).
  3. Evaluates if the learned Zermelo Wind autonomously recovers the biological flow.
"""

import unittest
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import NamedTuple, Tuple

from ham.training.losses import LossComponent, ReconstructionLoss, KLDivergenceLoss, KinematicPriorLoss, TimeAgnosticWindShooterLoss, OverdampedIVPShooterLoss
from ham.training.pipeline import TrainingPhase, HAMPipeline
from ham.geometry.surfaces import EuclideanSpace
from ham.models.learned import PullbackRanders
from ham.bio.vae import GeometricVAE
from ham.solvers.avbd import AVBDSolver # Kept for backward compatibility if needed

# ──────────────────────────────────────────────────────────
# 1. Biologically Realistic Data Generation (The Y-Branch)
# ──────────────────────────────────────────────────────────

def generate_waddington_branching_data(
    n: int = 600,
    data_dim: int = 50,
    latent_dim: int = 6,
    noise_level: float = 0.05,
    seed: int = 42,
):
    """
    Generates a high-dimensional dataset mirroring a biological 'Y' branch.
    Stem cells flow forward and bifurcate into two distinct terminal states.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    n_per_branch = n // 3
    
    # Time parameter (developmental pseudotime)
    t_trunk = jax.random.uniform(k1, (n_per_branch,), minval=0.0, maxval=1.0)
    t_branch = jax.random.uniform(k2, (n_per_branch * 2,), minval=1.0, maxval=2.0)
    
    # Trunk: Moves along y-axis
    z_trunk = jnp.stack([jax.random.normal(k3, (n_per_branch,)) * 0.05, t_trunk], axis=-1)
    
    # Branches: Move diagonally
    z_branch_a = jnp.stack([t_branch[:n_per_branch] - 1.0, t_branch[:n_per_branch]], axis=-1)
    z_branch_b = jnp.stack([-(t_branch[n_per_branch:] - 1.0), t_branch[n_per_branch:]], axis=-1)
    z_2d = jnp.concatenate([z_trunk, z_branch_a, z_branch_b], axis=0)
    
    # Heuristic Next States (dt = 0.1)
    dt = 0.1
    z_next_trunk = jnp.stack([z_trunk[:, 0], t_trunk + dt], axis=-1)
    z_next_a = jnp.stack([(t_branch[:n_per_branch] + dt) - 1.0, t_branch[:n_per_branch] + dt], axis=-1)
    z_next_b = jnp.stack([-((t_branch[n_per_branch:] + dt) - 1.0), t_branch[n_per_branch:] + dt], axis=-1)
    z_next_2d = jnp.concatenate([z_next_trunk, z_next_a, z_next_b], axis=0)

    # Lift to Ambient Latent Space
    rot_latent = jax.random.normal(k4, (latent_dim, 2))
    q_lat, _ = jnp.linalg.qr(rot_latent) 
    z_true = z_2d @ q_lat.T 
    z_next_true = z_next_2d @ q_lat.T

    # Dense Projection to Data Space
    rot_obs = jax.random.normal(jax.random.PRNGKey(99), (data_dim, latent_dim))
    q_obs, _ = jnp.linalg.qr(rot_obs)
    
    def embed_op(z):
        return jnp.tanh(z @ q_obs.T * 2.0)

    X = jax.vmap(embed_op)(z_true)
    X_next = jax.vmap(embed_op)(z_next_true)
    
    # Add noise
    X = X + jax.random.normal(jax.random.PRNGKey(100), X.shape) * noise_level
    X_next = X_next + jax.random.normal(jax.random.PRNGKey(101), X_next.shape) * noise_level

    return X, X_next, z_true

class RealisticPipelineDataset:
    def __init__(self, X, X_next):
        self.X = jnp.concatenate([X, X_next], axis=0)
        self.V = jnp.zeros_like(self.X)
        n = len(X)
        self.lineage_pairs = jnp.stack([jnp.arange(n), jnp.arange(n) + n], axis=1)

def get_filter_fn(selector):
    def filter_spec(model):
        base_mask = jax.tree_util.tree_map(lambda _: False, model)
        targets = selector(model)
        def make_true(n):
            return jax.tree_util.tree_map(
                lambda leaf: True if eqx.is_array(leaf) else False, n
            )
        if isinstance(targets, tuple):
            true_mask = tuple(make_true(t) for t in targets)
        else:
            true_mask = make_true(targets)
        return eqx.tree_at(selector, base_mask, replace=true_mask)
    return filter_spec


# ──────────────────────────────────────────────────────────
# 3. The Acid Tests
# ──────────────────────────────────────────────────────────

class TestTrueBiologicalInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.X_next, cls.z_true = generate_waddington_branching_data()
        ds = RealisticPipelineDataset(cls.X, cls.X_next)

        key = jax.random.PRNGKey(2026)
        k1, k2, k3 = jax.random.split(key, 3)
        latent_dim = 6
        manifold = EuclideanSpace(dim=latent_dim)
        
        decoder_net = eqx.nn.MLP(manifold.ambient_dim, 50, 128, 3, activation=jax.nn.gelu, key=k2)
        metric = PullbackRanders(manifold, decoder=decoder_net, key=k1, hidden_dim=32)
        
        # We pass a dummy solver to VAE to satisfy existing API
        solver = AVBDSolver(step_size=0.05, iterations=2) 
        model = GeometricVAE(data_dim=50, latent_dim=latent_dim, metric=metric, key=k3, solver=solver, decoder_net=decoder_net)
        
        # Phase 1: Pure VAE (Build the Euclidean Y-Branch)
        p1 = TrainingPhase(
            name="Manifold Learning",
            epochs=1000, 
            optimizer=optax.adam(1e-3),
            losses=[
                ReconstructionLoss(weight=1.0),
                KLDivergenceLoss(weight=1e-4), 
                KinematicPriorLoss(weight=1.0, margin=0.5)
            ],
            filter_spec=get_filter_fn(lambda m: (m.encoder_net, m.decoder_net)),
            requires_pairs=True, # Needs pairs for KinematicPrior
        )
        model_p1 = HAMPipeline(model).fit(ds, [p1], batch_size=64, seed=2026)

        # Sync the trained decoder into the Pullback Metric
        model_p1 = eqx.tree_at(lambda m: m.metric.decoder, model_p1, model_p1.decoder_net)

        # Phase 2: Forward IVP Geodesic Regression
        p2 = TrainingPhase(
            name="Metric Regression", 
            epochs=150, # Fast IVP epochs
            optimizer=optax.adam(1e-3),
            losses=[
                OverdampedIVPShooterLoss(weight=1.0)
            ],
            filter_spec=get_filter_fn(lambda m: m.metric.w_net),
            requires_pairs=True,
        )
        cls.model_p2 = HAMPipeline(model_p1).fit(ds, [p2], batch_size=64, seed=2026)

    def test_pullback_curvature(self):
        """
        Verifies that the Pullback Metric H(z) = J^T J is capturing the complex 
        geometry of the Y-branch rather than collapsing to a flat identity matrix.
        """
        keys = jax.random.split(jax.random.PRNGKey(0), 100)
        z_enc = jax.vmap(self.model_p2.encode)(self.X[:100], keys)
        
        def get_g_eigenvalues(z):
            h_matrix, _, _ = self.model_p2.metric._get_zermelo_data(z)
            evals, _ = jnp.linalg.eigh(h_matrix)
            return evals
        
        batch_evals = jax.vmap(get_g_eigenvalues)(z_enc)
        eval_variance = float(jnp.var(batch_evals[:, -1]))
        
        print(f"\n[Topology] Pullback Tensor Eigenvalue Variance: {eval_variance:.4f}")
        self.assertGreater(eval_variance, 0.1, 
                           "The VAE failed to warp the space. Pullback metric is flat.")

    def test_zermelo_causality_constraint(self):
        """
        Verifies the Zermelo condition h(W,W) < 1. 
        The Wind must be bounded by the Pullback metric's friction.
        """
        keys = jax.random.split(jax.random.PRNGKey(1), 100)
        z_enc = jax.vmap(self.model_p2.encode)(self.X[:100], keys)
        
        def check_zermelo(z):
            h_matrix, W, _ = self.model_p2.metric._get_zermelo_data(z)
            wind_norm_sq = jnp.dot(W, jnp.dot(h_matrix, W))
            return wind_norm_sq
            
        wind_norms = jax.vmap(check_zermelo)(z_enc)
        max_wind = float(jnp.max(jnp.sqrt(wind_norms)))
        
        print(f"\n[Causality] Maximum Latent Wind Speed (||W||_H): {max_wind:.4f}")
        self.assertLess(max_wind, 0.99, "Causality violated! Wind broke the speed limit.")

    def test_autonomous_wind_alignment(self):
        """
        The Ultimate Dynamical Test: 
        Does the learned Wind vector W(z), when pushed forward to the 50D space 
        via the Pullback Jacobian, align with the actual biological displacement?
        """
        N = 100
        X_s = self.X[:N]
        X_e = self.X_next[:N]
        keys = jax.random.split(jax.random.PRNGKey(2), N)
        
        true_disps = X_e - X_s
        true_mags = jnp.linalg.norm(true_disps, axis=-1)
        valid_mask = true_mags > 1e-4
        
        def pushforward_wind(x_start, k):
            z_s = self.model_p2.encode(x_start, k)
            _, W_latent, _ = self.model_p2.metric._get_zermelo_data(z_s)
            
            # Push Wind through Decoder Jacobian
            dec_fn = lambda z: self.model_p2.decode(z)
            _, W_data = jax.jvp(dec_fn, (z_s,), (W_latent,))
            return W_data
            
        W_data_preds = jax.vmap(pushforward_wind)(X_s, keys)
        
        eps = 1e-8
        dots = jnp.sum(W_data_preds * true_disps, axis=-1)
        norms_p = jnp.linalg.norm(W_data_preds, axis=-1) + eps
        norms_t = true_mags + eps
        cos_sims = dots / (norms_p * norms_t)
        
        valid_cosines = cos_sims[valid_mask]
        mean_cos = float(jnp.mean(valid_cosines))
        
        print(f"\n[Dynamics] Mean Cosine Similarity of Autonomous Flow: {mean_cos:.4f}")
        self.assertGreater(mean_cos, 0.40, 
                           "The autonomous Wind failed to capture the biological flow direction.")

if __name__ == "__main__":
    unittest.main()