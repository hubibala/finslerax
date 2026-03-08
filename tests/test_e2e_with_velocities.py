"""

End-to-End Geometric Learning Test

====================================



Tests the FULL pipeline: high-dimensional noisy data with dynamics

→ encode to latent manifold → learn Randers metric → geodesic tangents

→ decode back → compare with true dynamics.



This is the realistic scenario matching the biological application:

  1. Ground-truth: low-dim dynamics (rotation on a circle)

  2. Observation: nonlinear embedding into high-dim space + noise

  3. Learn: GeometricVAE (encoder/decoder) + NeuralRanders (metric)

  4. Verify: latent wind field tangents, decoded back to data space,

     recover the original dynamics.

"""



import unittest

import jax

import jax.numpy as jnp

import equinox as eqx

import optax

from typing import NamedTuple



from ham.geometry.surfaces import Hyperboloid

from ham.models.learned import NeuralRanders

from ham.bio.vae import GeometricVAE

from ham.training.losses import LossComponent, ReconstructionLoss, KLDivergenceLoss, GeodesicSprayLoss

from ham.training.pipeline import TrainingPhase, HAMPipeline

from ham.utils.math import safe_norm





# ──────────────────────────────────────────────────────────

# Synthetic high-dimensional data with known dynamics

# ──────────────────────────────────────────────────────────



class SyntheticHighDimDataset(NamedTuple):

    """High-dimensional observations of a low-dimensional dynamical system."""

    X: jnp.ndarray          # (N, D_high) observed states

    V: jnp.ndarray          # (N, D_high) observed velocities (dynamics)

    X_next: jnp.ndarray     # (N, D_high) next states (for pairs)

    z_true: jnp.ndarray     # (N, 2) ground-truth latent positions

    v_true: jnp.ndarray     # (N, 2) ground-truth latent velocities





def generate_rotating_circle_data(

    n: int = 500,

    data_dim: int = 50,

    latent_dim: int = 6,

    noise_level: float = 0.02,

    seed: int = 42,

) -> SyntheticHighDimDataset:

    """

    Ground truth: 2D orbit in a 6D latent manifold, rotated so it's dense and non-axis-aligned.

    Observation: A dense, non-linear projection into 50D space.

    """

    key = jax.random.PRNGKey(seed)

    k1, k2, k3, k4 = jax.random.split(key, 4)



    # 1. Start with a 2D circle signal

    angles = jax.random.uniform(k1, (n,), minval=0, maxval=2 * jnp.pi)

    z_2d = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)



    # 2. Lift to Ambient Latent Space (e.g. 7D for Sphere6)

    man_amb = latent_dim + 1

    rot_latent = jax.random.normal(k2, (man_amb, 2))

    q_lat, _ = jnp.linalg.qr(rot_latent)

    z_true = z_2d @ q_lat.T

   

    dt = 0.3

    omega = 1.0

    v_2d = dt * omega * jnp.stack([-jnp.sin(angles), jnp.cos(angles)], axis=-1)

    v_true = v_2d @ q_lat.T



    # 3. Dense Projection to Data Space

    # We use a random orthogonal matrix and a simple tanh non-linearity

    # to ensure the data is high-dim but has a 'smooth' manifold.

    rot_obs = jax.random.normal(k3, (data_dim, man_amb))

    q_obs, _ = jnp.linalg.qr(rot_obs)

   

    def embed_op(z):

        # A dense rotation followed by a smooth non-linearity

        return jnp.tanh(z @ q_obs.T * 2.0)



    X = jax.vmap(embed_op)(z_true)

    _, V = jax.vmap(lambda z, v: jax.jvp(embed_op, (z,), (v,)))(z_true, v_true)



    # Next state for lineage testing

    z_next = jnp.stack([jnp.cos(angles + dt * omega), jnp.sin(angles + dt * omega)], axis=-1) @ q_lat.T

    X_next = jax.vmap(embed_op)(z_next)



    # 4. Add noise

    X = X + jax.random.normal(k4, X.shape) * noise_level

    X_next = X_next + jax.random.normal(k1, X_next.shape) * noise_level



    return SyntheticHighDimDataset(X, V, X_next, z_true, v_true)





# ──────────────────────────────────────────────────────────

# Custom loss: directional wind alignment

# ──────────────────────────────────────────────────────────



class LatentVelocityAlignmentLoss(LossComponent):

    """

    Aligns W(z) DIRECTIONALLY with the JVP-projected velocity.



    Uses negative cosine similarity, because the Randers convexity

    constraint (|W|_H < 1) prevents matching magnitude — but

    the DIRECTION must be correct.

    """

    def __init__(self, weight: float = 1.0):

        super().__init__(weight, "VelAlign")



    def __call__(self, model, batch, key):

        x, v_data = batch[0], batch[1]

        z_mean, u_lat = model.project_control(x, v_data)

        _, W, _ = model.metric._get_zermelo_data(z_mean)



        u_norm = safe_norm(u_lat, axis=-1) + 1e-8

        w_norm = safe_norm(W, axis=-1) + 1e-8

        cos_sim = jnp.sum(u_lat * W, axis=-1) / (u_norm * w_norm)



        return (1.0 - jnp.mean(cos_sim)) * self.weight





# ──────────────────────────────────────────────────────────

# Utilities

# ──────────────────────────────────────────────────────────



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





class PipelineDataset:

    """Wraps synthetic data for HAMPipeline consumption."""

    def __init__(self, synth: SyntheticHighDimDataset):

        self.X = synth.X

        self.V = synth.V

        n = synth.X.shape[0]

        self.lineage_pairs = jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1)





def cosine_sim_batch(a, b):

    """Mean cosine similarity between two arrays of vectors."""

    na = safe_norm(a, axis=-1) + 1e-8

    nb = safe_norm(b, axis=-1) + 1e-8

    return float(jnp.mean(jnp.sum(a * b, axis=-1) / (na * nb)))





# ──────────────────────────────────────────────────────────

# Tests — trains once, validates incrementally

# ──────────────────────────────────────────────────────────



# Module-level cache: train once, test many

_trained_models = {}





def _get_trained_models():

    """Train Phase1 and Phase2 once, cache for all tests."""

    if _trained_models:

        return _trained_models



    latent_dim = 6

    synth = generate_rotating_circle_data(n=400, data_dim=50, latent_dim=latent_dim, noise_level=0.05, seed=42)



    key = jax.random.PRNGKey(2025)

    k1, k2 = jax.random.split(key)

    from ham.geometry.surfaces import Sphere

    manifold = Sphere(intrinsic_dim=latent_dim, radius=1.0)

    metric = NeuralRanders(manifold, k1, hidden_dim=32)

    model = GeometricVAE(data_dim=50, latent_dim=latent_dim, metric=metric, key=k2)

    ds = PipelineDataset(synth)



    # Phase 1: Train VAE (encoder/decoder)

    p1 = TrainingPhase(

        name="Manifold", epochs=2000, optimizer=optax.adam(1e-3),

        losses=[ReconstructionLoss(1.0), KLDivergenceLoss(1e-4)],

        filter_spec=get_filter_fn(lambda m: (m.encoder_net, m.decoder_net)),

        requires_pairs=False,

    )

    model_p1 = HAMPipeline(model).fit(ds, [p1], batch_size=64, seed=2025)



    # Phase 2: Train metric (wind field)

    p2 = TrainingPhase(

        name="Metric", epochs=1500, optimizer=optax.adam(1e-3),

        losses=[LatentVelocityAlignmentLoss(1.0), GeodesicSprayLoss(1)],

        filter_spec=get_filter_fn(lambda m: m.metric),

        requires_pairs=False,

    )

    model_p2 = HAMPipeline(model_p1).fit(ds, [p2], batch_size=64, seed=2025)



    _trained_models["synth"] = synth

    _trained_models["after_p1"] = model_p1

    _trained_models["after_p2"] = model_p2

    return _trained_models





class TestEndToEndGeodesicLearning(unittest.TestCase):

    """

    End-to-end test: high-dim noisy data → latent manifold → learned

    metric → dynamics recovery.

    """



    @classmethod

    def setUpClass(cls):

        """Train once for all tests."""

        models = _get_trained_models()

        cls.synth = models["synth"]

        cls.model_p1 = models["after_p1"]

        cls.model_p2 = models["after_p2"]



    def test_phase1_reconstruction(self):

        """VAE should learn to reconstruct high-dim data from latent space."""

        test_x = self.synth.X[:50]

        keys = jax.random.split(jax.random.PRNGKey(99), 50)



        def recon(x, k):

            z = self.model_p1.encode(x, k)

            return self.model_p1.decode(z)



        x_rec = jax.vmap(recon)(test_x, keys)

        mse = float(jnp.mean((test_x - x_rec) ** 2))



        print(f"\n[Phase1] Reconstruction MSE: {mse:.6f}")

        self.assertLess(mse, 0.05, f"Reconstruction MSE too high: {mse:.4f}")



    def test_latent_structure_preserves_topology(self):

        """Nearby data-space points should be nearby in latent space."""

        keys = jax.random.split(jax.random.PRNGKey(0), self.synth.X.shape[0])

        z_enc = jax.vmap(self.model_p1.encode)(self.synth.X, keys)



        # Sort by ground-truth angle

        z_true = self.synth.z_true

        angles = jnp.arctan2(z_true[:, 1], z_true[:, 0])

        idx = jnp.argsort(angles)

        z_sorted = z_enc[idx]



        dists = jnp.linalg.norm(z_sorted[1:] - z_sorted[:-1], axis=-1)

        median_d = float(jnp.median(dists))

        max_d = float(jnp.max(dists))



        print(f"\n[Topology] max/median = {max_d/median_d:.1f}")

        self.assertLess(max_d, median_d * 10,

                        f"Latent has fold-overs: max/median = {max_d/median_d:.1f}")



    def test_phase2_latent_velocity_alignment(self):

        """After metric training, W(z) should directionally align with

        the JVP-projected velocity in latent space."""

        test_x = self.synth.X[:100]

        test_v = self.synth.V[:100]



        def eval_pair(x, v):

            z, u_lat = self.model_p2.project_control(x, v)

            _, W, _ = self.model_p2.metric._get_zermelo_data(z)

            return u_lat, W



        u_lat, W_pred = jax.vmap(eval_pair)(test_x, test_v)

        cos = cosine_sim_batch(u_lat, W_pred)



        print(f"\n[Phase2] Latent velocity alignment (cosine): {cos:.4f}")

        self.assertGreater(cos, 0.65,

                           f"Wind should align with latent velocity, got cos={cos:.3f}")



    def test_full_pipeline_dynamics_recovery(self):

        """

        THE integration test: encode → wind field → decode ≈ true velocity.



        Validates the complete loop:

        1. Encoder maps high-dim data to latent manifold

        2. Metric's wind field captures dynamics direction in latent space

        3. Decoder maps predicted latent displacement back to data space

        4. Decoded velocity direction matches true data-space dynamics

        """

        test_x = self.synth.X[:100]

        test_v = self.synth.V[:100]

        eps = 0.01



        def predict_velocity(x):

            # Deterministic encoding via project_control (uses encoder mean)

            z_mean, _ = self.model_p2.project_control(x, jnp.zeros_like(x))

            _, W, _ = self.model_p2.metric._get_zermelo_data(z_mean)



            # Use exact pushforward (JVP) instead of finite difference

            _, v_pred = jax.jvp(self.model_p2.decode, (z_mean,), (W,))

            return v_pred



        v_pred = jax.vmap(predict_velocity)(test_x)

        cos = cosine_sim_batch(test_v, v_pred)



        print(f"\n[FullPipeline] Data-space dynamics recovery (cosine): {cos:.4f}")

        self.assertGreater(cos, 0.45,

                           f"Recovered dynamics should match observations, got cos={cos:.3f}")





if __name__ == "__main__":

    unittest.main()