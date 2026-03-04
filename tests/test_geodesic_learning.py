"""
Geodesic Learning Integration Tests
====================================

Verifies that the NeuralRanders metric can recover known synthetic vector
fields from trajectory pair data, using the modular training pipeline.

Strategy: Two-phase training via HAMPipeline.
  Phase 1 (Alignment):  Train W(x) to align with log_map(start, end).
  Phase 2 (Refinement): Optionally refine with geodesic action via AVBD.

Original intent preserved:
  - Scenario 1: Constant rightward flow on EuclideanSpace  ("River")
  - Scenario 2: Rotational flow on EuclideanSpace          ("Vortex")
  - Scenario 3: Rotational flow on Hyperboloid             ("Hyperboloid Vortex")
  - Scenario 4: Rotational flow on Sphere                  ("Sphere Vortex")
"""

import unittest
from functools import partial
from typing import Callable, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from ham.geometry.surfaces import EuclideanSpace, Hyperboloid, Sphere
from ham.geometry.metric import FinslerMetric
from ham.models.learned import NeuralRanders
from ham.training.losses import LossComponent
from ham.training.pipeline import TrainingPhase, HAMPipeline
from ham.utils.math import safe_norm

# ──────────────────────────────────────────────────────────
# Synthetic data helpers
# ──────────────────────────────────────────────────────────

class SyntheticDataset(NamedTuple):
    """Pairs of (start, end) points on a manifold."""
    starts: jnp.ndarray
    ends: jnp.ndarray


def generate_river_data(n: int = 400):
    """Constant rightward flow on R^2."""
    key = jax.random.PRNGKey(0)
    starts = jax.random.uniform(key, (n, 2), minval=-2.0, maxval=2.0)
    flow = jnp.array([1.0, 0.0])
    noise = jax.random.normal(key, (n, 2)) * 0.03
    ends = starts + flow * 0.5 + noise
    return SyntheticDataset(starts, ends), flow


def generate_vortex_data(n: int = 400):
    """Counter-clockwise rotation on R^2."""
    key = jax.random.PRNGKey(42)
    starts = jax.random.uniform(key, (n, 2), minval=-2.0, maxval=2.0)
    dt = 0.3
    c, s = jnp.cos(dt), jnp.sin(dt)
    R = jnp.array([[c, -s], [s, c]])
    ends = jnp.dot(starts, R.T)
    return SyntheticDataset(starts, ends), None  # no single flow vector


def generate_sphere_vortex(n: int = 400, noise: float = 0.0):
    """Rotational flow on S^2."""
    key = jax.random.PRNGKey(456)
    manifold = Sphere(radius=1.0)
    starts = jax.vmap(manifold.random_sample, in_axes=(0, None))(
        jax.random.split(key, n), ()
    )

    def true_wind(x):
        v = jnp.array([-x[1], x[0], 0.0])
        mag = 1.5 * jnp.exp(-5.0 * x[2] ** 2)
        return manifold.to_tangent(x, mag * v)

    dt = 0.15
    noise_keys = jax.random.split(key, n)

    def step(s, nk):
        tang = true_wind(s) * dt
        if noise > 0:
            raw_n = jax.random.normal(nk, (3,)) * noise
            tang = tang + manifold.to_tangent(s, raw_n)
        return manifold.retract(s, tang)

    ends = jax.vmap(step)(starts, noise_keys)
    return SyntheticDataset(starts, ends), true_wind

def generate_hyperboloid_vortex(n: int = 400, noise: float = 0.0):
    """Rotational flow on H^2 (upper sheet of two-sheeted hyperboloid).
    
    Samples points near the tip (x0 close to 1) where the wind field
    is well-defined and avoids the exponential spread of random_sample.
    """
    key = jax.random.PRNGKey(123)
    manifold = Hyperboloid(intrinsic_dim=2)

    # Sample points concentrated near the tip by using small tangent vectors
    origin = jnp.array([1.0, 0.0, 0.0])
    k1, k2 = jax.random.split(key)
    # Small tangent vectors → points stay close to the tip
    v_spatial = jax.random.normal(k1, (n, 2)) * 0.8  # scale controls spread
    v_tangent = jnp.concatenate([jnp.zeros((n, 1)), v_spatial], axis=1)
    starts = jax.vmap(manifold.exp_map, in_axes=(None, 0))(origin, v_tangent)

    def true_wind(x):
        # Rotation in spatial (x1, x2) plane, constant magnitude
        v_rot = jnp.array([0.0, -x[2], x[1]])
        return manifold.to_tangent(x, 0.5 * v_rot)

    dt = 0.3
    noise_keys = jax.random.split(k2, n)

    def step(s, nk):
        tang = true_wind(s) * dt
        if noise > 0:
            raw_n = jax.random.normal(nk, (3,)) * noise
            tang = tang + manifold.to_tangent(s, raw_n)
        return manifold.retract(s, tang)

    ends = jax.vmap(step)(starts, noise_keys)
    return SyntheticDataset(starts, ends), true_wind


# ──────────────────────────────────────────────────────────
# Modular losses for direct wind alignment
# ──────────────────────────────────────────────────────────

class DirectWindAlignmentLoss(LossComponent):
    """
    Aligns W(start) with the log_map displacement (start → end).
    
    This is the core learning signal: the wind should point from
    start to end with the correct magnitude.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "WindAlign")

    def __call__(self, model, batch, key):
        start, end = batch
        v_true = model.manifold.log_map(start, end)
        _, W, _ = model._get_zermelo_data(start)
        return jnp.mean((W - v_true) ** 2) * self.weight


class WindRegularizationLoss(LossComponent):
    """Light Jacobian penalty to encourage spatial smoothness of W."""
    def __init__(self, weight: float = 0.01):
        super().__init__(weight, "WindReg")

    def __call__(self, model, batch, key):
        start = batch[0]
        def get_w(pt):
            _, W, _ = model._get_zermelo_data(pt)
            return W
        jac = jax.jacfwd(get_w)(start)
        return jnp.mean(jac ** 2) * self.weight


class MetricIdentityLoss(LossComponent):
    """Anchors H(x) near Identity to prevent degenerate metric collapse."""
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "H_Id")

    def __call__(self, model, batch, key):
        start = batch[0]
        H, _, _ = model._get_zermelo_data(start)
        I = jnp.eye(H.shape[-1])
        return jnp.mean((H - I) ** 2) * self.weight


# ──────────────────────────────────────────────────────────
# Adapter: wraps NeuralRanders as a pipeline-compatible model
# ──────────────────────────────────────────────────────────

class MetricModel(eqx.Module):
    """Thin wrapper so the pipeline can treat the metric as a model."""
    metric: NeuralRanders
    manifold: object  # for LossComponent API compatibility

    def __init__(self, metric):
        self.metric = metric
        self.manifold = metric.manifold

    def _get_zermelo_data(self, x):
        return self.metric._get_zermelo_data(x)

    def encode(self, x, key):
        """Identity — data is already in latent space for this test."""
        return x


class PairDataset:
    """Minimal dataset wrapper compatible with HAMPipeline."""
    def __init__(self, starts, ends):
        self.X = starts
        self.V = ends  # V slot stores the "end" points for these tests
        n = starts.shape[0]
        self.lineage_pairs = jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1)


def _filter_all(model):
    """Unfreeze everything."""
    return jax.tree_util.tree_map(
        lambda leaf: True if eqx.is_array(leaf) else False, model
    )


def cosine_similarity(a, b):
    """Mean cosine similarity between two arrays of vectors."""
    na = safe_norm(a, axis=-1)
    nb = safe_norm(b, axis=-1)
    dots = jnp.sum(a * b, axis=-1)
    return float(jnp.mean(dots / (na * nb + 1e-8)))


# ──────────────────────────────────────────────────────────
# Training helper
# ──────────────────────────────────────────────────────────

def train_wind_field(manifold, dataset: SyntheticDataset,
                     epochs: int = 80, lr: float = 5e-3,
                     batch_size: int = 64, seed: int = 2025):
    """
    Train a NeuralRanders metric to recover the wind field from pair data.
    
    Uses direct alignment (MSE between W(start) and log_map(start, end)),
    which is far more stable than joint geodesic BVP optimization.
    """
    key = jax.random.PRNGKey(seed)
    metric = NeuralRanders(manifold, key, hidden_dim=32)
    model = MetricModel(metric)
    ds = PairDataset(dataset.starts, dataset.ends)

    phase = TrainingPhase(
        name="WindAlignment",
        epochs=epochs,
        optimizer=optax.adam(lr),
        losses=[
            DirectWindAlignmentLoss(weight=1.0),
            MetricIdentityLoss(weight=5.0),
            WindRegularizationLoss(weight=0.01),
        ],
        filter_spec=_filter_all,
        requires_pairs=False,  # We use X=starts, V=ends directly
    )

    pipeline = HAMPipeline(model)
    trained = pipeline.fit(ds, [phase], batch_size=batch_size, seed=seed)
    return trained


# ──────────────────────────────────────────────────────────
# Test Suite
# ──────────────────────────────────────────────────────────

class TestGeodesicLearning(unittest.TestCase):
    """
    Integration tests verifying that NeuralRanders can recover known
    synthetic vector fields from trajectory pair observations.
    """

    def test_river_direction(self):
        """Constant rightward flow: learned W should point right everywhere."""
        manifold = EuclideanSpace(2)
        dataset, true_flow = generate_river_data(400)
        trained = train_wind_field(manifold, dataset, epochs=60, lr=5e-3)

        # Evaluate on a grid
        eval_pts = jax.random.uniform(jax.random.PRNGKey(99), (100, 2),
                                      minval=-1.5, maxval=1.5)
        def get_w(pt):
            _, W, _ = trained._get_zermelo_data(pt)
            return W
        W_pred = jax.vmap(get_w)(eval_pts)

        # True flow is [1, 0] * 0.5
        true_W = jnp.broadcast_to(true_flow * 0.5, W_pred.shape)
        cos_sim = cosine_similarity(true_W, W_pred)

        print(f"[River] Cosine similarity: {cos_sim:.4f}")
        self.assertGreater(cos_sim, 0.85,
                           f"River wind should be aligned rightward, got cos={cos_sim:.3f}")

    def test_vortex_direction(self):
        """Rotational flow: learned W should capture CCW rotation."""
        manifold = EuclideanSpace(2)
        dataset, _ = generate_vortex_data(500)
        trained = train_wind_field(manifold, dataset, epochs=80, lr=5e-3)

        # Evaluate: true tangent direction at (x, y) is proportional to (-y, x)
        eval_pts = jax.random.uniform(jax.random.PRNGKey(99), (100, 2),
                                      minval=-1.5, maxval=1.5)
        def get_w(pt):
            _, W, _ = trained._get_zermelo_data(pt)
            return W
        W_pred = jax.vmap(get_w)(eval_pts)

        # True displacement per point
        true_disp = jax.vmap(manifold.log_map)(dataset.starts[:100], dataset.ends[:100])
        pred_disp = jax.vmap(get_w)(dataset.starts[:100])
        cos_sim = cosine_similarity(true_disp, pred_disp)

        print(f"[Vortex] Cosine similarity: {cos_sim:.4f}")
        self.assertGreater(cos_sim, 0.80,
                           f"Vortex wind should capture rotation, got cos={cos_sim:.3f}")

    def test_hyperboloid_vortex_direction(self):
        """Rotational flow on H^2: learned W should match tangent wind."""
        manifold = Hyperboloid(intrinsic_dim=2)
        dataset, true_wind_fn = generate_hyperboloid_vortex(500, noise=0.0)
        trained = train_wind_field(manifold, dataset, epochs=3000, lr=1e-3)

        eval_pts = jax.vmap(manifold.random_sample, in_axes=(0, None))(
            jax.random.split(jax.random.PRNGKey(999), 200), ()
        )
        def get_w(pt):
            _, W, _ = trained._get_zermelo_data(pt)
            return W
        W_pred = jax.vmap(get_w)(eval_pts)
        W_true = jax.vmap(true_wind_fn)(eval_pts)
        cos_sim = cosine_similarity(W_true, W_pred)

        print(f"[Hyperboloid] Cosine similarity: {cos_sim:.4f}")
        self.assertGreater(cos_sim, 0.70,
                           f"Hyperboloid wind should match true vortex, got cos={cos_sim:.3f}")

    def test_sphere_vortex_direction(self):
        """Rotational flow on S^2: learned W should match true tangent wind."""
        manifold = Sphere(radius=1.0)
        dataset, true_wind_fn = generate_sphere_vortex(500, noise=0.0)
        trained = train_wind_field(manifold, dataset, epochs=1000, lr=1e-3)

        # Evaluate on held-out points
        eval_pts = jax.vmap(manifold.random_sample, in_axes=(0, None))(
            jax.random.split(jax.random.PRNGKey(999), 200), ()
        )
        def get_w(pt):
            _, W, _ = trained._get_zermelo_data(pt)
            return W
        W_pred = jax.vmap(get_w)(eval_pts)
        W_true = jax.vmap(true_wind_fn)(eval_pts)

        # Scale: true_wind is normalized, so we compare directions
        cos_sim = cosine_similarity(W_true, W_pred)

        print(f"[Sphere] Cosine similarity: {cos_sim:.4f}")
        self.assertGreater(cos_sim, 0.70,
                           f"Sphere wind should match true vortex, got cos={cos_sim:.3f}")

    def test_loss_decreases(self):
        """Sanity: training loss should monotonically decrease (on average)."""
        manifold = EuclideanSpace(2)
        dataset, _ = generate_river_data(200)

        key = jax.random.PRNGKey(0)
        metric = NeuralRanders(manifold, key, hidden_dim=32)
        model = MetricModel(metric)

        loss_fn = DirectWindAlignmentLoss(weight=1.0)

        # Compute initial loss
        def eval_loss(m, starts, ends):
            def per_sample(s, e):
                return loss_fn(m, (s, e), jax.random.PRNGKey(0))
            return jnp.mean(jax.vmap(per_sample)(starts, ends))

        initial_loss = float(eval_loss(model, dataset.starts, dataset.ends))

        trained = train_wind_field(manifold, dataset, epochs=30, lr=5e-3)

        final_loss = float(eval_loss(trained, dataset.starts, dataset.ends))

        print(f"[LossCheck] Initial: {initial_loss:.4f} → Final: {final_loss:.4f}")
        self.assertLess(final_loss, initial_loss * 0.5,
                        "Loss should decrease significantly during training")


if __name__ == "__main__":
    unittest.main()
