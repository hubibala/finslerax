"""Preprocessing + embedding — the real scRNA-seq path, on synthetic counts.

Mirrors the standard single-cell preprocessing pipeline so that the synthetic
study exercises the *same* code the real Weinreb run will (PLAN §4, §5.3):

    counts → library-size normalize → log1p → (HVG) → z-score → PCA

plus an optional small autoencoder whose decoder feeds the **pullback** base
metric ``H(z) = JᵀJ`` (PLAN §6.1).

Everything is JAX so the deterministic ``embed_mean`` map is differentiable — the
generator pushes the latent drift through it with a JVP to produce a
PCA-frame velocity (`generator.emit_velocity`).
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Deterministic preprocessing transform (fit once, reused for velocity push)
# =============================================================================
@dataclass
class PCAEmbedding:
    """A fitted normalize→log1p→z-score→PCA transform.

    Stores the parameters so the *same* deterministic map can be applied to
    observed counts (to get ``X_pca``) and differentiated to push a latent
    velocity into PCA coordinates (to get ``velocity_pca``).
    """

    target_sum: float
    mean: np.ndarray  # (D_gene,) feature mean of log1p-normed counts
    std: np.ndarray  # (D_gene,)
    components: np.ndarray  # (d, D_gene) top-d principal axes
    d: int

    @staticmethod
    def fit(counts: np.ndarray, d: int, target_sum: float = 1e4) -> PCAEmbedding:
        """Fit the transform on observed integer ``counts`` (shape ``(n, D)``)."""
        x = _lognorm_np(counts, target_sum)
        mean = x.mean(axis=0)
        std = x.std(axis=0) + 1e-8
        xz = (x - mean) / std
        # Truncated PCA via SVD on the centred matrix.
        _u, _s, vt = np.linalg.svd(xz, full_matrices=False)
        comps = vt[:d]
        return PCAEmbedding(
            target_sum=float(target_sum),
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            components=comps.astype(np.float32),
            d=int(d),
        )

    def transform(self, counts: np.ndarray) -> np.ndarray:
        """Counts → PCA scores (numpy)."""
        x = _lognorm_np(counts, self.target_sum)
        xz = (x - self.mean) / self.std
        return (xz @ self.components.T).astype(np.float32)

    # ---- differentiable mean-rate path (for velocity push-forward) ----
    def embed_rates(self, rates: jax.Array) -> jax.Array:
        """Differentiable embedding of *mean* expression rates ``∈ ℝ^D``.

        Same arithmetic as :meth:`transform` but on continuous rates (no count
        sampling), so ``jax.jvp`` through it pushes a latent drift to PCA space.
        """
        lib = jnp.sum(rates) + 1e-8
        normed = rates * (self.target_sum / lib)
        x = jnp.log1p(normed)
        xz = (x - jnp.asarray(self.mean)) / jnp.asarray(self.std)
        return xz @ jnp.asarray(self.components).T


def _lognorm_np(counts: np.ndarray, target_sum: float) -> np.ndarray:
    counts = counts.astype(np.float64)
    lib = counts.sum(axis=1, keepdims=True) + 1e-8
    normed = counts * (target_sum / lib)
    return np.log1p(normed).astype(np.float32)


# =============================================================================
# Optional autoencoder — decoder feeds the pullback base metric H = JᵀJ
# =============================================================================
class MLPDecoder(eqx.Module):
    """A small smooth MLP decoder ``z -> ℝ^{D_out}`` (the pullback target)."""

    layers: list

    def __init__(self, d: int, out_dim: int, key, hidden: int = 64, depth: int = 2):
        keys = jax.random.split(key, depth + 1)
        dims = [d] + [hidden] * depth + [out_dim]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(len(dims) - 1)
        ]

    def __call__(self, z: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            z = jax.nn.tanh(layer(z))
        return self.layers[-1](z)


class _Encoder(eqx.Module):
    layers: list

    def __init__(self, in_dim: int, d: int, key, hidden: int = 64, depth: int = 2):
        keys = jax.random.split(key, depth + 1)
        dims = [in_dim] + [hidden] * depth + [d]
        self.layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]) for i in range(len(dims) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x)


def train_autoencoder(
    X_pca: np.ndarray,
    d: int,
    *,
    key,
    hidden: int = 64,
    depth: int = 2,
    steps: int = 1500,
    lr: float = 3e-3,
    batch: int = 256,
):
    """Train a tiny AE on PCA scores; return ``(encoder, decoder, final_loss)``.

    The decoder ``z -> X_pca`` is the frozen map fed to ``PullbackGNet``
    (``H = JᵀJ``) for the pullback metric arm (PLAN §6.1).  Latent dim ``d`` is
    the Stage-B collapse sweep axis.
    """
    import optax

    in_dim = X_pca.shape[1]
    ke, kd = jax.random.split(key)
    enc = _Encoder(in_dim, d, ke, hidden, depth)
    dec = MLPDecoder(d, in_dim, kd, hidden, depth)
    model = (enc, dec)
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    X = jnp.asarray(X_pca, dtype=jnp.float32)
    n = X.shape[0]

    @eqx.filter_jit
    def step(model, opt_state, xb):
        def loss_fn(m):
            enc, dec = m
            z = jax.vmap(enc)(xb)
            xr = jax.vmap(dec)(z)
            return jnp.mean((xr - xb) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = opt.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    rng = np.random.default_rng(0)
    loss = jnp.array(0.0)
    for _ in range(steps):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        model, opt_state, loss = step(model, opt_state, X[idx])
    enc, dec = model
    return enc, dec, float(loss)
