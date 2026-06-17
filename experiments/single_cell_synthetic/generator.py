"""SDE simulator → destructive snapshots → gene-space counts → a dataset.

Turns a :class:`~experiments.single_cell_synthetic.landscape.Landscape` into a
scRNA-seq-realistic :class:`SingleCellDataset` (PLAN §5.3):

1. **Progenitor cloud → clones.** ``n_clones`` ancestors near the progenitor;
   power-law clone sizes (realistic lineage statistics).
2. **SDE.** ``dx = f(x)dt + √(2D)dW`` (Euler–Maruyama), one independent
   trajectory per *observed* cell from its clone ancestor — clone-mates share an
   ancestor but have independent noise (shared origin, divergent fate).
3. **Destructive snapshots.** Each cell is observed at exactly one day
   ``{0..T-1}`` (you never see a cell twice).  A clone spanning several days
   yields the ``(early, mid, late)`` lineage triples.
4. **Nonlinear decoder → counts.** A fixed random smooth MLP ``x → rates`` makes
   PCA unable to trivially invert the embedding; rates → negative-binomial UMI
   counts with overdispersion + dropout (zero-inflation).
5. **Velocity emission.** The latent drift pushed through the deterministic
   count→PCA map (a JVP) gives a clean PCA-frame velocity, then corrupted with a
   documented noise model (per-dim scale, Gaussian, sign-flipped arrows) — the
   H2 SNR axis.

The decoder is a pure JAX function of ``x`` (frozen weights), so it is exactly
the kind of decoder ``PullbackGNet`` differentiates for the oracle pullback
metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .dataset import SingleCellDataset
from .embedding import PCAEmbedding
from .landscape import Landscape


# =============================================================================
# Config
# =============================================================================
@dataclass
class GeneratorConfig:
    """All knobs for one synthetic dataset (CPU-friendly defaults, PLAN §10)."""

    n_cells: int = 4000
    n_clones: int = 400
    clone_alpha: float = 2.0  # power-law exponent for clone sizes
    n_timepoints: int = 4
    dt: float = 1.0  # physical time between snapshots
    substeps: int = 50  # Euler–Maruyama substeps per unit time

    # progenitor cloud
    x0_mean: tuple[float, float] = (-2.0, 0.0)
    x0_spread: float = 0.15

    # gene space
    n_genes: int = 200
    latent_dim: int = 2  # PCA / embedding dim emitted as X_pca
    decoder_hidden: int = 64
    decoder_depth: int = 2
    decoder_scale: float = 3.0  # log-rate scale → realistic dynamic range
    library_size: float = 4000.0

    # negative-binomial count model
    nb_dispersion: float = 0.3  # smaller → more overdispersed (1/r style)
    dropout: float = 0.2  # zero-inflation probability

    # velocity noise model (H2 axis)
    vel_noise: float = 0.3  # Gaussian noise s.d. relative to clean velocity scale
    vel_flip_frac: float = 0.1  # fraction of sign-flipped "false-positive arrows"
    vel_bias: float = 0.0  # steady-state-style systematic shrink (0..1)

    seed: int = 0
    meta: dict = field(default_factory=dict)


# =============================================================================
# Fixed random nonlinear decoder x -> gene rates (pure JAX)
# =============================================================================
class RandomDecoder:
    """A frozen smooth MLP ``x ∈ ℝ² → rates ∈ ℝ^{D_gene}_{>0}``.

    Implemented with plain JAX arrays (not eqx) so it is trivially picklable /
    closure-friendly and differentiable via ``jax.jacfwd`` — the true decoder
    whose Jacobian defines the oracle pullback metric (PLAN §5.4).
    """

    def __init__(self, cfg: GeneratorConfig):
        key = jax.random.PRNGKey(cfg.seed + 17)
        dims = [2] + [cfg.decoder_hidden] * cfg.decoder_depth + [cfg.n_genes]
        self.Ws, self.bs = [], []
        for i in range(len(dims) - 1):
            key, k = jax.random.split(key)
            # He-ish init; keep tanh activations in a smooth regime.
            self.Ws.append(jax.random.normal(k, (dims[i + 1], dims[i])) * (1.5 / np.sqrt(dims[i])))
            self.bs.append(jnp.zeros(dims[i + 1]))
        self.scale = cfg.decoder_scale

    def __call__(self, x: jax.Array) -> jax.Array:
        h = x
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            h = jnp.tanh(W @ h + b)
        log_rate = self.Ws[-1] @ h + self.bs[-1]
        # Bounded log-rate → strictly positive rates with realistic dynamic range.
        return jnp.exp(self.scale * jnp.tanh(log_rate))


# =============================================================================
# SDE
# =============================================================================
def _simulate(landscape: Landscape, x0: jax.Array, n_timepoints: int, dt: float,
              substeps: int, key: jax.Array) -> jax.Array:
    """Integrate one trajectory, returning state at each snapshot day.

    Returns shape ``(n_timepoints, 2)`` — index 0 is the ancestor (day 0).
    """
    sub_dt = dt / substeps
    noise_scale = jnp.sqrt(2.0 * landscape.D * sub_dt)

    def one_interval(x, key):
        def em_step(x, k):
            x = x + landscape.drift(x) * sub_dt + noise_scale * jax.random.normal(k, x.shape)
            return x, None
        keys = jax.random.split(key, substeps)
        x, _ = jax.lax.scan(em_step, x, keys)
        return x, x

    keys = jax.random.split(key, n_timepoints - 1)
    _, later = jax.lax.scan(one_interval, x0, keys)  # (T-1, 2)
    return jnp.concatenate([x0[None], later], axis=0)


# =============================================================================
# Clone-size sampling
# =============================================================================
def _clone_sizes(n_clones: int, n_cells: int, alpha: float, rng) -> np.ndarray:
    """Power-law clone sizes summing (approximately) to ``n_cells`` (≥1 each)."""
    raw = rng.pareto(alpha, size=n_clones) + 1.0
    weights = raw / raw.sum()
    sizes = np.maximum(1, np.round(weights * n_cells).astype(int))
    # Trim/pad to hit n_cells exactly.
    diff = int(sizes.sum() - n_cells)
    order = np.argsort(-sizes)
    i = 0
    while diff > 0:
        j = order[i % n_clones]
        if sizes[j] > 1:
            sizes[j] -= 1
            diff -= 1
        i += 1
    while diff < 0:
        sizes[order[i % n_clones]] += 1
        diff += 1
        i += 1
    return sizes


# =============================================================================
# Velocity emission
# =============================================================================
def _emit_velocity(decoder: RandomDecoder, pca: PCAEmbedding, states: jax.Array,
                   drifts: jax.Array) -> jax.Array:
    """Clean PCA-frame velocity = J(count→PCA ∘ decoder)(x) · f(x), via JVP."""
    def embed_of_x(x):
        return pca.embed_rates(decoder(x))

    def push(x, f):
        _, jv = jax.jvp(embed_of_x, (x,), (f,))
        return jv

    return jax.vmap(push)(states, drifts)


# =============================================================================
# Top-level builder
# =============================================================================
def generate(landscape: Landscape, cfg: GeneratorConfig) -> SingleCellDataset:
    """Simulate a full synthetic single-cell dataset from a landscape."""
    rng = np.random.default_rng(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    # --- clones: ancestors + sizes + per-cell timepoint assignment ---
    sizes = _clone_sizes(cfg.n_clones, cfg.n_cells, cfg.clone_alpha, rng)
    x0_mean = jnp.asarray(cfg.x0_mean, dtype=jnp.float32)

    clone_ids, time_points, ancestors = [], [], []
    for c, sz in enumerate(sizes):
        anc = np.asarray(x0_mean) + cfg.x0_spread * rng.standard_normal(2)
        # Spread the clone's cells across timepoints so it spans ≥1 day; large
        # clones cover all days (yielding lineage triples), singletons land once.
        tps = rng.integers(0, cfg.n_timepoints, size=sz)
        if sz >= cfg.n_timepoints:
            tps[: cfg.n_timepoints] = np.arange(cfg.n_timepoints)  # guarantee span
        for t in tps:
            clone_ids.append(c)
            time_points.append(int(t))
            ancestors.append(anc)

    clone_id = np.asarray(clone_ids, dtype=np.int64)
    time_point = np.asarray(time_points, dtype=np.int64)
    ancestors = jnp.asarray(np.stack(ancestors), dtype=jnp.float32)
    n = clone_id.shape[0]

    # --- SDE: one trajectory per cell, read off at its assigned day ---
    keys = jax.random.split(key, n)
    sim = jax.vmap(
        lambda a, k: _simulate(landscape, a, cfg.n_timepoints, cfg.dt, cfg.substeps, k)
    )(ancestors, keys)  # (n, T, 2)
    states = sim[jnp.arange(n), jnp.asarray(time_point)]  # (n, 2) destructive read

    # --- oracle fields ---
    drifts = jax.vmap(landscape.drift)(states)
    hodge = jax.vmap(landscape.hodge_split)(states)
    true_grad, true_sol = hodge[0], hodge[1]
    fate_label = jax.vmap(landscape.fate_of)(states)
    branch = (states[:, 0] > landscape.x_b).astype(int)

    states_np = np.asarray(states)
    drifts_np = np.asarray(drifts)

    # --- gene space: decoder → NB counts with dropout ---
    decoder = RandomDecoder(cfg)
    rates = np.asarray(jax.vmap(decoder)(states))  # (n, D_gene), >0
    counts = _sample_nb_counts(rates, cfg, rng)

    # --- preprocessing + PCA (the real path) ---
    pca = PCAEmbedding.fit(counts, cfg.latent_dim, cfg.library_size)
    X_pca = pca.transform(counts)

    # --- velocity in the PCA frame: push latent drift, then corrupt ---
    vel_clean = np.asarray(_emit_velocity(decoder, pca, states, drifts))
    velocity_pca = _corrupt_velocity(vel_clean, cfg, rng)
    # --- oracle-frame velocity: the true latent drift, same corruption model ---
    velocity_true = _corrupt_velocity(drifts_np, cfg, rng)

    ds = SingleCellDataset(
        X_counts=counts,
        X_pca=X_pca.astype(np.float32),
        velocity_pca=velocity_pca.astype(np.float32),
        clone_id=clone_id,
        time_point=time_point,
        true_state=states_np.astype(np.float32),
        true_drift=drifts_np.astype(np.float32),
        true_grad=np.asarray(true_grad, dtype=np.float32),
        true_sol=np.asarray(true_sol, dtype=np.float32),
        fate_label=np.asarray(fate_label, dtype=np.int64),
        branch=np.asarray(branch, dtype=np.int64),
        velocity_true=velocity_true.astype(np.float32),
        decoder=lambda x, dec=decoder: np.asarray(dec(jnp.asarray(x))),
        pca_mean=pca.mean,
        pca_components=pca.components,
        meta={"kappa": landscape.kappa, "D": landscape.D, "seed": cfg.seed,
              "vel_noise": cfg.vel_noise, "vel_flip_frac": cfg.vel_flip_frac,
              **cfg.meta},
    )
    ds.build_triples(rng)
    return ds


def _sample_nb_counts(rates: np.ndarray, cfg: GeneratorConfig, rng) -> np.ndarray:
    """Negative-binomial UMI counts from mean ``rates`` + dropout zero-inflation.

    NB parametrised by mean ``μ = library·rate/Σrate`` and dispersion ``r`` (the
    gamma shape): variance ``μ + μ²/r`` (overdispersion).  Sampled as a
    Poisson–Gamma mixture; then a Bernoulli dropout mask zeroes entries.
    """
    lib = rates.sum(axis=1, keepdims=True) + 1e-8
    mu = rates * (cfg.library_size / lib)  # (n, D) per-gene mean
    r = cfg.nb_dispersion
    # Gamma–Poisson: λ ~ Gamma(shape=r, scale=μ/r), counts ~ Poisson(λ)
    lam = rng.gamma(shape=r, scale=mu / r)
    counts = rng.poisson(lam)
    if cfg.dropout > 0:
        mask = rng.random(counts.shape) < cfg.dropout
        counts = np.where(mask, 0, counts)
    return counts.astype(np.int64)


def _corrupt_velocity(vel_clean: np.ndarray, cfg: GeneratorConfig, rng) -> np.ndarray:
    """Apply the documented velocity noise model (PLAN §5.3, H2 axis)."""
    v = vel_clean.copy()
    scale = np.std(v) + 1e-8
    # systematic steady-state-style shrink toward zero
    v = (1.0 - cfg.vel_bias) * v
    # per-cell Gaussian noise
    v = v + cfg.vel_noise * scale * rng.standard_normal(v.shape)
    # sign-flipped "false-positive arrows"
    flip = rng.random(v.shape[0]) < cfg.vel_flip_frac
    v[flip] = -v[flip]
    return v
