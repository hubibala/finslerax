"""Covariate-encoder training through the differentiable eikonal solver.

One unbound :class:`~ham.models.wildfire.CovariateConditionedRanders` carries
the trainable weights; every step re-bakes one scene's rasters (constants) and
vmaps per-fire weather binding + CNN precompute + eikonal solve + dense
arrival loss over a fixed-size fire batch. Scenes rotate round-robin, so the
JIT cache holds one compiled step per scene shape.

The loss gauge follows the paper (absolute; ``alpha = 1``) after a short
affine warm-up (``alpha = 0``) that lets the encoder learn front *shapes*
before being billed for absolute speed.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ham.geometry.manifolds import EuclideanSpace
from ham.models.wildfire import CovariateConditionedRanders

from .ingest import SceneBundle, SceneNormalizer
from .medium import dense_arrival_loss, solve_arrival_encoder

CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def make_model(key, mode: str = "free", cnn_channels: int = 32,
               hidden_dim: int = 64) -> CovariateConditionedRanders:
    """Fresh encoder; ``mode`` is "free" (Gahtan drift) or "coupled" (T2)."""
    return CovariateConditionedRanders(
        EuclideanSpace(2), key,
        hidden_dim=hidden_dim, fuel_emb_dim=4, cnn_channels=cnn_channels,
        eps_G=0.1, max_G=10.0, max_b_norm=0.9, use_wind=mode,
    )


def scene_arrays(bundle: SceneBundle, normalizer: SceneNormalizer,
                 fire_idx: list[int]) -> dict:
    """Stack one scene's rasters + selected fires into jnp arrays."""
    elev_n, slope_n, canopy_n = normalizer.normalize_spatial(
        bundle.elev, bundle.slope, bundle.canopy
    )
    fires = [bundle.fires[i] for i in fire_idx]
    return {
        "elev": jnp.asarray(elev_n),
        "slope": jnp.asarray(slope_n),
        "aspect": jnp.asarray(bundle.aspect),
        "canopy": jnp.asarray(canopy_n),
        "fuel": jnp.asarray(bundle.fuel_codes),
        # Solver frame is pixels (see ingest.bind_scene_model): spacing 1.0.
        "spacing": 1.0,
        "shape": bundle.shape,
        "w4n": jnp.stack(
            [jnp.asarray(normalizer.normalize_weather(f.weather4_raw)) for f in fires]
        ),
        "wind": jnp.stack([jnp.asarray(f.measured_wind) for f in fires]),
        "ign": jnp.stack([jnp.asarray(f.ignition_rc) for f in fires]),
        "T_obs": jnp.stack([jnp.asarray(f.T_obs) for f in fires]),
        "burned": jnp.stack([jnp.asarray(f.burned) for f in fires]),
    }


def _batch_loss(model, sa, idx, alpha):
    scene_m = model.bind_scene_rasters(
        elev=sa["elev"], slope=sa["slope"], aspect=sa["aspect"],
        canopy=sa["canopy"], fuel_codes=sa["fuel"],
        pixel_spacing_m=sa["spacing"], origin_xy=jnp.zeros(2),
    )

    def one(i):
        fm = scene_m.bind_weather(sa["w4n"][i], measured_wind=sa["wind"][i])
        fm = fm.precompute_metric_field()
        T = solve_arrival_encoder(fm, sa["ign"][i], sa["shape"], sa["spacing"])
        return dense_arrival_loss(T, sa["T_obs"][i], sa["burned"][i], alpha)

    return jnp.mean(jax.vmap(one)(idx))


def curriculum(step: int, steps: int) -> float:
    """alpha: affine warm-up (first 20%), ramp to absolute by 50%, hold."""
    warm, ramp_end = 0.2 * steps, 0.5 * steps
    if step < warm:
        return 0.0
    return float(min((step - warm) / max(ramp_end - warm, 1), 1.0))


def train_encoder(
    bundles: list[SceneBundle],
    normalizer: SceneNormalizer,
    train_fire_idx: dict[str, list[int]],
    mode: str = "free",
    steps: int = 400,
    batch: int = 4,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[CovariateConditionedRanders, list]:
    """Train one encoder over the given scenes' training fires."""
    key = jax.random.PRNGKey(seed)
    model = make_model(key, mode=mode)
    arrays = [scene_arrays(b, normalizer, train_fire_idx[b.scene_id]) for b in bundles]

    optimizer = optax.adam(
        optax.exponential_decay(lr, transition_steps=200, decay_rate=0.85)
    )
    # Only float arrays train; static/int leaves are filtered out.
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step_fn(model, opt_state, sa, idx, alpha):
        loss, grads = eqx.filter_value_and_grad(_batch_loss)(model, sa, idx, alpha)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        s = step % len(arrays)
        sa = arrays[s]
        n_f = sa["w4n"].shape[0]
        idx = jnp.asarray(rng.choice(n_f, size=min(batch, n_f), replace=False))
        alpha = jnp.asarray(curriculum(step, steps))
        model, opt_state, loss = step_fn(model, opt_state, sa, idx, alpha)
        history.append(float(loss))
        if verbose and (step % 50 == 0 or step == steps - 1):
            print(f"    step {step:4d} [{bundles[s].scene_id}] "
                  f"alpha={float(alpha):.2f} loss={history[-1]:.4f}", flush=True)
    return model, history


def save_model(model, name: str):
    eqx.tree_serialise_leaves(CKPT_DIR / f"{name}.eqx", model)


def load_model(name: str, mode: str = "free") -> CovariateConditionedRanders:
    skeleton = make_model(jax.random.PRNGKey(0), mode=mode)
    return eqx.tree_deserialise_leaves(CKPT_DIR / f"{name}.eqx", skeleton)
