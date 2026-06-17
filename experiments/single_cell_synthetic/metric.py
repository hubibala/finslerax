"""Randers metric construction — base arms × wind variants (PLAN §6).

Assembles :class:`ham.geometry.zoo.Randers` metrics from a base "sea" tensor
``H(z)`` and a "wind" drift ``W(z)``:

Base-metric arms (the collapse question, §6.1)
    * **flat**      ``H = I``         — the design-doc reference / true-latent base.
    * **pullback**  ``H = JᵀJ + εI``  — decoder pullback (``PullbackGNet``); the
      collapse-prone HAM positioning metric we must characterise.
    * **conformal** ``H = c(x)·I``     — density-conformal (``ConformalEnergyBase``
      over a KDE); the SOTA-aligned arm that should degrade gracefully.

Wind variants (the asymmetry question, §6.2)
    * **full**           ``W ∝ f̂``               — the non-conservative Randers drift.
    * **potential-only** ``W ∝ ∇Φ̂`` (grad part)   — Hodge-predicted to fail at κ>0
      (negative control).
    * **β=0**            ``use_wind=False``         — symmetric-Riemannian control.

**Sign / navigability discipline.** The wind is the *current*: moving *with*
``f`` (down the developmental flow) is cheaper, so ``W = wind_scale · f``
directly (the marine convention; no one-form inversion since ``f`` is a
contravariant velocity).  ``wind_scale`` is sized by :func:`navigable_wind_scale`
so ``‖W‖_H < 1`` (the mild-wind regime, §5.2).  Learned/pullback winds use the
``"soft"`` causal clamp as a safety net; the trusted true metric uses ``"raw"``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.manifolds import EuclideanSpace
from ham.geometry.zoo import Randers
from ham.models.learned import ConformalEnergyBase, PullbackGNet
from ham.nn.kde import GaussianKDEEnergy

from .landscape import Landscape


# =============================================================================
# Small field wrappers (proper PyTrees so jit/vmap are clean)
# =============================================================================
class FlatSea(eqx.Module):
    """Identity sea ``H(z) = I`` (the flat / PCA reference base)."""

    dim: int = eqx.field(static=True)

    def __call__(self, z: jax.Array) -> jax.Array:
        return jnp.eye(self.dim, dtype=z.dtype)


class ScaledField(eqx.Module):
    """Wrap a vector field ``g(z)`` as a scaled wind ``W(z) = scale · g(z)``."""

    field: eqx.Module
    scale: float = eqx.field(static=True)

    def __call__(self, z: jax.Array) -> jax.Array:
        return self.scale * self.field(z)


class LandscapeDrift(eqx.Module):
    """The ground-truth drift ``f(z)`` of a :class:`Landscape` as a wind field."""

    landscape: Landscape
    scale: float = eqx.field(static=True)

    def __call__(self, z: jax.Array) -> jax.Array:
        return self.scale * self.landscape.drift(z)


# =============================================================================
# Navigability sizing (mild-wind regime)
# =============================================================================
def navigable_wind_scale(drift_fn, sea_fn, points, margin: float = 0.85) -> float:
    """Largest ``wind_scale`` keeping ``‖scale·f‖_H ≤ margin < 1`` over ``points``.

    Mirrors ``experiments/marine``'s navigability map: ``‖W‖²_H = Wᵀ H W`` must
    stay below 1 for the Zermelo corridor to be navigable.  Returns
    ``margin / max‖f‖_H`` so the worst-case point sits at ``margin``.
    """
    def w_norm(z):
        f = drift_fn(z)
        H = sea_fn(z)
        return jnp.sqrt(jnp.maximum(jnp.dot(f, jnp.dot(H, f)), 1e-12))
    worst = jnp.max(jax.vmap(w_norm)(jnp.asarray(points)))
    return float(margin / (worst + 1e-8))


# =============================================================================
# Builders
# =============================================================================
def build_true_metric(
    landscape: Landscape, points, *, margin: float = 0.85, dim: int = 2
) -> Randers:
    """The *true* Randers metric in the true latent (Stage A): ``H=I``, ``W∝f``.

    Uses ``wind_mode="raw"`` (trusted prescribed field) with a ``wind_scale``
    that :func:`navigable_wind_scale` guarantees is navigable, so the metric is
    the exact Zermelo travel-time geometry of the known drift.
    """
    manifold = EuclideanSpace(dim)
    sea = FlatSea(dim)
    scale = navigable_wind_scale(landscape.drift, sea, points, margin)
    wind = LandscapeDrift(landscape, scale)
    return Randers(manifold, sea, wind, epsilon=1e-5, use_wind=True, wind_mode="raw")


def build_randers(
    sea_fn,
    drift_fn,
    *,
    dim: int,
    points=None,
    wind_scale: float | None = None,
    margin: float = 0.85,
    use_wind: bool = True,
    wind_mode: str = "soft",
) -> Randers:
    """Generic Randers builder from a sea ``H(z)`` and a (raw) drift ``f(z)``.

    If ``wind_scale`` is ``None`` it is sized by :func:`navigable_wind_scale`
    over ``points``.  ``use_wind=False`` gives the symmetric-Riemannian control.
    """
    manifold = EuclideanSpace(dim)
    if isinstance(drift_fn, ScaledField):
        wind = drift_fn  # already scaled by the caller
    else:
        if wind_scale is None:
            if points is None:
                raise ValueError("provide wind_scale or points to size it")
            wind_scale = navigable_wind_scale(drift_fn, sea_fn, points, margin)
        wind = ScaledField(drift_fn, wind_scale)
    return Randers(manifold, sea_fn, wind, epsilon=1e-5, use_wind=use_wind, wind_mode=wind_mode)


def pullback_sea(decoder, dim: int) -> PullbackGNet:
    """Pullback sea ``H(z) = JᵀJ + εI`` from a decoder (the collapse-prone arm)."""
    return PullbackGNet(decoder=decoder, dim=dim)


def conformal_sea(points, dim: int, *, sigma: float = 0.5, beta: float = 1.0) -> ConformalEnergyBase:
    """Density-conformal sea ``H(z) = c(z)·I`` from a Gaussian-KDE energy.

    Low-density regions (off-manifold voids) get high energy → high conformal
    cost, so geodesics hug the data — the graceful-degradation arm (§6.1).
    """
    kde = GaussianKDEEnergy(centers=jnp.asarray(points, jnp.float32), sigma=sigma)
    return ConformalEnergyBase(ebm=kde, dim=dim, beta=beta)
