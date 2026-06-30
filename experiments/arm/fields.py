"""Learnable C-space distance fields — the Stage-D target and the L2 adjoint probe.

These satisfy the :class:`DistanceField` interface (``q -> clearance``) with
trainable parameters, so they can be recovered from demonstrations by backprop
through the AVBD solver. ``MLPDistance`` is the blank field Stage D fits;
``ScaledDistance`` wraps a fixed field with a single learnable scalar, which gives
a clean one-parameter finite-difference target for the L2 adjoint gradcheck.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


class MLPDistance(eqx.Module):
    """A blank C-space clearance field ``q -> scalar`` parameterized by an MLP."""

    mlp: eqx.nn.MLP

    def __init__(self, dof: int, *, width: int = 32, depth: int = 2, key: jax.Array):
        self.mlp = eqx.nn.MLP(
            in_size=dof,
            out_size="scalar",
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, q: jax.Array) -> jax.Array:
        return self.mlp(q)


class ScaledDistance(eqx.Module):
    """A fixed distance field rescaled by one learnable scalar (adjoint gradcheck probe)."""

    base: Any
    scale: jax.Array

    def __init__(self, base: Any, scale: Any = 1.0):
        self.base = base
        self.scale = jnp.asarray(scale)

    def __call__(self, q: jax.Array) -> jax.Array:
        return self.scale * self.base(q)
