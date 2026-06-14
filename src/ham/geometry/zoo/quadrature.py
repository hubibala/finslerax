"""Segment-quadrature wrapper for discrete-action geodesic solvers.

The discrete path solvers (:class:`~ham.solvers.AVBDSolver`,
:class:`~ham.solvers.GaussNewtonGeodesic`) score a segment ``[x_i, x_{i+1}]`` by
``metric.energy(x_i, v_i)`` with ``v_i = x_{i+1} - x_i`` — i.e. the metric is
sampled at the segment **start vertex**.  For a stiff, data-driven conformal
metric this under-samples the cost along long segments: a segment leaping from an
on-data point (where the metric is cheap) across a high-cost void pays only the
cheap start-vertex cost and never "sees" the void, so the solver can *tunnel*
through empty regions (see ``spec/AVBD_LATENT_FINDINGS_2026-06-14.md``).

:class:`SegmentQuadratureMetric` wraps any :class:`~ham.geometry.metric.FinslerMetric`
and replaces the single start-vertex sample with a multi-point quadrature of the
energy *along* the segment,

    E_quad(x, v) = Σ_k w_k · base.energy(x + s_k · v, v),

so that a segment straddling a void pays the (high) interior cost.  This matches
the midpoint rule already used by :meth:`FinslerMetric.arc_length` and is a
strictly better discretisation of the continuous action ``∫ ½ F² dt``.

Quadrature rules (selected by the static ``nquad`` field):

* ``nquad=1`` — start vertex only ``{s=0, w=1}`` (reproduces the base metric;
  a no-op, useful for A/B comparison).
* ``nquad=2`` — **midpoint** ``{s=0.5, w=1}`` (default; the cheapest anti-tunnel
  rule).
* ``nquad=3`` — Simpson ``{0, 0.5, 1}`` with weights ``{1/6, 4/6, 1/6}``.

Note:
    Segment quadrature is a *discrete-action* choice.  For ``nquad>1`` the
    resulting ``metric_fn`` is intentionally **not** 1-homogeneous in ``v``
    (``F(x, λv)`` samples the base metric at shifted points ``x + s_k λ v``); this
    is the correct behaviour for a discretised path integral and is consistent
    with the midpoint quadrature in :meth:`FinslerMetric.arc_length`.  It is meant
    for the boundary-value path solvers, not for continuous spray/curvature
    queries — for those, use the unwrapped base metric.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.metric import FinslerMetric
from ham.utils.math import GRAD_EPS

__all__ = ["SegmentQuadratureMetric"]


class SegmentQuadratureMetric(FinslerMetric):
    """Wrap a base metric with multi-point segment quadrature of the energy.

    Args:
        base: The wrapped :class:`~ham.geometry.metric.FinslerMetric`.  Its
            ``manifold`` is reused, so the wrapper is a drop-in replacement in any
            solver expecting ``metric.energy`` / ``metric.manifold``.
        nquad: Number of quadrature nodes along each segment: ``1`` (start vertex,
            no-op), ``2`` (midpoint, default), or ``3`` (Simpson).
    """

    base: FinslerMetric
    nquad: int = eqx.field(static=True, default=2)

    def __init__(self, base: FinslerMetric, nquad: int = 2):
        super().__init__(manifold=base.manifold)
        if nquad not in (1, 2, 3):
            raise ValueError(f"nquad must be 1, 2, or 3; got {nquad}")
        self.base = base
        self.nquad = nquad

    def __repr__(self) -> str:
        return f"SegmentQuadratureMetric(base={self.base!r}, nquad={self.nquad})"

    def _quadrature(self):
        """Return (nodes, weights) for the static ``nquad`` rule (sum of weights 1)."""
        if self.nquad == 1:
            return (0.0,), (1.0,)
        if self.nquad == 2:
            return (0.5,), (1.0,)
        return (0.0, 0.5, 1.0), (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0)

    def energy(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Quadrature energy ``Σ_k w_k · base.energy(x + s_k v, v)``."""
        nodes, weights = self._quadrature()
        total = 0.0
        for s, w in zip(nodes, weights):
            total = total + w * self.base.energy(x + s * v, v)
        return total

    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Effective cost ``F = sqrt(2 · E_quad)`` (consistent with ``E = ½F²``)."""
        v_sq = jnp.sum(v**2, axis=-1)
        is_zero = v_sq < GRAD_EPS
        val = 2.0 * self.energy(x, v)
        return jnp.where(is_zero, 0.0, jnp.sqrt(jnp.maximum(val, GRAD_EPS)))
