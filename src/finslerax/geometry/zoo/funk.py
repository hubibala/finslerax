"""Projectively flat Randers metrics of constant flag curvature."""

import equinox as eqx
import jax
import jax.numpy as jnp

from finslerax.geometry.manifold import Manifold
from finslerax.geometry.metric import FinslerMetric
from finslerax.utils.math import GRAD_EPS


class ProjectivelyFlatRanders(FinslerMetric):
    r"""Shen's standard model of projectively flat Randers metrics on the unit ball.

    On the open ball ``|x| < 1``, for a constant vector ``a`` with ``|a| < 1`` and
    ``ε = ±1``,

    .. math::
        F(x, y) = \frac{\sqrt{|y|^2 - (|x|^2|y|^2 - \langle x, y\rangle^2)}}{1 - |x|^2}
          + \varepsilon\left(\frac{\langle x, y\rangle}{1 - |x|^2}
                           + \frac{\langle a, y\rangle}{1 + \langle a, x\rangle}\right).

    Its flag curvature is the constant ``-1/4`` at every point, in every flag and
    from every flagpole, and Shen showed that every projectively flat Randers
    metric of non-zero constant flag curvature is isometric to one of these.
    With the default ``a = 0`` and ``ε = +1`` this is the standard Funk metric of
    the unit ball.

    That combination is what makes it useful here: a closed-form, genuinely
    non-Riemannian metric whose curvature is known exactly. A swapped index or a
    flipped sign in the curvature pipeline survives every Riemannian check and
    fails against this one.

    Reference:
        Shen, *Projectively flat Randers metrics with constant flag curvature*,
        Math. Ann. 325 (2003).
    """

    a: jax.Array
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        manifold: Manifold,
        a: jax.Array | None = None,
        epsilon: float = 1.0,
    ):
        """Initializes the projectively flat Randers metric.

        Args:
            manifold: The topological domain, which must carry the open unit
                ball. ``F`` blows up as ``|x| -> 1`` and is undefined beyond it.
            a: Constant drift vector, shape (D,), with ``|a| < 1``. Defaults to
                zero, which gives the Funk metric.
            epsilon: Orientation of the one-form, ``+1`` or ``-1``.
        """
        super().__init__(manifold=manifold)
        dim = manifold.ambient_dim
        self.a = jnp.zeros((dim,)) if a is None else jnp.asarray(a)
        if float(epsilon) not in (1.0, -1.0):
            raise ValueError(f"epsilon must be +1 or -1, got {epsilon!r}")
        self.epsilon = float(epsilon)

    def __repr__(self) -> str:
        return (
            f"ProjectivelyFlatRanders(manifold={self.manifold}, "
            f"a={self.a}, epsilon={self.epsilon})"
        )

    @property
    def flag_curvature_constant(self) -> float:
        """The exact flag curvature of the family, ``-1/4``."""
        return -0.25

    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """Computes F(x, v) for the standard model."""
        v_sq_raw = jnp.sum(v**2, axis=-1)
        is_zero = v_sq_raw < GRAD_EPS
        v_safe = jnp.where(is_zero[..., None], v + jnp.sqrt(GRAD_EPS), v)

        x_sq = jnp.sum(x**2, axis=-1)
        v_sq = jnp.sum(v_safe**2, axis=-1)
        xv = jnp.dot(x, v_safe)

        # |v|^2 - (|x|^2|v|^2 - <x,v>^2) = |v|^2 (1 - |x|^2) + <x,v>^2, which is
        # strictly positive inside the ball.
        discriminant = v_sq - (x_sq * v_sq - xv**2)
        one_minus_x_sq = jnp.maximum(1.0 - x_sq, GRAD_EPS)

        alpha = jnp.sqrt(jnp.maximum(discriminant, GRAD_EPS)) / one_minus_x_sq
        beta = self.epsilon * (
            xv / one_minus_x_sq + jnp.dot(self.a, v_safe) / (1.0 + jnp.dot(self.a, x))
        )
        return jnp.where(is_zero, 0.0, alpha + beta)
