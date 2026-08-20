"""
Curvature of a Finsler manifold, differentiated out of the geodesic spray.

The spray ``Gⁱ(x, y)`` follows from the energy, the Berwald coefficients
``Gⁱ_j = ∂Gⁱ/∂yʲ`` follow from the spray, and the curvatures below follow from
those, so no Christoffel symbol is ever written by hand and a metric cannot
disagree with its own curvature.

Four objects live here, and the literature often runs them together.
:func:`curvature_tensor` is ``Rⁱ_jk``, the obstruction to integrability of the
horizontal distribution. Contracting it with the direction gives the Jacobi
endomorphism ``R_y`` of :func:`riemannian_curvature`, whose trace is the Ricci
curvature and whose quadratic form is the flag curvature. Only the last of
these, :func:`sectional_curvature`, is the familiar Riemannian quantity, and it
exists only where the flagpole stops mattering.

The flagpole contracts into the *first* lower index,

    Rⁱ_k = Rⁱ_jk yʲ.

Sources differ on this, and ``Rⁱ_jk`` is antisymmetric in ``j`` and ``k``, so
the other choice flips the sign of both the flag and the Ricci curvature. This
one is pinned by ``K = +1`` on the round sphere and by
``Ric(y) = (n−1)λF²(x, y)`` at constant flag curvature ``λ``; the tests assert
both, against a Riemannian and a genuinely Finslerian metric respectively.

Performance note:
    The pipeline is third- and fourth-order autodiff: energy to spray, spray to
    Berwald coefficients, coefficients to curvature. Intermediates grow as
    ``O(D⁴)`` and XLA compilation grows sharply past ``D = 8``. Compile once with
    ``jax.jit`` and reuse. This is machinery for geometric analysis, not for an
    inner training loop.

Numerical stability:
    Every quantity here amplifies the Tikhonov term already present in the spray
    (``spray_reg``). For near-degenerate Randers metrics (``‖W‖_h → 1``) or for
    neural metrics early in training the values are indicative rather than exact.

Reference:
    ``spec/MATH_SPEC.md`` § 3.4. Bao, Chern & Shen, *An Introduction to
    Riemann-Finsler Geometry* (Springer GTM 200, 2000); Shen, *Lectures on
    Finsler Geometry* (World Scientific, 2001).
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp

from finslerax.geometry.metric import FinslerMetric
from finslerax.utils.math import NORM_EPS


def _berwald_coefficients(
    metric: FinslerMetric, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    r"""
    Berwald coefficients :math:`G^i_j = \partial G^i/\partial y^j`.

    The same quantity as :meth:`BerwaldConnection.connection_coefficients`,
    kept local so that curvature does not build a connection object per call.

    Args:
        metric: The Finsler metric.
        x: Position, shape (D,).
        y: Tangent vector, shape (D,).

    Returns:
        Coefficients, shape (D, D), indexed ``[i, j]``.
    """
    return jax.jacfwd(metric.spray, argnums=1)(x, y)


def curvature_tensor(
    metric: FinslerMetric, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    r"""
    Curvature tensor :math:`R^i_{jk}` of the Berwald connection.

    The obstruction to integrability of the horizontal distribution, and
    equivalently the Nijenhuis torsion of the horizontal projector,
    :math:`R(X, Y) = v[hX, hY]`. In coordinates,

    .. math::
        R^i_{jk} = \frac{\partial G^i_j}{\partial x^k}
                 - \frac{\partial G^i_k}{\partial x^j}
                 + G^m_j G^i_{km} - G^m_k G^i_{jm}.

    It vanishes identically exactly when the distribution is integrable.
    Antisymmetry in ``j`` and ``k`` holds to machine precision, for Finslerian
    metrics as well as Riemannian ones.

    Args:
        metric: The Finsler metric.
        x: Position, shape (D,).
        y: Direction, shape (D,).

    Returns:
        Curvature tensor, shape (D, D, D), indexed ``[i, j, k]``.
    """

    def coeffs(pos, vel):
        return _berwald_coefficients(metric, pos, vel)

    dG_dx = jax.jacfwd(coeffs, argnums=0)(x, y)  # [i, j, k] = dG^i_j/dx^k
    dG_dy = jax.jacfwd(coeffs, argnums=1)(x, y)  # [i, j, k] = G^i_jk
    G = coeffs(x, y)  # [m, j] = G^m_j

    return (
        dG_dx
        - jnp.transpose(dG_dx, (0, 2, 1))
        + jnp.einsum("mj,ikm->ijk", G, dG_dy)
        - jnp.einsum("mk,ijm->ijk", G, dG_dy)
    )


def riemannian_curvature(
    metric: FinslerMetric, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    r"""
    Jacobi endomorphism :math:`R_y = R(y, \cdot)`, i.e. :math:`R^i_k = R^i_{jk} y^j`.

    The operator appearing in the Jacobi equation, whose trace is the Ricci
    curvature and whose quadratic form is the flag curvature. Homogeneous of
    degree two in ``y``. The flagpole contracts into the first lower index;
    the module docstring says why.

    Args:
        metric: The Finsler metric.
        x: Position, shape (D,).
        y: Flagpole direction, shape (D,).

    Returns:
        Jacobi endomorphism, shape (D, D), indexed ``[i, k]``.
    """
    return jnp.einsum("ijk,j->ik", curvature_tensor(metric, x, y), y)


def ricci_curvature(
    metric: FinslerMetric, x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    r"""
    Ricci curvature :math:`\mathrm{Ric}(y) = \mathrm{tr}\, R_y`.

    Homogeneous of degree two in ``y``. On a manifold of constant flag
    curvature :math:`\lambda` it equals :math:`(n-1)\lambda F^2(x, y)`, which
    is the cheapest sharp check that a metric carries the curvature it claims.

    Args:
        metric: The Finsler metric.
        x: Position, shape (D,).
        y: Direction, shape (D,).

    Returns:
        Ricci curvature, shape ().
    """
    return jnp.trace(riemannian_curvature(metric, x, y))


def flag_curvature(
    metric: FinslerMetric, x: jnp.ndarray, y: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:
    r"""
    Flag curvature of the flag :math:`P = \mathrm{span}\{y, u\}` with flagpole ``y``.

    .. math::
        \mathbf{K}(P, y) = \frac{g_y\big(R_y(u),\, u\big)}
                                {g_y(y,y)\, g_y(u,u) - g_y(y,u)^2}

    It takes two arguments rather than one. A Riemannian sectional curvature
    depends only on the plane, but the fundamental tensor :math:`g_y` is
    itself direction-dependent, so raising the flag from a different edge of
    the same plane genuinely changes the number. Homogeneous of degree zero
    in both arguments.

    The denominator is the squared :math:`g_y`-area of the parallelogram on
    ``y`` and ``u``. It vanishes when the two are metrically parallel — a
    degenerate flag — where ``0.0`` is returned through a JAX-safe guard
    rather than a NaN.

    Args:
        metric: The Finsler metric.
        x: Position, shape (D,).
        y: Flagpole, shape (D,). Must be non-zero.
        u: Transverse edge, shape (D,). Independent of ``y``.

    Returns:
        Flag curvature, shape ().
    """
    R_u = riemannian_curvature(metric, x, y) @ u

    numerator = metric.inner_product(x, y, R_u, u)
    g_yy = metric.inner_product(x, y, y, y)
    g_uu = metric.inner_product(x, y, u, u)
    g_yu = metric.inner_product(x, y, y, u)
    denominator = g_yy * g_uu - g_yu**2

    safe = jnp.maximum(denominator, NORM_EPS)
    return jnp.where(denominator < NORM_EPS, 0.0, numerator / safe)


def sectional_curvature(
    metric: FinslerMetric,
    x: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    *,
    check: bool = True,
) -> jnp.ndarray:
    r"""
    Sectional curvature of the plane :math:`\mathrm{span}\{u, v\}`.

    Defined only where the flagpole does not matter, which is to say for
    Riemannian metrics. There it agrees with the flag curvature of the plane
    raised from either edge, with the classical sectional curvature of the
    Levi-Civita connection, and in two dimensions with the Gaussian curvature.

    The degeneration is automatic. A metric that declares itself Riemannian
    through :attr:`~finslerax.geometry.FinslerMetric.is_riemannian` takes the
    direct path; anything else is verified rather than assumed, by evaluating
    the same plane from both edges and requiring the two to agree. Passing a
    genuinely Finslerian metric therefore raises, rather than quietly
    returning one of several possible numbers.

    Args:
        metric: The Finsler metric.
        x: Position, shape (D,).
        u: First edge of the plane, shape (D,).
        v: Second edge, shape (D,). Independent of ``u``.
        check: Set False to skip the verification and halve the cost. Safe
            for a metric already known to be Riemannian.

    Returns:
        Sectional curvature, shape ().

    Raises:
        equinox.EquinoxRuntimeError: When the flag curvature of the plane
            depends on the flagpole, i.e. when the metric is not Riemannian.
            The check runs under ``jax.jit`` as well as eagerly.
    """
    K = flag_curvature(metric, x, u, v)
    if not check or metric.is_riemannian:
        return K

    K_swapped = flag_curvature(metric, x, v, u)
    return eqx.error_if(
        K,
        ~jnp.isclose(K, K_swapped, rtol=1e-4, atol=1e-6),
        "sectional_curvature is defined only for Riemannian metrics: the flag curvature "
        "of this plane depends on the flagpole. Use flag_curvature(metric, x, y, u), "
        "which takes the flagpole explicitly.",
    )


# ---------------------------------------------------------------------------
# Deprecated — removed in 2.0
# ---------------------------------------------------------------------------


def riemann_curvature_tensor(
    metric: FinslerMetric, x: jnp.ndarray, v: jnp.ndarray
) -> jnp.ndarray:
    """Deprecated alias for :func:`curvature_tensor`. Identical result."""
    warnings.warn(
        "riemann_curvature_tensor is deprecated and will be removed in 2.0; "
        "use curvature_tensor. Note that riemannian_curvature is a different "
        "object — the Jacobi endomorphism R^i_k = R^i_jk y^j.",
        DeprecationWarning,
        stacklevel=2,
    )
    return curvature_tensor(metric, x, v)


def flag_curvature_sample(
    metric: FinslerMetric, x: jnp.ndarray, key: jnp.ndarray
) -> jnp.ndarray:
    """
    Deprecated. Flag curvature at a random flag through ``x``.

    Draws a ``g``-orthonormal pair and evaluates :func:`flag_curvature` on it.
    Prefer passing the flag you actually mean. A Monte-Carlo sample over flags
    is rarely the question being asked, and it is not the Ricci scalar.
    """
    warnings.warn(
        "flag_curvature_sample is deprecated and will be removed in 2.0; "
        "draw the flag yourself and call flag_curvature.",
        DeprecationWarning,
        stacklevel=2,
    )
    dim = metric.manifold.ambient_dim
    k1, k2 = jax.random.split(key)

    t1 = metric.manifold.to_tangent(x, jax.random.normal(k1, (dim,)))
    t2 = metric.manifold.to_tangent(x, jax.random.normal(k2, (dim,)))

    g_11 = metric.inner_product(x, t1, t1, t1)
    t1 = t1 / jnp.maximum(jnp.sqrt(jnp.maximum(g_11, 0.0)), NORM_EPS)

    g_11 = metric.inner_product(x, t1, t1, t1)
    g_12 = metric.inner_product(x, t1, t1, t2)
    t2 = t2 - jnp.where(g_11 > NORM_EPS, g_12 / g_11, 0.0) * t1

    g_22 = metric.inner_product(x, t1, t2, t2)
    t2 = t2 / jnp.maximum(jnp.sqrt(jnp.maximum(g_22, 0.0)), NORM_EPS)

    return flag_curvature(metric, x, t1, t2)
