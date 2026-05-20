"""
Finsler curvature quantities derived from the geodesic spray via autodiff.

All curvature computations follow the Metric-First / Implicit Dynamics principle:
curvature is not computed from manually implemented Christoffel symbols but by
differentiating the spray G^i(x, v) that is itself derived from the energy E.

Mathematical reference:
    Bao-Chern-Shen, *An Introduction to Riemann-Finsler Geometry* (Springer GTM 200, 2000).
    Shen, *Lectures on Finsler Geometry* (World Scientific, 2001).

Performance note:
    The curvature pipeline involves 3rd-order autodiff: energy -> spray (2nd-order)
    -> nonlinear connection (3rd-order) -> Riemann tensor (4th-order). For D-dimensional
    inputs, intermediate Jacobians scale as O(D^4) and XLA compilation can take several
    minutes for D >= 8. Use jax.jit to compile once and reuse. The module is intended
    for geometric analysis and validation, not inner-loop training.

Numerical stability:
    All curvature computations amplify the Tikhonov regularization bias already present
    in the spray (spray_reg parameter of FinslerMetric). For near-degenerate Randers
    metrics (||W||_h -> 1) or neural metrics in early training, curvature values may
    be unreliable. Document this when using curvature for downstream analysis.

Sign convention:
    The Riemann curvature tensor follows R^i_jk = delta_k N^i_j - delta_j N^i_k,
    which gives positive flag curvature K = +1 for the unit sphere S^2.
    This matches Bao-Chern-Shen (2000), §6.2.
"""

import jax
import jax.numpy as jnp
from ham.geometry.metric import FinslerMetric
from ham.utils.math import safe_norm, NORM_EPS


def _nonlinear_connection(metric: FinslerMetric, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Nonlinear connection coefficients N^i_j = partial G^i / partial v^j.

    This is an internal helper. The returned (D, D) matrix satisfies the
    Euler homogeneity identity: N^i_j v^j = 2 G^i (for 2-homogeneous sprays).

    Args:
        metric: A FinslerMetric instance.
        x: Position, shape (D,).
        v: Velocity/direction, shape (D,).

    Returns:
        N: shape (D, D), where N[i, j] = N^i_j.
    """
    return jax.jacfwd(metric.spray, argnums=1)(x, v)


def riemann_curvature_tensor(metric: FinslerMetric, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Riemann curvature tensor R^i_jk of the nonlinear connection.

    Follows the formula (Bao-Chern-Shen, §6.2):
        R^i_jk = delta_k N^i_j - delta_j N^i_k
               = dN^i_j/dx^k - dN^i_k/dx^j
               + N^l_j dN^i_k/dv^l - N^l_k dN^i_j/dv^l

    where delta_k is the horizontal derivative of the nonlinear connection.

    Sign convention:
        This convention gives K = +1 for the unit sphere.
        The opposite sign (delta_j N^i_k - delta_k N^i_j) is also found in
        the literature (e.g., some editions of Bucataru-Miron).

    Antisymmetry:
        R^i_jk = -R^i_kj  (follows from the Jacobi identity for the horizontal frame).

    Args:
        metric: A FinslerMetric instance.
        x: Position, shape (D,).
        v: Reference direction (flagpole), shape (D,).

    Returns:
        R: shape (D, D, D), where R[i, j, k] = R^i_jk.
    """
    def N_fn(pos, vel):
        return _nonlinear_connection(metric, pos, vel)

    # dN^i_j / dx^k -> shape (D, D, D) as (i, j, k)
    dN_dx = jax.jacfwd(N_fn, argnums=0)(x, v)

    # dN^i_j / dv^k -> shape (D, D, D) as (i, j, k)
    dN_dv = jax.jacfwd(N_fn, argnums=1)(x, v)

    # Evaluate N once from the already-computed dN_dv via Euler's theorem:
    # N^i_j v^j = 2 G^i for 2-homogeneous sprays.
    # Alternatively we can just evaluate it directly.
    N = N_fn(x, v)

    # Term 1: +dN^i_j / dx^k
    term1 = dN_dx

    # Term 2: -dN^i_k / dx^j  (swap j <-> k indices)
    term2 = -jnp.transpose(dN_dx, (0, 2, 1))

    # Term 3: +N^l_j dN^i_k / dv^l
    # N[l, j], dN_dv[i, k, l] -> sum over l
    term3 = jnp.einsum('lj,ikl->ijk', N, dN_dv)

    # Term 4: -N^l_k dN^i_j / dv^l
    term4 = -jnp.einsum('lk,ijl->ijk', N, dN_dv)

    return term1 + term2 + term3 + term4


def sectional_curvature(metric: FinslerMetric, x: jnp.ndarray,
                        v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Flag (sectional) curvature K(x, v1; v2) for the plane spanned by v1 and v2.

    In Finsler geometry this is the *flag curvature* with flagpole v1 and
    transverse edge v2 (Bao-Chern-Shen, Definition 6.2.1):

        K(v1, v2) = g_{im}(x, v1) R^i_jk(x, v1) v1^j v2^k v2^m
                    -------------------------------------------------
                    g_v1(v1, v1) g_v1(v2, v2) - g_v1(v1, v2)^2

    where g_v1 = g(x, v1) is the fundamental tensor evaluated at the flagpole.

    The denominator is the squared area of the parallelogram spanned by v1, v2
    in the g_v1 metric.  It is zero when v1 and v2 are metric-parallel (degenerate
    flag plane); in that case, 0.0 is returned via a JAX-safe where guard.

    Note on the denominator guard:
        Uses NORM_EPS (1e-8) rather than 1e-12 to handle float32 precision.
        If the denominator is negative (indefinite metric), the result is
        mathematically undefined; a negative denominator indicates a bug in
        the metric definition.

    Args:
        metric: A FinslerMetric instance.
        x: Position, shape (D,).
        v1: Flagpole direction, shape (D,). Must be non-zero.
        v2: Transverse edge, shape (D,). Must be linearly independent of v1.

    Returns:
        K: Scalar flag curvature.
    """
    R_tensor = riemann_curvature_tensor(metric, x, v1)

    # Jacobi endomorphism: R^i = R^i_jk v1^j v2^k
    R_i = jnp.einsum('ijk,j,k->i', R_tensor, v1, v2)

    # Numerator: g_{im}(x, v1) R^i v2^m = g_v1(R_i, v2)
    numerator = metric.inner_product(x, v1, R_i, v2)

    # Denominator: area^2 of parallelogram in the g_v1 metric
    g_11 = metric.inner_product(x, v1, v1, v1)
    g_22 = metric.inner_product(x, v1, v2, v2)
    g_12 = metric.inner_product(x, v1, v1, v2)
    denominator = g_11 * g_22 - g_12**2

    # Safe division; NORM_EPS is appropriate for float32/64
    safe_denom = jnp.maximum(denominator, NORM_EPS)
    return jnp.where(denominator < NORM_EPS, 0.0, numerator / safe_denom)


def flag_curvature_sample(metric: FinslerMetric, x: jnp.ndarray,
                          key: jax.Array) -> jnp.ndarray:
    """
    Sample the flag curvature at x using a random pair of metric-orthogonal tangent vectors.

    This function computes the flag curvature for a single sampled tangent plane.
    For 2D manifolds this equals the Gaussian curvature. For higher dimensions,
    repeated calls (with different keys) provide a Monte-Carlo estimate of the
    curvature distribution.

    Note on Finsler direction-dependence:
        In Finsler geometry, all curvature quantities are direction-dependent:
        K = K(x, v1; v2). This function evaluates K at a randomly chosen flagpole
        v1 and transverse edge v2. The result is NOT the Ricci scalar (which
        requires summing over an orthonormal basis), but a single flag curvature
        sample.

    Gram-Schmidt orthogonalization:
        Uses the Finsler fundamental tensor g(x, v1) for orthogonalization and
        normalization, not the Euclidean inner product. This ensures the pair
        (v1, v2) is orthonormal w.r.t. the metric at the given flagpole.

    Args:
        metric: A FinslerMetric instance.
        x: Position, shape (D,).
        key: JAX PRNG key (following the JAX convention for stochastic functions).

    Returns:
        K: Scalar flag curvature for the sampled plane.
    """
    dim = metric.manifold.ambient_dim
    k1, k2 = jax.random.split(key)

    raw1 = jax.random.normal(k1, (dim,))
    raw2 = jax.random.normal(k2, (dim,))

    # Project to tangent space
    t1 = metric.manifold.to_tangent(x, raw1)
    t2 = metric.manifold.to_tangent(x, raw2)

    # Normalize t1 w.r.t. the metric
    g_t1_t1 = metric.inner_product(x, t1, t1, t1)
    t1 = t1 / jnp.maximum(jnp.sqrt(jnp.maximum(g_t1_t1, 0.0)), NORM_EPS)

    # Metric Gram-Schmidt: remove the t1 component from t2
    g_t1_t2 = metric.inner_product(x, t1, t1, t2)
    g_t1_t1_normed = metric.inner_product(x, t1, t1, t1)
    projection = jnp.where(g_t1_t1_normed > NORM_EPS,
                           g_t1_t2 / g_t1_t1_normed,
                           0.0)
    t2 = t2 - projection * t1

    # Normalize t2 w.r.t. the metric
    g_t2_t2 = metric.inner_product(x, t1, t2, t2)
    t2 = t2 / jnp.maximum(jnp.sqrt(jnp.maximum(g_t2_t2, 0.0)), NORM_EPS)

    return sectional_curvature(metric, x, t1, t2)


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------
# The old name 'scalar_curvature' was mathematically incorrect (it computed a
# single sectional curvature, not the Ricci trace). It is preserved as an
# alias with a deprecation note for any code that references it by name.
def scalar_curvature(metric: FinslerMetric, x: jnp.ndarray) -> jnp.ndarray:
    """
    .. deprecated::
        This function is retained for backward compatibility only.
        It is NOT the Ricci scalar curvature. For a single flag-curvature sample
        use ``flag_curvature_sample(metric, x, key)`` explicitly.

    Evaluates the flag curvature at a fixed canonical tangent plane (seed=42).
    The result is direction-dependent and only equals the Gaussian curvature
    for 2-dimensional surfaces.
    """
    key = jax.random.PRNGKey(42)
    return flag_curvature_sample(metric, x, key)
