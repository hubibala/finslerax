import jax
import jax.numpy as jnp
from ham.geometry.metric import FinslerMetric

def nonlinear_connection(metric: FinslerMetric, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the nonlinear connection N^i_j = \\partial G^i(x, v) / \\partial v^j.
    """
    return jax.jacfwd(metric.spray, argnums=1)(x, v)

def riemann_curvature_tensor(metric: FinslerMetric, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Riemann curvature tensor R^i_jk from the nonlinear connection:
    R^i_jk = \\partial N^i_j / \\partial x^k - \\partial N^i_k / \\partial x^j 
           + N^l_j \\partial N^i_k / \\partial v^l - N^l_k \\partial N^i_j / \\partial v^l
           
    Returns a tensor of shape (D, D, D) where D is the dimension of the manifold.
    R[i, j, k] corresponds to R^i_{jk}.
    """
    # N(x, v) is a (D, D) matrix mapping v to N*v
    def N_fn(pos, vel):
        return nonlinear_connection(metric, pos, vel)
        
    # \\partial N^i_j / \\partial x^k -> (D, D, D)  (i, j, k)
    dN_dx = jax.jacfwd(N_fn, argnums=0)(x, v)
    
    # \\partial N^i_j / \\partial v^k -> (D, D, D)  (i, j, k)
    dN_dv = jax.jacfwd(N_fn, argnums=1)(x, v)
    
    N = N_fn(x, v)
    
    # Term 1: \\partial N^i_j / \\partial x^k
    term1 = dN_dx
    
    # Term 2: - \\partial N^i_k / \\partial x^j
    term2 = -jnp.transpose(dN_dx, (0, 2, 1))
    
    # Term 3: N^l_j \\partial N^i_k / \\partial v^l
    # N is (l, j) ? No, N^i_j means N[i, j]. So l is the first index of N, j is the second.
    # dN_dv is \\partial N^i_k / \\partial v^l -> dN_dv[i, k, l]
    # So we want sum_l N[l, j] * dN_dv[i, k, l]
    term3 = jnp.einsum('lj,ikl->ijk', N, dN_dv)
    
    # Term 4: - N^l_k \\partial N^i_j / \\partial v^l
    term4 = -jnp.einsum('lk,ijl->ijk', N, dN_dv)
    
    R = term1 + term2 + term3 + term4
    return R

def sectional_curvature(metric: FinslerMetric, x: jnp.ndarray, v1: jnp.ndarray, v2: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the sectional curvature K(x, v1, v2) for the plane spanned by v1 and v2,
    with respect to the direction v1 (often taken as the flag pole in Finsler geometry).
    
    K = g_{im} R^i_{jk} v1^j v2^k v2^m / (||v1||^2 ||v2||^2 - <v1, v2>^2)
    where R^i_{jk} is evaluated at (x, v1).
    """
    R_tensor = riemann_curvature_tensor(metric, x, v1)
    
    # R^i = R^i_{jk} v1^j v2^k
    R_i = jnp.einsum('ijk,j,k->i', R_tensor, v1, v2)
    
    # Inner product g(R_i, v2) with respect to v1
    # Note: inner_product takes (x, v_ref, w1, w2)
    numerator = metric.inner_product(x, v1, R_i, v2)
    
    g_11 = metric.inner_product(x, v1, v1, v1)
    g_22 = metric.inner_product(x, v1, v2, v2)
    g_12 = metric.inner_product(x, v1, v1, v2)
    
    denominator = g_11 * g_22 - g_12**2
    
    # Safe division
    safe_denom = jnp.maximum(denominator, 1e-12)
    return jnp.where(denominator < 1e-12, 0.0, numerator / safe_denom)

def scalar_curvature(metric: FinslerMetric, x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes an averaged scalar curvature at point x.
    In Finsler geometry, scalar curvature is direction-dependent. As a simple metric, 
    we evaluate it by taking orthogonal basis vectors in the tangent space and averaging sectional curvatures.
    (Approximation for computational testing).
    """
    # Generate some tangent vectors to evaluate scalar curvature
    dim = metric.manifold.ambient_dim
    # We can just sample some random vectors and project to tangent space, or just build an orthonormal basis
    # But for a robust autodiff test, let's just pick one pair of vectors for surfaces (dim=2 or 3)
    
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    
    # For a general surface approximation:
    v1 = jax.random.normal(k1, (dim,))
    v2 = jax.random.normal(k2, (dim,))
    
    t1 = metric.manifold.to_tangent(x, v1)
    t1 = t1 / jnp.maximum(jnp.linalg.norm(t1), 1e-8)
    
    t2 = metric.manifold.to_tangent(x, v2)
    # Orthogonalize
    t2 = t2 - jnp.dot(t1, t2) * t1
    t2 = t2 / jnp.maximum(jnp.linalg.norm(t2), 1e-8)
    
    K = sectional_curvature(metric, x, t1, t2)
    return K
