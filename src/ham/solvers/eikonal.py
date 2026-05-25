import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional
from ham.geometry.metric import FinslerMetric, AsymmetricMetric

# =============================================================================
# STENCIL UPDATES
# =============================================================================

@jax.custom_vjp
def steady_state_min(old_val, new_val):
    return jnp.minimum(old_val, new_val)

def _steady_state_min_fwd(old_val, new_val):
    return jnp.minimum(old_val, new_val), (old_val, new_val)

def _steady_state_min_bwd(res, g):
    old_val, new_val = res
    route_to_new = new_val <= old_val + 1e-4
    g_new = jnp.where(route_to_new, g, 0.0)
    g_old = jnp.where(route_to_new, 0.0, g)
    return g_old, g_new

steady_state_min.defvjp(_steady_state_min_fwd, _steady_state_min_bwd)

@jax.custom_vjp
def safe_sqrt(x):
    return jnp.sqrt(jnp.maximum(x, 0.0))

def _safe_sqrt_fwd(x):
    y = safe_sqrt(x)
    return y, y

def _safe_sqrt_bwd(y, g):
    dx = jnp.where(y > 1e-8, g / (2.0 * y), 0.0)
    return (dx,)

safe_sqrt.defvjp(_safe_sqrt_fwd, _safe_sqrt_bwd)

def sharp_min(a, b):
    return jnp.where(a <= b, a, b)

def compute_two_point_update(T1: jax.Array, T2: jax.Array, 
                             g11: jax.Array, g12: jax.Array, g22: jax.Array, 
                             b1: jax.Array, b2: jax.Array, 
                             m1x: float, m1y: float, m2x: float, m2y: float, 
                             eps: float = 1e-12) -> jax.Array:
    """
    Computes the 2-point (triangular) upwind update for the anisotropic Eikonal PDE:
        (grad T - B)^T G^-1 (grad T - B) = 1
    """
    e11 = g11 * m1x**2 + 2 * g12 * (m1x * m1y) + g22 * m1y**2
    e22 = g11 * m2x**2 + 2 * g12 * (m2x * m2y) + g22 * m2y**2
    e12 = g11 * (m1x * m2x) + g12 * (m1x * m2y + m1y * m2x) + g22 * (m1y * m2y)
    
    detE = e11 * e22 - e12**2
    det_safe = jnp.maximum(detE, eps)
    q11 = e22 / det_safe
    q12 = -e12 / det_safe
    q22 = e11 / det_safe
    
    S1 = T1 + m1x * b1 + m1y * b2
    S2 = T2 + m2x * b1 + m2y * b2
    
    alpha = q11 + 2 * q12 + q22
    beta = (q11 + q12) * S1 + (q12 + q22) * S2
    gamma = q11 * S1**2 + 2 * q12 * S1 * S2 + q22 * S2**2 - 1.0
    
    discriminant = beta**2 - alpha * gamma
    alpha_safe = jnp.maximum(alpha, eps)
    
    t0_2pt = (beta + safe_sqrt(discriminant)) / alpha_safe
    t0_2pt = jnp.maximum(t0_2pt, jnp.maximum(T1, T2))
    
    lam1 = q11 * (t0_2pt - S1) + q12 * (t0_2pt - S2)
    lam2 = q12 * (t0_2pt - S1) + q22 * (t0_2pt - S2)
    
    inside_triangle = (lam1 >= 0) & (lam2 >= 0)
    valid = (detE > eps) & (alpha > eps) & (t0_2pt < 1e4) & inside_triangle
    
    return jnp.where(valid, t0_2pt, 1e5)

def compute_one_point_update(T_nbr: jax.Array, 
                             g11: jax.Array, g12: jax.Array, g22: jax.Array, 
                             b1: jax.Array, b2: jax.Array, 
                             mx: float, my: float) -> jax.Array:
    """Computes the 1-point (edge) update."""
    m_G_m = g11 * mx**2 + 2 * g12 * mx * my + g22 * my**2
    distance = safe_sqrt(m_G_m)
    m_dot_b = mx * b1 + my * b2
    return T_nbr + distance + m_dot_b

def stencil_update(T1: jax.Array, T2: jax.Array, T0: jax.Array, 
                   g11: jax.Array, g12: jax.Array, g22: jax.Array, 
                   b1: jax.Array, b2: jax.Array, 
                   m1x: float, m1y: float, m2x: float, m2y: float) -> jax.Array:
    """Takes the minimum of valid 2-point and 1-point updates."""
    t0_2pt = compute_two_point_update(T1, T2, g11, g12, g22, b1, b2, m1x, m1y, m2x, m2y)
    t0_1pt_m1 = compute_one_point_update(T1, g11, g12, g22, b1, b2, m1x, m1y)
    t0_1pt_m2 = compute_one_point_update(T2, g11, g12, g22, b1, b2, m2x, m2y)
    
    # 2-point overrides 1-point if valid and strictly smaller.
    t0_1pt = sharp_min(t0_1pt_m1, t0_1pt_m2)
    t0_candidate = sharp_min(t0_2pt, t0_1pt)
    return steady_state_min(T0, t0_candidate)

# =============================================================================
# SWEEPING PASSES
# =============================================================================

def sweep_y(T: jax.Array, G: jax.Array, B: jax.Array, source_mask: jax.Array, 
            hx: float, hy: float, direction: int) -> jax.Array:
    """
    Sweeps along axis 0 (Y-axis). 
    direction=1: top to bottom (row i depends on row i-1).
    direction=-1: bottom to top (row i depends on row i+1).
    """
    def scan_fn(T_prev, xs):
        T_curr, g11, g12, g22, b1, b2, s_mask = xs
        
        padded_T_prev = jnp.pad(T_prev, (1, 1), constant_values=1e5)
        
        T2 = T_prev
        T1_left = padded_T_prev[0:-2]
        T0 = T_curr
        
        # Vector points from current (i,j) to neighbor (i_nbr, j_nbr)
        # For Y sweep, neighbor is in Y direction.
        p = -direction * hy
        
        m2x = 0.0
        m2y = p
        
        # Left neighbor: (i_nbr, j-1)
        m1x_left = -1.0 * hx
        m1y_left = p
        
        T0_new1 = stencil_update(T1_left, T2, T0, g11, g12, g22, b1, b2, 
                                 m1x_left, m1y_left, m2x, m2y)
                                 
        # Right neighbor: (i_nbr, j+1)
        T1_right = padded_T_prev[2:]
        m1x_right = 1.0 * hx
        m1y_right = p
        
        T0_new2 = stencil_update(T1_right, T2, T0_new1, g11, g12, g22, b1, b2,
                                 m1x_right, m1y_right, m2x, m2y)
        
        T_curr_out = jnp.where(s_mask, 0.0, T0_new2)
        return T_curr_out, T_curr_out

    # Unrolling massively speeds up GPU/TPU by reducing scan dispatch overhead, 
    # but destroys CPU performance by causing instruction cache bloat.
    unroll_amt = 1 if jax.default_backend() == 'cpu' else 4

    if direction == 1:
        xs = (T[1:], G[0, 1:], G[1, 1:], G[2, 1:], B[0, 1:], B[1, 1:], source_mask[1:])
        _, T_out = jax.lax.scan(scan_fn, T[0], xs, unroll=unroll_amt)
        return jnp.concatenate([T[0:1], T_out], axis=0)
    else:
        xs = (T[:-1], G[0, :-1], G[1, :-1], G[2, :-1], B[0, :-1], B[1, :-1], source_mask[:-1])
        _, T_out = jax.lax.scan(scan_fn, T[-1], xs, reverse=True, unroll=unroll_amt)
        return jnp.concatenate([T_out, T[-1:]], axis=0)

def sweep_all(T: jax.Array, G: jax.Array, B: jax.Array, source_mask: jax.Array, 
              hx: float, hy: float) -> jax.Array:
    """Executes 4 directional sweeps (Y+, X+, Y-, X-)."""
    # 1. Sweep Y +1 (top to bottom)
    T = sweep_y(T, G, B, source_mask, hx, hy, 1)
    
    # 2. Sweep X +1 (left to right)
    # Transpose the domain to run a Y sweep along the X axis
    G_T = jnp.stack([G[2].T, G[1].T, G[0].T], axis=0)
    B_T = jnp.stack([B[1].T, B[0].T], axis=0)
    T = sweep_y(T.T, G_T, B_T, source_mask.T, hx=hy, hy=hx, direction=1).T
    
    # 3. Sweep Y -1 (bottom to top)
    T = sweep_y(T, G, B, source_mask, hx, hy, -1)
    
    # 4. Sweep X -1 (right to left)
    T = sweep_y(T.T, G_T, B_T, source_mask.T, hx=hy, hy=hx, direction=-1).T
    
    return T

# =============================================================================
# SOLVER & IMPLICIT GRADIENTS
# =============================================================================

@jax.custom_vjp
def _fast_sweeping_solve(G, B, source_mask, hx, hy, max_iters, tol):
    def cond_fun(val):
        T, max_diff, iters = val
        return (max_diff > tol) & (iters < max_iters)
        
    def body_fun(val):
        T, _, iters = val
        T_new = sweep_all(T, G, B, source_mask, hx, hy)
        diff = jnp.abs(T_new - T)
        max_diff = jnp.max(jnp.where(jnp.isnan(diff), 0.0, diff))
        return T_new, max_diff, iters + 1
        
    T_init = jnp.where(source_mask, 0.0, 1e5)
    final_val = jax.lax.while_loop(cond_fun, body_fun, (T_init, jnp.array(1e5), 0))
    return final_val[0]

def _solve_fwd(G, B, source_mask, hx, hy, max_iters, tol):
    T_final = _fast_sweeping_solve(G, B, source_mask, hx, hy, max_iters, tol)
    return T_final, (T_final, G, B, source_mask, hx, hy, max_iters)

def _solve_bwd(res, g_T):
    T_final, G, B, source_mask, hx, hy, max_iters = res
    
    def single_sweep(T_in, G_in, B_in):
        return sweep_all(T_in, G_in, B_in, source_mask, hx, hy)
        
    # Get the vector-Jacobian product function at the steady state
    _, vjp_fn = jax.vjp(single_sweep, T_final, G, B)
    
    def cond_fun(val):
        x_curr, x_prev, iters = val
        return (jnp.max(jnp.abs(x_curr - x_prev)) > 1e-6) & (iters < max_iters)
        
    def body_fun(val):
        x_curr, _, iters = val
        # dT is the J_T^T x_curr vector
        dT, _, _ = vjp_fn(x_curr)
        x_next = g_T + dT
        return x_next, x_curr, iters + 1
        
    # Solve adjoint system: x = g_T + J_T^T x
    x_init = jnp.zeros_like(g_T)
    x_prev = jnp.full_like(g_T, 1e9)
    x_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, (x_init, x_prev, 0))
    
    # Compute gradients with respect to metric using the converged adjoint state
    _, dG, dB = vjp_fn(x_final)
    
    dG = jnp.where(jnp.isnan(dG), 0.0, dG)
    dB = jnp.where(jnp.isnan(dB), 0.0, dB)
    
    return dG, dB, None, None, None, None, None

_fast_sweeping_solve.defvjp(_solve_fwd, _solve_bwd)

# =============================================================================
# EIKONAL SOLVER MODULE
# =============================================================================

class EikonalSolver(eqx.Module):
    max_iters: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    
    def __init__(self, max_iters: int = 50, tol: float = 1e-4):
        self.max_iters = max_iters
        self.tol = tol

    @eqx.filter_jit
    def solve(self, metric: FinslerMetric, source_coords: jax.Array, 
              grid_extent: Tuple[float, float, float, float], 
              grid_shape: Tuple[int, int]) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Solves the fully anisotropic Eikonal equation for multiple sources.
        """
        assert isinstance(metric, AsymmetricMetric), "Anisotropic solver strictly requires an AsymmetricMetric (e.g. Randers)."
        
        nx, ny = grid_shape
        xmin, xmax, ymin, ymax = grid_extent
        
        x = jnp.linspace(xmin, xmax, nx)
        y = jnp.linspace(ymin, ymax, ny)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        coords = jnp.stack([X, Y], axis=-1)
        
        hx = float((xmax - xmin) / max(1, nx - 1))
        hy = float((ymax - ymin) / max(1, ny - 1))
        
        def extract_GB(pt):
            # Zermelo data maps to Eikonal Godunov via inverse matrix
            H, W, lam = metric.zermelo_data(pt)
            B = - jnp.dot(H, W) / lam
            HW = jnp.dot(H, W)
            G = (H + jnp.outer(HW, HW) / lam) / lam
            return G, B
            
        G_mat, B_vec = jax.vmap(jax.vmap(extract_GB))(coords)
        
        # Format for fast sweeping (channels, nx, ny)
        g11 = G_mat[..., 0, 0]
        g12 = G_mat[..., 0, 1]
        g22 = G_mat[..., 1, 1]
        G = jnp.stack([g11, g12, g22], axis=0) 
        
        b1 = B_vec[..., 0]
        b2 = B_vec[..., 1]
        B = jnp.stack([b1, b2], axis=0)
        
        def get_source_mask(src):
            ix = jnp.clip(jnp.round((src[0] - xmin) / hx).astype(jnp.int32), 0, nx - 1)
            iy = jnp.clip(jnp.round((src[1] - ymin) / hy).astype(jnp.int32), 0, ny - 1)
            mask = jnp.zeros((nx, ny), dtype=bool)
            return mask.at[ix, iy].set(True)
            
        if source_coords.ndim == 1:
            source_coords = source_coords[None, :]
        masks = jax.vmap(get_source_mask)(source_coords)
        source_mask = jnp.any(masks, axis=0)
        
        T = _fast_sweeping_solve(G, B, source_mask, hx, hy, self.max_iters, self.tol)
        
        return T, X, Y
