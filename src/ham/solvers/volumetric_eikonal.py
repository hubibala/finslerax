"""
Volumetric Cartesian Eikonal Solver for 3D grids.

This module provides the :class:`VolumetricEikonalSolver` which evaluates the
Eikonal PDE over dense three-dimensional Cartesian grids using the Fast Sweeping
Method. It implements the full 3D anisotropic Godunov numerical Hamiltonian for
the dual Randers (Zermelo) arrival-time PDE

    (grad T - B)^T Q (grad T - B) = 1,    T(source) = 0,

where ``Q = lam * (H^{-1} - W W^T)`` is the *dual* (inverse) metric tensor and
``B = -H W / lam`` the dual drift, both derived from the Zermelo navigation
data ``(H, W, lam)``. The stencil enumerates all signed upwind donor
configurations per axis, which is required for correctness whenever the drift
``B`` or the off-diagonal couplings ``Q_ij`` are non-zero.

The solver alternates Gauss-Seidel sweeps across all 6 cardinal directions to
guarantee global causality for arbitrary spatial metrics and wind drifts.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.metric import FinslerMetric
from ham.solvers.eikonal import T_INF, safe_sqrt, sharp_min, steady_state_min

# =============================================================================
# SIGNED GODUNOV STENCIL UPDATES
# =============================================================================
#
# Convention: each donor sits at a signed displacement ``s_i = sigma_i * h_i``
# along axis i from the update target, and ``v_i = 1 / s_i = sigma_i / h_i``.
# The upwind gradient approximation is ``xi_i ~ (T_i - t) / s_i``, so with
# ``c_i = T_i * v_i - B_i`` the PDE residual becomes the quadratic
#
#     sum_ij Q_ij (t v_i - c_i)(t v_j - c_j) = 1,
#
# whose *larger* root is the causal candidate. Tracking the sign of ``v_i``
# (instead of folding +/- neighbors with a min) is what keeps the drift terms
# and the cross-couplings Q_ij consistent.


def compute_three_point_update(
    T1, T2, T3, Q11, Q22, Q33, Q12, Q13, Q23, B1, B2, B3, v1, v2, v3, eps=1e-12
):
    """3-point Godunov update; ``v_i`` are the signed inverse offsets."""
    c1 = T1 * v1 - B1
    c2 = T2 * v2 - B2
    c3 = T3 * v3 - B3

    a = (
        Q11 * v1**2
        + Q22 * v2**2
        + Q33 * v3**2
        + 2 * (Q12 * v1 * v2 + Q13 * v1 * v3 + Q23 * v2 * v3)
    )

    Qc1 = Q11 * c1 + Q12 * c2 + Q13 * c3
    Qc2 = Q12 * c1 + Q22 * c2 + Q23 * c3
    Qc3 = Q13 * c1 + Q23 * c2 + Q33 * c3

    b_half = -(v1 * Qc1 + v2 * Qc2 + v3 * Qc3)
    cc = c1 * Qc1 + c2 * Qc2 + c3 * Qc3 - 1.0

    desc = b_half**2 - a * cc
    T_cand = (-b_half + safe_sqrt(desc)) / jnp.maximum(a, eps)

    # Upwind (simplex) condition: every donor must carry nonnegative weight
    # d t / d T_k = v_k [Q (t v - c)]_k / (v^T Q (t v - c)) >= 0.
    r1 = Q11 * (T_cand * v1 - c1) + Q12 * (T_cand * v2 - c2) + Q13 * (T_cand * v3 - c3)
    r2 = Q12 * (T_cand * v1 - c1) + Q22 * (T_cand * v2 - c2) + Q23 * (T_cand * v3 - c3)
    r3 = Q13 * (T_cand * v1 - c1) + Q23 * (T_cand * v2 - c2) + Q33 * (T_cand * v3 - c3)
    upwind = (v1 * r1 >= 0) & (v2 * r2 >= 0) & (v3 * r3 >= 0)

    causal = (
        (desc >= 0)
        & (T_cand >= T1)
        & (T_cand >= T2)
        & (T_cand >= T3)
        & upwind
    )
    return jnp.where(causal, T_cand, T_INF)


def compute_two_point_update_3d(T1, T2, Q11, Q22, Q12, B1, B2, v1, v2, eps=1e-12):
    """2-point Godunov update on the (i, j) coordinate plane; signed ``v``."""
    c1 = T1 * v1 - B1
    c2 = T2 * v2 - B2

    a = Q11 * v1**2 + Q22 * v2**2 + 2 * Q12 * v1 * v2

    Qc1 = Q11 * c1 + Q12 * c2
    Qc2 = Q12 * c1 + Q22 * c2

    b_half = -(v1 * Qc1 + v2 * Qc2)
    cc = c1 * Qc1 + c2 * Qc2 - 1.0

    desc = b_half**2 - a * cc
    T_cand = (-b_half + safe_sqrt(desc)) / jnp.maximum(a, eps)

    r1 = Q11 * (T_cand * v1 - c1) + Q12 * (T_cand * v2 - c2)
    r2 = Q12 * (T_cand * v1 - c1) + Q22 * (T_cand * v2 - c2)
    upwind = (v1 * r1 >= 0) & (v2 * r2 >= 0)

    causal = (desc >= 0) & (T_cand >= T1) & (T_cand >= T2) & upwind
    return jnp.where(causal, T_cand, T_INF)


def compute_one_point_update_3d(T1, Q11, B1, v1, eps=1e-12):
    """1-point (axis-aligned) update; signed ``v1``."""
    c1 = T1 * v1 - B1

    a = Q11 * v1**2
    b_half = -(v1 * Q11 * c1)
    cc = c1 * Q11 * c1 - 1.0

    desc = b_half**2 - a * cc
    T_cand = (-b_half + safe_sqrt(desc)) / jnp.maximum(a, eps)

    causal = (desc >= 0) & (T_cand >= T1)
    return jnp.where(causal, T_cand, T_INF)


def voxel_update(
    T0,
    Tx_m,
    Tx_p,
    Ty_m,
    Ty_p,
    Tz,
    Q11,
    Q22,
    Q33,
    Q12,
    Q13,
    Q23,
    B1,
    B2,
    B3,
    hx,
    hy,
    hz,
    sz,
):
    """Godunov update of one z-plane.

    ``Tx_m/Tx_p`` (``Ty_m/Ty_p``) are the in-plane donor values on the -/+
    side along x (y); ``Tz`` is the donor plane along the sweep axis whose
    side is fixed by ``sz`` (+1 or -1). All signed donor configurations are
    enumerated and the minimum causal candidate is taken.
    """
    vz = sz / hz

    t_cand = compute_one_point_update_3d(Tz, Q33, B3, vz)

    for sx, Tx in ((-1.0, Tx_m), (1.0, Tx_p)):
        vx = sx / hx
        t_cand = sharp_min(t_cand, compute_one_point_update_3d(Tx, Q11, B1, vx))
        t_cand = sharp_min(
            t_cand,
            compute_two_point_update_3d(Tx, Tz, Q11, Q33, Q13, B1, B3, vx, vz),
        )
        for sy, Ty in ((-1.0, Ty_m), (1.0, Ty_p)):
            vy = sy / hy
            t_cand = sharp_min(
                t_cand,
                compute_two_point_update_3d(Tx, Ty, Q11, Q22, Q12, B1, B2, vx, vy),
            )
            t_cand = sharp_min(
                t_cand,
                compute_three_point_update(
                    Tx, Ty, Tz, Q11, Q22, Q33, Q12, Q13, Q23, B1, B2, B3, vx, vy, vz
                ),
            )

    for sy, Ty in ((-1.0, Ty_m), (1.0, Ty_p)):
        vy = sy / hy
        t_cand = sharp_min(t_cand, compute_one_point_update_3d(Ty, Q22, B2, vy))
        t_cand = sharp_min(
            t_cand,
            compute_two_point_update_3d(Ty, Tz, Q22, Q33, Q23, B2, B3, vy, vz),
        )

    return steady_state_min(T0, t_cand)


# =============================================================================
# SWEEPING PASSES
# =============================================================================


def sweep_axis_z(T, Q, B, source_mask, hx, hy, hz, direction):
    """
    Sweeps along the z axis (axis 2). T is (nx, ny, nz). Q is (6, nx, ny, nz).
    B is (3, nx, ny, nz). Q layout: (Q11, Q22, Q33, Q12, Q13, Q23).
    """
    # lax.scan unstacks along the leading axis, so move z to the front.
    Tm = jnp.moveaxis(T, 2, 0)  # (nz, nx, ny)
    Qm = jnp.moveaxis(Q, 3, 1)  # (6, nz, nx, ny)
    Bm = jnp.moveaxis(B, 3, 1)  # (3, nz, nx, ny)
    Sm = jnp.moveaxis(source_mask, 2, 0)

    # The donor plane sits -direction steps along z.
    sz = -float(direction)

    def scan_fn(T_prev, xs):
        T_curr, q11, q22, q33, q12, q13, q23, b1, b2, b3, s_mask = xs

        # In-plane donors use the not-yet-updated plane values (Jacobi within
        # the plane); the scanned-axis donor is the freshly updated previous
        # plane (Gauss-Seidel across planes).
        T_x_m = jnp.pad(T_curr[:-1, :], ((1, 0), (0, 0)), constant_values=T_INF)
        T_x_p = jnp.pad(T_curr[1:, :], ((0, 1), (0, 0)), constant_values=T_INF)
        T_y_m = jnp.pad(T_curr[:, :-1], ((0, 0), (1, 0)), constant_values=T_INF)
        T_y_p = jnp.pad(T_curr[:, 1:], ((0, 0), (0, 1)), constant_values=T_INF)

        T_new = voxel_update(
            T_curr,
            T_x_m,
            T_x_p,
            T_y_m,
            T_y_p,
            T_prev,
            q11,
            q22,
            q33,
            q12,
            q13,
            q23,
            b1,
            b2,
            b3,
            hx,
            hy,
            hz,
            sz,
        )
        T_out = jnp.where(s_mask, 0.0, T_new)
        return T_out, T_out

    if direction == 1:
        xs = (
            Tm[1:],
            Qm[0, 1:],
            Qm[1, 1:],
            Qm[2, 1:],
            Qm[3, 1:],
            Qm[4, 1:],
            Qm[5, 1:],
            Bm[0, 1:],
            Bm[1, 1:],
            Bm[2, 1:],
            Sm[1:],
        )
        _, T_out = jax.lax.scan(scan_fn, Tm[0], xs)
        Tm_new = jnp.concatenate([Tm[0:1], T_out], axis=0)
    else:
        xs = (
            Tm[:-1],
            Qm[0, :-1],
            Qm[1, :-1],
            Qm[2, :-1],
            Qm[3, :-1],
            Qm[4, :-1],
            Qm[5, :-1],
            Bm[0, :-1],
            Bm[1, :-1],
            Bm[2, :-1],
            Sm[:-1],
        )
        _, T_out = jax.lax.scan(scan_fn, Tm[-1], xs, reverse=True)
        Tm_new = jnp.concatenate([T_out, Tm[-1:]], axis=0)

    return jnp.moveaxis(Tm_new, 0, 2)


def sweep_all(T, Q, B, source_mask, hx, hy, hz):
    # Sweep Z
    T = sweep_axis_z(T, Q, B, source_mask, hx, hy, hz, 1)
    T = sweep_axis_z(T, Q, B, source_mask, hx, hy, hz, -1)

    # Sweep Y (swap the roles of Y and Z)
    T_y = jnp.transpose(T, (0, 2, 1))
    Q_y = jnp.stack([Q[0], Q[2], Q[1], Q[4], Q[3], Q[5]], axis=0).transpose(
        (0, 1, 3, 2)
    )
    B_y = jnp.stack([B[0], B[2], B[1]], axis=0).transpose((0, 1, 3, 2))
    S_y = jnp.transpose(source_mask, (0, 2, 1))

    T_y = sweep_axis_z(T_y, Q_y, B_y, S_y, hx, hz, hy, 1)
    T_y = sweep_axis_z(T_y, Q_y, B_y, S_y, hx, hz, hy, -1)
    T = jnp.transpose(T_y, (0, 2, 1))

    # Sweep X (swap the roles of X and Z)
    T_x = jnp.transpose(T, (2, 1, 0))
    Q_x = jnp.stack([Q[2], Q[1], Q[0], Q[5], Q[4], Q[3]], axis=0).transpose(
        (0, 3, 2, 1)
    )
    B_x = jnp.stack([B[2], B[1], B[0]], axis=0).transpose((0, 3, 2, 1))
    S_x = jnp.transpose(source_mask, (2, 1, 0))

    T_x = sweep_axis_z(T_x, Q_x, B_x, S_x, hz, hy, hx, 1)
    T_x = sweep_axis_z(T_x, Q_x, B_x, S_x, hz, hy, hx, -1)
    T = jnp.transpose(T_x, (2, 1, 0))

    return T


# =============================================================================
# SOLVER & IMPLICIT GRADIENTS
# =============================================================================


@jax.custom_vjp
def _volumetric_solve(Q, B, source_mask, hx, hy, hz, max_iters, tol):
    def cond_fun(val):
        _T, max_diff, iters = val
        return (max_diff > tol) & (iters < max_iters)

    def body_fun(val):
        T, _, iters = val
        T_new = sweep_all(T, Q, B, source_mask, hx, hy, hz)
        diff = jnp.abs(T_new - T)
        max_diff = jnp.max(jnp.where(jnp.isnan(diff), 0.0, diff))
        return T_new, max_diff, iters + 1

    T_init = jnp.where(source_mask, 0.0, T_INF)
    final_val = jax.lax.while_loop(cond_fun, body_fun, (T_init, jnp.array(T_INF), 0))
    return final_val[0]


def _solve_fwd(Q, B, source_mask, hx, hy, hz, max_iters, tol):
    T_final = _volumetric_solve(Q, B, source_mask, hx, hy, hz, max_iters, tol)
    return T_final, (T_final, Q, B, source_mask, hx, hy, hz, max_iters)


def _solve_bwd(res, g_T):
    T_final, Q, B, source_mask, hx, hy, hz, max_iters = res

    def single_sweep(T_in, Q_in, B_in):
        return sweep_all(T_in, Q_in, B_in, source_mask, hx, hy, hz)

    _, vjp_fn = jax.vjp(single_sweep, T_final, Q, B)

    def cond_fun(val):
        x_curr, x_prev, iters = val
        return (jnp.max(jnp.abs(x_curr - x_prev)) > 1e-6) & (iters < max_iters)

    def body_fun(val):
        x_curr, _, iters = val
        dT, _, _ = vjp_fn(x_curr)
        x_next = g_T + dT
        return x_next, x_curr, iters + 1

    x_init = jnp.zeros_like(g_T)
    x_prev = jnp.full_like(g_T, 1e9)
    x_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, (x_init, x_prev, 0))

    _, dQ, dB = vjp_fn(x_final)

    dQ = jnp.where(jnp.isnan(dQ), 0.0, dQ)
    dB = jnp.where(jnp.isnan(dB), 0.0, dB)

    return dQ, dB, None, None, None, None, None, None


_volumetric_solve.defvjp(_solve_fwd, _solve_bwd)


class VolumetricEikonalSolver(eqx.Module):
    """
    Volumetric Eulerian Eikonal Solver for 3D Cartesian grids.

    This solver extends the classical Fast Sweeping Method to 3D volumes. It
    propagates wavefronts by alternating Gauss-Seidel sweeps across the $X, Y$,
    and $Z$ axes (6 directional sweeps in total). The underlying numerical
    scheme is the fully anisotropic 3D Godunov stencil with signed upwind
    donor enumeration, evaluating the dual tensor
    $Q(x) = \\lambda (H^{-1} - W W^T)$ and dual drift $B(x) = -H W / \\lambda$
    derived from the Zermelo navigation data.

    The backward pass is implemented implicitly via `jax.custom_vjp` to
    provide $O(1)$ memory gradients with respect to the continuous metric.
    """

    max_iters: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)

    def __init__(self, max_iters: int = 50, tol: float = 1e-4):
        """
        Initializes the volumetric solver.

        Args:
            max_iters: Maximum sweeping iterations over the whole grid.
            tol: Tolerance for steady-state convergence.
        """
        self.max_iters = max_iters
        self.tol = tol

    def solve(
        self,
        metric: FinslerMetric,
        source_coords: jax.Array,
        grid_extent: tuple[float, float, float, float, float, float],
        grid_shape: tuple[int, int, int],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Solves the volumetric Eikonal PDE across the defined 3D Cartesian bounding box.

        Args:
            metric: The FinslerMetric to evaluate over the volume. Must implement `zermelo_data`.
            source_coords: Source ignition points. Shape ``(S, 3)``.
            grid_extent: Bounding box limits ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
            grid_shape: Voxel resolutions ``(nx, ny, nz)``.

        Returns:
            A tuple ``(T, Q, B)``:
            - **T**: Steady-state arrival time volume. Shape ``(nx, ny, nz)``.
            - **Q**: Dual (inverse) metric tensor field ``lam * (H^-1 - W W^T)``,
              flattened symmetric layout ``(6, nx, ny, nz)``.
            - **B**: Dual drift vector field ``-H W / lam``. Shape ``(3, nx, ny, nz)``.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = grid_extent
        nx, ny, nz = grid_shape
        hx = (xmax - xmin) / max(1, nx - 1)
        hy = (ymax - ymin) / max(1, ny - 1)
        hz = (zmax - zmin) / max(1, nz - 1)

        x_coords = jnp.linspace(xmin, xmax, nx)
        y_coords = jnp.linspace(ymin, ymax, ny)
        z_coords = jnp.linspace(zmin, zmax, nz)
        X, Y, Z = jnp.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        pts = jnp.stack([X, Y, Z], axis=-1)

        def get_zermelo(pt):
            return metric.zermelo_data(pt)

        H, W, lam = jax.vmap(jax.vmap(jax.vmap(get_zermelo)))(pts)

        # Dual drift: B = -H W / lam
        B_vec = -jnp.einsum("...ij,...j->...i", H, W) / lam[..., None]

        # Dual (inverse) metric: Q = lam * (H^-1 - W W^T)
        H_inv = jnp.linalg.inv(H)
        WW = jnp.einsum("...i,...j->...ij", W, W)
        Q_mat = lam[..., None, None] * (H_inv - WW)

        B_comp = jnp.stack([B_vec[..., 0], B_vec[..., 1], B_vec[..., 2]], axis=0)
        Q_comp = jnp.stack(
            [
                Q_mat[..., 0, 0],
                Q_mat[..., 1, 1],
                Q_mat[..., 2, 2],
                Q_mat[..., 0, 1],
                Q_mat[..., 0, 2],
                Q_mat[..., 1, 2],
            ],
            axis=0,
        )

        # Determine source mask
        def dist_to_src(src):
            return jnp.argmin(jnp.sum((pts - src) ** 2, axis=-1).flatten())

        closest_vs = jax.vmap(dist_to_src)(source_coords)
        source_mask = jnp.zeros(nx * ny * nz, dtype=bool)
        source_mask = source_mask.at[closest_vs].set(True)
        source_mask = source_mask.reshape((nx, ny, nz))

        T = _volumetric_solve(
            Q_comp, B_comp, source_mask, hx, hy, hz, self.max_iters, self.tol
        )
        return T, Q_comp, B_comp
