"""
Volumetric Cartesian Eikonal Solver for 3D grids.

This module provides the :class:`VolumetricEikonalSolver` which evaluates the
Eikonal PDE over dense three-dimensional Cartesian grids using the Fast Sweeping
Method. It implements the full 3D anisotropic Godunov numerical Hamiltonian and
alternates sweeps across all 6 cardinal directions to guarantee global causality
for arbitrary spatial metrics and wind drifts.
"""


import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.metric import FinslerMetric


def sharp_min(a, b):
    return jnp.where(a <= b, a, b)


def steady_state_min(t_old, t_new):
    return jnp.where(jnp.isnan(t_new), t_old, sharp_min(t_old, t_new))


def compute_three_point_update(
    T1, T2, T3, Q11, Q22, Q33, Q12, Q13, Q23, B1, B2, B3, dx, dy, dz, eps=1e-12
):
    vx, vy, vz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    c1 = T1 * vx + B1
    c2 = T2 * vy + B2
    c3 = T3 * vz + B3

    a = (
        Q11 * vx**2
        + Q22 * vy**2
        + Q33 * vz**2
        + 2 * Q12 * vx * vy
        + 2 * Q13 * vx * vz
        + 2 * Q23 * vy * vz
    )

    b_half = -(
        vx * (Q11 * c1 + Q12 * c2 + Q13 * c3)
        + vy * (Q12 * c1 + Q22 * c2 + Q23 * c3)
        + vz * (Q13 * c1 + Q23 * c2 + Q33 * c3)
    )

    c = (
        c1 * (Q11 * c1 + Q12 * c2 + Q13 * c3)
        + c2 * (Q12 * c1 + Q22 * c2 + Q23 * c3)
        + c3 * (Q13 * c1 + Q23 * c2 + Q33 * c3)
    ) - 1.0

    desc = b_half**2 - a * c
    valid = desc >= 0

    T_cand = (-b_half + jnp.sqrt(jnp.maximum(desc, 0.0))) / jnp.maximum(a, eps)
    causal = valid & (T_cand >= T1) & (T_cand >= T2) & (T_cand >= T3)
    return jnp.where(causal, T_cand, 1e5)


def compute_two_point_update_3d(T1, T2, Q11, Q22, Q12, B1, B2, dx, dy, eps=1e-12):
    vx, vy = 1.0 / dx, 1.0 / dy
    c1 = T1 * vx + B1
    c2 = T2 * vy + B2

    a = Q11 * vx**2 + Q22 * vy**2 + 2 * Q12 * vx * vy
    b_half = -(vx * (Q11 * c1 + Q12 * c2) + vy * (Q12 * c1 + Q22 * c2))
    c = (c1 * (Q11 * c1 + Q12 * c2) + c2 * (Q12 * c1 + Q22 * c2)) - 1.0

    desc = b_half**2 - a * c
    valid = desc >= 0

    T_cand = (-b_half + jnp.sqrt(jnp.maximum(desc, 0.0))) / jnp.maximum(a, eps)
    causal = valid & (T_cand >= T1) & (T_cand >= T2)
    return jnp.where(causal, T_cand, 1e5)


def compute_one_point_update_3d(T1, Q11, B1, dx, eps=1e-12):
    vx = 1.0 / dx
    c1 = T1 * vx + B1

    a = Q11 * vx**2
    b_half = -(vx * Q11 * c1)
    c = (c1 * Q11 * c1) - 1.0

    desc = b_half**2 - a * c
    valid = desc >= 0

    T_cand = (-b_half + jnp.sqrt(jnp.maximum(desc, 0.0))) / jnp.maximum(a, eps)
    causal = valid & (T_cand >= T1)
    return jnp.where(causal, T_cand, 1e5)


def voxel_update(T0, Tx, Ty, Tz, Q11, Q22, Q33, Q12, Q13, Q23, B1, B2, B3, hx, hy, hz):
    t3 = compute_three_point_update(
        Tx, Ty, Tz, Q11, Q22, Q33, Q12, Q13, Q23, B1, B2, B3, hx, hy, hz
    )

    t2_xy = compute_two_point_update_3d(Tx, Ty, Q11, Q22, Q12, B1, B2, hx, hy)
    t2_xz = compute_two_point_update_3d(Tx, Tz, Q11, Q33, Q13, B1, B3, hx, hz)
    t2_yz = compute_two_point_update_3d(Ty, Tz, Q22, Q33, Q23, B2, B3, hy, hz)
    t2 = sharp_min(t2_xy, sharp_min(t2_xz, t2_yz))

    t1_x = compute_one_point_update_3d(Tx, Q11, B1, hx)
    t1_y = compute_one_point_update_3d(Ty, Q22, B2, hy)
    t1_z = compute_one_point_update_3d(Tz, Q33, B3, hz)
    t1 = sharp_min(t1_x, sharp_min(t1_y, t1_z))

    t_cand = sharp_min(t3, sharp_min(t2, t1))
    return steady_state_min(T0, t_cand)


def sweep_axis_z(T, Q, B, source_mask, hx, hy, hz, direction):
    """
    Sweeps along Z axis. T is (nx, ny, nz). Q is (6, nx, ny, nz). B is (3, nx, ny, nz).
    """
    _nx, _ny, _nz = T.shape

    def scan_fn(T_prev, xs):
        T_curr, q11, q22, q33, q12, q13, q23, b1, b2, b3, s_mask = xs

        # We need neighbors in X and Y from T_curr (or T_prev depending on causality)
        # But to avoid complex intra-plane dependencies in JAX, we do Jacobi inside the plane
        # by using the old T_curr for X and Y neighbors, and T_prev for Z.
        # This matches the vectorized parallel sweeps.

        T_x_p = jnp.pad(T_curr[1:, :], ((0, 1), (0, 0)), constant_values=1e5)
        T_x_m = jnp.pad(T_curr[:-1, :], ((1, 0), (0, 0)), constant_values=1e5)
        T_y_p = jnp.pad(T_curr[:, 1:], ((0, 0), (0, 1)), constant_values=1e5)
        T_y_m = jnp.pad(T_curr[:, :-1], ((0, 0), (1, 0)), constant_values=1e5)

        T_x = sharp_min(T_x_p, T_x_m)
        T_y = sharp_min(T_y_p, T_y_m)
        T_z = T_prev

        T_new = voxel_update(
            T_curr, T_x, T_y, T_z, q11, q22, q33, q12, q13, q23, b1, b2, b3, hx, hy, hz
        )
        T_out = jnp.where(s_mask, 0.0, T_new)
        return T_out, T_out

    # Q layout: Q11, Q22, Q33, Q12, Q13, Q23
    if direction == 1:
        xs = (
            T[:, :, 1:],
            Q[0, :, :, 1:],
            Q[1, :, :, 1:],
            Q[2, :, :, 1:],
            Q[3, :, :, 1:],
            Q[4, :, :, 1:],
            Q[5, :, :, 1:],
            B[0, :, :, 1:],
            B[1, :, :, 1:],
            B[2, :, :, 1:],
            source_mask[:, :, 1:],
        )
        _, T_out = jax.lax.scan(scan_fn, T[:, :, 0], xs)
        return jnp.concatenate([T[:, :, 0:1], T_out], axis=2)
    else:
        xs = (
            T[:, :, :-1],
            Q[0, :, :, :-1],
            Q[1, :, :, :-1],
            Q[2, :, :, :-1],
            Q[3, :, :, :-1],
            Q[4, :, :, :-1],
            Q[5, :, :, :-1],
            B[0, :, :, :-1],
            B[1, :, :, :-1],
            B[2, :, :, :-1],
            source_mask[:, :, :-1],
        )
        _, T_out = jax.lax.scan(scan_fn, T[:, :, -1], xs, reverse=True)
        return jnp.concatenate([T_out, T[:, :, -1:]], axis=2)


def sweep_all(T, Q, B, source_mask, hx, hy, hz):
    # Sweep Z
    T = sweep_axis_z(T, Q, B, source_mask, hx, hy, hz, 1)
    T = sweep_axis_z(T, Q, B, source_mask, hx, hy, hz, -1)

    # Sweep Y (Transpose Y and Z)
    T_y = jnp.transpose(T, (0, 2, 1))
    Q_y = jnp.stack([Q[0], Q[2], Q[1], Q[4], Q[3], Q[5]], axis=0).transpose(
        (0, 1, 3, 2)
    )
    B_y = jnp.stack([B[0], B[2], B[1]], axis=0).transpose((0, 1, 3, 2))
    S_y = jnp.transpose(source_mask, (0, 2, 1))

    T_y = sweep_axis_z(T_y, Q_y, B_y, S_y, hx, hz, hy, 1)
    T_y = sweep_axis_z(T_y, Q_y, B_y, S_y, hx, hz, hy, -1)
    T = jnp.transpose(T_y, (0, 2, 1))

    # Sweep X (Transpose X and Z)
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

    T_init = jnp.where(source_mask, 0.0, 1e5)
    final_val = jax.lax.while_loop(cond_fun, body_fun, (T_init, jnp.array(1e5), 0))
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
    and $Z$ axes (6 directional sweeps in total). The underlying numerical scheme
    is the fully anisotropic 3D Godunov stencil, which correctly evaluates the
    full $3 \\times 3$ Zermelo spatial metric tensor $G(x)$ and 3D drift $W(x)$.

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
            - **Q**: Extracted $3 \\times 3$ inverse metric tensor field (flattened symmetric).
            - **B**: Extracted 3D drift vector field.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = grid_extent
        nx, ny, nz = grid_shape
        hx = (xmax - xmin) / (nx - 1)
        hy = (ymax - ymin) / (ny - 1)
        hz = (zmax - zmin) / (nz - 1)

        x_coords = jnp.linspace(xmin, xmax, nx)
        y_coords = jnp.linspace(ymin, ymax, ny)
        z_coords = jnp.linspace(zmin, zmax, nz)
        X, Y, Z = jnp.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        pts = jnp.stack([X, Y, Z], axis=-1)

        def get_zermelo(pt):
            return metric.zermelo_data(pt)

        H, W, lam = jax.vmap(jax.vmap(jax.vmap(get_zermelo)))(pts)

        # B = - H * W / lam
        # H is (..., 3, 3), W is (..., 3)
        B_vec = -jnp.einsum("...ij,...j->...i", H, W) / lam[..., None]

        HW = jnp.einsum("...ij,...j->...i", H, W)
        HW_outer = jnp.einsum("...i,...j->...ij", HW, HW)
        Q_mat = (H + HW_outer / lam[..., None, None]) / lam[..., None, None]

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
