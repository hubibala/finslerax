"""Mesh-based fully anisotropic Godunov Eikonal solver.

Extends the Eulerian Fast Sweeping Method to unstructured triangulations.
"""


import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.mesh_adjacency import MeshAdjacency
from ham.geometry.metric import AsymmetricMetric
from ham.solvers.eikonal import (
    T_INF,
    compute_one_point_update,
    compute_two_point_update,
    sharp_min,
    steady_state_min,
)


def _triangle_update(
    T_nbr1: jax.Array,
    T_nbr2: jax.Array,
    T0: jax.Array,
    g11: jax.Array,
    g12: jax.Array,
    g22: jax.Array,
    b1: jax.Array,
    b2: jax.Array,
    m1x: jax.Array,
    m1y: jax.Array,
    m2x: jax.Array,
    m2y: jax.Array,
) -> jax.Array:
    """Computes the Godunov update for a single triangle."""
    t0_2pt = compute_two_point_update(
        T_nbr1, T_nbr2, g11, g12, g22, b1, b2, m1x, m1y, m2x, m2y
    )
    t0_1pt_m1 = compute_one_point_update(T_nbr1, g11, g12, g22, b1, b2, m1x, m1y)
    t0_1pt_m2 = compute_one_point_update(T_nbr2, g11, g12, g22, b1, b2, m2x, m2y)

    t0_1pt = sharp_min(t0_1pt_m1, t0_1pt_m2)
    t0_candidate = sharp_min(t0_2pt, t0_1pt)
    return steady_state_min(T0, t0_candidate)


def build_stencil_data(
    vertices: jax.Array,
    adjacency: jax.Array,
    G_faces: jax.Array,
    B_faces: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Precomputes per-(vertex, adjacent-face) stencil data.

    The local 2D basis, donor offsets and projected metric/drift components
    depend only on the mesh and the per-face tensors — they are invariant
    across sweeps, so they are hoisted out of the fixed-point loop (XLA
    cannot move computation out of a ``while_loop`` body).

    This function is plain differentiable JAX: gradients flow through
    ``g_loc``/``b_loc`` back to ``G_faces``/``B_faces``.

    Args:
        vertices: Vertex coordinates, shape ``(V, D)``.
        adjacency: Per-vertex face table ``(V, max_adj, 4)`` with rows
            ``(center, v1, v2, face_idx)``; padding has ``face_idx = -1``.
        G_faces: Per-face primal Randers tensors, shape ``(F, D, D)``.
        B_faces: Per-face primal drift vectors, shape ``(F, D)``.

    Returns:
        Tuple ``(nbr1, nbr2, valid, m_loc, g_loc, b_loc)`` with shapes
        ``(V, max_adj)`` (int donor indices), ``(V, max_adj)`` (bool),
        ``(V, max_adj, 4)`` for ``(m1x, m1y, m2x, m2y)``,
        ``(V, max_adj, 3)`` for ``(g11, g12, g22)`` and
        ``(V, max_adj, 2)`` for ``(b1, b2)``.
    """
    num_vertices = adjacency.shape[0]

    def per_pair(v_idx, face_data):
        v1_idx = face_data[1]
        v2_idx = face_data[2]
        face_idx = face_data[3]

        # Valid face mask
        is_valid = face_idx >= 0

        # Vectors from target (v_idx) to donors (v1, v2)
        target_pt = vertices[v_idx]
        m1 = jnp.where(
            is_valid, vertices[v1_idx] - target_pt, jnp.zeros_like(target_pt)
        )
        m2 = jnp.where(
            is_valid, vertices[v2_idx] - target_pt, jnp.zeros_like(target_pt)
        )

        # Construct local 2D orthonormal basis (u1, u2) for the triangle
        norm1_sq = jnp.sum(m1**2)
        norm1 = jnp.sqrt(jnp.maximum(norm1_sq, 1e-12))
        u1 = m1 / norm1

        proj = jnp.sum(m2 * u1)
        u2 = m2 - proj * u1
        norm2 = jnp.sqrt(jnp.maximum(jnp.sum(u2**2), 1e-12))
        u2 = u2 / norm2

        # 2D coordinates of the edges in the (u1, u2) basis
        m1x = norm1
        m1y = jnp.zeros_like(norm1)
        m2x = proj
        m2y = jnp.sum(m2 * u2)

        # Retrieve the N x N metric and N x 1 drift for this face
        G_face = jnp.where(is_valid, G_faces[face_idx], jnp.eye(G_faces.shape[-1]))
        B_face = jnp.where(is_valid, B_faces[face_idx], jnp.zeros(B_faces.shape[-1]))

        # Project metric G and drift B into the 2D tangent plane
        # g_local = U^T G U
        g11 = jnp.dot(u1, jnp.dot(G_face, u1))
        g12 = jnp.dot(u1, jnp.dot(G_face, u2))
        g22 = jnp.dot(u2, jnp.dot(G_face, u2))

        b1 = jnp.dot(u1, B_face)
        b2 = jnp.dot(u2, B_face)

        return (
            v1_idx,
            v2_idx,
            is_valid,
            jnp.stack([m1x, m1y, m2x, m2y]),
            jnp.stack([g11, g12, g22]),
            jnp.stack([b1, b2]),
        )

    def per_vertex(v_idx, adj_rows):
        return jax.vmap(lambda fd: per_pair(v_idx, fd))(adj_rows)

    return jax.vmap(per_vertex)(jnp.arange(num_vertices), adjacency)


def _vertex_update(
    v_idx: jax.Array,
    T: jax.Array,
    nbr1: jax.Array,
    nbr2: jax.Array,
    valid: jax.Array,
    m_loc: jax.Array,
    g_loc: jax.Array,
    b_loc: jax.Array,
    source_mask: jax.Array,
) -> jax.Array:
    """Updates a single vertex by checking all its adjacent triangles.

    Consumes the precomputed stencil data of :func:`build_stencil_data`;
    only the donor arrival-time gathers happen per sweep.
    """
    T0 = T[v_idx]

    def process_face(T0_curr, xs):
        v1_idx, v2_idx, is_valid, m, g, b = xs

        T1 = jnp.where(is_valid, T[v1_idx], T_INF)
        T2 = jnp.where(is_valid, T[v2_idx], T_INF)

        T0_new = _triangle_update(
            T1, T2, T0_curr, g[0], g[1], g[2], b[0], b[1], m[0], m[1], m[2], m[3]
        )
        T0_new = jnp.where(is_valid, T0_new, T0_curr)
        return T0_new, None

    T0_final, _ = jax.lax.scan(
        process_face,
        T0,
        (nbr1[v_idx], nbr2[v_idx], valid[v_idx], m_loc[v_idx], g_loc[v_idx], b_loc[v_idx]),
    )

    # Enforce boundary condition: source nodes remain 0
    return jnp.where(source_mask[v_idx], 0.0, T0_final)


def sweep_mesh(
    T: jax.Array,
    orderings: jax.Array,
    nbr1: jax.Array,
    nbr2: jax.Array,
    valid: jax.Array,
    m_loc: jax.Array,
    g_loc: jax.Array,
    b_loc: jax.Array,
    source_mask: jax.Array,
) -> jax.Array:
    """Executes all topological sweeps over the mesh."""

    def sweep_pass(T_curr, ordering):
        def update_step(T_state, v_idx):
            # Update the vertex
            T_new_val = _vertex_update(
                v_idx, T_state, nbr1, nbr2, valid, m_loc, g_loc, b_loc, source_mask
            )
            # Write it back into the state array
            T_state = T_state.at[v_idx].set(T_new_val)
            return T_state, None

        T_out, _ = jax.lax.scan(update_step, T_curr, ordering)
        return T_out, None

    T_final, _ = jax.lax.scan(sweep_pass, T, orderings)
    return T_final


@jax.custom_vjp
def _fast_mesh_solve_local(
    g_loc, b_loc, source_mask, orderings, nbr1, nbr2, valid, m_loc, max_iters, tol
):
    """Fixed-point fast sweeping solve on precomputed stencil data.

    Differentiable w.r.t. the projected components ``g_loc``/``b_loc`` via the
    implicit adjoint fixed-point iteration in the custom VJP.
    """

    def cond_fun(val):
        _T, max_diff, iters = val
        return (max_diff > tol) & (iters < max_iters)

    def body_fun(val):
        T, _, iters = val
        T_new = sweep_mesh(
            T, orderings, nbr1, nbr2, valid, m_loc, g_loc, b_loc, source_mask
        )
        diff = jnp.abs(T_new - T)
        max_diff = jnp.max(jnp.where(jnp.isnan(diff), 0.0, diff))
        return T_new, max_diff, iters + 1

    T_init = jnp.where(source_mask, 0.0, T_INF)
    final_val = jax.lax.while_loop(cond_fun, body_fun, (T_init, jnp.array(T_INF), 0))
    return final_val[0]


def _solve_fwd(
    g_loc, b_loc, source_mask, orderings, nbr1, nbr2, valid, m_loc, max_iters, tol
):
    T_final = _fast_mesh_solve_local(
        g_loc, b_loc, source_mask, orderings, nbr1, nbr2, valid, m_loc, max_iters, tol
    )
    return T_final, (
        T_final,
        g_loc,
        b_loc,
        source_mask,
        orderings,
        nbr1,
        nbr2,
        valid,
        m_loc,
        max_iters,
    )


def _solve_bwd(res, g_T):
    (
        T_final,
        g_loc,
        b_loc,
        source_mask,
        orderings,
        nbr1,
        nbr2,
        valid,
        m_loc,
        max_iters,
    ) = res

    def single_sweep(T_in, g_in, b_in):
        return sweep_mesh(
            T_in, orderings, nbr1, nbr2, valid, m_loc, g_in, b_in, source_mask
        )

    _, vjp_fn = jax.vjp(single_sweep, T_final, g_loc, b_loc)

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

    _, dg_loc, db_loc = vjp_fn(x_final)

    dg_loc = jnp.where(jnp.isnan(dg_loc), 0.0, dg_loc)
    db_loc = jnp.where(jnp.isnan(db_loc), 0.0, db_loc)

    return dg_loc, db_loc, None, None, None, None, None, None, None, None


_fast_mesh_solve_local.defvjp(_solve_fwd, _solve_bwd)


def _fast_mesh_solve(
    G_faces, B_faces, source_mask, orderings, adjacency, vertices, max_iters, tol
):
    """Solves the mesh Eikonal PDE from per-face tensors.

    Thin differentiable wrapper: projects the per-face ``G``/``B`` onto the
    per-(vertex, face) stencil data once (outside the fixed-point loop), then
    runs :func:`_fast_mesh_solve_local`. Gradients w.r.t. ``G_faces`` and
    ``B_faces`` flow through the projection automatically.
    """
    nbr1, nbr2, valid, m_loc, g_loc, b_loc = build_stencil_data(
        vertices, adjacency, G_faces, B_faces
    )
    return _fast_mesh_solve_local(
        g_loc, b_loc, source_mask, orderings, nbr1, nbr2, valid, m_loc, max_iters, tol
    )


class MeshEikonalSolver(eqx.Module):
    """
    Mesh-based Eulerian Eikonal Solver for arbitrary triangulations.

    This solver extends the Fast Sweeping Method to unstructured manifolds (e.g.
    triangle meshes). By utilizing a precomputed sequence of topologically-sorted
    vertex sweeps, it propagates arrival times across the mesh in $O(N)$ time per
    sweep, naturally supporting complex topologies like spheres or terrain models.

    The Godunov Hamiltonian evaluates the full N-dimensional ambient metric
    projected strictly onto the 2D local tangent plane of each element.
    It integrates fully with ``jax.custom_vjp``, providing exact gradients of
    the solution $T(x)$ with respect to the continuous metric tensors with
    $O(1)$ memory consumption.
    """

    max_iters: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)

    def __init__(self, max_iters: int = 50, tol: float = 1e-4):
        self.max_iters = max_iters
        self.tol = tol

    def solve(
        self,
        metric: AsymmetricMetric,
        mesh_adj: MeshAdjacency,
        vertices: jax.Array,
        faces: jax.Array,
        source_coords: jax.Array,
    ) -> jax.Array:
        """
        Solves the Eikonal PDE on the provided mesh.

        Evaluates the metric tensor per-face, identifies the nearest vertices to
        the source coordinates to seed the initial conditions, and executes the
        fast sweeping while-loop to reach a steady-state global arrival time field.

        Args:
            metric: The AsymmetricMetric evaluated over the surface.
            mesh_adj: Precomputed topological adjacency containing the sorted sweep orderings.
            vertices: Coordinates of mesh vertices. Shape ``(V, D)`` where D is 2, 3, or N.
            faces: Triangular connectivity array. Shape ``(F, 3)``.
            source_coords: Source points (e.g. ignition sites). Shape ``(S, D)``.

        Returns:
            T: Global arrival times for each vertex. Shape ``(V,)``.

        Reference:
            Implementation utilizes characteristic ray clamping (Option A) for strict
            causality preservation.
        """

        # 1. Map source points to closest vertices
        # We assign source_mask = True for the closest vertex to each source.
        def get_closest_v(src):
            return jnp.argmin(jnp.sum((vertices - src) ** 2, axis=-1))

        closest_vs = jax.vmap(get_closest_v)(source_coords)
        source_mask = jnp.zeros(len(vertices), dtype=bool)
        source_mask = source_mask.at[closest_vs].set(True)

        # 2. Evaluate Metric on Faces
        # The Godunov scheme assumes piecewise constant metric per face
        face_centroids = jnp.mean(vertices[faces], axis=1)

        def extract_GB(pt):
            H, W, lam = metric.zermelo_data(pt)
            B = -jnp.dot(H, W) / lam
            HW = jnp.dot(H, W)
            G = (H + jnp.outer(HW, HW) / lam) / lam
            return G, B

        G_faces, B_faces = jax.vmap(extract_GB)(face_centroids)

        # 3. Solve!
        T = _fast_mesh_solve(
            G_faces,
            B_faces,
            source_mask,
            mesh_adj.sweep_orderings,
            mesh_adj.vertex_adjacency,
            vertices,
            self.max_iters,
            self.tol,
        )
        return T
