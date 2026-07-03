"""Evaluation — provider-agnostic path metrics and an independent geodesic oracle.

Every metric is computed from ``(path, robot, metric/dist)`` and never reaches
inside a concrete provider, so the same numbers apply to a synthetic or a real
arm. ``spray_geodesic`` is the HAM-internal cross-check: it re-derives the geodesic
by integrating the spray ODE (``ExponentialMap``) and shooting to the goal — a
different computation from AVBD's discrete BVP, so agreement is real evidence the
metric's whole autodiff chain and both solvers are correct (validation rungs
L3b/L7b). Use it on the smooth intrinsic-angle metric at moderate distance;
shooting is unstable on stiff or long geodesics, where the eikonal (L3) governs.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ham.geometry.manifolds import FlatTorus
from ham.solvers import ExponentialMap


def to_angle_path(manifold: Any, path: jax.Array) -> jax.Array:
    """Joint-angle path ``(T+1, dof)`` from an ambient path (identity off the torus)."""
    if isinstance(manifold, FlatTorus):
        return jax.vmap(manifold.to_angles)(path)
    return path


# ---------------------------------------------------------------------------
# Path metrics (the per-stage report table)
# ---------------------------------------------------------------------------
def kinetic_cost(metric: Any, path: jax.Array) -> jax.Array:
    """Directed Finsler arc length under the kinetic-energy metric (the primary cost)."""
    return metric.arc_length(path)


def cspace_length(manifold: Any, path: jax.Array) -> jax.Array:
    """Configuration-space path length ``Σ ‖log(x_i, x_{i+1})‖`` (wrap-aware)."""
    seg = jax.vmap(manifold.log_map)(path[:-1], path[1:])
    return jnp.sum(jnp.linalg.norm(seg, axis=-1))


def dirichlet_energy(manifold: Any, path: jax.Array) -> jax.Array:
    """Smoothness ``Σ ‖Δ‖²`` (the sampling-paper Dirichlet energy)."""
    seg = jax.vmap(manifold.log_map)(path[:-1], path[1:])
    return jnp.sum(jnp.sum(seg**2, axis=-1))


def task_length(robot: Any, manifold: Any, path: jax.Array) -> jax.Array:
    """End-effector (workspace) path length."""
    q = to_angle_path(manifold, path)
    ee = jax.vmap(robot.fk)(q)
    return jnp.sum(jnp.linalg.norm(ee[1:] - ee[:-1], axis=-1))


def acceleration_energy(manifold: Any, path: jax.Array) -> jax.Array:
    """Discrete ``∫‖q̈‖²`` from second differences of the angle path (jerk proxy)."""
    q = to_angle_path(manifold, path)
    accel = q[2:] - 2 * q[1:-1] + q[:-2]
    return jnp.sum(jnp.sum(accel**2, axis=-1))


def min_clearance(
    dist: Any, manifold: Any, path: jax.Array, *, samples_per_segment: int = 8
) -> jax.Array:
    """Minimum C-space clearance along the path (negative ⇒ collision).

    Evaluated on a densified path — ``samples_per_segment`` interpolated points
    per segment, not just vertices. Vertex-only clearance is blind to
    *tunneling*: a long segment can straddle a thin obstacle body with both
    endpoints (and even its midpoint) clear, which is also the failure mode of
    AVBD's midpoint-rule energy on coarse discretizations.
    """
    q = to_angle_path(manifold, path)
    if samples_per_segment > 1:
        a, b = q[:-1], q[1:]
        t = jnp.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        pts = a[:, None, :] + t[None, :, None] * (b - a)[:, None, :]
        q = jnp.concatenate([pts.reshape(-1, q.shape[-1]), q[-1:]], axis=0)
    return jnp.min(jax.vmap(dist)(q))


def is_collision_free(dist: Any, manifold: Any, path: jax.Array, margin: float = 0.0) -> bool:
    """Whether the whole path stays clear of obstacles."""
    return bool(min_clearance(dist, manifold, path) > margin)


def path_metrics(
    metric: Any,
    robot: Any,
    path: jax.Array,
    dist: Any | None = None,
) -> dict:
    """Assemble the per-stage report dict from a solved path."""
    manifold = metric.manifold
    report = {
        "kinetic_cost": float(kinetic_cost(metric, path)),
        "cspace_length": float(cspace_length(manifold, path)),
        "task_length": float(task_length(robot, manifold, path)),
        "dirichlet_energy": float(dirichlet_energy(manifold, path)),
        "accel_energy": float(acceleration_energy(manifold, path)),
    }
    if dist is not None:
        report["min_clearance"] = float(min_clearance(dist, manifold, path))
    return report


# ---------------------------------------------------------------------------
# Independent geodesic oracle (spray-ODE shooting) — the HAM-internal cross-check
# ---------------------------------------------------------------------------
def spray_geodesic(
    metric: Any,
    start: jax.Array,
    goal: jax.Array,
    *,
    n_steps: int = 64,
    iters: int = 25,
    reg: float = 1e-7,
    damping: float = 1.0,
) -> jax.Array:
    """Geodesic ``start -> goal`` by shooting the spray ODE (intrinsic-angle metric).

    Root-finds the initial velocity ``v0`` so that ``Exp_start(v0) = goal`` with a
    damped Gauss-Newton iteration, then traces the spray. Independent of AVBD.
    Intended for the smooth, full-rank intrinsic-angle representation.

    Returns:
        The geodesic path, shape ``(n_steps + 1, dof)``.
    """
    em = ExponentialMap(max_steps=n_steps)
    manifold = metric.manifold
    dim = start.shape[0]
    v0 = manifold.log_map(start, goal)

    def resid(v):
        return em.shoot(metric, start, v) - goal

    def step(v, _):
        r = resid(v)
        J = jax.jacfwd(resid)(v)
        dv = jnp.linalg.solve(J.T @ J + reg * jnp.eye(dim, dtype=v.dtype), -J.T @ r)
        return v + damping * dv, None

    v0, _ = jax.lax.scan(step, v0, None, length=iters)
    xs, _ = em.trace(metric, start, v0)
    return xs


def spray_endpoint_error(metric: Any, start: jax.Array, goal: jax.Array, **kw) -> float:
    """Residual ``‖Exp_start(v0) - goal‖`` of the shot geodesic (shooting quality)."""
    xs = spray_geodesic(metric, start, goal, **kw)
    return float(jnp.linalg.norm(xs[-1] - goal))


# ---------------------------------------------------------------------------
# Identifiability — the a-priori diagnostic and per-channel recovery metrics
# ---------------------------------------------------------------------------
def el_residual_score(metric: Any, demos: list[jax.Array], *, normal_only: bool = False) -> jax.Array:
    """Mean squared discrete Euler–Lagrange residual of the demos under ``metric``.

    With ``normal_only`` the residual at each inner node is projected orthogonal
    to the local path tangent, discarding the component that only reparametrizes
    the curve. This is the discrete unparametrized-geodesic (shape) residual —
    exactly the split in Bucataru–Muzsnay: tangential residual = parametrization
    freedom (gauge), normal residual = genuine shape bending.
    """
    from ham.solvers.avbd import _el_residual

    def demo_resid(demo):
        inner = demo[1:-1]
        z = jnp.zeros((inner.shape[0], 0))
        r = _el_residual(inner, metric, demo[0], demo[-1], None, z, z)
        r = r.reshape(inner.shape)
        if normal_only:
            t = demo[2:] - demo[:-2]
            t = t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-12)
            r = r - jnp.sum(r * t, axis=-1, keepdims=True) * t
        return jnp.mean(r**2)

    return jnp.mean(jnp.stack([demo_resid(d) for d in demos]))


def shape_identifiability(base_metric: Any, demos: list[jax.Array], solver: Any) -> float:
    """A-priori identifiability of the drift from path *shape* (holonomy-rank proxy).

    Mean point-set deviation of the demos from the **drift-free base metric's**
    geodesics with the same endpoints, normalized by mean demo length. An exact
    drift is projectively invisible — its demos already trace base geodesics as
    point sets — so the score is ≈ 0 on a pure-gradient regime, predicting
    *before any fit* that no path-shape loss can recover the drift there. A
    rotational drift bends demos off the base geodesics at first order — score
    grows with its strength, predicting recoverability. This is the discrete
    stand-in for Muzsnay's projective-metrizability test ("is the demo spray
    already metrizable by the base?"); it needs only forward solves of the known
    base, no inverse fit. Parametrization is deliberately quotiented out
    (:func:`polyline_deviation`): vertex spacing is the timing channel, not shape.
    """
    devs, lens = [], []
    for d in demos:
        # Cold solve, same init family as the demos. (Warm-starting from the
        # demo does NOT sharpen the test: the demo's constant-F-speed spacing
        # fights the base solver's uniform-spacing optimum and both regimes
        # collapse to a parametrization-relaxation floor.) In high DoF the
        # cold floor grows with geodesic basin multiplicity — this diagnostic
        # is a low-dimensional instrument; the convex asymmetry estimators are
        # what carry to high DoF.
        ref = solver.solve(base_metric, d[0], d[-1], n_steps=d.shape[0] - 1,
                           train_mode=False).xs
        dev = jnp.maximum(polyline_deviation(d, ref), polyline_deviation(ref, d))
        devs.append(float(dev))
        lens.append(float(cspace_length(base_metric.manifold, d)))
    return float(jnp.mean(jnp.array(devs))) / (float(jnp.mean(jnp.array(lens))) + 1e-12)


def polyline_deviation(path_a: jax.Array, path_b: jax.Array) -> jax.Array:
    """Max distance from ``path_a``'s vertices to the polyline ``path_b`` (point-set metric).

    Parametrization-blind by construction: two geodesics sharing a point set but
    parametrized at different speeds score ≈ 0, unlike a vertexwise difference.
    (One-sided discrete Hausdorff; symmetrize by calling twice if needed.)
    """
    a, b = path_b[:-1], path_b[1:]  # segments of path_b
    ab = b - a

    def point_dist(p):
        t = jnp.sum((p - a) * ab, axis=-1) / (jnp.sum(ab * ab, axis=-1) + 1e-12)
        t = jnp.clip(t, 0.0, 1.0)
        proj = a + t[:, None] * ab
        return jnp.min(jnp.linalg.norm(p - proj, axis=-1))

    return jnp.max(jax.vmap(point_dist)(path_a))


def wind_to_form(robot: Any, wind_fn: Any, q: jax.Array) -> jax.Array:
    """First-order Randers 1-form of a joint-space wind: ``b(q) = -M(q) W(q)``."""
    return -(robot.inertia(q) @ wind_fn(q))


def form_cosine(b_pred: jax.Array, b_true: jax.Array) -> float:
    """Global cosine between two sampled 1-form fields, shape ``(N, dof)`` each."""
    num = float(jnp.sum(b_pred * b_true))
    den = float(jnp.linalg.norm(b_pred) * jnp.linalg.norm(b_true))
    return num / (den + 1e-12)


def hodge_split_form(
    b_samples: jax.Array,
    qs: jax.Array,
    centers: jax.Array,
    *,
    width: float = 0.5,
    ridge: float = 1e-4,
):
    """Approximate Helmholtz/Hodge split of a sampled 1-form on the demo region.

    Fits ``b(q) ≈ Σ_j w_j ∇φ_j(q) + Σ_j v_j ⋆dφ_j(q)`` (RBF basis ``φ``) by ridge
    least-squares and returns callables ``(b_exact, b_rot)`` for the two
    components. A region-level LSQ projection, not a boundary-value Hodge solve —
    adequate for channel-recovery cosines, not for topology.
    """

    def phi(q):
        r2 = jnp.sum((q[None, :] - centers) ** 2, axis=-1)
        return jnp.exp(-r2 / (2.0 * width**2))

    def grad_atoms(q):  # (K, dof) rows ∇φ_j
        return jax.jacfwd(phi)(q)

    def rot_atoms(q):  # (K, dof) rows ⋆dφ_j = (-∂₂φ_j, ∂₁φ_j)
        g = grad_atoms(q)
        return jnp.stack([-g[:, 1], g[:, 0]], axis=-1)

    Ag = jax.vmap(grad_atoms)(qs)  # (N, K, dof)
    Ar = jax.vmap(rot_atoms)(qs)
    N, K, dof = Ag.shape
    A = jnp.concatenate([Ag, Ar], axis=1).transpose(0, 2, 1).reshape(N * dof, 2 * K)
    y = b_samples.reshape(-1)
    w = jnp.linalg.solve(A.T @ A + ridge * jnp.eye(2 * K), A.T @ y)
    wg, wr = w[:K], w[K:]

    def b_exact(q):
        return grad_atoms(q).T @ wg

    def b_rot(q):
        return rot_atoms(q).T @ wr

    return b_exact, b_rot
