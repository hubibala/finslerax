"""Inverse learning — recover a C-space field from demonstrations through the solver.

A ``DemoLoss`` is a callable ``(field, demos) -> scalar`` built around a
``make_metric(field)`` factory, so the same training driver fits any learnable
field under any loss formulation. The menu is organized by **observation
channel**, because that is what decides identifiability
(``spec/exact_drift_gauge_equivalence.md``):

*Path-shape channel* — blind to any exact (gradient) drift, by the projective-
invariance / reward-shaping gauge theorem:

* **L-B — Euler–Lagrange residual.** Demos should be discrete geodesics of the
  fitted metric. Reuses AVBD's ``_el_residual``; no solver in the loop.
* **L-A — path matching.** Solve the geodesic with the current field and match
  the demo; exercises the implicit-diff adjoint (gated by L2).
* **L-C — max-margin**, **L-D — density.** Landscape-shaping fallbacks.

*Timing/cost channel* — carries exactly the component the shape channel misses
(Bucataru–Muzsnay parametrization-rigidity):

* **L-T — segment-cost matching** (:func:`cost_matching_loss`). Matches per-
  segment traversal costs of timed demos; no inner solve.
* **Round-trip asymmetry LSQ** (:func:`recover_potential_lsq`). The closed-form
  estimator ``cost(γ) − cost(γ⁻¹) = 2∫β = 2(U(B) − U(A))`` inverted as a convex
  least-squares for the potential — seedless, global.

Every path-shape arm carries an anti-collapse regularizer (``free_field_reg``):
without it, a field can satisfy stationarity trivially.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ham.solvers.avbd import _el_residual

MakeMetric = Callable[[Any], Any]


# ---------------------------------------------------------------------------
# Anti-collapse regularizer
# ---------------------------------------------------------------------------
def free_field_reg(field: Any, sample_q: jax.Array, margin: float = 0.3) -> jax.Array:
    """Penalize predicting collision (small clearance) where data does not force it.

    Pulls the learned field toward "free" (clearance ≥ ``margin``) on the sampled
    configurations, so a barrier only emerges where the demonstrations require it.
    """
    clr = jax.vmap(field)(sample_q)
    return jnp.mean(jax.nn.relu(margin - clr) ** 2)


# ---------------------------------------------------------------------------
# Loss arms
# ---------------------------------------------------------------------------
def el_residual_loss(
    field: Any,
    demos: list[jax.Array],
    make_metric: MakeMetric,
    *,
    sample_q: jax.Array | None = None,
    reg_weight: float = 0.1,
    normal_only: bool = False,
) -> jax.Array:
    """L-B: mean squared discrete Euler–Lagrange residual of the demos (no inner solve).

    With ``normal_only`` the residual is projected orthogonal to the path
    tangent, discarding the parametrization (vertex-spacing) component. This
    models *unparametrized* demonstrations — pure path shape. Demo vertex
    spacing is timing information in disguise (solver demos are constant-F-speed
    sampled), so leaving it in quietly mixes the two observation channels.
    """
    metric = make_metric(field)

    def demo_resid(demo):
        inner = demo[1:-1]
        z = jnp.zeros((inner.shape[0], 0))
        r = _el_residual(inner, metric, demo[0], demo[-1], None, z, z)
        if normal_only:
            r = r.reshape(inner.shape)
            t = demo[2:] - demo[:-2]
            t = t / (jnp.linalg.norm(t, axis=-1, keepdims=True) + 1e-12)
            r = r - jnp.sum(r * t, axis=-1, keepdims=True) * t
        return jnp.mean(r**2)

    data = jnp.mean(jnp.stack([demo_resid(d) for d in demos]))
    reg = free_field_reg(field, sample_q) if sample_q is not None else 0.0
    return data + reg_weight * reg


def path_matching_loss(
    field: Any,
    demos: list[jax.Array],
    make_metric: MakeMetric,
    *,
    solver: Any,
    sample_q: jax.Array | None = None,
    reg_weight: float = 0.1,
) -> jax.Array:
    """L-A: mean squared distance between the solved geodesic and the demo (uses the adjoint)."""
    metric = make_metric(field)

    def demo_loss(demo):
        traj = solver.solve(metric, demo[0], demo[-1], n_steps=demo.shape[0] - 1)
        return jnp.mean((traj.xs - demo) ** 2)

    data = jnp.mean(jnp.stack([demo_loss(d) for d in demos]))
    reg = free_field_reg(field, sample_q) if sample_q is not None else 0.0
    return data + reg_weight * reg


def max_margin_loss(
    field: Any,
    demos: list[jax.Array],
    make_metric: MakeMetric,
    *,
    margin: float = 0.1,
    noise: float = 0.1,
    key: jax.Array | None = None,
    sample_q: jax.Array | None = None,
    reg_weight: float = 0.1,
) -> jax.Array:
    """L-C: hinge loss — the demo's path energy should beat a perturbed alternative."""
    metric = make_metric(field)
    key = jax.random.PRNGKey(0) if key is None else key

    def path_energy(path):
        vs = jax.vmap(metric.manifold.log_map)(path[:-1], path[1:])
        return jnp.sum(jax.vmap(metric.energy)(path[:-1], vs))

    def demo_loss(demo, k):
        alt = demo.at[1:-1].add(noise * jax.random.normal(k, demo[1:-1].shape))
        return jax.nn.relu(margin + path_energy(demo) - path_energy(alt))

    keys = jax.random.split(key, len(demos))
    data = jnp.mean(jnp.stack([demo_loss(d, k) for d, k in zip(demos, keys)]))
    reg = free_field_reg(field, sample_q) if sample_q is not None else 0.0
    return data + reg_weight * reg


def density_loss(
    field: Any,
    demos: list[jax.Array],
    make_metric: MakeMetric,
    *,
    sample_q: jax.Array,
    margin: float = 0.3,
    reg_weight: float = 1.0,
) -> jax.Array:
    """L-D: visited states should be clear (clearance ≥ margin), off-data may be barriered."""
    visited = jnp.concatenate(list(demos), axis=0)
    clr_visited = jax.vmap(field)(visited)
    data = jnp.mean(jax.nn.relu(margin - clr_visited) ** 2)
    # Encourage some barrier somewhere off-data (else the trivial free field wins).
    clr_sample = jax.vmap(field)(sample_q)
    spread = jnp.mean(jax.nn.relu(clr_sample - 2.0) ** 2)
    return data + reg_weight * spread


# ---------------------------------------------------------------------------
# Timing/cost channel — the complementary observations that break the gauge
# ---------------------------------------------------------------------------
def segment_costs(metric: Any, path: jax.Array) -> jax.Array:
    """Per-segment Finsler cost ``F(mid_i, Δ_i)`` (midpoint rule, matches ``arc_length``)."""
    mids = jax.vmap(metric.manifold.project)(0.5 * (path[:-1] + path[1:]))
    return jax.vmap(metric.metric_fn)(mids, path[1:] - path[:-1])


def segment_asymmetry(metric: Any, path: jax.Array) -> jax.Array:
    """Directed round-trip asymmetry per segment: ``(F(mid, Δ) − F(mid, −Δ))/2 = ∫β``.

    This is the note's clean estimator on the finest endpoint graph: for an exact
    drift ``β = dU`` it equals ``U(x_{i+1}) − U(x_i)`` exactly (same point set,
    reversed) — the observation model is "run the demo backwards and time it".
    """
    mids = jax.vmap(metric.manifold.project)(0.5 * (path[:-1] + path[1:]))
    d = path[1:] - path[:-1]
    fwd = jax.vmap(metric.metric_fn)(mids, d)
    rev = jax.vmap(metric.metric_fn)(mids, -d)
    return 0.5 * (fwd - rev)


def cost_matching_loss(
    field: Any,
    demos: list[jax.Array],
    make_metric: MakeMetric,
    *,
    demo_costs: list[jax.Array],
    reg_weight: float = 0.0,
    sample_q: jax.Array | None = None,
) -> jax.Array:
    """L-T: match observed per-segment traversal costs (the timing channel; no inner solve).

    By parametrization-rigidity this channel *does* identify the exact drift
    component that L-A/L-B provably cannot see.
    """
    metric = make_metric(field)

    def demo_loss(demo, costs):
        return jnp.mean((segment_costs(metric, demo) - costs) ** 2)

    data = jnp.mean(jnp.stack([demo_loss(d, c) for d, c in zip(demos, demo_costs)]))
    reg = free_field_reg(field, sample_q) if sample_q is not None else 0.0
    return data + reg_weight * reg


def rbf_features(q: jax.Array, centers: jax.Array, width: float) -> jax.Array:
    """Gaussian RBF feature vector ``φ(q)``, shape ``(K,)``."""
    r2 = jnp.sum((q[None, :] - centers) ** 2, axis=-1)
    return jnp.exp(-r2 / (2.0 * width**2))


def recover_potential_lsq(
    starts: jax.Array,
    ends: jax.Array,
    deltas: jax.Array,
    centers: jax.Array,
    *,
    width: float = 0.5,
    ridge: float = 1e-4,
):
    """Recover the drift potential from directed cost asymmetries — convex, seedless.

    Solves ``min_w Σ_k ((φ(end_k) − φ(start_k))ᵀw − ΔU_k)² + ridge‖w‖²`` where
    ``ΔU_k`` is the observed round-trip asymmetry ``(cost_fwd − cost_rev)/2``
    over the segment/endpoint graph. For an exact drift this is a linear system
    in the node potentials (unique up to the additive-constant gauge). Returns
    ``(U_hat, w)`` with ``U_hat: q -> φ(q)ᵀw``.
    """
    Phi = jax.vmap(lambda a, b: rbf_features(b, centers, width) - rbf_features(a, centers, width))(
        starts, ends
    )
    A = Phi.T @ Phi + ridge * jnp.eye(Phi.shape[1])
    w = jnp.linalg.solve(A, Phi.T @ deltas)

    def U_hat(q: jax.Array) -> jax.Array:
        return rbf_features(q, centers, width) @ w

    return U_hat, w


def recover_form_lsq(
    starts: jax.Array,
    ends: jax.Array,
    deltas: jax.Array,
    centers: jax.Array,
    *,
    width: float = 0.5,
    ridge: float = 1e-4,
):
    """Recover the *full* drift 1-form ``β = dU + ⋆dψ`` from directed cost asymmetries.

    The round-trip asymmetry observes ``∫β`` along each segment — the whole form,
    not just its exact part — so with demos crossing the region in varied
    directions a single ridge LSQ identifies both Hodge channels at once:
    ``b(mid_k)·Δ_k ≈ ΔU_k`` with atoms ``∇φ_j`` (exact) and ``⋆dφ_j`` (co-exact).
    Returns ``(U_hat, psi_hat, b_hat)``; the recovered channels can be compared
    to ground truth separately. Convex and seedless, like
    :func:`recover_potential_lsq` (which is the exact-channel-only special case).
    """
    mids = 0.5 * (starts + ends)
    disp = ends - starts

    def phi(q):
        r2 = jnp.sum((q[None, :] - centers) ** 2, axis=-1)
        return jnp.exp(-r2 / (2.0 * width**2))

    def row(mid, d):
        G = jax.jacfwd(phi)(mid)  # (K, dof) rows ∇φ_j
        R = jnp.stack([-G[:, 1], G[:, 0]], axis=-1)  # ⋆dφ_j
        return jnp.concatenate([G @ d, R @ d])

    A = jax.vmap(row)(mids, disp)
    K = centers.shape[0]
    w = jnp.linalg.solve(A.T @ A + ridge * jnp.eye(2 * K), A.T @ deltas)
    wg, wr = w[:K], w[K:]

    def U_hat(q):
        return phi(q) @ wg

    def psi_hat(q):
        return phi(q) @ wr

    def b_hat(q):
        G = jax.jacfwd(phi)(q)
        R = jnp.stack([-G[:, 1], G[:, 0]], axis=-1)
        return G.T @ wg + R.T @ wr

    return U_hat, psi_hat, b_hat


LOSS_ARMS = {
    "el": el_residual_loss,       # L-B (path shape)
    "path": path_matching_loss,   # L-A (path shape, adjoint showcase)
    "margin": max_margin_loss,    # L-C
    "density": density_loss,      # L-D
    "cost": cost_matching_loss,   # L-T (timing channel)
}


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------
def fit_field(
    field0: Any,
    demos: list[jax.Array],
    make_metric: MakeMetric,
    *,
    loss_fn: Callable,
    steps: int = 400,
    lr: float = 3e-3,
    grad_clip: float = 1.0,
    **loss_kwargs,
) -> tuple[Any, list[float]]:
    """Fit a learnable ``field`` to ``demos`` by minimizing ``loss_fn`` with Adam.

    Gradients are global-norm clipped (``grad_clip``) so a transiently bad field
    (e.g. a near-causal wind making the inner solve stiff) cannot blow up training.
    Returns the fitted field and the loss history.
    """
    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(lr))
    state = opt.init(eqx.filter(field0, eqx.is_array))
    field = field0

    @eqx.filter_jit
    def step(field, state):
        loss, grads = eqx.filter_value_and_grad(
            lambda f: loss_fn(f, demos, make_metric, **loss_kwargs)
        )(field)
        updates, state = opt.update(grads, state, eqx.filter(field, eqx.is_array))
        return eqx.apply_updates(field, updates), state, loss

    history = []
    for _ in range(steps):
        field, state, loss = step(field, state)
        history.append(float(loss))
    return field, history
