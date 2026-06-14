"""Stage B — reconstruct the current from passive drifters (direct regression).

Drifter buoys float *with* the current, so their tracks are integral curves of
``W`` (``dx/dt = W(x, t)``), NOT time-optimal geodesics. The current is therefore
recovered by **direct velocity regression**, not by inverting an optimal-control
solver. This is the honest framing (a drifter measures the flow it sits in).

Two reconstructions:

* **Geostrophic stream function** (primary): fit a scalar ``ψ`` network so that
  ``∇^⊥ψ`` matches the finite-difference drifter velocities. Divergence-free *by
  construction* — the physically correct prior for mesoscale surface flow, and it
  halves the degrees of freedom. Its honest blind spot is the divergent Ekman
  drift (not representable by any ``ψ``), which the reconstruction cannot recover.
* **Kernel smoother** (ablation): :class:`ham.models.learned.KernelWindField`,
  a non-parametric Nadaraya–Watson average of the observed velocities — captures
  divergence but extrapolates poorly off-track.

Reconstruction is on the surface (``z = 0``) horizontal current; recovering the
depth structure needs profiling floats (a documented extension).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ham.models.learned import KernelWindField


class DrifterObs(NamedTuple):
    """Sparse drifter observations (surface, horizontal)."""

    positions: jax.Array  # (M, 2)
    times: jax.Array  # (M,)
    velocities: jax.Array  # (M, 2) finite-difference, noisy


def simulate_drifters(
    medium,
    n_drifters: int = 12,
    t_span: float = 12.0,
    dt: float = 0.1,
    ping_interval: int = 5,
    noise: float = 0.01,
    key=None,
    region=((1.0, 9.0), (1.0, 9.0)),
) -> DrifterObs:
    """Release passive drifters and record noisy GPS pings → FD velocities.

    Integrates ``dx/dt = W_horiz(x, t)`` (RK4) at the surface, samples positions
    every ``ping_interval`` steps with Gaussian GPS noise, and finite-differences
    consecutive pings into velocity observations.
    """
    key = jax.random.PRNGKey(0) if key is None else key
    (x0, x1), (y0, y1) = region
    k_pos, k_noise = jax.random.split(key)
    starts = jax.random.uniform(
        k_pos, (n_drifters, 2), minval=jnp.array([x0, y0]), maxval=jnp.array([x1, y1])
    )

    def vel(xy, t):
        x3 = jnp.array([xy[0], xy[1], 0.0])
        return medium.physical_current(x3, t)[:2]

    n_steps = int(t_span / dt)

    def rollout(xy0):
        def step(xy, i):
            t = i * dt
            k1 = vel(xy, t)
            k2 = vel(xy + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = vel(xy + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = vel(xy + dt * k3, t + dt)
            xy_next = xy + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return xy_next, (xy, t)

        _, (traj, ts) = jax.lax.scan(step, xy0, jnp.arange(n_steps))
        return traj, ts

    trajs, ts = jax.vmap(rollout)(starts)  # (N, n_steps, 2), (N, n_steps)

    idx = jnp.arange(0, n_steps - ping_interval, ping_interval)
    pos = trajs[:, idx, :]  # (N, P, 2)
    pos_next = trajs[:, idx + ping_interval, :]
    t_obs = ts[:, idx]
    dt_ping = ping_interval * dt
    vel_obs = (pos_next - pos) / dt_ping

    pos = pos.reshape(-1, 2)
    t_obs = t_obs.reshape(-1)
    vel_obs = vel_obs.reshape(-1, 2)

    pos = pos + noise * jax.random.normal(k_noise, pos.shape)
    return DrifterObs(positions=pos, times=t_obs, velocities=vel_obs)


# =============================================================================
# Geostrophic stream-function reconstruction
# =============================================================================
class StreamFunctionField(eqx.Module):
    """Divergence-free current ``W = ∇^⊥ψ`` from a scalar ``ψ`` network."""

    mlp: eqx.nn.MLP
    scale: float = eqx.field(static=True)

    def __init__(self, key, hidden=64, depth=3, scale=1.0):
        self.mlp = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            width_size=hidden,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        self.scale = float(scale)

    def psi(self, xy: jax.Array) -> jax.Array:
        return self.scale * self.mlp(xy)[0]

    def horizontal(self, xy: jax.Array) -> jax.Array:
        g = jax.grad(self.psi)(xy)  # (∂ψ/∂x, ∂ψ/∂y)
        return jnp.array([g[1], -g[0]])  # ∇^⊥ψ

    def __call__(self, x: jax.Array) -> jax.Array:
        """Return a 3D current (horizontal from ψ; ``W_z = 0``)."""
        wh = self.horizontal(x[:2])
        return jnp.array([wh[0], wh[1], 0.0])


def fit_streamfunction(obs: DrifterObs, key=None, iters=1500, lr=3e-3, smooth=1e-3):
    """Fit ``ψ`` so ``∇^⊥ψ`` matches observed velocities (divergence-free)."""
    key = jax.random.PRNGKey(1) if key is None else key
    field = StreamFunctionField(key)

    def loss_fn(model):
        pred = jax.vmap(model.horizontal)(obs.positions)
        data = jnp.mean(jnp.sum((pred - obs.velocities) ** 2, axis=-1))
        # mild curvature penalty on ψ for smooth extrapolation off-track
        lap = jax.vmap(lambda p: jnp.trace(jax.hessian(model.psi)(p)))(obs.positions)
        return data + smooth * jnp.mean(lap**2)

    opt = optax.adam(lr)
    state = opt.init(eqx.filter(field, eqx.is_array))

    @eqx.filter_jit
    def step(model, state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, state = opt.update(grads, state)
        return eqx.apply_updates(model, updates), state, loss

    for _ in range(iters):
        field, state, _ = step(field, state)
    return field


def fit_kernel(obs: DrifterObs, sigma=0.8):
    """Non-parametric Nadaraya–Watson reconstruction (ablation baseline)."""
    kf = KernelWindField(obs.positions, obs.velocities, sigma=sigma)

    def current_3d(x):
        wh = kf(x[:2])
        return jnp.array([wh[0], wh[1], 0.0])

    return current_3d
