from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from ham.geometry.metric import FinslerMetric
from ham.solvers.avbd import Trajectory

__all__ = ["GeodesicLearningSolver"]


class GeodesicLearningSolver(eqx.Module):
    """Geodesic Learning approach (as used in RiemannEBM).

    Finds the energy-minimizing path by optimizing the entire discrete path
    simultaneously using the optax.adam optimizer.
    """

    step_size: float = eqx.field(static=True, default=1e-2)
    iterations: int = eqx.field(static=True, default=5000)

    def solve(
        self,
        metric: FinslerMetric,
        p_start: jax.Array,
        p_end: jax.Array,
        n_steps: int = 10,
        constraints: Optional[list[Callable[[jax.Array], jax.Array]]] = None,
        train_mode: bool = True,
        key: Optional[jax.Array] = None,
    ) -> Trajectory:

        t = jnp.linspace(0, 1, n_steps + 1)[:, None]
        linear_path = (1 - t) * p_start + t * p_end

        if key is None:
            seed_val = jnp.sum(p_start + p_end).astype(jnp.int32)
            key = jax.random.fold_in(jax.random.PRNGKey(0), seed_val)

        noise = jax.random.normal(key, shape=linear_path.shape) * 1e-4
        path_guess = jax.vmap(metric.manifold.project)(linear_path + noise)
        init_inner = path_guess[1:-1]

        optimizer = optax.adam(self.step_size)
        opt_state = optimizer.init(init_inner)

        def loss_fn(inner):
            full_xs = jnp.concatenate([p_start[None, :], inner, p_end[None, :]], axis=0)
            full_vs = jax.vmap(metric.manifold.log_map)(full_xs[:-1], full_xs[1:])
            # Total path energy
            energy = jnp.sum(jax.vmap(metric.energy)(full_xs[:-1], full_vs))
            return energy

        def step_fn(carry, _):
            inner, opt_s = carry
            loss, grad = jax.value_and_grad(loss_fn)(inner)

            # Project Euclidean gradient to tangent space
            grad_tan = jax.vmap(metric.manifold.to_tangent)(inner, grad)

            # Apply optax update rule
            updates, new_opt_s = optimizer.update(grad_tan, opt_s, inner)

            # Since updates is typically -lr * grad_tan, we retract the update delta directly
            new_inner = jax.vmap(metric.manifold.retract)(inner, updates)

            return (new_inner, new_opt_s), loss

        if train_mode:
            (final_inner, _), _ = jax.lax.scan(
                step_fn, (init_inner, opt_state), None, length=self.iterations
            )
        else:
            (final_inner, _), _ = jax.lax.scan(
                step_fn, (init_inner, opt_state), None, length=self.iterations
            )

        full_xs = jnp.concatenate(
            [p_start[None, :], final_inner, p_end[None, :]], axis=0
        )
        full_vs = jax.vmap(metric.manifold.log_map)(full_xs[:-1], full_xs[1:])
        final_energy = jnp.sum(jax.vmap(metric.energy)(full_xs[:-1], full_vs))

        n_inner = n_steps - 1
        return Trajectory(
            xs=full_xs,
            vs=full_vs,
            energy=final_energy,
            constraint_violation=jnp.array(0.0),
            lambdas=jnp.zeros((n_inner, 0)),
            stiffness=jnp.ones((n_inner, 0)),
        )
