from typing import Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from experiments.wildfire.synthetic.experiment_base import SyntheticZermeloMetric
from ham.solvers.avbd import AVBDSolver
from ham.solvers.eikonal import EikonalSolver
from ham.training.losses import ArrivalTimeLoss


def compute_tv_regularization(
    H_grid: jax.Array, W_grid: Optional[jax.Array], lambda_H: float, lambda_W: float
) -> jax.Array:
    """Compute Total Variation (TV) regularization for H and W grids.

    Uses mean-reduction — appropriate for the AVBD solver whose data loss
    is also mean-normalised (via ArrivalTimeLoss).
    """
    loss_reg = 0.0

    if lambda_H > 0:
        # H_grid shape: (3, M, N)
        dh_dx = H_grid[:, :, 1:] - H_grid[:, :, :-1]
        dh_dy = H_grid[:, 1:, :] - H_grid[:, :-1, :]
        tv_H = jnp.mean(jnp.abs(dh_dx)) + jnp.mean(jnp.abs(dh_dy))
        loss_reg += lambda_H * tv_H

    if W_grid is not None and lambda_W > 0:
        # W_grid shape: (2, M, N)
        dw_dx = W_grid[:, :, 1:] - W_grid[:, :, :-1]
        dw_dy = W_grid[:, 1:, :] - W_grid[:, :-1, :]
        tv_W = jnp.mean(jnp.abs(dw_dx)) + jnp.mean(jnp.abs(dw_dy))
        loss_reg += lambda_W * tv_W

    return loss_reg


def compute_tv_regularization_eikonal(
    G: jax.Array, B: Optional[jax.Array], lambda_G: float, lambda_B: float
) -> jax.Array:
    """TV regularization matching Gahtan et al. exactly.

    Operates on Godunov parameters (G, B) — not Zermelo (H, W) — and uses
    *sum* (not mean) reduction so the effective strength is grid-size-proportional,
    identical to Gahtan's ``compute_tv_regularization`` in experiment_base.py.
    """
    loss_reg = 0.0

    if lambda_G > 0:
        # G shape: (3, M, N) — iterate over each symmetric-tensor component
        for c in range(3):
            Gc = G[c]
            loss_reg += lambda_G * (
                jnp.sum(jnp.abs(Gc[:, 1:] - Gc[:, :-1]))
                + jnp.sum(jnp.abs(Gc[1:, :] - Gc[:-1, :]))
            )

    if B is not None and lambda_B > 0:
        # B shape: (2, M, N)
        for c in range(2):
            Bc = B[c]
            loss_reg += lambda_B * (
                jnp.sum(jnp.abs(Bc[:, 1:] - Bc[:, :-1]))
                + jnp.sum(jnp.abs(Bc[1:, :] - Bc[:-1, :]))
            )

    return loss_reg


def eikonal_to_zermelo(G: jax.Array, B: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Convert Eikonal Godunov variables (G, B) to Zermelo (H, W) assuming lam=1.
    G shape (3, ...), B shape (2, ...)
    """
    # G = (H + (HW)(HW)^T)
    # B = -HW
    # Therefore, HW = -B
    # H = G - (-B)(-B)^T = G - BB^T
    # And since HW = -B => H W = -B => W = H^-1 (-B)

    # H = G - B B^T
    g11, g12, g22 = G[0], G[1], G[2]
    b1, b2 = B[0], B[1]

    h11 = g11 - b1 * b1
    h12 = g12 - b1 * b2
    h22 = g22 - b2 * b2
    H = jnp.stack([h11, h12, h22], axis=0)

    # Det H
    detH = h11 * h22 - h12**2
    detH_safe = jnp.maximum(detH, 1e-6)

    # H^-1
    inv_h11 = h22 / detH_safe
    inv_h12 = -h12 / detH_safe
    inv_h22 = h11 / detH_safe

    # W = H^-1 (-B)
    w1 = inv_h11 * (-b1) + inv_h12 * (-b2)
    w2 = inv_h12 * (-b1) + inv_h22 * (-b2)
    W = jnp.stack([w1, w2], axis=0)

    return H, W


def zermelo_to_eikonal(H: jax.Array, W: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Convert Zermelo (H, W) back to Eikonal (G, B) with lam=1.
    H shape (3, ...), W shape (2, ...)
    """
    h11, h12, h22 = H[0], H[1], H[2]
    w1, w2 = W[0], W[1]

    b1 = -(h11 * w1 + h12 * w2)
    b2 = -(h12 * w1 + h22 * w2)
    B = jnp.stack([b1, b2], axis=0)

    g11 = h11 + b1 * b1
    g12 = h12 + b1 * b2
    g22 = h22 + b2 * b2
    G = jnp.stack([g11, g12, g22], axis=0)

    return G, B


class MetricRecoveryModel(eqx.Module):
    """Holds the parameters to be optimized."""

    H_grid: jax.Array
    W_grid: jax.Array

    def __init__(
        self, M: int, N: int, init_H: jax.Array = None, init_W: jax.Array = None
    ):
        if init_H is None:
            # Initialize with isotropic H=I
            self.H_grid = jnp.zeros((3, M, N))
            self.H_grid = self.H_grid.at[0].set(1.0)
            self.H_grid = self.H_grid.at[2].set(1.0)
        else:
            self.H_grid = init_H

        if init_W is None:
            self.W_grid = jnp.zeros((2, M, N))
        else:
            self.W_grid = init_W

    def project(
        self, constrain_isotropic: bool, constant_W: bool = False
    ) -> "MetricRecoveryModel":
        """Projects H to be SPD and optionally isotropic.

        If ``constant_W``, the drift field is projected onto spatially
        constant fields (its spatial mean). This restores identifiability
        for single-source recovery: per-pixel drift is only constrained
        along characteristics, while a constant drift (2 parameters) is
        fully determined by dense arrival times.
        """
        H = self.H_grid

        # Clip diagonals to avoid blowup or collapse
        h11 = jnp.clip(H[0], 0.1, 10.0)
        h22 = jnp.clip(H[2], 0.1, 10.0)
        h12 = H[1]

        if constrain_isotropic:
            avg = (h11 + h22) / 2.0
            h11 = avg
            h22 = avg
            h12 = jnp.zeros_like(h12)
        else:
            # Enforce determinant > 0: h12^2 < h11 * h22
            max_h12 = jnp.sqrt(h11 * h22) * 0.95
            h12 = jnp.clip(h12, -max_h12, max_h12)

        H_new = jnp.stack([h11, h12, h22], axis=0)

        W = self.W_grid
        if constant_W:
            W = jnp.mean(W, axis=(1, 2), keepdims=True) * jnp.ones_like(W)

        # Project W for causal bound: ||W||_H < 1
        # This is roughly equivalent to ||B||_G^-1 < 1
        W_norm_sq = W[0] * (h11 * W[0] + h12 * W[1]) + W[1] * (h12 * W[0] + h22 * W[1])

        mask = W_norm_sq > 0.81  # Max magnitude 0.9
        scale = jnp.where(mask, 0.9 / jnp.maximum(jnp.sqrt(W_norm_sq), 1e-8), 1.0)
        W_new = W * scale

        return eqx.tree_at(lambda m: (m.H_grid, m.W_grid), self, (H_new, W_new))


class MetricRecoveryOptimizer:
    """Optimizer for synthetic metric recovery using JAX and Optax.
    Supports both 'eikonal' and 'avbd' solvers.
    """

    def __init__(
        self,
        M: int,
        N: int,
        solver_type: str = "eikonal",
        recover_H: bool = True,
        recover_W: bool = False,
        constrain_isotropic: bool = True,
        constant_W: bool = False,
        lambda_H: float = 0.01,
        lambda_W: float = 0.01,
        reg_type: str = "tv",
        optimizer_type: str = "adam",
    ):
        self.M = M
        self.N = N
        self.solver_type = solver_type
        self.recover_H = recover_H
        self.recover_W = recover_W
        self.constrain_isotropic = constrain_isotropic
        self.constant_W = constant_W
        self.lambda_H = lambda_H
        self.lambda_W = lambda_W
        self.reg_type = reg_type
        self.optimizer_type = optimizer_type

        self.model = MetricRecoveryModel(M, N)

        if self.solver_type == "eikonal":
            self.solver = EikonalSolver(max_iters=50, tol=1e-5)
            # Data loss is the Gahtan-exact sum-reduced MSE computed inline in
            # `fit` (0.5 * sum(diff^2)); no separate loss module is needed.
            self.loss_fn = None
        elif self.solver_type == "avbd":
            self.solver = AVBDSolver(step_size=0.1, iterations=40, parallel=True)
            self.loss_fn = ArrivalTimeLoss(
                solver=self.solver, solver_steps=20, weight=1.0
            )
        else:
            raise ValueError(f"Unknown solver_type: {solver_type}")

        self.history = {"loss": [], "loss_data": [], "loss_reg": []}

    def fit(
        self,
        source_coords: jax.Array,
        T_obs: jax.Array,
        obs_mask: jax.Array,
        n_iter: int = 500,
        lr: float = 0.1,
        verbose: bool = True,
        alpha: float = 0.0,
        patience: int = 20,
        min_delta: float = 1e-4,
        test_mask: Optional[jax.Array] = None,
    ) -> Dict:
        """Fit parameters to observed arrival times."""

        # We need continuous coordinates for observations if using AVBD
        M, N = self.M, self.N
        I, J = jnp.meshgrid(jnp.arange(M), jnp.arange(N), indexing="ij")
        obs_coords = jnp.stack([I[obs_mask], J[obs_mask]], axis=-1).astype(jnp.float32)
        T_obs_values = T_obs[obs_mask]

        if source_coords.ndim == 1:
            source_coords = source_coords[None, :]

        # Setup Optax: exponentially decaying learning rate (x0.8 every 100 steps)
        schedule = optax.exponential_decay(
            init_value=lr, transition_steps=100, decay_rate=0.8
        )
        # Filter trainable parameters
        filter_spec = jax.tree_util.tree_map(lambda _: False, self.model)
        filter_spec = eqx.tree_at(lambda m: m.H_grid, filter_spec, self.recover_H)
        filter_spec = eqx.tree_at(lambda m: m.W_grid, filter_spec, self.recover_W)
        if self.optimizer_type == "adam":
            optimizer = optax.adam(learning_rate=schedule)
        else:
            optimizer = optax.sgd(learning_rate=schedule)

        opt_state = optimizer.init(eqx.filter(self.model, filter_spec))

        @eqx.filter_value_and_grad(has_aux=True)
        def compute_loss(
            diff_model: MetricRecoveryModel,
            static_model: MetricRecoveryModel,
            current_alpha: jax.Array,
        ):
            model = eqx.combine(diff_model, static_model)
            metric = SyntheticZermeloMetric(model.H_grid, model.W_grid)

            if self.solver_type == "eikonal":
                T_pred, _, _ = self.solver.solve(
                    metric,
                    source_coords,
                    grid_extent=(0, M - 1, 0, N - 1),
                    grid_shape=(M, N),
                )

                diff = T_pred[obs_mask] - T_obs_values
                loss_data = 0.5 * jnp.sum(diff**2)

                lam_H = self.lambda_H if self.recover_H else 0.0
                lam_W = self.lambda_W if self.recover_W else 0.0
                if self.reg_type == "tv":
                    # Gahtan-exact regularization: sum-reduced TV on G/B params.
                    G, B = zermelo_to_eikonal(model.H_grid, model.W_grid)
                    loss_reg = compute_tv_regularization_eikonal(
                        G, B if self.recover_W else None, lam_H, lam_W
                    )
                elif self.reg_type == "tikhonov":
                    # Sum-reduced Tikhonov (ridge) on the Godunov params, to
                    # match the TV reduction so lambda is comparable.
                    G, B = zermelo_to_eikonal(model.H_grid, model.W_grid)
                    loss_reg = lam_H * jnp.sum(G**2)
                    if self.recover_W:
                        loss_reg += lam_W * jnp.sum(B**2)
                else:
                    loss_reg = 0.0
            else:
                # Pass all ignition points: ArrivalTimeLoss takes the min over
                # per-source arc lengths (multi-ignition union). Using only
                # source_coords[0] here previously skewed every multi-source
                # AVBD recovery (e.g. C3/C10) against union-field observations.
                src = source_coords if source_coords.shape[0] > 1 else source_coords[0]
                loss_data = self.loss_fn(
                    metric, src, obs_coords, T_obs_values, alpha=current_alpha
                )

                if self.reg_type == "tv":
                    loss_reg = compute_tv_regularization(
                        model.H_grid,
                        model.W_grid,
                        self.lambda_H if self.recover_H else 0.0,
                        self.lambda_W if self.recover_W else 0.0,
                    )
                elif self.reg_type == "tikhonov":
                    loss_reg = 0.0
                    if self.recover_H:
                        loss_reg += self.lambda_H * jnp.sum(model.H_grid**2)
                    if self.recover_W:
                        loss_reg += self.lambda_W * jnp.sum(model.W_grid**2)
                else:
                    loss_reg = 0.0

            loss = loss_data + loss_reg
            return loss, (loss_data, loss_reg)

        @eqx.filter_jit
        def make_step(model, opt_state, current_alpha):
            diff_model, static_model = eqx.partition(model, filter_spec)
            (loss, (loss_data, loss_reg)), grads = compute_loss(
                diff_model, static_model, current_alpha
            )
            updates, opt_state = optimizer.update(grads, opt_state, diff_model)
            diff_model = eqx.apply_updates(diff_model, updates)
            model = eqx.combine(diff_model, static_model)
            model = model.project(self.constrain_isotropic, self.constant_W)
            return model, opt_state, loss, loss_data, loss_reg

        try:
            from tqdm.auto import tqdm

            pbar = tqdm(
                range(n_iter),
                desc=f"Training ({self.solver_type})",
                disable=not verbose,
                leave=False,
            )
        except ImportError:
            pbar = range(n_iter)

        best_loss = float("inf")
        no_improvement_count = 0
        converged = False

        # Determine when alpha curriculum finishes so we don't early stop during ramp
        alpha_done_iter = 0
        warmup = int(0.2 * n_iter)
        ramp = int(0.6 * n_iter)
        if self.solver_type == "avbd":
            alpha_done_iter = warmup + ramp

        for it in pbar:
            if it < warmup:
                alpha_val = 0.0
            elif it > alpha_done_iter:
                alpha_val = 1.0
            else:
                alpha_val = (it - warmup) / max(ramp, 1)

            current_alpha = jnp.array(alpha_val, dtype=jnp.float32)
            self.model, opt_state, loss, loss_data, loss_reg = make_step(
                self.model, opt_state, current_alpha
            )

            loss_val = float(loss)
            self.history["loss"].append(loss_val)
            self.history["loss_data"].append(float(loss_data))
            self.history["loss_reg"].append(float(loss_reg))

            if verbose:
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        loss=f"{loss_val:.4f}", data=f"{float(loss_data):.4f}"
                    )
                elif it % 50 == 0 or it == n_iter - 1:
                    print(
                        f"  Iter {it}: loss={loss_val:.4f} (data={float(loss_data):.4f}, reg={float(loss_reg):.4f})"
                    )

            if it < alpha_done_iter:
                # Reset best loss continuously while curriculum changes loss landscape
                best_loss = loss_val
                no_improvement_count = 0
            else:
                if loss_val < best_loss - min_delta:
                    best_loss = loss_val
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Relative plateau check
                if it >= 10:
                    loss_10_ago = self.history["loss"][-11]
                    rel_change = abs(loss_val - loss_10_ago) / max(abs(loss_10_ago), 1e-10)
                    if rel_change < 1e-5:
                        converged = True
                        if verbose:
                            msg = f"  Converged at iteration {it} (relative tolerance)"
                            if hasattr(pbar, "write"):
                                pbar.write(msg)
                            else:
                                print(msg)
                        break

            if no_improvement_count >= patience:
                if verbose:
                    msg = f"  Early stopping at iteration {it} (no improvement in {patience} iters)"
                    if hasattr(pbar, "write"):
                        pbar.write(msg)
                    else:
                        print(msg)
                break

        results = {
            "final_loss": self.history["loss"][-1],
            "final_data": self.history["loss_data"][-1],
            "final_reg": self.history["loss_reg"][-1],
            "iterations": len(self.history["loss"]),
            "converged": converged,
        }

        if test_mask is not None:
            # Recompute T_pred to evaluate train/test error
            from experiments.wildfire.synthetic.experiment_base import compute_errors
            from ham.solvers.eikonal import EikonalSolver
            
            metric_final = SyntheticZermeloMetric(self.model.H_grid, self.model.W_grid)
            eval_solver = EikonalSolver(max_iters=100, tol=1e-4)
            T_pred, _, _ = eval_solver.solve(
                metric_final,
                source_coords,
                grid_extent=(0, M - 1, 0, N - 1),
                grid_shape=(M, N),
            )
            train_err = compute_errors(T_pred, T_obs, obs_mask)
            test_err = compute_errors(T_pred, T_obs, test_mask)
            results["train_mre"] = train_err["mre"]
            results["test_mre"] = test_err["mre"]

        return results

    def get_G_B(self) -> Tuple[jax.Array, jax.Array]:
        """Returns the recovered parameters in Eikonal G/B form."""
        return zermelo_to_eikonal(self.model.H_grid, self.model.W_grid)
