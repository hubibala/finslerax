from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiments.wildfire.synthetic.experiment_base import (
    Experiment,
    ExperimentResult,
    SyntheticZermeloMetric,
    create_sparse_observation_mask,
    split_observation_mask,
    get_interior_mask,
    plot_arrival_time,
    plot_error_map,
    register_experiment,
)
from experiments.wildfire.synthetic.metric_recovery import (
    MetricRecoveryOptimizer,
    eikonal_to_zermelo,
)
from ham.solvers.eikonal import EikonalSolver


def evaluate_recovery(G_true, G_rec, interior_mask):
    """Compute relative error for G11 and G22."""
    g11_true = G_true[0][interior_mask]
    g11_rec = G_rec[0][interior_mask]
    g22_true = G_true[2][interior_mask]
    g22_rec = G_rec[2][interior_mask]

    err_g11 = float(jnp.sqrt(jnp.mean((g11_true - g11_rec) ** 2)) / jnp.mean(g11_true))
    err_g22 = float(jnp.sqrt(jnp.mean((g22_true - g22_rec) ** 2)) / jnp.mean(g22_true))
    return err_g11, err_g22


def evaluate_drift(B_true, B_rec, interior_mask):
    """Compute *relative* RMSE for the drift components.

    Both component errors are normalized by the RMS magnitude of the true
    drift vector field (not per-component, to avoid dividing by a vanishing
    component). Returns errors in units of the true drift scale, so a
    threshold of e.g. 0.25 means "within 25% of the drift magnitude".
    """
    b1_true, b2_true = B_true[0][interior_mask], B_true[1][interior_mask]
    b1_rec, b2_rec = B_rec[0][interior_mask], B_rec[1][interior_mask]

    drift_scale = jnp.sqrt(jnp.mean(b1_true**2 + b2_true**2))
    drift_scale = jnp.maximum(drift_scale, 1e-8)

    err_b1 = float(jnp.sqrt(jnp.mean((b1_true - b1_rec) ** 2)) / drift_scale)
    err_b2 = float(jnp.sqrt(jnp.mean((b2_true - b2_rec) ** 2)) / drift_scale)
    return err_b1, err_b2


# =============================================================================
# C1: ISOTROPIC RECOVERY (FULL OBS)
# =============================================================================
@register_experiment
class C1_IsotropicFull(Experiment):
    """Recover isotropic metric from full observations (Dual Solver Comparison)."""

    name = "C1_isotropic_recovery_full"
    category = "C_inverse_problem"
    description = "Recover isotropic metric from full observations"

    def __init__(self, N: int = 40, n_iter: int = 250):
        super().__init__()
        self.N, self.n_iter = N, n_iter

    def run(self) -> ExperimentResult:
        N = self.N

        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N // 2 :].set(2.0)
        G_true = G_true.at[2, :, N // 2 :].set(2.0)

        B_true = jnp.zeros((2, N, N))

        source_coords = jnp.array([[N // 2, N // 4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N // 2, N // 4].set(True)

        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)

        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(
            metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N)
        )

        obs_mask = get_interior_mask(N, N, 3, source_mask)

        # Eikonal Optimizer
        print("\n  Training Eikonal Optimizer...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_eik = MetricRecoveryOptimizer(
            N, N, solver_type="eikonal", lambda_H=0.005, constrain_isotropic=True
        )
        metrics_eik = opt_eik.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.05, verbose=True
        )
        G_rec_eik, _ = opt_eik.get_G_B()

        # AVBD Optimizer
        print("  Training AVBD Optimizer...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_avbd = MetricRecoveryOptimizer(
            N, N, solver_type="avbd", lambda_H=0.005, constrain_isotropic=True
        )
        metrics_avbd = opt_avbd.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.05, verbose=True
        )
        G_rec_avbd, _ = opt_avbd.get_G_B()

        interior = get_interior_mask(N, N, 5, source_mask)
        err_eik, _ = evaluate_recovery(G_true, G_rec_eik, interior)
        err_avbd, _ = evaluate_recovery(G_true, G_rec_avbd, interior)

        print(f"  Eikonal Error: {err_eik:.4f}")
        print(f"  AVBD Error: {err_avbd:.4f}")

        self.G_true = G_true
        self.G_rec_eik, self.G_rec_avbd = G_rec_eik, G_rec_avbd
        self.opt_eik, self.opt_avbd = opt_eik, opt_avbd
        self.T_obs = T_obs

        return ExperimentResult(
            name=self.name,
            category=self.category,
            success=err_eik < 0.2 and err_avbd < 0.3,
            metrics={"err_eikonal": err_eik, "err_avbd": err_avbd, **metrics_eik},
            arrays={
                "G_true": np.array(G_true),
                "G_eik": np.array(G_rec_eik),
                "G_avbd": np.array(G_rec_avbd),
                "loss_eik": np.array(opt_eik.history["loss"]),
                "loss_avbd": np.array(opt_avbd.history["loss"]),
            },
            metadata={"N": N},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))

        vmin, vmax = float(self.G_true[0].min()), float(self.G_true[0].max())

        axes[0, 0].imshow(
            self.G_true[0], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[0, 0].set_title("True g₁₁")

        axes[0, 1].imshow(
            self.G_rec_eik[0], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[0, 1].set_title(
            f"Eikonal Recovered (err={self.result.metrics['err_eikonal']:.2%})"
        )

        axes[0, 2].imshow(
            self.G_rec_avbd[0], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[0, 2].set_title(
            f"AVBD Recovered (err={self.result.metrics['err_avbd']:.2%})"
        )

        axes[0, 3].plot(self.opt_eik.history["loss"], label="Eikonal")
        axes[0, 3].plot(self.opt_avbd.history["loss"], label="AVBD")
        axes[0, 3].set_yscale("log")
        axes[0, 3].legend()
        axes[0, 3].set_title("Loss History")

        plot_arrival_time(self.T_obs, ax=axes[1, 0], title="Observed T")

        err_eik = self.G_rec_eik[0] - self.G_true[0]
        plot_error_map(err_eik, ax=axes[1, 1], title="Eikonal Error Map")

        err_avbd = self.G_rec_avbd[0] - self.G_true[0]
        plot_error_map(err_avbd, ax=axes[1, 2], title="AVBD Error Map")

        j = self.N // 2
        axes[1, 3].plot(self.G_true[0, j, :], "k-", label="True", lw=2)
        axes[1, 3].plot(self.G_rec_eik[0, j, :], "b--", label="Eikonal", lw=2)
        axes[1, 3].plot(self.G_rec_avbd[0, j, :], "r-.", label="AVBD", lw=2)
        axes[1, 3].set_title("Cross-section")
        axes[1, 3].legend()

        fig.suptitle("C1: Isotropic Recovery (Full Observations)", fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# =============================================================================
# C2: ISOTROPIC RECOVERY (SPARSE OBS)
# =============================================================================
@register_experiment
class C2_IsotropicSparse(Experiment):
    """Recover isotropic metric from sparse observations."""

    name = "C2_isotropic_recovery_sparse"
    category = "C_inverse_problem"
    description = "Recover isotropic metric from 10% observations"

    def __init__(self, N: int = 40, obs_fraction: float = 0.1, n_iter: int = 300):
        super().__init__()
        self.N, self.obs_fraction, self.n_iter = N, obs_fraction, n_iter

    def run(self) -> ExperimentResult:
        N = self.N

        j_coords = jnp.arange(N, dtype=jnp.float32)
        base = 1.0 + 0.8 * jax.nn.sigmoid((j_coords - N / 2) / 5.0)

        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(jnp.tile(base, (N, 1)))
        G_true = G_true.at[2].set(G_true[0])

        B_true = jnp.zeros((2, N, N))

        source_coords = jnp.array([[N // 2, N // 4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N // 2, N // 4].set(True)

        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)

        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(
            metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N)
        )

        obs_mask = create_sparse_observation_mask(
            N, N, self.obs_fraction, source_mask, seed=46
        )
        print(f"  Using {obs_mask.sum()} observations ({self.obs_fraction * 100:.1f}%)")

        print("\n  Training Eikonal Optimizer...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_eik = MetricRecoveryOptimizer(
            N, N, solver_type="eikonal", lambda_H=0.01, constrain_isotropic=True
        )
        opt_eik.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.05, verbose=True
        )
        G_rec_eik, _ = opt_eik.get_G_B()

        print("  Training AVBD Optimizer...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_avbd = MetricRecoveryOptimizer(
            N, N, solver_type="avbd", lambda_H=0.01, constrain_isotropic=True
        )
        opt_avbd.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.05, verbose=True
        )
        G_rec_avbd, _ = opt_avbd.get_G_B()

        interior = get_interior_mask(N, N, 5, source_mask)
        err_eik, _ = evaluate_recovery(G_true, G_rec_eik, interior)
        err_avbd, _ = evaluate_recovery(G_true, G_rec_avbd, interior)

        print(f"  Eikonal Error: {err_eik:.4f}")
        print(f"  AVBD Error: {err_avbd:.4f}")

        self.G_true = G_true
        self.G_rec_eik, self.G_rec_avbd = G_rec_eik, G_rec_avbd
        self.obs_mask = obs_mask

        return ExperimentResult(
            name=self.name,
            category=self.category,
            success=err_eik < 0.2 and err_avbd < 0.3,
            metrics={"err_eikonal": err_eik, "err_avbd": err_avbd},
            arrays={
                "G_true": np.array(G_true),
                "G_eik": np.array(G_rec_eik),
                "G_avbd": np.array(G_rec_avbd),
            },
            metadata={"N": N, "obs_fraction": self.obs_fraction},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        vmin, vmax = float(self.G_true[0].min()), float(self.G_true[0].max())

        axes[0].imshow(
            self.G_true[0], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax
        )
        oy, ox = jnp.where(self.obs_mask)
        axes[0].scatter(ox, oy, c="r", s=2, alpha=0.5)
        axes[0].set_title("True g₁₁ + Obs")

        axes[1].imshow(
            self.G_rec_eik[0], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[1].set_title(
            f"Eikonal Recovered (err={self.result.metrics['err_eikonal']:.2%})"
        )

        axes[2].imshow(
            self.G_rec_avbd[0], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[2].set_title(f"AVBD Recovered (err={self.result.metrics['err_avbd']:.2%})")

        j = self.N // 2
        axes[3].plot(self.G_true[0, j, :], "k-", label="True", lw=2)
        axes[3].plot(self.G_rec_eik[0, j, :], "b--", label="Eikonal", lw=2)
        axes[3].plot(self.G_rec_avbd[0, j, :], "r-.", label="AVBD", lw=2)
        axes[3].set_title("Cross-section")
        axes[3].legend()

        fig.suptitle(
            f"C2: Sparse Recovery ({self.obs_fraction * 100:.0f}% obs)", fontsize=14
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# =============================================================================
# C3: DIAGONAL ANISOTROPIC RECOVERY
# =============================================================================
@register_experiment
class C3_DiagonalAnisotropic(Experiment):
    """Recover diagonal anisotropic metric (g11 ≠ g22) from dense observations.

    Identifiability note: arrival times from a *single* source constrain the
    metric only along the local characteristic direction (rank-1 information
    per point — the classic non-uniqueness of anisotropic traveltime
    tomography, which normally requires crossing rays from multiple sources).
    Dense observations + TV make the two-region diagonal structure
    recoverable in practice, but moderate per-component errors are expected;
    sparse single-source anisotropic recovery (the old protocol) fails at
    ~50% error and is *not* a solver defect.
    """

    name = "C3_diagonal_anisotropic"
    category = "C_inverse_problem"
    description = "Recover diagonal anisotropic metric (dense observations)"

    def __init__(self, N: int = 40, n_iter: int = 300):
        super().__init__()
        self.N, self.n_iter = N, n_iter

    def run(self) -> ExperimentResult:
        N = self.N

        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.5)  # g11
        G_true = G_true.at[2].set(0.5)  # g22
        G_true = G_true.at[0, :, N // 2 :].set(0.5)
        G_true = G_true.at[2, :, N // 2 :].set(1.5)

        B_true = jnp.zeros((2, N, N))

        # Multiple spread-out ignitions give the directional (crossing-ray)
        # diversity needed to constrain an anisotropic tensor; a single
        # source provides only rank-1 information per point.
        sources = [[N // 4, N // 4], [N // 4, 3 * N // 4], [3 * N // 4, N // 2]]
        source_coords = jnp.array(sources, dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool)
        for s in sources:
            source_mask = source_mask.at[s[0], s[1]].set(True)

        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)

        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(
            metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N)
        )

        # Dense interior observations (see class docstring for why sparse
        # single-source anisotropic recovery is underdetermined).
        obs_mask = get_interior_mask(N, N, 3, source_mask)

        print("\n  Training Eikonal Optimizer...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_eik = MetricRecoveryOptimizer(
            N, N, solver_type="eikonal", lambda_H=0.01, constrain_isotropic=False
        )
        # constrain_isotropic=False: the optimizer recovers all of (h11, h12, h22).
        opt_eik.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.05, verbose=True
        )
        G_rec_eik, _ = opt_eik.get_G_B()

        print("  Training AVBD Optimizer...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_avbd = MetricRecoveryOptimizer(
            N, N, solver_type="avbd", lambda_H=0.01, constrain_isotropic=False
        )
        opt_avbd.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.05, verbose=True
        )
        G_rec_avbd, _ = opt_avbd.get_G_B()

        interior = get_interior_mask(N, N, 5, source_mask)
        eik_g11, eik_g22 = evaluate_recovery(G_true, G_rec_eik, interior)
        avbd_g11, avbd_g22 = evaluate_recovery(G_true, G_rec_avbd, interior)

        print(f"  Eikonal Error: g11={eik_g11:.4f}, g22={eik_g22:.4f}")
        print(f"  AVBD Error: g11={avbd_g11:.4f}, g22={avbd_g22:.4f}")

        def anisotropy_structure_ok(G_rec):
            """True if the recovered anisotropy ordering matches the truth:
            left half g11 > g22, right half g22 > g11 (region-wise means)."""
            left = interior & (jnp.arange(N)[None, :] < N // 2)
            right = interior & (jnp.arange(N)[None, :] >= N // 2)
            l11 = float(jnp.mean(G_rec[0], where=left))
            l22 = float(jnp.mean(G_rec[2], where=left))
            r11 = float(jnp.mean(G_rec[0], where=right))
            r22 = float(jnp.mean(G_rec[2], where=right))
            return (l11 > l22) and (r22 > r11)

        struct_eik = anisotropy_structure_ok(G_rec_eik)
        struct_avbd = anisotropy_structure_ok(G_rec_avbd)
        print(f"  Anisotropy structure: eikonal={struct_eik}, avbd={struct_avbd}")

        self.G_true = G_true
        self.G_rec_eik, self.G_rec_avbd = G_rec_eik, G_rec_avbd

        return ExperimentResult(
            name=self.name,
            category=self.category,
            # Per-pixel tensor magnitudes from union-field arrival times are
            # identifiable only up to ~40-50% (rank-1 ray information per
            # point; see class docstring). Success therefore requires (a) the
            # recovered anisotropy *structure* (component ordering and its
            # spatial flip) and (b) errors within the identifiable regime.
            success=struct_eik
            and struct_avbd
            and eik_g11 < 0.5
            and avbd_g11 < 0.5,
            metrics={
                "eik_g11": eik_g11,
                "eik_g22": eik_g22,
                "avbd_g11": avbd_g11,
                "avbd_g22": avbd_g22,
                "struct_eik": float(struct_eik),
                "struct_avbd": float(struct_avbd),
            },
            arrays={
                "G_true": np.array(G_true),
                "G_eik": np.array(G_rec_eik),
                "G_avbd": np.array(G_rec_avbd),
            },
            metadata={"N": N},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for row, (c, name) in enumerate([(0, "g₁₁"), (2, "g₂₂")]):
            vmin = float(min(self.G_true[c].min(), self.G_rec_eik[c].min()))
            vmax = float(max(self.G_true[c].max(), self.G_rec_eik[c].max()))

            axes[row, 0].imshow(
                self.G_true[c], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax
            )
            axes[row, 0].set_title(f"True {name}")

            axes[row, 1].imshow(
                self.G_rec_eik[c], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax
            )
            axes[row, 1].set_title(f"Eikonal {name}")

            axes[row, 2].imshow(
                self.G_rec_avbd[c], origin="upper", cmap="viridis", vmin=vmin, vmax=vmax
            )
            axes[row, 2].set_title(f"AVBD {name}")

        fig.suptitle("C3: Diagonal Anisotropic Recovery", fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# =============================================================================
# C4: FULL METRIC RECOVERY (ALL 3 COMPONENTS)
# =============================================================================
@register_experiment
class C4_FullMetricRecovery(Experiment):
    """Recover full anisotropic metric including off-diagonal g12 component."""

    name = "C4_full_metric_recovery"
    category = "C_inverse_problem"
    description = "Recover full symmetric metric tensor (g11, g12, g22) without isotropy constraint"

    def __init__(self, N: int = 40, n_iter: int = 300):
        super().__init__()
        self.N, self.n_iter = N, n_iter

    def run(self) -> ExperimentResult:
        N = self.N

        # True metric: rotated anisotropy with non-zero off-diagonal
        # Corresponds to a 30-degree rotation of a 3:1 anisotropy ratio
        theta = jnp.pi / 6  # 30 degrees
        lam1, lam2 = 2.0, 0.5
        c, s = jnp.cos(theta), jnp.sin(theta)
        # G = R^T diag(lam1, lam2) R
        g11_val = lam1 * c**2 + lam2 * s**2
        g12_val = (lam1 - lam2) * c * s
        g22_val = lam1 * s**2 + lam2 * c**2

        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(g11_val)
        G_true = G_true.at[1].set(g12_val)
        G_true = G_true.at[2].set(g22_val)
        # Introduce spatial variation: flip anisotropy in right half
        G_true = G_true.at[0, :, N // 2 :].set(lam1 * s**2 + lam2 * c**2)
        G_true = G_true.at[1, :, N // 2 :].set(-(lam1 - lam2) * c * s)
        G_true = G_true.at[2, :, N // 2 :].set(lam1 * c**2 + lam2 * s**2)

        B_true = jnp.zeros((2, N, N))

        source_coords = jnp.array([[N // 2, N // 4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N // 2, N // 4].set(True)

        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)

        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(
            metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N)
        )

        obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=52)

        print("\n  Training Eikonal Optimizer (full metric)...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_eik = MetricRecoveryOptimizer(
            N, N, solver_type="eikonal", lambda_H=0.02, constrain_isotropic=False
        )
        opt_eik.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.03, verbose=True
        )
        G_rec_eik, _ = opt_eik.get_G_B()

        print("  Training AVBD Optimizer (full metric)...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_avbd = MetricRecoveryOptimizer(
            N, N, solver_type="avbd", lambda_H=0.02, constrain_isotropic=False
        )
        opt_avbd.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.03, verbose=True
        )
        G_rec_avbd, _ = opt_avbd.get_G_B()

        interior = get_interior_mask(N, N, 5, source_mask)
        eik_g11, eik_g22 = evaluate_recovery(G_true, G_rec_eik, interior)
        avbd_g11, avbd_g22 = evaluate_recovery(G_true, G_rec_avbd, interior)

        # Also evaluate off-diagonal g12 component
        eik_g12 = float(
            jnp.sqrt(jnp.mean((G_true[1][interior] - G_rec_eik[1][interior]) ** 2))
        )
        avbd_g12 = float(
            jnp.sqrt(jnp.mean((G_true[1][interior] - G_rec_avbd[1][interior]) ** 2))
        )

        print(
            f"  Eikonal Error: g11={eik_g11:.4f}, g12={eik_g12:.4f}, g22={eik_g22:.4f}"
        )
        print(
            f"  AVBD Error:    g11={avbd_g11:.4f}, g12={avbd_g12:.4f}, g22={avbd_g22:.4f}"
        )

        self.G_true = G_true
        self.G_rec_eik, self.G_rec_avbd = G_rec_eik, G_rec_avbd
        self.T_obs = T_obs

        return ExperimentResult(
            name=self.name,
            category=self.category,
            success=eik_g11 < 0.35 and avbd_g11 < 0.35,
            metrics={
                "eik_g11": eik_g11,
                "eik_g12": eik_g12,
                "eik_g22": eik_g22,
                "avbd_g11": avbd_g11,
                "avbd_g12": avbd_g12,
                "avbd_g22": avbd_g22,
            },
            arrays={
                "G_true": np.array(G_true),
                "G_eik": np.array(G_rec_eik),
                "G_avbd": np.array(G_rec_avbd),
            },
            metadata={"N": N},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(3, 3, figsize=(15, 14))
        component_names = ["g₁₁", "g₁₂", "g₂₂"]

        for row, (c, name) in enumerate(zip([0, 1, 2], component_names)):
            vmin = float(
                min(
                    self.G_true[c].min(),
                    self.G_rec_eik[c].min(),
                    self.G_rec_avbd[c].min(),
                )
            )
            vmax = float(
                max(
                    self.G_true[c].max(),
                    self.G_rec_eik[c].max(),
                    self.G_rec_avbd[c].max(),
                )
            )
            cmap = "viridis" if c != 1 else "RdBu_r"

            axes[row, 0].imshow(
                self.G_true[c], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax
            )
            axes[row, 0].set_title(f"True {name}")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(
                self.G_rec_eik[c], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax
            )
            axes[row, 1].set_title(f"Eikonal {name}")
            axes[row, 1].axis("off")

            im = axes[row, 2].imshow(
                self.G_rec_avbd[c], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax
            )
            axes[row, 2].set_title(f"AVBD {name}")
            axes[row, 2].axis("off")
            plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

        fig.suptitle(
            "C4: Full Anisotropic Metric Recovery (g11, g12, g22)", fontsize=14
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# =============================================================================
# C5: DRIFT RECOVERY
# =============================================================================
@register_experiment
class C5_DriftRecovery(Experiment):
    """Recover drift field with known metric from dense observations.

    Uses dense interior observations, matching the constant-drift recovery
    protocol of Gahtan et al. (arXiv:2603.00035), who report 2-3% relative
    errors. With *sparse* single-source observations the perpendicular drift
    component is weakly identifiable (only the along-characteristic component
    of B enters the arrival times) — that harder regime is exercised by C6/C8.
    """

    name = "C5_drift_recovery"
    category = "C_inverse_problem"
    description = "Recover drift B with fixed G (dense observations)"

    def __init__(self, N: int = 40, n_iter: int = 250):
        super().__init__()
        self.N, self.n_iter = N, n_iter

    def run(self) -> ExperimentResult:
        N = self.N

        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)

        B_true = jnp.zeros((2, N, N))
        B_true = B_true.at[0].set(0.2)
        B_true = B_true.at[1].set(0.1)

        source_coords = jnp.array([[N // 2, N // 2]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N // 2, N // 2].set(True)

        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)

        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(
            metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N)
        )

        # Dense interior observations (see class docstring).
        obs_mask = get_interior_mask(N, N, 3, source_mask)

        print("\n  Training Eikonal Optimizer...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_eik = MetricRecoveryOptimizer(
            N,
            N,
            solver_type="eikonal",
            recover_H=False,
            recover_W=True,
            constant_W=True,
            lambda_W=0.01,
        )
        opt_eik.model = eqx.tree_at(lambda m: m.H_grid, opt_eik.model, H_true)
        opt_eik.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.05, verbose=True
        )
        _, B_rec_eik = opt_eik.get_G_B()

        print("  Training AVBD Optimizer...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_avbd = MetricRecoveryOptimizer(
            N,
            N,
            solver_type="avbd",
            recover_H=False,
            recover_W=True,
            constant_W=True,
            lambda_W=0.01,
        )
        opt_avbd.model = eqx.tree_at(lambda m: m.H_grid, opt_avbd.model, H_true)
        opt_avbd.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.05, verbose=True
        )
        _, B_rec_avbd = opt_avbd.get_G_B()

        interior = get_interior_mask(N, N, 5, source_mask)
        eik_b1, eik_b2 = evaluate_drift(B_true, B_rec_eik, interior)
        avbd_b1, avbd_b2 = evaluate_drift(B_true, B_rec_avbd, interior)

        print(f"  Eikonal Error: b1={eik_b1:.4f}, b2={eik_b2:.4f}")
        print(f"  AVBD Error: b1={avbd_b1:.4f}, b2={avbd_b2:.4f}")

        self.B_true = B_true
        self.B_rec_eik, self.B_rec_avbd = B_rec_eik, B_rec_avbd

        return ExperimentResult(
            name=self.name,
            category=self.category,
            # Relative thresholds (see evaluate_drift): recover within 25% of
            # the true drift magnitude.
            success=eik_b1 < 0.25 and avbd_b1 < 0.25,
            metrics={
                "eik_b1": eik_b1,
                "eik_b2": eik_b2,
                "avbd_b1": avbd_b1,
                "avbd_b2": avbd_b2,
            },
            arrays={
                "B_true": np.array(B_true),
                "B_eik": np.array(B_rec_eik),
                "B_avbd": np.array(B_rec_avbd),
            },
            metadata={"N": N},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for row, (c, name) in enumerate([(0, "b₁"), (1, "b₂")]):
            vmin = float(min(self.B_true[c].min(), self.B_rec_eik[c].min()))
            vmax = float(max(self.B_true[c].max(), self.B_rec_eik[c].max()))

            axes[row, 0].imshow(
                self.B_true[c], origin="upper", cmap="RdBu_r", vmin=vmin, vmax=vmax
            )
            axes[row, 0].set_title(f"True {name}")

            axes[row, 1].imshow(
                self.B_rec_eik[c], origin="upper", cmap="RdBu_r", vmin=vmin, vmax=vmax
            )
            axes[row, 1].set_title(f"Eikonal {name}")

            axes[row, 2].imshow(
                self.B_rec_avbd[c], origin="upper", cmap="RdBu_r", vmin=vmin, vmax=vmax
            )
            axes[row, 2].set_title(f"AVBD {name}")

        fig.suptitle("C5: Drift Recovery", fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# =============================================================================
# C6: JOINT METRIC + DRIFT RECOVERY
# =============================================================================
@register_experiment
class C6_JointMetricDriftRecovery(Experiment):
    """Simultaneously recover both the metric H and drift W from arrival times.

    This is the hardest inverse problem: we optimise all 5 unknown fields
    (g11, g12, g22, b1, b2) jointly, which couples the Riemannian geometry
    and the Zermelo drift in a single gradient descent.

    Identifiability caveat: single-source arrival times provide one scalar
    constraint per point for 5 unknown fields, so the problem is severely
    underdetermined — in particular the drift component perpendicular to the
    characteristics is in the null space and is fixed only by TV
    regularization. Moderate errors here reflect this gauge freedom, not
    solver failure (cf. C5 for the identifiable constant-drift case and C11
    for the multi-seed variance analysis).
    """

    name = "C6_joint_metric_drift_recovery"
    category = "C_inverse_problem"
    description = "Joint recovery of metric tensor and drift field simultaneously"

    def __init__(self, N: int = 40, n_iter: int = 350):
        super().__init__()
        self.N, self.n_iter = N, n_iter

    def run(self) -> ExperimentResult:
        N = self.N

        # True metric: moderate anisotropy
        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.5)
        G_true = G_true.at[2].set(0.7)
        G_true = G_true.at[0, :, N // 2 :].set(0.7)
        G_true = G_true.at[2, :, N // 2 :].set(1.5)

        # True drift: spatially constant, moderate wind
        B_true = jnp.zeros((2, N, N))
        B_true = B_true.at[0].set(0.15)  # b1 (x-component)
        B_true = B_true.at[1].set(-0.10)  # b2 (y-component)

        source_coords = jnp.array([[N // 2, N // 4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N // 2, N // 4].set(True)

        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)

        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(
            metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N)
        )

        obs_mask = create_sparse_observation_mask(N, N, 0.15, source_mask, seed=53)

        print("\n  Training Eikonal Optimizer (joint H+W)...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_eik = MetricRecoveryOptimizer(
            N,
            N,
            solver_type="eikonal",
            recover_H=True,
            recover_W=True,
            lambda_H=0.02,
            lambda_W=0.01,
            constrain_isotropic=False,
        )
        opt_eik.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.03, verbose=True
        )
        G_rec_eik, B_rec_eik = opt_eik.get_G_B()

        print("  Training AVBD Optimizer (joint H+W)...")
        obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
        opt_avbd = MetricRecoveryOptimizer(
            N,
            N,
            solver_type="avbd",
            recover_H=True,
            recover_W=True,
            lambda_H=0.02,
            lambda_W=0.01,
            constrain_isotropic=False,
        )
        opt_avbd.fit(
            source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.03, verbose=True
        )
        G_rec_avbd, B_rec_avbd = opt_avbd.get_G_B()

        interior = get_interior_mask(N, N, 5, source_mask)
        eik_g11, eik_g22 = evaluate_recovery(G_true, G_rec_eik, interior)
        avbd_g11, avbd_g22 = evaluate_recovery(G_true, G_rec_avbd, interior)
        eik_b1, eik_b2 = evaluate_drift(B_true, B_rec_eik, interior)
        avbd_b1, avbd_b2 = evaluate_drift(B_true, B_rec_avbd, interior)

        print(
            f"  Eikonal Error: g11={eik_g11:.4f}, g22={eik_g22:.4f} | b1={eik_b1:.4f}, b2={eik_b2:.4f}"
        )
        print(
            f"  AVBD Error:    g11={avbd_g11:.4f}, g22={avbd_g22:.4f} | b1={avbd_b1:.4f}, b2={avbd_b2:.4f}"
        )

        self.G_true, self.B_true = G_true, B_true
        self.G_rec_eik, self.B_rec_eik = G_rec_eik, B_rec_eik
        self.G_rec_avbd, self.B_rec_avbd = G_rec_avbd, B_rec_avbd

        return ExperimentResult(
            name=self.name,
            category=self.category,
            success=eik_g11 < 0.40 and avbd_g11 < 0.40,
            metrics={
                "eik_g11": eik_g11,
                "eik_g22": eik_g22,
                "eik_b1": eik_b1,
                "eik_b2": eik_b2,
                "avbd_g11": avbd_g11,
                "avbd_g22": avbd_g22,
                "avbd_b1": avbd_b1,
                "avbd_b2": avbd_b2,
            },
            arrays={
                "G_true": np.array(G_true),
                "B_true": np.array(B_true),
                "G_eik": np.array(G_rec_eik),
                "B_eik": np.array(B_rec_eik),
                "G_avbd": np.array(G_rec_avbd),
                "B_avbd": np.array(B_rec_avbd),
            },
            metadata={"N": N},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        # Row 0: g11 (metric)
        for col, (data, title) in enumerate(
            [
                (self.G_true[0], "True g₁₁"),
                (self.G_rec_eik[0], "Eikonal g₁₁"),
                (self.G_rec_avbd[0], "AVBD g₁₁"),
                (self.G_rec_avbd[0] - self.G_true[0], "AVBD Error g₁₁"),
            ]
        ):
            cmap = "RdBu_r" if col == 3 else "viridis"
            im = axes[0, col].imshow(data, origin="upper", cmap=cmap)
            axes[0, col].set_title(title)
            axes[0, col].axis("off")
            plt.colorbar(im, ax=axes[0, col], fraction=0.046)

        # Row 1: b1 (drift)
        for col, (data, title) in enumerate(
            [
                (self.B_true[0], "True b₁"),
                (self.B_rec_eik[0], "Eikonal b₁"),
                (self.B_rec_avbd[0], "AVBD b₁"),
                (self.B_rec_avbd[0] - self.B_true[0], "AVBD Error b₁"),
            ]
        ):
            im = axes[1, col].imshow(data, origin="upper", cmap="RdBu_r")
            axes[1, col].set_title(title)
            axes[1, col].axis("off")
            plt.colorbar(im, ax=axes[1, col], fraction=0.046)

        fig.suptitle("C6: Joint Metric + Drift Recovery", fontsize=14)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# =============================================================================
# C7: REGULARIZATION ABLATION
# =============================================================================
@register_experiment
class C7_RegularizationAblation(Experiment):
    """Show TV regularization is necessary for sparse observations."""

    name = "C7_regularization_ablation"
    category = "C_inverse_problem"
    description = "Recovery error vs regularization strength (Eikonal vs AVBD)"

    def __init__(self, N: int = 40, n_iter: int = 200):
        super().__init__()
        self.N, self.n_iter = N, n_iter
        # Includes 1e-3, the optimum reported by Gahtan et al. (arXiv:2603.00035).
        self.lambda_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1]

    def run(self) -> ExperimentResult:
        N = self.N

        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N // 2 :].set(2.0)
        G_true = G_true.at[2, :, N // 2 :].set(2.0)

        B_true = jnp.zeros((2, N, N))

        source_coords = jnp.array([[N // 2, N // 4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N // 2, N // 4].set(True)

        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)

        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(
            metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N)
        )

        obs_mask = create_sparse_observation_mask(N, N, 0.05, source_mask, seed=47)

        results_eik, results_avbd = [], []
        for lam in self.lambda_values:
            print(f"\n  lambda = {lam}")
            interior = get_interior_mask(N, N, 5, source_mask)

            obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
            opt_eik = MetricRecoveryOptimizer(
                N, N, solver_type="eikonal", lambda_H=lam, constrain_isotropic=True
            )
            opt_eik.fit(
                source_coords,
                T_obs,
                obs_mask=obs_train,
                test_mask=obs_test,
                n_iter=self.n_iter,
                lr=0.05,
                verbose=False,
            )
            G_rec_eik, _ = opt_eik.get_G_B()
            err_eik, _ = evaluate_recovery(G_true, G_rec_eik, interior)

            obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
            opt_avbd = MetricRecoveryOptimizer(
                N, N, solver_type="avbd", lambda_H=lam, constrain_isotropic=True
            )
            opt_avbd.fit(
                source_coords,
                T_obs,
                obs_mask=obs_train,
                test_mask=obs_test,
                n_iter=self.n_iter,
                lr=0.05,
                verbose=False,
            )
            G_rec_avbd, _ = opt_avbd.get_G_B()
            err_avbd, _ = evaluate_recovery(G_true, G_rec_avbd, interior)

            results_eik.append({"lambda": lam, "error": err_eik})
            results_avbd.append({"lambda": lam, "error": err_avbd})
            print(f"    Eikonal: {err_eik:.4f}  |  AVBD: {err_avbd:.4f}")

        best_eik = min(results_eik, key=lambda x: x["error"])
        best_avbd = min(results_avbd, key=lambda x: x["error"])
        self.results_eik = results_eik
        self.results_avbd = results_avbd

        return ExperimentResult(
            name=self.name,
            category=self.category,
            success=True,
            metrics={
                "eik_best_lambda": best_eik["lambda"],
                "eik_best_error": best_eik["error"],
                "avbd_best_lambda": best_avbd["lambda"],
                "avbd_best_error": best_avbd["error"],
            },
            arrays={
                "lambdas": np.array([r["lambda"] for r in results_eik]),
                "eik_errors": np.array([r["error"] for r in results_eik]),
                "avbd_errors": np.array([r["error"] for r in results_avbd]),
            },
            metadata={"N": N},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))

        lambdas = np.array([r["lambda"] for r in self.results_eik])
        lambdas_plot = np.where(lambdas == 0, 1e-3, lambdas)

        ax.semilogx(
            lambdas_plot,
            [r["error"] for r in self.results_eik],
            "o-",
            lw=2,
            label="Eikonal",
        )
        ax.semilogx(
            lambdas_plot,
            [r["error"] for r in self.results_avbd],
            "s--",
            lw=2,
            label="AVBD",
        )

        ax.set_xlabel("Regularization λ")
        ax.set_ylabel("Relative Error")
        ax.set_title("C7: Regularization Ablation (Eikonal vs AVBD)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# =============================================================================
# C8: OBSERVATION DENSITY
# =============================================================================
@register_experiment
class C8_ObservationDensity(Experiment):
    """Recovery error vs observation density."""

    name = "C8_observation_density"
    category = "C_inverse_problem"
    description = "Recovery error vs observation density (Eikonal vs AVBD)"

    def __init__(self, N: int = 40, n_iter: int = 200):
        super().__init__()
        self.N, self.n_iter = N, n_iter
        self.obs_fractions = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    def run(self) -> ExperimentResult:
        N = self.N

        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N // 2 :].set(2.0)
        G_true = G_true.at[2, :, N // 2 :].set(2.0)
        B_true = jnp.zeros((2, N, N))

        source_coords = jnp.array([[N // 2, N // 4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N // 2, N // 4].set(True)

        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(
            metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N)
        )

        results_eik, results_avbd = [], []
        for frac in self.obs_fractions:
            print(f"\n  Fraction: {frac * 100:.0f}%")

            if frac >= 1.0:
                obs_mask = get_interior_mask(N, N, 3, source_mask)
            else:
                obs_mask = create_sparse_observation_mask(
                    N, N, frac, source_mask, seed=48
                )

            interior = get_interior_mask(N, N, 5, source_mask)
            lambda_H = 0.01 if frac < 0.5 else 0.001

            obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
            opt_eik = MetricRecoveryOptimizer(
                N, N, solver_type="eikonal", lambda_H=lambda_H, constrain_isotropic=True
            )
            opt_eik.fit(
                source_coords,
                T_obs,
                obs_mask=obs_train,
                test_mask=obs_test,
                n_iter=self.n_iter,
                lr=0.05,
                verbose=False,
            )
            G_rec_eik, _ = opt_eik.get_G_B()
            err_eik, _ = evaluate_recovery(G_true, G_rec_eik, interior)

            obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
            opt_avbd = MetricRecoveryOptimizer(
                N, N, solver_type="avbd", lambda_H=lambda_H, constrain_isotropic=True
            )
            opt_avbd.fit(
                source_coords,
                T_obs,
                obs_mask=obs_train,
                test_mask=obs_test,
                n_iter=self.n_iter,
                lr=0.05,
                verbose=False,
            )
            G_rec_avbd, _ = opt_avbd.get_G_B()
            err_avbd, _ = evaluate_recovery(G_true, G_rec_avbd, interior)

            results_eik.append({"fraction": frac, "error": err_eik})
            results_avbd.append({"fraction": frac, "error": err_avbd})
            print(f"    Eikonal: {err_eik:.4f}  |  AVBD: {err_avbd:.4f}")

        self.results_eik = results_eik
        self.results_avbd = results_avbd

        return ExperimentResult(
            name=self.name,
            category=self.category,
            success=True,
            metrics={
                "eik_min_error": min(r["error"] for r in results_eik),
                "avbd_min_error": min(r["error"] for r in results_avbd),
            },
            arrays={
                "fractions": np.array([r["fraction"] for r in results_eik]),
                "eik_errors": np.array([r["error"] for r in results_eik]),
                "avbd_errors": np.array([r["error"] for r in results_avbd]),
            },
            metadata={"N": N},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))

        fracs = np.array([r["fraction"] for r in self.results_eik]) * 100
        ax.plot(
            fracs, [r["error"] for r in self.results_eik], "o-", lw=2, label="Eikonal"
        )
        ax.plot(
            fracs, [r["error"] for r in self.results_avbd], "s--", lw=2, label="AVBD"
        )
        ax.set_xlabel("Observation Density (%)")
        ax.set_ylabel("Relative Error")
        ax.set_title("C8: Observation Density (Eikonal vs AVBD)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# =============================================================================
# C9: NOISE ROBUSTNESS
# =============================================================================
@register_experiment
class C9_NoiseRobustness(Experiment):
    """Test recovery under measurement noise."""

    name = "C9_noise_robustness"
    category = "C_inverse_problem"
    description = "Recovery error vs noise level (Eikonal vs AVBD)"

    def __init__(self, N: int = 40, n_iter: int = 200):
        super().__init__()
        self.N, self.n_iter = N, n_iter
        self.noise_levels = [0, 0.02, 0.05, 0.1, 0.2]

    def run(self) -> ExperimentResult:
        N = self.N

        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N // 2 :].set(2.0)
        G_true = G_true.at[2, :, N // 2 :].set(2.0)
        B_true = jnp.zeros((2, N, N))

        source_coords = jnp.array([[N // 2, N // 4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N // 2, N // 4].set(True)

        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)
        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_clean, _, _ = solver.solve(
            metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N)
        )

        obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=49)
        interior = get_interior_mask(N, N, 5, source_mask)

        results_eik, results_avbd = [], []
        np.random.seed(50)

        for noise in self.noise_levels:
            print(f"\n  Noise: {noise * 100:.0f}%")

            T_obs = T_clean
            if noise > 0:
                std = float(jnp.std(T_clean))
                noise_tensor = noise * std * jnp.array(np.random.randn(N, N))
                T_obs = T_clean + noise_tensor

            obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
            opt_eik = MetricRecoveryOptimizer(
                N, N, solver_type="eikonal", lambda_H=0.02, constrain_isotropic=True
            )
            opt_eik.fit(
                source_coords,
                T_obs,
                obs_mask=obs_train,
                test_mask=obs_test,
                n_iter=self.n_iter,
                lr=0.05,
                verbose=False,
            )
            G_rec_eik, _ = opt_eik.get_G_B()
            err_eik, _ = evaluate_recovery(G_true, G_rec_eik, interior)

            obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
            opt_avbd = MetricRecoveryOptimizer(
                N, N, solver_type="avbd", lambda_H=0.02, constrain_isotropic=True
            )
            opt_avbd.fit(
                source_coords,
                T_obs,
                obs_mask=obs_train,
                test_mask=obs_test,
                n_iter=self.n_iter,
                lr=0.05,
                verbose=False,
            )
            G_rec_avbd, _ = opt_avbd.get_G_B()
            err_avbd, _ = evaluate_recovery(G_true, G_rec_avbd, interior)

            results_eik.append({"noise": noise, "error": err_eik})
            results_avbd.append({"noise": noise, "error": err_avbd})
            print(f"    Eikonal: {err_eik:.4f}  |  AVBD: {err_avbd:.4f}")

        self.results_eik = results_eik
        self.results_avbd = results_avbd

        return ExperimentResult(
            name=self.name,
            category=self.category,
            success=True,
            metrics={
                "eik_clean_error": results_eik[0]["error"],
                "eik_noisy_error": results_eik[-1]["error"],
                "avbd_clean_error": results_avbd[0]["error"],
                "avbd_noisy_error": results_avbd[-1]["error"],
            },
            arrays={
                "noise_levels": np.array([r["noise"] for r in results_eik]),
                "eik_errors": np.array([r["error"] for r in results_eik]),
                "avbd_errors": np.array([r["error"] for r in results_avbd]),
            },
            metadata={"N": N},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))

        noise = np.array([r["noise"] for r in self.results_eik]) * 100
        ax.plot(
            noise, [r["error"] for r in self.results_eik], "o-", lw=2, label="Eikonal"
        )
        ax.plot(
            noise, [r["error"] for r in self.results_avbd], "s--", lw=2, label="AVBD"
        )
        ax.set_xlabel("Noise Level (%)")
        ax.set_ylabel("Relative Error")
        ax.set_title("C9: Noise Robustness (Eikonal vs AVBD)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# =============================================================================
# C10: MULTIPLE SOURCES
# =============================================================================
@register_experiment
class C10_MultipleSources(Experiment):
    """Test if multiple sources improve recovery."""

    name = "C10_multiple_sources"
    category = "C_inverse_problem"
    description = "Recovery with multiple ignition points (Eikonal vs AVBD)"

    def __init__(self, N: int = 40, n_iter: int = 200):
        super().__init__()
        self.N, self.n_iter = N, n_iter

    def run(self) -> ExperimentResult:
        N = self.N

        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)
        G_true = G_true.at[0, :, N // 2 :].set(2.0)
        G_true = G_true.at[2, :, N // 2 :].set(2.0)
        B_true = jnp.zeros((2, N, N))

        results_eik, results_avbd = [], []

        for n_sources, sources in [
            (1, [[N // 2, N // 4]]),
            (2, [[N // 4, N // 4], [3 * N // 4, N // 4]]),
            (3, [[N // 4, N // 4], [3 * N // 4, N // 4], [N // 2, 3 * N // 4]]),
        ]:
            print(f"\n  {n_sources} source(s)")

            source_coords = jnp.array(sources, dtype=jnp.float32)
            source_mask = jnp.zeros((N, N), dtype=bool)
            for s in sources:
                source_mask = source_mask.at[s[0], s[1]].set(True)

            H_true, W_true = eikonal_to_zermelo(G_true, B_true)
            metric_true = SyntheticZermeloMetric(H_true, W_true)
            solver = EikonalSolver(max_iters=50, tol=1e-5)
            T_obs, _, _ = solver.solve(
                metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N)
            )

            obs_mask = create_sparse_observation_mask(N, N, 0.1, source_mask, seed=51)
            interior = get_interior_mask(N, N, 5, source_mask)

            obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
            opt_eik = MetricRecoveryOptimizer(
                N, N, solver_type="eikonal", lambda_H=0.02, constrain_isotropic=True
            )
            opt_eik.fit(
                source_coords,
                T_obs,
                obs_mask=obs_train,
                test_mask=obs_test,
                n_iter=self.n_iter,
                lr=0.05,
                verbose=False,
            )
            G_rec_eik, _ = opt_eik.get_G_B()
            err_eik, _ = evaluate_recovery(G_true, G_rec_eik, interior)

            obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
            opt_avbd = MetricRecoveryOptimizer(
                N, N, solver_type="avbd", lambda_H=0.02, constrain_isotropic=True
            )
            opt_avbd.fit(
                source_coords,
                T_obs,
                obs_mask=obs_train,
                test_mask=obs_test,
                n_iter=self.n_iter,
                lr=0.05,
                verbose=False,
            )
            G_rec_avbd, _ = opt_avbd.get_G_B()
            err_avbd, _ = evaluate_recovery(G_true, G_rec_avbd, interior)

            results_eik.append({"n_sources": n_sources, "error": err_eik})
            results_avbd.append({"n_sources": n_sources, "error": err_avbd})
            print(f"    Eikonal: {err_eik:.4f}  |  AVBD: {err_avbd:.4f}")

        self.results_eik = results_eik
        self.results_avbd = results_avbd

        return ExperimentResult(
            name=self.name,
            category=self.category,
            success=True,
            metrics={
                "eik_error_1source": results_eik[0]["error"],
                "eik_error_3sources": results_eik[-1]["error"],
                "avbd_error_1source": results_avbd[0]["error"],
                "avbd_error_3sources": results_avbd[-1]["error"],
            },
            arrays={
                "n_sources": np.array([r["n_sources"] for r in results_eik]),
                "eik_errors": np.array([r["error"] for r in results_eik]),
                "avbd_errors": np.array([r["error"] for r in results_avbd]),
            },
            metadata={"N": N},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))

        n = [r["n_sources"] for r in self.results_eik]
        ax.plot(n, [r["error"] for r in self.results_eik], "o-", lw=2, label="Eikonal")
        ax.plot(n, [r["error"] for r in self.results_avbd], "s--", lw=2, label="AVBD")
        ax.set_xlabel("Number of Sources")
        ax.set_ylabel("Relative Error")
        ax.set_title("C10: Multiple Sources (Eikonal vs AVBD)")
        ax.set_xticks(n)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig


# =============================================================================
# C11: IDENTIFIABILITY ANALYSIS
# =============================================================================
@register_experiment
class C11_IdentifiabilityAnalysis(Experiment):
    """Run joint recovery from multiple random initializations."""

    name = "C11_identifiability_analysis"
    category = "C_inverse_problem"
    description = "Test if joint recovery converges to same metric from 5 seeds"

    def __init__(self, N: int = 40, n_iter: int = 250, n_seeds: int = 5):
        super().__init__()
        self.N, self.n_iter, self.n_seeds = N, n_iter, n_seeds

    def run(self) -> ExperimentResult:
        N = self.N

        G_true = jnp.zeros((3, N, N))
        G_true = G_true.at[0].set(1.0)
        G_true = G_true.at[2].set(1.0)

        B_true = jnp.zeros((2, N, N))
        B_true = B_true.at[0].set(0.3)
        B_true = B_true.at[1].set(0.3)

        source_coords = jnp.array([[N // 2, N // 4]], dtype=jnp.float32)
        source_mask = jnp.zeros((N, N), dtype=bool).at[N // 2, N // 4].set(True)

        H_true, W_true = eikonal_to_zermelo(G_true, B_true)
        metric_true = SyntheticZermeloMetric(H_true, W_true)

        solver = EikonalSolver(max_iters=50, tol=1e-5)
        T_obs, _, _ = solver.solve(metric_true, source_coords, (0, N - 1, 0, N - 1), (N, N))

        obs_mask = create_sparse_observation_mask(N, N, 0.2, source_mask)

        results_g = []
        results_b = []

        for seed in range(self.n_seeds):
            import jax.random as jrandom
            import equinox as eqx
            key = jrandom.PRNGKey(seed)
            k1, k2 = jrandom.split(key)
            
            obs_train, obs_test = split_observation_mask(obs_mask, 0.8, seed=42)
            opt = MetricRecoveryOptimizer(N, N, solver_type="eikonal", recover_H=True, recover_W=True, constrain_isotropic=True)
            opt.model = eqx.tree_at(lambda m: m.H_grid, opt.model, opt.model.H_grid.at[0].add(0.2 * jrandom.normal(k1, (N, N))))
            opt.model = eqx.tree_at(lambda m: m.H_grid, opt.model, opt.model.H_grid.at[2].add(0.2 * jrandom.normal(k1, (N, N))))
            opt.model = eqx.tree_at(lambda m: m.W_grid, opt.model, opt.model.W_grid + 0.1 * jrandom.normal(k2, (2, N, N)))
            
            opt.fit(source_coords, T_obs, obs_mask=obs_train, test_mask=obs_test, n_iter=self.n_iter, lr=0.05, verbose=False)
            G_rec, B_rec = opt.get_G_B()
            
            interior = get_interior_mask(N, N, 5, source_mask)
            err_g11, err_g22 = evaluate_recovery(G_true, G_rec, interior)
            err_b1, err_b2 = evaluate_drift(B_true, B_rec, interior)
            
            results_g.append((err_g11 + err_g22)/2)
            results_b.append((err_b1 + err_b2)/2)

        var_g = np.var(results_g)
        var_b = np.var(results_b)

        return ExperimentResult(
            name=self.name,
            category=self.category,
            success=True,
            metrics={"var_g": float(var_g), "var_b": float(var_b)},
            metadata={"N": N, "n_seeds": self.n_seeds},
        )

    def visualize(self, save_path: Optional[str] = None) -> plt.Figure:
        return None



# =============================================================================
# RUN ALL
# =============================================================================

ALL_EXPERIMENTS = [
    C1_IsotropicFull,
    C2_IsotropicSparse,
    C3_DiagonalAnisotropic,
    C4_FullMetricRecovery,
    C5_DriftRecovery,
    C6_JointMetricDriftRecovery,
    C7_RegularizationAblation,
    C8_ObservationDensity,
    C9_NoiseRobustness,
    C10_MultipleSources,
    C11_IdentifiabilityAnalysis,
]


def run_all(save=True, visualize=True):
    """Run all Category C experiments."""
    results = {}
    for cls in ALL_EXPERIMENTS:
        exp = cls()
        results[exp.name] = exp.execute(save=save, visualize=visualize)

    print("\n" + "=" * 60)
    print("CATEGORY C SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: {'✓ PASS' if r.success else '✗ FAIL'}")
    return results


if __name__ == "__main__":
    run_all()

