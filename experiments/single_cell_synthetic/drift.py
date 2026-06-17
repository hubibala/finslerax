"""Drift reconstruction (sparseVFC) + mesh-free Helmholtz–Hodge decomposition.

The drift estimator takes the noisy, projected RNA velocities and reconstructs a
smooth analytic field ``f̂(z)`` in latent space, then splits it into its
reversible (gradient) and irreversible (solenoidal) parts — the two quantities
the metric's wind and the flux-recovery score care about (PLAN §6.2, §8).

* :class:`SparseVFC` — a Dynamo-style **Sparse Vector Field Consensus** RKHS fit
  (Ma et al. 2013): a vector-valued Gaussian-kernel field on sparse control
  points, with **EM outlier rejection** that down-weights the sign-flipped
  "false-positive arrows" in the velocity noise model.  It is an ``eqx.Module``,
  so the fitted field is a valid, differentiable ``w_net`` for the Randers metric
  and conceptually upgrades HAM's Nadaraya–Watson ``KernelWindField``.

* :func:`helmholtz_hodge_rbf` — a **mesh-free** Helmholtz–Hodge decomposition
  (RBF scalar/stream potentials fit by ridge least squares).  Avoids a discrete
  grid Poisson solve and works directly at the data points.  2-D (the ``∇^⊥``
  rotation is planar); applied in the 2-D latent where the flux is defined.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Sparse Vector Field Consensus (sparseVFC)
# =============================================================================
class SparseVFC(eqx.Module):
    """A fitted sparse RKHS vector field ``f̂(z) = Σ_i K(z, c_i) C_i``.

    Construct via :meth:`fit`.  The Gaussian kernel ``K(z,c)=exp(-‖z-c‖²/2β²)``
    and the coefficients ``C`` (one row per control point) are frozen arrays, so
    ``__call__`` is a pure differentiable JAX function usable as a Randers wind.
    """

    centers: jax.Array  # (M, d)
    coeffs: jax.Array  # (M, d)
    beta: float = eqx.field(static=True)

    def __call__(self, z: jax.Array) -> jax.Array:
        d2 = jnp.sum((self.centers - z) ** 2, axis=-1)
        k = jnp.exp(-d2 / (2.0 * self.beta**2))
        return k @ self.coeffs

    @staticmethod
    def fit(
        z: np.ndarray,
        v: np.ndarray,
        *,
        n_control: int = 100,
        beta: float | None = None,
        reg: float = 1e-3,
        max_iter: int = 30,
        outlier_ratio: float = 0.9,
        seed: int = 0,
    ) -> SparseVFC:
        """Fit a sparseVFC field to noisy velocity samples ``(z, v)``.

        Args:
            z: Positions, shape ``(N, d)``.
            v: Velocities, shape ``(N, d)``.
            n_control: Number of sparse control points (random subset of ``z``).
            beta: Gaussian kernel bandwidth; defaults to the median pairwise
                distance of the control points (a robust scale heuristic).
            reg: RKHS ridge regularisation ``λ``.
            max_iter: EM iterations.
            outlier_ratio: Initial inlier prior ``γ`` (the EM re-estimates it).
            seed: RNG seed for control-point subsampling.
        """
        z = np.asarray(z, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        n, d = z.shape
        rng = np.random.default_rng(seed)
        m = min(n_control, n)
        ctrl_idx = rng.choice(n, size=m, replace=False)
        centers = z[ctrl_idx]

        if beta is None:
            # median heuristic on control points
            diffs = centers[:, None, :] - centers[None, :, :]
            dists = np.sqrt((diffs**2).sum(-1))
            beta = float(np.median(dists[dists > 0])) or 1.0

        def kernel(a, b):
            d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
            return np.exp(-d2 / (2.0 * beta**2))

        K = kernel(z, centers)  # (N, M)
        U = kernel(centers, centers)  # (M, M)

        C = np.zeros((m, d))
        sigma2 = np.mean(np.var(v, axis=0)) + 1e-8
        gamma = outlier_ratio
        # uniform outlier density over the data bounding box volume
        vol = np.prod(v.max(0) - v.min(0) + 1e-6)
        a_unif = 1.0 / max(vol, 1e-6)

        for _ in range(max_iter):
            pred = K @ C  # (N, d)
            resid = np.sum((v - pred) ** 2, axis=1)  # (N,)
            # E-step: posterior inlier probability
            inlier = np.exp(-resid / (2.0 * sigma2))
            outlier = (2.0 * np.pi * sigma2) ** (d / 2.0) * (1.0 - gamma) / max(gamma, 1e-8) * a_unif
            p = inlier / (inlier + outlier + 1e-300)
            # M-step: weighted RKHS ridge  (KᵀPK + λσ²U) C = KᵀPV
            P = p[:, None]
            A = K.T @ (P * K) + reg * sigma2 * U
            rhs = K.T @ (P * v)
            C = np.linalg.solve(A + 1e-8 * np.eye(m), rhs)
            pred = K @ C
            resid = np.sum((v - pred) ** 2, axis=1)
            sp = p.sum()
            sigma2 = float((p * resid).sum() / (d * sp + 1e-8)) + 1e-8
            gamma = float(np.clip(sp / n, 1e-3, 1 - 1e-3))

        return SparseVFC(
            centers=jnp.asarray(centers, dtype=jnp.float32),
            coeffs=jnp.asarray(C, dtype=jnp.float32),
            beta=float(beta),
        )


# =============================================================================
# Mesh-free Helmholtz–Hodge decomposition (2-D)
# =============================================================================
class HodgeField(eqx.Module):
    """Decomposed field callables produced by :func:`helmholtz_hodge_rbf`.

    Exposes ``grad_part(z)`` (curl-free / irrotational), ``sol_part(z)``
    (divergence-free / solenoidal) and ``full(z) = grad_part + sol_part``.  Built
    from **matrix-valued** curl-free and divergence-free Gaussian RBF kernels
    (Fuselier–Wright), whose column spaces are the orthogonal Hodge subspaces, so
    the split is well-posed (unlike a primal ``∇Φ + ∇^⊥Ψ`` least-squares fit).
    For a Gaussian ``φ(r)=exp(-‖r‖²/2β²)`` with ``r = z - c``:

        K_cf = (I/β² - r rᵀ/β⁴) φ                      (curl-free)
        K_df = (r rᵀ/β⁴ + (1/β² - ‖r‖²/β⁴) I) φ        (divergence-free)
    """

    centers: jax.Array  # (M, 2)
    a: jax.Array  # (M, 2) curl-free (gradient) coeffs
    b: jax.Array  # (M, 2) divergence-free (solenoidal) coeffs
    beta: float = eqx.field(static=True)

    def _matrices(self, z):
        r = z - self.centers  # (M, 2)
        b2 = self.beta**2
        b4 = self.beta**4
        s = jnp.sum(r**2, axis=-1)  # (M,)
        phi = jnp.exp(-s / (2.0 * b2))  # (M,)
        rrt = r[:, :, None] * r[:, None, :]  # (M, 2, 2)
        eye = jnp.eye(2)
        k_cf = (eye / b2 - rrt / b4) * phi[:, None, None]
        k_df = (rrt / b4 + ((1.0 / b2 - s / b4)[:, None, None]) * eye) * phi[:, None, None]
        return k_cf, k_df

    def grad_part(self, z: jax.Array) -> jax.Array:
        k_cf, _ = self._matrices(z)
        return jnp.einsum("mij,mj->i", k_cf, self.a)

    def sol_part(self, z: jax.Array) -> jax.Array:
        _, k_df = self._matrices(z)
        return jnp.einsum("mij,mj->i", k_df, self.b)

    def full(self, z: jax.Array) -> jax.Array:
        return self.grad_part(z) + self.sol_part(z)

    def __call__(self, z: jax.Array) -> jax.Array:
        return self.full(z)


def helmholtz_hodge_rbf(
    field_fn,
    z: np.ndarray,
    *,
    n_control: int = 80,
    beta: float | None = None,
    reg: float = 1e-3,
    seed: int = 0,
) -> HodgeField:
    """Mesh-free Helmholtz–Hodge split of a 2-D vector field at samples ``z``.

    Uses **matrix-valued divergence-free / curl-free RBF kernels**
    (Fuselier–Wright): the field is fit as ``f ≈ Σ K_cf(·,cᵢ) aᵢ + Σ K_df(·,cᵢ) bᵢ``
    by ridge least squares, where every ``K_cf`` column is exactly curl-free and
    every ``K_df`` column is exactly divergence-free.  Because the two kernel
    spans are the orthogonal Hodge subspaces, the gradient (reversible) and
    solenoidal (irreversible flux) parts are identified uniquely — the well-posed
    cure for the ill-conditioned primal ``∇Φ + ∇^⊥Ψ`` fit and the unstable
    inverse-Laplacian (Poisson) fit (PLAN §6.2 step 2, §12).  Any spatially
    *constant* (harmonic) component is absorbed into the curl-free part.

    Args:
        field_fn: Differentiable callable ``z -> ℝ²`` (e.g. a fitted ``SparseVFC``).
        z: Sample positions, shape ``(N, 2)``.
        n_control: RBF control points (random subset of ``z``).
        beta: Kernel bandwidth (median heuristic if ``None``).
        reg: Ridge regularisation.
        seed: RNG seed.

    Returns:
        A :class:`HodgeField` with ``grad_part`` / ``sol_part`` / ``full``.
    """
    z = np.asarray(z, dtype=np.float64)
    assert z.shape[1] == 2, "Helmholtz-Hodge RBF is 2-D"
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    m = min(n_control, n)
    centers = z[rng.choice(n, size=m, replace=False)]

    if beta is None:
        diffs = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt((diffs**2).sum(-1))
        beta = float(np.median(dists[dists > 0])) or 1.0

    u = np.asarray(jax.vmap(field_fn)(jnp.asarray(z, jnp.float32)), dtype=np.float64)  # (N,2)

    b2, b4 = beta**2, beta**4
    r = z[:, None, :] - centers[None, :, :]  # (N, M, 2)
    s = (r**2).sum(-1)  # (N, M)
    phi = np.exp(-s / (2.0 * b2))  # (N, M)
    rrt = r[:, :, :, None] * r[:, :, None, :]  # (N, M, 2, 2)
    eye = np.eye(2)
    k_cf = (eye / b2 - rrt / b4) * phi[:, :, None, None]  # (N, M, 2, 2)
    k_df = (rrt / b4 + (1.0 / b2 - s / b4)[:, :, None, None] * eye) * phi[:, :, None, None]

    # Project the field *separately* onto each Hodge subspace (best approximation
    # in its own kernel span).  A joint fit lets the two finite-sampled bases
    # trade off — assigning huge cancelling coefficients that reconstruct the
    # field but inflate each part (spurious solenoidal energy even at κ=0).
    # Separate ridge projections avoid that; the (small) harmonic component of
    # the field is simply left out of both parts, as Hodge theory prescribes.
    A_cf = np.transpose(k_cf, (0, 2, 1, 3)).reshape(n * 2, m * 2)
    A_df = np.transpose(k_df, (0, 2, 1, 3)).reshape(n * 2, m * 2)
    rhs = u.reshape(n * 2)

    # The spatially-constant component of the field is harmonic (both curl- and
    # divergence-free); a developmental landscape's dominant constant is the
    # forward drift ``(c,0) = -∇(-c·x₁)``, which is conservative.  Remove the
    # global mean from the divergence-free target so this harmonic constant is
    # attributed to the gradient part (where it belongs) rather than leaking
    # spurious solenoidal energy.
    u_mean = u.mean(axis=0)
    rhs_df = (u - u_mean).reshape(n * 2)
    a = np.linalg.solve(A_cf.T @ A_cf + reg * np.eye(2 * m), A_cf.T @ rhs).reshape(m, 2)
    b = np.linalg.solve(A_df.T @ A_df + reg * np.eye(2 * m), A_df.T @ rhs_df).reshape(m, 2)

    return HodgeField(
        centers=jnp.asarray(centers, jnp.float32),
        a=jnp.asarray(a, jnp.float32),
        b=jnp.asarray(b, jnp.float32),
        beta=float(beta),
    )
