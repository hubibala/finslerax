"""
Finsler Metric base class — the core geometric abstraction of HAMTools.

This module defines :class:`FinslerMetric`, from which all concrete
metrics (Euclidean, Riemannian, Randers, learned neural metrics) inherit.
The metric provides the fundamental cost function F(x, v) and its derived
energy E = ½ F². Every downstream geometric object — geodesic spray,
curvature, parallel transport — is auto-differentiated from ``metric_fn``.

Mathematical reference: spec/MATH_SPEC.md §§ 1–2.
Architecture reference: spec/ARCH_SPEC.md § 2.2.

See Also:
    ham.geometry.zoo : Concrete metric implementations.
    ham.geometry.transport : Berwald parallel transport built on this class.
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.manifold import Manifold
from ham.utils.math import PSD_EPS


class FinslerMetric(eqx.Module):
    """
    Abstract base class for all Finsler metrics.

    Defines the geometry of a manifold via the fundamental cost function $F(x, v)$
    and its derived energy $E = \\tfrac{1}{2}F^2$. All downstream geometry
    (geodesic spray, curvature, parallel transport) is auto-differentiated from
    `metric_fn`. Inherits from `eqx.Module` so subclasses are valid JAX PyTrees.

    Implementations may be vmapped externally; methods operate on single points.
    """

    manifold: Manifold
    spray_reg: float = eqx.field(static=True, default=PSD_EPS)

    @abstractmethod
    def metric_fn(self, x: jax.Array, v: jax.Array) -> jax.Array:
        r"""
        Computes the fundamental Finsler cost function F(x, v).

        The metric must be positively 1-homogeneous in v, meaning F(x, λv) = λF(x, v) for λ > 0.
        Implementations must be gradient-safe at v = 0 (e.g., using `safe_norm`).

        Args:
            x: Position vector on the manifold.
            v: Tangent vector at position x.

        Returns:
            Scalar Finsler cost $F(x, v) \geq 0$. Shape: `()`.
        """
        pass

    def energy(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Lagrangian energy E(x, v) = ½ F²(x, v).

        This scalar is the root of the computational graph: the geodesic
        spray, fundamental tensor, and inner product are all derived from
        automatic differentiation of this function.

        Args:
            x: Position on the manifold. Shape ``(D,)`` or ``(N,)``.
            v: Tangent vector at x. Shape ``(D,)`` or ``(N,)``.

        Returns:
            Scalar energy. Shape ``()``.

        Reference:
            spec/MATH_SPEC.md § 1.2.
        """
        return 0.5 * self.metric_fn(x, v) ** 2

    def inner_product(
        self, x: jax.Array, v: jax.Array, w1: jax.Array, w2: jax.Array
    ) -> jax.Array:
        """
        Finsler inner product <w1, w2>_v using the fundamental tensor g_ij(x, v).

        Computes  w1ᵀ · g(x, v) · w2  where g_ij = ∂²E/∂vⁱ∂vʲ (the Hessian
        of the energy with respect to velocity).  Note: unlike Riemannian
        geometry, the inner product depends on a *reference direction* v.

        Args:
            x: Position on the manifold. Shape ``(D,)`` or ``(N,)``.
            v: Reference tangent direction for the fundamental tensor. Shape ``(D,)`` or ``(N,)``.
            w1: First tangent vector. Shape ``(D,)`` or ``(N,)``.
            w2: Second tangent vector. Shape ``(D,)`` or ``(N,)``.

        Returns:
            Scalar inner product. Shape ``()``.

        Reference:
            spec/MATH_SPEC.md § 1.1 (fundamental tensor);
            spec/ARCH_SPEC.md § 2.2.
        """
        # Hessian of E w.r.t v, evaluated at (x, v).
        # We compute this dynamically on every call; JAX optimizes this away under JIT.
        g_fn = jax.hessian(self.energy, argnums=1)
        g_x_v = g_fn(x, v)
        return jnp.dot(w1, jnp.dot(g_x_v, w2))

    def spray(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Geodesic spray coefficients Gⁱ(x, v).

        Solves the implicit linear system derived from Euler-Lagrange:
            Hess_v(E) · (−2G) = ∇ₓE − Jacₓ(∇ᵥE) · v
        for the spray coefficients $G^i$, per `spec/MATH_SPEC.md` § 2.2.

        Args:
            x: Position on the manifold. Shape ``(D,)`` or ``(N,)``.
            v: Velocity (tangent vector). Shape ``(D,)`` or ``(N,)``.

        Returns:
            Spray vector G(x, v). Shape ``(D,)`` or ``(N,)``.

        Note:
            A Tikhonov term ε·I (ε = spray_reg) is added to the Hessian to
            regularize near-degenerate directions (Randers boundary).
            See spec/MATH_SPEC.md § 6.1.
        """
        grad_v_fn = jax.grad(self.energy, argnums=1)

        # We need grad_x(E) and Jac_x(grad_v E) * v
        # Using JVP for the mixed term is efficient
        grad_x = jax.grad(self.energy, argnums=0)(x, v)

        def d_dv_fixed_v(pos):
            return grad_v_fn(pos, v)

        _, mixed_term = jax.jvp(d_dv_fixed_v, (x,), (v,))
        rhs = grad_x - mixed_term

        hess_v = jax.hessian(self.energy, argnums=1)(x, v)

        # Trace-scaled Tikhonov regularisation: ε·(tr(H)/D)·I instead of a
        # bare ε·I.  Scaling by the mean eigenvalue keeps the relative
        # perturbation constant regardless of the metric's overall magnitude,
        # preventing over-regularisation for small metrics and under-
        # regularisation for large ones (spec/MATH_SPEC.md § 6.1).
        dim = v.shape[-1]
        reg_scale = self.spray_reg * jnp.maximum(jnp.trace(hess_v) / dim, 1.0)
        safe_hess = hess_v + reg_scale * jnp.eye(dim)
        acc = jnp.linalg.solve(safe_hess, rhs)
        return -0.5 * acc

    def geod_acceleration(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Geodesic acceleration  ẍⁱ = −2 Gⁱ(x, v).

        Returns the acceleration vector that, combined with velocity v,
        defines the geodesic ODE:  dx/dt = v,  dv/dt = −2G(x, v).

        Args:
            x: Position on the manifold. Shape ``(D,)`` or ``(N,)``.
            v: Velocity (tangent vector). Shape ``(D,)`` or ``(N,)``.

        Returns:
            Acceleration vector. Shape ``(D,)`` or ``(N,)``.

        Reference:
            spec/MATH_SPEC.md § 2.1.
        """
        return -2.0 * self.spray(x, v)

    def arc_length(self, gamma: jax.Array) -> jax.Array:
        """
        Approximate Finsler arc length of a discrete path.

        Uses the midpoint rule: each segment [γᵢ, γᵢ₊₁] is evaluated at
        the projected midpoint with velocity v = γᵢ₊₁ − γᵢ. This is a first-order
        quadrature; accuracy improves with finer discretization.

        Args:
            gamma: Waypoints of the path. Shape ``(N, D)`` where N ≥ 2.

        Returns:
            Total arc length (scalar). Shape ``()``. If N < 2, returns 0.0.

        Example:
            >>> path = jnp.linspace(start, end, num=50)
            >>> length = metric.arc_length(path)
        """
        if gamma.shape[0] < 2:
            return jnp.array(0.0)

        def segment_length(x1, x2):
            v = x2 - x1
            midpoint = self.manifold.project(0.5 * (x1 + x2))
            return self.metric_fn(midpoint, v)

        return jnp.sum(jax.vmap(segment_length)(gamma[:-1], gamma[1:]))


class AsymmetricMetric(FinslerMetric):
    """Base class for Randers-type metrics defined by a Riemannian sea and a drift.

    Subclasses must implement :meth:`zermelo_data`, which returns the triple
    ``(H, W, lambda)`` needed for the Zermelo navigation formula.  Inheriting
    from this class (rather than bare :class:`FinslerMetric`) allows consumers
    (losses, VAE) to branch on ``isinstance(metric, AsymmetricMetric)`` instead
    of the fragile ``hasattr(metric, '_get_zermelo_data')`` duck-typing pattern,
    which is not safe inside ``jax.jit`` due to Python-level conditional tracing.

    Architecture reference: spec/ARCH_SPEC.md § 2.2.
    """

    @abstractmethod
    def zermelo_data(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the Zermelo navigation triple ``(H, W, lambda)`` at position x.

        Args:
            x: Position on the manifold, shape ``(D,)``.

        Returns:
            H:      Riemannian sea metric tensor, shape ``(D, D)``.
            W:      Wind (drift) vector, shape ``(D,)``.
            lambda: Causality scalar ``1 - ||W||²_H``, shape ``()``.
        """
        pass
