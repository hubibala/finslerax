"""
Parallel translation along curves on Finsler manifolds.

The geodesic spray ``G^i(x, y)`` induces the coefficients ``G^i_j = ∂G^i/∂y^j``
of the Berwald connection. They span the horizontal distribution, and their
horizontal curves are the parallel translation of the Finsler manifold:

    dX^i/dt + G^i_j(γ(t), X(t)) γ̇^j(t) = 0.

The coefficients are evaluated at the *translated vector* ``X(t)``, not at the
curve's velocity, so the equation is homogeneous of degree one in ``X`` but in
general nonlinear. It preserves the Finsler norm ``F``, which makes translation
a map between indicatrices and the holonomy group a subgroup of the
diffeomorphism group of the indicatrix rather than of ``O(n)``.

``G^i_j`` is linear in ``y`` exactly on Berwald manifolds — in particular
Riemannian ones, where ``G^i_j(x, y) = Γ^i_{jk}(x) y^k`` for the Levi-Civita
symbols and the translation above collapses to the usual linear one.

See ``spec/MATH_SPEC.md § 3``.
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from ham.geometry.metric import FinslerMetric


class Connection(eqx.Module):
    """
    Base class for geometric connections.
    Inherits from eqx.Module to ensure valid JAX PyTree behavior.
    """

    metric: FinslerMetric

    @abstractmethod
    def parallel_transport(
        self, path_x: jax.Array, path_v: jax.Array, vec_start: jax.Array
    ) -> jax.Array:
        """
        Transport a vector along a discrete path.

        Args:
            path_x: Positions along the curve, shape (T, D).
            path_v: Velocities along the curve, shape (T, D).
            vec_start: Initial tangent vector to transport, shape (D,).

        Returns:
            Transported vectors at each point, shape (T, D).
        """
        pass


class BerwaldConnection(Connection):
    r"""
    The Berwald connection of a Finsler spray.

    Its coefficients are the first velocity derivatives of the spray,
    :math:`G^i_j = \partial G^i / \partial y^j`. They span the horizontal
    distribution, and :meth:`parallel_transport` integrates the horizontal
    curves — the parallel translation whose loops generate the holonomy group.

    See ``spec/MATH_SPEC.md § 3``.
    """

    def connection_coefficients(self, x: jax.Array, y: jax.Array) -> jax.Array:
        r"""
        Berwald coefficients :math:`G^i_j = \partial G^i/\partial y^j`.

        Homogeneous of degree one in ``y``, which is what makes the translation
        they define homogeneous of degree one in the translated vector. Linear
        in ``y`` exactly on Berwald manifolds, where they reduce to
        :math:`\Gamma^i_{jk}(x) y^k`.

        Args:
            x: Position, shape (D,).
            y: Tangent vector, shape (D,).

        Returns:
            Coefficients, shape (D, D), indexed ``[i, j]``.
        """
        return jax.jacfwd(self.metric.spray, argnums=1)(x, y)

    def parallel_transport(
        self, path_x: jax.Array, path_v: jax.Array, vec_start: jax.Array
    ) -> jax.Array:
        r"""
        Parallel translation of ``vec_start`` along ``(path_x, path_v)``.

        Integrates the horizontality condition

        .. math::
            \dot X^i + G^i_j(\gamma, X)\,\dot\gamma^j = 0,

        in which the coefficients are evaluated at the *translated vector*
        rather than at the curve's velocity. The equation is therefore nonlinear
        in ``X``, and the resulting map is positively homogeneous of degree one
        and preserves the Finsler norm.

        Args:
            path_x: Discrete positions along the curve, shape (T, D).
            path_v: Velocities at each position, shape (T, D).
            vec_start: Initial tangent vector, shape (D,).

        Returns:
            Transported vectors aligned with path_x, shape (T, D).
            Entry ``i`` is the transported vector at ``path_x[i]``; in particular
            ``result[0] == vec_start``.

        Note:
            Assumes the curve is parameterised over [0, 1] with uniform spacing.
            Implementations may be vmapped externally for batched paths.
        """
        if path_x.shape[0] < 2:
            return jnp.broadcast_to(vec_start, path_x.shape)

        dt = 1.0 / (len(path_x) - 1)

        def transport_ode(carry_vec, inputs):
            x, x_next, v = inputs

            # G^i_j evaluated at the carried vector — this is what makes the
            # translation horizontal rather than linear. (D, D)
            g_coeff = self.connection_coefficients(x, carry_vec)

            # dX^i/dt = - G^i_j(x, X) v^j
            dvec = -jnp.einsum("ij,j->i", g_coeff, v)

            new_vec = carry_vec + dvec * dt

            # Project onto tangent space at the NEXT point to prevent drift bias
            new_vec = self.metric.manifold.to_tangent(x_next, new_vec)
            return new_vec, new_vec

        _, transported_vecs = jax.lax.scan(
            transport_ode, vec_start, (path_x[:-1], path_x[1:], path_v[:-1])
        )

        return jnp.concatenate([vec_start[None, :], transported_vecs], axis=0)
