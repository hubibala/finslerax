"""
Parallel transport along curves on Finsler manifolds.

The canonical parallel translation of a Finsler manifold is the *horizontal*
one, carried by the nonlinear connection ``N^i_j = ∂G^i/∂y^j`` that the geodesic
spray induces. It is positively homogeneous of degree one rather than linear,
and it preserves the Finsler norm ``F``. That norm preservation is what makes
translation a map between indicatrices, and therefore what makes the holonomy
group a subgroup of the diffeomorphism group of the indicatrix.

Differentiating the nonlinear coefficients once more in the velocity gives the
linear Berwald coefficients ``^BΓ^i_jk = ∂²G^i/∂y^j∂y^k``. They are the
Christoffel symbols of Levi-Civita whenever the spray is quadratic in the
velocity, and :meth:`BerwaldConnection.christoffel_symbols` exposes them for
analysis. They do not define the transport: the linear equation they generate is
not metric-compatible and is not an isometry of ``F``.

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
    def christoffel_symbols(self, x: jax.Array, v: jax.Array) -> jax.Array:
        """
        Compute the linear connection coefficients at (x, v).

        Args:
            x: Position on the manifold, shape (D,).
            v: Tangent vector at x, shape (D,).

        Returns:
            Connection coefficients Γ^i_{jk}, shape (D, D, D).
        """
        pass

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
    The connection induced by a Finsler spray.

    Two objects live here, both derived from the spray :math:`G^i(x, y)`:

    * the nonlinear connection :math:`N^i_j = \partial G^i / \partial y^j`,
      whose horizontal curves define the canonical parallel translation; and
    * the linear Berwald connection
      :math:`^B\Gamma^i_{jk} = \partial N^i_j / \partial y^k`.

    :meth:`parallel_transport` uses the first. See ``spec/MATH_SPEC.md § 3``.
    """

    def nonlinear_coefficients(self, x: jax.Array, y: jax.Array) -> jax.Array:
        r"""
        Nonlinear connection coefficients :math:`N^i_j = \partial G^i/\partial y^j`.

        Homogeneous of degree one in ``y``, which is what makes the transport it
        defines homogeneous of degree one in the transported vector.

        Args:
            x: Position, shape (D,).
            y: Tangent vector, shape (D,).

        Returns:
            Coefficients, shape (D, D), indexed ``[i, j]``.
        """
        return jax.jacfwd(self.metric.spray, argnums=1)(x, y)

    def christoffel_symbols(self, x: jax.Array, v: jax.Array) -> jax.Array:
        r"""
        Linear Berwald coefficients :math:`^B\Gamma^i_{jk}=\partial^2 G^i/\partial v^j \partial v^k`.

        Torsion-free, and equal to the Christoffel symbols of Levi-Civita when
        the metric is Riemannian. A derived quantity for analysis; transport is
        :meth:`parallel_transport`, which is horizontal rather than linear.

        Args:
            x: Position, shape (D,).
            v: Tangent vector, shape (D,).

        Returns:
            Coefficients tensor, shape (D, D, D).

        Note:
            This differentiates through the linear solver in `metric.spray`. It assumes
            the Hessian of the energy is reasonably conditioned (regularised).
        """
        jacobian_v = jax.jacfwd(self.metric.spray, argnums=1)
        hessian_v = jax.jacfwd(jacobian_v, argnums=1)
        return hessian_v(x, v)

    def parallel_transport(
        self, path_x: jax.Array, path_v: jax.Array, vec_start: jax.Array
    ) -> jax.Array:
        r"""
        Horizontal parallel translation of ``vec_start`` along ``(path_x, path_v)``.

        Integrates the horizontality condition

        .. math::
            \dot Y^i + N^i_j(\gamma, Y)\,\dot\gamma^j = 0,

        in which the coefficients are evaluated at the *transported vector*
        rather than at the curve's velocity. The equation is therefore nonlinear
        in ``Y``, and the resulting map is positively homogeneous of degree one
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

            # N^i_j evaluated at the carried vector — this is what makes the
            # translation horizontal rather than linear. (D, D)
            n_coeff = self.nonlinear_coefficients(x, carry_vec)

            # dY^i/dt = - N^i_j(x, Y) v^j
            dvec = -jnp.einsum("ij,j->i", n_coeff, v)

            new_vec = carry_vec + dvec * dt

            # Project onto tangent space at the NEXT point to prevent drift bias
            new_vec = self.metric.manifold.to_tangent(x_next, new_vec)
            return new_vec, new_vec

        _, transported_vecs = jax.lax.scan(
            transport_ode, vec_start, (path_x[:-1], path_x[1:], path_v[:-1])
        )

        return jnp.concatenate([vec_start[None, :], transported_vecs], axis=0)
