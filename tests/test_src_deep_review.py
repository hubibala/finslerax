"""Regression tests added during the 2026-06-13 deep src/ham review.

Covers findings that previously had no direct test:

* W-RAND: the Randers/DiscreteRanders wind squash must be continuous (and in
  fact C^1) -- the old ``where(w_norm < 0.5, 1.0, ...)`` gate left a jump
  discontinuity at the 0.5 boundary.
* W-MK: every manifold exposes a generic ``tangent_dot`` / ``tangent_norm`` so
  downstream code never reaches for hyperboloid-private ``_minkowski_dot``.
* Finsler axioms: positive 1-homogeneity F(x, c v) = c F(x, v) and positive
  definiteness of the fundamental tensor g_ij = d^2 E / dv^i dv^j.
"""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ham.geometry import (
    Euclidean,
    EuclideanSpace,
    Hyperboloid,
    Randers,
    Riemannian,
    Sphere,
)
from ham.geometry.manifold import Manifold


class _FlatPlane(Manifold):
    @property
    def ambient_dim(self):
        return 2

    @property
    def intrinsic_dim(self):
        return 2

    def project(self, x):
        return x

    def to_tangent(self, x, v):
        return v

    def retract(self, x, delta):
        return x + delta

    def random_sample(self, key, shape):
        return jax.random.normal(key, (*shape, 2))


def _randers_with_constant_wind(wind, *, epsilon=1e-5):
    """A 2-D Randers metric with H = I and a constant raw wind vector."""
    manifold = _FlatPlane()
    return Randers(
        manifold,
        h_net=lambda x: jnp.eye(2),
        w_net=lambda x: jnp.asarray(wind, dtype=jnp.float32),
        epsilon=epsilon,
    )


class TestWindSquashContinuity(unittest.TestCase):
    """W-RAND: the squashed wind magnitude must vary continuously."""

    def test_no_jump_across_old_boundary(self):
        x = jnp.zeros(2)
        # Sweep the raw wind magnitude finely through the old 0.5 branch point.
        mags = np.linspace(0.40, 0.60, 201)
        squashed = []
        for m in mags:
            metric = _randers_with_constant_wind([m, 0.0])
            _, W, _ = metric.zermelo_data(x)
            squashed.append(float(jnp.linalg.norm(W)))
        squashed = np.asarray(squashed)
        max_step = np.max(np.abs(np.diff(squashed)))
        # Continuous map over a step of ~1e-3 in input must move by O(1e-3); the
        # pre-fix code jumped by ~0.04 (=0.5 - max_speed*tanh(0.5)) at m=0.5.
        self.assertLess(
            max_step,
            5e-3,
            f"wind magnitude jumped by {max_step:.4f}; squash is discontinuous",
        )

    def test_causal_bound_holds(self):
        """||W||_H < 1 must hold even for very strong raw winds."""
        x = jnp.zeros(2)
        for m in [0.1, 0.49, 0.51, 1.0, 5.0, 50.0]:
            metric = _randers_with_constant_wind([m, 0.0])
            H, W, lam = metric.zermelo_data(x)
            w_norm_h = float(jnp.sqrt(W @ H @ W))
            self.assertLess(w_norm_h, 1.0, f"causal bound violated at m={m}")
            self.assertGreater(float(lam), 0.0, f"lambda <= 0 at m={m}")

    def test_squash_c1_via_grad(self):
        """The squashed magnitude is differentiable across the old boundary."""

        def squashed_mag(m):
            metric = _randers_with_constant_wind([m, 0.0])
            _, W, _ = metric.zermelo_data(jnp.zeros(2))
            return jnp.sqrt(jnp.sum(W**2))

        g = jax.grad(squashed_mag)
        # Finite, non-NaN gradient on both sides of and at the old branch point.
        for m in [0.45, 0.499, 0.5, 0.501, 0.55]:
            val = float(g(jnp.float32(m)))
            self.assertTrue(np.isfinite(val), f"non-finite d|W|/dm at m={m}")


class TestGenericTangentDot(unittest.TestCase):
    """W-MK: tangent_dot / tangent_norm available on every manifold."""

    def test_euclidean_submanifolds_use_ambient_dot(self):
        for manifold in [EuclideanSpace(3), Sphere(intrinsic_dim=2)]:
            x = manifold.project(jnp.array([0.3, 0.4, 1.0]))
            u = manifold.to_tangent(x, jnp.array([1.0, 0.0, 0.0]))
            v = manifold.to_tangent(x, jnp.array([0.0, 1.0, 0.0]))
            expected = float(jnp.sum(u * v))
            got = float(manifold.tangent_dot(x, u, v))
            np.testing.assert_allclose(got, expected, atol=1e-6)
            np.testing.assert_allclose(
                float(manifold.tangent_norm(x, u)),
                float(jnp.linalg.norm(u)),
                atol=1e-6,
            )

    def test_hyperboloid_uses_minkowski(self):
        manifold = Hyperboloid(intrinsic_dim=2)
        x = jnp.array([1.0, 0.0, 0.0])  # pole
        u = jnp.array([0.0, 1.0, 0.0])
        v = jnp.array([0.0, 0.0, 2.0])
        # Generic API must agree with the private Minkowski helper.
        np.testing.assert_allclose(
            float(manifold.tangent_dot(x, u, v)),
            float(manifold._minkowski_dot(u, v)),
            atol=1e-6,
        )
        # On a spacelike tangent vector the Minkowski norm is the spatial norm.
        np.testing.assert_allclose(
            float(manifold.tangent_norm(x, u)), 1.0, atol=1e-6
        )


class TestFinslerAxioms(unittest.TestCase):
    """Positive 1-homogeneity and positive-definite fundamental tensor."""

    def _metrics(self):
        plane = _FlatPlane()
        return {
            "euclidean": Euclidean(plane),
            "riemannian": Riemannian(
                plane, g_net=lambda x: jnp.array([[2.0, 0.3], [0.3, 1.0]])
            ),
            "randers": _randers_with_constant_wind([0.3, -0.1]),
        }

    def test_positive_homogeneity(self):
        x = jnp.array([0.2, -0.4])
        v = jnp.array([0.7, 1.3])
        for name, metric in self._metrics().items():
            f_v = float(metric.metric_fn(x, v))
            for c in [0.25, 1.0, 3.0]:
                f_cv = float(metric.metric_fn(x, c * v))
                np.testing.assert_allclose(
                    f_cv, c * f_v, rtol=1e-4,
                    err_msg=f"{name}: F(x,{c}v) != {c} F(x,v)",
                )

    def test_fundamental_tensor_positive_definite(self):
        x = jnp.array([0.1, 0.2])
        key = jax.random.PRNGKey(0)
        for name, metric in self._metrics().items():
            for i in range(8):
                v = jax.random.normal(jax.random.fold_in(key, i), (2,))
                v = v / jnp.linalg.norm(v)  # nonzero direction
                g = jax.hessian(metric.energy, argnums=1)(x, v)
                g = 0.5 * (g + g.T)
                eigs = np.linalg.eigvalsh(np.asarray(g))
                self.assertGreater(
                    float(eigs.min()), 0.0,
                    f"{name}: fundamental tensor not PD (min eig {eigs.min():.3e})",
                )


class _StubModel(eqx.Module):
    """Minimal GenerativeModel-shaped stub for loss-component tests."""

    metric: object
    manifold: object

    def encode(self, x, key):
        return x  # identity encoder: data space == latent space

    def project_control(self, x, v):
        return x, v

    def decode(self, z):
        return z


class TestEulerLagrangeResidualLoss(unittest.TestCase):
    """W-EL: residual differentiates the model's true metric.energy."""

    def _loss_on_traj(self, traj):
        from ham.training.losses import EulerLagrangeResidualLoss

        manifold = EuclideanSpace(2)
        model = _StubModel(metric=Euclidean(manifold), manifold=manifold)
        loss = EulerLagrangeResidualLoss(weight=1.0)
        # batch = (x, v, trajectory); only batch[2] is used.
        return float(loss(model, (None, None, traj), jax.random.PRNGKey(0)))

    def test_zero_on_geodesic(self):
        # A straight line is the Euclidean geodesic -> EL residual ~ 0.
        t = jnp.linspace(0.0, 1.0, 11)[:, None]
        straight = jnp.concatenate([t, 2.0 * t], axis=1)
        self.assertLess(self._loss_on_traj(straight), 1e-6)

    def test_positive_on_nongeodesic(self):
        # A parabola is not a straight line -> nonzero EL residual.
        t = jnp.linspace(0.0, 1.0, 11)
        curved = jnp.stack([t, t**2], axis=1)
        self.assertGreater(self._loss_on_traj(curved), 1e-3)


class TestAlignmentLossesManifoldAgnostic(unittest.TestCase):
    """W-MK: alignment losses run on non-hyperboloid manifolds."""

    def test_zermelo_alignment_on_euclidean_space(self):
        from ham.training.losses import ZermeloAlignmentLoss

        manifold = EuclideanSpace(2)
        from ham.models.learned import NeuralRanders

        metric = NeuralRanders(manifold, jax.random.PRNGKey(1))
        model = _StubModel(metric=metric, manifold=manifold)
        loss = ZermeloAlignmentLoss(weight=1.0)
        x = jnp.array([0.1, -0.2])
        v = jnp.array([1.0, 0.5])
        # Previously raised AttributeError (manifold._minkowski_norm) on a
        # EuclideanSpace; must now return a finite scalar.
        val = float(loss(model, (x, v), jax.random.PRNGKey(2)))
        self.assertTrue(np.isfinite(val))


def _zermelo_reconstruct(metric, x, v):
    """F via the zoo/eikonal Zermelo form using metric.zermelo_data."""
    H, W, lam = metric.zermelo_data(x)
    Hv = H @ v
    HW = H @ W
    v_sq_h = float(v @ Hv)
    Wdotv = float(v @ HW)
    disc = float(lam) * v_sq_h + Wdotv**2
    return (np.sqrt(max(disc, 0.0)) - Wdotv) / float(lam)


class TestZermeloDataConsistency(unittest.TestCase):
    """MATH-ZD1: zermelo_data must reconstruct the same metric as metric_fn.

    The eikonal / mesh solvers rebuild the metric from ``zermelo_data``; if it
    disagrees with ``metric_fn`` (which AVBD and the energy/spray use), the two
    solver families silently optimize different geometries.
    """

    def test_covariate_conditioned_randers(self):
        from ham.models.wildfire import CovariateConditionedRanders

        m = CovariateConditionedRanders(
            EuclideanSpace(2), jax.random.PRNGKey(0),
            hidden_dim=16, cnn_channels=8, use_wind=True,
        )
        h, w = 8, 8
        z = jnp.zeros((h, w))
        m = m.bind_scene(
            z, z, z, z, jnp.zeros((h, w), dtype=jnp.int32),
            jnp.array([0.5, 0.5, 0.1, 0.9]), 1.0, jnp.zeros(2),
        ).precompute_metric_field()

        for v in [jnp.array([1.0, 0.3]), jnp.array([-0.5, 1.2])]:
            x = jnp.array([3.0, 4.0])
            np.testing.assert_allclose(
                _zermelo_reconstruct(m, x, v),
                float(m.metric_fn(x, v)),
                rtol=1e-4,
                err_msg="wildfire zermelo_data inconsistent with metric_fn",
            )

    def test_covariate_mesh_randers(self):
        from ham.geometry.mesh import TriangularMesh
        from ham.utils.terrain import CovariateMeshRanders

        verts = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        faces = jnp.array([[0, 1, 2], [3, 2, 1]], dtype=jnp.int32)
        mesh = TriangularMesh(verts, faces, grid_size=4)
        m = CovariateMeshRanders(mesh, jax.random.PRNGKey(0), hidden_dim=16)
        m = m.bind_scene(jnp.ones((2, 9)) * 0.3, jnp.array([0.5, 0.5, 0.1, 0.9]))

        x = jnp.array([0.3, 0.3, 0.0])
        v = jnp.array([1.0, 0.4, 0.0])
        np.testing.assert_allclose(
            _zermelo_reconstruct(m, x, v),
            float(m.metric_fn(x, v)),
            rtol=1e-4,
            err_msg="terrain zermelo_data inconsistent with metric_fn",
        )


if __name__ == "__main__":
    unittest.main()
