import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from ham.geometry import Torus
from ham.geometry.manifold import Manifold, _safe_norm_ratio


class MockManifold(Manifold):
    """A minimal concrete manifold for testing default methods."""
    @property
    def ambient_dim(self) -> int:
        return 3

    @property
    def intrinsic_dim(self) -> int:
        return 2

    def project(self, x: jax.Array) -> jax.Array:
        # Project onto the sphere
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return jnp.where(norm > 0, x / norm, x)

    def to_tangent(self, x: jax.Array, v: jax.Array) -> jax.Array:
        # Tangent projection for a sphere
        return v - jnp.sum(v * x, axis=-1, keepdims=True) * x

    def retract(self, x: jax.Array, delta: jax.Array) -> jax.Array:
        # Projected retraction
        return self.project(x + delta)

    def random_sample(self, key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        pts = jax.random.normal(key, (*shape, 3))
        return self.project(pts)


class TestManifold(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        self.manifold = MockManifold()

    def test_safe_norm_ratio_primal(self):
        # Normal case
        x = jnp.array([3.0, 4.0]) # norm 5
        y = jnp.array([1.0, 0.0]) # norm 1
        np.testing.assert_allclose(_safe_norm_ratio(x, y), jnp.array([5.0]), atol=1e-5)

        # y is zero vector
        y_zero = jnp.array([0.0, 0.0])
        np.testing.assert_allclose(_safe_norm_ratio(x, y_zero), jnp.array([1.0]), atol=1e-5)

        # Both zero
        x_zero = jnp.array([0.0, 0.0])
        np.testing.assert_allclose(_safe_norm_ratio(x_zero, y_zero), jnp.array([1.0]), atol=1e-5)

    def test_safe_norm_ratio_grad(self):
        # check_grads verifies the custom JVP against numerical differentiation.
        # We check a normal case.
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])

        # Test normal case
        check_grads(_safe_norm_ratio, (x, y), order=1, modes=['fwd'])

        # Test edge case x=0, y!=0
        x_zero = jnp.array([0.0, 0.0, 0.0])
        # Numerically, diff at x=0 is undefined (conical point), but the JVP should handle it gracefully without NaNs.
        # check_grads might fail finite diff at strict zero due to non-differentiability,
        # so we just test that jax.jacfwd doesn't produce NaNs.
        J_x = jax.jacfwd(lambda x_: _safe_norm_ratio(x_, y))(x_zero)
        self.assertFalse(jnp.any(jnp.isnan(J_x)))

        # Test edge case y=0
        y_zero = jnp.array([0.0, 0.0, 0.0])
        J_y = jax.jacfwd(lambda y_: _safe_norm_ratio(x, y_))(y_zero)
        self.assertFalse(jnp.any(jnp.isnan(J_y)))

    def test_default_log_map_coincident(self):
        x = jnp.array([1.0, 0.0, 0.0])
        v = self.manifold.log_map(x, x)
        # Should return exactly 0
        np.testing.assert_allclose(v, jnp.zeros(3), atol=1e-5)

    def test_default_log_map_normal_secant(self):
        # Use Torus where a purely normal secant can be easily created.
        torus = Torus(major_R=2.0, minor_r=1.0)
        # Point on outer equator:
        x = jnp.array([3.0, 0.0, 0.0])
        # Displace radially outward (purely normal to the surface)
        y = jnp.array([4.0, 0.0, 0.0])

        v = torus.log_map(x, y)
        # Since displacement is purely normal, tangent projection is zero.
        # log_map should handle this and return zero, avoiding division by zero.
        np.testing.assert_allclose(v, jnp.zeros(3), atol=1e-5)

    def test_log_map_jit_vmap(self):
        # Verify composition with JAX transforms
        batch_size = 10
        k1, k2 = jax.random.split(self.key)
        xs = self.manifold.random_sample(k1, (batch_size,))
        ys = self.manifold.random_sample(k2, (batch_size,))

        # vmap
        vmap_log = jax.vmap(self.manifold.log_map)
        vs = vmap_log(xs, ys)
        self.assertEqual(vs.shape, (batch_size, 3))

        # jit
        jit_vmap_log = jax.jit(vmap_log)
        vs_jit = jit_vmap_log(xs, ys)
        np.testing.assert_allclose(vs, vs_jit, atol=1e-5)

        # Ensure all vectors are tangent
        projs = jax.vmap(self.manifold.to_tangent)(xs, vs)
        np.testing.assert_allclose(vs, projs, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
