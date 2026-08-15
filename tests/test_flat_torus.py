"""Exactness suite for the Clifford ``FlatTorus`` manifold (validation rung L0).

Run: ``JAX_PLATFORMS=cpu pytest tests/test_flat_torus.py`` (and again with
``JAX_ENABLE_X64=1`` for the tight float64 guarantee).

The Clifford torus is intrinsically flat, so every operation is closed-form and
*exact* — a far stronger gate than the donut :class:`~finslerax.geometry.manifolds.torus.Torus`
(whose ``log_map`` is only approximate). These tests are the geometry contract the
robot-arm experiment's metric/solver stack stands on: a wrong torus silently bends
every geodesic.
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from _precision import tol

from finslerax.geometry.manifolds import EuclideanSpace, FlatTorus

jax.config.update("jax_platform_name", "cpu")


def _wrap(theta):
    """Wrap angles to (-pi, pi]."""
    return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


@pytest.mark.parametrize("dim", [1, 2, 4, 7])
def test_round_trip_exp_log(dim):
    """``exp_map(x, log_map(x, y)) == y`` exactly, and ``log`` is tangent."""
    atol, rtol = tol(atol32=1e-5, atol64=1e-10)
    T = FlatTorus(dim)
    key = jax.random.PRNGKey(dim)
    k0, k1 = jax.random.split(key)
    x = T.random_sample(k0)
    y = T.random_sample(k1)

    v = T.log_map(x, y)
    # log lands in the tangent space (no radial component).
    np.testing.assert_allclose(T.to_tangent(x, v), v, atol=atol, rtol=rtol)
    # exp inverts log exactly.
    np.testing.assert_allclose(T.exp_map(x, v), y, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dim", [2, 4, 7])
def test_closed_form_geodesic_length(dim):
    """Geodesic length equals ``||wrap(q1 - q0)||`` — the exact flat ground truth."""
    atol, rtol = tol(atol32=1e-4, atol64=1e-9)
    T = FlatTorus(dim)
    key = jax.random.PRNGKey(100 + dim)
    k0, k1 = jax.random.split(key)
    q0 = jax.random.uniform(k0, (dim,), minval=-jnp.pi, maxval=jnp.pi)
    q1 = jax.random.uniform(k1, (dim,), minval=-jnp.pi, maxval=jnp.pi)
    x0, x1 = T.embed_angles(q0), T.embed_angles(q1)

    length_closed = jnp.linalg.norm(_wrap(q1 - q0))
    # The log vector's ambient norm equals the wrapped-angle norm (orthonormal frame).
    np.testing.assert_allclose(
        jnp.linalg.norm(T.log_map(x0, x1)), length_closed, atol=atol, rtol=rtol
    )
    # Summing chord lengths along the sampled geodesic recovers the same length.
    v = T.log_map(x0, x1)
    ts = jnp.linspace(0.0, 1.0, 64)
    geo = jax.vmap(lambda t: T.exp_map(x0, t * v))(ts)
    seg = jax.vmap(lambda a, b: jnp.linalg.norm(T.log_map(a, b)))(geo[:-1], geo[1:])
    np.testing.assert_allclose(jnp.sum(seg), length_closed, atol=atol, rtol=rtol)


def test_seam_wraps():
    """Across the +/-pi seam the step is small, not ~2 pi (no naive subtraction)."""
    T = FlatTorus(1)
    a = T.embed_angles(jnp.array([jnp.pi - 0.01]))
    b = T.embed_angles(jnp.array([-jnp.pi + 0.01]))
    step = jnp.linalg.norm(T.log_map(a, b))
    assert float(step) < 0.05, float(step)  # ~0.02, the short way round


def test_matches_euclidean_away_from_seam():
    """Away from the seam the log's angular content matches the Euclidean difference."""
    atol, rtol = tol(atol32=1e-5, atol64=1e-10)
    dim = 4
    T = FlatTorus(dim)
    EuclideanSpace(dim)  # reference semantics: log = y - x in angle space
    qa = jnp.array([0.1, -0.2, 0.3, 0.0])
    qb = jnp.array([0.25, -0.05, 0.5, -0.15])
    xa, xb = T.embed_angles(qa), T.embed_angles(qb)
    # Recover per-joint angular increments from the ambient log via the frame.
    frame = T.tangent_frame(xa)  # (2n, n), orthonormal columns
    increments = frame.T @ T.log_map(xa, xb)
    np.testing.assert_allclose(increments, qb - qa, atol=atol, rtol=rtol)


def test_project_and_angle_round_trip():
    """``project`` is idempotent on-manifold; ``to_angles``/``embed_angles`` invert."""
    atol, rtol = tol(atol32=1e-5, atol64=1e-11)
    T = FlatTorus(3)
    q = jnp.array([0.4, -1.2, 2.9])
    x = T.embed_angles(q)
    np.testing.assert_allclose(T.project(x), x, atol=atol, rtol=rtol)
    np.testing.assert_allclose(_wrap(T.to_angles(x)), _wrap(q), atol=atol, rtol=rtol)
    # project pulls an off-manifold point to unit blocks (each block norm 1).
    off = x * 2.5 + 0.3
    blocks = T.project(off).reshape(3, 2)
    np.testing.assert_allclose(
        jnp.linalg.norm(blocks, axis=-1), jnp.ones(3), atol=atol, rtol=rtol
    )


def test_tangent_frame_lifts_metric():
    """``E(x)`` is an isometry: ``E^T E = I`` and it lifts a joint tensor faithfully.

    This is the identity the robot-arm metric relies on — a joint-space inertia
    ``M(q)`` lifts to the ambient metric ``E M E^T`` whose segment energy equals
    the intrinsic ``dtheta^T M dtheta`` (so AVBD on the torus == AVBD on angles).
    """
    atol, rtol = tol(atol32=1e-5, atol64=1e-10)
    dim = 5
    T = FlatTorus(dim)
    key = jax.random.PRNGKey(7)
    k0, k1, k2 = jax.random.split(key, 3)
    x = T.random_sample(k0)
    frame = T.tangent_frame(x)  # (2n, n)
    # Orthonormal columns.
    np.testing.assert_allclose(frame.T @ frame, jnp.eye(dim), atol=atol, rtol=rtol)

    # Lift identity: v = E q_dot has the same quadratic form under E M E^T.
    A = jax.random.normal(k1, (dim, dim))
    M = A @ A.T + jnp.eye(dim)
    q_dot = jax.random.normal(k2, (dim,))
    v = frame @ q_dot  # ambient tangent
    G = frame @ M @ frame.T
    np.testing.assert_allclose(v @ G @ v, q_dot @ M @ q_dot, atol=atol, rtol=rtol)
    # The lifted ambient vector is genuinely tangent at x.
    np.testing.assert_allclose(T.to_tangent(x, v), v, atol=atol, rtol=rtol)


def test_dims_and_vmap_jit():
    """Dimensions are correct and the ops are vmap/jit-compatible."""
    dim = 6
    T = FlatTorus(dim)
    assert T.ambient_dim == 12
    assert T.intrinsic_dim == 6
    key = jax.random.PRNGKey(0)
    batch = jax.vmap(T.project)(jax.random.normal(key, (8, 12)))
    assert batch.shape == (8, 12)
    ang = jax.jit(T.to_angles)(T.random_sample(key))
    assert ang.shape == (6,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
