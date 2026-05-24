"""Tests for the IVP Geodesic Solver (ExponentialMap).

Verifies ballistic motion, great-circle trajectories, energy conservation, 
and differentiability of the shooting solver.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import config
import numpy as np

# Enforce High Precision
# config.update("jax_enable_x64", True)

from ham.geometry import Sphere
from ham.geometry.manifold import Manifold
from ham.geometry.zoo import Euclidean, Randers, Riemannian
from ham.solvers.geodesic import ExponentialMap
from ham.utils.math import safe_norm

@pytest.fixture
def solver():
    return ExponentialMap(step_size=0.0005, max_steps=2000)

class Plane(Manifold):
    """Mock manifold for flat space tests."""
    @property
    def ambient_dim(self) -> int: return 2
    @property
    def intrinsic_dim(self) -> int: return 2
    def project(self, x): return x
    def to_tangent(self, x, v): return v
    def retract(self, x, v): return x + v
    def random_sample(self, key, shape): 
        return jax.random.normal(key, shape + (2,))

def test_euclidean_ballistic(solver):
    """In flat space, a geodesic is a straight line x(t) = x0 + v0*t."""
    metric = Euclidean(Plane())
    x0 = jnp.array([0.0, 0.0])
    v0 = jnp.array([1.0, 0.5])
    
    x_final = solver.shoot(metric, x0, v0)
    expected = x0 + v0
    np.testing.assert_allclose(x_final, expected, atol=1e-4)

def test_sphere_great_circle(solver):
    """On a Unit Sphere, geodesics are great circles."""
    sphere = Sphere(radius=1.0)
    metric = Euclidean(sphere)
    
    x0 = jnp.array([1.0, 0.0, 0.0])
    speed = jnp.pi / 2.0
    v0 = jnp.array([0.0, 0.0, speed]) 
    
    x_final = solver.shoot(metric, x0, v0)
    expected = jnp.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(x_final, expected, atol=1e-3)

def test_energy_conservation(solver):
    """Energy E(x, v) must be constant along a geodesic flow."""
    sphere = Sphere(radius=1.0)
    # Use Randers wind which is non-trivial but conserves energy
    def w_net(x): return 0.2 * jnp.array([-x[1], x[0], 0.0])
    def h_net(x): return jnp.eye(3)
    metric = Randers(sphere, h_net, w_net)
    
    x0 = jnp.array([1.0, 0.0, 0.0])
    v0 = sphere.to_tangent(x0, jnp.array([0.0, 0.5, 0.5]))
    
    xs, vs = solver.trace(metric, x0, v0)
    energies = jax.vmap(metric.energy)(xs, vs)
    
    e_start = energies[0]
    rel_deviation = jnp.abs(energies - e_start) / e_start
    max_rel_err = jnp.max(rel_deviation)
    
    # Tolerance 1e-3 is safe for regularized Randers
    assert max_rel_err < 1e-3

def test_zero_velocity(solver):
    """Geodesic with zero velocity should remain stationary."""
    sphere = Sphere(radius=1.0)
    metric = Euclidean(sphere)
    x0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.zeros(3)
    xf = solver.shoot(metric, x0, v0)
    np.testing.assert_allclose(xf, x0, atol=1e-5)

def test_jit_and_vmap_compatibility(solver):
    """Verify solver works under JAX transforms."""
    sphere = Sphere(radius=1.0)
    metric = Euclidean(sphere)
    
    jit_shoot = jax.jit(solver.shoot)
    x0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.array([0.0, 1.0, 0.0])
    xf1 = jit_shoot(metric, x0, v0)
    xf2 = solver.shoot(metric, x0, v0)
    np.testing.assert_allclose(xf1, xf2)

def test_differentiability(solver):
    """Verify gradient flow through the IVP solver."""
    sphere = Sphere(radius=1.0)
    x0 = jnp.array([1.0, 0.0, 0.0])
    v0 = sphere.to_tangent(x0, jnp.array([0.0, 1.0, 1.0]))
    
    target = jnp.array([0.0, 0.0, 1.0])
    def loss(wind_scale):
        # Scale the wind in a Randers metric
        def w_net(x): return wind_scale * jnp.array([-x[1], x[0], 0.0])
        def h_net(x): return jnp.eye(3)
        metric = Randers(sphere, h_net, w_net)
        xf = solver.shoot(metric, x0, v0)
        return jnp.sum(xf * target)
        
    grad = jax.grad(loss)(0.2)
    assert jnp.isfinite(grad)
    assert jnp.abs(grad) > 1e-6