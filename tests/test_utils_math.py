import jax
import jax.numpy as jnp
import pytest

from ham.utils.math import GRAD_EPS, safe_norm, safe_norm_additive


def test_safe_norm_forward_zero():
    """Verify safe_norm(0) returns sqrt(eps)."""
    x = jnp.zeros((3,))
    result = safe_norm(x)
    expected = jnp.sqrt(GRAD_EPS)
    assert jnp.allclose(result, expected)

def test_safe_norm_grad_at_zero():
    """Verify gradient at zero is finite (zero)."""
    def f(x):
        return safe_norm(x)

    g = jax.grad(f)
    grad_val = g(jnp.zeros((3,)))
    assert jnp.all(jnp.isfinite(grad_val))
    assert jnp.all(grad_val == 0.0)

def test_safe_norm_matches_linalg():
    """Verify safe_norm matches jnp.linalg.norm for large values."""
    x = jnp.array([3.0, 4.0])
    result = safe_norm(x)
    expected = 5.0
    assert jnp.allclose(result, expected)

def test_safe_norm_vmap():
    """Verify vmap support."""
    x = jnp.array([[3.0, 4.0], [0.0, 0.0]])
    results = jax.vmap(safe_norm)(x)
    assert jnp.allclose(results[0], 5.0)
    assert jnp.allclose(results[1], jnp.sqrt(GRAD_EPS))

def test_safe_norm_jit():
    """Verify JIT support."""
    x = jnp.array([3.0, 4.0])
    result = jax.jit(safe_norm)(x)
    assert jnp.allclose(result, 5.0)

def test_safe_norm_additive_smoothness():
    """Verify additive norm is smooth at zero (non-zero 2nd derivative)."""
    def f(x):
        return safe_norm_additive(x, eps=1e-3)

    # 1st derivative should be 0 at origin
    g = jax.grad(f)
    assert jnp.allclose(g(jnp.zeros((3,))), 0.0)

    # 2nd derivative (Hessian) should be 1/eps * Identity at origin
    # d^2/dv^2 sqrt(v^2 + eps^2) = 1/eps at v=0
    h = jax.hessian(f)
    hess = h(jnp.zeros((3,)))
    assert jnp.allclose(hess, jnp.eye(3) * (1.0 / 1e-3))

if __name__ == "__main__":
    pytest.main([__file__])
