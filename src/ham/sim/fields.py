"""Analytical vector field generators for spherical and planar simulations.

Provides divergence-free spherical flows via the stream-function construction
v = ∇ψ × x (see get_stream_function_flow), as well as classical 2-D planar
vortex models (Lamb–Oseen, Rankine).

All factory functions in this module return a closure `flow(x) -> v` that 
operates on a single point `x`. Use `jax.vmap` to evaluate on batched inputs.
"""

import jax
import jax.numpy as jnp
from typing import Callable
from ham.utils.math import NORM_EPS, GRAD_EPS, safe_norm

def get_stream_function_flow(psi_fn: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array], jax.Array]:
    """Converts a scalar stream function into a divergence-free vector field on S².

    The resulting flow v = ∇ψ × x is guaranteed to be tangential to the 
    unit sphere (v · x = 0) and divergence-free.

    Args:
        psi_fn: Scalar stream function ψ: ℝ³ → ℝ. Must be a differentiable
            function accepting a single point x of shape (3,).

    Returns:
        A vector field function (3,) → (3,) mapping a point on the sphere to
        a divergence-free tangent vector.
        
    Note:
        Tangency (v · x = 0) is only guaranteed if |x| = 1.
    """
    def flow(x: jax.Array) -> jax.Array:
        # Ensure x is a flat (3,) array
        x_flat = jnp.asarray(x).reshape(-1)
        # jax.grad computes the ambient gradient in ℝ³
        grad_psi = jax.grad(psi_fn)(x_flat)
        return jnp.cross(grad_psi, x_flat)
    
    return flow

def tilted_rotation(alpha_deg: float = 45.0) -> Callable[[jax.Array], jax.Array]:
    """Constant rotation around a tilted axis on the sphere.

    The rotation axis is (sin α, 0, cos α) in ambient ℝ³, where α = alpha_deg.
    When α = 0 the axis is the north pole (z-axis).

    Args:
        alpha_deg: Tilt angle from the z-axis in degrees.

    Returns:
        Tangential vector field (3,) → (3,) on the unit sphere.
    """
    alpha = jnp.radians(alpha_deg)
    axis = jnp.array([jnp.sin(alpha), 0.0, jnp.cos(alpha)])
    # axis is already unit-norm, but we add a safety guard correctly
    axis = axis / jnp.maximum(jnp.linalg.norm(axis), NORM_EPS)
    
    def psi(x: jax.Array) -> jax.Array:
        return jnp.dot(axis, x.reshape(-1))
    
    return get_stream_function_flow(psi)

def rossby_haurwitz(R: int = 4, omega: float = 1.0, K: float = 1.0) -> Callable[[jax.Array], jax.Array]:
    """Rossby–Haurwitz wave – classic wavy planetary-scale flow on the sphere.

    The stream function is defined as:
        ψ = -ω sin(lat) + K cos^R(lat) sin(lat) cos(R lon)
    
    Reference:
        Haurwitz, B. (1940). The motion of atmospheric disturbances on the 
        spherical earth. J. Mar. Res., 3, 254–267.

    Args:
        R: Azimuthal wave number (integer, default 4).
        omega: Amplitude of the solid-body rotation component.
        K: Amplitude of the Rossby–Haurwitz wave component.

    Returns:
        Tangential vector field (3,) → (3,) on the unit sphere.
    """
    def psi(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x).reshape(-1)
        # Coordinate map: z ≡ sin(lat), ρ_xy ≡ cos(lat), lon ≡ atan2(y, x)
        z = x[2]
        rho_xy = jnp.sqrt(jnp.sum(x[:2]**2) + GRAD_EPS)
        
        # Avoid branch cuts by using complex representation for cos(R*lon)
        xy_complex = x[0] + 1j * x[1]
        xy_norm = jnp.maximum(jnp.abs(xy_complex), GRAD_EPS)
        xy_unit = xy_complex / xy_norm
        cos_R_lon = jnp.real(xy_unit ** R)
        
        term1 = -omega * z
        term2 = K * (rho_xy ** R) * z * cos_R_lon
        
        return term1 + term2
    
    return get_stream_function_flow(psi)

def harmonic_vortices(ell: int = 5, m: int = 3) -> Callable[[jax.Array], jax.Array]:
    """Cellular vortex flow using a sinusoidal latitudinal profile.

    Generates a high-frequency grid of vortices on the sphere surface.

    Args:
        ell: Degree (controls latitudinal frequency).
        m: Azimuthal wavenumber (controls longitudinal frequency).

    Returns:
        Tangential vector field (3,) → (3,) on the unit sphere.
    """
    def psi(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x).reshape(-1)
        z = x[2]
        
        cos_lat = jnp.sqrt(1.0 - z**2 + GRAD_EPS)
        cos_lat_m = cos_lat ** m
        
        xy_complex = x[0] + 1j * x[1]
        xy_norm = jnp.maximum(jnp.abs(xy_complex), GRAD_EPS)
        xy_unit = xy_complex / xy_norm
        cos_m_lon = jnp.real(xy_unit ** m)
        
        # High-frequency oscillation in latitude
        z_poly = jnp.sin(ell * jnp.pi * z)
        
        return cos_lat_m * z_poly * cos_m_lon
    
    return get_stream_function_flow(psi)

def lamb_oseen_vortex(center: jax.Array, core_radius: float = 1.0, circulation: float = 1.0) -> Callable[[jax.Array], jax.Array]:
    """2-D Lamb–Oseen vortex (smoothed point vortex).

    The velocity profile is defined as:
        v_θ(r) = (Γ / 2πr)(1 − exp(−r² / r_c²))

    Args:
        center: Vortex center position, shape (2,) or (D,).
        core_radius: Viscous core radius r_c.
        circulation: Total circulation Γ.

    Returns:
        Vector field function (D,) -> (D,). Output dimension matches input.
    """
    def flow(x: jax.Array) -> jax.Array:
        r_vec = x[:2] - center[:2]
        r_sq = jnp.sum(r_vec**2) + GRAD_EPS
        r = jnp.sqrt(r_sq)
        
        v_theta = (circulation / (2 * jnp.pi * r)) * (1.0 - jnp.exp(-r_sq / (core_radius**2)))
        
        # Map azimuthal velocity to Cartesian components
        v_x = -v_theta * r_vec[1] / r
        v_y = v_theta * r_vec[0] / r
        
        # Preserve D-dimensional shape by padding with zeros
        v = jnp.zeros_like(x)
        v = v.at[0].set(v_x)
        v = v.at[1].set(v_y)
        return v
        
    return flow

def rankine_vortex(center: jax.Array, core_radius: float = 1.0, circulation: float = 1.0) -> Callable[[jax.Array], jax.Array]:
    """2-D Rankine vortex (solid-body core, irrotational exterior).

    The velocity profile is defined piecewise:
        v_θ(r) = Γr / (2π r_c²)   for r ≤ r_c
        v_θ(r) = Γ / (2πr)         for r > r_c

    Args:
        center: Vortex center position, shape (2,) or (D,).
        core_radius: Core radius r_c separating the two regimes.
        circulation: Total circulation Γ.

    Returns:
        Vector field function (D,) -> (D,). Output dimension matches input.
    """
    def flow(x: jax.Array) -> jax.Array:
        r_vec = x[:2] - center[:2]
        r = safe_norm(r_vec, eps=GRAD_EPS)
        
        v_theta = jnp.where(
            r <= core_radius, 
            (circulation * r) / (2 * jnp.pi * core_radius**2),
            circulation / (2 * jnp.pi * r)
        )
        
        v_x = -v_theta * r_vec[1] / r
        v_y = v_theta * r_vec[0] / r
        
        v = jnp.zeros_like(x)
        v = v.at[0].set(v_x)
        v = v.at[1].set(v_y)
        return v
        
    return flow