import jax
import jax.numpy as jnp
from typing import Callable

def get_stream_function_flow(psi_fn: Callable[[jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Converts a scalar stream function ψ(x,y,z) → ℝ into a divergence-free vector field
    on the sphere via v = ∇ψ × x (cross product with position vector).

    This ensures the flow is tangential to the sphere (if x is on unit sphere).
    """
    def flow(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x).reshape(-1)               # ensure shape (3,)
        grad_psi = jax.grad(psi_fn)(x)               # shape (3,)
        grad_psi = jnp.asarray(grad_psi).reshape(-1) # force (3,)
        
        v = jnp.cross(grad_psi, x)                   # v = ∇ψ × x
        return v
    
    return flow


def tilted_rotation(alpha_deg: float = 45.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Constant rotation around a tilted axis.
    
    Args:
        alpha_deg: Tilt angle from z-axis in degrees
    
    Returns:
        Tangential vector field (rotation around tilted axis)
    """
    alpha = jnp.radians(alpha_deg)
    axis = jnp.array([jnp.sin(alpha), 0.0, jnp.cos(alpha)])
    axis = axis / jnp.linalg.norm(axis + 1e-10)  # safe unit vector
    
    def psi(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x).reshape(-1)
        return jnp.dot(axis, x)  # linear projection onto axis
    
    return get_stream_function_flow(psi)


def rossby_haurwitz(R: int = 4, omega: float = 1.0, K: float = 1.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Rossby-Haurwitz wave – classic wavy planetary-scale flow on the sphere.
    
    ψ = -ω sin(lat) + K cos^R(lat) sin(lat) cos(R lon)
    v = ∇ψ × x
    """
    def psi(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x).reshape(-1)
        z = x[2]                            # sin(lat)
        rho_xy = jnp.sqrt(jnp.sum(x[:2]**2) + 1e-10)  # cos(lat) safe
        
        # Avoid div-by-zero in lon phase
        xy_complex = x[0] + 1j * x[1]
        xy_norm = jnp.maximum(jnp.abs(xy_complex), 1e-10)
        xy_unit = xy_complex / xy_norm
        
        cos_R_lon = jnp.real(xy_unit ** R)
        
        term1 = -omega * z
        term2 = K * (rho_xy ** R) * z * cos_R_lon
        
        return term1 + term2
    
    return get_stream_function_flow(psi)


def harmonic_vortices(l: int = 5, m: int = 3) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    High-frequency cellular / vortex flow using associated Legendre-like structure.
    
    Args:
        l: degree (controls latitudinal variation)
        m: azimuthal wavenumber
    
    Returns:
        Tangential vector field
    """
    def psi(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x).reshape(-1)
        z = x[2]
        
        # cos(lat)^m term (safe)
        cos_lat = jnp.sqrt(1.0 - z**2 + 1e-10)
        cos_lat_m = cos_lat ** m
        
        # Lon phase
        xy_complex = x[0] + 1j * x[1]
        xy_norm = jnp.maximum(jnp.abs(xy_complex), 1e-10)
        xy_unit = xy_complex / xy_norm
        cos_m_lon = jnp.real(xy_unit ** m)
        
        # Simple latitudinal polynomial (can be replaced with real Legendre)
        z_poly = jnp.sin(l * jnp.pi * z)  # high-frequency oscillation
        
        return cos_lat_m * z_poly * cos_m_lon
    
    return get_stream_function_flow(psi)

def lamb_oseen_vortex(center: jnp.ndarray, core_radius: float = 1.0, circulation: float = 1.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """2D Lamb-Oseen vortex (smoothed point vortex)."""
    def flow(x: jnp.ndarray) -> jnp.ndarray:
        r_vec = x[:2] - center[:2]
        r_sq = jnp.sum(r_vec**2) + 1e-10
        r = jnp.sqrt(r_sq)
        velocity_mag = (circulation / (2 * jnp.pi * r)) * (1.0 - jnp.exp(-r_sq / (core_radius**2)))
        v_x = -velocity_mag * r_vec[1] / r
        v_y = velocity_mag * r_vec[0] / r
        v = jnp.array([v_x, v_y])
        if x.shape[0] > 2:
            v_z = jnp.zeros(x.shape[0] - 2)
            v = jnp.concatenate([v, v_z])
        return v
    return flow

def rankine_vortex(center: jnp.ndarray, core_radius: float = 1.0, circulation: float = 1.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """2D Rankine vortex (solid body inside, irrotational outside)."""
    def flow(x: jnp.ndarray) -> jnp.ndarray:
        r_vec = x[:2] - center[:2]
        r = jnp.linalg.norm(r_vec) + 1e-10
        
        v_theta = jnp.where(r <= core_radius, 
                            (circulation * r) / (2 * jnp.pi * core_radius**2),
                            circulation / (2 * jnp.pi * r))
        v_x = -v_theta * r_vec[1] / r
        v_y = v_theta * r_vec[0] / r
        v = jnp.array([v_x, v_y])
        if x.shape[0] > 2:
            v_z = jnp.zeros(x.shape[0] - 2)
            v = jnp.concatenate([v, v_z])
        return v
    return flow