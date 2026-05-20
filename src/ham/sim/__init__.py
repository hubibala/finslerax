"""Physics-simulated vector fields and wind fields.

This module provides analytical generators for divergence-free vector fields, 
primarily for use as wind fields W(x) in Zermelo navigation experiments 
on the sphere and in 2D space.

Key features:
- Stream-function based flows on the sphere (v = ∇ψ × x).
- Classic planetary wave models (Rossby-Haurwitz).
- Standard 2D vortex models (Lamb-Oseen, Rankine).

All functions are designed to operate on single points but are fully 
compatible with `jax.vmap` for batched evaluation.

See also:
    spec/MATH_SPEC.md § 5 (Zermelo Navigation).
"""

from ham.sim.fields import (
    get_stream_function_flow,
    tilted_rotation,
    rossby_haurwitz,
    harmonic_vortices,
    lamb_oseen_vortex,
    rankine_vortex
)

__all__ = [
    "get_stream_function_flow",
    "tilted_rotation",
    "rossby_haurwitz",
    "harmonic_vortices",
    "lamb_oseen_vortex",
    "rankine_vortex"
]
