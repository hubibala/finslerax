"""Vehicle models — the Riemannian "sea" ``H`` and the operating constraints.

The vehicle is where drag/power physics enters as geometry. Quadratic drag
``D = ½ρ C_d A v²`` against a fixed power budget sets the through-water speed
``s_max``; a spatial covariate (cold/dense water) lowers it locally via
``speed_factor``. The achievable-speed map becomes an isotropic Riemannian sea:

    H(x) = I / (s_max · speed_factor(x))²      ("slow lens", demo_eikonal_fronts)

so slower water costs more Finsler distance. Anisotropic ``H`` (a directional
"speed polar" from wave added resistance) is the dominant effect for *surface*
vessels and is the next-session target — the frame supports it by overriding
``sea_tensor``.

The :class:`Glider` is a buoyancy-driven AUV: no propeller, slow (~0.3 in basin
units → genuinely *near* the Zermelo mild-wind boundary), and it advances by
choosing depth to ride favorable current layers. Its operating envelope (depth
bounds, glide-angle kinematics) is exposed as :class:`Constraint` objects that
both planners consume.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .constraints import Constraint, depth_envelope, glide_angle_limit


class Glider(eqx.Module):
    """Buoyancy-driven underwater glider (3D).

    Attributes:
        s_max: Through-water speed (basin units). Near current magnitudes by
            design — the physically interesting near-boundary regime.
        dim: Spatial dimension (3).
        z_min, z_max: Depth operating envelope (surface = 0, downward positive).
        glide_angle_max_deg: Steepest dive/climb path angle (from L/D limits).
            ``None`` disables the glide-angle constraint.
        epsilon: Randers causality margin passed to the metric.
    """

    s_max: float = eqx.field(static=True, default=0.85)
    dim: int = eqx.field(static=True, default=3)
    z_min: float = eqx.field(static=True, default=0.0)
    z_max: float = eqx.field(static=True, default=1.0)
    glide_angle_max_deg: float | None = eqx.field(static=True, default=60.0)
    epsilon: float = eqx.field(static=True, default=1e-5)

    def sea_tensor(self, medium, x: jax.Array) -> jax.Array:
        """Isotropic Riemannian sea ``H(x) = I / (s_max · speed_factor)²``."""
        s = self.s_max * medium.speed_factor(x)
        s2 = jnp.maximum(s * s, 1e-6)
        return jnp.eye(self.dim, dtype=x.dtype) / s2

    def constraints(self) -> list[Constraint]:
        """Operating-envelope constraints (depth bounds + glide angle).

        These are inequality / segment-wise / (optionally) time-aware, so they are
        enforced by the time-lifted planner's penalty layer; equality constraints
        (e.g. fixed-depth legs) would additionally pass into AVBD natively.
        """
        cons = list(depth_envelope(self.z_min, self.z_max, axis=2))
        if self.glide_angle_max_deg is not None:
            cons.append(
                glide_angle_limit(self.glide_angle_max_deg, horiz_axes=(0, 1), vert_axis=2)
            )
        return cons
