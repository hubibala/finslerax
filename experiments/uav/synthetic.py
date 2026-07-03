"""Synthetic fleet generator — the U2 bridge from the power model to the estimator.

Flights are waypoint missions (vertical climb, cruise legs with altitude
changes, vertical descent) flown at constant velocity per leg through a uniform
or spatially varying horizontal wind. Segment energies come from the §2 power
model in closed form, so ground truth is exact:

    P = c₀ + c₁·v_air + c₂·v_air² + c_ind/(v_air + 1) + (mg/η↑)·ż₊ − η↓·mg·ż₋

With ``c₁ = c_ind = 0`` (the default) the linear segment model of
``medium.py`` is *exact* for uniform wind — the estimator must recover
``k = mg(1/η↑ + η↓)/2`` and ``b = −2c₂·W`` to solver precision. The other
knobs inject what real data adds: measurement noise, per-flight mass jitter
(the per-log ``k`` spread), linear-drag / induced-power model mismatch (which
the even bracket must absorb without leaking into the odd part).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

Wind = Callable[[np.ndarray], np.ndarray]  # (2,) position xy -> (2,) wind m/s


@dataclass(frozen=True)
class PowerModel:
    """Instantaneous electrical power of a multirotor (SI units)."""

    mass: float = 1.5       # kg
    g: float = 9.81         # m/s²
    hover_w: float = 250.0  # c₀ [W]
    lin_w: float = 0.0      # c₁ [W/(m/s)] — breaks the exact wind mapping
    quad_w: float = 0.7     # c₂ [W/(m/s)²]
    induced_w: float = 0.0  # c_ind [W·(m/s)] — mismatch term c_ind/(v_air+1)
    eta_up: float = 0.7     # climb efficiency
    eta_dn: float = 0.2     # descent recovery fraction (multirotors: small)

    @property
    def k(self) -> float:
        """The climb-cost ledger: odd coefficient of dz [J/m]."""
        return self.mass * self.g * (1.0 / self.eta_up + self.eta_dn) / 2.0

    @property
    def s_even(self) -> float:
        """Symmetric vertical overhead: even coefficient of |dz| [J/m]."""
        return self.mass * self.g * (1.0 / self.eta_up - self.eta_dn) / 2.0

    def segment_energy(self, v: np.ndarray, dt: float, wind_xy: np.ndarray) -> float:
        """Exact energy of a constant-velocity segment (ground velocity ``v``)."""
        v_air = float(np.linalg.norm(v - np.array([wind_xy[0], wind_xy[1], 0.0])))
        e = (
            self.hover_w * dt
            + self.lin_w * v_air * dt
            + self.quad_w * v_air**2 * dt
            + self.induced_w * dt / (v_air + 1.0)
        )
        dz = v[2] * dt
        mg = self.mass * self.g
        e += mg / self.eta_up * max(dz, 0.0) - self.eta_dn * mg * max(-dz, 0.0)
        return e


@dataclass(frozen=True)
class FleetTruth:
    """Everything the generator knows and the estimator must recover."""

    model: PowerModel
    k_by_log: pd.Series                       # per-log ledger (mass jitter included)
    wind_fn: Wind                             # true horizontal wind field
    uniform_wind: np.ndarray | None = None  # (2,) if the field is uniform
    wind_form: np.ndarray | None = None     # b = −2c₂·W if uniform
    params: dict = field(default_factory=dict)


def make_vortex(center: tuple, strength: float, radius: float) -> Wind:
    """Solenoidal swirl: tangential speed ``strength·2Rr/(r²+R²)``, peak at ``r=R``."""
    c = np.asarray(center, float)

    def wind(xy: np.ndarray) -> np.ndarray:
        d = np.asarray(xy, float) - c
        r2 = float(d @ d)
        coef = strength * 2.0 * radius / (r2 + radius**2)
        return coef * np.array([-d[1], d[0]])

    return wind


def synthesize_fleet(
    n_flights: int = 20,
    *,
    model: PowerModel | None = None,
    wind: object = (0.0, 0.0),
    n_legs: tuple = (4, 8),
    seg_dt: tuple = (3.0, 5.0),
    speed_jitter: float = 0.15,
    noise: float = 0.02,
    mass_cv: float = 0.0,
    wind_meas_noise: float = 0.3,
    airframe: str = "quad_a",
    seed: int = 0,
) -> tuple[pd.DataFrame, FleetTruth]:
    """Generate a fleet of waypoint missions → (segment table, ground truth).

    ``wind`` is a uniform ``(Wx, Wy)`` tuple or a callable ``xy -> (2,)`` field.
    ``speed_jitter`` wobbles each segment's velocity around the leg setpoint —
    without it every segment of a leg is a scalar multiple of one feature row
    and the per-log design is rank-deficient (a real controller never flies
    that cleanly). ``noise`` scales Gaussian energy noise by the hover energy
    per segment; ``mass_cv`` jitters each flight's mass (⇒ per-log ledger
    spread). EKF wind columns are the true wind at the segment midpoint plus
    ``wind_meas_noise``.
    """
    model = PowerModel() if model is None else model
    rng = np.random.default_rng(seed)
    if callable(wind):
        wind_fn, uniform = wind, None
    else:
        w = np.asarray(wind, float)
        wind_fn, uniform = (lambda xy: w), w

    rows, k_by_log = [], {}
    for i in range(n_flights):
        log_id = f"synth-{i:03d}"
        mass_i = model.mass * max(0.5, 1.0 + mass_cv * rng.standard_normal())
        m_i = PowerModel(
            mass=mass_i, g=model.g, hover_w=model.hover_w, lin_w=model.lin_w,
            quad_w=model.quad_w, induced_w=model.induced_w,
            eta_up=model.eta_up, eta_dn=model.eta_dn,
        )
        k_by_log[log_id] = m_i.k

        # --- waypoint mission: climb, cruise legs, descend -------------------
        legs = []
        alt = float(rng.uniform(25.0, 60.0))
        legs.append((np.array([0.0, 0.0, rng.uniform(1.5, 3.0)]), None, alt))
        for _ in range(int(rng.integers(n_legs[0], n_legs[1] + 1))):
            th = rng.uniform(0.0, 2.0 * np.pi)
            dist = rng.uniform(60.0, 240.0)
            vh = rng.uniform(8.0, 16.0)
            new_alt = float(np.clip(alt + rng.uniform(-15.0, 15.0), 15.0, 80.0))
            T = dist / vh
            vel = np.array([vh * np.cos(th), vh * np.sin(th), (new_alt - alt) / T])
            legs.append((vel, T, None))
            alt = new_alt
        legs.append((np.array([0.0, 0.0, -rng.uniform(1.5, 2.5)]), None, 0.0))

        # --- slice legs into constant-velocity segments ----------------------
        pos, t = np.zeros(3), 0.0
        for vel, T, target_alt in legs:
            if T is None:  # vertical leg: duration from altitude target
                T = abs(target_alt - pos[2]) / max(abs(vel[2]), 1e-9)
            t_leg = 0.0
            while t_leg + seg_dt[1] <= T:
                sdt = float(rng.uniform(*seg_dt))
                v_seg = vel * (1.0 + speed_jitter * rng.uniform(-1.0, 1.0))
                delta = v_seg * sdt
                mid = pos + 0.5 * delta
                w_mid = wind_fn(mid[:2])
                e = m_i.segment_energy(v_seg, sdt, w_mid)
                e += noise * model.hover_w * sdt * rng.standard_normal()
                w_meas = w_mid + wind_meas_noise * rng.standard_normal(2)
                rows.append({
                    "log_id": log_id, "airframe": airframe, "t0": t, "dt": sdt,
                    "dx": delta[0], "dy": delta[1], "dz": delta[2],
                    "length": float(np.linalg.norm(delta)), "energy": e,
                    "mid_x": mid[0], "mid_y": mid[1], "mid_z": mid[2],
                    "wind_x": w_meas[0], "wind_y": w_meas[1],
                })
                pos, t, t_leg = pos + delta, t + sdt, t_leg + sdt

    df = pd.DataFrame(rows)
    truth = FleetTruth(
        model=model,
        k_by_log=pd.Series(k_by_log, name="k"),
        wind_fn=wind_fn,
        uniform_wind=uniform,
        wind_form=None if uniform is None else -2.0 * model.quad_w * uniform,
        params={"noise": noise, "mass_cv": mass_cv, "seed": seed},
    )
    return df, truth
