"""Semi-empirical electrolyzer stack model with a lumped thermal model.

Illustrative PEM-style stack: a polarization curve with temperature-dependent
ohmic and activation terms, a Faraday efficiency that drops at low load, and a
single-node thermal model (stack plus water inventory).

All parameter values are illustrative and NOT calibrated to any real stack.
The point is the structure: voltage, hydrogen yield and heat generation all
depend on load AND temperature, which is what couples the fast (thermal) and
slow (scheduling) layers of the control problem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FARADAY = 96485.0   # C/mol
M_H2 = 2.016e-3     # kg/mol
U_TN = 1.481        # thermoneutral voltage, V


@dataclass(frozen=True)
class StackParams:
    # Geometry
    n_cells: int = 250
    area_cm2: float = 2000.0
    i_min: float = 0.05            # A/cm^2, lowest stable current density
    i_max: float = 1.0             # A/cm^2, rated current density
    # Polarization curve: U = U_rev(T) + r(T)*i + s*log10(t(T)*i + 1)
    r0: float = 0.25               # ohm cm^2 at 25 C
    r_slope: float = 0.0016        # ohm cm^2 per K
    s: float = 0.25                # V
    t0: float = 100.0              # cm^2/A at 60 C
    t_slope: float = 0.02          # 1/K
    # Faraday efficiency: eta = f2 * i^2 / (i^2 + f1)
    f1: float = 2.5e-4
    f2: float = 0.99
    # Thermal model
    c_th: float = 1.5e7            # J/K, stack + water thermal mass
    h_loss: float = 1500.0         # W/K, loss to ambient
    t_min: float = 50.0            # C
    t_max: float = 75.0            # C
    t_nom: float = 65.0            # C, used for linearised / scheduling models
    q_cool_max_kw: float = 150.0   # kW thermal
    cop_cool: float = 8.0          # kW thermal per kW electric
    q_heat_max_kw: float = 60.0    # kW electric heater (COP 1), keeps the stack warm at low load
    # Operating power window valid across t_min..t_max
    p_min_kw: float = 80.0
    p_max_kw: float = 900.0


class Electrolyzer:
    """Stack model: maps (current density, temperature) to voltage, H2 and heat."""

    def __init__(self, params: StackParams | None = None):
        self.p = params or StackParams()

    # --- electrochemistry -------------------------------------------------
    def cell_voltage(self, i, T):
        p = self.p
        u_rev = 1.229 - 0.9e-3 * (T - 25.0)
        r = p.r0 - p.r_slope * (T - 25.0)
        t = p.t0 * np.exp(-p.t_slope * (T - 60.0))
        return u_rev + r * i + p.s * np.log10(t * i + 1.0)

    def faraday_efficiency(self, i):
        p = self.p
        return p.f2 * i**2 / (i**2 + p.f1)

    def stack_power_kw(self, i, T):
        p = self.p
        return p.n_cells * self.cell_voltage(i, T) * i * p.area_cm2 / 1000.0

    def h2_rate_kg_h(self, i, T):
        p = self.p
        current = i * p.area_cm2
        mol_s = self.faraday_efficiency(i) * p.n_cells * current / (2.0 * FARADAY)
        return mol_s * M_H2 * 3600.0

    def heat_kw(self, i, T):
        """Heat released in the stack (negative = endothermic at very low load)."""
        p = self.p
        current = i * p.area_cm2
        return p.n_cells * (self.cell_voltage(i, T) - U_TN) * current / 1000.0

    # --- inversion: power -> current density -------------------------------
    def current_density_for_power(self, p_kw: float, T: float) -> float:
        lo, hi = 1e-4, 2.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.stack_power_kw(mid, T) < p_kw:
                lo = mid
            else:
                hi = mid
        return float(np.clip(0.5 * (lo + hi), self.p.i_min, self.p.i_max))

    def operating_point(self, p_kw: float, T: float) -> dict:
        i = self.current_density_for_power(p_kw, T)
        return {
            "i": i,
            "voltage": float(self.cell_voltage(i, T)),
            "power_kw": float(self.stack_power_kw(i, T)),
            "h2_kg_h": float(self.h2_rate_kg_h(i, T)),
            "heat_kw": float(self.heat_kw(i, T)),
        }

    # --- thermal dynamics ---------------------------------------------------
    def thermal_step(self, T: float, heat_kw: float, cool_kw: float,
                     t_amb: float, dt_s: float) -> float:
        p = self.p
        dT = (1000.0 * heat_kw - p.h_loss * (T - t_amb) - 1000.0 * cool_kw) / p.c_th
        return T + dt_s * dT


if __name__ == "__main__":
    ely = Electrolyzer()
    print(f"{'load kW':>8} {'U cell V':>9} {'H2 kg/h':>8} {'kWh/kg':>7} {'heat kW':>8}")
    for p_kw in (80, 200, 400, 600, 800, 900):
        op = ely.operating_point(p_kw, ely.p.t_nom)
        print(f"{p_kw:8.0f} {op['voltage']:9.3f} {op['h2_kg_h']:8.2f} "
              f"{op['power_kw'] / op['h2_kg_h']:7.1f} {op['heat_kw']:8.1f}")