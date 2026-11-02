"""Synthetic one-day scenario: electricity price, ambient temperature, hydrogen
demand and short power-curtailment events. All profiles are made up for the demo."""
from __future__ import annotations

import numpy as np

DEMAND_KG_H = 9.0          # constant hydrogen offtake
TANK_CAP_KG = 120.0
TANK_START_KG = 60.0

# (start hour, end hour, power cap in kW): surprise curtailments the scheduler does not know about
CURTAILMENTS = [(7.2, 7.6, 300.0), (12.3, 12.7, 400.0), (15.3, 15.7, 350.0)]


def price_eur_mwh(t_h):
    """Day-ahead style price: morning and evening peaks, deep midday solar dip. Periodic in 24 h."""
    t = np.asarray(t_h, dtype=float) % 24.0
    return (70.0
            + 35.0 * np.exp(-((t - 19.0) / 2.5) ** 2)
            + 20.0 * np.exp(-((t - 8.0) / 2.0) ** 2)
            - 62.0 * np.exp(-((t - 13.0) / 3.0) ** 2))


def ambient_c(t_h: float) -> float:
    return 18.0 + 5.0 * np.sin(2.0 * np.pi * (t_h - 9.0) / 24.0)


def power_cap_kw(t_h: float, p_max_kw: float) -> float:
    for start, end, cap in CURTAILMENTS:
        if start <= t_h % 24.0 < end:
            return min(cap, p_max_kw)
    return p_max_kw