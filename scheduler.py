"""Slow layer: rolling-horizon day-ahead scheduling as a linear program.

The nonlinear hydrogen-vs-power curve at nominal temperature is replaced by a
concave piecewise-linear upper bound, so the whole scheduling problem is an LP
(no binaries). Limitations: minimum load is always enforced (no off state), and
the scheduler is blind to stack temperature, which is exactly the information
the fast layer has and the scheduler lacks.
"""
from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

from electrolyzer import Electrolyzer
from scenario import DEMAND_KG_H, TANK_CAP_KG, price_eur_mwh


def efficiency_map(ely: Electrolyzer, n_seg: int = 6):
    """Chord lines h2 <= a_j * P + b_j (P in MW, h2 in kg/h) through the model at t_nom."""
    sp = ely.p
    p_pts = np.linspace(sp.p_min_kw, sp.p_max_kw, n_seg + 1)
    h_pts = np.array([ely.operating_point(p, sp.t_nom)["h2_kg_h"] for p in p_pts])
    p_mw = p_pts / 1000.0
    a = np.diff(h_pts) / np.diff(p_mw)
    b = h_pts[:-1] - a * p_mw[:-1]
    if np.any(np.diff(a) > 1e-6):
        warnings.warn("H2(P) is not concave on the chosen window; the LP bound is not tight.")
    return a, b


def plan_schedule(ely: Electrolyzer, eff_map, tank_kg: float, t_start_h: float,
                  tank_target_kg: float, horizon_steps: int = 96, dt_h: float = 0.25):
    """Minimise electricity cost over the horizon. Returns stack power plan in MW (one value per 15 min)."""
    a, b = eff_map
    sp = ely.p
    t = t_start_h + dt_h * np.arange(horizon_steps)
    price = price_eur_mwh(t)

    p = cp.Variable(horizon_steps)            # MW
    h2 = cp.Variable(horizon_steps)           # kg/h
    m = cp.Variable(horizon_steps + 1)        # kg

    cons = [m[0] == tank_kg,
            m[1:] == m[:-1] + dt_h * (h2 - DEMAND_KG_H),
            m >= 0, m <= TANK_CAP_KG,
            m[-1] >= tank_target_kg,
            p >= sp.p_min_kw / 1000.0, p <= sp.p_max_kw / 1000.0,
            h2 >= 0]
    cons += [h2 <= a[j] * p + b[j] for j in range(len(a))]

    cost = cp.sum(cp.multiply(price, p)) * dt_h
    smooth = 0.5 * cp.norm1(cp.diff(p))       # tiny tie-breaker, discourages needless ramping
    prob = cp.Problem(cp.Minimize(cost + smooth), cons)
    prob.solve(solver=cp.HIGHS)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Scheduling LP failed: {prob.status}")
    return np.asarray(p.value).ravel()