"""Fast layer: 1-minute MPC that tracks the schedule while respecting stack temperature.

Internal model: linear thermal dynamics with heat generation linearised in
power around the nominal temperature. The real plant is nonlinear, so there is
deliberate model mismatch. Temperature limits are soft (slack) so the problem
stays feasible; cooling and heating cost electricity, so they are only used when needed.

Solver note: the thermal dynamics mix very different magnitudes (W vs K), which
makes first-order OSQP hit its iteration limit; the interior-point solver
Clarabel solves each 15-step QP in about a millisecond.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from electrolyzer import Electrolyzer


class FastMPC:
    def __init__(self, ely: Electrolyzer, dt_s: float = 60.0, horizon: int = 15,
                 ramp_up_kw_per_step: float = 150.0):
        self.ely, self.dt, self.N = ely, dt_s, horizon
        sp = ely.p

        # Linearise heat generation q(P) ~ q0 + q1*P at t_nom
        p_grid = np.linspace(sp.p_min_kw, sp.p_max_kw, 30)
        heat = np.array([ely.operating_point(p, sp.t_nom)["heat_kw"] for p in p_grid])
        self.q1, self.q0 = np.polyfit(p_grid, heat, 1)          # kW per kW, kW

        N = horizon
        self.x = cp.Variable(N)                    # stack power, MW
        self.qc = cp.Variable(N, nonneg=True)      # cooling, MW thermal
        self.qh = cp.Variable(N, nonneg=True)      # electric heater, MW thermal
        self.T = cp.Variable(N + 1)                # stack temperature, C
        self.s = cp.Variable(N + 1, nonneg=True)   # soft-constraint slack, K

        self.T0 = cp.Parameter()
        self.x_prev = cp.Parameter()
        self.x_ref = cp.Parameter(N)
        self.x_cap = cp.Parameter(nonneg=True)
        self.t_amb = cp.Parameter()
        self.price = cp.Parameter(nonneg=True)     # EUR/MWh

        a = dt_s / sp.c_th
        cons = [self.T[0] == self.T0]
        for k in range(N):
            cons.append(
                self.T[k + 1] == self.T[k] + a * (
                    1000.0 * self.q0 + 1e6 * self.q1 * self.x[k]
                    - sp.h_loss * self.T[k] + sp.h_loss * self.t_amb
                    - 1e6 * self.qc[k] + 1e6 * self.qh[k]))
        cons += [self.T[1:] <= sp.t_max + self.s[1:],
                 self.T[1:] >= sp.t_min - self.s[1:],
                 self.qc <= sp.q_cool_max_kw / 1000.0,
                 self.qh <= sp.q_heat_max_kw / 1000.0,
                 self.x >= sp.p_min_kw / 1000.0,
                 self.x <= sp.p_max_kw / 1000.0,
                 self.x <= self.x_cap,
                 self.x[0] - self.x_prev <= ramp_up_kw_per_step / 1000.0]
        if N > 1:
            cons.append(self.x[1:] - self.x[:-1] <= ramp_up_kw_per_step / 1000.0)

        dt_h = dt_s / 3600.0
        x_all = cp.hstack([self.x_prev, self.x])
        cost = (50.0 * cp.sum_squares(self.x - self.x_ref)
                + 5.0 * cp.sum_squares(cp.diff(x_all))
                + self.price * (cp.sum(self.qc) / sp.cop_cool + cp.sum(self.qh)) * dt_h
                + 1e3 * cp.sum_squares(self.s))
        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def solve(self, T_meas, p_prev_kw, p_ref_kw, cap_kw, t_amb, price):
        sp = self.ely.p
        self.T0.value = float(T_meas)
        self.x_prev.value = p_prev_kw / 1000.0
        self.x_ref.value = np.asarray(p_ref_kw) / 1000.0
        # persistence forecast: the cap seen now is assumed to hold over the horizon
        self.x_cap.value = max(cap_kw, sp.p_min_kw) / 1000.0
        self.t_amb.value = float(t_amb)
        self.price.value = float(max(price, 0.0))
        try:
            self.prob.solve(solver=cp.CLARABEL)
            ok = self.prob.status in ("optimal", "optimal_inaccurate") and self.x.value is not None
        except cp.SolverError:
            ok = False
        if not ok:   # conservative fallback: hold power, full cooling if hot
            cool = sp.q_cool_max_kw if T_meas > sp.t_max - 2 else 0.0
            heat = sp.q_heat_max_kw if T_meas < sp.t_min + 2 else 0.0
            return min(p_prev_kw, cap_kw), cool, heat
        return (float(self.x.value[0]) * 1000.0, float(self.qc.value[0]) * 1000.0,
                float(self.qh.value[0]) * 1000.0)