"""Two-layer (multi-timescale) electrolyzer operation demo.

Slow layer : hourly re-planned 24 h LP schedule (15 min resolution), price-driven.
Fast layer : 1-minute MPC tracking the schedule under temperature and ramp limits.
Plant      : nonlinear electrolyzer + thermal model (mismatched w.r.t. the MPC model).

Compared against a flat-load baseline that runs the same fast layer but with a
constant power reference chosen to just meet average hydrogen demand.

Usage:  python run_demo.py [--hours 24] [--out results] [--no-plot]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from electrolyzer import Electrolyzer
from fast_mpc import FastMPC
from scenario import (DEMAND_KG_H, TANK_CAP_KG, TANK_START_KG, ambient_c,
                      power_cap_kw, price_eur_mwh)
from scheduler import efficiency_map, plan_schedule

DT_S = 60.0
SLOW_DT_H = 0.25
REPLAN_STEPS = 60          # re-plan every hour


def flat_power_kw(ely: Electrolyzer) -> float:
    """Constant power whose hydrogen output at nominal temperature equals demand."""
    lo, hi = ely.p.p_min_kw, ely.p.p_max_kw
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ely.operating_point(mid, ely.p.t_nom)["h2_kg_h"] < DEMAND_KG_H:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate(strategy: str, hours: int, ely: Electrolyzer, mpc: FastMPC, eff_map, seed: int = 0):
    rng = np.random.default_rng(seed)
    sp = ely.p
    steps = int(hours * 3600 / DT_S)
    dt_h = DT_S / 3600.0
    ref_per_slow = int(SLOW_DT_H * 3600 / DT_S)       # fast steps per 15-min plan step

    T, tank, p_prev = 65.0, TANK_START_KG, 0.5 * (sp.p_min_kw + sp.p_max_kw)
    p_flat = flat_power_kw(ely)
    plan, plan_start = None, 0

    log = {k: np.zeros(steps) for k in
           ("t_h", "price", "p_kw", "p_ref_kw", "cap_kw", "T", "qc_kw", "qh_kw", "h2_kg_h", "tank", "cost")}
    cost_cum, unmet, vented = 0.0, 0.0, 0.0

    for k in range(steps):
        t_h = k * dt_h
        if strategy == "scheduled" and k % REPLAN_STEPS == 0:
            plan = plan_schedule(ely, eff_map, tank, t_h, tank_target_kg=TANK_START_KG) * 1000.0
            plan_start = k

        if strategy == "scheduled":
            idx = np.minimum((k + np.arange(mpc.N) - plan_start) // ref_per_slow, len(plan) - 1)
            p_ref = plan[idx]
        else:
            p_ref = np.full(mpc.N, p_flat)

        cap = power_cap_kw(t_h, sp.p_max_kw)
        price = float(price_eur_mwh(t_h))
        T_meas = T + rng.normal(0.0, 0.1)
        p_cmd, qc, qh = mpc.solve(T_meas, p_prev, p_ref, cap, ambient_c(t_h), price)
        p_cmd = float(np.clip(p_cmd, sp.p_min_kw, min(cap, sp.p_max_kw)))
        qc = float(np.clip(qc, 0.0, sp.q_cool_max_kw))
        qh = float(np.clip(qh, 0.0, sp.q_heat_max_kw))

        # --- nonlinear plant ---
        op = ely.operating_point(p_cmd, T)
        T = ely.thermal_step(T, op["heat_kw"] + qh, qc, ambient_c(t_h), DT_S)
        tank += (op["h2_kg_h"] - DEMAND_KG_H) * dt_h
        if tank < 0.0:
            unmet, tank = unmet - tank, 0.0
        if tank > TANK_CAP_KG:
            vented, tank = vented + (tank - TANK_CAP_KG), TANK_CAP_KG
        e_mwh = (op["power_kw"] + qc / sp.cop_cool + qh) / 1000.0 * dt_h
        cost_cum += price * e_mwh
        p_prev = op["power_kw"]

        for key, val in (("t_h", t_h), ("price", price), ("p_kw", op["power_kw"]),
                         ("p_ref_kw", p_ref[0]), ("cap_kw", cap), ("T", T), ("qc_kw", qc), ("qh_kw", qh),
                         ("h2_kg_h", op["h2_kg_h"]), ("tank", tank), ("cost", cost_cum)):
            log[key][k] = val

    h2_kg = log["h2_kg_h"].sum() * dt_h
    energy_kwh = (log["p_kw"] + log["qc_kw"] / sp.cop_cool + log["qh_kw"]).sum() * dt_h
    summary = {
        "cost_eur": cost_cum,
        "h2_kg": h2_kg,
        "energy_mwh": energy_kwh / 1000.0,
        "kwh_per_kg": energy_kwh / h2_kg,
        "avg_price_paid": cost_cum / (energy_kwh / 1000.0),
        "tank_end_kg": log["tank"][-1],
        "tank_min_kg": log["tank"].min(),
        "unmet_kg": unmet,
        "vented_kg": vented,
        "T_max_c": log["T"].max(),
        "min_above_tmax": int((log["T"] > sp.t_max + 0.5).sum() * DT_S / 60),
        "cost_per_kg": cost_cum / h2_kg,
        "thermal_mwh": (log["qc_kw"] / sp.cop_cool + log["qh_kw"]).sum() * dt_h / 1000.0,
    }
    return log, summary


def print_summary(flat, sched):
    rows = [("Electricity cost (EUR)", "cost_eur", "{:.0f}"),
            ("Hydrogen produced (kg)", "h2_kg", "{:.0f}"),
            ("Cost per kg H2 (EUR/kg)", "cost_per_kg", "{:.2f}"),
            ("Energy used (MWh)", "energy_mwh", "{:.2f}"),
            ("Specific energy (kWh/kg)", "kwh_per_kg", "{:.1f}"),
            ("Average price paid (EUR/MWh)", "avg_price_paid", "{:.1f}"),
            ("Tank at end (kg)", "tank_end_kg", "{:.0f}"),
            ("Tank minimum (kg)", "tank_min_kg", "{:.0f}"),
            ("Unmet demand (kg)", "unmet_kg", "{:.1f}"),
            ("Peak stack temperature (C)", "T_max_c", "{:.1f}"),
            ("Minutes above T limit", "min_above_tmax", "{:d}"),
            ("Cooling + heater electricity (MWh)", "thermal_mwh", "{:.3f}")]
    print(f"\n{'':<32}{'Flat load':>12}{'Scheduled':>12}")
    for label, key, fmt in rows:
        print(f"{label:<32}{fmt.format(flat[key]):>12}{fmt.format(sched[key]):>12}")
    saving = 100.0 * (flat["cost_per_kg"] - sched["cost_per_kg"]) / flat["cost_per_kg"]
    print(f"\nSaving in cost per kg H2 vs flat load: {saving:.1f}%  "
          f"(specific energy {sched['kwh_per_kg'] - flat['kwh_per_kg']:+.1f} kWh/kg)")


def make_plot(flat_log, sched_log, ely, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = ely.p
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    t = sched_log["t_h"]

    ax[0].plot(t, sched_log["price"], color="tab:red", label="Price (EUR/MWh)")
    ax[0].set_ylabel("EUR/MWh")
    ax0b = ax[0].twinx()
    ax0b.plot(t, sched_log["p_ref_kw"], "--", color="tab:blue", label="Schedule (slow layer)")
    ax0b.plot(t, sched_log["p_kw"], color="tab:blue", label="Delivered power (scheduled)")
    ax0b.plot(flat_log["t_h"], flat_log["p_kw"], color="gray", label="Delivered power (flat)")
    ax0b.plot(t, sched_log["cap_kw"], ":", color="black", label="Power cap")
    ax0b.set_ylabel("Stack power (kW)")
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax0b.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=2)
    ax[0].set_ylim(0, 150)
    ax0b.set_ylim(0, 1350)
    ax[0].set_title("Slow layer follows the price; fast layer follows the schedule and the cap")

    ax[1].plot(t, sched_log["tank"], color="tab:blue", label="Scheduled")
    ax[1].plot(flat_log["t_h"], flat_log["tank"], color="gray", label="Flat load")
    ax[1].set_ylabel("H2 tank (kg)")
    ax[1].legend(fontsize=8)

    ax[2].plot(t, sched_log["T"], color="tab:blue", label="Scheduled")
    ax[2].plot(flat_log["t_h"], flat_log["T"], color="gray", label="Flat load")
    ax[2].axhline(sp.t_max, color="tab:red", ls="--", lw=1)
    ax[2].axhline(sp.t_min, color="tab:red", ls="--", lw=1)
    ax[2].set_ylabel("Stack temp (C)")
    ax2b = ax[2].twinx()
    ax2b.fill_between(t, sched_log["qc_kw"], color="tab:cyan", alpha=0.3)
    ax2b.set_ylabel("Cooling, scheduled (kW)")
    ax[2].legend(fontsize=8, loc="upper left")

    ax[3].plot(t, sched_log["cost"], color="tab:blue", label="Scheduled")
    ax[3].plot(flat_log["t_h"], flat_log["cost"], color="gray", label="Flat load")
    ax[3].set_ylabel("Cumulative cost (EUR)")
    ax[3].set_xlabel("Hour of day")
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    ely = Electrolyzer()
    eff_map = efficiency_map(ely)
    mpc = FastMPC(ely, dt_s=DT_S)
    print(f"Linearised heat model: q = {mpc.q0:.1f} kW + {mpc.q1:.3f} * P")

    t0 = time.time()
    flat_log, flat = simulate("flat", args.hours, ely, mpc, eff_map)
    sched_log, sched = simulate("scheduled", args.hours, ely, mpc, eff_map)
    print(f"Simulated 2 x {args.hours} h in {time.time() - t0:.0f} s")
    print_summary(flat, sched)

    if not args.no_plot:
        out = Path(args.out)
        out.mkdir(exist_ok=True)
        make_plot(flat_log, sched_log, ely, out / "overview.png")
        print(f"Figure saved to {out / 'overview.png'}")


if __name__ == "__main__":
    main()