# Multi-timescale control of an electrolyzer: a small digital-twin prototype

A compact, self-contained Python prototype showing how a **slow scheduling layer**
and a **fast model-predictive control (MPC) layer** can be coordinated around a
single electrolyzer model. It is a proof of concept for the research direction
*digital-twin modelling and multi-timescale optimization of electrolysis systems*.

> **All parameters are illustrative and not calibrated to any real stack.**
> The goal is the structure of the problem, not quantitative claims about a
> specific electrolyzer.

## What it does

| Layer | Timescale | Method | Job |
| --- | --- | --- | --- |
| Slow | 15 min steps, 24 h horizon, re-planned hourly | LP (CVXPY + HiGHS) | Shift production to cheap-electricity hours using a piecewise-linear efficiency map and a hydrogen tank |
| Fast | 1 min steps, 15 min horizon | QP-based MPC (CVXPY + Clarabel) | Track the schedule while keeping stack temperature in its window, limiting ramp-up, and obeying surprise power curtailments |
| Plant | 1 min | Nonlinear semi-empirical model | Polarization curve with temperature dependence, Faraday efficiency, lumped thermal model |

The fast MPC uses a **linearised** heat model, while the plant is nonlinear, so
there is deliberate model mismatch. The scheduler is **blind to temperature**:
it plans at nominal temperature and only gets feedback through the tank level at
each re-plan. That gap is exactly the interface question studied in the research
project (what information should pass between the layers?).

## Relation to the PhD proposal

This repository is a **starting point**, not an implementation of the proposed method.
It corresponds to the temperature-blind baseline (architecture A in the proposal):
the scheduler plans at nominal temperature and the fast MPC enforces the temperature
limit, with the only feedback being the tank level at each re-plan.

Not yet implemented:

- Architecture B (scheduler with a static thermal model) and C (feedback from the fast
  layer: updated parameters, state estimate, power envelope, tracking-error report)
- The envelope check with margin tightening and the safe fallback
- Renewable-supply forecasts and the cap on available power
- Calibration to measured data, so the model is a simulation testbed and not a validated digital twin

The numbers in the table below are from the synthetic day in `scenario.py` and are
indicative only. They were reproduced with `python run_demo.py` (cost per kg H2:
3.04 vs 2.44 EUR/kg, a 19.6% saving).

## Quick start

```bash
pip install -r requirements.txt
python electrolyzer.py     # prints the stack model at a few loads
python run_demo.py         # 24 h comparison, saves results/overview.png
python run_demo.py --hours 12 --no-plot
```

The full run takes roughly 10 seconds.

## Example result (24 h synthetic day)

Baseline = flat load chosen to meet average hydrogen demand, using the same fast layer.

| | Flat load | Scheduled |
| --- | --- | --- |
| Electricity cost (EUR) | 642 | 509 |
| Cost per kg H2 (EUR/kg) | 3.04 | 2.44 |
| Specific energy (kWh/kg) | 46.3 | 49.0 |
| Peak stack temperature (C) | 64.9 | 75.2 |

Scheduled operation is about **20% cheaper per kg of hydrogen**, even though it
uses more energy per kg (electrolyzers are less efficient at high load). The price of
flexibility is visible in the thermal panel: high-load periods push the stack to
its 75 C limit and the fast layer must use cooling to stay inside it.

## Files

- `electrolyzer.py`: stack and thermal model
- `scenario.py`: synthetic price, ambient temperature, demand, and curtailment events
- `scheduler.py`: slow-layer LP
- `fast_mpc.py`: fast-layer MPC
- `run_demo.py`: simulation, baseline comparison, plots

## Known limitations

- Parameters are illustrative; the model is not validated against measurements.
- The scheduler always keeps a minimum load (no off state or start-up costs).
- Degradation is not modelled.
- Temperature limits are soft constraints, so cooling chatters at the upper limit.
- One synthetic day and one price profile, so results are indicative only.

## Natural extensions

1. Calibrate the stack model to measured data and add uncertainty.
2. Replace the linearised thermal model with a reduced-order model derived from a detailed one.
3. Pass thermal headroom and ramp limits up to the scheduler (instead of power setpoints only).
4. Add degradation-aware costs and on/off decisions (MILP).
5. Run the fast layer in a hardware-in-the-loop setup.