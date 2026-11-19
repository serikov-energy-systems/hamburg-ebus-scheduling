"""Uncertain energy consumption: scenarios from Hamburg winter temperatures, the cost of a
schedule over the scenarios, and value of the stochastic solution (VSS) and of perfect
information (EVPI) for the two-stage program."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .exact import solve_milp
from .model import Instance, Solution, cost_breakdown, expected_shortfall

NOMINAL_KWH_KM = 1.6   # consumption at the factor 1.0 (about 5 C, the winter median)


def consumption_per_km(temp_c):
    """Assumed law: 1.15 kWh/km above 15 C, plus 0.045 kWh/km for every degree below."""
    return 1.15 + 0.045 * np.maximum(0.0, 15.0 - np.asarray(temp_c, float))


def scenarios_from_temperatures(csv_path: str, edges=(0.0, 0.2, 0.5, 0.8, 0.95, 1.0)):
    """Consumption factors (relative to the nominal 1.6 kWh/km) and probabilities from the
    daily mean temperatures of the Hamburg heating seasons (November to March, 2015-2025).
    Scenarios are the mean factor of each quantile band, cold days last."""
    df = pd.read_csv(csv_path)
    f = consumption_per_km(df.t) / NOMINAL_KWH_KM
    f = np.sort(f)                      # warm to cold
    n = len(f)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = int(round(a * n)), int(round(b * n))
        out.append((float(f[lo:hi].mean()), (hi - lo) / n))
    return out


def total_cost(inst: Instance, sol: Solution, scen):
    """Deterministic cost plus the expected price of emergency energy over the scenarios."""
    facs = [s[0] for s in scen]
    probs = [s[1] for s in scen]
    es, per = expected_shortfall(inst, sol, facs, probs)
    base = cost_breakdown(inst, sol)
    return {**base, "exp_shortfall_kwh": es, "per_scenario_kwh": per,
            "exp_cost": base["cost"] + inst.params.c_resc * es}


def evpi_vss(inst: Instance, scen, time_limit=120.0, mip_gap=1e-3):
    """Two-stage program: RP (stochastic), EEV (plan for the mean consumption, evaluated over
    all scenarios) and WS (wait and see, one plan per scenario)."""
    mean_f = sum(f * p for f, p in scen)
    rp = solve_milp(inst, scen, time_limit=time_limit, mip_gap=mip_gap)
    ev = solve_milp(inst, [(mean_f, 1.0)], time_limit=time_limit, mip_gap=mip_gap)
    eev = solve_milp(inst, scen, time_limit=time_limit, mip_gap=mip_gap, fix=ev["first_stage"])
    ws = []
    for f, p in scen:
        r = solve_milp(inst, [(f, 1.0)], time_limit=time_limit, mip_gap=mip_gap)
        ws.append((f, p, r["objective"]))
    ws_val = sum(p * o for _, p, o in ws)
    return {
        "mean_factor": mean_f,
        "RP": rp["objective"], "RP_bound": rp["bound"], "RP_gap": rp.get("gap"), "RP_time": rp["time"],
        "EV_plan": ev["objective"], "EEV": eev["objective"], "EEV_time": eev["time"],
        "WS": ws_val, "WS_by_scenario": ws,
        "VSS": eev["objective"] - rp["objective"],
        "EVPI": rp["objective"] - ws_val,
        "rp_solution": rp["solution"], "ev_solution": ev["solution"],
        "rp_emergency_kwh": rp["emergency_kwh"], "eev_emergency_kwh": eev["emergency_kwh"],
    }
