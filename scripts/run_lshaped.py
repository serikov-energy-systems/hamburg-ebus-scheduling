"""Solve the two-stage program of instance T with the L-shaped method; VSS and EVPI from exact values.

    python scripts/run_lshaped.py [--small N]     # --small N keeps the first N trips (check against the full MILP)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from joblib import Parallel, delayed  # noqa: E402

from ebus.exact import solve_milp  # noqa: E402
from ebus.lshaped import lshaped  # noqa: E402
from ebus.model import Instance  # noqa: E402
from ebus.stoch import scenarios_from_temperatures  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--small", type=int, default=0)
ap.add_argument("--time", type=float, default=7200)
ap.add_argument("--master-time", type=float, default=900)
ap.add_argument("--stress", type=float, default=1.0, help="multiply the consumption factors (forces emergency energy in small tests)")
ap.add_argument("--battery-scale", type=float, default=1.0, help="scale the usable battery energy (small tests with real recourse)")
ap.add_argument("--lp-iters", type=int, default=0, help="Benders cuts at LP-relaxed master optima before the integer master (phase 1)")
ap.add_argument("--check", action="store_true", help="also solve the full two-stage MILP (small instances only)")
a = ap.parse_args()

d = json.loads((ROOT / "data" / "instances" / "T.json").read_text())
if a.small:
    d["trips"] = sorted(d["trips"], key=lambda t: t["start_min"])[:a.small]
inst = Instance.from_json(d)
if a.battery_scale != 1.0:
    inst = inst.with_params(cap_kwh=inst.params.cap_kwh * a.battery_scale)
scen = scenarios_from_temperatures(str(ROOT / "data" / "hamburg_winter_temps.csv"))
scen = [(f * a.stress, p) for f, p in scen]
mean_f = sum(f * p for f, p in scen)
print("trips", inst.n, "scenarios", [(round(f, 3), round(p, 3)) for f, p in scen], flush=True)

jobs = [[(mean_f, 1.0)]] + [[(f, 1.0)] for f, _ in scen]
t0 = time.time()
sols = Parallel(n_jobs=len(jobs))(delayed(solve_milp)(inst, j, time_limit=1200, mip_gap=1e-3) for j in jobs)
ev, ws = sols[0], sols[1:]
print("deterministic solves", round(time.time() - t0), "s; EV", ev["objective"], "WS", [round(w["objective"], 1) for w in ws], flush=True)
seeds = [s["first_stage"] for s in sols]
r = lshaped(inst, scen, seeds=seeds, time_limit=a.time, master_time=a.master_time, lp_iters=a.lp_iters)
ws_val = sum(p * w["objective"] for (f, p), w in zip(scen, ws))
# EEV: first stage of the expected-value plan, exact recourse
x_ev = None
P = r["problem"]
from ebus.lshaped import Problem  # noqa: E402,F401
import numpy as np  # noqa: E402
x = np.zeros(P.nfirst)
M = P.M
fs = ev["first_stage"]
for k, key in enumerate(M["D"]): x[M["xd"] + k] = 1.0 if key in fs["xd"] else 0.0
for k, key in enumerate(M["V"]): x[M["xv"] + k] = 1.0 if key in fs["xv"] else 0.0
for i in range(M["n"]):
    x[M["ys"] + i] = 1.0 if i in fs["ys"] else 0.0
    x[M["ye"] + i] = 1.0 if i in fs["ye"] else 0.0
for k in range(inst.T):
    x[M["o"] + k] = 1.0 if k in fs["o"] else 0.0
    x[M["K"] + k] = fs["K"].get(k, 0)
c1, qs, _ = P.total(x)
eev = c1 + sum(qs)
out = {"trips": inst.n, "scenarios": scen, "WS": ws_val, "EV_deterministic": ev["objective"], "EEV": eev,
       "RP_upper": r["RP_ub"], "RP_lower": r["RP_lb"], "gap": r["gap"],
       "VSS_range": [eev - r["RP_ub"], eev - r["RP_lb"]], "EVPI_range": [r["RP_lb"] - ws_val, r["RP_ub"] - ws_val],
       "first_stage_cost_RP": r["first_stage_cost"], "recourse_by_scenario_RP": r["recourse_by_scenario"],
       "iterations": len(r["history"]), "history": r["history"]}
if a.check:
    full = solve_milp(inst, scen, time_limit=1800, mip_gap=1e-4)
    out["full_milp_RP"] = full["objective"]
    out["full_milp_bound"] = full["bound"]
name = f"lshaped_small{a.small}.json" if a.small else "lshaped_T.json"
(ROOT / "results" / name).write_text(json.dumps(out, indent=2, default=float))
print(json.dumps({k: v for k, v in out.items() if k not in ("history", "scenarios")}, indent=2, default=float))
