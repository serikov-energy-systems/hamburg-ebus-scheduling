"""Experiments of the project. Usage: python scripts/run_experiments.py <stage> [...]

stages: solvers   ALNS (5 seeds), CP-SAT and HiGHS MILP on the instances, no-charger and all-terminal baselines
        margin    design consumption sweep, expected cost over the Hamburg winter scenarios
        evpi      two-stage MILP: RP, EEV, WS, VSS and EVPI on instance T
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebus.alns import alns, initial_solution  # noqa: E402
from ebus.cpsat import solve_cpsat  # noqa: E402
from ebus.exact import solve_milp  # noqa: E402
from ebus.model import Instance, Solution, cost_breakdown, min_fleet_unlimited  # noqa: E402
from ebus.stoch import evpi_vss, scenarios_from_temperatures, total_cost  # noqa: E402
from ebus.validate import check_solution  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)


def load(name):
    return Instance.from_json(json.loads((ROOT / "data/instances" / f"{name}.json").read_text()))


def num(x):
    return float(x) if x is not None else None


def save(name, obj):
    (RES / name).write_text(json.dumps(obj, indent=1, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))


def stage_solvers():
    out = {}
    budget = {"S": (30, 120, 200), "T": (30, 120, 200), "M": (60, 180, 300), "L": (240, 300, None)}
    for name in ["S", "T", "M", "L"]:
        inst = load(name)
        ta, tc, tm = budget[name]
        row = {"trips": inst.n, "terminals": inst.T, "service_km": float(inst.km.sum()),
               "unlimited_fleet": min_fleet_unlimited(inst)}
        # baselines
        st = initial_solution(inst, frozenset(range(inst.T)))
        row["greedy_all_terminals"] = cost_breakdown(inst, Solution(st.plans, st.sites))
        try:
            nocharge = alns(inst, time_limit=max(10, ta / 3), seed=0, allowed_sites=frozenset())
            row["depot_only"] = {**cost_breakdown(inst, nocharge), "valid": not check_solution(inst, nocharge)}
        except ValueError as e:
            row["depot_only"] = {"error": str(e)}
        runs = []
        for seed in range(5 if name != "L" else 3):
            log = []
            sol = alns(inst, time_limit=ta, seed=seed, log=log)
            cb = cost_breakdown(inst, sol)
            runs.append({"seed": seed, **cb, "valid": not check_solution(inst, sol),
                         "trace": [(round(t, 1), it, float(c)) for t, it, c in log[::max(1, len(log) // 40)]]})
            print(name, "alns", seed, round(cb["cost"], 1), cb["vehicles"], cb["charge_points"], flush=True)
        row["alns"] = runs
        r = solve_cpsat(inst, time_limit=tc)
        row["cpsat"] = {k: (num(v) if k in ("objective", "bound", "gap", "time") else v) for k, v in r.items() if k != "solution"}
        if r["solution"] is not None:
            row["cpsat"]["reeval"] = cost_breakdown(inst, r["solution"])
            row["cpsat"]["valid"] = not check_solution(inst, r["solution"])
        print(name, "cpsat", row["cpsat"].get("objective"), row["cpsat"].get("status"), flush=True)
        if tm:
            r = solve_milp(inst, time_limit=tm, mip_gap=1e-3)
            row["milp"] = {k: (num(v) if k in ("objective", "bound", "gap", "time") else v)
                           for k, v in r.items() if k not in ("solution", "first_stage")}
            if r["solution"] is not None:
                row["milp"]["reeval"] = cost_breakdown(inst, r["solution"])
                row["milp"]["valid"] = not check_solution(inst, r["solution"])
            print(name, "milp", row["milp"].get("objective"), row["milp"].get("gap"), flush=True)
        out[name] = row
        save("solvers.json", out)


def stage_margin():
    scen = scenarios_from_temperatures(str(ROOT / "data/hamburg_winter_temps.csv"))
    mean_f = sum(f * p for f, p in scen)
    facs = [0.9, mean_f, 1.1, 1.2, 1.3, 1.4, 1.5]
    out = {"scenarios": scen, "mean_factor": mean_f, "design": {}}
    limits = {"S": 30, "T": 30, "M": 60, "L": 240}
    for name in ["S", "T", "M", "L"]:
        inst = load(name)
        rows = []
        for fd in facs:
            sols = [alns(inst, time_limit=limits[name], seed=s, f=fd) for s in range(2)]
            tcs = [total_cost(inst, s, scen) for s in sols]
            k = int(np.argmin([t["exp_cost"] for t in tcs]))
            t = tcs[k]
            rows.append({"design_factor": fd, **{x: t[x] for x in ("vehicles", "sites", "charge_points", "deadhead_km", "cost", "exp_shortfall_kwh", "exp_cost")},
                         "per_scenario_kwh": t["per_scenario_kwh"]})
            print(name, round(fd, 2), round(t["cost"], 1), round(t["exp_shortfall_kwh"], 1), round(t["exp_cost"], 1), flush=True)
        out["design"][name] = rows
        save("margin.json", out)


def stage_evpi():
    inst = load("T")
    scen = scenarios_from_temperatures(str(ROOT / "data/hamburg_winter_temps.csv"))
    r = evpi_vss(inst, scen, time_limit=600, mip_gap=2e-3)
    out = {k: v for k, v in r.items() if k not in ("rp_solution", "ev_solution")}
    out["rp_cost"] = cost_breakdown(inst, r["rp_solution"])
    out["ev_cost"] = cost_breakdown(inst, r["ev_solution"])
    out["scenarios"] = scen
    save("evpi.json", out)
    print(json.dumps({k: out[k] for k in ("RP", "EEV", "WS", "VSS", "EVPI")}))


if __name__ == "__main__":
    for st in sys.argv[1:]:
        t0 = time.time()
        {"solvers": stage_solvers, "margin": stage_margin, "evpi": stage_evpi}[st]()
        print(st, "done in", round(time.time() - t0), "s", flush=True)
