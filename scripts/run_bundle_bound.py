"""Scenario-bundle lower bound for RP on instance T. For a partition of the scenarios into bundles, RP >= sum_B P_B * RP_B, where RP_B is the two-stage
optimum of the bundle (first stage shared inside a bundle, probabilities normalised). Every RP_B is bounded below by the dual bound of HiGHS, so the sum
of P_B * dual bound is a rigorous lower bound; it is at least the wait-and-see value (singleton bundles).
    python scripts/run_bundle_bound.py 3,4 --time 3000      # bundle of scenario indices (0-4)"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ebus.exact import solve_milp
from ebus.model import Instance
from ebus.stoch import scenarios_from_temperatures
ap = argparse.ArgumentParser(); ap.add_argument("bundle"); ap.add_argument("--time", type=float, default=3000); ap.add_argument("--gap", type=float, default=1e-3)
a = ap.parse_args()
inst = Instance.from_json(json.loads((ROOT / "data/instances/T.json").read_text()))
scen = scenarios_from_temperatures(str(ROOT / "data/hamburg_winter_temps.csv"))
idx = [int(i) for i in a.bundle.split(",")]
PB = sum(scen[i][1] for i in idx)
sub = [(scen[i][0], scen[i][1] / PB) for i in idx]
t0 = time.time()
r = solve_milp(inst, sub, time_limit=a.time, mip_gap=a.gap)
out = {"bundle": idx, "P_B": PB, "objective": r["objective"], "bound": r["bound"], "seconds": time.time() - t0, "weighted_bound": PB * r["bound"], "weighted_obj": PB * r["objective"]}
(ROOT / "results" / f"bundle_{a.bundle.replace(',', '_')}.json").write_text(json.dumps(out, indent=1)); print(out)
