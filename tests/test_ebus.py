import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ebus.alns import alns, initial_solution
from ebus.cpsat import solve_cpsat
from ebus.exact import solve_milp
from ebus.model import (Instance, Params, Solution, charge_points, cost_breakdown, min_fleet_unlimited,
                        plan_block, simulate_plan)
from ebus.stoch import consumption_per_km, scenarios_from_temperatures, total_cost
from ebus.validate import check_solution

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return Instance.from_json(json.loads((ROOT / "data/instances" / f"{name}.json").read_text()))


def small(name="S", t0=6 * 60, t1=9 * 60):
    inst = load(name)
    sub = inst.trips[(inst.trips.start_min >= t0) & (inst.trips.start_min < t1)].reset_index(drop=True)
    return Instance("sub", sub, inst.term_name, inst.term_lat, inst.term_lon, inst.depot, inst.params)


def line_instance(n=6, km=40.0, params=None):
    """Shuttle between two terminals, one trip every 60 minutes, 40 km each: range matters."""
    trips = pd.DataFrame({
        "trip_id": [str(i) for i in range(n)], "route_short_name": "x",
        "start_min": [60.0 * i for i in range(n)], "end_min": [60.0 * i + 50 for i in range(n)],
        "length_km": km, "sterm": [i % 2 for i in range(n)], "eterm": [(i + 1) % 2 for i in range(n)],
    })
    return Instance("line", trips, ["A", "B"], np.array([53.55, 53.56]), np.array([10.0, 10.0]), 0,
                    params or Params())


def test_block_needs_charger_when_range_is_short():
    inst = line_instance(n=8, km=40.0)       # 8 trips x 40 km x 1.6 kWh = 512 kWh, window 280
    all_trips = list(range(inst.n))
    assert plan_block(inst, all_trips, frozenset()) is None
    plan = plan_block(inst, all_trips, frozenset({0, 1}))
    assert plan is not None
    sol = Solution([plan], frozenset({0, 1}))
    assert check_solution(inst, sol) == []


def test_higher_consumption_creates_shortfall():
    inst = line_instance(n=8, km=40.0)
    plan = plan_block(inst, list(range(8)), frozenset({0, 1}))
    short0, _ = simulate_plan(inst, plan, 1.0, frozenset({0, 1}))
    short1, _ = simulate_plan(inst, plan, 1.6, frozenset({0, 1}))
    assert short0 == 0
    assert short1 > 0


def test_unlimited_fleet_lower_bound():
    inst = small()
    sol = alns(inst, time_limit=3, seed=1)
    assert len(sol.plans) >= min_fleet_unlimited(inst)
    assert check_solution(inst, sol) == []


def test_alns_improves_on_greedy_start():
    inst = small("S", 6 * 60, 12 * 60)
    start = initial_solution(inst, frozenset(range(inst.T)))
    c0 = cost_breakdown(inst, Solution(start.plans, start.sites))["cost"]
    sol = alns(inst, time_limit=4, seed=0)
    assert cost_breakdown(inst, sol)["cost"] <= c0 + 1e-9


def test_exact_methods_agree_on_a_small_instance():
    inst = small("S", 6 * 60, 8 * 60 + 30)
    a = alns(inst, time_limit=3, seed=0)
    c = solve_cpsat(inst, time_limit=30)
    m = solve_milp(inst, time_limit=60)
    assert check_solution(inst, c["solution"]) == []
    assert check_solution(inst, m["solution"]) == []
    best = min(cost_breakdown(inst, s)["cost"] for s in (a, c["solution"], m["solution"]))
    assert cost_breakdown(inst, c["solution"])["cost"] <= cost_breakdown(inst, a)["cost"] + 1e-6 or c["gap"] > 0
    assert abs(cost_breakdown(inst, m["solution"])["cost"] - m["objective"]) < 1e-3
    assert best <= cost_breakdown(inst, a)["cost"] + 1e-9


def test_charge_points_count_overlapping_waits():
    inst = line_instance(n=4, km=10.0)
    p = [plan_block(inst, [i], frozenset({0, 1})) for i in range(4)]
    assert charge_points(inst, p, frozenset({0, 1})) == {0: 0, 1: 0}   # single-trip blocks never wait


def test_consumption_law_and_scenarios():
    assert consumption_per_km(15.0) == pytest.approx(1.15)
    assert consumption_per_km(5.0) == pytest.approx(1.6)
    sc = scenarios_from_temperatures(str(ROOT / "data/hamburg_winter_temps.csv"))
    assert sum(p for _, p in sc) == pytest.approx(1.0)
    fs = [f for f, _ in sc]
    assert fs == sorted(fs) and fs[0] < 1.0 < fs[-1]


def test_stochastic_cost_grows_with_cold():
    inst = line_instance(n=8, km=40.0)
    plan = plan_block(inst, list(range(8)), frozenset({0, 1}))
    sol = Solution([plan], frozenset({0, 1}))
    tc = total_cost(inst, sol, [(1.0, 0.5), (1.5, 0.5)])
    assert tc["per_scenario_kwh"][1] > tc["per_scenario_kwh"][0]
    assert tc["exp_cost"] > tc["cost"]


def test_service():
    from fastapi.testclient import TestClient
    from ebus.service import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert {i["name"] for i in c.get("/instances").json()} >= {"S", "T", "M", "L"}
    r = c.post("/schedule", json={"instance": "T", "method": "alns", "time_limit": 3, "scenarios": True})
    assert r.status_code == 200, r.text
    d = r.json()
    n_trips = sum(len(b["trips"]) for b in d["blocks"])
    assert n_trips == d["trips"]
    assert d["cost_eur_per_day"]["vehicles"] >= d["unlimited_range_fleet"]
    assert "scenarios" in d
    assert c.post("/schedule", json={"instance": "nope"}).status_code == 404
    assert c.post("/schedule", json={"method": "alns"}).status_code == 422
    assert c.post("/schedule", json={"instance": "T", "time_limit": 9999}).status_code == 422
