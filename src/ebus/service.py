"""HTTP service: schedule electric bus blocks and charging terminals for a bundled Hamburg
instance or for trips posted by the caller."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .alns import alns
from .cpsat import solve_cpsat
from .exact import solve_milp
from .model import Instance, Params, Solution, charge_points, cost_breakdown, min_fleet_unlimited
from .stoch import scenarios_from_temperatures, total_cost

ROOT = Path(os.environ.get("EBUS_ROOT", Path(__file__).resolve().parents[2]))
INSTANCE_DIR = ROOT / "data" / "instances"
TEMPS = ROOT / "data" / "hamburg_winter_temps.csv"
MAX_TRIPS = 3000
MAX_SECONDS = 120.0

app = FastAPI(title="Electric bus scheduling", version="0.1.0")


class Terminal(BaseModel):
    name: str
    lat: float
    lon: float


class TripIn(BaseModel):
    trip_id: str
    route_short_name: str = ""
    start_min: float
    end_min: float
    length_km: float = Field(gt=0)
    sterm: int
    eterm: int


class InstanceIn(BaseModel):
    name: str = "custom"
    terminals: list[Terminal]
    depot: int
    trips: list[TripIn]


class ScheduleRequest(BaseModel):
    instance: str | None = Field(None, description="name of a bundled instance (S, T, M, L)")
    custom: InstanceIn | None = None
    method: str = Field("alns", pattern="^(alns|cpsat|milp)$")
    time_limit: float = Field(20.0, gt=0, le=MAX_SECONDS)
    consumption_factor: float = Field(1.0, ge=0.5, le=2.0, description="design consumption relative to 1.6 kWh/km")
    seed: int = 0
    scenarios: bool = Field(False, description="also report the shortfall over the Hamburg winter scenarios")
    cost_overrides: dict[str, float] = Field(default_factory=dict, description="override fields of the parameter set")


def _bundled() -> dict:
    return {p.stem: p for p in sorted(INSTANCE_DIR.glob("*.json"))}


def _load(req: ScheduleRequest) -> Instance:
    base_params = None
    if req.cost_overrides:
        try:
            base_params = Params(**req.cost_overrides)
        except TypeError as e:
            raise HTTPException(422, f"unknown parameter: {e}")
    if req.custom is not None:
        c = req.custom
        if len(c.trips) > MAX_TRIPS:
            raise HTTPException(422, f"at most {MAX_TRIPS} trips")
        if not (0 <= c.depot < len(c.terminals)):
            raise HTTPException(422, "depot must index a terminal")
        for t in c.trips:
            if not (0 <= t.sterm < len(c.terminals) and 0 <= t.eterm < len(c.terminals)):
                raise HTTPException(422, f"trip {t.trip_id}: terminal index out of range")
            if t.end_min <= t.start_min:
                raise HTTPException(422, f"trip {t.trip_id}: ends before it starts")
        d = c.model_dump()
        d["trips"] = sorted(d["trips"], key=lambda t: (t["start_min"], t["end_min"]))
        return Instance.from_json(d, base_params)
    if req.instance is None:
        raise HTTPException(422, "give a bundled instance name or a custom instance")
    files = _bundled()
    if req.instance not in files:
        raise HTTPException(404, f"unknown instance, choose from {sorted(files)}")
    return Instance.from_json(json.loads(files[req.instance].read_text()), base_params)


@app.get("/health")
def health():
    return {"status": "ok", "instances": sorted(_bundled())}


@app.get("/instances")
def instances():
    out = []
    for name, path in _bundled().items():
        d = json.loads(path.read_text())
        out.append({"name": name, "trips": len(d["trips"]), "terminals": len(d["terminals"]),
                    "lines": sorted({t["route_short_name"] for t in d["trips"]})})
    return out


@app.post("/schedule")
def schedule(req: ScheduleRequest):
    inst = _load(req)
    if req.method == "alns":
        sol = alns(inst, time_limit=req.time_limit, seed=req.seed, f=req.consumption_factor)
        info = {"method": "alns", "optimality_gap": None}
    elif req.method == "cpsat":
        r = solve_cpsat(inst, time_limit=req.time_limit, f=req.consumption_factor)
        sol = r["solution"]
        info = {"method": "cpsat", "status": r["status"], "optimality_gap": r["gap"]}
    else:
        if inst.n > 400:
            raise HTTPException(422, "the MILP is limited to 400 trips; use alns or cpsat")
        r = solve_milp(inst, [(req.consumption_factor, 1.0)], time_limit=req.time_limit)
        sol = r["solution"]
        info = {"method": "milp", "status": r["message"], "optimality_gap": r.get("gap")}
    if sol is None:
        raise HTTPException(422, "no schedule found within the time limit")
    cb = cost_breakdown(inst, sol)
    cp = charge_points(inst, sol.plans, sol.sites)
    ids = inst.trips.trip_id.tolist()
    out = {
        **info,
        "instance": inst.name,
        "trips": inst.n,
        "unlimited_range_fleet": min_fleet_unlimited(inst) if inst.n <= 3000 else None,
        "cost_eur_per_day": {k: float(v) for k, v in cb.items()},
        "charging_terminals": [{"terminal": inst.term_name[s], "charge_points": int(k)} for s, k in cp.items() if k > 0],
        "depot": inst.term_name[inst.depot],
        "blocks": [
            {"trips": [ids[i] for i in pl.trips], "connections": pl.conn, "deadhead_km": float(pl.dh_km)}
            for pl in sol.plans
        ],
    }
    if req.scenarios and TEMPS.exists():
        tc = total_cost(inst, sol, scenarios_from_temperatures(str(TEMPS)))
        out["scenarios"] = {"expected_shortfall_kwh": float(tc["exp_shortfall_kwh"]),
                            "per_scenario_kwh": [float(x) for x in tc["per_scenario_kwh"]]}
    return out
