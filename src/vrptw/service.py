"""HTTP service around the VRPTW solver with run telemetry.

    uvicorn vrptw.service:app --port 8010
    POST /solve   {"instance": "C101", "iterations": 2000, "seed": 1}   or inline coordinates, demands and windows
    GET  /metrics counts, latency percentiles and solution quality of the runs since the process started
    GET  /health

Every request records its solve time, total time, vehicles and distance; `VRPTW_LOG` (a file path) also
appends one JSON line per request. Limits on iterations and size keep the latency bounded.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import _alns_cpp, check, load, solve
from .instance import Instance, names

MAX_ITER = 20000
MAX_CUSTOMERS = 200
app = FastAPI(title="vrptw", version="0.1")
_runs: deque = deque(maxlen=1000)


class Inline(BaseModel):
    capacity: float
    x: list[float]
    y: list[float]
    demand: list[float]
    ready: list[float]
    due: list[float]
    service: list[float]


class Request(BaseModel):
    instance: str | None = Field(None, description="Solomon instance name, e.g. C101")
    inline: Inline | None = None
    iterations: int = Field(2000, ge=10, le=MAX_ITER)
    seed: int = 0
    customers: int | None = Field(None, ge=1, le=MAX_CUSTOMERS, description="use only the first k customers of a named instance")


def _instance(req: Request) -> Instance:
    if (req.instance is None) == (req.inline is None):
        raise HTTPException(422, "give either instance or inline")
    if req.instance is not None:
        if req.instance not in names():
            raise HTTPException(404, f"unknown instance {req.instance}")
        inst = load(req.instance)
        return inst.subset(req.customers) if req.customers else inst
    a = req.inline
    lens = {len(a.x), len(a.y), len(a.demand), len(a.ready), len(a.due), len(a.service)}
    if len(lens) != 1 or len(a.x) < 2 or len(a.x) - 1 > MAX_CUSTOMERS:
        raise HTTPException(422, f"arrays must have equal length between 2 and {MAX_CUSTOMERS + 1}")
    return Instance("inline", a.capacity, a.x, a.y, a.demand, a.ready, a.due, a.service)


@app.get("/health")
def health():
    return {"status": "ok", "kernel": "cpp" if _alns_cpp is not None else "python"}


@app.post("/solve")
def post_solve(req: Request):
    t0 = time.perf_counter()
    inst = _instance(req)
    backend = "cpp" if _alns_cpp is not None else "python"
    if backend == "python" and req.iterations * inst.n > 20000:
        raise HTTPException(422, "without the C++ kernel use at most 20000 / customers iterations")
    routes, dist, vehicles, ms = solve(inst, iterations=req.iterations, seed=req.seed, backend=backend)
    chk = check(inst, routes)
    rec = dict(instance=inst.name, customers=inst.n, iterations=req.iterations, backend=backend, vehicles=vehicles,
               distance=round(dist, 3), solve_ms=round(ms, 1), total_ms=round((time.perf_counter() - t0) * 1000, 1),
               feasible=chk["feasible"])
    _runs.append(rec)
    if os.environ.get("VRPTW_LOG"):
        with open(os.environ["VRPTW_LOG"], "a") as f:
            f.write(json.dumps(rec) + "\n")
    return {"routes": routes, **rec}


@app.get("/metrics")
def metrics():
    if not _runs:
        return {"runs": 0}
    lat = np.array([r["total_ms"] for r in _runs])
    return {"runs": len(_runs), "latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p95": float(np.percentile(lat, 95)),
            "latency_ms_max": float(lat.max()), "infeasible_runs": sum(1 for r in _runs if not r["feasible"]),
            "mean_vehicles": float(np.mean([r["vehicles"] for r in _runs])), "last": _runs[-1]}
