"""Solomon VRPTW instances, an independent feasibility checker and the best known solutions (SINTEF)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "solomon"


@dataclass
class Instance:
    name: str
    capacity: float
    x: list
    y: list
    demand: list
    ready: list
    due: list
    service: list

    @property
    def n(self) -> int:
        return len(self.x) - 1

    def dist(self, a: int, b: int) -> float:
        return math.hypot(self.x[a] - self.x[b], self.y[a] - self.y[b])

    def args(self):
        return (self.x, self.y, self.demand, self.ready, self.due, self.service, self.capacity)

    def subset(self, k: int) -> "Instance":
        """The depot and the first k customers (small instances for the exact methods)."""
        s = slice(0, k + 1)
        return Instance(f"{self.name}-{k}", self.capacity, self.x[s], self.y[s], self.demand[s], self.ready[s],
                        self.due[s], self.service[s])


def load(name: str, path: Path | None = None) -> Instance:
    p = (path or DATA) / f"{name}.txt"
    lines = [ln.split() for ln in p.read_text().splitlines() if ln.strip()]
    cap = None
    rows = []
    for i, ln in enumerate(lines):
        if ln[:2] == ["NUMBER", "CAPACITY"]:
            cap = float(lines[i + 1][1])
        elif len(ln) == 7 and ln[0].isdigit():
            rows.append([float(v) for v in ln])
    rows.sort(key=lambda r: r[0])
    col = lambda k: [r[k] for r in rows]
    return Instance(name, cap, col(1), col(2), col(3), col(4), col(5), col(6))


def check(inst: Instance, routes, tol: float = 1e-6) -> dict:
    """Independent check: every customer once, capacity, time windows, depot due date. Returns the distance."""
    seen = sorted(c for r in routes for c in r)
    problems = []
    if seen != list(range(1, inst.n + 1)):
        problems.append("customers not served exactly once")
    total = 0.0
    for r in routes:
        load_, t, prev = 0.0, 0.0, 0
        for v in r:
            load_ += inst.demand[v]
            t = max(t + inst.dist(prev, v), inst.ready[v])
            if t > inst.due[v] + tol:
                problems.append(f"late at {v}")
            t += inst.service[v]
            total += inst.dist(prev, v)
            prev = v
        total += inst.dist(prev, 0)
        if t + inst.dist(prev, 0) > inst.due[0] + tol:
            problems.append("late at the depot")
        if load_ > inst.capacity + tol:
            problems.append("capacity exceeded")
    return dict(feasible=not problems, problems=problems, vehicles=len(routes), distance=total)


def bks() -> dict:
    return json.loads((DATA / "bks_sintef.json").read_text())


def names() -> list:
    return sorted(p.stem for p in DATA.glob("*.txt"))
