"""Independent check of a schedule: every trip once, blocks in time order with feasible
connections, battery between the reserve floor and the ceiling at the design consumption."""
from __future__ import annotations

from .model import Instance, Solution, simulate_plan


def check_solution(inst: Instance, sol: Solution, f: float = 1.0, tol: float = 1e-6) -> list:
    """Return the list of violations (empty when the schedule is valid)."""
    errs = []
    seen = [x for pl in sol.plans for x in pl.trips]
    if sorted(seen) != list(range(inst.n)):
        missing = sorted(set(range(inst.n)) - set(seen))
        dup = sorted({x for x in seen if seen.count(x) > 1})
        errs.append(f"trips not covered exactly once: missing {missing[:5]} duplicated {dup[:5]}")
    for b, pl in enumerate(sol.plans):
        if len(pl.conn) != len(pl.trips) - 1:
            errs.append(f"block {b}: connection list has the wrong length")
            continue
        for k, c in enumerate(pl.conn):
            i, j = pl.trips[k], pl.trips[k + 1]
            ok = inst.direct_ok(i, j) if c == "D" else inst.depot_ok(i, j)
            if not ok:
                errs.append(f"block {b}: connection {c} from trip {i} to {j} is not possible in time")
        short, _ = simulate_plan(inst, pl, f, sol.sites)
        if short > tol:
            errs.append(f"block {b}: battery shortfall {short:.1f} kWh at consumption factor {f}")
    return errs


def soc_trace(inst: Instance, pl, f: float, sites):
    """Battery level of one block over the day as (minutes, kWh) points, without the emergency
    floor, so shortfalls show as levels under the reserve."""
    p = inst.params
    cf = p.cons * f
    t = pl.trips
    soc = p.smax
    pts = [(inst.start[t[0]] - inst.dh_min[inst.depot, inst.sterm[t[0]]], soc)]
    soc -= inst.km_out(t[0]) * cf
    for k, i in enumerate(t):
        pts.append((inst.start[i], soc))
        soc -= inst.km[i] * cf
        pts.append((inst.end[i], soc))
        if k == len(t) - 1:
            soc -= inst.km_in(i) * cf
            pts.append((inst.end[i] + inst.dh_min[inst.eterm[i], inst.depot], soc))
            break
        j = t[k + 1]
        if pl.conn[k] == "D":
            if inst.eterm[i] in sites:
                soc = min(p.smax, soc + inst.cap_direct(i, j))
                pts.append((inst.end[i] + inst.occupancy(i, j), soc))
            soc -= inst.km_direct(i, j) * cf
        else:
            soc -= inst.km_in(i) * cf
            pts.append((inst.end[i] + inst.dh_min[inst.eterm[i], inst.depot], soc))
            soc = min(p.smax, soc + inst.cap_depot(i, j))
            pts.append((inst.start[j] - inst.dh_min[inst.depot, inst.sterm[j]], soc))
            soc -= inst.dh_km[inst.depot, inst.sterm[j]] * cf
    return pts
