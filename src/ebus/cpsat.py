"""Constraint programming model of the same problem (OR-Tools CP-SAT): a multiple circuit over
the trips for the vehicle blocks, integer battery levels in kWh and a cumulative constraint
for the charge points of each terminal."""
from __future__ import annotations

import math
import time

from ortools.sat.python import cp_model

from .exact import arcs, solution_from_arcs, KMAX
from .model import Instance


def solve_cpsat(inst: Instance, time_limit: float = 60.0, f: float = 1.0, workers: int = 8, hint=None, log: bool = False):
    p = inst.params
    n = inst.n
    cf = p.cons * f
    D, V = arcs(inst)
    smin, smax = math.ceil(p.smin), math.floor(p.smax)
    m = cp_model.CpModel()
    unit = p.c_dh_km + p.c_kwh * p.cons
    SC = 100  # cents

    def ceil_e(km): return int(math.ceil(km * cf))

    ys = [m.NewBoolVar(f"ys{i}") for i in range(n)]
    ye = [m.NewBoolVar(f"ye{i}") for i in range(n)]
    d = {a: m.NewBoolVar(f"d{a}") for a in D}
    v = {a: m.NewBoolVar(f"v{a}") for a in V}
    s = [m.NewIntVar(smin, smax, f"s{i}") for i in range(n)]
    o = [m.NewBoolVar(f"o{k}") for k in range(inst.T)]
    K = [m.NewIntVar(0, KMAX, f"K{k}") for k in range(inst.T)]
    circ = []
    for i in range(n):
        circ.append((0, i + 1, ys[i]))
        circ.append((i + 1, 0, ye[i]))
    pair = {}
    for a in D:
        pair.setdefault(a, []).append(d[a])
    for a in V:
        pair.setdefault(a, []).append(v[a])
    for (i, j), lits in pair.items():
        lit = m.NewBoolVar(f"a{i}_{j}")
        m.Add(sum(lits) == lit)
        circ.append((i + 1, j + 1, lit))
    m.AddMultipleCircuit(circ)
    for i in range(n):
        eo = ceil_e(inst.km_out(i))
        m.Add(s[i] <= smax - eo).OnlyEnforceIf(ys[i])
        ein = ceil_e(inst.km_in(i))
        ei = ceil_e(inst.km[i])
        m.Add(s[i] - ei - ein >= smin).OnlyEnforceIf(ye[i])
        m.Add(s[i] - ei >= smin)
    occ = {}
    for (i, j) in D:
        cap = int(math.floor(inst.cap_direct(i, j)))
        q = m.NewIntVar(0, cap, f"q{i}_{j}")
        ei = ceil_e(inst.km[i])
        dh = ceil_e(inst.km_direct(i, j))
        m.Add(s[j] <= s[i] - ei - dh + q).OnlyEnforceIf(d[(i, j)])
        m.Add(q == 0).OnlyEnforceIf(d[(i, j)].Not())
        if cap > 0:
            m.Add(q == 0).OnlyEnforceIf(o[int(inst.eterm[i])].Not())
            a = inst.end[i]
            occ.setdefault(int(inst.eterm[i]), []).append(
                (m.NewOptionalFixedSizeIntervalVar(int(round(a * 10)), max(1, int(round(inst.occupancy(i, j) * 10))),
                                                   d[(i, j)], f"occ{i}_{j}")))
    big = 10 * n
    for k, ivs in occ.items():
        cap_k = m.NewIntVar(0, big, f"cap{k}")
        m.Add(cap_k == K[k]).OnlyEnforceIf(o[k])
        m.Add(cap_k == big).OnlyEnforceIf(o[k].Not())
        m.AddCumulative(ivs, [1] * len(ivs), cap_k)
    for k in range(inst.T):
        m.Add(K[k] == 0).OnlyEnforceIf(o[k].Not())
    for (i, j) in V:
        cap = int(math.floor(inst.cap_depot(i, j)))
        q = m.NewIntVar(0, cap, f"qv{i}_{j}")
        ei = ceil_e(inst.km[i])
        din = ceil_e(inst.km_in(i))
        dout = ceil_e(inst.dh_km[inst.depot, inst.sterm[j]])
        m.Add(s[j] <= s[i] - ei - din - dout + q).OnlyEnforceIf(v[(i, j)])
        m.Add(s[i] - ei - din >= smin).OnlyEnforceIf(v[(i, j)])
        m.Add(q == 0).OnlyEnforceIf(v[(i, j)].Not())
    obj = []
    for i in range(n):
        obj.append(int(round(SC * (p.c_veh + unit * inst.km_out(i)))) * ys[i])
        obj.append(int(round(SC * unit * inst.km_in(i))) * ye[i])
    for (i, j) in D:
        obj.append(int(round(SC * unit * inst.km_direct(i, j))) * d[(i, j)])
    for (i, j) in V:
        obj.append(int(round(SC * unit * inst.km_depot(i, j))) * v[(i, j)])
    for k in range(inst.T):
        obj.append(int(SC * p.c_site) * o[k])
        obj.append(int(SC * p.c_cp) * K[k])
    m.Minimize(sum(obj))
    if hint is not None:
        for i in range(n):
            m.AddHint(ys[i], 1 if i in hint["ys"] else 0)
            m.AddHint(ye[i], 1 if i in hint["ye"] else 0)
        for a in D:
            m.AddHint(d[a], 1 if a in hint["xd"] else 0)
        for a in V:
            m.AddHint(v[a], 1 if a in hint["xv"] else 0)
        for k in range(inst.T):
            m.AddHint(o[k], 1 if k in hint["o"] else 0)
            m.AddHint(K[k], hint["K"].get(k, 0))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = log
    t0 = time.time()
    st = solver.Solve(m)
    out = {"status": solver.StatusName(st), "time": time.time() - t0, "n_arcs": len(D) + len(V)}
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out["objective"] = solver.ObjectiveValue() / SC
        out["bound"] = solver.BestObjectiveBound() / SC
        out["gap"] = (out["objective"] - out["bound"]) / max(out["objective"], 1e-9)
        xd = {a for a in D if solver.Value(d[a])}
        xv = {a for a in V if solver.Value(v[a])}
        ysel = {i for i in range(n) if solver.Value(ys[i])}
        osel = {k for k in range(inst.T) if solver.Value(o[k])}
        out["solution"] = solution_from_arcs(inst, xd, xv, ysel, osel)
    else:
        out["objective"] = out["bound"] = out["gap"] = None
        out["solution"] = None
    return out
