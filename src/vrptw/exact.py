"""Exact methods for small VRPTW instances: a MIP (Pyomo + HiGHS, and the same model in PySCIPOpt) and CP-SAT.

All three minimise  BIG * vehicles + distance  like the ALNS (BIG = 1e4), so their optimum is directly comparable.
The MIP is the two-index formulation with arrival-time and load variables (big-M); it is only practical for
about ten customers, which is why the benchmark in bench.py uses first-k subsets of the Solomon instances.
"""
from __future__ import annotations

import math

from .instance import Instance

BIG = 1.0e4


def _bigm(I: Instance, i: int, j: int) -> float:
    return max(0.0, I.due[i] + I.service[i] + I.dist(i, j) - I.ready[j])


def _arcs(I: Instance):
    """Arcs that cannot be dropped by a simple time-window or capacity test."""
    n = I.n
    out = []
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                continue
            if i and j and (I.ready[i] + I.service[i] + I.dist(i, j) > I.due[j] + 1e-9 or I.demand[i] + I.demand[j] > I.capacity):
                continue
            out.append((i, j))
    return out


def solve_highs(I: Instance, time_limit: float = 120.0) -> dict:
    import pyomo.environ as pyo

    n = I.n
    arcs = _arcs(I)
    m = pyo.ConcreteModel()
    m.A = pyo.Set(initialize=arcs, dimen=2)
    m.C = pyo.RangeSet(1, n)
    m.x = pyo.Var(m.A, domain=pyo.Binary)
    m.t = pyo.Var(m.C, bounds=lambda m, i: (I.ready[i], I.due[i]))
    m.q = pyo.Var(m.C, bounds=lambda m, i: (I.demand[i], I.capacity))
    m.obj = pyo.Objective(expr=sum((I.dist(i, j) + (BIG if i == 0 else 0.0)) * m.x[i, j] for i, j in arcs))
    m.out = pyo.Constraint(m.C, rule=lambda m, i: sum(m.x[a, b] for a, b in arcs if a == i) == 1)
    m.inn = pyo.Constraint(m.C, rule=lambda m, j: sum(m.x[a, b] for a, b in arcs if b == j) == 1)
    m.depot = pyo.Constraint(expr=sum(m.x[0, j] for j in range(1, n + 1) if (0, j) in m.A) ==
                             sum(m.x[i, 0] for i in range(1, n + 1) if (i, 0) in m.A))
    cons = pyo.ConstraintList()
    m.cons = cons
    for i, j in arcs:
        if i and j:
            M = _bigm(I, i, j)
            cons.add(m.t[j] >= m.t[i] + I.service[i] + I.dist(i, j) - M * (1 - m.x[i, j]))
            cons.add(m.q[j] >= m.q[i] + I.demand[j] - I.capacity * (1 - m.x[i, j]))
        elif i == 0:
            cons.add(m.t[j] >= I.dist(0, j) - I.due[j] * (1 - m.x[0, j]))
        elif j == 0:
            cons.add(m.t[i] + I.service[i] + I.dist(i, 0) <= I.due[0] + 1e5 * (1 - m.x[i, 0]))
    s = pyo.SolverFactory("appsi_highs")
    s.config.time_limit = time_limit
    res = s.solve(m)
    term = str(res.solver.termination_condition)
    try:
        obj = float(pyo.value(m.obj))
        xs = {(i, j) for i, j in arcs if pyo.value(m.x[i, j]) > 0.5}
    except (ValueError, TypeError):
        return dict(status=term, objective=None)
    return dict(status=term, objective=obj, optimal="optimal" in term.lower(), routes=_routes_from(xs, n))


def solve_scip(I: Instance, time_limit: float = 120.0) -> dict:
    from pyscipopt import Model, quicksum

    n = I.n
    arcs = _arcs(I)
    m = Model()
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    x = {(i, j): m.addVar(vtype="B", name=f"x{i}_{j}") for i, j in arcs}
    t = {i: m.addVar(lb=I.ready[i], ub=I.due[i], name=f"t{i}") for i in range(1, n + 1)}
    q = {i: m.addVar(lb=I.demand[i], ub=I.capacity, name=f"q{i}") for i in range(1, n + 1)}
    m.setObjective(quicksum((I.dist(i, j) + (BIG if i == 0 else 0.0)) * x[i, j] for i, j in arcs), "minimize")
    for i in range(1, n + 1):
        m.addCons(quicksum(x[a, b] for a, b in arcs if a == i) == 1)
        m.addCons(quicksum(x[a, b] for a, b in arcs if b == i) == 1)
    m.addCons(quicksum(x[0, j] for j in range(1, n + 1) if (0, j) in x) == quicksum(x[i, 0] for i in range(1, n + 1) if (i, 0) in x))
    for i, j in arcs:
        if i and j:
            M = _bigm(I, i, j)
            m.addCons(t[j] >= t[i] + I.service[i] + I.dist(i, j) - M * (1 - x[i, j]))
            m.addCons(q[j] >= q[i] + I.demand[j] - I.capacity * (1 - x[i, j]))
        elif i == 0:
            m.addCons(t[j] >= I.dist(0, j) - I.due[j] * (1 - x[0, j]))
        else:
            m.addCons(t[i] + I.service[i] + I.dist(i, 0) <= I.due[0] + 1e5 * (1 - x[i, 0]))
    m.optimize()
    if m.getNSols() == 0:
        return dict(status=m.getStatus(), objective=None)
    return dict(status=m.getStatus(), objective=float(m.getObjVal()), optimal=m.getStatus() == "optimal",
                routes=_routes_from({a for a in arcs if m.getVal(x[a]) > 0.5}, n))


def solve_cpsat(I: Instance, time_limit: float = 120.0, scale: int = 1000) -> dict:
    from ortools.sat.python import cp_model

    n = I.n
    arcs = _arcs(I)
    m = cp_model.CpModel()
    lit = {a: m.NewBoolVar(f"x{a}") for a in arcs}
    m.AddMultipleCircuit([(i, j, lit[(i, j)]) for i, j in arcs])
    horizon = int(math.ceil(I.due[0] * scale))
    t = {i: m.NewIntVar(int(math.ceil(I.ready[i] * scale)), int(math.floor(I.due[i] * scale)), f"t{i}") for i in range(1, n + 1)}
    q = {i: m.NewIntVar(int(I.demand[i]), int(I.capacity), f"q{i}") for i in range(1, n + 1)}
    for i, j in arcs:
        if i and j:
            m.Add(t[j] >= t[i] + int(math.ceil((I.service[i] + I.dist(i, j)) * scale))).OnlyEnforceIf(lit[(i, j)])
            m.Add(q[j] >= q[i] + int(I.demand[j])).OnlyEnforceIf(lit[(i, j)])
        elif i == 0:
            m.Add(t[j] >= int(math.ceil(I.dist(0, j) * scale))).OnlyEnforceIf(lit[(0, j)])
            m.Add(q[j] >= int(I.demand[j])).OnlyEnforceIf(lit[(0, j)])
        else:
            m.Add(t[i] + int(math.ceil((I.service[i] + I.dist(i, 0)) * scale)) <= horizon).OnlyEnforceIf(lit[(i, 0)])
    m.Minimize(sum(int(round((I.dist(i, j) + (BIG if i == 0 else 0.0)) * scale)) * lit[(i, j)] for i, j in arcs))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 4
    st = s.Solve(m)
    name = s.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(status=name, objective=None)
    chosen = {a for a in arcs if s.Value(lit[a])}
    routes = _routes_from(chosen, n)
    d = sum(I.dist(*a) for a in chosen)
    return dict(status=name, objective=d + BIG * len(routes), optimal=st == cp_model.OPTIMAL, routes=routes)


def _routes_from(chosen, n):
    nxt = {}
    starts = []
    for i, j in chosen:
        if i == 0:
            starts.append(j)
        else:
            nxt[i] = j
    routes = []
    for s in starts:
        r, v = [], s
        while v != 0:
            r.append(v)
            v = nxt.get(v, 0)
        routes.append(r)
    return routes
