"""Mixed-integer program of the electric vehicle scheduling problem with opportunity
charging (HiGHS through SciPy), deterministic or two-stage with uncertain consumption.

First stage: trip-to-trip connections (direct or via the depot), vehicles, open charging
terminals and charge points. Second stage per consumption scenario: battery level, charging
energy and emergency energy bought when the reserve floor would be violated.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp
from scipy.sparse import coo_matrix

from .model import BlockPlan, Instance, Solution, block_conn_km

KMAX = 15  # charge points per terminal


class _Vars:
    def __init__(self):
        self.n = 0
        self.lb, self.ub, self.integ, self.cost = [], [], [], []

    def add(self, count, lb, ub, integer, cost=0.0):
        start = self.n
        self.n += count
        self.lb += [lb] * count if np.isscalar(lb) else list(lb)
        self.ub += [ub] * count if np.isscalar(ub) else list(ub)
        self.integ += [1 if integer else 0] * count
        self.cost += [cost] * count if np.isscalar(cost) else list(cost)
        return start


class _Rows:
    def __init__(self):
        self.r, self.c, self.v, self.lo, self.hi = [], [], [], [], []
        self.m = 0

    def add(self, cols, vals, lo, hi):
        self.r += [self.m] * len(cols)
        self.c += list(cols)
        self.v += list(vals)
        self.lo.append(lo)
        self.hi.append(hi)
        self.m += 1


def arcs(inst: Instance):
    D, V = [], []
    for i in range(inst.n):
        for j in range(i + 1, inst.n):
            if inst.start[j] - inst.end[i] > inst.params.dep_gap_max + 120:
                break
            if inst.direct_ok(i, j):
                D.append((i, j))
            if inst.depot_ok(i, j):
                V.append((i, j))
    return D, V


def build_model(inst: Instance, scenarios=((1.0, 1.0),)):
    """Variables and rows of the two-stage program. Returns a dict with the variable table W, the row table R, the
    column offsets and the arc lists. Shared by solve_milp and the L-shaped method (lshaped.py)."""
    p = inst.params
    n = inst.n
    D, V = arcs(inst)
    nd, nv = len(D), len(V)
    smin, smax = p.smin, p.smax
    W = _Vars()
    unit_dh = p.c_dh_km + p.c_kwh * p.cons
    xd = W.add(nd, 0, 1, True, [unit_dh * inst.km_direct(i, j) for i, j in D])
    xv = W.add(nv, 0, 1, True, [unit_dh * inst.km_depot(i, j) for i, j in V])
    ys = W.add(n, 0, 1, True, [p.c_veh + unit_dh * inst.km_out(i) for i in range(n)])
    ye = W.add(n, 0, 1, True, [unit_dh * inst.km_in(i) for i in range(n)])
    o = W.add(inst.T, 0, 1, True, p.c_site)
    K = W.add(inst.T, 0, KMAX, True, p.c_cp)
    first_end = W.n
    S = len(scenarios)
    sv, qd, qv, uu = [], [], [], []
    for (f, pr) in scenarios:
        sv.append(W.add(n, smin, smax, False))
        qd.append(W.add(nd, 0, [inst.cap_direct(i, j) for i, j in D], False))
        qv.append(W.add(nv, 0, [inst.cap_depot(i, j) for i, j in V], False))
        uu.append(W.add(n, 0, smax, False, p.c_resc * pr))
    R = _Rows()
    # flow conservation
    out_d = [[] for _ in range(n)]
    in_d = [[] for _ in range(n)]
    out_v = [[] for _ in range(n)]
    in_v = [[] for _ in range(n)]
    for a, (i, j) in enumerate(D):
        out_d[i].append(a)
        in_d[j].append(a)
    for a, (i, j) in enumerate(V):
        out_v[i].append(a)
        in_v[j].append(a)
    for i in range(n):
        R.add([xd + a for a in out_d[i]] + [xv + a for a in out_v[i]] + [ye + i], [1] * (len(out_d[i]) + len(out_v[i]) + 1), 1, 1)
        R.add([xd + a for a in in_d[i]] + [xv + a for a in in_v[i]] + [ys + i], [1] * (len(in_d[i]) + len(in_v[i]) + 1), 1, 1)
    # charge points: o and K
    for k in range(inst.T):
        R.add([K + k, o + k], [1, -KMAX], -np.inf, 0)
    # a charger is needed to charge: handled through q <= cap * o below
    # occupancy: at every trip end at terminal k, arcs whose occupancy covers that time <= K_k
    occ = {}
    for a, (i, j) in enumerate(D):
        if inst.direct_tau(i, j) >= p.tmin_chg:
            occ.setdefault(int(inst.eterm[i]), []).append((inst.end[i], inst.end[i] + inst.occupancy(i, j), a))
    for k, lst in occ.items():
        starts = sorted({s for s, _, _ in lst})
        for t in starts:
            cols = [xd + a for s, e, a in lst if s <= t < e - 1e-9]
            big = len(cols)
            # buses wait at a closed terminal without holding a charge point: relax with o_k = 0
            R.add(cols + [K + k, o + k], [1] * len(cols) + [-1, big], -np.inf, big)
    # per scenario battery constraints
    for w, (f, pr) in enumerate(scenarios):
        cf = p.cons * f
        s0, q0, v0, u0 = sv[w], qd[w], qv[w], uu[w]
        for i in range(n):
            ei = inst.km[i] * cf
            # pull-out: s_i <= smax - cf*km_out + M(1 - ys)
            eo = inst.km_out(i) * cf
            M = smax - smin + eo
            R.add([s0 + i, ys + i], [1, M], -np.inf, smax - eo + M)
            # reach the depot after the last trip: s_i - e_i + u_i - cf*km_in >= smin - M(1 - ye)
            ein = inst.km_in(i) * cf
            M2 = smax - smin + ei + ein
            R.add([s0 + i, u0 + i, ye + i], [1, 1, -M2], smin + ei + ein - M2, np.inf)
            # end of every trip: s_i - e_i + u_i >= smin
            R.add([s0 + i, u0 + i], [1, 1], smin + ei, np.inf)
        for a, (i, j) in enumerate(D):
            ei = inst.km[i] * cf
            dh = inst.km_direct(i, j) * cf
            M = smax - smin + ei + dh
            # s_j <= s_i - e_i + u_i - dh + q + M(1 - x)
            R.add([s0 + j, s0 + i, u0 + i, q0 + a, xd + a], [1, -1, -1, -1, M], -np.inf, -ei - dh + M)
            cap = inst.cap_direct(i, j)
            R.add([q0 + a, xd + a], [1, -cap], -np.inf, 0)
            R.add([q0 + a, o + int(inst.eterm[i])], [1, -cap], -np.inf, 0)
        for a, (i, j) in enumerate(V):
            ei = inst.km[i] * cf
            din = inst.km_in(i) * cf
            dout = inst.dh_km[inst.depot, inst.sterm[j]] * cf
            M = smax - smin + ei + din + dout
            R.add([s0 + j, s0 + i, u0 + i, v0 + a, xv + a], [1, -1, -1, -1, M], -np.inf, -ei - din - dout + M)
            cap = inst.cap_depot(i, j)
            R.add([v0 + a, xv + a], [1, -cap], -np.inf, 0)
            M3 = smax - smin + ei + din
            R.add([s0 + i, u0 + i, xv + a], [1, 1, -M3], smin + ei + din - M3, np.inf)
    return {"W": W, "R": R, "D": D, "V": V, "nd": nd, "nv": nv, "n": n, "xd": xd, "xv": xv, "ys": ys, "ye": ye,
            "o": o, "K": K, "first_end": first_end, "sv": sv, "qd": qd, "qv": qv, "uu": uu, "p": p}


def solve_milp(inst: Instance, scenarios=((1.0, 1.0),), time_limit: float = 60.0, mip_gap: float = 1e-4,
               fix: dict | None = None, first_stage_only_cost: bool = False, log: bool = False):
    """Solve the (two-stage) program. `scenarios` is a list of (consumption factor, probability).
    `fix` pins the first stage: {'xd': set of arcs, 'xv': set of arcs, 'ys': set, 'ye': set, 'o': set, 'K': dict}."""
    M_ = build_model(inst, scenarios)
    W, R, D, V, nd, nv, n, xd, xv, ys, ye, o, K, sv, qd, qv, uu = (M_[k] for k in (
        "W", "R", "D", "V", "nd", "nv", "n", "xd", "xv", "ys", "ye", "o", "K", "sv", "qd", "qv", "uu"))
    S = len(scenarios)
    lb = np.array(W.lb, float)
    ub = np.array(W.ub, float)
    if fix is not None:
        def pin(base, items, universe):
            for a, key in enumerate(universe):
                v = 1.0 if key in items else 0.0
                lb[base + a] = ub[base + a] = v
        pin(xd, fix["xd"], D)
        pin(xv, fix["xv"], V)
        pin(ys, fix["ys"], range(n))
        pin(ye, fix["ye"], range(n))
        pin(o, fix["o"], range(inst.T))
        for k in range(inst.T):
            lb[K + k] = ub[K + k] = fix["K"].get(k, 0)
    A = coo_matrix((R.v, (R.r, R.c)), shape=(R.m, W.n)).tocsr()
    t0 = time.time()
    res = milp(c=np.array(W.cost), constraints=LinearConstraint(A, R.lo, R.hi), integrality=np.array(W.integ),
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": log})
    out = {"status": int(res.status), "message": res.message, "time": time.time() - t0, "n_vars": W.n, "n_rows": R.m,
           "n_arcs": nd + nv}
    if res.x is None:
        out.update(objective=None, bound=None, solution=None)
        return out
    x = res.x
    out["objective"] = float(res.fun)
    out["bound"] = float(getattr(res, "mip_dual_bound", np.nan))
    out["gap"] = float(getattr(res, "mip_gap", np.nan))
    sol_d = {D[a] for a in range(nd) if x[xd + a] > 0.5}
    sol_v = {V[a] for a in range(nv) if x[xv + a] > 0.5}
    ysel = {i for i in range(n) if x[ys + i] > 0.5}
    yesel = {i for i in range(n) if x[ye + i] > 0.5}
    osel = {k for k in range(inst.T) if x[o + k] > 0.5}
    Ksel = {k: int(round(x[K + k])) for k in range(inst.T)}
    out["first_stage"] = {"xd": sol_d, "xv": sol_v, "ys": ysel, "ye": yesel, "o": osel, "K": Ksel}
    out["solution"] = solution_from_arcs(inst, sol_d, sol_v, ysel, osel)
    out["emergency_kwh"] = [float(x[uu[w]:uu[w] + n].sum()) for w in range(S)]
    return out


def solution_from_arcs(inst: Instance, xd, xv, ys, sites) -> Solution:
    succ = {}
    for (i, j) in xd:
        succ[i] = (j, "D")
    for (i, j) in xv:
        succ[i] = (j, "V")
    plans = []
    for i0 in sorted(ys):
        trips, conn = [i0], []
        cur = i0
        while cur in succ:
            nxt, c = succ[cur]
            trips.append(nxt)
            conn.append(c)
            cur = nxt
        plans.append(BlockPlan(trips, conn, block_conn_km(inst, trips, conn), conn.count("V")))
    return Solution(plans, frozenset(sites))
