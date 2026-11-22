"""Multi-cut L-shaped (Benders) method for the two-stage electric bus program.

First stage (integer): connections, vehicles, charging terminals and charge points. Second stage per consumption
scenario: battery levels, charging energy and emergency energy, a linear program for a fixed first stage. The
recourse cost Q_w(x) of scenario w is convex and piecewise linear in the first-stage variables x, so the dual
solution of the scenario program gives an optimality cut  theta_w >= Q_w(x_hat) + g_w . (x - x_hat).
Master problem: first-stage rows, one variable theta_w per scenario and the cuts, solved by HiGHS (SciPy milp).
Lower bound: dual bound of the master; upper bound: first-stage cost of the best x_hat plus its exact recourse cost.
The recourse program is a linear program, so every upper bound is an exact cost of a feasible plan.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .exact import build_model, solve_milp
from .model import Instance


class Problem:
    def __init__(self, inst: Instance, scenarios):
        self.inst, self.scen = inst, list(scenarios)
        M = build_model(inst, self.scen)
        self.M = M
        W, R = M["W"], M["R"]
        self.nfirst = M["first_end"]
        n_all = W.n
        self.cost = np.array(W.cost, float)
        self.lb, self.ub = np.array(W.lb, float), np.array(W.ub, float)
        self.integ = np.array(W.integ)
        A = sparse.coo_matrix((R.v, (R.r, R.c)), shape=(R.m, n_all)).tocsr()
        lo, hi = np.array(R.lo, float), np.array(R.hi, float)
        # scenario blocks of columns
        S = len(self.scen)
        starts = [M["sv"][w] for w in range(S)] + [n_all]
        self.blocks = [(starts[w], starts[w + 1]) for w in range(S)]
        # row ownership: first stage if no column at or after nfirst
        Ac = A.tocsc()
        rows_first, rows_w = [], [[] for _ in range(S)]
        later = (A[:, self.nfirst:] != 0).sum(axis=1).A1 > 0
        for r in range(A.shape[0]):
            if not later[r]:
                rows_first.append(r)
            else:
                cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
                c = cols[cols >= self.nfirst][0]
                w = next(k for k, (a, b) in enumerate(self.blocks) if a <= c < b)
                rows_w[w].append(r)
        self.A, self.lo, self.hi = A, lo, hi
        self.rows_first = np.array(rows_first)
        self.rows_w = [np.array(r) for r in rows_w]
        self.cost_first = self.cost[:self.nfirst]

    # ----- second stage -----
    def recourse(self, w: int, x: np.ndarray):
        """Exact recourse cost Q_w(x) (already weighted by the scenario probability) and the cut gradient."""
        a, b = self.blocks[w]
        rows = self.rows_w[w]
        Ax = self.A[rows][:, :self.nfirst]
        Ay = self.A[rows][:, a:b]
        lo, hi = self.lo[rows], self.hi[rows]
        shift = Ax @ x
        up = np.isfinite(hi)
        dn = np.isfinite(lo)
        G = sparse.vstack([Ay[up], -Ay[dn]]).tocsr()
        bvec = np.concatenate([hi[up] - shift[up], -(lo[dn] - shift[dn])])
        res = linprog(self.cost[a:b], A_ub=G, b_ub=bvec, bounds=np.column_stack([self.lb[a:b], self.ub[a:b]]),
                      method="highs")
        if res.status != 0:
            raise RuntimeError(f"recourse LP scenario {w}: {res.message}")
        lam = res.ineqlin.marginals
        lu, ld = lam[:up.sum()], lam[up.sum():]
        g = -(Ax[up].T @ lu) + (Ax[dn].T @ ld)
        return float(res.fun), np.asarray(g).ravel()

    def total(self, x: np.ndarray):
        qs, gs = zip(*[self.recourse(w, x) for w in range(len(self.scen))])
        return float(self.cost_first @ x), list(qs), list(gs)


def lshaped(inst: Instance, scenarios, seeds=(), time_limit: float = 7200.0, master_time: float = 900.0,
            gap: float = 0.005, log=print):
    """seeds: first-stage plans (dicts as returned by solve_milp) used to create the initial cuts."""
    t0 = time.time()
    P = Problem(inst, scenarios)
    S, nf = len(P.scen), P.nfirst
    M = P.M
    first_rows = P.A[P.rows_first][:, :nf]
    cuts_A, cuts_lo = [], []                       # rows of [x, theta] space
    history = []
    best = {"ub": np.inf, "x": None}

    def plan_to_x(fs):
        x = np.zeros(nf)
        for a, key in enumerate(M["D"]): x[M["xd"] + a] = 1.0 if key in fs["xd"] else 0.0
        for a, key in enumerate(M["V"]): x[M["xv"] + a] = 1.0 if key in fs["xv"] else 0.0
        for i in range(M["n"]):
            x[M["ys"] + i] = 1.0 if i in fs["ys"] else 0.0
            x[M["ye"] + i] = 1.0 if i in fs["ye"] else 0.0
        for k in range(inst.T):
            x[M["o"] + k] = 1.0 if k in fs["o"] else 0.0
            x[M["K"] + k] = fs["K"].get(k, 0)
        return x

    def add_cuts(x):
        c1, qs, gs = P.total(x)
        ub = c1 + sum(qs)
        for w in range(S):
            row = sparse.hstack([sparse.csr_matrix(-gs[w].reshape(1, -1)),
                                 sparse.csr_matrix(([1.0], ([0], [w])), shape=(1, S))]).tocsr()
            cuts_A.append(row)
            cuts_lo.append(qs[w] - float(gs[w] @ x))
        if ub < best["ub"]:
            best.update(ub=ub, x=x.copy(), first=c1, q=qs)
        return ub

    for fs in seeds:
        ub = add_cuts(plan_to_x(fs))
        log(f"seed cut: ub {ub:.2f}  best {best['ub']:.2f}  ({time.time() - t0:.0f} s)")
    cost_m = np.concatenate([P.cost_first, np.ones(S)])
    lb_m = np.concatenate([P.lb[:nf], np.zeros(S)])
    ub_m = np.concatenate([P.ub[:nf], np.full(S, 1e7)])
    integ = np.concatenate([P.integ[:nf], np.zeros(S)])
    base_rows = sparse.hstack([first_rows, sparse.csr_matrix((first_rows.shape[0], S))]).tocsr()
    base_lo, base_hi = P.lo[P.rows_first], P.hi[P.rows_first]
    lb_best, it = 0.0, 0
    while time.time() - t0 < time_limit:
        it += 1
        A = sparse.vstack([base_rows] + cuts_A).tocsr()
        lo = np.concatenate([base_lo, cuts_lo]); hi = np.concatenate([base_hi, np.full(len(cuts_lo), np.inf)])
        res = milp(cost_m, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb_m, ub_m),
                   options={"time_limit": master_time, "mip_rel_gap": 1e-4})
        if res.x is None:
            log("master returned no solution"); break
        dual = float(getattr(res, "mip_dual_bound", res.fun))
        lb_best = max(lb_best, dual)
        x = np.round(res.x[:nf]) if False else res.x[:nf]
        ub = add_cuts(x)
        g = (best["ub"] - lb_best) / best["ub"]
        history.append({"iter": it, "lb": lb_best, "ub": best["ub"], "gap": g, "master_obj": float(res.fun),
                        "time": time.time() - t0, "cuts": len(cuts_lo)})
        log(f"iter {it}: lb {lb_best:.2f} ub {best['ub']:.2f} gap {100 * g:.2f}%  master {res.fun:.2f}  ({time.time() - t0:.0f} s)")
        if g <= gap:
            break
    return {"RP_ub": best["ub"], "RP_lb": lb_best, "gap": (best["ub"] - lb_best) / best["ub"], "history": history,
            "first_stage_cost": best.get("first"), "recourse_by_scenario": best.get("q"), "x": best["x"], "problem": P}
