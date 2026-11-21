"""ALNS with learned guidance: a bandit instead of fixed operator weights, and a gradient-boosting surrogate that
predicts the cost change of a destroy move before it is repaired and costed.

The surrogate never decides what is accepted. Every candidate that the search looks at is repaired and costed by the
original cost function (`model.cost_breakdown`, through `State.cost`); the surrogate only chooses, among `k` cheap
removal sets, the one that is tried (with probability `eps` a random one, so that it keeps exploring). Whether its
predictions were right is recorded: precision of "predicted to improve" against the recomputed cost.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .alns import State, _replan_all, _unit_dh_cost, initial_solution, repair
from .model import Solution, plan_block

OPS = ["random", "related", "block", "worst"]          # removal operators; "site" (open or close a charger) is separate
FEATURES = (["op_" + o for o in OPS] + ["q", "n_blocks", "span_min", "mean_start_h", "wait_sum", "wait_mean", "km_sum",
            "n_terminals", "frac_time", "rel_cost"])


def removal_set(inst, plans, op, nq, rng):
    """Cheap destroy: which trips to remove (no replanning of the remaining blocks yet)."""
    q = int(rng.integers(max(2, nq // 3), nq + 1))
    if op == "random":
        flat = [x for pl in plans for x in pl.trips]
        pick = rng.choice(len(flat), size=min(q, len(flat)), replace=False)
        rem = {flat[x] for x in pick}
    elif op == "related":
        s = int(rng.integers(inst.n))
        d = np.abs(inst.start - inst.start[s]) + 30.0 * (inst.eterm != inst.eterm[s])
        rem = set(np.argsort(d)[:q].tolist())
    elif op == "block":
        sizes = np.array([len(pl.trips) for pl in plans], float)
        idx = rng.choice(len(plans), size=min(2, len(plans)), replace=False, p=(1 / sizes) / (1 / sizes).sum())
        rem = {x for b in np.atleast_1d(idx) for x in plans[int(b)].trips}
    else:
        cand = []
        for pl in plans:
            t = pl.trips
            for a in range(1, len(t)):
                cand.append((inst.start[t[a]] - inst.end[t[a - 1]], t[a]))
        cand.sort(reverse=True)
        rem = {x for _, x in cand[:q]}
        if not rem:
            rem = {int(rng.integers(inst.n))}
    return {int(x) for x in rem}


def features(inst, pred_wait, rem, op, frac, rel_cost):
    r = np.fromiter(rem, int)
    st = inst.start[r]
    w = pred_wait[r]
    f = [float(op == o) for o in OPS]
    f += [len(r), 0.0, float(st.max() - st.min()), float(st.mean() / 60), float(w.sum()), float(w.mean()),
          float(inst.km[r].sum()), float(len(set(inst.eterm[r].tolist()))), frac, rel_cost]
    return f


def wait_before(inst, plans):
    """Minutes between each trip and its predecessor in the current blocks (0 for the first trip of a block)."""
    w = np.zeros(inst.n)
    for pl in plans:
        t = pl.trips
        for a in range(1, len(t)):
            w[t[a]] = inst.start[t[a]] - inst.end[t[a - 1]]
    return w


def guided_alns(inst, time_limit=60.0, seed=0, f=1.0, selector="ucb", surrogate=None, k=4, eps=0.1,
                trace=None, moves=None, init=None):
    """selector: 'weights' (as in alns.py, adaptive scores) or 'ucb' (discounted UCB1 on relative cost improvement).
    surrogate: fitted regressor with .predict(X) -> predicted cost change, or None. k: removal sets drawn per
    iteration when a surrogate is given. trace: list of (seconds, iterations, best cost). moves: list of dict rows
    (features, cost change, predicted change) for training and for the precision check."""
    rng = np.random.default_rng(seed)
    unit = _unit_dh_cost(inst)
    all_sites = frozenset(range(inst.T))
    site_list = sorted(all_sites)
    cur = init if init is not None else initial_solution(inst, all_sites, f)
    best = cur.copy()
    best._cost = cur.cost
    c0 = cur.cost
    T0 = max(1.0, 0.002 * c0)
    nq = max(3, min(40, inst.n // 12))
    arms = OPS + ["site"]
    w = np.ones(len(arms))
    scores, uses = np.zeros(len(arms)), np.zeros(len(arms))
    ucb_n, ucb_r = np.full(len(arms), 1e-9), np.zeros(len(arms))
    t0 = time.time()
    it = full_evals = 0
    n_pred_improve = n_pred_improve_true = 0
    wait = wait_before(inst, cur.plans)
    while time.time() - t0 < time_limit:
        it += 1
        frac = (time.time() - t0) / time_limit
        Temp = T0 * (0.01 ** frac)
        if selector == "ucb":
            tot = ucb_n.sum()
            score = ucb_r / ucb_n + np.sqrt(2 * math.log(max(tot, 2.0)) / ucb_n)
            a = int(np.argmax(score))
        else:
            a = int(rng.choice(len(arms), p=w / w.sum()))
        op = arms[a]
        uses[a] += 1
        plans, sites, removed = list(cur.plans), cur.sites, []
        pred, feat = None, None
        if op == "site":
            s = int(site_list[int(rng.integers(len(site_list)))])
            sites = frozenset(sites - {s}) if s in sites else frozenset(sites | {s})
            plans, removed = _replan_all(inst, plans, sites, f)
        else:
            cands = [removal_set(inst, plans, op, nq, rng) for _ in range(k if surrogate is not None else 1)]
            fs = [features(inst, wait, c, op, frac, cur.cost / c0) for c in cands]
            if surrogate is not None:
                p = surrogate.predict(np.array(fs))
                j = int(rng.integers(len(cands))) if rng.random() < eps else int(np.argmin(p))
                pred = float(p[j])
            else:
                j = 0
            rem, feat = cands[j], fs[j]
            newplans = []
            for pl in plans:
                keep = [x for x in pl.trips if x not in rem]
                if len(keep) == len(pl.trips):
                    newplans.append(pl)
                elif keep:
                    npl = plan_block(inst, keep, sites, f)
                    if npl is None:
                        rem.update(keep)
                    else:
                        newplans.append(npl)
            plans, removed = newplans, list(rem)
        plans = repair(inst, plans, removed, sites, f, rng, unit)
        if plans is None:
            ucb_n[a] += 1
            continue
        cand = State(inst, plans, sites)
        c_new, c_cur = cand.cost, cur.cost               # original cost function
        full_evals += 1
        if feat is not None and moves is not None:
            moves.append({"features": feat, "delta": c_new - c_cur, "pred": pred, "op": op})
        if pred is not None and pred < 0:
            n_pred_improve += 1
            n_pred_improve_true += int(c_new < c_cur - 1e-9)
        gain = 0.0
        if c_new < best.cost - 1e-9:
            best, gain = cand, 33.0
        elif c_new < c_cur - 1e-9:
            gain = 9.0
        elif rng.random() < math.exp(-(c_new - c_cur) / Temp):
            gain = 3.0
        reward = float(np.clip((c_cur - c_new) / (0.002 * c0), 0.0, 1.0))
        ucb_r[a] = 0.99 * ucb_r[a] + reward
        ucb_n[a] = 0.99 * ucb_n[a] + 1.0
        if gain > 0:
            cur = cand
            wait = wait_before(inst, cur.plans)
        scores[a] += gain
        if it % 50 == 0:
            upd = np.where(uses > 0, scores / np.maximum(uses, 1), 0)
            w = 0.8 * w + 0.2 * (upd + 0.1)
            scores[:] = 0
            uses[:] = 0
            if trace is not None:
                trace.append((time.time() - t0, it, best.cost))
    if trace is not None:
        trace.append((time.time() - t0, it, best.cost))
    stats = {"iterations": it, "full_evals": full_evals, "pred_improve": n_pred_improve,
             "pred_improve_true": n_pred_improve_true}
    return Solution(best.plans, best.sites), stats
