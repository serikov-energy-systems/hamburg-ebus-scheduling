"""Adaptive large neighbourhood search for the electric vehicle scheduling problem with
opportunity charging: trips are removed and re-inserted, charging terminals are opened and
closed, and every candidate is costed exactly by `model.cost_breakdown`."""
from __future__ import annotations

import math
import time
from bisect import bisect_left

import numpy as np

from .model import (BlockPlan, Instance, Solution, cost_breakdown, plan_block)


def _unit_dh_cost(inst: Instance) -> float:
    p = inst.params
    return p.c_dh_km + p.c_kwh * p.cons


class State:
    def __init__(self, inst: Instance, plans: list, sites: frozenset):
        self.inst, self.plans, self.sites = inst, plans, sites
        self._cost = None

    def copy(self):
        return State(self.inst, list(self.plans), self.sites)

    @property
    def cost(self):
        if self._cost is None:
            self._cost = cost_breakdown(self.inst, Solution(self.plans, self.sites))["cost"]
        return self._cost


def _insert_options(inst, plans, i, sites, f):
    """Best feasible insertion of trip i into an existing block: (delta_km, block index, new plan)."""
    best = None
    s_i, e_i = inst.start[i], inst.end[i]
    for b, pl in enumerate(plans):
        t = pl.trips
        starts = [inst.start[x] for x in t]
        pos = bisect_left(starts, s_i)
        if pos > 0 and inst.end[t[pos - 1]] > s_i:
            continue
        if pos < len(t) and inst.start[t[pos]] < e_i:
            continue
        if pos > 0:
            a = t[pos - 1]
            if not (inst.direct_ok(a, i) or inst.depot_ok(a, i)):
                continue
        if pos < len(t):
            c = t[pos]
            if not (inst.direct_ok(i, c) or inst.depot_ok(i, c)):
                continue
        new = plan_block(inst, t[:pos] + [i] + t[pos:], sites, f)
        if new is None:
            continue
        d = new.dh_km - pl.dh_km
        if best is None or d < best[0]:
            best = (d, b, new)
    return best


def repair(inst, plans, removed, sites, f, rng, unit):
    plans = list(plans)
    order = sorted(removed, key=lambda x: (rng.random() * 0.0, inst.start[x]))
    rng.shuffle(order)
    for i in order:
        opt = _insert_options(inst, plans, i, sites, f)
        new_block = plan_block(inst, [i], sites, f)
        if new_block is None:
            return None
        cost_new = inst.params.c_veh + new_block.dh_km * unit
        if opt is not None and opt[0] * unit < cost_new:
            plans[opt[1]] = opt[2]
        else:
            plans.append(new_block)
    return plans


def initial_solution(inst: Instance, sites: frozenset, f: float = 1.0):
    """Sequential greedy: append each trip (by start time) to the block that ends latest before it."""
    plans: list = []
    for i in range(inst.n):
        best = None
        for b, pl in enumerate(plans):
            a = pl.trips[-1]
            if not (inst.direct_ok(a, i) or inst.depot_ok(a, i)):
                continue
            new = plan_block(inst, pl.trips + [i], sites, f)
            if new is None:
                continue
            key = inst.end[a]
            if best is None or key > best[0]:
                best = (key, b, new)
        if best is not None:
            plans[best[1]] = best[2]
        else:
            nb = plan_block(inst, [i], sites, f)
            if nb is None:
                raise ValueError(f"trip {i} cannot be driven by a bus even from the depot")
            plans.append(nb)
    return State(inst, plans, sites)


def _replan_all(inst, plans, sites, f):
    out, removed = [], []
    for pl in plans:
        new = plan_block(inst, pl.trips, sites, f)
        if new is None:
            removed.extend(pl.trips)
        else:
            out.append(new)
    return out, removed


def alns(inst: Instance, time_limit: float = 60.0, seed: int = 0, f: float = 1.0,
         max_iter: int = 10**9, init: State | None = None, log: list | None = None,
         allowed_sites=None) -> Solution:
    rng = np.random.default_rng(seed)
    unit = _unit_dh_cost(inst)
    all_sites = frozenset(range(inst.T)) if allowed_sites is None else frozenset(allowed_sites)
    site_list = sorted(all_sites)
    cur = init if init is not None else initial_solution(inst, all_sites, f)
    best = cur.copy()
    best._cost = cur.cost
    T0 = max(1.0, 0.002 * cur.cost)
    t0 = time.time()
    it = 0
    ops = ["random", "related", "block", "worst", "site"]
    w = np.ones(len(ops))
    scores = np.zeros(len(ops))
    uses = np.zeros(len(ops))
    seg = 50
    nq = max(3, min(40, inst.n // 12))
    while time.time() - t0 < time_limit and it < max_iter:
        it += 1
        frac = (time.time() - t0) / time_limit
        Temp = T0 * (0.01 ** frac)
        k = rng.choice(len(ops), p=w / w.sum())
        op = ops[k]
        uses[k] += 1
        plans = list(cur.plans)
        sites = cur.sites
        removed: list = []
        if op == "site":
            if not site_list:
                continue
            s = int(site_list[int(rng.integers(len(site_list)))])
            sites = frozenset(sites - {s}) if s in sites else frozenset(sites | {s})
            plans, removed = _replan_all(inst, plans, sites, f)
        else:
            q = int(rng.integers(max(2, nq // 3), nq + 1))
            if op == "random":
                flat = [(b, x) for b, pl in enumerate(plans) for x in pl.trips]
                pick = rng.choice(len(flat), size=min(q, len(flat)), replace=False)
                rem = {flat[x][1] for x in pick}
            elif op == "related":
                seed_trip = int(rng.integers(inst.n))
                d = np.abs(inst.start - inst.start[seed_trip]) + 30.0 * (inst.eterm != inst.eterm[seed_trip])
                rem = set(np.argsort(d)[:q].tolist())
            elif op == "block":
                sizes = np.array([len(pl.trips) for pl in plans], float)
                idx = rng.choice(len(plans), size=min(2, len(plans)), replace=False, p=(1 / sizes) / (1 / sizes).sum())
                rem = {x for b in np.atleast_1d(idx) for x in plans[int(b)].trips}
            else:  # worst: trips with the longest connection to their predecessor
                cand = []
                for pl in plans:
                    t = pl.trips
                    for a in range(1, len(t)):
                        cand.append((inst.start[t[a]] - inst.end[t[a - 1]], t[a]))
                cand.sort(reverse=True)
                rem = {x for _, x in cand[:q]}
                if not rem:
                    rem = {int(rng.integers(inst.n))}
            rem = set(int(x) for x in rem)
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
            continue
        cand = State(inst, plans, sites)
        c_new, c_cur = cand.cost, cur.cost
        gain = 0.0
        if c_new < best.cost - 1e-9:
            best = cand
            gain = 33.0
        elif c_new < c_cur - 1e-9:
            gain = 9.0
        elif rng.random() < math.exp(-(c_new - c_cur) / Temp):
            gain = 3.0
        if gain > 0:
            cur = cand
        scores[k] += gain
        if it % seg == 0:
            upd = np.where(uses > 0, scores / np.maximum(uses, 1), 0)
            w = 0.8 * w + 0.2 * (upd + 0.1)
            scores[:] = 0
            uses[:] = 0
            if log is not None:
                log.append((time.time() - t0, it, best.cost))
    if log is not None:
        log.append((time.time() - t0, it, best.cost))
    return Solution(best.plans, best.sites)
