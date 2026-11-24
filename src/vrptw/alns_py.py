"""Pure Python reference of the ALNS in cpp/alns_core.cpp: same operators, same weights, same schedule.

Cost = total distance + BIG * routes (fewer vehicles first). The C++ kernel and this code use different
random number generators, so the same seed does not give the same route; they are compared by speed
(iterations per second) and by the distribution of results over seeds.
"""
from __future__ import annotations

import math
import random
import time

BIG = 1.0e4
INF = float("inf")


class _I:
    def __init__(self, x, y, dem, ready, due, serv, cap):
        self.n = len(x) - 1
        self.cap, self.dem, self.ready, self.due, self.serv = cap, dem, ready, due, serv
        self.d = [[math.hypot(x[i] - x[j], y[i] - y[j]) for j in range(self.n + 1)] for i in range(self.n + 1)]


def _feasible(I, r, pos, c, load):
    if load + I.dem[c] > I.cap + 1e-9:
        return False
    t, prev, d = 0.0, 0, I.d
    for k in range(len(r) + 1):
        v = c if k == pos else (r[k] if k < pos else r[k - 1])
        t = max(t + d[prev][v], I.ready[v])
        if t > I.due[v] + 1e-9:
            return False
        t += I.serv[v]
        prev = v
    return t + d[prev][0] <= I.due[0] + 1e-9


def _dist(I, s):
    tot = 0.0
    for r in s:
        prev = 0
        for v in r:
            tot += I.d[prev][v]
            prev = v
        tot += I.d[prev][0]
    return tot


def _cost(I, s):
    return _dist(I, s) + BIG * len(s)


def _best_insert(I, s, loads, c):
    best, bi, bp, sec = INF, -1, -1, INF
    for ri, r in enumerate(s):
        rb, rp = INF, -1
        for p in range(len(r) + 1):
            a = 0 if p == 0 else r[p - 1]
            b = 0 if p == len(r) else r[p]
            delta = I.d[a][c] + I.d[c][b] - I.d[a][b]
            if delta >= rb:
                continue
            if _feasible(I, r, p, c, loads[ri]):
                rb, rp = delta, p
        if rp >= 0:
            if rb < best:
                sec, best, bi, bp = best, rb, ri, rp
            elif rb < sec:
                sec = rb
    return best, bi, bp, sec


def _repair(I, s, U, regret):
    loads = [sum(I.dem[v] for v in r) for r in s]
    U = list(U)
    while U:
        pick, pkey, pb = -1, -INF, None
        for k, c in enumerate(U):
            b = _best_insert(I, s, loads, c)
            delta, ri, _, sec = b
            if ri < 0:
                key = 1.0e9 if regret else -(2 * I.d[0][c] + BIG)
            elif regret:
                key = (1.0e8 if sec >= INF else sec) - delta
            else:
                key = -delta
            if key > pkey:
                pkey, pick, pb = key, k, b
        c = U.pop(pick)
        _, ri, p, _ = pb
        if ri < 0:
            s.append([c])
            loads.append(I.dem[c])
        else:
            s[ri].insert(p, c)
            loads[ri] += I.dem[c]


def _initial(I):
    order = sorted(range(1, I.n + 1), key=lambda c: I.due[c])
    s = []
    _repair(I, s, order, False)
    return s


def _destroy(I, s, op, q, rng):
    removed = []
    if op == 0:
        allc = [v for r in s for v in r]
        rng.shuffle(allc)
        removed = allc[:q]
    elif op == 1:
        gains = []
        for r in s:
            for p, c in enumerate(r):
                a = 0 if p == 0 else r[p - 1]
                b = 0 if p + 1 == len(r) else r[p + 1]
                gains.append((I.d[a][c] + I.d[c][b] - I.d[a][b], c))
        gains.sort(key=lambda g: -g[0])
        for _ in range(min(q, len(gains))):
            idx = int(rng.random() ** 3 * len(gains))
            removed.append(gains.pop(idx)[1])
    elif op == 2:
        allc = [v for r in s for v in r]
        seed = allc[rng.randrange(len(allc))]
        rel = sorted((I.d[seed][c] / 100.0 + abs(I.ready[seed] - I.ready[c]) / 100.0 +
                      abs(I.dem[seed] - I.dem[c]) / I.cap, c) for c in allc)
        removed = [c for _, c in rel[:q]]
    else:
        best = min(range(len(s)), key=lambda ri: len(s[ri]))
        removed = list(s[best])
    removed = sorted(set(removed))
    rs = set(removed)
    for r in s:
        r[:] = [v for v in r if v not in rs]
    s[:] = [r for r in s if r]
    return removed


def solve(x, y, dem, ready, due, serv, cap, iterations, seed):
    t0 = time.perf_counter()
    I = _I(x, y, dem, ready, due, serv, cap)
    rng = random.Random(seed)
    cur = _initial(I)
    best = [list(r) for r in cur]
    cc = bc = _cost(I, cur)
    T0 = 50.0
    T = T0
    cool = (0.01 / T0) ** (1.0 / max(1, iterations))
    wd, wr = [1.0] * 4, [1.0] * 2
    sd, sr, ud, ur = [0.0] * 4, [0.0] * 2, [0] * 4, [0] * 2

    def pick(w):
        r = rng.random() * sum(w)
        a = 0.0
        for k, v in enumerate(w):
            a += v
            if r <= a:
                return k
        return len(w) - 1

    qmax = max(4, min(40, int(0.3 * I.n)))
    qmin = max(3, int(0.1 * I.n))
    for it in range(iterations):
        od, orr = pick(wd), pick(wr)
        cand = [list(r) for r in cur]
        q = rng.randint(qmin, qmax)
        U = _destroy(I, cand, od, q, rng)
        _repair(I, cand, U, orr == 1)
        nc = _cost(I, cand)
        score = 0.0
        if nc < bc - 1e-9:
            best, bc, score, cur, cc = [list(r) for r in cand], nc, 33.0, cand, nc
        elif nc < cc - 1e-9:
            cur, cc, score = cand, nc, 9.0
        elif rng.random() < math.exp(-(nc - cc) / max(T, 1e-9)):
            cur, cc, score = cand, nc, 3.0
        sd[od] += score
        ud[od] += 1
        sr[orr] += score
        ur[orr] += 1
        if (it + 1) % 100 == 0:
            for k in range(4):
                if ud[k]:
                    wd[k] = 0.9 * wd[k] + 0.1 * sd[k] / ud[k]
                wd[k] = max(wd[k], 0.05)
                sd[k], ud[k] = 0.0, 0
            for k in range(2):
                if ur[k]:
                    wr[k] = 0.9 * wr[k] + 0.1 * sr[k] / ur[k]
                wr[k] = max(wr[k], 0.05)
                sr[k], ur[k] = 0.0, 0
        T *= cool
    return best, _dist(I, best), len(best), (time.perf_counter() - t0) * 1000.0
