"""Instance, parameters and the exact evaluation of a schedule.

A schedule is a set of vehicle blocks (ordered trips) and a set of open charging terminals.
Between two consecutive trips of a block the bus either drives directly (and is connected to a
charger when the terminal has one) or visits the depot, where it charges at depot power.

Occupancy rule used by every solver: a bus that waits for at least `tmin_chg` minutes at an
open charging terminal holds one charge point from arrival until it leaves for the next trip,
at most `t_cap` minutes. The number of charge points of a terminal is the largest number of
buses holding one at the same time.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

from .gtfs import haversine_km


@dataclass(frozen=True)
class Params:
    cons: float = 1.6          # kWh per km, 12 m bus incl. heating at about 5 C, the Hamburg winter median (assumed)
    cap_kwh: float = 350.0     # nominal battery (assumed)
    smin_frac: float = 0.10    # reserve floor
    smax_frac: float = 0.90    # charging ceiling
    p_opp: float = 300.0       # kW pantograph at terminals
    p_dep: float = 120.0       # kW average depot charger (mid-day top-up)
    overhead_min: float = 2.0  # connection and disconnection
    tmin_chg: float = 5.0      # a wait shorter than this does not use a charger
    t_cap: float = 60.0        # longest charge occupancy of one visit
    layover_min: float = 2.0   # minimum wait between trips
    wait_max: float = 90.0     # longest direct wait between two trips of a block
    dep_gap_min: float = 25.0  # slack needed for a visit to the depot (beyond driving)
    dep_gap_max: float = 150.0
    speed_kmh: float = 25.0    # deadhead speed
    circuity: float = 1.3      # road distance over straight line
    # costs per day (EUR), annualised and assumed
    c_veh: float = 120.0       # one e-bus incl. depot charger and maintenance
    c_site: float = 40.0       # one charging terminal (grid connection)
    c_cp: float = 90.0         # one charge point (pantograph and charger)
    c_dh_km: float = 1.1       # driver time and wear per deadhead km
    c_kwh: float = 0.22        # electricity
    c_resc: float = 2.5        # EUR per kWh of emergency energy (tow, replacement bus, quick charge)

    @property
    def smin(self): return self.smin_frac * self.cap_kwh

    @property
    def smax(self): return self.smax_frac * self.cap_kwh

    @property
    def window(self): return self.smax - self.smin


@dataclass
class Instance:
    name: str
    trips: pd.DataFrame           # trip_id, route_short_name, start_min, end_min, length_km, sterm, eterm
    term_name: list
    term_lat: np.ndarray
    term_lon: np.ndarray
    depot: int                    # index into terminals (depot sits at this terminal location)
    params: Params = field(default_factory=Params)

    def __post_init__(self):
        t = self.trips
        self.n = len(t)
        self.start = t.start_min.to_numpy(float)
        self.end = t.end_min.to_numpy(float)
        self.km = t.length_km.to_numpy(float)
        self.sterm = t.sterm.to_numpy(int)
        self.eterm = t.eterm.to_numpy(int)
        self.T = len(self.term_name)
        p = self.params
        lat, lon = self.term_lat, self.term_lon
        d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]) * p.circuity
        self.dh_km = d
        self.dh_min = np.where(d > 0, np.maximum(2.0, d / p.speed_kmh * 60.0), 0.0)
        self.e_trip = self.km * p.cons

    # ---- derived arc data -------------------------------------------------
    def direct_ok(self, i, j):
        p = self.params
        gap = self.start[j] - self.end[i]
        dm = self.dh_min[self.eterm[i], self.sterm[j]]
        return p.layover_min + dm <= gap <= p.wait_max + dm

    def depot_ok(self, i, j):
        p = self.params
        gap = self.start[j] - self.end[i]
        need = self.dh_min[self.eterm[i], self.depot] + self.dh_min[self.depot, self.sterm[j]] + p.dep_gap_min
        return need <= gap <= p.dep_gap_max + need

    def direct_tau(self, i, j):
        """Minutes available at the terminal of trip i before driving to the start of j."""
        return self.start[j] - self.end[i] - self.dh_min[self.eterm[i], self.sterm[j]]

    def occupancy(self, i, j):
        p = self.params
        return max(0.0, min(self.direct_tau(i, j), p.t_cap))

    def cap_direct(self, i, j):
        """Maximum energy (kWh) charged at the terminal of i during the wait before j."""
        p = self.params
        tau = self.direct_tau(i, j)
        if tau < p.tmin_chg:
            return 0.0
        return min(p.window, p.p_opp * (min(tau, p.t_cap) - p.overhead_min) / 60.0)

    def cap_depot(self, i, j):
        p = self.params
        tau = (self.start[j] - self.end[i] - self.dh_min[self.eterm[i], self.depot]
               - self.dh_min[self.depot, self.sterm[j]])
        return min(p.window, max(0.0, p.p_dep * (tau - p.overhead_min) / 60.0))

    def km_direct(self, i, j): return self.dh_km[self.eterm[i], self.sterm[j]]
    def km_depot(self, i, j): return self.dh_km[self.eterm[i], self.depot] + self.dh_km[self.depot, self.sterm[j]]
    def km_out(self, i): return self.dh_km[self.depot, self.sterm[i]]
    def km_in(self, i): return self.dh_km[self.eterm[i], self.depot]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "terminals": [{"name": n, "lat": float(a), "lon": float(b)}
                          for n, a, b in zip(self.term_name, self.term_lat, self.term_lon)],
            "depot": int(self.depot),
            "trips": self.trips[["trip_id", "route_short_name", "start_min", "end_min", "length_km", "sterm", "eterm"]]
                     .to_dict("records"),
            "params": asdict(self.params),
        }

    @staticmethod
    def from_json(d: dict, params: Params | None = None) -> "Instance":
        tr = pd.DataFrame(d["trips"])
        term = d["terminals"]
        return Instance(
            name=d.get("name", "instance"),
            trips=tr,
            term_name=[t["name"] for t in term],
            term_lat=np.array([t["lat"] for t in term], float),
            term_lon=np.array([t["lon"] for t in term], float),
            depot=int(d["depot"]),
            params=params or Params(**d.get("params", {})),
        )

    def with_params(self, **kw) -> "Instance":
        from dataclasses import replace
        return Instance(self.name, self.trips, self.term_name, self.term_lat, self.term_lon,
                        self.depot, replace(self.params, **kw))


def make_instance(name: str, trips: pd.DataFrame, params: Params | None = None,
                  depot_name: str | None = None) -> Instance:
    """Terminals from stop names; the depot sits at the terminal that minimises the trip-weighted distance."""
    params = params or Params()
    trips = trips.copy()
    names = sorted(set(trips.start_name) | set(trips.end_name))
    idx = {n: k for k, n in enumerate(names)}
    lat = np.zeros(len(names))
    lon = np.zeros(len(names))
    for n in names:
        a = trips[trips.start_name == n]
        b = trips[trips.end_name == n]
        la = np.concatenate([a.start_lat.to_numpy(), b.end_lat.to_numpy()])
        lo = np.concatenate([a.start_lon.to_numpy(), b.end_lon.to_numpy()])
        lat[idx[n]], lon[idx[n]] = la.mean(), lo.mean()
    trips["sterm"] = trips.start_name.map(idx)
    trips["eterm"] = trips.end_name.map(idx)
    if depot_name is None:
        w = np.bincount(np.concatenate([trips.sterm, trips.eterm]), minlength=len(names)).astype(float)
        d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        depot = int(np.argmin(d @ w))
    else:
        depot = idx[depot_name]
    return Instance(name, trips.reset_index(drop=True), names, lat, lon, depot, params)


# =========================================================================
# Block planning: dynamic programme over the energy requirement
# =========================================================================

@dataclass
class BlockPlan:
    trips: list
    conn: list                 # 'D' direct, 'V' via depot, len = len(trips) - 1
    dh_km: float
    n_visits: int


def plan_block(inst: Instance, trips: list, sites, f: float = 1.0):
    """Plan of the connections of a block with the least deadhead that keeps the battery
    between the reserve floor and the ceiling, or None when there is none.
    Energy per km is scaled by f. Backward dynamic programme over the energy needed at the
    start of each trip, indexed by the number of depot visits that follow."""
    p = inst.params
    m = len(trips)
    smin, smax = p.smin, p.smax
    cf = p.cons * f
    INF = float("inf")
    VMAX = 4
    e = [inst.km[i] * cf for i in trips]
    last = trips[-1]
    r = [[INF] * (VMAX + 1) for _ in range(m)]
    choice = [[None] * (VMAX + 1) for _ in range(m)]
    ra0 = smin + inst.km_in(last) * cf
    if ra0 + e[-1] <= smax:
        r[m - 1][0] = ra0 + e[-1]
    for k in range(m - 2, -1, -1):
        i, j = trips[k], trips[k + 1]
        okd = inst.direct_ok(i, j)
        cap = edh = 0.0
        if okd:
            cap = inst.cap_direct(i, j) if inst.eterm[i] in sites else 0.0
            edh = inst.km_direct(i, j) * cf
        okv = inst.depot_ok(i, j)
        capv = ekd = edv = 0.0
        if okv:
            capv = inst.cap_depot(i, j)
            ekd = inst.km_in(i) * cf
            edv = inst.dh_km[inst.depot, inst.sterm[j]] * cf
        for v in range(VMAX + 1):
            rn = r[k + 1][v]
            if rn < INF and okd:
                rs = max(smin, rn + edh - cap) + e[k]
                if rs <= smax and rs < r[k][v]:
                    r[k][v] = rs
                    choice[k][v] = ("D", v)
            if v >= 1:
                rn = r[k + 1][v - 1]
                if rn < INF and okv:
                    rs = ekd + max(smin, rn + edv - capv) + e[k]
                    if rs <= smax and rs < r[k][v]:
                        r[k][v] = rs
                        choice[k][v] = ("V", v - 1)
    best = None
    for v in range(VMAX + 1):
        if r[0][v] + inst.km_out(trips[0]) * cf <= smax + 1e-9:
            conn = []
            vv = v
            for k in range(m - 1):
                c, vv2 = choice[k][vv]
                conn.append(c)
                vv = vv2
            km = block_conn_km(inst, trips, conn)
            if best is None or km < best.dh_km - 1e-9:
                best = BlockPlan(list(trips), conn, km, v)
    return best


def block_conn_km(inst: Instance, trips: list, conn: list) -> float:
    km = inst.km_out(trips[0]) + inst.km_in(trips[-1])
    for k, c in enumerate(conn):
        km += inst.km_direct(trips[k], trips[k + 1]) if c == "D" else inst.km_depot(trips[k], trips[k + 1])
    return km


# =========================================================================
# Solutions and their cost
# =========================================================================

@dataclass
class Solution:
    plans: list
    sites: frozenset

    def blocks(self):
        return [pl.trips for pl in self.plans]


def charge_points(inst: Instance, plans: list, sites) -> dict:
    """Largest number of buses holding a charge point at once, per open terminal."""
    ev: dict = {}
    for pl in plans:
        for k, c in enumerate(pl.conn):
            if c != "D":
                continue
            i, j = pl.trips[k], pl.trips[k + 1]
            s = inst.eterm[i]
            if s in sites and inst.direct_tau(i, j) >= inst.params.tmin_chg:
                a = inst.end[i]
                b = a + inst.occupancy(i, j)
                ev.setdefault(s, []).extend([(a, 1), (b, -1)])
    out = {}
    for s in sites:
        cur = best = 0
        for _, d in sorted(ev.get(s, []), key=lambda x: (x[0], x[1])):
            cur += d
            best = max(best, cur)
        out[s] = best
    return out


def cost_breakdown(inst: Instance, sol: Solution) -> dict:
    p = inst.params
    cp = charge_points(inst, sol.plans, sol.sites)
    used = [s for s, k in cp.items() if k > 0]
    nveh = len(sol.plans)
    dh = sum(pl.dh_km for pl in sol.plans)
    dh_kwh = dh * p.cons
    cost = (p.c_veh * nveh + p.c_site * len(used) + p.c_cp * sum(cp.values())
            + p.c_dh_km * dh + p.c_kwh * dh_kwh)   # trip energy is the same in every schedule
    return {
        "vehicles": nveh,
        "sites": len(used),
        "charge_points": int(sum(cp.values())),
        "deadhead_km": dh,
        "cost_vehicles": p.c_veh * nveh,
        "cost_chargers": p.c_site * len(used) + p.c_cp * sum(cp.values()),
        "cost_deadhead": p.c_dh_km * dh + p.c_kwh * dh_kwh,
        "cost": cost,
    }


# =========================================================================
# Simulation under another consumption level (second stage)
# =========================================================================

def simulate_plan(inst: Instance, pl: BlockPlan, f: float, sites):
    """Run a block with consumption factor f. Returns (shortfall kWh, minimum SOC reached).
    Charging is the maximum the layover allows; when the battery would fall below the reserve
    floor the shortfall is bought as emergency energy and the floor is restored."""
    p = inst.params
    cf = p.cons * f
    trips = pl.trips
    soc = p.smax - inst.km_out(trips[0]) * cf
    short = 0.0
    lo = soc

    def floor(s):
        nonlocal short
        if s < p.smin:
            short += p.smin - s
            return p.smin
        return s

    soc = floor(soc)
    for k, i in enumerate(trips):
        soc = floor(soc - inst.km[i] * cf)
        lo = min(lo, soc)
        if k == len(trips) - 1:
            soc = floor(soc - inst.km_in(i) * cf)
            break
        j = trips[k + 1]
        if pl.conn[k] == "D":
            if inst.eterm[i] in sites:
                soc = min(p.smax, soc + inst.cap_direct(i, j))
            soc = floor(soc - inst.km_direct(i, j) * cf)
        else:
            soc = floor(soc - inst.km_in(i) * cf)
            soc = min(p.smax, soc + inst.cap_depot(i, j))
            soc = floor(soc - inst.dh_km[inst.depot, inst.sterm[j]] * cf)
        lo = min(lo, soc)
    return short, lo


def expected_shortfall(inst: Instance, sol: Solution, factors, probs):
    per = []
    for f in factors:
        per.append(sum(simulate_plan(inst, pl, f, sol.sites)[0] for pl in sol.plans))
    return float(np.dot(per, probs)), per


# =========================================================================
# Baseline: fleet without range limit (the diesel-like vehicle scheduling problem)
# =========================================================================

def min_fleet_unlimited(inst: Instance) -> int:
    """Minimum number of vehicles when the battery does not matter: maximum matching of
    compatible trip pairs (a bus may wait up to `wait_max` between trips)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    n = inst.n
    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if inst.start[j] - inst.end[i] > inst.params.wait_max + 60:
                break
            if inst.direct_ok(i, j):
                rows.append(i)
                cols.append(j)
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    match = maximum_bipartite_matching(g, perm_type="column")
    return n - int((match >= 0).sum())
