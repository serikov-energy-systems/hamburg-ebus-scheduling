"""VRPTW benchmark: exact methods on small subsets, Python against C++ speed, all 56 Solomon instances.

    python scripts/run_vrptw_bench.py exact | speed | full | all
Results go to results/vrptw/*.csv and *.json; every run records its time (the telemetry the CI job keeps).
"""
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import vrptw  # noqa: E402
from vrptw import exact, instance as ins  # noqa: E402

OUT = ROOT / "results" / "vrptw"
OUT.mkdir(parents=True, exist_ok=True)
BIG = 1e4


def run_exact():
    rows = []
    for name in ("C101", "R101", "RC101", "C201", "R201", "RC201"):
        for k in (8, 10, 12):
            I = ins.load(name).subset(k)
            rec = dict(instance=I.name, customers=k)
            for lab, fn in (("cpsat", exact.solve_cpsat), ("highs", exact.solve_highs), ("scip", exact.solve_scip)):
                t = time.perf_counter()
                try:
                    r = fn(I, time_limit=30)
                    obj = r["objective"]
                    ok = bool(obj is not None and ins.check(I, r["routes"])["feasible"])
                    rec[f"{lab}_objective"] = None if obj is None else round(obj, 3)
                    rec[f"{lab}_optimal"] = bool(r.get("optimal"))
                    rec[f"{lab}_feasible"] = ok
                except Exception as e:  # report, do not hide
                    rec[f"{lab}_objective"], rec[f"{lab}_optimal"], rec[f"{lab}_feasible"] = None, False, f"error {type(e).__name__}"
                rec[f"{lab}_seconds"] = round(time.perf_counter() - t, 2)
            alns = []
            for seed in range(3):
                r, d, v, ms = vrptw.solve(I, iterations=2000, seed=seed)
                assert ins.check(I, r)["feasible"]
                alns.append(d + BIG * v)
            rec["alns_best"] = round(min(alns), 3)
            rec["alns_worst"] = round(max(alns), 3)
            rows.append(rec)
            print(rec, flush=True)
    keys = list(rows[0])
    with open(OUT / "exact_small.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def run_speed():
    rows = []
    for name in ("C101", "R101", "RC101"):
        I = ins.load(name)
        it = 100
        rp = vrptw.solve(I, iterations=it, seed=1, backend="python")
        rc = vrptw.solve(I, iterations=it, seed=1, backend="cpp")
        rows.append(dict(instance=name, iterations=it, python_ms=round(rp[3], 1), cpp_ms=round(rc[3], 1),
                         speedup=round(rp[3] / rc[3], 1), python_ms_per_iteration=round(rp[3] / it, 2),
                         cpp_ms_per_iteration=round(rc[3] / it, 3)))
        print(rows[-1], flush=True)
    json.dump(dict(rows=rows, median_speedup=st.median(r["speedup"] for r in rows)), open(OUT / "speed.json", "w"), indent=1)


def _one(args):
    name, seed, iters = args
    bks = ins.bks()
    I = ins.load(name)
    r, d, v, ms = vrptw.solve(I, iterations=iters, seed=seed)
    chk = ins.check(I, r)
    assert chk["feasible"], (name, chk["problems"])
    b = bks[name.upper()]
    return dict(instance=name, seed=seed, iterations=iters, vehicles=v, distance=round(d, 2), bks_vehicles=b["vehicles"],
                bks_distance=b["distance"], ms=round(ms), extra_vehicles=v - b["vehicles"],
                gap_pct=round(100 * (d / b["distance"] - 1), 3) if v == b["vehicles"] else None)


def run_full(iters=4000, seeds=(0, 1, 2)):
    from multiprocessing import Pool

    tasks = [(n, s, iters) for n in ins.names() for s in seeds]
    with Pool(6) as pool:
        rows = pool.map(_one, tasks, chunksize=1)
    rows.sort(key=lambda r: (r["instance"], r["seed"]))
    with open(OUT / "solomon_full.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    groups = {"C1": "C1", "C2": "C2", "R1": "R1", "R2": "R2", "RC1": "RC1", "RC2": "RC2"}
    for g in groups:
        sel = [r for r in rows if r["instance"].startswith(g) and (g not in ("R1", "R2") or not r["instance"].startswith("RC"))]
        if not sel:
            continue
        best = {}
        for r in sel:
            k = r["instance"]
            if k not in best or (r["vehicles"], r["distance"]) < (best[k]["vehicles"], best[k]["distance"]):
                best[k] = r
        summary[g] = dict(
            instances=len(best), runs=len(sel),
            best_of_seeds_matching_bks_vehicles=sum(1 for b in best.values() if b["extra_vehicles"] <= 0),
            best_of_seeds_mean_extra_vehicles=round(st.mean(b["extra_vehicles"] for b in best.values()), 3),
            mean_gap_pct_where_vehicles_match=round(st.mean(b["gap_pct"] for b in best.values() if b["gap_pct"] is not None), 3)
            if any(b["gap_pct"] is not None for b in best.values()) else None,
            mean_ms_per_run=round(st.mean(r["ms"] for r in sel)),
        )
    json.dump(summary, open(OUT / "solomon_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


def run_quick():
    """CI telemetry: six instances, one seed, 500 iterations; time and quality of every run."""
    bks = ins.bks()
    rows = []
    for name in ("C101", "C201", "R101", "R201", "RC101", "RC201"):
        I = ins.load(name)
        r, d, v, ms = vrptw.solve(I, iterations=500, seed=0)
        assert ins.check(I, r)["feasible"]
        rows.append(dict(instance=name, vehicles=v, distance=round(d, 2), bks_vehicles=bks[name]["vehicles"],
                         bks_distance=bks[name]["distance"], ms=round(ms)))
    json.dump(rows, open(OUT / "ci_quick.json", "w"), indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "quick":
        run_quick()
        sys.exit(0)
    if mode in ("exact", "all"):
        run_exact()
    if mode in ("speed", "all"):
        run_speed()
    if mode in ("full", "all"):
        run_full()
