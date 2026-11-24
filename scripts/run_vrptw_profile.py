"""Where does the time go in the C++ kernel, and what did the O(1) insertion test change?

Runs the naive and the fast variant on the same seeds (they must return identical routes), reports the share of the
time spent in the feasibility test, destroy and the rest of repair, and the speed-up. Results: results/vrptw/profile.json
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import vrptw  # noqa: E402
from vrptw import instance as ins  # noqa: E402

OUT = ROOT / "results" / "vrptw"
rows = []
for name in ("C101", "R101", "RC101", "C201", "R201", "RC201"):
    I = ins.load(name)
    rec = dict(instance=name)
    for label, fast in (("naive", False), ("fast", True)):
        times, shares, same = [], [], []
        for seed in (1, 2, 3):
            r, d, v, ms, prof = vrptw.profile(I, iterations=500, seed=seed, fast=fast)
            plain = vrptw.solve(I, iterations=500, seed=seed, fast=fast)
            times.append(plain[3])
            tot = prof["total_seconds"]
            shares.append(dict(feasibility=prof["feasibility_seconds"] / tot, destroy=prof["destroy_seconds"] / tot,
                               repair_total=prof["repair_seconds"] / tot, calls=prof["feasibility_calls"]))
            same.append(r)
            rec[f"{label}_routes_seed{seed}"] = [len(x) for x in r]
        rec[f"{label}_ms_median"] = round(st.median(times), 1)
        rec[f"{label}_feasibility_share"] = round(st.mean(s["feasibility"] for s in shares), 3)
        rec[f"{label}_destroy_share"] = round(st.mean(s["destroy"] for s in shares), 3)
        rec[f"{label}_repair_share"] = round(st.mean(s["repair_total"] for s in shares), 3)
        rec[f"{label}_feasibility_calls_per_run"] = int(st.mean(s["calls"] for s in shares))
        rec[f"_{label}_solutions"] = same
    rec["identical_routes_all_seeds"] = rec.pop("_naive_solutions") == rec.pop("_fast_solutions")
    rec["speedup_fast_vs_naive"] = round(rec["naive_ms_median"] / rec["fast_ms_median"], 2)
    for k in [k for k in rec if "_routes_seed" in k]:
        del rec[k]
    rows.append(rec)
    print(rec, flush=True)
json.dump(dict(rows=rows, median_speedup=st.median(r["speedup_fast_vs_naive"] for r in rows)), open(OUT / "profile.json", "w"), indent=1)
