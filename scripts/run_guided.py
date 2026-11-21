"""v0.3 experiments: learned guidance for ALNS. Usage: python scripts/run_guided.py collect | fit | compare

collect  ALNS move logs (features, recomputed cost change) on S and M, seeds 100-103 -> results/guided_moves.npz
fit      gradient boosting on the logs; held-out score by run -> results/guided_surrogate.joblib, guided_fit.json
compare  ALNS, bandit, surrogate, bandit + surrogate on S, M (trained on) and T, L (never seen) -> results/guided_runs.json
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"

from ebus.alns import alns  # noqa: E402
from ebus.guided import FEATURES, guided_alns  # noqa: E402
from ebus.model import Instance, cost_breakdown  # noqa: E402
from ebus.validate import check_solution  # noqa: E402

TRAIN_SEEDS = [100, 101, 102, 103]
TRAIN_T = {"S": 40, "M": 90}
BUDGET = {"S": 30, "T": 30, "M": 60, "L": 120}
SEEDS = {"S": range(10), "T": range(10), "M": range(10), "L": range(3)}
METHODS = ["alns", "bandit", "surrogate", "bandit+surrogate"]


def load(name):
    return Instance.from_json(json.loads((ROOT / "data/instances" / f"{name}.json").read_text()))


def collect_one(args):
    name, seed = args
    moves = []
    guided_alns(load(name), time_limit=TRAIN_T[name], seed=seed, selector="weights", moves=moves)
    return name, seed, moves


def stage_collect():
    tasks = [(n, s) for n in TRAIN_T for s in TRAIN_SEEDS]
    X, y, run, inst = [], [], [], []
    with ProcessPoolExecutor(8) as ex:
        for n, s, mv in ex.map(collect_one, tasks):
            for m in mv:
                X.append(m["features"]); y.append(m["delta"]); run.append(f"{n}{s}"); inst.append(n)
            print(n, s, len(mv), flush=True)
    np.savez(RES / "guided_moves.npz", X=np.array(X), y=np.array(y), run=np.array(run), inst=np.array(inst))


def stage_fit():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    import joblib
    d = np.load(RES / "guided_moves.npz")
    X, y, run = d["X"], d["y"], d["run"]
    # features are on different scales across instances: the cost change is scaled by the instance's initial cost scale
    scale = np.array([np.abs(y[run == r]).mean() for r in run])
    ys = y / np.maximum(scale, 1e-9)
    ups = {"S": 0, "M": 0}
    held = ["S103", "M103"]                       # one run per instance held out for the score
    tr = ~np.isin(run, held)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    gb.fit(X[tr], ys[tr])
    p = gb.predict(X[~tr])
    imp = (y[~tr] < -1e-9)
    info = {"moves": int(len(y)), "train_moves": int(tr.sum()), "held_out_moves": int((~tr).sum()),
            "share_improving_moves": float((y < -1e-9).mean()),
            "held_out_corr": float(np.corrcoef(p, ys[~tr])[0, 1]),
            "held_out_auc_improving": float(roc_auc_score(imp, -p)) if imp.any() and not imp.all() else None,
            "importance": dict(zip(FEATURES, [float(v) for v in gb.feature_importances_]))}
    gb_all = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    gb_all.fit(X, ys)
    joblib.dump(gb_all, RES / "guided_surrogate.joblib")
    (RES / "guided_fit.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def run_one(args):
    import joblib
    name, method, seed = args
    inst = load(name)
    sur = joblib.load(RES / "guided_surrogate.joblib") if "surrogate" in method else None
    tr = []
    t = time.time()
    if method == "alns":
        sol = alns(inst, time_limit=BUDGET[name], seed=seed, log=tr)
        st = {}
    else:
        sol, st = guided_alns(inst, time_limit=BUDGET[name], seed=seed, selector="ucb" if "bandit" in method else "weights",
                              surrogate=sur, trace=tr)
    cb = cost_breakdown(inst, sol)               # recomputed by the original cost function, independent of the search
    return {"inst": name, "method": method, "seed": seed, "cost": float(cb["cost"]), "valid": not check_solution(inst, sol), "errors": check_solution(inst, sol)[:3],
            "trace": [(float(a), int(b), float(c)) for a, b, c in tr[::max(1, len(tr) // 40)]], **{k: int(v) for k, v in st.items()}}


def stage_compare():
    only = sys.argv[2] if len(sys.argv) > 2 else None
    tasks = [(n, m, s) for n in ("L", "M", "S", "T") for m in METHODS for s in SEEDS[n] if only in (None, n)]
    out = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(run_one, tasks):
            out.append(r)
            print(r["inst"], r["method"], r["seed"], round(r["cost"], 1), flush=True)
    (RES / (f"guided_runs_{only}.json" if only else "guided_runs.json")).write_text(json.dumps(out))


if __name__ == "__main__":
    {"collect": stage_collect, "fit": stage_fit, "compare": stage_compare}[sys.argv[1]]()
