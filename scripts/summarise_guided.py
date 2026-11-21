"""Table and figure of the v0.3 comparison from results/guided_runs.json (M rows replaced by guided_runs_M.json)."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
runs = json.loads((RES / "guided_runs.json").read_text())
runs = [r for r in runs if r["inst"] != "M"] + json.loads((RES / "guided_runs_M.json").read_text())
df = pd.DataFrame(runs)
bad = df[~df["valid"]]
ok = df[df["valid"]]
order = ["alns", "bandit", "surrogate", "bandit+surrogate"]
rows = []
for inst in ["S", "T", "M", "L"]:
    base = ok[(ok.inst == inst) & (ok.method == "alns")]["cost"].to_numpy()
    for m in order:
        x = ok[(ok.inst == inst) & (ok.method == m)]
        c = x["cost"].to_numpy()
        p = stats.mannwhitneyu(c, base).pvalue if m != "alns" else np.nan
        rows.append({"instance": inst, "method": m, "runs": len(c), "mean": c.mean(), "sd": c.std(ddof=1), "best": c.min(),
                     "vs_alns_pct": 100 * (c.mean() - base.mean()) / base.mean() if m != "alns" else 0.0, "p_mannwhitney": p,
                     "iterations": x["iterations"].mean() if m != "alns" else np.nan,
                     "pred_improve": int(x["pred_improve"].sum()) if "surrogate" in m else None,
                     "pred_improve_true": int(x["pred_improve_true"].sum()) if "surrogate" in m else None})
tab = pd.DataFrame(rows)
tab.to_csv(RES / "guided_summary.csv", index=False, float_format="%.3f")
bad[["inst", "method", "seed", "cost", "errors"]].to_csv(RES / "guided_invalid_runs.csv", index=False)
print(tab.round(2).to_string())
print(bad[["inst", "method", "seed", "cost", "errors"]].to_string())
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
for a, inst in zip(ax, ["S", "T", "M", "L"]):
    data = [ok[(ok.inst == inst) & (ok.method == m)]["cost"] for m in order]
    a.boxplot(data, tick_labels=["ALNS", "bandit", "surr.", "b+s"], widths=0.6)
    a.set_title(f"{inst} ({'trained on' if inst in 'SM' else 'unseen'})", fontsize=9)
    a.tick_params(labelsize=7)
ax[0].set_ylabel("daily cost, EUR")
fig.tight_layout()
fig.savefig(RES / "fig_guided.png", dpi=130)
