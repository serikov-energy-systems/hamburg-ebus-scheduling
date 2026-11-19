"""Figures from results/*.json and a fresh ALNS run for the battery traces."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"

from ebus.alns import alns  # noqa: E402
from ebus.model import Instance, cost_breakdown  # noqa: E402
from ebus.stoch import scenarios_from_temperatures  # noqa: E402
from ebus.validate import soc_trace  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C = {"alns": "#2a6f97", "cpsat": "#e07a1f", "milp": "#6a994e", "base": "#999999"}


def fig_solvers():
    d = json.loads((RES / "solvers.json").read_text())
    names = [n for n in "STML" if n in d]
    fig, ax = plt.subplots(1, len(names), figsize=(2.6 * len(names), 3.2))
    for a, n in zip(np.atleast_1d(ax), names):
        r = d[n]
        ref = min(x["cost"] for x in r["alns"])
        vals, labs, cols = [], [], []
        vals.append(r["greedy_all_terminals"]["cost"]); labs.append("greedy"); cols.append(C["base"])
        if "cost" in r.get("depot_only", {}):
            vals.append(r["depot_only"]["cost"]); labs.append("no\nchargers"); cols.append(C["base"])
        al = [x["cost"] for x in r["alns"]]
        vals.append(np.mean(al)); labs.append("ALNS"); cols.append(C["alns"])
        for key, lab in (("cpsat", "CP-SAT"), ("milp", "MILP")):
            if key in r and r[key].get("reeval"):
                vals.append(r[key]["reeval"]["cost"]); labs.append(lab); cols.append(C[key])
        a.bar(range(len(vals)), [v / ref for v in vals], color=cols)
        a.errorbar([labs.index("ALNS")], [np.mean(al) / ref], yerr=[[np.mean(al) / ref - min(al) / ref], [max(al) / ref - np.mean(al) / ref]], color="k", capsize=3)
        a.set_xticks(range(len(vals)))
        a.set_xticklabels(labs, fontsize=7)
        lo = min(0.97, min(v / ref for v in vals) - 0.03)
        a.set_ylim(lo, max(v / ref for v in vals) * 1.03)
        a.set_title(f"{n}: {r['trips']} trips", fontsize=9)
    np.atleast_1d(ax)[0].set_ylabel("daily cost / best ALNS run")
    fig.tight_layout()
    fig.savefig(RES / "fig_solvers.png", dpi=170)
    plt.close(fig)


def fig_convergence():
    d = json.loads((RES / "solvers.json").read_text())
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8))
    for a, n in zip(ax, "STML"):
        if n not in d:
            continue
        ref = min(x["cost"] for x in d[n]["alns"])
        for run in d[n]["alns"]:
            t = [x[0] for x in run["trace"]]
            c = [x[2] / ref for x in run["trace"]]
            a.plot(t, c, lw=1, color=C["alns"], alpha=0.7)
        cp = d[n].get("cpsat", {})
        if cp.get("objective"):
            a.axhline(cp["objective"] / ref, color=C["cpsat"], ls="--", lw=1, label="CP-SAT")
        a.set_title(n)
        a.set_xlabel("seconds")
        top = max(x["trace"][min(2, len(x["trace"]) - 1)][2] for x in d[n]["alns"]) / ref
        a.set_ylim(0.995, min(top, 1.15))
    ax[0].set_ylabel("cost / best")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "fig_convergence.png", dpi=170)
    plt.close(fig)


def fig_margin():
    d = json.loads((RES / "margin.json").read_text())
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.9))
    for a, n in zip(ax, "STML"):
        if n not in d["design"]:
            continue
        rows = d["design"][n]
        x = [r["design_factor"] for r in rows]
        base = rows[0]["cost"]
        a.plot(x, [r["cost"] for r in rows], color=C["alns"], label="schedule cost")
        a.plot(x, [r["exp_cost"] for r in rows], color=C["cpsat"], label="plus expected emergency energy")
        k = int(np.argmin([r["exp_cost"] for r in rows]))
        a.plot(x[k], rows[k]["exp_cost"], "o", color="k", ms=4)
        a.axvline(d["mean_factor"], color=C["base"], ls=":", lw=1)
        a.set_title(n)
        a.set_xlabel("design consumption factor")
    ax[0].set_ylabel("EUR per day")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "fig_margin.png", dpi=170)
    plt.close(fig)


def fig_battery():
    inst = Instance.from_json(json.loads((ROOT / "data/instances/T.json").read_text()))
    sol = alns(inst, time_limit=20, seed=0)
    scen = scenarios_from_temperatures(str(ROOT / "data/hamburg_winter_temps.csv"))
    fs = [1.0, scen[-1][0]]
    fig, ax = plt.subplots(len(sol.plans), 1, figsize=(7.5, 1.6 * len(sol.plans)), sharex=True)
    p = inst.params
    for a, pl in zip(np.atleast_1d(ax), sol.plans):
        for f, col in zip(fs, (C["alns"], C["cpsat"])):
            pts = soc_trace(inst, pl, f, sol.sites)
            a.plot([t / 60 for t, _ in pts], [s for _, s in pts], color=col, lw=1,
                   label=f"consumption factor {f:.2f}")
        a.axhline(p.smin, color="#aa3333", lw=0.8, ls="--")
        a.axhline(p.smax, color=C["base"], lw=0.8, ls="--")
        a.set_ylabel("kWh")
        a.set_ylim(0, p.cap_kwh)
    np.atleast_1d(ax)[0].legend(frameon=False, fontsize=7, ncol=2)
    np.atleast_1d(ax)[-1].set_xlabel("hour of day")
    fig.tight_layout()
    fig.savefig(RES / "fig_battery.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["solvers", "convergence", "margin", "battery"]
    for w in which:
        {"solvers": fig_solvers, "convergence": fig_convergence, "margin": fig_margin, "battery": fig_battery}[w]()
        print("figure", w)
