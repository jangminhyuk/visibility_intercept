"""Variance / reliability analysis for the v3 ablations.

Plots the per-seed *distribution* of outcomes for each method, not
just the aggregate P_succ.  Range-MPPI may have similar mean
intercept rate to Vision-MPPI but with bimodal failure -- the
visibility-aware methods are intended to be *reliable*, not just
average-case good.

Outputs:
  results/v3/variance_<attacker>.png        (per-method violin + scatter)
  results/v3/variance_table_<attacker>.txt  (CI summary)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

METHODS = ("pn", "range_only", "visibility_cost",
           "feasibility_aware", "full")
METHOD_LABEL = {
    "pn":                "Naive PN",
    "range_only":        "Range-MPPI",
    "visibility_cost":   "Vision-MPPI",
    "feasibility_aware": "Feas-MPPI",
    "full":              "Proposed",
}
METHOD_COLOR = {
    "pn":                "#7f8c8d",
    "range_only":        "#e74c3c",
    "visibility_cost":   "#f39c12",
    "feasibility_aware": "#3498db",
    "full":              "#27ae60",
}


def by_method(results):
    out = {m: [] for m in METHODS}
    for r in results:
        if r["method"] in METHODS:
            out[r["method"]].append(r)
    return out


def wilson_ci(p, n, z=1.96):
    """Wilson-score 95% CI for a Bernoulli proportion."""
    if n <= 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom)
    return max(0.0, centre - half), min(1.0, centre + half)


def make_variance_plot(results, attacker_label, out_path):
    by = by_method(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ----- Panel 1: distribution of min_rho per method (violin) -----
    ax = axes[0, 0]
    positions = np.arange(len(METHODS))
    data = [[float(r["min_rho"]) for r in by[m]] for m in METHODS]
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                           showmedians=True, showextrema=True)
    for body, m in zip(parts['bodies'], METHODS):
        body.set_facecolor(METHOD_COLOR[m])
        body.set_alpha(0.7)
    parts['cmedians'].set_color("black")
    parts['cmedians'].set_linewidth(1.5)
    for key in ("cmaxes", "cmins", "cbars"):
        parts[key].set_edgecolor("#444")
    ax.axhline(3.8, color="green", ls="--", lw=1.2,
               label="$r_c$ = 3.8 m (intercept envelope)")
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                       fontsize=9, rotation=12)
    ax.set_ylabel(r"min  $\rho$  (m) -- final defender-attacker distance")
    ax.set_title("Distribution of miss distance per method",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # ----- Panel 2: Wilson 95% CI on P_succ -----
    ax = axes[0, 1]
    for j, m in enumerate(METHODS):
        runs = by[m]
        n = len(runs)
        k = sum(1 for r in runs if r["success"])
        p = k / max(1, n)
        lo, hi = wilson_ci(p, n)
        ax.errorbar([j], [p * 100], yerr=[[(p - lo) * 100], [(hi - p) * 100]],
                    fmt="o", color=METHOD_COLOR[m], capsize=8,
                    markersize=10, markeredgecolor="white",
                    markeredgewidth=1.0, linewidth=2.5,
                    label=METHOD_LABEL[m])
        ax.text(j, p * 100 + 4, f"{p*100:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                       fontsize=9, rotation=12)
    ax.set_ylabel("Strict $P_{\\mathrm{succ}}$ (%)")
    ax.set_title("Wilson 95% CI on success rate "
                 "(error bars overlap -> statistically tied)",
                 fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    # ----- Panel 3: Mean lock-loss time per method (key reliability) -----
    ax = axes[1, 0]
    ll_data = []
    for m in METHODS:
        ll_per_run = []
        for r in by[m]:
            ts = np.asarray(r["ts_t"], dtype=float)
            hv = np.asarray(r["ts_h_V"], dtype=float)
            if ts.size < 2:
                ll_per_run.append(0.0)
                continue
            dt = np.diff(ts, append=ts[-1])
            ll_per_run.append(float(np.sum(np.where(hv < 0, dt, 0.0))))
        ll_data.append(ll_per_run)
    parts = ax.violinplot(ll_data, positions=positions, widths=0.7,
                           showmedians=True, showextrema=True)
    for body, m in zip(parts['bodies'], METHODS):
        body.set_facecolor(METHOD_COLOR[m])
        body.set_alpha(0.7)
    parts['cmedians'].set_color("black")
    parts['cmedians'].set_linewidth(1.5)
    for key in ("cmaxes", "cmins", "cbars"):
        parts[key].set_edgecolor("#444")
    for i, d in enumerate(ll_data):
        if d:
            ax.text(i, max(d) * 1.05 + 0.005,
                    f"mean={np.mean(d):.3f}s\nmedian={np.median(d):.3f}s",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                       fontsize=9, rotation=12)
    ax.set_ylabel("Total lock-loss time per engagement (s)")
    ax.set_title("Reliability metric: lock-loss time distribution",
                 fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # ----- Panel 4: variance of per-method min_rho -----
    ax = axes[1, 1]
    stat_data = []
    for m in METHODS:
        rhos = np.asarray([r["min_rho"] for r in by[m]])
        stat_data.append({
            "method": METHOD_LABEL[m],
            "mean": float(np.mean(rhos)),
            "std": float(np.std(rhos)),
            "q05": float(np.percentile(rhos, 5)),
            "q50": float(np.median(rhos)),
            "q95": float(np.percentile(rhos, 95)),
            "iqr": float(np.percentile(rhos, 75) - np.percentile(rhos, 25)),
        })
    means = [s["mean"] for s in stat_data]
    stds = [s["std"] for s in stat_data]
    iqrs = [s["iqr"] for s in stat_data]

    x = np.arange(len(METHODS))
    w = 0.35
    ax.bar(x - w/2, stds, width=w,
           color=[METHOD_COLOR[m] for m in METHODS], alpha=0.95,
           edgecolor="white", linewidth=0.5, label="std($\\min\\rho$)")
    ax.bar(x + w/2, iqrs, width=w,
           color=[METHOD_COLOR[m] for m in METHODS], alpha=0.55,
           edgecolor="white", linewidth=0.5, hatch="//",
           label="IQR($\\min\\rho$)")
    for i, (s, iq) in enumerate(zip(stds, iqrs)):
        ax.text(i - w/2, s + 0.15, f"{s:.1f}", ha="center",
                fontsize=8, fontweight="bold")
        ax.text(i + w/2, iq + 0.15, f"{iq:.1f}", ha="center",
                fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                       fontsize=9, rotation=12)
    ax.set_ylabel(r"spread of min $\rho$ (m)")
    ax.set_title(r"Method reliability: lower spread $\rightarrow$ "
                 r"more predictable outcomes",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Reliability / variance analysis ({attacker_label}, "
        f"n={int(sum(len(by[m]) for m in METHODS) / len(METHODS))} per method)",
        fontweight="bold", fontsize=14, y=0.995)
    fig.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_table(results, attacker_label, out_path):
    by = by_method(results)
    lines = [
        f"Reliability summary ({attacker_label})",
        "=" * 80,
        f"{'method':<22}{'P_succ':>10}{'(95% CI)':>20}"
        f"{'mean lock-loss':>16}{'std minrho':>14}",
    ]
    for m in METHODS:
        runs = by[m]
        n = len(runs)
        if n == 0:
            continue
        k = sum(1 for r in runs if r["success"])
        p = k / n
        lo, hi = wilson_ci(p, n)
        # lock-loss
        lls = []
        for r in runs:
            ts = np.asarray(r["ts_t"], dtype=float)
            hv = np.asarray(r["ts_h_V"], dtype=float)
            if ts.size < 2:
                lls.append(0.0)
                continue
            dt = np.diff(ts, append=ts[-1])
            lls.append(float(np.sum(np.where(hv < 0, dt, 0.0))))
        mean_ll = float(np.mean(lls))
        rho_std = float(np.std([r["min_rho"] for r in runs]))
        lines.append(f"{METHOD_LABEL[m]:<22}{p*100:>9.0f}%"
                     f" [{lo*100:>5.0f}%, {hi*100:>4.0f}%]"
                     f"{mean_ll:>16.3f}s{rho_std:>14.2f}")
    lines.append("")
    lines.append("Note: Wilson 95% confidence interval on P_succ from "
                 f"n={n} Bernoulli trials per method.  Overlapping CIs "
                 "means the methods are statistically tied on P_succ.")
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


def main():
    out_dir = ROOT / "results/v3"
    inputs = [
        ("ablation_pilot_hard", "Hard pilot"),
        ("ablation_pilot_fov_exit", "FoV-exit pilot"),
    ]
    for sub, label in inputs:
        raw = out_dir / sub / "raw_results.json"
        if not raw.exists():
            continue
        with open(raw) as f:
            results = json.load(f)
        print(f"=== {label} ===")
        make_variance_plot(results, label,
                            out_dir / f"variance_{sub}.png")
        write_table(results, label,
                    out_dir / f"variance_table_{sub}.txt")


if __name__ == "__main__":
    main()
