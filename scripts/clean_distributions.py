"""Plot success-vs-failure-separated distributions of key metrics.

Failed engagements contaminate aggregate means: when a defender
breaches, min_rho is whatever the distance happened to be at the
breach moment (often 10+ m), not a meaningful "miss distance".  This
script splits each method's runs into successful (intercept before
any visual-loss) and failed (everything else) and plots the
distributions separately.

Output:
  results/v3/clean_distributions_<attacker>.png
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


def lock_loss_seconds(r):
    ts = np.asarray(r["ts_t"], dtype=float)
    hv = np.asarray(r["ts_h_V"], dtype=float)
    if ts.size < 2:
        return 0.0
    dt = np.diff(ts, append=ts[-1])
    return float(np.sum(np.where(hv < 0, dt, 0.0)))


def make_plot(results, attacker_label, out_path):
    by = by_method(results)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # ---- Panel 1: Outcome counts (n breach / n intercept / n vis_loss) ----
    ax = axes[0]
    cats = ["intercept", "breach", "visual_loss", "timeout"]
    cat_colors = ["#27ae60", "#c0392b", "#f39c12", "#7f8c8d"]
    x = np.arange(len(METHODS))
    bottoms = np.zeros(len(METHODS))
    for cat, c in zip(cats, cat_colors):
        counts = [sum(1 for r in by[m] if r["outcome"] == cat) for m in METHODS]
        ax.bar(x, counts, bottom=bottoms, color=c, alpha=0.92,
               edgecolor="white", linewidth=0.5, label=cat)
        for i, (cnt, b) in enumerate(zip(counts, bottoms)):
            if cnt > 0:
                ax.text(i, b + cnt / 2, f"{cnt}",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")
        bottoms += np.asarray(counts)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                       rotation=20, fontsize=10, ha="right")
    ax.set_ylabel("count of engagements")
    ax.set_title("Outcome breakdown (n=60 per method)",
                 fontweight="bold", fontsize=11.5)
    ax.legend(loc="lower right", fontsize=9)

    # ---- Panel 2: Miss distance on SUCCESSFUL intercepts only ----
    ax = axes[1]
    positions = np.arange(len(METHODS))
    data = []
    counts = []
    for m in METHODS:
        # Only runs that intercepted -- min_rho is then a meaningful
        # "closest approach during the kill manoeuvre".
        succ_runs = [r for r in by[m] if r["outcome"] == "intercept"]
        data.append([float(r["min_rho"]) for r in succ_runs])
        counts.append(len(succ_runs))
    parts = ax.violinplot(
        [d if d else [0.0] for d in data],
        positions=positions, widths=0.7, showmedians=True,
        showextrema=True)
    for body, m in zip(parts['bodies'], METHODS):
        body.set_facecolor(METHOD_COLOR[m])
        body.set_alpha(0.75)
    parts['cmedians'].set_color("black")
    parts['cmedians'].set_linewidth(1.5)
    for key in ("cmaxes", "cmins", "cbars"):
        parts[key].set_edgecolor("#444")
    # Compute the actual data range to set ylim sensibly, then place
    # the n=... labels just above the lower edge of the panel.
    flat_vals = [v for d in data for v in d if d]
    if flat_vals:
        y_lo = min(flat_vals) - 0.10
        y_hi = max(max(flat_vals), 3.85) + 0.20
    else:
        y_lo, y_hi = 0.0, 4.0
    ax.set_ylim(y_lo, y_hi)
    for i, (d, c) in enumerate(zip(data, counts)):
        ax.text(i, y_lo + 0.03, f"n={c}", ha="center", va="bottom",
                fontsize=8, color="#444")
        if d:
            ax.text(i, max(d) + 0.04,
                    f"med={np.median(d):.2f}m",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")
    ax.axhline(3.8, color="green", ls="--", lw=1.2,
               label="$r_c$ = 3.8 m (intercept envelope)")
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                       rotation=20, fontsize=10, ha="right")
    ax.set_ylabel(r"min $\rho$ (m) -- closest defender-attacker approach")
    ax.set_title("Miss distance distribution (intercepts only)\n"
                 "Failed engagements excluded to avoid statistical noise",
                 fontweight="bold", fontsize=11.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # ---- Panel 3: Lock-loss time distribution (ALL runs) ----
    ax = axes[2]
    data = [[lock_loss_seconds(r) * 1000 for r in by[m]] for m in METHODS]
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                           showmedians=True, showextrema=True)
    for body, m in zip(parts['bodies'], METHODS):
        body.set_facecolor(METHOD_COLOR[m])
        body.set_alpha(0.75)
    parts['cmedians'].set_color("black")
    parts['cmedians'].set_linewidth(1.5)
    for key in ("cmaxes", "cmins", "cbars"):
        parts[key].set_edgecolor("#444")
    for i, d in enumerate(data):
        if d:
            mean_ms = np.mean(d)
            med_ms = np.median(d)
            ax.text(i, max(d) * 1.03,
                    f"mean={mean_ms:.0f}ms\nmed={med_ms:.0f}ms",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")
    ax.axhline(50, color="red", ls="--", lw=1.0, alpha=0.7,
               label="50 ms threshold (above $\\rightarrow$ 0% intercept)")
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                       rotation=20, fontsize=10, ha="right")
    ax.set_ylabel("Total lock-loss time (ms)")
    ax.set_title("Lock-loss time distribution (all runs)\n"
                 "Above 50 ms threshold $\\rightarrow$ engagement fails",
                 fontweight="bold", fontsize=11.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Clean per-method distributions ({attacker_label}, n=60 per method)",
        fontweight="bold", fontsize=13.5, y=0.99)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86,
                        bottom=0.16, wspace=0.30)
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    out_dir = ROOT / "results/v3"
    for sub, label in [
        ("ablation_pilot_hard", "Hard pilot"),
        ("ablation_pilot_fov_exit", "FoV-exit pilot"),
        ("ablation_pilot_easy", "Easy pilot"),
    ]:
        raw = out_dir / sub / "raw_results.json"
        if not raw.exists():
            continue
        with open(raw) as f:
            results = json.load(f)
        print(f"=== {label} ===")
        make_plot(results, label,
                  out_dir / f"clean_dist_{sub}.png")


if __name__ == "__main__":
    main()
