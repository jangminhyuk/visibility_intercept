"""Build the headline paper-summary figure (5-method version).

Combines:
    (row 1) Bar chart: intercept rate + strict success per method for both
            attackers, with the naive PN baseline first to anchor the gap.
    (row 2) Visibility/feasibility margin trajectories (h_V, mu_V, eta_V)
            over time, median + IQR band per method.
    (row 3) Per-method comparison panel + key numbers table.
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

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


def ts_matrix(runs, field, t_grid):
    M = np.full((len(runs), len(t_grid)), np.nan)
    for i, r in enumerate(runs):
        t = np.asarray(r["ts_t"])
        y = np.asarray(r[field])
        if t.size < 2:
            continue
        mask = t_grid <= float(t[-1]) + 1e-9
        M[i, mask] = np.interp(t_grid[mask], t, y)
    return M


def main():
    base = ROOT / "results/v2/final_pilot"
    abl_smart = json.load(open(base / "ablation_pilot_hard/raw_results.json"))
    abl_juke = json.load(open(base / "ablation_pilot_easy/raw_results.json"))

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 5, hspace=0.50, wspace=0.40,
                          height_ratios=[1.0, 1.05, 1.0])

    # Row 1: bar charts for each attacker
    for ax_idx, (data, attacker_name) in enumerate([
            (abl_smart, "Hard pilot (multi-break, banking)"),
            (abl_juke, "Easy pilot (gentle banking)")]):
        by = by_method(data)
        ax = fig.add_subplot(gs[0, ax_idx * 2 + ax_idx:(ax_idx + 1) * 2 + ax_idx + 1])
        xs = np.arange(len(METHODS))
        n_arr = np.array([len(by[m]) for m in METHODS], dtype=float)
        inter = np.array([sum(1 for r in by[m] if r["outcome"] == "intercept")
                          for m in METHODS], dtype=float) / np.maximum(n_arr, 1.0)
        succ = np.array([sum(1 for r in by[m] if r["success"])
                         for m in METHODS], dtype=float) / np.maximum(n_arr, 1.0)
        w = 0.36
        ax.bar(xs - w/2, inter, width=w, color=[METHOD_COLOR[m] for m in METHODS],
               alpha=0.95, edgecolor="white", linewidth=0.5,
               label=r"intercept rate $\mathbb{P}[\tau_I<\infty]$")
        ax.bar(xs + w/2, succ, width=w,
               color=[METHOD_COLOR[m] for m in METHODS],
               alpha=0.6, edgecolor="white", linewidth=0.5, hatch="//",
               label=r"strict $P_{\rm succ}$  $= \mathbb{P}[\tau_I \leq \min(\tau_B,\tau_L)]$")
        for i, (a, b) in enumerate(zip(inter, succ)):
            ax.text(xs[i] - w/2, a + 0.025, f"{a*100:.0f}%",
                    ha="center", fontsize=9, fontweight="bold")
            ax.text(xs[i] + w/2, b + 0.025, f"{b*100:.0f}%",
                    ha="center", fontsize=9, fontweight="bold",
                    color="#444")
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                           fontsize=10, rotation=10)
        ax.set_ylim(0, 1.10)
        ax.set_ylabel("rate")
        ax.set_title(f"{attacker_name}  (n={int(max(n_arr))} per method)",
                     fontweight="bold", fontsize=12)
        if ax_idx == 0:
            ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3, axis="y")
        # Headline gap annotation
        gap_intercept = (inter[METHODS.index("full")]
                          - inter[METHODS.index("pn")])
        ax.text(0.99, 0.98,
                f"Proposed - Naive PN gap:\n"
                f"  intercept rate: {gap_intercept*100:+.0f} pp",
                transform=ax.transAxes,
                fontsize=11, ha="right", va="top", fontweight="bold",
                color="#27ae60",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf7ed",
                          edgecolor="#27ae60"))

    # Row 2: time-series bands for hard attacker
    by = by_method(abl_smart)
    dt = 0.02
    t_max = min(4.0, max(r["end_time"] for r in abl_smart))
    t_grid = np.arange(0, t_max + dt, dt)
    panels = [
        ("ts_h_V",   r"$h_V$  (visibility margin)",  (-1.5, 0.45),
         [(0, "lock lost (h_V=0)", "red")]),
        ("ts_mu_V",  r"$\mu_V$  (feasibility margin)", (-2, 45),
         [(0, "infeasible (mu_V=0)", "red")]),
        ("ts_eta_V", r"$\eta_V$  (command residual)", (-4, 4),
         [(0, "", "red")]),
        ("ts_rho",   r"$\rho$  (defender-attacker range, m)", (0, 35),
         [(3.8, "r_c (intercept envelope)", "green")]),
        ("ts_Omega_inf", r"$\|\Omega\|_\infty$  (body rate, rad/s)", (0, 16),
         [(14.0, "Omega_max", "red")]),
    ]
    for col, (field, ylab, ylim, refs) in enumerate(panels):
        ax = fig.add_subplot(gs[1, col])
        for m in METHODS:
            runs = by[m]
            if not runs:
                continue
            M_mat = ts_matrix(runs, field, t_grid)
            if field == "ts_eta_V":
                M_mat = np.clip(M_mat, -8, 8)
            with np.errstate(all="ignore"):
                med = np.nanmedian(M_mat, axis=0)
                lo = np.nanpercentile(M_mat, 35, axis=0)
                hi = np.nanpercentile(M_mat, 65, axis=0)
            ax.fill_between(t_grid, lo, hi, color=METHOD_COLOR[m],
                            alpha=0.18, edgecolor="none")
            ax.plot(t_grid, med, color=METHOD_COLOR[m], lw=2.0,
                    label=METHOD_LABEL[m])
        for v, lab, c in refs:
            ax.axhline(v, color=c, ls="--", lw=1.0, alpha=0.7,
                       label=lab if lab else None)
        ax.set_xlim(0, t_max)
        ax.set_ylim(*ylim)
        ax.set_xlabel("t (s)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, loc="lower right", ncol=1)
        if col == 2:
            ax.set_title("Hard pilot time-series  (median + 35-65 pctile band)",
                         fontweight="bold", fontsize=11)

    # Row 3: summary table + per-method visibility "scorecard"
    ax_tbl = fig.add_subplot(gs[2, 0:2])
    ax_tbl.axis("off")
    header = ["Method", "intercept", r"$P_{\rm succ}$",
              r"$\langle h_{\min}\rangle$", r"$\langle\eta_{\rm viol}\rangle$"]
    rows = []
    for m in METHODS:
        runs = by[m]
        n = max(1, len(runs))
        succ = sum(1 for r in runs if r["success"]) / n
        inter = sum(1 for r in runs if r["outcome"] == "intercept") / n
        hmin = np.median([r["min_h_V"] for r in runs])
        evi = np.mean([r["eta_viol_integral"] for r in runs])
        rows.append([METHOD_LABEL[m], f"{inter*100:.0f}%",
                     f"{succ*100:.0f}%", f"{hmin:+.2f}", f"{evi:.2f}"])
    tbl = ax_tbl.table(cellText=rows, colLabels=header,
                       cellLoc="center", loc="center",
                       colColours=["#222"] * 5)
    for k in range(5):
        tbl[0, k].set_text_props(color="white", fontweight="bold")
    # Color the Proposed row green
    for k in range(5):
        tbl[len(METHODS), k].set_facecolor("#eaf7ed")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.0, 2.0)
    ax_tbl.set_title(f"Hard pilot — summary table (n={int(max(len(by[m]) for m in METHODS))})",
                     fontweight="bold", fontsize=12)

    # Row 3: bar chart of h_min per method
    ax_h = fig.add_subplot(gs[2, 2])
    h_vals = [np.median([r["min_h_V"] for r in by[m]]) for m in METHODS]
    ax_h.bar(np.arange(len(METHODS)), h_vals,
             color=[METHOD_COLOR[m] for m in METHODS], alpha=0.95,
             edgecolor="white", linewidth=0.5)
    for i, v in enumerate(h_vals):
        ax_h.text(i, v - 0.04, f"{v:+.2f}", ha="center",
                  va="top" if v > 0 else "bottom",
                  fontsize=9, fontweight="bold")
    ax_h.axhline(0, color="red", ls="--", lw=1.0,
                 label="lock lost (h_V<0)")
    ax_h.set_xticks(np.arange(len(METHODS)))
    ax_h.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                          fontsize=9, rotation=10)
    ax_h.set_ylabel(r"median  $\min_t h_V$")
    ax_h.set_title("Worst visibility margin per run", fontweight="bold")
    ax_h.legend(fontsize=8)
    ax_h.grid(alpha=0.3, axis="y")

    # Row 3: bar chart of eta_viol per method
    ax_e = fig.add_subplot(gs[2, 3])
    e_vals = [np.mean([r["eta_viol_integral"] for r in by[m]]) for m in METHODS]
    ax_e.bar(np.arange(len(METHODS)), e_vals,
             color=[METHOD_COLOR[m] for m in METHODS], alpha=0.95,
             edgecolor="white", linewidth=0.5)
    for i, v in enumerate(e_vals):
        ax_e.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9,
                  fontweight="bold")
    ax_e.set_xticks(np.arange(len(METHODS)))
    ax_e.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                          fontsize=9, rotation=10)
    ax_e.set_ylabel(r"$\langle\int [-\eta_V]_+\, dt\rangle$  (s)")
    ax_e.set_title("Command-level visibility-rate violation",
                   fontweight="bold")
    ax_e.grid(alpha=0.3, axis="y")

    # Row 3: bar chart of frac_eta_V_negative
    ax_f = fig.add_subplot(gs[2, 4])
    f_vals = [np.mean([r["fraction_eta_V_negative"] for r in by[m]])
              for m in METHODS]
    ax_f.bar(np.arange(len(METHODS)), f_vals,
             color=[METHOD_COLOR[m] for m in METHODS], alpha=0.95,
             edgecolor="white", linewidth=0.5)
    for i, v in enumerate(f_vals):
        ax_f.text(i, v + 0.01, f"{v*100:.0f}%", ha="center",
                  fontsize=9, fontweight="bold")
    ax_f.set_xticks(np.arange(len(METHODS)))
    ax_f.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                          fontsize=9, rotation=10)
    ax_f.set_ylabel(r"fraction of time  $\eta_V < 0$")
    ax_f.set_title("Fraction of engagement under visibility-rate violation",
                   fontweight="bold")
    ax_f.grid(alpha=0.3, axis="y")

    fig.suptitle("Visibility-Constrained Terminal Guidance — Headline Results",
                 fontweight="bold", fontsize=15, y=0.99)
    out = ROOT / "results/v2/PAPER_SUMMARY_v3.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
