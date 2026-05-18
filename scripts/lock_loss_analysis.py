"""Analysis: does visibility failure cause intercept failure?

For the v3 ablation runs already on disk, plot the *within-engagement*
correlation between time spent without lock and the eventual outcome.
This directly answers the question "why do we need visibility-aware
planning if Range-MPPI works fine" -- if the lock-loss -> miss chain
is real, Range-MPPI's lock-loss runs should fail more often than its
locked runs, while Vision-MPPI's runs simply lose lock less.

Inputs: results/v3/ablation_pilot_{hard,fov_exit}/raw_results.json
Output: results/v3/lock_loss_analysis.png  +  lock_loss_table.txt

Usage:
    py scripts/lock_loss_analysis.py
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


def lock_loss_time_seconds(r: dict) -> float:
    """Total seconds with h_V < 0 during the engagement."""
    ts = np.asarray(r["ts_t"], dtype=float)
    hv = np.asarray(r["ts_h_V"], dtype=float)
    if ts.size < 2:
        return 0.0
    dt = np.diff(ts, append=ts[-1])
    return float(np.sum(np.where(hv < 0, dt, 0.0)))


def by_method(results):
    out = {m: [] for m in METHODS}
    for r in results:
        if r["method"] in METHODS:
            out[r["method"]].append(r)
    return out


def make_plot(results: list[dict], attacker_label: str,
              out_path: Path):
    """4-panel: scatter, intercept-rate by lock-loss bucket,
    distribution of lock-loss time, miss distance vs lock-loss."""
    by = by_method(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Panel 1: Scatter intercept-vs-lock-loss-time per method ----
    ax = axes[0, 0]
    for m in METHODS:
        runs = by[m]
        xs, ys = [], []
        for r in runs:
            ll = lock_loss_time_seconds(r)
            xs.append(ll)
            ys.append(1 if r["outcome"] == "intercept" else 0)
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        # Jitter y for visibility, color by method
        jitter = np.random.RandomState(42).uniform(-0.025, 0.025, size=len(ys))
        ax.scatter(xs, ys + jitter, color=METHOD_COLOR[m],
                   label=METHOD_LABEL[m], alpha=0.6, s=40,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Total time without lock during engagement (s)")
    ax.set_ylabel("Outcome (1 = intercept, 0 = breach)")
    ax.set_title("Lock-loss vs outcome (each dot = one engagement)",
                 fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["breach", "intercept"])
    ax.set_ylim(-0.15, 1.15)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=8)

    # ---- Panel 2: Intercept rate as a function of binned lock-loss ----
    ax = axes[0, 1]
    bins = [0.0, 0.05, 0.15, 0.30, 1.0]
    bin_centres = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
    bin_labels = [f"[{bins[i]:.2f}-{bins[i+1]:.2f})"
                  for i in range(len(bins) - 1)]
    n_methods = len(METHODS)
    bar_w = 0.16
    xs_base = np.arange(len(bin_centres))
    for j, m in enumerate(METHODS):
        runs = by[m]
        rates = []
        counts = []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            in_bin = [r for r in runs
                      if lo <= lock_loss_time_seconds(r) < hi]
            counts.append(len(in_bin))
            if not in_bin:
                rates.append(np.nan)
            else:
                rates.append(sum(1 for r in in_bin
                                 if r["outcome"] == "intercept")
                             / len(in_bin))
        rates = np.asarray(rates, dtype=float)
        xs = xs_base + (j - n_methods / 2 + 0.5) * bar_w
        ax.bar(xs, np.nan_to_num(rates, nan=0.0), width=bar_w,
               color=METHOD_COLOR[m], alpha=0.95,
               edgecolor="white", linewidth=0.5,
               label=METHOD_LABEL[m])
        for k, (x_pos, rate, cnt) in enumerate(zip(xs, rates, counts)):
            if cnt == 0 or np.isnan(rate):
                ax.text(x_pos, 0.03, "n=0", ha="center", va="bottom",
                        fontsize=7, color="#888")
            else:
                ax.text(x_pos, rate + 0.02, f"{rate*100:.0f}%",
                        ha="center", va="bottom",
                        fontsize=7, fontweight="bold")
    ax.set_xticks(xs_base)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_xlabel("Lock-loss duration bucket (s)")
    ax.set_ylabel("Intercept rate within bucket")
    ax.set_title("Intercept rate degrades with lock-loss time",
                 fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    # ---- Panel 3: Distribution of lock-loss time per method ----
    ax = axes[1, 0]
    positions = np.arange(len(METHODS))
    data = [
        [lock_loss_time_seconds(r) for r in by[m]]
        for m in METHODS
    ]
    parts = ax.violinplot(data, positions=positions, widths=0.7,
                           showmeans=False, showmedians=True,
                           showextrema=False)
    for body, m in zip(parts['bodies'], METHODS):
        body.set_facecolor(METHOD_COLOR[m])
        body.set_alpha(0.7)
        body.set_edgecolor("black")
    parts['cmedians'].set_color("black")
    parts['cmedians'].set_linewidth(1.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                       fontsize=9, rotation=12)
    ax.set_ylabel("Total lock-loss time (s)")
    ax.set_title("Per-method distribution of lock-loss time",
                 fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    for i, m in enumerate(METHODS):
        d = data[i]
        if d:
            ax.text(i, max(d) * 1.02 + 0.01,
                    f"median={np.median(d):.2f}s\nmean={np.mean(d):.2f}s",
                    ha="center", va="bottom", fontsize=8)

    # ---- Panel 4: miss distance vs lock-loss time ----
    ax = axes[1, 1]
    for m in METHODS:
        runs = by[m]
        xs = [lock_loss_time_seconds(r) for r in runs]
        ys = [float(r["min_rho"]) for r in runs]
        ax.scatter(xs, ys, color=METHOD_COLOR[m],
                   label=METHOD_LABEL[m], alpha=0.6, s=40,
                   edgecolor="white", linewidth=0.5)
    ax.axhline(3.8, color="green", ls="--", lw=1.0,
               label="$r_c$ = 3.8 m (intercept envelope)")
    ax.set_xlabel("Total time without lock (s)")
    ax.set_ylabel(r"min  $\rho$  (defender-attacker distance, m)")
    ax.set_title("Miss distance grows with lock-loss time",
                 fontweight="bold")
    ax.set_ylim(0, max(15, max(
        max(r["min_rho"] for r in by[m]) for m in METHODS if by[m]) + 1))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"Lock-loss -> intercept-failure analysis ({attacker_label}, n="
        f"{int(sum(len(by[m]) for m in METHODS) / len(METHODS))} per method)",
        fontweight="bold", fontsize=14, y=0.995)
    fig.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_table(results, attacker_label, out_path):
    """Text summary: within each method, intercept-rate stratified by
    lock-loss bucket.  Makes the chain quantitative."""
    by = by_method(results)
    bins = [0.0, 0.05, 0.15, 0.30, 1.0]
    lines = [
        f"Lock-loss -> intercept analysis ({attacker_label})",
        "=" * 80,
        "Within each method, intercept rate by lock-loss-time bucket:",
        "",
        f"{'method':<22}" + "".join(
            f"{'[' + str(bins[i]) + '-' + str(bins[i+1]) + ')':>14}"
            for i in range(len(bins) - 1)
        ),
    ]
    for m in METHODS:
        line = f"{METHOD_LABEL[m]:<22}"
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            in_bin = [r for r in by[m]
                      if lo <= lock_loss_time_seconds(r) < hi]
            if not in_bin:
                line += f"{'(n=0)':>14}"
            else:
                ic = sum(1 for r in in_bin if r["outcome"] == "intercept")
                rate = ic / len(in_bin)
                line += f"{rate*100:>8.0f}%(n={len(in_bin)})"
        lines.append(line)
    lines.append("")
    lines.append("Per-method aggregate:")
    lines.append(f"{'method':<22}{'mean ll(s)':>12}{'median ll(s)':>14}"
                 f"{'intercept':>12}{'strict P_succ':>16}")
    for m in METHODS:
        runs = by[m]
        if not runs:
            continue
        ll = [lock_loss_time_seconds(r) for r in runs]
        inter = sum(1 for r in runs if r["outcome"] == "intercept") / len(runs)
        succ = sum(1 for r in runs if r["success"]) / len(runs)
        lines.append(f"{METHOD_LABEL[m]:<22}{np.mean(ll):>12.3f}"
                     f"{np.median(ll):>14.3f}{inter*100:>11.0f}%"
                     f"{succ*100:>15.0f}%")
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
            print(f"  [skip] {raw} not found")
            continue
        with open(raw) as f:
            results = json.load(f)
        print(f"=== {label} ===")
        make_plot(results, label,
                  out_dir / f"lock_loss_analysis_{sub}.png")
        write_table(results, label,
                    out_dir / f"lock_loss_table_{sub}.txt")


if __name__ == "__main__":
    main()
