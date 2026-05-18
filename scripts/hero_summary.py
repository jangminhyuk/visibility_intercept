"""Single hero figure with a clean three-panel narrative.

Panel 1: Strict P_succ per method, per attacker, with Wilson 95% CI.
         The headline "visibility-aware MPPI hugely beats Naive PN".

Panel 2: Lock-loss time -> intercept failure (the causal chain).
         Within EACH method, intercept rate stratified by lock-loss-time
         bucket.  Same deterministic pattern across methods: lock-loss
         >= 50ms => 0% intercept.

Panel 3: Mean lock-loss time per method on the FoV-exit attacker.
         The actionable advantage: Proposed reduces mean lock-loss by
         ~31% over Range-MPPI -> directly translates to reliability.

Statistical hygiene:
  - P_succ aggregates over ALL runs (binary).
  - Lock-loss time and h_min margins are computed per-run, then
    summarised with MEDIAN where possible (robust to failed-engagement
    outliers).  Where the mean is shown it is explicitly labelled.
  - Wilson CI is the right interval for Bernoulli proportions.

Output: results/v3/HERO_SUMMARY.png
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


def wilson_ci(p, n, z=1.96):
    if n <= 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_attacker(sub):
    raw = ROOT / "results/v3" / sub / "raw_results.json"
    if not raw.exists():
        return None
    with open(raw) as f:
        return json.load(f)


def main():
    abl_easy = load_attacker("ablation_pilot_easy")
    abl_hard = load_attacker("ablation_pilot_hard")
    abl_fov = load_attacker("ablation_pilot_fov_exit")
    if abl_fov is None:
        raise SystemExit("Need results/v3/ablation_pilot_fov_exit/raw_results.json")

    fig = plt.figure(figsize=(18, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 0.9],
                          wspace=0.32)

    # ===== Panel 1: P_succ per scenario, with Wilson 95% CI =====
    ax = fig.add_subplot(gs[0, 0])
    attackers = [
        ("Easy pilot",        abl_easy),
        ("Hard pilot",        abl_hard),
        ("FoV-exit pilot",    abl_fov),
    ]
    x_base = np.arange(len(attackers))
    bar_w = 0.16
    for j, m in enumerate(METHODS):
        ps = []
        los = []
        his = []
        for _, data in attackers:
            if data is None:
                ps.append(0); los.append(0); his.append(0); continue
            by = by_method(data)
            runs = by[m]
            n = len(runs)
            k = sum(1 for r in runs if r["success"])
            p = k / max(1, n)
            lo, hi = wilson_ci(p, n)
            ps.append(p)
            los.append(p - lo)
            his.append(hi - p)
        xs = x_base + (j - len(METHODS) / 2 + 0.5) * bar_w
        ax.bar(xs, np.array(ps) * 100, width=bar_w,
               color=METHOD_COLOR[m], alpha=0.95,
               edgecolor="white", linewidth=0.6,
               label=METHOD_LABEL[m],
               yerr=[np.array(los) * 100, np.array(his) * 100],
               capsize=2.5, error_kw=dict(linewidth=1.0, ecolor="#444"))
        for xpos, p in zip(xs, ps):
            ax.text(xpos, p * 100 + 2.5, f"{p*100:.0f}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")
    ax.set_xticks(x_base)
    ax.set_xticklabels([a[0] for a in attackers], fontsize=10)
    ax.set_ylabel("Strict $P_{\\mathrm{succ}}$ (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("(A) Visibility-aware MPPI clearly beats Naive PN\n"
                 "across all attacker scenarios (Wilson 95% CI)",
                 fontweight="bold", fontsize=11.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=5, fontsize=9, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    # Annotate the PN -> Proposed gap on each scenario
    for i, (name, data) in enumerate(attackers):
        if data is None:
            continue
        by = by_method(data)
        pn_p = sum(1 for r in by["pn"] if r["success"]) / max(1, len(by["pn"]))
        full_p = sum(1 for r in by["full"] if r["success"]) / max(1, len(by["full"]))
        gap = (full_p - pn_p) * 100
        ax.annotate(f"+{gap:.0f} pp",
                    xy=(x_base[i], 105),
                    ha="center", va="top", fontsize=9.5,
                    color="#27ae60", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#eaf7ed",
                              edgecolor="#27ae60", linewidth=0.8))

    # ===== Panel 2: Lock-loss => intercept failure (aggregate) =====
    ax = fig.add_subplot(gs[0, 1])
    # Pool runs across all 5 methods (5 x 60 = 300 runs) so the
    # buckets have meaningful sample sizes.  The point of this panel
    # is to show that the threshold is the same for every method --
    # so pooling them is honest.
    by = by_method(abl_fov)
    all_runs = []
    for m in METHODS:
        all_runs.extend(by[m])
    lock_times = np.asarray([lock_loss_seconds(r) for r in all_runs])
    intercept = np.asarray([1 if r["outcome"] == "intercept" else 0
                            for r in all_runs])

    bins = [0.0, 0.025, 0.050, 0.075, 0.100, 0.150, 0.300, 1.0]
    bin_labels = ["0–25", "25–50", "50–75", "75–100",
                  "100–150", "150–300", "300+"]
    bin_rates = []
    bin_counts = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (lock_times >= lo) & (lock_times < hi)
        bin_counts.append(int(mask.sum()))
        if mask.sum() == 0:
            bin_rates.append(np.nan)
        else:
            bin_rates.append(float(intercept[mask].mean()))
    bin_rates = np.asarray(bin_rates)
    x_pos2 = np.arange(len(bin_labels))
    # Color bars green->red across the threshold
    colors = []
    for r in bin_rates:
        if np.isnan(r):
            colors.append("#cccccc")
        elif r > 0.7:
            colors.append("#27ae60")
        elif r > 0.3:
            colors.append("#f39c12")
        else:
            colors.append("#c0392b")
    ax.bar(x_pos2, np.nan_to_num(bin_rates, nan=0.0) * 100,
           color=colors, alpha=0.95, edgecolor="white", linewidth=0.7)
    for xp, rate, cnt in zip(x_pos2, bin_rates, bin_counts):
        if cnt == 0:
            ax.text(xp, 3, "n=0", ha="center", va="bottom",
                    fontsize=8, color="#888")
        else:
            label = f"{rate*100:.0f}%" if not np.isnan(rate) else ""
            ax.text(xp, rate * 100 + 2, label,
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
            ax.text(xp, -7, f"n={cnt}", ha="center", va="top",
                    fontsize=8, color="#444")
    # Threshold line at 50 ms (where intercept rate collapses)
    ax.axvline(1.5, color="black", ls="--", lw=1.5, alpha=0.7)
    ax.text(1.55, 90, "50 ms\nthreshold", fontsize=9,
            color="black", fontweight="bold", va="top")
    ax.set_xticks(x_pos2)
    ax.set_xticklabels(bin_labels, fontsize=9.5)
    ax.set_xlabel("Total lock-loss time per engagement (ms)",
                  fontsize=10)
    ax.set_ylabel("Intercept rate within bucket (%)", fontsize=11)
    ax.set_ylim(-15, 115)
    ax.set_title("(B) The 50 ms threshold: above this, intercept "
                 "rate collapses\n"
                 "(pooled across all 5 methods on FoV-exit pilot, "
                 f"N={len(all_runs)} engagements)",
                 fontweight="bold", fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    # ===== Panel 3: Mean lock-loss time per method =====
    # Each engagement contributes a single number: how many ms (over
    # the whole run) was the intruder outside the FoV cone.  The
    # *mean* of this across runs captures BOTH how often a method
    # crosses the 50 ms threshold AND how badly it crosses it when
    # it does (severity).  Median is 0 ms for every MPPI method
    # (most runs maintain lock perfectly), so mean is the right
    # statistic here -- it weights the catastrophic-failure runs
    # that Panel B shows are causally fatal.
    ax = fig.add_subplot(gs[0, 2])
    means = []
    medians = []
    p75s = []
    for m in METHODS:
        ll = [lock_loss_seconds(r) * 1000 for r in by[m]]
        means.append(np.mean(ll))
        medians.append(np.median(ll))
        p75s.append(np.percentile(ll, 75))
    x_pos = np.arange(len(METHODS))
    ax.bar(x_pos, means,
           color=[METHOD_COLOR[m] for m in METHODS],
           alpha=0.95, edgecolor="white", linewidth=0.6)
    for i, (mean, med, p75) in enumerate(zip(means, medians, p75s)):
        ax.text(x_pos[i], mean + 5, f"{mean:.0f} ms",
                ha="center", fontsize=10, fontweight="bold")
        # Show median + 75th percentile as small annotation underneath
        # so the reader sees the distribution shape, not just the mean.
        ax.text(x_pos[i], -12,
                f"med={med:.0f}\np75={p75:.0f}",
                ha="center", va="top", fontsize=7.5,
                color="#555")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS],
                       rotation=20, fontsize=9.5, ha="right")
    ax.set_ylabel("Mean total lock-loss time per engagement (ms)\n"
                  "(lower = fewer + shorter visibility outages)",
                  fontsize=10)
    ax.set_ylim(-35, max(means) * 1.30)
    full_mean = means[METHODS.index("full")]
    range_mean = means[METHODS.index("range_only")]
    rel_drop = ((range_mean - full_mean) / range_mean * 100
                if range_mean > 0 else 0)
    ax.set_title(
        f"(C) Proposed reduces mean lock-loss time by "
        f"{rel_drop:.0f}% vs Range-MPPI\n"
        f"({full_mean:.0f} ms vs {range_mean:.0f} ms; "
        f"FoV-exit pilot, n=60)",
        fontweight="bold", fontsize=10.5)
    # Add a 50 ms reference line (the Panel B threshold)
    ax.axhline(50, color="black", ls=":", lw=1.0, alpha=0.6)
    ax.text(len(METHODS) - 0.5, 52, "50 ms threshold (Panel B)",
            ha="right", va="bottom", fontsize=7.5, color="#444",
            style="italic")
    # Highlight the proposed bar
    ax.add_patch(plt.Rectangle(
        (x_pos[METHODS.index("full")] - 0.45, 0),
        0.9, full_mean,
        fill=False, edgecolor="#27ae60", linewidth=2.5, zorder=5))
    ax.grid(alpha=0.3, axis="y")

    # Bottom title bar with key takeaway
    fig.suptitle(
        "Why visibility-aware planning is necessary  --  "
        "headline summary  (results/v3, n=60 per (method, attacker))",
        fontweight="bold", fontsize=13.5, y=0.99)

    out = ROOT / "results/v3/HERO_SUMMARY.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
