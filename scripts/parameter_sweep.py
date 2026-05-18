"""Parameter sweep: vary a single SimConfig parameter and run an ablation
of all 5 methods at each value.  Produces a P_succ-vs-parameter plot and
a margin-vs-parameter plot showing how each method degrades.

This is the primary defence for "Range-MPPI works fine, why visibility?"
- at nominal parameter values the methods cluster
- as the parameter gets harsher, Range-MPPI falls off a cliff while
  visibility-aware methods survive

Usage:
    py scripts/parameter_sweep.py \
        --param visibility.theta_F_deg \
        --values 5 10 15 20 25 30 35 \
        --attacker pilot_hard --n 40 --workers 6 \
        --out-dir results/v3/sweeps/fov

    py scripts/parameter_sweep.py \
        --param estimator.latency \
        --values 0.05 0.10 0.15 0.20 0.25 0.30 \
        --attacker pilot_hard --n 40 --workers 6 \
        --out-dir results/v3/sweeps/latency

    py scripts/parameter_sweep.py \
        --param geom.r_c \
        --values 1.0 1.5 2.0 2.5 3.0 3.5 3.8 \
        --attacker pilot_hard --n 40 --workers 6 \
        --out-dir results/v3/sweeps/r_c
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import main as M  # noqa: E402

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


def run_sweep(
    param: str,
    values: list[float],
    attacker: str,
    n: int,
    workers: int,
    t_end: float,
    out_dir: Path,
) -> dict:
    """Run an ablation of all methods at each parameter value.

    Returns dict {value: {method: list[trial_result]}}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    seeds = list(range(n))
    for val in values:
        jobs = [(m, attacker, seed, t_end, {param: float(val)})
                for m in METHODS for seed in seeds]
        print(f"\n[{param}={val}] {len(jobs)} sims, {workers} workers", flush=True)
        t0 = time.time()
        results = M._run_trials_parallel(jobs, workers)
        elapsed = time.time() - t0
        # Group by method
        by_method = {m: [] for m in METHODS}
        for r in results:
            by_method[r["method"]].append(r)
        all_results[float(val)] = by_method
        # Per-value summary
        summary_line = f"  {param}={val:>7.4g}  "
        for m in METHODS:
            runs = by_method[m]
            n_ok = max(1, len(runs))
            psucc = sum(1 for r in runs if r["success"]) / n_ok
            summary_line += f"{METHOD_LABEL[m][:9]:>9s}={psucc*100:>4.0f}%  "
        print(f"{summary_line}  ({elapsed:.1f}s)", flush=True)

    # Save raw
    out_raw = out_dir / "raw_sweep.json"
    serializable = {
        str(v): {m: by_method[m] for m in METHODS}
        for v, by_method in all_results.items()
    }
    with open(out_raw, "w") as f:
        json.dump({
            "param": param,
            "attacker": attacker,
            "n_per_value_per_method": n,
            "values": list(map(float, values)),
            "results": serializable,
        }, f)
    print(f"  raw -> {out_raw.relative_to(ROOT)}", flush=True)
    return all_results


def make_sweep_plot(
    all_results: dict,
    param: str,
    attacker: str,
    out_dir: Path,
):
    """Plot P_succ, intercept rate, h_min, eta_viol vs parameter value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = sorted(all_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    panels = [
        ("Strict $P_{\\mathrm{succ}}$",
         lambda runs: sum(1 for r in runs if r["success"]) / max(1, len(runs)),
         "rate", (0, 1.0)),
        ("Intercept rate $\\mathbb{P}[\\tau_I < \\infty]$",
         lambda runs: sum(1 for r in runs if r["outcome"] == "intercept") / max(1, len(runs)),
         "rate", (0, 1.0)),
        ("Median worst visibility margin $\\langle\\min_t h_V\\rangle$",
         lambda runs: float(np.median([r["min_h_V"] for r in runs])),
         "$h_V$ (median)", None),
        ("Mean $\\eta_V$ violation integral $\\langle\\int [-\\eta_V]_+\\,dt\\rangle$",
         lambda runs: float(np.mean([r["eta_viol_integral"] for r in runs])),
         "violation (s)", None),
    ]

    for ax, (title, fn, ylab, ylim) in zip(axes, panels):
        for m in METHODS:
            ys = []
            for v in values:
                runs = all_results[v][m]
                ys.append(fn(runs))
            ax.plot(values, ys, "o-", color=METHOD_COLOR[m],
                    label=METHOD_LABEL[m], linewidth=2.0,
                    markersize=7, markeredgecolor="white",
                    markeredgewidth=0.8)
        ax.set_xlabel(param)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        # Reference line at h_V = 0
        if "min_t h_V" in title:
            ax.axhline(0, color="red", ls="--", lw=1.0, alpha=0.6,
                       label="lock lost ($h_V<0$)")
        if "Strict" in title:
            ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Parameter sweep: {param}  ({M.ATTACKER_LABEL.get(attacker, attacker)},  "
        f"n={len(next(iter(all_results.values()))['pn'])} per point per method)",
        fontweight="bold", fontsize=13, y=0.995)
    fig.tight_layout()
    out = out_dir / f"sweep_{param.replace('.', '_')}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {out.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", required=True,
                        help="Dotted config path, e.g. visibility.theta_F_deg")
    parser.add_argument("--values", type=float, nargs="+", required=True,
                        help="Parameter values to sweep over")
    parser.add_argument("--attacker", default="pilot_hard",
                        help="Attacker mode (pilot_hard, pilot_easy, "
                             "pilot_fov_exit)")
    parser.add_argument("--n", type=int, default=40,
                        help="Trials per (value, method)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--t-end", type=float, default=4.5)
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Parameter sweep: {args.param} ===")
    print(f"  attacker: {args.attacker}")
    print(f"  values:   {args.values}")
    print(f"  n/point:  {args.n}    workers: {args.workers}")
    print(f"  out_dir:  {out_dir.relative_to(ROOT)}")

    all_results = run_sweep(
        param=args.param,
        values=args.values,
        attacker=args.attacker,
        n=args.n,
        workers=args.workers,
        t_end=args.t_end,
        out_dir=out_dir,
    )

    make_sweep_plot(all_results, args.param, args.attacker, out_dir)

    # Compact table
    table_lines = [f"# Parameter sweep: {args.param}",
                   f"# attacker: {args.attacker}, n={args.n} per point per method",
                   ""]
    table_lines.append(f"{args.param:>18}  " +
                       "  ".join(f"{METHOD_LABEL[m]:>12}" for m in METHODS))
    for v in sorted(all_results.keys()):
        line = f"{v:>18.4g}  "
        for m in METHODS:
            runs = all_results[v][m]
            psucc = sum(1 for r in runs if r["success"]) / max(1, len(runs))
            line += f"{psucc*100:>10.0f}%   "
        table_lines.append(line)
    table_lines.append("")
    table_lines.append("(Columns: strict P_succ per method.)")
    out_tbl = out_dir / "sweep_table.txt"
    out_tbl.write_text("\n".join(table_lines))
    print(f"  table -> {out_tbl.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
