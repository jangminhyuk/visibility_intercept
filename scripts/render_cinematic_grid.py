"""Render a cinematic 5-method comparison grid video.

For each showcase seed:
  1. Run all 5 methods (same seed, same attacker).
  2. Render each method's engagement using the cinematic OpenGL renderer
     at 640x360 (small but still high-quality).
  3. Stitch the 5 videos into a 2x3 grid (3 across, 2 down) using
     ffmpeg's xstack filter, with the bottom-right cell holding a
     side-by-side numbers panel.

Output: results/v3/cinematic_grids/cinematic_grid_seed<N>.mp4

Usage:
    py scripts/render_cinematic_grid.py --seeds 0 1 5 7 9
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import main as M  # noqa: E402
from gtsim.world import run_headless  # noqa: E402
from gtsim.viz_opengl import render_video_opengl, composite_pip_video  # noqa: E402


METHOD_ORDER = ("pn", "range_only", "visibility_cost",
                "feasibility_aware", "full")
METHOD_TITLE = {
    "pn":                "NAIVE PN (no MPPI)",
    "range_only":        "RANGE-MPPI",
    "visibility_cost":   "VISION-MPPI (h_V)",
    "feasibility_aware": "FEAS-MPPI (+mu_V)",
    "full":              "PROPOSED (full)",
}
# Per-cell tint band drawn under the title using ffmpeg drawbox.
METHOD_COLOR_HEX = {
    "pn":                "7f8c8d",
    "range_only":        "e74c3c",
    "visibility_cost":   "f39c12",
    "feasibility_aware": "3498db",
    "full":              "27ae60",
}


def render_per_method_videos(
    seed: int, out_dir: Path, attacker: str = "smart",
    t_end: float = 3.5, width: int = 640, height: int = 360,
    fps: int = 24, stride: int = 4,
    pip: bool = False, pip_scale: float = 0.34,
    pov_width: int = 720, pov_height: int = 540,
) -> dict:
    """Run each method on the same seed and render a cinematic OpenGL
    video for each.  Returns a dict {method: (mp4_path, summary)}.

    If `pip=True`, also renders a first-person defender-POV pass and
    composites it as a labelled inset on the third-person video.  The
    returned `mp4_path` is the PiP-composited version in that mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in METHOD_ORDER:
        cfg = M._config_for(method, attacker)
        cfg = M._vary_initial_conditions(cfg, seed)
        cfg.sim.t_end = t_end
        # Title shown in HUD; renderer uppercases scenario.name.
        cfg.scenario.name = METHOD_TITLE[method]
        print(f"  [seed {seed}] {method:18s} -> sim ...", flush=True)
        t0 = time.time()
        world = run_headless(cfg, collect_plan_debug_steps=1)
        sim_t = time.time() - t0
        s = world.metrics.summary
        print(f"    outcome={s.outcome:10s} success={s.success!s:5s} "
              f"min_rho={s.min_rho:5.2f} h_min={s.min_h_V:+.2f}  "
              f"({sim_t:.1f}s)", flush=True)
        # Render the cinematic OpenGL video at the per-cell resolution.
        third_person_path = out_dir / f"seed{seed}_{method}.mp4"
        try:
            render_video_opengl(
                world.metrics, cfg, third_person_path,
                plan_debug_history=world.plan_debug_history,
                fps=fps, stride=stride,
                width=width, height=height,
                cam_yaw_deg=38.0, cam_pitch_deg=22.0,
                cam_orbit_speed_deg_s=4.0,
                trail_seconds=1.2,
                extra_seconds_after_outcome=1.0,
                show_mppi_samples=True,
                n_samples_drawn=48,
                show_intruder_tube=False,
                hidden=True,
                view_mode="thirdperson",
                method_tag=METHOD_TITLE[method],
            )
            print(f"    cinematic mp4: {third_person_path.name}", flush=True)
        except Exception as e:
            print(f"    OpenGL render failed: {e!r}", flush=True)
            raise

        cell_path = third_person_path
        if pip:
            pov_path = out_dir / f"seed{seed}_{method}_pov.mp4"
            pip_path = out_dir / f"seed{seed}_{method}_pip.mp4"
            try:
                render_video_opengl(
                    world.metrics, cfg, pov_path,
                    plan_debug_history=world.plan_debug_history,
                    fps=fps, stride=stride,
                    width=pov_width, height=pov_height,
                    trail_seconds=1.2,
                    extra_seconds_after_outcome=1.0,
                    show_mppi_samples=True,
                    n_samples_drawn=48,
                    show_intruder_tube=False,
                    hidden=True,
                    view_mode="pov",
                    method_tag=METHOD_TITLE[method],
                )
                composite_pip_video(
                    third_person_path, pov_path, pip_path,
                    inset_scale=pip_scale, margin_px=16,
                    corner="top_right",
                    border_color="white", border_thickness=3,
                    label="Defender camera POV",
                )
                cell_path = pip_path
                print(f"    pip mp4:       {pip_path.name}", flush=True)
            except Exception as e:
                print(f"    POV/PiP render failed: {e!r}", flush=True)
                # Fall back to plain third-person cell.
        results[method] = (cell_path, asdict(s))
    return results


def _probe_duration(path: Path) -> float:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def composite_grid(per_method: dict, out_path: Path,
                   cell_w: int = 640, cell_h: int = 360,
                   seed: int = 0, attacker_label: str = "Smart attacker"):
    """Composite five per-method MP4s into a 2x3 grid using ffmpeg.

    Layout (3 wide, 2 tall):
        [pn]              [range]    [vision]
        [feasibility]     [full]     [summary panel]

    The summary panel is a tinted png with the headline numbers.  The
    output is truncated to the longest cell video duration plus a
    small post-roll, so the final video matches the engagement length
    rather than the looped panel duration.
    """
    ordered = [per_method[m][0] for m in METHOD_ORDER]
    # Build summary panel image
    panel_path = out_path.parent / f".panel_seed{seed}.png"
    _make_summary_panel(per_method, panel_path, cell_w, cell_h,
                        seed, attacker_label)

    # Determine the longest cell duration so the grid output ends with
    # the engagement and doesn't tail off into a 30 s loop of the panel.
    durations = [_probe_duration(p) for p in ordered]
    max_dur = max(durations) if durations else 4.0
    grid_dur = max_dur + 0.4   # tiny post-roll

    # ffmpeg command: 6 inputs, xstack to 3x2, output truncated to
    # max cell duration.
    inputs = []
    for v in ordered:
        inputs.extend(["-i", str(v)])
    # Loop the panel image for the engagement duration.
    inputs.extend(["-loop", "1", "-t", f"{grid_dur:.2f}",
                   "-i", str(panel_path)])

    # Build filter_complex.  The OpenGL HUD already shows the method
    # name (we set cfg.scenario.name), so we just need to add a thin
    # colored border per cell to make the method distinction obvious
    # at a glance.  Then xstack into a 3x2 grid.
    parts = []
    labels = []
    border = 4
    for idx, m in enumerate(METHOD_ORDER):
        color = METHOD_COLOR_HEX[m]
        # Scale to inner cell size minus border, then pad with the
        # method's color (border).
        inner_w = cell_w - 2 * border
        inner_h = cell_h - 2 * border
        parts.append(
            f"[{idx}:v]scale={inner_w}:{inner_h},"
            f"pad={cell_w}:{cell_h}:{border}:{border}:color=0x{color}"
            f"[v{idx}]"
        )
        labels.append(f"[v{idx}]")
    # Summary panel
    parts.append(f"[5:v]scale={cell_w}:{cell_h}[v5]")
    labels.append("[v5]")
    grid_filter = (
        "".join(labels) +
        f"xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0[grid]"
    )
    filter_complex = ";".join(parts + [grid_filter])

    cmd = ["ffmpeg", "-y", *inputs,
           "-filter_complex", filter_complex,
           "-map", "[grid]",
           "-t", f"{grid_dur:.2f}",           # cap output length
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "medium", "-crf", "20",
           "-loglevel", "warning",
           str(out_path)]
    print(f"  ffmpeg compositing -> {out_path.name} ...", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("FFmpeg stderr:", res.stderr, flush=True)
        raise RuntimeError(f"ffmpeg failed: {res.returncode}")
    print(f"    wrote {out_path.name} "
          f"({out_path.stat().st_size/1024:.0f} KB, {time.time()-t0:.1f}s)",
          flush=True)
    try:
        panel_path.unlink()
    except Exception:
        pass


def _make_summary_panel(per_method: dict, out_path: Path,
                        width: int, height: int, seed: int,
                        attacker_label: str):
    """Render a static PNG summary panel for the grid's 6th cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100,
                      facecolor="#101218")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#101218")
    ax.axis("off")
    ax.text(0.5, 0.96,
            f"Per-method summary",
            ha="center", va="top", color="#e6e8ee",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.87,
            f"{attacker_label}, seed {seed}",
            ha="center", va="top", color="#b0b4ba",
            fontsize=8.5, transform=ax.transAxes)
    # Table-like text rows
    rows = []
    for m in METHOD_ORDER:
        s = per_method[m][1]
        outcome = s["outcome"]
        rho = s["min_rho"]
        hmin = s["min_h_V"]
        rows.append((m, outcome, rho, hmin))
    y0 = 0.74
    dy = 0.13
    headers = "outcome     min ρ    min h_V"
    ax.text(0.04, y0, "", color="#7f8c8d", fontsize=8.5,
            transform=ax.transAxes)
    ax.text(0.52, y0, headers, color="#7f8c8d", fontsize=8.5,
            family="monospace", transform=ax.transAxes)
    for i, (m, outcome, rho, hmin) in enumerate(rows):
        y = y0 - (i + 1) * dy
        color = "#" + METHOD_COLOR_HEX[m]
        ax.text(0.04, y, METHOD_TITLE[m].split(" (")[0],
                color=color, fontsize=9, fontweight="bold",
                transform=ax.transAxes)
        outcome_color = ("#27ae60" if outcome == "intercept"
                         else "#c0392b" if outcome == "breach"
                         else "#bdc3c7")
        ax.text(0.52, y, f"{outcome:10s}", color=outcome_color,
                fontsize=9, family="monospace", transform=ax.transAxes)
        ax.text(0.70, y, f"{rho:5.2f}", color="#d0d3da",
                fontsize=9, family="monospace", transform=ax.transAxes)
        ax.text(0.82, y, f"{hmin:+.2f}", color="#d0d3da",
                fontsize=9, family="monospace", transform=ax.transAxes)

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=100, facecolor="#101218")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 5, 9])
    parser.add_argument("--attacker", type=str, default="smart")
    parser.add_argument("--t-end", type=float, default=3.5)
    parser.add_argument("--cell-width", type=int, default=640)
    parser.add_argument("--cell-height", type=int, default=360)
    parser.add_argument("--out-dir", type=str,
                        default=str(ROOT / "results/v3/cinematic_grids"))
    parser.add_argument("--pip", action="store_true",
                        help="Render defender first-person POV and composite "
                             "as a PiP inset on each cell.")
    parser.add_argument("--pip-scale", type=float, default=0.34,
                        help="Inset size as fraction of cell width (0.20-0.40).")
    parser.add_argument("--pov-width", type=int, default=720,
                        help="POV-pass render width before scaling.")
    parser.add_argument("--pov-height", type=int, default=540,
                        help="POV-pass render height before scaling.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_dir = out_dir / "cells"
    cell_dir.mkdir(exist_ok=True)
    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        per_method = render_per_method_videos(
            seed, cell_dir,
            attacker=args.attacker, t_end=args.t_end,
            width=args.cell_width, height=args.cell_height,
            pip=args.pip, pip_scale=args.pip_scale,
            pov_width=args.pov_width, pov_height=args.pov_height,
        )
        suffix = "_pip" if args.pip else ""
        out_path = out_dir / f"cinematic_grid{suffix}_seed{seed}.mp4"
        composite_grid(per_method, out_path,
                       cell_w=args.cell_width,
                       cell_h=args.cell_height,
                       seed=seed,
                       attacker_label=M.ATTACKER_LABEL.get(args.attacker,
                                                          args.attacker))
    print("\nDone.")


if __name__ == "__main__":
    main()
