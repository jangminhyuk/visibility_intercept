"""Render a single high-quality snapshot frame for a paper figure.

For a chosen seed/method/attacker, this script:
  1. Runs the simulation once.
  2. Renders a third-person snapshot tightly framed on the defender and
     intruder (so they fill the image rather than getting lost in the
     full engagement).
  3. Renders a defender-camera POV snapshot at the same instant.
  4. Composites the POV image as a labelled PiP inset on the
     third-person image (PIL, no ffmpeg required).

HUD chrome (method name, stats line, legend, POV brackets/strips) is
suppressed by default; the FoV ring and reticle stay because they make
the POV inset visually meaningful.  Background is white by default.

Usage:
    py scripts/render_paper_snapshot.py
    py scripts/render_paper_snapshot.py --seed 4 --method full \\
        --attacker pilot_fov_exit --t-snap 0.0 --out figs/snapshot.png

If `--t-snap` is omitted, the snapshot is taken at the moment of
outcome (the final simulated record); pass an explicit time to capture
an earlier moment when the drones are more separated.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import main as M  # noqa: E402
from gtsim.world import run_headless  # noqa: E402
from gtsim.viz_opengl import render_snapshot_opengl  # noqa: E402


def composite_pip_image(base_path: Path, inset_path: Path, out_path: Path,
                        inset_scale: float = 0.30,
                        margin: int = 24,
                        corner: str = "top_right",
                        border_color=(0, 0, 0),
                        border_thickness: int = 3,
                        label: str | None = "Defender camera POV",
                        label_bg=(0, 0, 0, 190),
                        label_fg=(255, 255, 255),
                        label_font_size: int = 28,
                        ) -> Path:
    """Overlay `inset_path` on `base_path` as a bordered PiP, write PNG."""
    from PIL import Image, ImageDraw, ImageFont
    base = Image.open(base_path).convert("RGBA")
    inset = Image.open(inset_path).convert("RGBA")
    new_w = max(64, int(base.width * inset_scale))
    new_h = max(64, int(inset.height * new_w / inset.width))
    inset = inset.resize((new_w, new_h), Image.LANCZOS)
    bt = max(1, int(border_thickness))
    bordered = Image.new("RGBA",
                         (new_w + 2 * bt, new_h + 2 * bt),
                         (*border_color, 255))
    bordered.paste(inset, (bt, bt))

    if corner == "top_right":
        x = base.width - bordered.width - margin
        y = margin
    elif corner == "top_left":
        x = margin
        y = margin
    elif corner == "bottom_right":
        x = base.width - bordered.width - margin
        y = base.height - bordered.height - margin
    else:
        x = margin
        y = base.height - bordered.height - margin
    base.alpha_composite(bordered, dest=(x, y))

    if label:
        draw = ImageDraw.Draw(base)
        font = None
        for name in ("consola.ttf", "DejaVuSans.ttf", "arial.ttf"):
            try:
                font = ImageFont.truetype(name, label_font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = max(8, label_font_size // 3)
        lx = x + bt + 6
        ly = y + bordered.height - bt - text_h - 2 * pad - 6
        draw.rectangle([lx, ly,
                        lx + text_w + 2 * pad, ly + text_h + 2 * pad],
                       fill=label_bg)
        draw.text((lx + pad, ly + pad), label,
                  fill=label_fg, font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(out_path)
    return out_path


def _auto_t_snap(world, lead_seconds: float, target_rho: float | None,
                 rewind_seconds: float = 0.0) -> float:
    """Pick a snapshot time that gives a clean POV separation.

    Strategy: if `target_rho` is given, pick the first time when rho
    drops to that value (drones close-but-not-touching).  Otherwise,
    take the outcome time minus `lead_seconds`.  Then rewind by
    `rewind_seconds` so the drones aren't already at minimum range.
    """
    recs = world.metrics.records
    if not recs:
        return 0.0
    if target_rho is not None:
        candidate = None
        for r in recs:
            if r.rho <= target_rho:
                candidate = float(r.t)
                break
        if candidate is not None:
            return max(0.0, candidate - float(rewind_seconds))
    s = world.metrics.summary
    import math as _m
    t_end = (s.tau_I if not _m.isnan(s.tau_I)
             else s.tau_B if not _m.isnan(s.tau_B)
             else s.tau_L if not _m.isnan(s.tau_L)
             else recs[-1].t)
    return max(0.0,
               float(t_end) - float(lead_seconds) - float(rewind_seconds))


# Camera angle presets.  Each entry is (yaw, pitch, mode) where mode is
# either "absolute" (yaw/pitch interpreted in world frame) or "relative"
# (yaw is added to the engagement axis -- the direction from the
# protected asset toward the drones midpoint).  Relative presets adapt
# their orientation per seed so the asset stays in a consistent spot in
# the frame regardless of where the engagement happens in the world.
ANGLE_PRESETS: dict[str, tuple[float, float, str]] = {
    # Absolute (world-frame yaw)
    "side":  (42.0, 20.0, "absolute"),
    "rear":  (160.0, 18.0, "absolute"),
    "high":  (60.0, 40.0, "absolute"),
    # Relative to the engagement axis (asset --> drones).
    # Pitch > 0 looks down from above the engagement plane; pitch < 0
    # looks up from below.
    "axis":         ( 90.0,  20.0, "relative"),  # broadside, mid-pitch
    "axis_high":    ( 60.0,  35.0, "relative"),  # elevated three-quarter
    "axis_back":    (135.0,  18.0, "relative"),  # over-the-shoulder
    "axis_below":   ( 90.0, -20.0, "relative"),  # broadside, looking up
    "axis_below_3q":( 60.0, -18.0, "relative"),  # three-quarter, low
    "axis_low":     ( 90.0,   6.0, "relative"),  # near-horizon, dramatic
    "axis_top":     ( 90.0,  65.0, "relative"),  # near top-down
}


def _engagement_axis_yaw(world, cfg, t_snap: float) -> float:
    """World-frame yaw (deg) of the vector from protected asset to the
    drones midpoint at the snapshot time."""
    recs = world.metrics.records
    if len(recs) < 2:
        return 0.0
    dt = recs[1].t - recs[0].t
    idx = max(0, min(int(round(t_snap / max(dt, 1e-6))), len(recs) - 1))
    cur = recs[idx]
    M_d = 0.5 * (np.asarray(cur.p_D, dtype=float)
                 + np.asarray(cur.p_A, dtype=float))
    asset = np.asarray(cfg.geom.p_P, dtype=float)
    v = M_d - asset
    if abs(v[0]) < 1e-6 and abs(v[1]) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(v[1], v[0]))


def _render_one(args, seed: int, angle_name: str, out_dir: Path) -> Path:
    """Render and composite a single snapshot for (seed, angle)."""
    yaw_raw, pitch_deg, mode = ANGLE_PRESETS[angle_name]
    if args.cam_pitch_deg is not None:
        pitch_deg = args.cam_pitch_deg

    cfg = M._config_for(args.method, args.attacker)
    cfg = M._vary_initial_conditions(cfg, seed)
    cfg.sim.t_end = args.t_end
    cfg.scenario.name = ""

    t0 = time.time()
    world = run_headless(cfg, collect_plan_debug_steps=1)
    s = world.metrics.summary
    print(f"\n=== {args.attacker} seed {seed} method {args.method} "
          f"angle={angle_name} ===")
    print(f"  outcome={s.outcome:10s} success={s.success!s:5s} "
          f"min_rho={s.min_rho:5.2f}  ({time.time() - t0:.1f}s)")

    if args.t_snap is not None:
        t_snap = args.t_snap
    else:
        t_snap = _auto_t_snap(world, args.lead_seconds, args.target_rho,
                              args.rewind_seconds)
    print(f"  snapshot time t = {t_snap:.2f} s")

    if args.cam_yaw_deg is not None:
        yaw_deg = args.cam_yaw_deg
    elif mode == "relative":
        yaw_deg = _engagement_axis_yaw(world, cfg, t_snap) + yaw_raw
    else:
        yaw_deg = yaw_raw
    print(f"  camera yaw={yaw_deg:.0f}, pitch={pitch_deg:.0f}")

    stem = (f"snapshot_{args.attacker}_{args.method}"
            f"_seed{seed}_a-{angle_name}")
    third_path = out_dir / f"{stem}_thirdperson.png"
    pov_path = out_dir / f"{stem}_pov.png"
    out_path = out_dir / f"{stem}.png"

    # Optionally inflate the rendered drone size for the third-person
    # pass only -- POV stays at true scale so the FoV ring stays
    # geometrically meaningful.
    vs = float(args.visual_scale)
    if vs != 1.0:
        orig_def = cfg.defender.visual_arm_length
        orig_int = cfg.intruder.visual_arm_length
        cfg.defender.visual_arm_length = orig_def * vs
        cfg.intruder.visual_arm_length = orig_int * vs

    render_snapshot_opengl(
        world.metrics, cfg, third_path,
        plan_debug_history=world.plan_debug_history,
        t_snap=t_snap,
        width=args.width, height=args.height,
        cam_yaw_deg=yaw_deg,
        cam_pitch_deg=pitch_deg,
        cam_distance=args.cam_distance,
        cam_target_mode=args.cam_target,
        view_mode="thirdperson",
        clean_hud=not args.show_hud,
        white_bg=not args.dark_bg,
        line_scale=args.line_scale,
    )

    if vs != 1.0:
        cfg.defender.visual_arm_length = orig_def
        cfg.intruder.visual_arm_length = orig_int

    if args.no_pip:
        print(f"  wrote {third_path.name}")
        return third_path

    render_snapshot_opengl(
        world.metrics, cfg, pov_path,
        plan_debug_history=world.plan_debug_history,
        t_snap=t_snap,
        width=args.pov_width, height=args.pov_height,
        view_mode="pov",
        clean_hud=not args.show_hud,
        white_bg=not args.dark_bg,
    )

    border_color = (0, 0, 0) if not args.dark_bg else (255, 255, 255)
    composite_pip_image(
        third_path, pov_path, out_path,
        inset_scale=args.pip_scale,
        margin=28,
        corner=args.pip_corner,
        border_color=border_color,
        border_thickness=3,
        label=args.label or None,
        label_font_size=args.label_font_size,
    )
    if not args.keep_intermediates:
        for p in (third_path, pov_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    print(f"  wrote {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1, 4, 5, 6, 7, 12, 14, 16],
                        help="One or more seeds to render. One PNG per "
                             "(seed, angle) combination is written.")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep the *_thirdperson.png and *_pov.png "
                             "components alongside the final PiP image.")
    parser.add_argument("--method", type=str, default="full",
                        choices=["pn", "range_only", "visibility_cost",
                                 "feasibility_aware", "full"])
    parser.add_argument("--attacker", type=str, default="pilot_fov_exit")
    parser.add_argument("--t-end", type=float, default=3.5)
    parser.add_argument("--t-snap", type=float, default=None,
                        help="Snapshot time in seconds. Default = "
                             "outcome time minus --lead-seconds (or the "
                             "first time rho <= --target-rho if set).")
    parser.add_argument("--lead-seconds", type=float, default=0.25,
                        help="Lead time before the engagement outcome to "
                             "snapshot at, used only if --target-rho is "
                             "never reached.")
    parser.add_argument("--target-rho", type=float, default=7.0,
                        help="Snapshot the first time rho drops to this "
                             "value -- close enough to read the geometry "
                             "but not yet at the intercept envelope.")
    parser.add_argument("--rewind-seconds", type=float, default=0.3,
                        help="Shift the auto-picked snapshot time this "
                             "much earlier so the drones are not already "
                             "at minimum range.")
    parser.add_argument("--width", type=int, default=3200)
    parser.add_argument("--height", type=int, default=1800)
    parser.add_argument("--pov-width", type=int, default=1440)
    parser.add_argument("--pov-height", type=int, default=1152)
    parser.add_argument("--angles", type=str, nargs="+",
                        default=["axis", "axis_high", "axis_back",
                                 "axis_below", "axis_below_3q",
                                 "axis_low", "axis_top"],
                        choices=list(ANGLE_PRESETS.keys()),
                        help="Camera angle preset(s) to render per seed. "
                             "Relative presets (axis*) adapt to the "
                             "engagement geometry so framing is consistent "
                             "across seeds.  Negative-pitch presets look "
                             "up from below the engagement plane.")
    parser.add_argument("--visual-scale", type=float, default=2.2,
                        help="Multiplier on drone visual size in the "
                             "third-person pass; lets the drones read "
                             "clearly even when the camera has pulled "
                             "back to include the asset. Set 1.0 for "
                             "true scale.")
    parser.add_argument("--line-scale", type=float, default=2.8,
                        help="Multiplier on every line width (trails, "
                             "MPPI candidate fan, FoV ring, LOS line). "
                             "Scales with render resolution: at 3200px "
                             "the default 2.8 keeps trails legible.")
    parser.add_argument("--cam-yaw-deg", type=float, default=None,
                        help="Override the preset yaw (degrees).")
    parser.add_argument("--cam-pitch-deg", type=float, default=None,
                        help="Override the preset pitch (degrees).")
    parser.add_argument("--cam-distance", type=float, default=None,
                        help="Camera orbit distance. Default = auto-fit so "
                             "the two drones fill the frame nicely.")
    parser.add_argument("--cam-target", type=str, default="all",
                        choices=["drones", "engagement", "all"],
                        help="`all` (default) frames defender + intruder + "
                             "protected asset together; `drones` zooms in "
                             "on just the two aircraft.")
    parser.add_argument("--pip-scale", type=float, default=0.30)
    parser.add_argument("--pip-corner", type=str, default="top_right",
                        choices=["top_left", "top_right",
                                 "bottom_left", "bottom_right"])
    parser.add_argument("--label", type=str, default="",
                        help="Text label to overlay on the PiP inset. "
                             "Empty (default) = no label.")
    parser.add_argument("--label-font-size", type=int, default=28,
                        help="Font size (pt) for the PiP inset label.")
    parser.add_argument("--no-pip", action="store_true")
    parser.add_argument("--show-hud", action="store_true")
    parser.add_argument("--dark-bg", action="store_true")
    parser.add_argument("--out-dir", type=str,
                        default=str(ROOT / "results/v3/paper_snapshot"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for seed in args.seeds:
        for angle_name in args.angles:
            outputs.append(_render_one(args, seed, angle_name, out_dir))

    print(f"\nWrote {len(outputs)} snapshot(s) to {out_dir}")


if __name__ == "__main__":
    main()
