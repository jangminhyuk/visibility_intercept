"""CLI entrypoint for the visibility-feasibility terminal-guidance sim (v2).

Modes:
    demo       single-run demo for one (method, attacker, seed) combination.
    videos     render N seeds per (method, attacker) into per-scenario folders.
    ablation   4-method ablation table: P_succ, h_min, mu_min, eta_viol, etc.
               + median+IQR time-series plots of the diagnostics.
    map        visibility-feasibility map: P_succ heatmap over
               (intruder a_max) x (defender Omega_max) for each method.

Examples:
    py main.py --mode demo --method full --attacker smart --seed 7 --video
    py main.py --mode videos --method full --attacker smart --n 4
    py main.py --mode ablation --n 30 --workers 6
    py main.py --mode map --n 20 --workers 6
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from gtsim.config import SimConfig, PLANNER_METHODS  # noqa: E402
from gtsim.world import run_headless  # noqa: E402
from gtsim.viz import (  # noqa: E402
    plot_mppi_snapshots,
    plot_run_timeseries,
    plot_trajectory_3d,
)
from gtsim.video3d import render_video_3d  # noqa: E402

try:
    from gtsim.viz_opengl import render_video_opengl  # noqa: E402
    from gtsim.viz_opengl import composite_pip_video  # noqa: E402
    _HAVE_OPENGL = True
except Exception as _e:
    _HAVE_OPENGL = False
    _OPENGL_IMPORT_ERR = _e


METHOD_LABEL = {
    "pn":                "Naive PN\n(proportional navigation,\nno anticipation)",
    "range_only":        "Range-MPPI\n(no visibility cost)",
    "visibility_cost":   "Vision-MPPI\n(field-of-view penalty)",
    "feasibility_aware": "Feasibility-MPPI\n(+ body-rate margin)",
    "full":              "Proposed (Scenario-risk\nMPPI + rollout gate)",
}
METHOD_LABEL_SHORT = {
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
ATTACKER_LABEL = {
    "smart":            "Hard pilot (banking break)",
    "pilot_hard":       "Hard pilot (banking break)",
    "pilot_easy":       "Easy pilot (gentle banking)",
    "pilot_fov_exit":   "FoV-exit pilot (camera-aware break)",
    "juke_sinusoidal":  "Sinusoidal juke attacker",
    "juke_bangbang":    "Bang-bang juke attacker",
    "straight":         "Straight-line attacker",
}


def _render_video(metrics, cfg, out_path, fps, stride, plan_debug_history,
                  prefer_opengl: bool = True,
                  pip: bool = False,
                  pip_width: int = 640,
                  pip_height: int = 480,
                  pip_scale: float = 0.30,
                  method_tag: str | None = None):
    """Render an MP4 of a recorded run.

    If `pip=True` and the OpenGL renderer is available, also renders a
    first-person defender-POV pass and composites it as a PiP inset.
    The cinematic third-person video is always written to `out_path`;
    the PiP variant lands at `out_path.with_name(stem + '_pip.mp4')`
    and the raw POV is kept alongside as `*_pov.mp4` for reference.
    """
    if prefer_opengl and _HAVE_OPENGL:
        try:
            third_person_path = render_video_opengl(
                metrics, cfg, out_path,
                plan_debug_history=plan_debug_history,
                fps=fps, stride=stride,
                view_mode="thirdperson",
                method_tag=method_tag,
            )
            if pip:
                stem = Path(out_path).stem
                parent = Path(out_path).parent
                pov_path = parent / f"{stem}_pov.mp4"
                pip_path = parent / f"{stem}_pip.mp4"
                render_video_opengl(
                    metrics, cfg, pov_path,
                    plan_debug_history=plan_debug_history,
                    fps=fps, stride=stride,
                    width=pip_width, height=pip_height,
                    view_mode="pov",
                    method_tag=method_tag,
                )
                label = "Defender camera POV (FoV \\u00B130\\u00B0)"
                # Keep the label ASCII-safe — ffmpeg's drawtext does not
                # play well with curly degree signs across builds.
                label = "Defender camera POV"
                try:
                    composite_pip_video(
                        third_person_path, pov_path, pip_path,
                        inset_scale=pip_scale, margin_px=28,
                        corner="top_right",
                        border_color="white", border_thickness=3,
                        label=label,
                    )
                except Exception as e:
                    print(f"  [warn] PiP composite failed: {e!r}")
            return third_person_path
        except Exception as e:
            print(f"  [warn] OpenGL render failed: {e!r}; falling back to mpl.")
    return render_video_3d(
        metrics, cfg, out_path,
        fps=fps, stride=stride,
        plan_debug_history=plan_debug_history,
    )


# --------------------------------------------------------------------------- #
# Scenario configuration / initial conditions
# --------------------------------------------------------------------------- #


def _config_for(method: str, attacker: str,
                base: SimConfig | None = None) -> SimConfig:
    if method not in PLANNER_METHODS:
        raise SystemExit(f"Unknown method {method!r}. Known: {PLANNER_METHODS}")
    cfg = deepcopy(base) if base is not None else SimConfig()
    cfg.scenario.attacker_mode = attacker
    cfg.scenario.planner_method = method
    cfg.scenario.name = f"{attacker}_{method}"
    # PN is the naive baseline: no MPPI, just reactive proportional navigation.
    cfg.scenario.use_mppi = (method != "pn")
    return cfg


def _vary_initial_conditions(cfg: SimConfig, seed: int) -> SimConfig:
    """Vary intruder and defender initial state widely by seed.

    Head-on geometry (paper baseline): the defender sits on the threat
    axis ~14-20 m from the asset at similar altitude to the incoming
    intruder.  Body-z (thrust + camera) is naturally roughly aligned
    with the LOS to the target, so the engagement is physically
    interceptable; the visibility cost manifests as LOS angular-rate
    tracking during the attacker's lateral break.
    """
    rng = np.random.default_rng(seed * 7919 + 1009)
    p_asset = np.asarray(cfg.geom.p_P, dtype=float)

    # Intruder spawns farther out so the engagement runs long enough
    # for multi-pass defender attempts.
    int_az = float(rng.uniform(0.0, 2.0 * np.pi))
    int_dist = float(rng.uniform(44.0, 54.0))
    z_int = float(rng.uniform(5.0, 9.0))
    cfg.intruder.p0 = [p_asset[0] + int_dist * float(np.cos(int_az)),
                       p_asset[1] + int_dist * float(np.sin(int_az)),
                       z_int]
    p_int = np.asarray(cfg.intruder.p0, dtype=float)
    to_asset = p_asset - p_int
    to_asset = to_asset / max(float(np.linalg.norm(to_asset)), 1e-9)
    speed0 = float(rng.uniform(12.0, 14.5))
    cfg.intruder.v0 = (to_asset * speed0).tolist()

    geom_mode = getattr(cfg.scenario, "geometry_mode", "head_on")
    if geom_mode == "crossing":
        # Defender spawns ~90 deg off the attacker's bearing, so the
        # defender flies "across" the attack line.  LOS rotates rapidly
        # at terminal range -- the regime where mu_V becomes binding.
        side = 1.0 if rng.uniform() > 0.5 else -1.0
        def_az = int_az + side * (np.pi / 2 + float(rng.uniform(-np.pi / 12,
                                                                  np.pi / 12)))
        def_dist = float(rng.uniform(12.0, 18.0))
    else:
        # head_on: defender on threat axis (default)
        def_az = int_az + float(rng.uniform(-np.pi / 12, np.pi / 12))
        def_dist = float(rng.uniform(15.0, 22.0))
    z_def = z_int + float(rng.uniform(2.0, 5.0))
    cfg.defender.p0 = [p_asset[0] + def_dist * float(np.cos(def_az)),
                       p_asset[1] + def_dist * float(np.sin(def_az)),
                       z_def]
    # Initial velocity: small loiter velocity (not already in sprint).
    p_def = np.asarray(cfg.defender.p0, dtype=float)
    to_int_from_def = p_int - p_def
    to_int_from_def = to_int_from_def / max(
        float(np.linalg.norm(to_int_from_def)), 1e-9)
    def_speed = float(rng.uniform(2.0, 5.0))
    cfg.defender.v0 = (to_int_from_def * def_speed).tolist()

    cfg.sim.random_seed = int(seed)
    return cfg


# --------------------------------------------------------------------------- #
# Single-trial worker (pickle-safe top-level)
# --------------------------------------------------------------------------- #


def _run_one_trial(args_tuple):
    method, attacker, seed, t_end, overrides = args_tuple
    cfg = _config_for(method, attacker)
    cfg = _vary_initial_conditions(cfg, seed)
    if t_end is not None:
        cfg.sim.t_end = float(t_end)
    # Allow per-trial overrides (used by `map` mode to vary a_max / Omega_max).
    if overrides:
        for path, val in overrides.items():
            parts = path.split(".")
            obj = cfg
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)
    # Local import so pickle works under joblib loky workers.
    from gtsim.world import run_headless as _rh
    world = _rh(cfg, collect_plan_debug_steps=0)
    recs = world.metrics.records
    s = world.metrics.summary
    return {
        "method": method,
        "attacker": attacker,
        "seed": int(seed),
        "outcome": s.outcome,
        "success": bool(s.success),
        "end_time": float(s.end_time),
        "tau_I": float(s.tau_I) if not math.isnan(s.tau_I) else None,
        "tau_B": float(s.tau_B) if not math.isnan(s.tau_B) else None,
        "tau_L": float(s.tau_L) if not math.isnan(s.tau_L) else None,
        "min_rho": float(s.min_rho),
        "min_attacker_hvu_dist": float(s.min_attacker_hvu_dist),
        "min_h_V": float(s.min_h_V),
        "min_mu_V": float(s.min_mu_V),
        "min_eta_V": float(s.min_eta_V),
        "mean_h_V": float(s.mean_h_V),
        "mean_mu_V": float(s.mean_mu_V),
        "mean_eta_V": float(s.mean_eta_V),
        "eta_viol_integral": float(s.eta_viol_integral),
        "fraction_h_V_negative": float(s.fraction_h_V_negative),
        "fraction_mu_V_negative": float(s.fraction_mu_V_negative),
        "fraction_eta_V_negative": float(s.fraction_eta_V_negative),
        "fraction_planner_stressed": float(s.fraction_planner_stressed),
        # Compact time series (interpolated later on a common grid)
        "ts_t":     [r.t for r in recs],
        "ts_rho":   [r.rho for r in recs],
        "ts_h_V":   [r.h_V for r in recs],
        "ts_mu_V":  [r.mu_V for r in recs],
        "ts_eta_V": [r.eta_V for r in recs],
        "ts_Omega_inf": [float(max(abs(x) for x in r.Omega_applied))
                         for r in recs],
        "ts_stressed": [1.0 if r.planner_stressed else 0.0 for r in recs],
        # Initial geometry for scatter plots
        "intruder_p0": list(map(float, cfg.intruder.p0)),
        "intruder_v0": list(map(float, cfg.intruder.v0)),
        "defender_p0": list(map(float, cfg.defender.p0)),
        "defender_v0": list(map(float, cfg.defender.v0)),
        "Omega_max": float(cfg.defender.Omega_max),
        "a_max": float(cfg.intruder.a_max),
    }


def _run_trials_parallel(jobs, workers):
    try:
        from joblib import Parallel, delayed
        return Parallel(n_jobs=workers, backend="loky", verbose=5)(
            delayed(_run_one_trial)(j) for j in jobs)
    except ImportError:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            return pool.map(_run_one_trial, jobs)


# --------------------------------------------------------------------------- #
# Demo mode
# --------------------------------------------------------------------------- #


def cmd_compare(args: argparse.Namespace) -> int:
    """Run all four planner methods on the same seed and overlay their
    trajectories + diagnostics into a single comparison figure.

    Produces:
        compare_<seed>_traj.png    — 3D + XY view side-by-side
        compare_<seed>_diag.png    — h_V, mu_V, eta_V, rho over time
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    global plt; plt = _plt

    attacker = args.attacker
    seed = int(args.seed) if args.seed is not None else 0
    out_dir = Path(args.out_dir) / f"compare_{attacker}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    worlds = {}
    cfgs = {}
    for method in PLANNER_METHODS:
        cfg = _config_for(method, attacker)
        cfg = _vary_initial_conditions(cfg, seed)
        if args.t_end is not None:
            cfg.sim.t_end = float(args.t_end)
        t0 = time.time()
        world = run_headless(cfg)
        wall = time.time() - t0
        s = world.metrics.summary
        print(f"  {method:18s} outcome={s.outcome:10s} succ={s.success!s:5s} "
              f"min_rho={s.min_rho:5.2f} min_h={s.min_h_V:+5.3f} "
              f"min_mu={s.min_mu_V:+5.2f} eta_viol={s.eta_viol_integral:5.2f} "
              f"({wall:.1f}s)")
        worlds[method] = world
        cfgs[method] = cfg

    # Trajectory plot (3D + XY)
    fig = plt.figure(figsize=(13, 6))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    axxy = fig.add_subplot(1, 2, 2)
    # Asset
    p_P = np.asarray(cfgs[next(iter(cfgs))].geom.p_P)
    r_P = float(cfgs[next(iter(cfgs))].geom.r_P)
    for ax_ in (ax3, axxy):
        ax_.plot([p_P[0]], [p_P[1]], "k*", markersize=14, label="Asset"
                 if ax_ is axxy else None)
    # Asset sphere on 3D
    u = np.linspace(0, 2 * np.pi, 18); v = np.linspace(0, np.pi, 9)
    x = p_P[0] + r_P * np.outer(np.cos(u), np.sin(v))
    y = p_P[1] + r_P * np.outer(np.sin(u), np.sin(v))
    z = p_P[2] + r_P * np.outer(np.ones_like(u), np.cos(v))
    ax3.plot_surface(x, y, z, color="black", alpha=0.18, linewidth=0)
    # Intruder (same for all methods)
    rec_int = np.array([r.p_A for r in worlds["range_only"].metrics.records])
    ax3.plot(rec_int[:, 0], rec_int[:, 1], rec_int[:, 2],
             "k-", lw=2.5, label="Intruder")
    axxy.plot(rec_int[:, 0], rec_int[:, 1], "k-", lw=2.0,
              label="Intruder")
    axxy.plot(rec_int[0, 0], rec_int[0, 1], "k^", markersize=10)
    for method in PLANNER_METHODS:
        recs = worlds[method].metrics.records
        pD = np.array([r.p_D for r in recs])
        color = METHOD_COLOR[method]
        ax3.plot(pD[:, 0], pD[:, 1], pD[:, 2],
                 color=color, lw=2.0, label=METHOD_LABEL[method].split(" (")[0])
        # Mark contact / end
        ax3.scatter(pD[-1, 0], pD[-1, 1], pD[-1, 2],
                    color=color, s=80, marker="X", edgecolor="black",
                    linewidth=0.5)
        axxy.plot(pD[:, 0], pD[:, 1], color=color, lw=2.0,
                  label=METHOD_LABEL[method].split(" (")[0])
        axxy.plot(pD[0, 0], pD[0, 1], color=color, marker="^", markersize=8,
                  markeredgecolor="black")
        axxy.plot(pD[-1, 0], pD[-1, 1], color=color, marker="X", markersize=10,
                  markeredgecolor="black")
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.set_zlabel("Z (m)")
    ax3.set_title(f"3D trajectories — {ATTACKER_LABEL.get(attacker)} seed {seed}")
    ax3.legend(fontsize=8, loc="upper right")
    axxy.set_xlabel("X (m)"); axxy.set_ylabel("Y (m)")
    axxy.set_title("Top-down (XY) view")
    axxy.set_aspect("equal", adjustable="datalim")
    axxy.grid(alpha=0.3)
    axxy.legend(fontsize=8, loc="best")
    asset_circle = _plt.Circle((p_P[0], p_P[1]), r_P,
                                color="black", alpha=0.15)
    axxy.add_patch(asset_circle)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_traj.png", dpi=140)
    plt.close(fig)

    # Diagnostics plot
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    panels = [
        ("rho", lambda r: r.rho, "ρ (m)",
         [(cfgs["full"].geom.r_c, "r_c", "green")]),
        ("h_V", lambda r: r.h_V, r"$h_V$",
         [(0.0, "h_V=0 (lock lost)", "red"),
          (cfgs["full"].visibility.h_safe, "h_safe", "orange")]),
        ("mu_V", lambda r: r.mu_V, r"$\mu_V$",
         [(0.0, "μ_V=0 (infeasible)", "red")]),
        ("eta_V", lambda r: r.eta_V, r"$\eta_V$",
         [(0.0, "η_V=0", "red")]),
    ]
    for ax, (name, getter, ylabel, refs) in zip(axes, panels):
        for method in PLANNER_METHODS:
            recs = worlds[method].metrics.records
            t = np.array([r.t for r in recs])
            y = np.array([getter(r) for r in recs])
            ax.plot(t, y, color=METHOD_COLOR[method], lw=1.8,
                    label=METHOD_LABEL[method].split(" (")[0])
        for val, lab, col in refs:
            ax.axhline(val, color=col, linestyle="--", linewidth=1.0,
                       alpha=0.7, label=lab)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if name == "rho":
            ax.legend(fontsize=8, loc="upper right", ncol=2)
        if name == "eta_V":
            ax.set_ylim(-6, 6)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(f"Method comparison — {ATTACKER_LABEL.get(attacker)} "
                 f"seed {seed}", fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_dir / "compare_diag.png", dpi=150)
    plt.close(fig)

    # Save raw summaries as JSON
    summaries = {}
    for m, w in worlds.items():
        summaries[m] = asdict(w.metrics.summary)
    with open(out_dir / "compare_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"  outputs: {out_dir}/")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = _config_for(args.method, args.attacker)
    seed = int(args.seed) if args.seed is not None else 0
    cfg = _vary_initial_conditions(cfg, seed)
    if args.t_end is not None:
        cfg.sim.t_end = float(args.t_end)
    out_dir = Path(args.out_dir) / f"demo_{cfg.scenario.name}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    world = run_headless(cfg, collect_plan_debug_steps=4)
    t_elapsed = time.time() - t0
    world.metrics.print_summary()
    print(f"  wall time:            {t_elapsed:.2f} s")
    cfg.save_json(out_dir / "config.json")
    world.metrics.save_json(out_dir / "metrics.json")
    summary = asdict(world.metrics.summary)
    summary["wall_time_s"] = t_elapsed
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_trajectory_3d(world.metrics, cfg, out_dir / "traj3d.png")
    plot_run_timeseries(world.metrics, cfg, out_dir / "run_timeseries.png")
    plot_mppi_snapshots(world.plan_debug_history, world.metrics, cfg,
                        out_dir / "mppi_snapshots.png", n_snapshots=4)
    if args.video:
        t0v = time.time()
        path = _render_video(
            world.metrics, cfg, out_dir / "run.mp4",
            fps=args.video_fps, stride=args.video_stride,
            plan_debug_history=world.plan_debug_history,
            prefer_opengl=(not args.video_mpl),
            pip=args.pip,
            pip_width=args.pip_width,
            pip_height=args.pip_height,
            pip_scale=args.pip_scale,
            method_tag=METHOD_LABEL.get(cfg.scenario.planner_method,
                                        cfg.scenario.planner_method).split("\n")[0],
        )
        print(f"  video:                {path.name}  ({time.time()-t0v:.1f}s)")
    print(f"  outputs:              {out_dir}/")
    return 0


# --------------------------------------------------------------------------- #
# Videos mode (render N seeds per method/attacker)
# --------------------------------------------------------------------------- #


def cmd_videos(args: argparse.Namespace) -> int:
    methods = args.methods.split(",") if args.methods else list(PLANNER_METHODS)
    if args.attackers:
        attackers = args.attackers.split(",")
    elif args.attacker:
        attackers = [args.attacker]
    else:
        attackers = ["smart"]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.n))
    rows = []
    rendered = []
    for atk in attackers:
        for method in methods:
            scen_name = f"{atk}_{method}"
            scen_dir = out_root / scen_name
            scen_dir.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                cfg = _config_for(method, atk)
                cfg = _vary_initial_conditions(cfg, seed)
                if args.t_end is not None:
                    cfg.sim.t_end = float(args.t_end)
                prefix = f"seed{seed}_"
                print(f"[videos] {scen_name} seed={seed} ...")
                t0 = time.time()
                world = run_headless(cfg, collect_plan_debug_steps=1)
                sim_wall = time.time() - t0
                s = world.metrics.summary
                print(f"  sim: outcome={s.outcome} success={s.success} "
                      f"min_rho={s.min_rho:.2f} min_h_V={s.min_h_V:+.3f} "
                      f"min_mu_V={s.min_mu_V:+.2f} ({sim_wall:.1f}s)")
                cfg.save_json(scen_dir / f"{prefix}config.json")
                world.metrics.save_json(scen_dir / f"{prefix}metrics.json")
                summary = asdict(s)
                summary["scenario"] = scen_name
                summary["seed"] = seed
                summary["sim_wall_s"] = sim_wall
                with open(scen_dir / f"{prefix}summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                plot_run_timeseries(world.metrics, cfg,
                                    scen_dir / f"{prefix}run_timeseries.png")
                t0v = time.time()
                path = _render_video(
                    world.metrics, cfg, scen_dir / f"{prefix}run.mp4",
                    fps=args.video_fps, stride=args.video_stride,
                    plan_debug_history=world.plan_debug_history,
                    prefer_opengl=(not args.video_mpl),
                    pip=args.pip,
                    pip_width=args.pip_width,
                    pip_height=args.pip_height,
                    pip_scale=args.pip_scale,
                    method_tag=METHOD_LABEL.get(
                        method, method).split("\n")[0],
                )
                video_wall = time.time() - t0v
                print(f"  video: {path.relative_to(out_root)} "
                      f"({video_wall:.1f}s)")
                rendered.append(path)
                rows.append({
                    "scenario": scen_name, "method": method, "attacker": atk,
                    "seed": int(seed), "outcome": s.outcome,
                    "success": bool(s.success),
                    "end_time": float(s.end_time),
                    "min_rho": float(s.min_rho),
                    "min_h_V": float(s.min_h_V),
                    "min_mu_V": float(s.min_mu_V),
                    "min_eta_V": float(s.min_eta_V),
                    "eta_viol_integral": float(s.eta_viol_integral),
                    "sim_wall_s": float(sim_wall),
                    "video_wall_s": float(video_wall),
                })
    csv_path = out_root / "videos_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n[videos] wrote {len(rendered)} videos under {out_root}/")
    print(f"[videos] aggregate: {csv_path}")
    return 0


# --------------------------------------------------------------------------- #
# Ablation mode
# --------------------------------------------------------------------------- #


def _ts_matrix(runs, field, t_grid):
    M = np.full((len(runs), len(t_grid)), np.nan)
    for i, r in enumerate(runs):
        t = np.asarray(r["ts_t"], dtype=float)
        y = np.asarray(r[field], dtype=float)
        if t.size < 2:
            continue
        mask = t_grid <= float(t[-1]) + 1e-9
        M[i, mask] = np.interp(t_grid[mask], t, y)
    return M


def _binom_se(p, n):
    n = max(int(n), 1)
    return float(np.sqrt(max(0.0, p * (1.0 - p)) / n))


def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _by_method(results, methods):
    out = {m: [] for m in methods}
    for r in results:
        if r["method"] in out:
            out[r["method"]].append(r)
    return out


def _plot_method_bars(results, methods, field, ylabel, title, out_path,
                      reference_lines=None, log_y=False):
    """Mean +/- SEM bar plot."""
    by = _by_method(results, methods)
    means, sems = [], []
    for m in methods:
        d = np.array([r[field] for r in by[m]], dtype=float)
        if d.size == 0:
            means.append(0.0); sems.append(0.0); continue
        means.append(float(np.mean(d)))
        sems.append(float(np.std(d, ddof=1) / np.sqrt(d.size))
                    if d.size > 1 else 0.0)
    xs = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(methods) + 3), 4.4))
    ax.bar(xs, means, yerr=sems,
           color=[METHOD_COLOR[m] for m in methods], alpha=0.95,
           capsize=6, error_kw=dict(ecolor="#222", elinewidth=1.4))
    for i, (v, s_e) in enumerate(zip(means, sems)):
        ax.text(i, v + (s_e if not log_y else 0) + 0.02 * max(1.0, max(abs(x) for x in means)),
                f"{v:.2f}", ha="center", va="bottom",
                fontweight="bold", fontsize=9)
    if reference_lines:
        for lab, val, col in reference_lines:
            ax.axhline(val, color=col, linestyle="--", linewidth=1.2,
                       alpha=0.7, label=lab)
        ax.legend(loc="best", fontsize=8)
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods],
                       rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_success_panel(results, methods, out_path, title):
    """Headline panel: success rate + intercept / breach / vis-loss counts."""
    by = _by_method(results, methods)
    n_arr = np.array([len(by[m]) for m in methods], dtype=float)
    succ = np.array([sum(1 for r in by[m] if r["success"])
                     for m in methods], dtype=float)
    rate = succ / np.maximum(n_arr, 1.0)
    los, his = [], []
    for k, n in zip(succ, n_arr):
        _, lo, hi = _wilson_ci(int(k), int(n))
        los.append(lo); his.append(hi)
    los = np.array(los); his = np.array(his)
    # Outcome breakdown
    cats = ("intercept", "breach", "visual_loss", "timeout")
    cat_colors = {"intercept": "#27ae60", "breach": "#c0392b",
                  "visual_loss": "#f1c40f", "timeout": "#7f8c8d"}
    by_cat = {c: np.array([sum(1 for r in by[m] if r["outcome"] == c)
                            for m in methods], dtype=float) for c in cats}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xs = np.arange(len(methods))
    ax = axes[0]
    ax.bar(xs, rate, color=[METHOD_COLOR[m] for m in methods], alpha=0.95,
           edgecolor="white", linewidth=0.5)
    yerr_lo = np.maximum(rate - los, 0.0)
    yerr_hi = np.maximum(his - rate, 0.0)
    ax.errorbar(xs, rate, yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor="black", capsize=4)
    for i, p in enumerate(rate):
        ax.text(xs[i], p + 0.02, f"{p*100:.0f}%", ha="center",
                fontweight="bold", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods],
                       rotation=15, ha="right")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel(r"$P_{\rm succ}$  $= \mathbb{P}[\tau_I \leq \min(\tau_B,\tau_L)]$")
    ax.set_title("Success rate (95% Wilson CI)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    bottoms = np.zeros(len(methods))
    for c in cats:
        vals = by_cat[c]
        ax.bar(xs, vals, bottom=bottoms, color=cat_colors[c],
               label=c, edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods],
                       rotation=15, ha="right")
    ax.set_ylabel("trials")
    ax.set_title("Outcome composition")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_timeseries_band(results, methods, field, ylabel, title, out_path,
                          t_max=4.0, dt=0.02, reference_lines=None,
                          y_lim=None, band_lo=25, band_hi=75):
    by = _by_method(results, methods)
    t_grid = np.arange(0.0, t_max + dt, dt)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for m in methods:
        runs = by[m]
        if not runs:
            continue
        M = _ts_matrix(runs, field, t_grid)
        with np.errstate(all="ignore"):
            med = np.nanmedian(M, axis=0)
            q_lo = np.nanpercentile(M, band_lo, axis=0)
            q_hi = np.nanpercentile(M, band_hi, axis=0)
        color = METHOD_COLOR[m]
        ax.fill_between(t_grid, q_lo, q_hi, color=color, alpha=0.18,
                        edgecolor="none")
        ax.plot(t_grid, med, color=color, lw=2.2, label=METHOD_LABEL[m])
    if reference_lines:
        for lab, val, col in reference_lines:
            ax.axhline(val, color=col, linestyle="--", linewidth=1.2,
                       alpha=0.7, label=lab)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, t_max)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_headline_panel(results, methods, out_path, attacker_label,
                         t_max=3.0):
    """Combined 2x2 panel: success-rate, h_V, mu_V, eta_V time-series.

    The single figure paper authors will actually quote in a results
    section.  Top-left: success-rate + outcome breakdown.  Other three:
    median-and-band trajectories of h_V, mu_V, eta_V over time.
    """
    import matplotlib.pyplot as _plt
    by = _by_method(results, methods)
    dt = 0.02
    t_grid = np.arange(0.0, t_max + dt, dt)
    fig, axes = _plt.subplots(2, 2, figsize=(13, 8))

    # (0,0) success + outcome bar
    n_arr = np.array([len(by[m]) for m in methods], dtype=float)
    succ = np.array([sum(1 for r in by[m] if r["success"])
                     for m in methods], dtype=float)
    rate = succ / np.maximum(n_arr, 1.0)
    inter = np.array([sum(1 for r in by[m] if r["outcome"] == "intercept")
                      for m in methods], dtype=float) / np.maximum(n_arr, 1.0)
    ax = axes[0, 0]
    xs = np.arange(len(methods))
    w = 0.36
    ax.bar(xs - w/2, inter, width=w, color="#2980b9", alpha=0.85,
           label="intercept (rho < r_c)")
    ax.bar(xs + w/2, rate, width=w, color="#27ae60", alpha=0.85,
           label=r"$P_{\rm succ}$ = $\tau_I \leq \min(\tau_B,\tau_L)$")
    for i, (a, b) in enumerate(zip(inter, rate)):
        ax.text(xs[i] - w/2, a + 0.02, f"{a*100:.0f}%", ha="center",
                fontsize=9, fontweight="bold")
        ax.text(xs[i] + w/2, b + 0.02, f"{b*100:.0f}%", ha="center",
                fontsize=9, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL[m].split(" (")[0] for m in methods],
                       rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.10)
    ax.set_title("Closure vs strict success")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (0,1) h_V time-series
    ax = axes[0, 1]
    for m in methods:
        runs = by[m]
        if not runs:
            continue
        M = _ts_matrix(runs, "ts_h_V", t_grid)
        with np.errstate(all="ignore"):
            med = np.nanmedian(M, axis=0)
            q_lo = np.nanpercentile(M, 35, axis=0)
            q_hi = np.nanpercentile(M, 65, axis=0)
        ax.fill_between(t_grid, q_lo, q_hi, color=METHOD_COLOR[m], alpha=0.20,
                        edgecolor="none")
        ax.plot(t_grid, med, color=METHOD_COLOR[m], lw=2.0,
                label=METHOD_LABEL[m].split(" (")[0])
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.2,
               alpha=0.7, label=r"$h_V=0$ (lock lost)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$h_V$ (visibility margin)")
    ax.set_xlim(0, t_max)
    ax.set_title(r"$h_V$ trajectory (median + 35-65 pctile band)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) mu_V time-series
    ax = axes[1, 0]
    for m in methods:
        runs = by[m]
        if not runs:
            continue
        M = _ts_matrix(runs, "ts_mu_V", t_grid)
        with np.errstate(all="ignore"):
            med = np.nanmedian(M, axis=0)
            q_lo = np.nanpercentile(M, 35, axis=0)
            q_hi = np.nanpercentile(M, 65, axis=0)
        ax.fill_between(t_grid, q_lo, q_hi, color=METHOD_COLOR[m], alpha=0.20,
                        edgecolor="none")
        ax.plot(t_grid, med, color=METHOD_COLOR[m], lw=2.0,
                label=METHOD_LABEL[m].split(" (")[0])
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.2,
               alpha=0.7, label=r"$\mu_V=0$ (infeasible)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$\mu_V$ (feasibility margin)")
    ax.set_xlim(0, t_max)
    ax.set_title(r"$\mu_V$ trajectory")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) eta_V time-series
    ax = axes[1, 1]
    for m in methods:
        runs = by[m]
        if not runs:
            continue
        M = _ts_matrix(runs, "ts_eta_V", t_grid)
        # Clip outliers for clarity
        M = np.clip(M, -8, 8)
        with np.errstate(all="ignore"):
            med = np.nanmedian(M, axis=0)
            q_lo = np.nanpercentile(M, 35, axis=0)
            q_hi = np.nanpercentile(M, 65, axis=0)
        ax.fill_between(t_grid, q_lo, q_hi, color=METHOD_COLOR[m], alpha=0.20,
                        edgecolor="none")
        ax.plot(t_grid, med, color=METHOD_COLOR[m], lw=2.0,
                label=METHOD_LABEL[m].split(" (")[0])
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.2,
               alpha=0.7, label=r"$\eta_V=0$")
    ax.set_ylim(-6, 6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"$\eta_V$ (command-level residual)")
    ax.set_xlim(0, t_max)
    ax.set_title(r"$\eta_V$ trajectory")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Headline ablation — {attacker_label}  "
                 f"(n={int(n_arr.max())} trials per method)",
                 fontweight="bold")
    _plt.tight_layout(rect=(0, 0, 1, 0.96))
    _plt.savefig(out_path, dpi=150)
    _plt.close(fig)


def _print_ablation_table(results, methods, out_dir, attacker_label):
    by = _by_method(results, methods)
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    out("=" * 96)
    out(f"ABLATION — attacker: {attacker_label}   (n={len(by[methods[0]])} "
        f"per method)")
    out("=" * 96)
    hdr = (f"{'method':35s} {'P_succ':>8s} {'intercept':>10s} "
           f"{'breach':>8s} {'vis_loss':>10s} "
           f"{'h_min':>8s} {'mu_min':>8s} {'eta_viol':>9s} {'tau_I':>7s}")
    out(hdr)
    out("-" * len(hdr))
    for m in methods:
        runs = by[m]; n = max(1, len(runs))
        n_succ = sum(1 for r in runs if r["success"])
        n_int  = sum(1 for r in runs if r["outcome"] == "intercept")
        n_brc  = sum(1 for r in runs if r["outcome"] == "breach")
        n_vl   = sum(1 for r in runs if r["outcome"] == "visual_loss")
        h_min  = np.median([r["min_h_V"]   for r in runs]) if runs else 0.0
        mu_min = np.median([r["min_mu_V"]  for r in runs]) if runs else 0.0
        eta_v  = np.mean(  [r["eta_viol_integral"] for r in runs]) if runs else 0.0
        tau_I  = np.median([r["tau_I"] for r in runs if r["tau_I"] is not None]) \
                    if any(r["tau_I"] is not None for r in runs) else float("nan")
        out(f"{METHOD_LABEL[m]:35s} {n_succ/n:8.0%} {n_int/n:10.0%} "
            f"{n_brc/n:8.0%} {n_vl/n:10.0%} "
            f"{h_min:+8.3f} {mu_min:+8.2f} {eta_v:9.4f} {tau_I:7.3f}")
    out("=" * 96)
    with open(out_dir / "ablation_table.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _run_ablation_for_attacker(attacker, methods, n, workers, out_root,
                               t_end):
    out_dir = out_root / f"ablation_{attacker}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(n))
    jobs = [(m, attacker, seed, t_end, None)
            for m in methods for seed in seeds]
    print(f"\n[ablation:{attacker}] {len(jobs)} sims, {workers} workers")
    t0 = time.time()
    results = _run_trials_parallel(jobs, workers)
    print(f"[ablation:{attacker}] done in {time.time()-t0:.1f}s")

    with open(out_dir / "raw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
    flat_fields = ["method", "attacker", "seed", "outcome", "success",
                   "end_time", "tau_I", "tau_B", "tau_L",
                   "min_rho", "min_attacker_hvu_dist",
                   "min_h_V", "min_mu_V", "min_eta_V",
                   "mean_h_V", "mean_mu_V", "mean_eta_V",
                   "eta_viol_integral",
                   "fraction_h_V_negative", "fraction_mu_V_negative",
                   "fraction_eta_V_negative", "fraction_planner_stressed",
                   "Omega_max", "a_max"]
    with open(out_dir / "summary.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in flat_fields})

    print(f"[ablation:{attacker}] generating plots ...")
    attacker_label = ATTACKER_LABEL.get(attacker, attacker)

    _plot_success_panel(
        results, methods, out_dir / "success_panel.png",
        title=f"Method ablation — {attacker_label}",
    )
    _plot_method_bars(
        results, methods, "min_h_V",
        ylabel=r"median $\min_t h_V$",
        title=f"Worst visibility margin — {attacker_label}",
        out_path=out_dir / "min_h_V_bars.png",
        reference_lines=[(r"$h_V=0$", 0.0, "red")],
    )
    _plot_method_bars(
        results, methods, "min_mu_V",
        ylabel=r"mean $\min_t \mu_V$",
        title=f"Worst visibility-feasibility margin — {attacker_label}",
        out_path=out_dir / "min_mu_V_bars.png",
        reference_lines=[(r"$\mu_V=0$ (infeasible)", 0.0, "red")],
    )
    _plot_method_bars(
        results, methods, "eta_viol_integral",
        ylabel=r"$\int [-\eta_V]_+\, dt$ (s)",
        title=f"Command-level visibility-rate violation — {attacker_label}",
        out_path=out_dir / "eta_viol_bars.png",
    )
    _plot_method_bars(
        results, methods, "fraction_mu_V_negative",
        ylabel=r"fraction of time $\mu_V < 0$",
        title=f"Time spent visibility-infeasible — {attacker_label}",
        out_path=out_dir / "frac_mu_neg_bars.png",
    )
    _plot_method_bars(
        results, methods, "min_attacker_hvu_dist",
        ylabel=r"mean $\min_t \|p_A - p_P\|$ (m)",
        title=f"Closest the attacker got to the asset — {attacker_label}",
        out_path=out_dir / "min_atk_hvu_bars.png",
        reference_lines=[(r"$r_P$ (breach)", 3.0, "red")],
    )
    # Time-series bands
    t_max_plot = min(4.0, max(r["end_time"] for r in results))
    _plot_timeseries_band(
        results, methods, "ts_h_V", ylabel=r"$h_V$",
        title=f"Visibility margin over time — {attacker_label} "
              f"(median + 25-75 pctile)",
        out_path=out_dir / "h_V_timeseries.png",
        t_max=t_max_plot, y_lim=(-1.5, 0.45),
        reference_lines=[(r"$h_V=0$", 0.0, "red")],
    )
    _plot_timeseries_band(
        results, methods, "ts_mu_V", ylabel=r"$\mu_V$",
        title=f"Visibility-feasibility margin over time — {attacker_label}",
        out_path=out_dir / "mu_V_timeseries.png",
        t_max=t_max_plot, y_lim=(-5, 45),
        reference_lines=[(r"$\mu_V=0$", 0.0, "red")],
    )
    _plot_timeseries_band(
        results, methods, "ts_eta_V", ylabel=r"$\eta_V$",
        title=f"Command-level residual over time — {attacker_label}",
        out_path=out_dir / "eta_V_timeseries.png",
        t_max=t_max_plot, y_lim=(-6, 6),
        reference_lines=[(r"$\eta_V=0$", 0.0, "red")],
    )
    _plot_timeseries_band(
        results, methods, "ts_rho", ylabel=r"$\rho$ (m)",
        title=f"Range over time — {attacker_label}",
        out_path=out_dir / "rho_timeseries.png",
        t_max=t_max_plot,
        reference_lines=[(r"$r_c$", 2.5, "green")],
    )
    _plot_headline_panel(results, methods,
                         out_dir / "headline_panel.png",
                         attacker_label, t_max=t_max_plot)
    _print_ablation_table(results, methods, out_dir, attacker_label)
    print(f"[ablation:{attacker}] outputs in {out_dir}/")
    return results


def cmd_ablation(args: argparse.Namespace) -> int:
    n = int(args.n)
    workers = int(args.workers) if args.workers else max(
        1, (os.cpu_count() or 2) - 1)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(",") if args.methods else list(PLANNER_METHODS)
    if args.attackers:
        attackers = args.attackers.split(",")
    elif args.attacker:
        attackers = [args.attacker]
    else:
        attackers = ["smart"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    global plt; plt = _plt
    for atk in attackers:
        _run_ablation_for_attacker(atk, methods, n, workers, out_root,
                                   args.t_end)
    return 0


# --------------------------------------------------------------------------- #
# Feasibility-map mode
# --------------------------------------------------------------------------- #


def _grid_aggregate(grid_results, methods, a_max_vals, Omega_max_vals,
                    accumulator):
    """Helper: build per-method (n_o, n_a) matrix via `accumulator(rs)`."""
    n_a, n_o = len(a_max_vals), len(Omega_max_vals)
    by = {m: {} for m in methods}
    for r in grid_results:
        m = r["method"]
        key = (Omega_max_vals.index(r["Omega_max"]),
               a_max_vals.index(r["a_max"]))
        by[m].setdefault(key, []).append(r)
    out = {}
    for m in methods:
        mat = np.full((n_o, n_a), np.nan)
        for (i_o, i_a), rs in by[m].items():
            mat[i_o, i_a] = float(accumulator(rs))
        out[m] = mat
    return out


def _imshow_grid(matrices, methods, a_max_vals, Omega_max_vals,
                 title, cbar_label, vmin, vmax, cmap, out_path,
                 fmt="{:.0f}", text_threshold=None):
    n_m = len(methods)
    n_a, n_o = len(a_max_vals), len(Omega_max_vals)
    fig, axes = plt.subplots(1, n_m, figsize=(3.7 * n_m + 1.5, 4.7),
                             sharey=True)
    if n_m == 1:
        axes = [axes]
    im = None
    for ax, m in zip(axes, methods):
        mat = matrices[m]
        im = ax.imshow(mat, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto",
                       extent=[-0.5, n_a - 0.5, -0.5, n_o - 0.5])
        ax.set_xticks(np.arange(n_a))
        ax.set_xticklabels([f"{v:.0f}" for v in a_max_vals])
        ax.set_yticks(np.arange(n_o))
        ax.set_yticklabels([f"{v:.0f}" for v in Omega_max_vals])
        ax.set_xlabel(r"intruder $a_A^{\max}$ (m/s²)")
        ax.set_title(METHOD_LABEL[m].split(" (")[0], fontsize=11)
        for i in range(n_o):
            for j in range(n_a):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = "white"
                if text_threshold is not None:
                    lo, hi = text_threshold
                    if lo <= v <= hi:
                        color = "black"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")
    axes[0].set_ylabel(r"defender $\Omega_D^{\max}$ (rad/s)")
    fig.suptitle(title, fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.85, label=cbar_label)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feasibility_map(grid_results, methods, a_max_vals, Omega_max_vals,
                          out_dir, attacker_label):
    """Multiple grid heatmaps: P_succ, intercept rate, <min mu_V>, <h_min>, eta_viol."""
    # P_succ
    rate = _grid_aggregate(
        grid_results, methods, a_max_vals, Omega_max_vals,
        lambda rs: sum(1 for r in rs if r["success"]) / max(1, len(rs)),
    )
    _imshow_grid(rate, methods, a_max_vals, Omega_max_vals,
                 title=rf"$P_{{\rm succ}} = \mathbb{{P}}[\tau_I\leq\min(\tau_B,\tau_L)]$ "
                       rf"(%) — {attacker_label}",
                 cbar_label=r"$P_{\rm succ}$",
                 vmin=0.0, vmax=1.0, cmap="RdYlGn",
                 out_path=out_dir / "feasibility_map_Psucc.png",
                 fmt="{:.0%}", text_threshold=(0.4, 0.8))
    # Intercept rate (closure-only metric, no visibility gating)
    inter = _grid_aggregate(
        grid_results, methods, a_max_vals, Omega_max_vals,
        lambda rs: sum(1 for r in rs if r["outcome"] == "intercept")
                   / max(1, len(rs)),
    )
    _imshow_grid(inter, methods, a_max_vals, Omega_max_vals,
                 title=rf"Closure rate $\mathbb{{P}}[\tau_I < \infty]$ (%) — "
                       rf"{attacker_label}",
                 cbar_label="closure rate",
                 vmin=0.0, vmax=1.0, cmap="RdYlGn",
                 out_path=out_dir / "feasibility_map_intercept.png",
                 fmt="{:.0%}", text_threshold=(0.4, 0.8))
    # <min mu_V>
    mu = _grid_aggregate(
        grid_results, methods, a_max_vals, Omega_max_vals,
        lambda rs: np.mean([r["min_mu_V"] for r in rs]),
    )
    abs_max = max(2.5, max(abs(float(np.nanmax(mu[m])))
                           for m in methods),
                  max(abs(float(np.nanmin(mu[m]))) for m in methods))
    _imshow_grid(mu, methods, a_max_vals, Omega_max_vals,
                 title=rf"Worst feasibility margin $\langle\min_t\mu_V\rangle$ "
                       rf"— {attacker_label}",
                 cbar_label=r"$\langle\min_t \mu_V\rangle$",
                 vmin=-abs_max, vmax=abs_max, cmap="RdBu",
                 out_path=out_dir / "feasibility_map_min_mu.png",
                 fmt="{:+.1f}")
    # <h_min>
    hm = _grid_aggregate(
        grid_results, methods, a_max_vals, Omega_max_vals,
        lambda rs: np.mean([r["min_h_V"] for r in rs]),
    )
    abs_max = max(0.4, max(abs(float(np.nanmax(hm[m])))
                           for m in methods),
                  max(abs(float(np.nanmin(hm[m]))) for m in methods))
    _imshow_grid(hm, methods, a_max_vals, Omega_max_vals,
                 title=rf"Worst visibility margin $\langle\min_t h_V\rangle$ "
                       rf"— {attacker_label}",
                 cbar_label=r"$\langle\min_t h_V\rangle$",
                 vmin=-abs_max, vmax=abs_max, cmap="RdBu",
                 out_path=out_dir / "feasibility_map_min_h.png",
                 fmt="{:+.2f}")
    # <eta_viol>
    ev = _grid_aggregate(
        grid_results, methods, a_max_vals, Omega_max_vals,
        lambda rs: np.mean([r["eta_viol_integral"] for r in rs]),
    )
    vmax = max(0.5, max(float(np.nanmax(ev[m])) for m in methods))
    _imshow_grid(ev, methods, a_max_vals, Omega_max_vals,
                 title=rf"Command-level violation $\langle\int [-\eta_V]_+ dt\rangle$ "
                       rf"— {attacker_label}",
                 cbar_label=r"$\langle\int [-\eta_V]_+ dt\rangle$",
                 vmin=0.0, vmax=vmax, cmap="Reds",
                 out_path=out_dir / "feasibility_map_eta_viol.png",
                 fmt="{:.2f}")


def cmd_map(args: argparse.Namespace) -> int:
    n = int(args.n)
    workers = int(args.workers) if args.workers else max(
        1, (os.cpu_count() or 2) - 1)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(",") if args.methods else list(PLANNER_METHODS)
    attacker = args.attacker
    if args.a_max_vals:
        a_max_vals = [float(x) for x in args.a_max_vals.split(",")]
    else:
        a_max_vals = [25.0, 40.0, 55.0, 70.0]
    if args.Omega_max_vals:
        Omega_max_vals = [float(x) for x in args.Omega_max_vals.split(",")]
    else:
        Omega_max_vals = [15.0, 25.0, 35.0, 50.0]

    out_dir = out_root / f"map_{attacker}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    global plt; plt = _plt

    seeds = list(range(n))
    jobs = []
    for m in methods:
        for a_max in a_max_vals:
            for Om_max in Omega_max_vals:
                for seed in seeds:
                    overrides = {
                        "intruder.a_max": float(a_max),
                        "defender.Omega_max": float(Om_max),
                    }
                    jobs.append((m, attacker, seed, args.t_end, overrides))
    print(f"\n[map:{attacker}] {len(jobs)} sims "
          f"({len(methods)} methods x {len(a_max_vals)} a_max x "
          f"{len(Omega_max_vals)} Omega_max x {n} seeds), {workers} workers")
    t0 = time.time()
    results = _run_trials_parallel(jobs, workers)
    print(f"[map:{attacker}] done in {time.time()-t0:.1f}s")

    with open(out_dir / "raw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
    flat = ["method", "a_max", "Omega_max", "seed", "outcome", "success",
            "min_rho", "min_h_V", "min_mu_V", "min_eta_V",
            "eta_viol_integral"]
    with open(out_dir / "summary.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in flat})
    _plot_feasibility_map(results, methods, a_max_vals, Omega_max_vals,
                          out_dir, ATTACKER_LABEL.get(attacker, attacker))
    print(f"[map:{attacker}] outputs in {out_dir}/")
    return 0


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["demo", "compare", "videos", "ablation", "map"],
                        default="demo")
    parser.add_argument("--out-dir", type=str,
                        default=str(ROOT / "results" / "v2"))
    parser.add_argument("--method", type=str, default="full",
                        choices=list(PLANNER_METHODS))
    parser.add_argument("--methods", type=str, default="",
                        help="Comma-separated list (overrides --method).")
    parser.add_argument("--attacker", type=str, default="pilot_hard",
                        choices=["smart", "pilot_hard", "pilot_easy",
                                 "pilot_fov_exit",
                                 "juke_sinusoidal",
                                 "juke_bangbang", "straight"])
    parser.add_argument("--attackers", type=str, default="",
                        help="Comma-separated list (overrides --attacker).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=20,
                        help="Trials per scenario (ablation/videos/map).")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--workers", type=int, default=0)
    # Video options
    parser.add_argument("--video", action="store_true",
                        help="In demo mode, also render an MP4.")
    parser.add_argument("--video-fps", type=int, default=24)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--video-mpl", action="store_true",
                        help="Force matplotlib renderer (fallback).")
    parser.add_argument("--pip", action="store_true",
                        help="Also render defender first-person POV and "
                             "composite as a PiP inset on the cinematic.")
    parser.add_argument("--pip-width", type=int, default=640,
                        help="POV pass render width before scaling.")
    parser.add_argument("--pip-height", type=int, default=480,
                        help="POV pass render height before scaling.")
    parser.add_argument("--pip-scale", type=float, default=0.30,
                        help="Inset size as fraction of base width (0.20-0.40).")
    # Map mode
    parser.add_argument("--a-max-vals", type=str, default="",
                        help="Comma-separated intruder a_max values for map.")
    parser.add_argument("--Omega-max-vals", type=str, default="",
                        help="Comma-separated defender Omega_max values.")
    args = parser.parse_args(argv)

    if args.mode == "demo":
        return cmd_demo(args)
    if args.mode == "compare":
        return cmd_compare(args)
    if args.mode == "videos":
        return cmd_videos(args)
    if args.mode == "ablation":
        return cmd_ablation(args)
    if args.mode == "map":
        return cmd_map(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
