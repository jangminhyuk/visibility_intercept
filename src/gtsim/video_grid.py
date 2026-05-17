"""Combined 2x2 method-comparison video.

Renders the same engagement seed under all four planner methods in a
single 2x2 grid MP4, with synchronized time, so the viewer can directly
compare body slewing, camera tracking, and closure across methods.

Each panel shows:
    - 3D world: asset (green sphere), defender body (blue X-quad),
      intruder body (red X-quad), camera FoV cone, trails.
    - HUD top-left: method label, outcome marker.
    - HUD bottom: t, rho, h_V (color-coded: green/yellow/red), tau_I if reached.

Uses pure matplotlib (Agg backend) -- no OpenGL dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  side-effect register

from .config import SimConfig, cos_theta_F
from .metrics import MetricsLog


# --------------------------------------------------------------------------- #


def _sphere_mesh(center, radius, n=12):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, max(7, n // 2))
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _x_quad_segments(center, R, arm=0.6):
    """Return list of (start, end) 3D segments for an X-frame quadrotor body."""
    body_arms = arm * np.array([
        [+1, 0, 0], [-1, 0, 0],
        [0, +1, 0], [0, -1, 0],
    ])
    segs = []
    for a in body_arms:
        end = center + R @ a
        segs.append((center, end))
    # Boresight (body-z) as a longer pointer
    bz = R @ np.array([0, 0, 1.0])
    segs.append((center, center + bz * arm * 1.6))
    return segs


def _cone_circle(apex, axis, half_angle_rad, n=24, length=4.0):
    """Return Nx3 array of points on the FoV cone base."""
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    base_center = apex + axis * length
    r = length * np.tan(half_angle_rad)
    if abs(axis @ np.array([0, 0, 1.0])) < 0.95:
        e1 = np.cross(axis, np.array([0, 0, 1.0]))
    else:
        e1 = np.cross(axis, np.array([1, 0, 0]))
    e1 = e1 / max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(axis, e1)
    theta = np.linspace(0, 2 * np.pi, n)
    pts = (base_center[None, :]
           + r * np.cos(theta)[:, None] * e1[None, :]
           + r * np.sin(theta)[:, None] * e2[None, :])
    return pts


# --------------------------------------------------------------------------- #


OUTCOME_COLOR = {
    "intercept":   "#27ae60",
    "breach":      "#c0392b",
    "visual_loss": "#f1c40f",
    "timeout":     "#7f8c8d",
    "in_progress": "#bdc3c7",
}


def render_compare_grid_video(
    metrics_by_method: dict,
    cfg_by_method: dict,
    method_label_by_id: dict,
    method_color_by_id: dict,
    out_path: str | Path,
    fps: int = 24,
    stride: int = 4,
    trail_seconds: float = 1.0,
    title: str = "Method comparison",
) -> Path:
    """Render a 2x2 method-comparison MP4.

    `metrics_by_method`: dict {method_id: MetricsLog}
    `cfg_by_method`:     dict {method_id: SimConfig}
    `method_label_by_id`: short display label per method
    """
    method_ids = list(metrics_by_method.keys())
    n_methods = len(method_ids)
    # Pick grid: 2x2 for 4, 2x3 (one empty) for 5, 2x3 for 6
    if n_methods <= 4:
        n_rows, n_cols = 2, 2
    elif n_methods <= 6:
        n_rows, n_cols = 2, 3
    else:
        raise ValueError("up to 6 methods supported")

    # Build per-method per-frame arrays on a common grid (use the longest
    # method's records as the timeline).
    arrays = {}
    t_max = 0.0
    for m in method_ids:
        recs = metrics_by_method[m].records
        if not recs:
            continue
        arr = dict(
            t=np.array([r.t for r in recs]),
            p_D=np.array([r.p_D for r in recs]),
            p_A=np.array([r.p_A for r in recs]),
            R_D=np.array([np.asarray(r.R_D_flat).reshape(3, 3)
                          for r in recs]),
            rho=np.array([r.rho for r in recs]),
            h_V=np.array([r.h_V for r in recs]),
            mu_V=np.array([r.mu_V for r in recs]),
            outcome=metrics_by_method[m].summary.outcome,
            success=bool(metrics_by_method[m].summary.success),
        )
        arrays[m] = arr
        t_max = max(t_max, float(arr["t"][-1]))
    # Common time grid in real seconds, decimated by stride*dt_sim.
    dt_sim = cfg_by_method[method_ids[0]].sim.dt_sim
    dt_frame = dt_sim * stride
    n_frames = int(t_max / dt_frame) + 4
    # Hold last frame for 1s post-outcome
    n_hold = int(1.0 / dt_frame)
    n_frames += n_hold
    t_frames = np.arange(n_frames) * dt_frame

    # Bounding box for axes (same across panels)
    all_p = np.vstack(
        [arrays[m]["p_D"] for m in arrays] + [arrays[m]["p_A"] for m in arrays]
        + [np.asarray(cfg_by_method[method_ids[0]].geom.p_P)[None, :]]
    )
    p_min = all_p.min(axis=0) - 2.0
    p_max = all_p.max(axis=0) + 2.0
    p_ctr = 0.5 * (p_min + p_max)
    p_rng = max(float((p_max - p_min).max()), 4.0) * 0.55

    cfg0 = cfg_by_method[method_ids[0]]
    p_P = np.asarray(cfg0.geom.p_P)
    r_P = cfg0.geom.r_P
    r_c = cfg0.geom.r_c
    theta_F = float(np.deg2rad(cfg0.visibility.theta_F_deg))

    fig = plt.figure(figsize=(6.0 * n_cols, 4.5 * n_rows),
                      facecolor="#101218")
    axes = []
    for k in range(n_methods):
        ax = fig.add_subplot(n_rows, n_cols, k + 1, projection="3d",
                              facecolor="#0c0e14")
        ax.set_facecolor("#0c0e14")
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.fill = False
            pane.pane.set_edgecolor("#22262d")
        ax.tick_params(colors="#666870", labelsize=7)
        ax.set_xlim(p_ctr[0] - p_rng, p_ctr[0] + p_rng)
        ax.set_ylim(p_ctr[1] - p_rng, p_ctr[1] + p_rng)
        ax.set_zlim(max(0, p_ctr[2] - p_rng * 0.6), p_ctr[2] + p_rng * 0.6)
        ax.view_init(elev=20, azim=35)
        axes.append(ax)

    fig.suptitle(title, color="#e6e8ee", fontsize=14, fontweight="bold",
                 y=0.985)

    # Artists per panel, initialized empty -- updated in animate()
    panel_artists = []
    for k, m in enumerate(method_ids):
        ax = axes[k]
        # Asset
        xs, ys, zs = _sphere_mesh(p_P, r_P, n=14)
        ax.plot_surface(xs, ys, zs, color="#1d3a1f", alpha=0.30,
                        linewidth=0)
        ax.scatter(*p_P, color="#27ae60", s=80, marker="o",
                   edgecolor="white", linewidth=0.6)
        # Persistent trail line
        trail_def, = ax.plot([], [], [], color=method_color_by_id[m],
                              lw=2.0, alpha=0.85)
        trail_int, = ax.plot([], [], [], color="#e74c3c", lw=1.8,
                              alpha=0.85)
        # Body marker artists (re-drawn each frame using set_data_3d)
        def_arms = []
        for _ in range(5):
            ln, = ax.plot([], [], [], color=method_color_by_id[m],
                           lw=2.0)
            def_arms.append(ln)
        int_arms = []
        for _ in range(4):
            ln, = ax.plot([], [], [], color="#e74c3c", lw=1.6)
            int_arms.append(ln)
        # FoV cone -- updated as a Poly3DCollection-like line ring
        cone_ring, = ax.plot([], [], [], color="#3498db", lw=1.0,
                              alpha=0.6, linestyle="-")
        cone_spokes = [ax.plot([], [], [], color="#3498db", lw=0.6,
                                alpha=0.30)[0] for _ in range(8)]
        # LOS dashed line
        los_line, = ax.plot([], [], [], color="white", lw=0.8,
                             linestyle="--", alpha=0.5)
        # Hit/breach marker (kept zero-size until event)
        hit_pt, = ax.plot([], [], [], "o", markersize=14, alpha=0.0)
        # HUD text artists -- 2D in figure coords overlaid via ax.text2D
        title_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes,
                                color=method_color_by_id[m],
                                fontsize=11, fontweight="bold")
        info_text = ax.text2D(0.02, 0.06, "", transform=ax.transAxes,
                               color="#d0d3da", fontsize=9,
                               family="monospace")
        outcome_text = ax.text2D(0.98, 0.96, "", transform=ax.transAxes,
                                  color="#bdc3c7", fontsize=10,
                                  fontweight="bold", ha="right")

        panel_artists.append(dict(
            method=m, ax=ax, trail_def=trail_def, trail_int=trail_int,
            def_arms=def_arms, int_arms=int_arms,
            cone_ring=cone_ring, cone_spokes=cone_spokes,
            los_line=los_line, hit_pt=hit_pt,
            title_text=title_text, info_text=info_text,
            outcome_text=outcome_text,
        ))

    def update(frame_idx: int):
        t_now = t_frames[frame_idx]
        for art in panel_artists:
            m = art["method"]
            arr = arrays.get(m)
            if arr is None:
                continue
            # Find closest record index
            i = int(np.searchsorted(arr["t"], t_now))
            i = min(i, len(arr["t"]) - 1)
            # Trail window
            t_lo = max(arr["t"][0], t_now - trail_seconds)
            i_lo = int(np.searchsorted(arr["t"], t_lo))
            pD_trail = arr["p_D"][i_lo:i + 1]
            pA_trail = arr["p_A"][i_lo:i + 1]
            art["trail_def"].set_data_3d(pD_trail[:, 0], pD_trail[:, 1],
                                          pD_trail[:, 2])
            art["trail_int"].set_data_3d(pA_trail[:, 0], pA_trail[:, 1],
                                          pA_trail[:, 2])
            # Bodies
            pD = arr["p_D"][i]; pA = arr["p_A"][i]; R = arr["R_D"][i]
            for ln, (a, b) in zip(art["def_arms"],
                                   _x_quad_segments(pD, R, arm=0.7)):
                ln.set_data_3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
            R_int = np.eye(3)
            for ln, (a, b) in zip(art["int_arms"],
                                   _x_quad_segments(pA, R_int, arm=0.55)[:4]):
                ln.set_data_3d([a[0], b[0]], [a[1], b[1]], [a[2], b[2]])
            # FoV cone
            b_c = np.asarray(cfg_by_method[m].defender.b_c, dtype=float)
            axis = R @ b_c
            cone_pts = _cone_circle(pD, axis, theta_F, n=24, length=3.5)
            art["cone_ring"].set_data_3d(cone_pts[:, 0],
                                          cone_pts[:, 1], cone_pts[:, 2])
            for k_s, ln_s in enumerate(art["cone_spokes"]):
                step = len(cone_pts) // len(art["cone_spokes"])
                p_edge = cone_pts[k_s * step]
                ln_s.set_data_3d([pD[0], p_edge[0]],
                                  [pD[1], p_edge[1]],
                                  [pD[2], p_edge[2]])
            # LOS
            art["los_line"].set_data_3d([pD[0], pA[0]],
                                         [pD[1], pA[1]],
                                         [pD[2], pA[2]])
            # HUD
            art["title_text"].set_text(method_label_by_id[m])
            h_V = float(arr["h_V"][i])
            h_color = ("#27ae60" if h_V > 0.05
                        else "#f1c40f" if h_V > -0.05 else "#c0392b")
            art["info_text"].set_text(
                f"t={t_now:5.2f}s\nρ={arr['rho'][i]:5.2f} m\n"
                f"h_V={h_V:+.2f}\nμ_V={arr['mu_V'][i]:+.1f}"
            )
            art["info_text"].set_color(h_color)
            # Outcome marker once final
            t_end_method = float(arr["t"][-1])
            if t_now >= t_end_method:
                oc = arr["outcome"]
                art["outcome_text"].set_text(
                    f"{oc.upper()}"
                    + ("  ✓" if arr["success"] else "  ✗" if oc != "intercept" else "")
                )
                art["outcome_text"].set_color(
                    OUTCOME_COLOR.get(oc, "#bdc3c7"))
                # Hit marker
                if oc == "intercept":
                    pos = 0.5 * (pD + pA)
                    art["hit_pt"].set_data_3d([pos[0]], [pos[1]], [pos[2]])
                    art["hit_pt"].set_color("#27ae60")
                    art["hit_pt"].set_alpha(0.85)
                elif oc == "breach":
                    art["hit_pt"].set_data_3d([pA[0]], [pA[1]], [pA[2]])
                    art["hit_pt"].set_color("#c0392b")
                    art["hit_pt"].set_alpha(0.85)
        return []

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                    interval=1000.0 / fps, blit=False)
    # Save as MP4
    writer = animation.FFMpegWriter(fps=fps, bitrate=4000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out_path), writer=writer, dpi=110)
    plt.close(fig)
    return out_path
