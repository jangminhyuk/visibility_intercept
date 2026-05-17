"""Cinematic 3D engagement video.

Designed to give the same "looks really cool" feel as the omnidrone OpenGL
renderer but built entirely on matplotlib so there is no GL dependency.

Per frame the renderer paints:
    * Dark theme + faint grid + protected-asset translucent sphere.
    * Defender as a small X-frame quadrotor wireframe (4 arms, 4 rotor
      discs, body-x forward indicator), oriented by the logged R_D.
    * Intruder as a red X-frame quadrotor wireframe.
    * Camera FoV cone along the *actual* defender boresight b_D = R_D b_c,
      translucent skyblue.
    * The most recent MPPI plan's K defender candidate rollouts, alpha-
      weighted by their softmin weight and viridis-coloured by their risk
      score -- the "spaghetti" fan that visualises the planner deliberating.
    * The most recent intruder tube's M scenario rollouts in faint dashed
      gray -- the prediction uncertainty.
    * Trails for both vehicles, fading with age.
    * Velocity arrows for both vehicles.
    * LOS line defender->intruder, dashed white.
    * On outcome: multi-layer explosion VFX (5 concentric expanding
      spheres + 16 radial fragment streaks) at the contact point,
      red sphere of doom at the asset on breach, gold flash for
      visual_loss.
    * A HUD strip with scenario name, simulator time, rho, h_V, outcome.
    * Camera that orbits + smoothly zooms in during terminal phase
      (auto-zoom kicks in when rho drops below 5*r_c).

Output: MP4 via ffmpeg when available, GIF fallback via PillowWriter.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .config import SimConfig, cos_theta_F
from .metrics import MetricsLog


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _sphere_mesh(center: np.ndarray, radius: float, n: int = 14):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, max(7, n // 2))
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _disk_mesh(center: np.ndarray, axis: np.ndarray, radius: float,
               n_theta: int = 18):
    """Flat disk mesh perpendicular to `axis`, centred on `center`."""
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    # Build orthonormal basis (e1, e2) perpendicular to axis.
    if abs(axis @ np.array([0.0, 0.0, 1.0])) < 0.95:
        e1 = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    else:
        e1 = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    e1 = e1 / max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(axis, e1)
    theta = np.linspace(0.0, 2 * np.pi, n_theta)
    pts = (center[None, :]
           + radius * (np.cos(theta)[:, None] * e1[None, :]
                       + np.sin(theta)[:, None] * e2[None, :]))
    return pts


def _cone_mesh(apex: np.ndarray, axis: np.ndarray, half_angle: float,
               length: float, n_theta: int = 22, n_r: int = 5):
    """Open cone surface from `apex` along unit `axis`."""
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    if abs(axis @ np.array([0.0, 0.0, 1.0])) < 0.95:
        e1 = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    else:
        e1 = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    e1 = e1 / max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(axis, e1)
    rs = np.linspace(0.0, length, n_r)
    thetas = np.linspace(0.0, 2 * np.pi, n_theta)
    R, T = np.meshgrid(rs, thetas)
    s = R
    rho = s * np.tan(half_angle)
    X = apex[0] + s * axis[0] + rho * (np.cos(T) * e1[0] + np.sin(T) * e2[0])
    Y = apex[1] + s * axis[1] + rho * (np.cos(T) * e1[1] + np.sin(T) * e2[1])
    Z = apex[2] + s * axis[2] + rho * (np.cos(T) * e1[2] + np.sin(T) * e2[2])
    return X, Y, Z


def _quadrotor_lines(center: np.ndarray, R: np.ndarray, arm_length: float
                     ) -> tuple[list, np.ndarray, np.ndarray, list]:
    """Return geometry pieces for an X-frame quadrotor.

    Returns:
        arm_segments: list of 4 (start, end) tuples for arms.
        body_z_world: world-frame thrust axis (R e_3).
        body_x_world: world-frame heading (R e_1, used for forward fin).
        rotor_disks: list of 4 disk polylines centred at the arm tips.
    """
    bx = R @ np.array([1.0, 0.0, 0.0])
    by = R @ np.array([0.0, 1.0, 0.0])
    bz = R @ np.array([0.0, 0.0, 1.0])
    # X-frame: arms at +-45 deg from body-x.
    d1 = (bx + by) / np.sqrt(2.0)
    d2 = (bx - by) / np.sqrt(2.0)
    tips = [center + d1 * arm_length, center + d2 * arm_length,
            center - d1 * arm_length, center - d2 * arm_length]
    arms = [(center, t) for t in tips]
    rotor_disks = [_disk_mesh(t + bz * 0.05 * arm_length, bz,
                              arm_length * 0.32, n_theta=14)
                   for t in tips]
    return arms, bz, bx, rotor_disks


def _R_from_velocity(v: np.ndarray) -> np.ndarray:
    """Visualisation-only rotation: align body-x with horizontal velocity,
    keep body-z roughly world-up.  Used for the intruder where we don't
    log a true attitude.
    """
    vh = np.array([v[0], v[1], 0.0])
    n = float(np.linalg.norm(vh))
    if n < 1e-3:
        return np.eye(3)
    x = vh / n
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    y = y / max(np.linalg.norm(y), 1e-9)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


# --------------------------------------------------------------------------- #
# Main animation
# --------------------------------------------------------------------------- #


# Palette tuned for the dark theme.
COL_ASSET = "#2ecc71"
COL_DEF_BODY = "#4dc8ff"      # cyan-blue defender
COL_DEF_TRAIL = "#1e90ff"
COL_DEF_ARM = "#bcd0e0"
COL_DEF_ROTOR = "#e6f0ff"
COL_DEF_FIN = "#ffd54f"
COL_INT_BODY = "#ff5c4d"      # red-orange intruder
COL_INT_TRAIL = "#ff4040"
COL_INT_ARM = "#f3b6b6"
COL_INT_ROTOR = "#fde0e0"
COL_FOV = "#62c8ff"
COL_LOS = "#aaaaaa"
COL_INTRUDER_TUBE = "#888888"
COL_VFX_HOT = "#ffae42"       # explosion core
COL_VFX_OUTER = "#ff5e3a"     # explosion outer ring
COL_BREACH = "#ff2b1c"
COL_VLOSS = "#f9d423"


def render_video_3d(metrics: MetricsLog, cfg: SimConfig,
                    out_path: str | Path,
                    fps: int = 24,
                    stride: int = 3,
                    plan_debug_history: list | None = None,
                    azim_start_deg: float = -55.0,
                    azim_speed_deg_s: float = 8.0,
                    elev_deg: float = 22.0,
                    fov_length: float = 5.0,
                    trail_seconds: float = 1.5,
                    extra_seconds_after_outcome: float = 1.4,
                    show_mppi_samples: bool = True,
                    n_samples_drawn: int = 64,
                    show_intruder_tube: bool = True,
                    dpi: int = 130) -> Path:
    """Render the engagement as a dark-themed cinematic MP4.

    Parameters
    ----------
    plan_debug_history : list[(t, PlanDebug)] or None
        If supplied, MPPI candidate fans and intruder scenarios are drawn
        for each frame, sourced from the most recent plan_debug whose
        timestamp <= the frame's sim time.
    show_mppi_samples : bool
        Toggle MPPI candidate overlay.
    n_samples_drawn : int
        Cap on K candidates per frame (highest softmin weight kept).
        Pure performance optimisation; default 64 from K=128.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    recs = metrics.records
    if not recs:
        return out_path
    dt = (recs[1].t - recs[0].t) if len(recs) > 1 else cfg.sim.dt_sim
    n_pad = int(round(extra_seconds_after_outcome / max(dt, 1e-6)))
    full_records = recs + [recs[-1]] * n_pad
    full_records = full_records[::stride]

    p_D = np.array([r.p_D for r in full_records])
    p_A = np.array([r.p_A for r in full_records])
    v_D = np.array([r.v_D for r in full_records])
    v_A = np.array([r.v_A for r in full_records])
    R_D = np.array([np.asarray(r.R_D_flat).reshape(3, 3)
                    for r in full_records])
    ts = np.array([r.t for r in full_records])
    rho_log = np.array([r.rho for r in full_records])
    hv_log = np.array([r.h_V for r in full_records])
    plan_ids = [r.plan_debug_idx for r in full_records]
    n_real = (len(recs) + stride - 1) // stride  # frames before VFX tail
    final_outcome = metrics.summary.outcome

    # Axis limits.
    all_pts = np.vstack([p_D, p_A, np.array([cfg.geom.p_P])])
    pad = 4.0
    bbox_min = all_pts.min(axis=0) - pad
    bbox_max = all_pts.max(axis=0) + pad
    span = float((bbox_max - bbox_min).max())
    centre = 0.5 * (bbox_min + bbox_max)
    half = span / 2.0
    x_lim = (centre[0] - half, centre[0] + half)
    y_lim = (centre[1] - half, centre[1] + half)
    z_lim = (max(0.0, centre[2] - half), centre[2] + half)

    plt.style.use("dark_background")
    # Figsize chosen so dpi*figsize yields even pixel dimensions
    # (required by libx264 / yuv420p).
    fig = plt.figure(figsize=(10, 6.4), facecolor="#0a0a14")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a14")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0)

    theta_F = float(np.deg2rad(cfg.visibility.theta_F_deg))
    ctf = cos_theta_F(cfg)
    arm_def = float(cfg.defender.visual_arm_length)
    arm_int = float(cfg.intruder.visual_arm_length)
    b_c_body = np.asarray(cfg.defender.b_c, dtype=float)
    trail_n = max(2, int(trail_seconds / max(dt * stride, 1e-6)))

    # ------------------------------------------------------------------ #
    # Per-frame draw
    # ------------------------------------------------------------------ #
    def draw_quad(center: np.ndarray, R: np.ndarray, arm_length: float,
                  color_arm: str, color_rotor: str, color_body: str,
                  draw_fin: bool = False):
        arms, bz, bx, rotors = _quadrotor_lines(center, R, arm_length)
        for a, b in arms:
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                    color=color_arm, lw=1.8, alpha=0.95)
        for disk in rotors:
            ax.plot(disk[:, 0], disk[:, 1], disk[:, 2],
                    color=color_rotor, lw=1.3, alpha=0.85)
        # Body core sphere (small).
        sx, sy, sz = _sphere_mesh(center, arm_length * 0.25, n=10)
        ax.plot_surface(sx, sy, sz, color=color_body, alpha=0.95,
                        linewidth=0, shade=False)
        if draw_fin:
            tip = center + bx * arm_length * 0.55
            ax.plot([center[0], tip[0]], [center[1], tip[1]],
                    [center[2], tip[2]], color=COL_DEF_FIN, lw=2.5,
                    alpha=0.95)

    def vfx_explosion(center: np.ndarray, phase: float):
        """phase in [0, 1]: 0 = ignition, 1 = max bloom + fade."""
        # Five concentric expanding spheres with decaying alpha.
        for k, (rscale, alpha_base) in enumerate([(0.5, 0.95),
                                                  (1.0, 0.80),
                                                  (1.7, 0.55),
                                                  (2.5, 0.30),
                                                  (3.5, 0.18)]):
            r = cfg.geom.r_c * (0.6 + rscale * (0.3 + 1.6 * phase))
            a = alpha_base * (1.0 - 0.7 * phase)
            color = COL_VFX_HOT if k < 2 else COL_VFX_OUTER
            sx, sy, sz = _sphere_mesh(center, r, n=12)
            ax.plot_surface(sx, sy, sz, color=color, alpha=max(0.05, a),
                            linewidth=0, shade=False)
        # 16 radial fragment streaks (deterministic by phase to keep them
        # stable across frames).
        rng = np.random.default_rng(42)
        dirs = rng.normal(size=(16, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        L = cfg.geom.r_c * (1.5 + 6.0 * phase)
        for d in dirs:
            ax.plot([center[0], center[0] + d[0] * L],
                    [center[1], center[1] + d[1] * L],
                    [center[2], center[2] + d[2] * L],
                    color="white", lw=1.0, alpha=max(0.0, 0.85 - phase))

    def vfx_breach(center: np.ndarray, phase: float):
        for k, rscale in enumerate([0.6, 1.1, 1.8]):
            r = cfg.geom.r_P * (0.8 + rscale * (0.3 + 1.2 * phase))
            sx, sy, sz = _sphere_mesh(center, r, n=14)
            ax.plot_surface(sx, sy, sz, color=COL_BREACH,
                            alpha=max(0.08, 0.6 - phase * 0.4),
                            linewidth=0, shade=False)

    def vfx_vloss(center: np.ndarray, phase: float):
        for k, rscale in enumerate([0.4, 0.9, 1.5]):
            r = 1.2 * (1.0 + rscale * phase)
            sx, sy, sz = _sphere_mesh(center, r, n=10)
            ax.plot_surface(sx, sy, sz, color=COL_VLOSS,
                            alpha=max(0.05, 0.5 - phase * 0.35),
                            linewidth=0, shade=False)

    def draw_static_world():
        ax.cla()
        ax.set_facecolor("#0a0a14")
        # Hide the three big matplotlib "panes" -- they otherwise paint
        # giant gray walls behind the scene that wreck the dark theme.
        for ax_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            ax_axis.pane.fill = False
            ax_axis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
            ax_axis._axinfo["grid"]["color"] = (0.2, 0.25, 0.35, 0.25)
            ax_axis._axinfo["grid"]["linewidth"] = 0.3
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
        # Protected asset.
        pp = np.asarray(cfg.geom.p_P)
        sx, sy, sz = _sphere_mesh(pp, cfg.geom.r_P, n=18)
        ax.plot_surface(sx, sy, sz, color=COL_ASSET, alpha=0.18,
                        linewidth=0, shade=False)
        sx, sy, sz = _sphere_mesh(pp, cfg.geom.r_P * 0.2, n=10)
        ax.plot_surface(sx, sy, sz, color=COL_ASSET, alpha=0.9,
                        linewidth=0, shade=False)

    def update(i):
        draw_static_world()
        # ---- Tracking + zoom camera ----
        rho_now = float(rho_log[min(i, len(rho_log) - 1)])
        # The visible half-span starts at "engagement view" (~12-18m so
        # the asset, defender and intruder all fit) and shrinks during
        # terminal phase so the impact is dramatic.
        far_half = min(22.0, max(14.0, rho_now * 0.65 + 6.0))
        if rho_now < 6 * cfg.geom.r_c:
            close_half = max(4.5, rho_now * 1.5 + cfg.geom.r_c * 2.0)
            half_z = close_half
        else:
            half_z = far_half
        # Camera centre: weighted blend of the two vehicles' midpoint and
        # the asset.  In the close-up phase we look almost entirely at
        # the contact point; far away we keep the asset in view.
        mid = 0.5 * (p_D[i] + p_A[i])
        pp = np.asarray(cfg.geom.p_P, dtype=float)
        if rho_now < 6 * cfg.geom.r_c:
            cam_centre = mid
        else:
            w = float(np.clip(rho_now / 40.0, 0.0, 0.5))
            cam_centre = (1.0 - w) * mid + w * pp
        ax.set_xlim(cam_centre[0] - half_z, cam_centre[0] + half_z)
        ax.set_ylim(cam_centre[1] - half_z, cam_centre[1] + half_z)
        ax.set_zlim(max(0.0, cam_centre[2] - half_z * 0.6),
                    cam_centre[2] + half_z * 0.6)

        # ---- MPPI candidate fan + intruder tube ----
        if (show_mppi_samples and plan_debug_history
                and plan_ids[i] >= 0):
            idx = min(plan_ids[i], len(plan_debug_history) - 1)
            _t, dbg = plan_debug_history[idx]
            weights = dbg.weights
            risk = dbg.risk_scores
            vmin = float(risk.min()); vmax = float(risk.max())
            if vmax - vmin < 1e-6:
                vmax = vmin + 1.0
            # Sort by weight, keep top n_samples_drawn.
            order = np.argsort(-weights)
            order = order[:min(n_samples_drawn, len(order))]
            # Use a high-contrast cool-to-hot palette over the dark
            # background.  Low-cost = bright cyan, high-cost = pink/red.
            cmap = plt.get_cmap("cool")
            for k in order:
                col = cmap(1.0 - (risk[k] - vmin) / (vmax - vmin))
                alpha = float(min(1.0, 0.18 + weights[k] * 8.0))
                ax.plot(dbg.p_D[k, :, 0], dbg.p_D[k, :, 1],
                        dbg.p_D[k, :, 2],
                        color=col, lw=1.0, alpha=alpha)
            if show_intruder_tube:
                for m in range(dbg.p_A.shape[0]):
                    ax.plot(dbg.p_A[m, :, 0], dbg.p_A[m, :, 1],
                            dbg.p_A[m, :, 2],
                            color=COL_INTRUDER_TUBE, lw=0.6,
                            alpha=0.32, linestyle="--")

        # ---- Trails ----
        s = max(0, i - trail_n)
        if i > 0:
            ax.plot(p_D[s:i + 1, 0], p_D[s:i + 1, 1], p_D[s:i + 1, 2],
                    color=COL_DEF_TRAIL, lw=2.0, alpha=0.85)
            ax.plot(p_A[s:i + 1, 0], p_A[s:i + 1, 1], p_A[s:i + 1, 2],
                    color=COL_INT_TRAIL, lw=2.0, alpha=0.85)

        # ---- Vehicles ----
        # Hide vehicles in the VFX tail to let the explosion shine.
        in_tail = i >= n_real
        if not in_tail:
            draw_quad(p_D[i], R_D[i], arm_def,
                      COL_DEF_ARM, COL_DEF_ROTOR, COL_DEF_BODY,
                      draw_fin=True)
            R_int = _R_from_velocity(v_A[i])
            draw_quad(p_A[i], R_int, arm_int,
                      COL_INT_ARM, COL_INT_ROTOR, COL_INT_BODY,
                      draw_fin=False)

            # Velocity arrows.
            if np.linalg.norm(v_D[i]) > 0.1:
                ax.quiver(*p_D[i], *(v_D[i] * 0.18), color=COL_DEF_BODY,
                          arrow_length_ratio=0.3, lw=1.2, alpha=0.85)
            if np.linalg.norm(v_A[i]) > 0.1:
                ax.quiver(*p_A[i], *(v_A[i] * 0.15), color=COL_INT_BODY,
                          arrow_length_ratio=0.3, lw=1.2, alpha=0.85)

            # FoV cone along ACTUAL boresight b_D = R_D b_c.
            b_world = R_D[i] @ b_c_body
            X, Y, Z = _cone_mesh(p_D[i], b_world, theta_F, fov_length)
            ax.plot_surface(X, Y, Z, color=COL_FOV, alpha=0.10,
                            linewidth=0, shade=False)

            # LOS line defender->intruder.
            ax.plot([p_D[i, 0], p_A[i, 0]],
                    [p_D[i, 1], p_A[i, 1]],
                    [p_D[i, 2], p_A[i, 2]],
                    color=COL_LOS, lw=0.9, linestyle="--", alpha=0.55)

        # ---- VFX ----
        if in_tail:
            tail_frac = (i - n_real) / max(1, len(full_records) - n_real)
            if final_outcome == "intercept":
                vfx_explosion(0.5 * (p_D[i] + p_A[i]), tail_frac)
            elif final_outcome == "breach":
                vfx_breach(np.asarray(cfg.geom.p_P), tail_frac)
            elif final_outcome == "visual_loss":
                vfx_vloss(p_D[i], tail_frac)

        # ---- Camera orbit ----
        az = azim_start_deg + azim_speed_deg_s * ts[i]
        # Pitch drops slightly during terminal phase for drama.
        terminal = max(0.0, 1.0 - rho_now / (6 * cfg.geom.r_c))
        ev = elev_deg + 6.0 * terminal
        ax.view_init(elev=ev, azim=az)

        # ---- HUD (title strip) ----
        outcome_color = {
            "intercept": "#7CFC8B",
            "breach": "#ff6e63",
            "visual_loss": "#f9d423",
            "timeout": "#cccccc",
            "in_progress": "#cccccc",
        }.get(final_outcome, "#cccccc")
        ax.set_title(
            f"{cfg.scenario.name}    "
            f"t = {ts[i]:5.2f}s    "
            f"rho = {rho_now:5.2f} m    "
            f"h_V = {hv_log[min(i, len(hv_log)-1)]:+.3f}    "
            f"outcome: {final_outcome}",
            color=outcome_color, fontsize=11, pad=8,
        )

    anim = animation.FuncAnimation(fig, update, frames=len(full_records),
                                   interval=1000 / fps, blit=False)
    writer = _pick_writer(fps)
    if writer is None:
        out_path = out_path.with_suffix(".gif")
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        if not str(out_path).endswith(".mp4"):
            out_path = out_path.with_suffix(".mp4")
        anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    plt.style.use("default")
    return out_path


def _pick_writer(fps: int):
    if animation.FFMpegWriter.isAvailable():
        return animation.FFMpegWriter(
            fps=fps, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-loglevel", "warning",
                        "-crf", "20"],
        )
    return None
