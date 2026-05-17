"""Visualisation: 3D trajectory, MPPI spaghetti snapshots, time-series.

All figures are plain matplotlib so they render headlessly with the
Agg backend.  Callers pass a `MetricsLog` (and optionally the captured
`plan_debug_history` from the World) and an output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  side-effect register

from .config import SimConfig
from .metrics import MetricsLog


# --------------------------------------------------------------------------- #
# 3D trajectory
# --------------------------------------------------------------------------- #


def _draw_sphere(ax, center, radius, color, alpha=0.2):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 12)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)


def plot_trajectory_3d(metrics: MetricsLog, cfg: SimConfig,
                       out_path: str | Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    p_D = np.array([r.p_D for r in metrics.records])
    p_A = np.array([r.p_A for r in metrics.records])
    ax.plot(p_D[:, 0], p_D[:, 1], p_D[:, 2], "b-", lw=2, label="Defender")
    ax.plot(p_A[:, 0], p_A[:, 1], p_A[:, 2], "r-", lw=2, label="Intruder")
    ax.scatter(*p_D[0], color="blue", s=40, marker="^", label="Def start")
    ax.scatter(*p_A[0], color="red", s=40, marker="^", label="Int start")
    p_P = np.asarray(cfg.geom.p_P)
    _draw_sphere(ax, p_P, cfg.geom.r_P, color="green", alpha=0.20)
    ax.scatter(*p_P, color="green", s=80, marker="o", label="Asset")
    if metrics.summary.outcome == "intercept":
        # Draw collision sphere at the last defender position.
        _draw_sphere(ax, p_D[-1], cfg.geom.r_c, color="orange", alpha=0.4)
    elif metrics.summary.outcome == "breach":
        _draw_sphere(ax, p_A[-1], cfg.geom.r_c, color="red", alpha=0.4)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"3D Engagement -- outcome: {metrics.summary.outcome}")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# MPPI spaghetti snapshots
# --------------------------------------------------------------------------- #


def plot_mppi_snapshots(plan_debug_history: list,
                        metrics: MetricsLog,
                        cfg: SimConfig,
                        out_path: str | Path,
                        n_snapshots: int = 4) -> None:
    """plan_debug_history: list of (t, PlanDebug) tuples captured during the
    sim.  For each chosen snapshot we render two subplots: XY top-down and
    XZ side view, showing all K defender rollouts (line alpha proportional
    to softmin weight; cost-coloured) and all M intruder scenarios.
    """
    if not plan_debug_history:
        return
    n = min(n_snapshots, len(plan_debug_history))
    # Evenly spaced indices including the first and last entry.
    if n == 1:
        idxs = [0]
    else:
        idxs = [round(i * (len(plan_debug_history) - 1) / (n - 1))
                for i in range(n)]
    chosen = [plan_debug_history[i] for i in idxs]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)
    p_executed = np.array([r.p_D for r in metrics.records])
    p_atk_exec = np.array([r.p_A for r in metrics.records])
    t_executed = np.array([r.t for r in metrics.records])
    for col, (t_snap, dbg) in enumerate(chosen):
        K = dbg.p_D.shape[0]
        M = dbg.p_A.shape[0]
        risk = dbg.risk_scores
        # Normalise cost colours
        vmin, vmax = float(risk.min()), float(risk.max())
        if vmax - vmin < 1e-9:
            vmax = vmin + 1.0
        # 2D top-down (XY)
        ax = axes[0, col]
        # Intruder scenarios first (background)
        for m in range(M):
            ax.plot(dbg.p_A[m, :, 0], dbg.p_A[m, :, 1],
                    color="gray", alpha=0.25, lw=0.6, linestyle="--")
        # Defender candidates
        cmap = plt.get_cmap("viridis")
        for i in range(K):
            c = cmap((risk[i] - vmin) / (vmax - vmin))
            alpha = float(min(1.0, 0.05 + dbg.weights[i] * 8.0))
            ax.plot(dbg.p_D[i, :, 0], dbg.p_D[i, :, 1],
                    color=c, alpha=alpha, lw=0.7)
        # Executed trajectory up to this snapshot
        mask = t_executed <= t_snap + 1e-6
        ax.plot(p_executed[mask, 0], p_executed[mask, 1], "b-",
                lw=2.5, label="Defender (executed)")
        ax.plot(p_atk_exec[mask, 0], p_atk_exec[mask, 1], "r-",
                lw=2.5, label="Intruder (true)")
        ax.add_patch(Circle((cfg.geom.p_P[0], cfg.geom.p_P[1]),
                            cfg.geom.r_P, color="green", alpha=0.20))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"t = {t_snap:.2f}s  (XY)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        if col == 0:
            ax.legend(fontsize=7, loc="upper right")
        # 2D side view (XZ)
        ax = axes[1, col]
        for m in range(M):
            ax.plot(dbg.p_A[m, :, 0], dbg.p_A[m, :, 2],
                    color="gray", alpha=0.25, lw=0.6, linestyle="--")
        for i in range(K):
            c = cmap((risk[i] - vmin) / (vmax - vmin))
            alpha = float(min(1.0, 0.05 + dbg.weights[i] * 8.0))
            ax.plot(dbg.p_D[i, :, 0], dbg.p_D[i, :, 2],
                    color=c, alpha=alpha, lw=0.7)
        ax.plot(p_executed[mask, 0], p_executed[mask, 2], "b-", lw=2.5)
        ax.plot(p_atk_exec[mask, 0], p_atk_exec[mask, 2], "r-", lw=2.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"t = {t_snap:.2f}s  (XZ)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
    # Single colorbar referencing the last drawn cmap call.
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6,
                        label="Defender sample risk R_i")
    fig.suptitle(
        f"MPPI rollout snapshots -- K={cfg.mppi.K}, M={cfg.mppi.M}, N={cfg.mppi.N}",
        y=1.02,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #


def plot_run_timeseries(metrics: MetricsLog, cfg: SimConfig,
                        out_path: str | Path) -> None:
    """Six-panel diagnostic: rho, h_V, mu_V, eta_V, |Omega|, sigma."""
    t = np.array([r.t for r in metrics.records])
    rho = np.array([r.rho for r in metrics.records])
    hV = np.array([r.h_V for r in metrics.records])
    muV = np.array([r.mu_V for r in metrics.records])
    etaV = np.array([r.eta_V for r in metrics.records])
    Om = np.array([r.Omega_applied for r in metrics.records])
    sig = np.array([r.sigma_mppi for r in metrics.records])
    stressed = np.array([1.0 if r.planner_stressed else 0.0
                         for r in metrics.records])

    fig, axes = plt.subplots(6, 1, figsize=(9, 12), sharex=True)
    axes[0].plot(t, rho, "b-")
    axes[0].axhline(cfg.geom.r_c, color="r", linestyle="--", label="r_c")
    axes[0].set_ylabel(r"$\rho$ (m)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, hV, "b-")
    axes[1].axhline(0.0, color="r", linestyle="--", label=r"$h_V=0$")
    axes[1].axhline(cfg.visibility.h_safe, color="g", linestyle=":",
                    label=r"$h_{safe}$")
    axes[1].set_ylabel(r"$h_V$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, muV, "b-")
    axes[2].axhline(0.0, color="r", linestyle="--", label=r"$\mu_V=0$")
    axes[2].axhline(cfg.visibility.mu_safe, color="g", linestyle=":",
                    label=r"$\mu_{safe}$")
    axes[2].set_ylabel(r"$\mu_V$")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, etaV, "b-")
    axes[3].axhline(0.0, color="r", linestyle="--", label=r"$\eta_V=0$")
    axes[3].axhline(cfg.visibility.eta_safe, color="g", linestyle=":",
                    label=r"$\eta_{safe}$")
    axes[3].set_ylabel(r"$\eta_V$")
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)

    axes[4].plot(t, np.max(np.abs(Om), axis=1), "b-",
                 label=r"$\|\Omega\|_\infty$")
    axes[4].axhline(cfg.defender.Omega_max, color="r", linestyle="--",
                    label=r"$\Omega_{max}$")
    axes[4].set_ylabel("body rate (rad/s)")
    axes[4].legend(fontsize=8)
    axes[4].grid(alpha=0.3)

    axes[5].plot(t, sig, "b-", label=r"$\sigma$")
    axes[5].axhline(cfg.defender.sigma_max, color="r", linestyle="--",
                    label=r"$\sigma_{max}$")
    axes[5].axhline(cfg.sim.g, color="g", linestyle=":", label="hover")
    axes[5].set_ylabel("thrust (m/s²)")
    axes[5].set_xlabel("t (s)")
    axes[5].legend(fontsize=8)
    axes[5].grid(alpha=0.3)
    # Mark stressed-planning regions as a thin orange ribbon at the top
    # of the rho panel rather than shading every subplot (which buries the
    # actual trajectories).
    if stressed.any():
        ax = axes[0]
        ymin, ymax = ax.get_ylim()
        ribbon_y = ymax + 0.04 * (ymax - ymin)
        ax.scatter(t[stressed > 0.5],
                   np.full(int((stressed > 0.5).sum()), ribbon_y),
                   color="orange", s=10, marker="s",
                   label="planner stressed (gate fallback)")
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Run time series — outcome: {metrics.summary.outcome}  "
                 f"(success={metrics.summary.success})")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Batch outcome bars
# --------------------------------------------------------------------------- #


def plot_batch_outcomes(rows: list[dict], out_path: str | Path) -> None:
    """Stacked bar of outcome counts per scenario."""
    by_scen: dict[str, dict[str, int]] = {}
    for r in rows:
        scen = r.get("scenario", "?")
        out = r.get("outcome", "in_progress")
        by_scen.setdefault(scen, {"intercept": 0, "breach": 0,
                                  "visual_loss": 0, "timeout": 0})
        by_scen[scen][out] = by_scen[scen].get(out, 0) + 1
    scens = sorted(by_scen.keys())
    keys = ["intercept", "breach", "visual_loss", "timeout"]
    colors = {"intercept": "tab:green", "breach": "tab:red",
              "visual_loss": "tab:orange", "timeout": "tab:gray"}
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(scens))
    for k in keys:
        vals = np.array([by_scen[s].get(k, 0) for s in scens], dtype=float)
        ax.bar(scens, vals, bottom=bottoms, color=colors[k], label=k)
        bottoms += vals
    ax.set_ylabel("count")
    ax.set_title("Batch outcomes by scenario")
    ax.legend(fontsize=8)
    plt.xticks(rotation=15)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
