"""Simulation loop + outcome detection (v2: no CBF shield).

Per simulator tick (dt_sim):
    1. Estimator updates noisy/delayed observation of intruder.
    2. If enough sim time elapsed since last plan, run MPPI -> (sigma, Omega).
       (Optional: at close range, override with a deterministic terminal
       LOS-pursuit law for the kill stroke.)
    3. Step defender with (sigma, Omega) -- NO CBF SHIELD.
    4. Step intruder.
    5. Log diagnostics (h_V, mu_V, eta_V, gate-stressed flag, event times).
    6. Check terminal events:
         tau_I: first intercept (rho <= r_c)              -> always terminal
         tau_B: first breach (attacker in protected zone) -> terminal
         tau_L: first visual loss (h_V < 0)               -> recorded; the
              simulation continues unless cfg.scenario.terminate_on_visual_loss
              is True.  This lets us observe long-horizon visibility/feasibility
              evolution after the first loss event -- the central new
              diagnostic of the v2 paper.

Success per paper Eq. 7:  tau_I <= min(tau_B, tau_L).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .attacker import (
    attacker_attitude_from_accel,
    attacker_command,
    smart_attacker_command,
    SmartAttackerState,
)
from .config import (
    SimConfig,
    cos_theta_F,
    gravity,
)
from .dynamics import (
    DefenderState,
    IntruderState,
    step_defender,
    step_intruder_kinematic,
)
from .metrics import MetricsLog, StepRecord
from .mppi import PlanDebug, PlannerState, plan
from .visibility import evaluate_visibility, evaluate_eta_V


# --------------------------------------------------------------------------- #
# Simple latency-aware estimator
# --------------------------------------------------------------------------- #


@dataclass
class _Sample:
    t: float
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray
    R: np.ndarray
    h_V: float = 1.0     # visibility margin at observation time


class IntruderEstimator:
    """Returns a noisy, latency-delayed estimate of intruder kinematics.

    When `estimator.vision_gated=True`, the estimator only ingests a new
    sample when `h_V > 0` (target inside FoV).  Otherwise it coasts on
    the last visible sample under a constant-velocity model.  This is
    what makes camera lock loss physically consequential: the planner's
    information channel goes stale.
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.buf: deque[_Sample] = deque(maxlen=512)
        # Track the most recent VISIBLE sample for the vision-gated case.
        self.last_visible: _Sample | None = None

    def observe(self, t: float, state: IntruderState, g: float,
                h_V_now: float = 1.0) -> None:
        """Try to ingest a new sample.  When vision_gated, only accept
        samples taken while h_V > 0.  Always keep the noiseless ground
        truth in the long buffer so latency-delayed estimates still
        work (the buffer drops samples below h_V only at the gating
        check inside `estimate`)."""
        R = attacker_attitude_from_accel(state.a, g)
        s = _Sample(t=t, p=state.p.copy(), v=state.v.copy(),
                    a=state.a.copy(), R=R, h_V=float(h_V_now))
        self.buf.append(s)
        if h_V_now >= 0.0:
            self.last_visible = s

    def estimate(self, t_now: float) -> _Sample | None:
        """Realistic camera-pipeline estimator.

        Pipeline delay: an observation taken at time t is AVAILABLE to
        the planner at time t + latency.  Therefore the most recent data
        available at t_now is from physical time target_t = t_now - latency.

        Vision-gated: the planner only receives an observation if the
        target was inside the FoV at the observation time (h_V > 0).
        If the most-recent-visible observation is older than target_t,
        the estimator coasts on a constant-velocity model forward to
        target_t with noise that inflates with coast age.
        """
        ep = self.cfg.estimator
        if not self.buf:
            return None
        target_t = t_now - ep.latency
        if ep.vision_gated:
            # The latest visible buffered sample with t <= target_t is
            # what physically reaches the planner right now.  If no such
            # sample exists, coast on `last_visible`.
            candidates = [s for s in self.buf
                          if s.t <= target_t + 1e-9 and s.h_V > 0.0]
            if candidates:
                base = candidates[-1]   # most recent visible-and-arrived
                coast_age = max(0.0, target_t - base.t)
                coast_factor = 1.0
            elif self.last_visible is not None:
                # Lock was lost before any sample at target_t arrived.
                # Coast from last visible.
                base = self.last_visible
                coast_age = max(0.0, target_t - base.t)
                if coast_age > ep.coast_max_age:
                    coast_factor = float(ep.coast_max_noise_scale)
                else:
                    coast_factor = 1.0 + (
                        (ep.coast_max_noise_scale - 1.0)
                        * coast_age / max(1e-6, ep.coast_max_age))
            else:
                # No lock ever acquired.  Use latest available buffered
                # sample (we still want the planner to have *some*
                # initial state guess).
                base = self.buf[-1]
                coast_age = 0.0
                coast_factor = float(ep.coast_max_noise_scale)
            # Constant-velocity coast from base to target_t.
            p_coast = base.p + base.v * coast_age
            v_coast = base.v.copy()
            scale = coast_factor
            p_hat = p_coast + self.rng.normal(0.0,
                                              ep.pos_noise_std * scale,
                                              size=3)
            v_hat = v_coast + self.rng.normal(0.0,
                                              ep.vel_noise_std * scale,
                                              size=3)
            a_hat = base.a.copy()
            return _Sample(t=target_t, p=p_hat, v=v_hat, a=a_hat,
                            R=base.R, h_V=base.h_V)
        # Legacy (non-gated) path: pick closest buffered sample.
        target_t = t_now - ep.latency
        best = self.buf[0]
        best_err = abs(best.t - target_t)
        for s in self.buf:
            err = abs(s.t - target_t)
            if err < best_err:
                best, best_err = s, err
        p_hat = best.p + self.rng.normal(0.0, ep.pos_noise_std, size=3)
        v_hat = best.v + self.rng.normal(0.0, ep.vel_noise_std, size=3)
        a_hat = best.a.copy()
        if len(self.buf) >= 2:
            s_old, s_new = self.buf[-2], self.buf[-1]
            dt = max(1e-3, s_new.t - s_old.t)
            a_hat = (s_new.v - s_old.v) / dt
        return _Sample(t=best.t, p=p_hat, v=v_hat, a=a_hat, R=best.R)


# --------------------------------------------------------------------------- #
# World
# --------------------------------------------------------------------------- #


@dataclass
class World:
    cfg: SimConfig
    defender: DefenderState
    intruder: IntruderState
    estimator: IntruderEstimator
    planner_state: PlannerState
    rng: np.random.Generator
    metrics: MetricsLog = field(default_factory=MetricsLog)
    t: float = 0.0
    last_plan_t: float = -1e9
    last_sigma: float = 0.0           # raw planner output
    last_Omega_applied: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # LPF state: smoothed commands actually fed to the dynamics.
    sigma_lpf: float | None = None
    Omega_lpf: np.ndarray | None = None
    in_terminal_commit: bool = False
    terminal_commit_engaged: bool = False
    lock_loss_t0: float = -1.0
    last_plan_debug: Optional[PlanDebug] = None
    plan_debug_history: list = field(default_factory=list)
    last_planner_stressed: bool = False
    outcome: str = "in_progress"
    intercept_point: Optional[np.ndarray] = None
    # First-loss-of-lock time tau_L (NaN if never lost).  Recorded but does
    # not terminate the sim unless cfg.scenario.terminate_on_visual_loss.
    tau_L: float = float("nan")
    tau_B: float = float("nan")
    tau_I: float = float("nan")
    attacker_g: SmartAttackerState = field(default_factory=SmartAttackerState)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: SimConfig,
                    rng: Optional[np.random.Generator] = None) -> "World":
        if rng is None:
            rng = np.random.default_rng(cfg.sim.random_seed)
        defender = DefenderState(
            p=np.asarray(cfg.defender.p0, dtype=float),
            v=np.asarray(cfg.defender.v0, dtype=float),
            R=np.eye(3),
        )
        b_c = np.asarray(cfg.defender.b_c, dtype=float)
        nb = float(np.linalg.norm(b_c))
        to_int = np.asarray(cfg.intruder.p0, dtype=float) - defender.p
        nto = float(np.linalg.norm(to_int))
        if nb > 1e-6 and nto > 1e-6:
            from .so3 import project_so3
            b_c_hat = b_c / nb
            target_dir = to_int / nto
            z_col = target_dir
            world_x = np.array([1.0, 0.0, 0.0])
            x_hint = world_x - (world_x @ z_col) * z_col
            if float(np.linalg.norm(x_hint)) < 1e-3:
                x_hint = np.cross(z_col, np.array([0.0, 0.0, 1.0]))
            x_col = x_hint / max(float(np.linalg.norm(x_hint)), 1e-9)
            y_col = np.cross(z_col, x_col)
            y_col = y_col / max(float(np.linalg.norm(y_col)), 1e-9)
            x_col = np.cross(y_col, z_col)
            body_basis = np.column_stack([x_col, y_col, z_col])
            ez = np.array([0.0, 0.0, 1.0])
            cos_a = float(np.clip(np.dot(b_c_hat, ez), -1.0, 1.0))
            if cos_a > 1.0 - 1e-9:
                R_align = np.eye(3)
            elif cos_a < -1.0 + 1e-9:
                R_align = np.diag([1.0, -1.0, -1.0])
            else:
                axis_b = np.cross(ez, b_c_hat)
                axis_b /= max(float(np.linalg.norm(axis_b)), 1e-9)
                from .so3 import exp_so3
                import math
                R_align = exp_so3(axis_b * math.acos(cos_a))
            defender.R = project_so3(body_basis @ R_align.T)
        intruder = IntruderState(
            p=np.asarray(cfg.intruder.p0, dtype=float),
            v=np.asarray(cfg.intruder.v0, dtype=float),
        )
        estimator = IntruderEstimator(cfg, rng)
        planner_state = PlannerState.from_config(cfg)
        return cls(
            cfg=cfg,
            defender=defender,
            intruder=intruder,
            estimator=estimator,
            planner_state=planner_state,
            rng=rng,
        )

    # ------------------------------------------------------------------ #
    def _step_attacker(self) -> None:
        mode = self.cfg.scenario.attacker_mode
        if mode in ("smart", "pilot_hard", "pilot_easy"):
            a = smart_attacker_command(
                self.cfg, self.intruder,
                self.defender.p, self.defender.v,
                self.t, self.attacker_g, self.cfg.sim.dt_sim,
            )
        else:
            a = attacker_command(self.cfg, self.intruder, self.t)
        self.intruder.a = a
        self.intruder = step_intruder_kinematic(self.intruder, a,
                                                self.cfg.sim.dt_sim)

    # ------------------------------------------------------------------ #
    def _check_outcomes(self, vis: dict) -> None:
        """Update tau_I/tau_B/tau_L (record first-event timestamps) and set
        self.outcome.  Default policy: terminate on intercept, breach, or
        timeout; visual loss is recorded but does NOT terminate.
        """
        cfg = self.cfg
        rho = float(np.linalg.norm(self.intruder.p - self.defender.p))
        if rho <= cfg.geom.r_c:
            if np.isnan(self.tau_I):
                self.tau_I = self.t
            self.outcome = "intercept"
            self.intercept_point = 0.5 * (self.intruder.p + self.defender.p)
            return

        d_aP = float(np.linalg.norm(self.intruder.p
                                    - np.asarray(cfg.geom.p_P)))
        if d_aP <= cfg.geom.r_P:
            if np.isnan(self.tau_B):
                self.tau_B = self.t
            if cfg.scenario.terminate_on_breach:
                self.outcome = "breach"
                self.intercept_point = self.intruder.p.copy()
                return

        if vis["h_V"] < 0.0 and np.isnan(self.tau_L):
            self.tau_L = self.t
        if (cfg.scenario.terminate_on_visual_loss
                and not np.isnan(self.tau_L)
                and self.t - self.tau_L > 0.05):
            self.outcome = "visual_loss"
            self.intercept_point = self.defender.p.copy()
            return

        if self.t >= cfg.sim.t_end:
            self.outcome = "timeout"

    # ------------------------------------------------------------------ #
    def step(self, collect_plan_debug: bool = False) -> None:
        if self.outcome != "in_progress":
            return
        cfg = self.cfg
        g = cfg.sim.g

        # Visibility at the CURRENT pre-step state -- used to gate the
        # estimator measurement.  When h_V < 0 the camera doesn't see the
        # target so no new sample lands in the visible buffer.
        vis_pre = evaluate_visibility(cfg, self.defender.R, self.defender.p,
                                      self.defender.v, self.intruder.p,
                                      self.intruder.v)
        self.estimator.observe(self.t, self.intruder, g,
                               h_V_now=float(vis_pre["h_V"]))

        # Replan if due.
        if self.t - self.last_plan_t >= cfg.sim.dt_plan or self.last_plan_t < 0:
            est = self.estimator.estimate(self.t)
            if est is None:
                p_hat = self.intruder.p.copy()
                v_hat = self.intruder.v.copy()
                R_hat = np.eye(3)
                a_hat = np.zeros(3)
            else:
                p_hat, v_hat, R_hat, a_hat = est.p, est.v, est.R, est.a
            if cfg.scenario.use_mppi:
                sigma_0, Omega_0, self.planner_state, dbg, stressed = plan(
                    cfg, self.planner_state, self.defender,
                    p_hat, v_hat, R_hat, a_hat,
                    self.rng,
                    collect_debug=collect_plan_debug,
                )
                self.last_sigma = sigma_0
                self.last_Omega_applied = Omega_0
                self.last_planner_stressed = bool(stressed)
                if dbg is not None:
                    self.last_plan_debug = dbg
                    self.plan_debug_history.append((self.t, dbg))
            else:
                # PN-style baseline.
                sigma_0, Omega_0 = self._pn_command(p_hat, v_hat)
                self.last_sigma = sigma_0
                self.last_Omega_applied = Omega_0
            self.last_plan_t = self.t

        # Terminal-phase commit with hysteresis on lock.
        self.in_terminal_commit = False
        h_now = float(vis_pre["h_V"])
        d_cfg = cfg.defender
        # Track sustained-loss-of-lock timer (used by hysteresis).
        if h_now > d_cfg.terminal_commit_enter_h:
            self.lock_loss_t0 = -1.0
        elif self.lock_loss_t0 < 0.0:
            self.lock_loss_t0 = self.t
        sustained_loss = (self.lock_loss_t0 >= 0.0
                           and (self.t - self.lock_loss_t0)
                                > d_cfg.terminal_commit_lock_loss_grace)
        if d_cfg.terminal_commit_requires_lock:
            if not self.terminal_commit_engaged:
                # Enter only on solid lock
                self.terminal_commit_engaged = (h_now
                                                 > d_cfg.terminal_commit_enter_h)
            else:
                # Exit only after sustained lock loss below the exit
                # threshold.
                if h_now <= d_cfg.terminal_commit_exit_h and sustained_loss:
                    self.terminal_commit_engaged = False
        else:
            self.terminal_commit_engaged = True
        if (cfg.scenario.use_mppi
                and d_cfg.terminal_commit_distance > 0.0
                and self.terminal_commit_engaged):
            est_term = self.estimator.estimate(self.t)
            if est_term is not None:
                rho_est = float(np.linalg.norm(est_term.p - self.defender.p))
                if rho_est < d_cfg.terminal_commit_distance:
                    sigma_pn, Omega_pn = self._terminal_commit_command(
                        est_term.p, est_term.v,
                    )
                    self.last_sigma = sigma_pn
                    self.last_Omega_applied = Omega_pn
                    self.in_terminal_commit = True

        # Single-pole LPF on the executed command.  Kills the high-frequency
        # wobble between MPPI replans and against estimator noise; tau is
        # short (~20 ms) so the response delay is negligible relative to
        # the engagement timescale (~1.5 s).
        raw_sigma = float(self.last_sigma)
        raw_Omega = self.last_Omega_applied.copy()
        tau = float(cfg.defender.cmd_lpf_tau)
        bypass = (self.in_terminal_commit
                  and cfg.defender.cmd_lpf_bypass_in_terminal)
        if tau > 0.0 and not bypass:
            alpha = cfg.sim.dt_sim / (tau + cfg.sim.dt_sim)
            if self.sigma_lpf is None:
                self.sigma_lpf = raw_sigma
                self.Omega_lpf = raw_Omega.copy()
            self.sigma_lpf = alpha * raw_sigma + (1.0 - alpha) * self.sigma_lpf
            self.Omega_lpf = alpha * raw_Omega + (1.0 - alpha) * self.Omega_lpf
            sigma_exec = float(self.sigma_lpf)
            Omega_exec = self.Omega_lpf.copy()
        else:
            # In terminal commit, snap LPF to raw command so the kill
            # stroke is unfiltered AND subsequent re-entry into MPPI mode
            # starts from a fresh state.
            self.sigma_lpf = raw_sigma
            self.Omega_lpf = raw_Omega.copy()
            sigma_exec = raw_sigma
            Omega_exec = raw_Omega
        Omega_applied = Omega_exec

        # Step defender.
        self.defender = step_defender(
            self.defender, sigma_exec, Omega_applied,
            cfg.sim.dt_sim, g,
        )
        # Step attacker.
        self._step_attacker()

        # Diagnostics: visibility tensor at the just-stepped state.
        vis = evaluate_visibility(cfg, self.defender.R, self.defender.p,
                                  self.defender.v, self.intruder.p,
                                  self.intruder.v)
        eta_V_now = evaluate_eta_V(
            cfg, self.defender.R, self.defender.p, self.defender.v,
            self.intruder.p, self.intruder.v, Omega_applied,
        )

        # Outcomes.
        self._check_outcomes(vis)

        rho = float(np.linalg.norm(self.intruder.p - self.defender.p))
        atk_hvu = float(np.linalg.norm(
            self.intruder.p - np.asarray(cfg.geom.p_P)
        ))
        est_now = self.estimator.estimate(self.t) if self.estimator.buf else None
        a_hat_log = est_now.a if est_now is not None else np.zeros(3)
        self.metrics.record(StepRecord(
            t=self.t,
            p_D=list(map(float, self.defender.p)),
            v_D=list(map(float, self.defender.v)),
            p_A=list(map(float, self.intruder.p)),
            v_A=list(map(float, self.intruder.v)),
            a_A_true=list(map(float, self.intruder.a)),
            a_A_hat=list(map(float, a_hat_log)),
            sigma_mppi=float(sigma_exec),
            Omega_mppi=list(map(float, self.last_Omega_applied)),
            Omega_applied=list(map(float, Omega_applied)),
            rho=rho,
            h_V=float(vis["h_V"]),
            mu_V=float(vis["mu_V"]),
            eta_V=float(eta_V_now),
            attacker_hvu_dist=atk_hvu,
            outcome=self.outcome,
            R_D_flat=list(map(float, self.defender.R.reshape(9))),
            plan_debug_idx=(len(self.plan_debug_history) - 1
                            if self.plan_debug_history else -1),
            planner_stressed=bool(self.last_planner_stressed),
        ))
        self.t += cfg.sim.dt_sim

    # ------------------------------------------------------------------ #
    def _terminal_commit_command(self, p_hat: np.ndarray, v_hat: np.ndarray
                                 ) -> tuple[float, np.ndarray]:
        cfg = self.cfg
        to_int = p_hat - self.defender.p
        n = float(np.linalg.norm(to_int))
        if n < 1e-6:
            return cfg.defender.sigma_max, np.zeros(3)
        los = to_int / n
        if cfg.defender.terminal_commit_lead:
            v_rel = v_hat - self.defender.v
            v_closing = float(-np.dot(v_rel, los))
            if v_closing > 0.5:
                T_imp = min(n / v_closing, 0.5)
                p_lead = p_hat + v_hat * T_imp
                to_lead = p_lead - self.defender.p
                n_lead = float(np.linalg.norm(to_lead))
                if n_lead > 1e-6:
                    los = to_lead / n_lead
        b_c = np.asarray(cfg.defender.b_c, dtype=float)
        b_c = b_c / max(float(np.linalg.norm(b_c)), 1e-9)
        b_D = self.defender.R @ b_c
        axis_world = np.cross(b_D, los)
        gain = float(cfg.defender.terminal_commit_omega_gain)
        Omega_body = self.defender.R.T @ (gain * axis_world)
        Omega_body = np.clip(Omega_body,
                             -cfg.defender.Omega_max,
                             cfg.defender.Omega_max)
        sigma = float(cfg.defender.sigma_max)
        return sigma, Omega_body

    # ------------------------------------------------------------------ #
    def _pn_command(self, p_hat: np.ndarray, v_hat: np.ndarray
                    ) -> tuple[float, np.ndarray]:
        """Classical proportional-navigation baseline.

        Slew body so the boresight (R @ b_c) tracks the line-of-sight
        toward the estimated attacker; thrust at near-maximum sigma so
        the defender closes at maximum available acceleration.  This is
        the "naive but committed" baseline: no anticipation, no
        visibility planning, just track-and-thrust.
        """
        to_int = p_hat - self.defender.p
        n = float(np.linalg.norm(to_int))
        if n < 1e-6:
            return self.cfg.sim.g, np.zeros(3)
        los = to_int / n
        b_c = np.asarray(self.cfg.defender.b_c, dtype=float)
        b_c = b_c / max(float(np.linalg.norm(b_c)), 1e-9)
        b_D = self.defender.R @ b_c
        axis_world = np.cross(b_D, los)
        # PN body-rate gain: proportional to the angular error.  Higher
        # gain means tighter tracking but more noise sensitivity.
        Omega_body = self.defender.R.T @ (7.0 * axis_world)
        Omega_body = np.clip(Omega_body, -self.cfg.defender.Omega_max,
                             self.cfg.defender.Omega_max)
        # Thrust at near-max sigma so the naive baseline isn't hampered
        # by under-thrusting; the trade-off the naive baseline gives up
        # is anticipation, not raw acceleration.
        v_rel_los = float((v_hat - self.defender.v) @ los)
        sigma = (0.85 * self.cfg.defender.sigma_max
                  + max(0.0, -v_rel_los) * 0.3)
        sigma = float(np.clip(sigma, 0.0, self.cfg.defender.sigma_max))
        return sigma, Omega_body

    # ------------------------------------------------------------------ #
    def finalize(self) -> None:
        self.metrics.finalize(self.outcome, self.t,
                              tau_I=self.tau_I,
                              tau_B=self.tau_B,
                              tau_L=self.tau_L)


def run_headless(cfg: SimConfig,
                 collect_plan_debug_steps: int = 0,
                 max_steps: int | None = None,
                 rng: Optional[np.random.Generator] = None) -> World:
    """Run a single episode and return the World object."""
    world = World.from_config(cfg, rng=rng)
    if max_steps is None:
        max_steps = int(cfg.sim.t_end / cfg.sim.dt_sim) + 10
    always_collect = collect_plan_debug_steps > 0
    for step_idx in range(max_steps):
        world.step(collect_plan_debug=always_collect)
        if world.outcome != "in_progress":
            break
    world.finalize()
    return world
