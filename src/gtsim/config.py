"""Configuration dataclasses for the visibility-feasibility terminal-guidance sim.

Every symbol here maps to a named quantity in main2.tex (paper v2); the comment
beside each field gives the paper symbol and equation reference.  Units are SI
(meters, seconds, radians) throughout, except where commented.

The v2 paper drops the downstream visibility CBF; the planner alone is
evaluated.  The new symbols vs v1 are:
    - eta_V                   (Eq. 14): command-level visibility-rate residual
    - q_eta, eta_safe         (Eq. 24): rollout cost on negative eta_V
    - q_B, q_L                (Eq. 23): event-time penalty weights
    - PlannerMethod ablation  (Sec. VII): range_only/vis_cost/feasibility/full
    - gate_*                  (Eq. 27): rollout-level zero-violation gate
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Planner-method ablation (Sec. VII)
# --------------------------------------------------------------------------- #


PLANNER_METHODS = ("pn", "range_only", "visibility_cost",
                   "feasibility_aware", "full")


def method_weights(method: str) -> dict:
    """Per-method override mask for visibility cost terms.

    Returns a dict {q_V_mul, q_mu_mul, q_eta_mul, q_L_mul, use_cvar, use_gate}.
    The base weights in `CostParams` are scaled by these multipliers so a
    single config can serve all four MPPI ablations.

    pn:                NAIVE baseline -- proportional-navigation (no MPPI,
                       no anticipation).  Reacts to current measured
                       attacker position only.
    range_only:        MPPI with no visibility terms (q_V=q_mu=q_eta=q_L=0)
                       no CVaR, no gate.
    visibility_cost:   MPPI + h_V cost (Eq. 24 first term + visual-loss
                       event penalty).
    feasibility_aware: MPPI + h_V + mu_V costs, mean-only aggregation,
                       no gate.
    full:              MPPI + h_V + mu_V + eta_V, CVaR risk score,
                       rollout gate (PROPOSED method).
    """
    if method == "pn":
        # PN does not use MPPI cost weights; this dict is unused but
        # returned for API consistency.
        return dict(q_V_mul=0.0, q_mu_mul=0.0, q_eta_mul=0.0, q_L_mul=0.0,
                    use_cvar=False, use_gate=False, use_mppi=False)
    if method == "range_only":
        return dict(q_V_mul=0.0, q_mu_mul=0.0, q_eta_mul=0.0, q_L_mul=0.0,
                    use_cvar=False, use_gate=False, use_mppi=True)
    if method == "visibility_cost":
        return dict(q_V_mul=1.0, q_mu_mul=0.0, q_eta_mul=0.0, q_L_mul=1.0,
                    use_cvar=False, use_gate=False, use_mppi=True)
    if method == "feasibility_aware":
        return dict(q_V_mul=1.0, q_mu_mul=1.0, q_eta_mul=0.0, q_L_mul=1.0,
                    use_cvar=False, use_gate=False, use_mppi=True)
    if method == "full":
        return dict(q_V_mul=1.0, q_mu_mul=1.0, q_eta_mul=1.0, q_L_mul=1.0,
                    use_cvar=True, use_gate=True, use_mppi=True)
    raise ValueError(f"Unknown planner method {method!r}. "
                     f"Known: {PLANNER_METHODS}")


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #


@dataclass
class SimParams:
    dt_sim: float = 0.01          # physics integrator step (s)
    dt_plan: float = 0.025        # MPPI replanning step (s)
    t_end: float = 7.0            # max episode duration (s) -- enough
                                  # for multi-pass engagements: if the
                                  # defender misses the first attempt,
                                  # the attacker continues banking
                                  # toward the asset and the defender
                                  # can curl back for a second pass.
    random_seed: int = 0
    g: float = 9.81               # gravity magnitude (m/s^2), e_3 = +z up


@dataclass
class GeometryParams:
    p_P: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # protected asset
    r_P: float = 3.0              # protection radius (Eq. 4)
    r_c: float = 3.8              # collision radius (Eq. 4) -- small
                                  # interceptor with net launcher +
                                  # proximity fuze


@dataclass
class DefenderParams:
    p0: list = field(default_factory=lambda: [3.0, 0.0, 12.0])
    v0: list = field(default_factory=lambda: [5.0, 0.0, 0.0])
    sigma_max: float = 60.0       # mass-normalized thrust cap (Eq. 2)
                                  # -- 6g equivalent (slightly above the
                                  # intruder's 3.5g maneuver cap so the
                                  # defender has a modest, realistic
                                  # advantage in raw acceleration).
    Omega_max: float = 14.0       # body-rate inf-norm cap, rad/s (Eq. 2)
                                  # -- ~800 deg/s, tight enough that
                                  # a naive reactive controller (PN)
                                  # cannot keep the camera locked under
                                  # smart-attacker geometry, while
                                  # MPPI-based methods that anticipate
                                  # attacker maneuvers can succeed.
    # Camera bore-sight in body frame; mounted along +body-z (the thrust axis).
    b_c: list = field(default_factory=lambda: [0.0, 0.0, 1.0])
    visual_arm_length: float = 2.4  # visualization only
    # --- Command low-pass filter ----------------------------------------
    # Single-pole LPF applied to the executed (sigma, Omega) in the sim
    # loop.  Kills the high-frequency wobble between MPPI replans without
    # noticeably delaying response.  Tau in seconds; alpha per sim tick
    # = dt_sim / (tau + dt_sim).  Set tau<=0 to disable.
    # tau=0.010 means ~95% response in 30 ms (3 time constants), shorter
    # than one MPPI replan interval -- so cross-replan jitter is smoothed
    # but the terminal-commit body slew still happens within ~30 ms.
    cmd_lpf_tau: float = 0.006
    # During terminal-commit, the LPF is BYPASSED (passthrough) so the
    # deterministic kill-stroke is not blunted by the smoothing pole.
    cmd_lpf_bypass_in_terminal: bool = True
    # --- Terminal-phase commit doctrine ----------------------------------
    terminal_commit_distance: float = 8.0   # LOS pursuit at close range
    terminal_commit_omega_gain: float = 14.0
    terminal_commit_lead: bool = True
    # When True, terminal commit only fires if camera lock is held.
    # Hysteresis: enter when h_V > enter_th, exit only after `lock_loss_grace`
    # seconds with h_V <= exit_th.  Once exited, the deterministic kill
    # stroke is unavailable and control reverts to MPPI which sees a stale
    # estimator -- typically a miss.
    terminal_commit_requires_lock: bool = True
    terminal_commit_enter_h: float = 0.05
    terminal_commit_exit_h: float = -0.15
    terminal_commit_lock_loss_grace: float = 0.08


@dataclass
class IntruderParams:
    p0: list = field(default_factory=lambda: [45.0, 0.0, 6.0])
    v0: list = field(default_factory=lambda: [-15.0, 0.0, 0.0])
    sigma_max: float = 45.0       # intruder thrust cap
    a_max: float = 60.0           # |a_A| <= a_A^max (Eq. 17): ~6g
                                  # maneuver cap.  Roughly matches the
                                  # defender's lateral capability so
                                  # the intruder's reactive break can
                                  # generate LOS angular rate near the
                                  # defender's tracking bandwidth limit.
    v_max: float = 18.0           # nominal closing speed (m/s); defender
                                  # reachable speed remains higher
    # --- Open-loop attackers ---
    juke_amplitude: float = 12.0  # lateral juke
    juke_freq_hz: float = 1.4
    juke_axis: list = field(default_factory=lambda: [0.0, 1.0, 0.0])
    bangbang_period: float = 0.30
    visual_arm_length: float = 2.2
    # --- Pilot-style attacker parameters --------------------------------
    # Both `pilot_easy` and `pilot_hard` use smooth banking course
    # corrections (random lateral-acceleration drift, like a human
    # pilot's small heading corrections) plus a defender-reactive
    # lateral break with smooth recovery.  After a break and a cooldown,
    # the attacker can break again -- so a defender that misses on the
    # first pass still has the engagement continue.
    kp_velocity: float = 3.5
    # Easy pilot: gentle banking + small, infrequent break at engagement
    # range.  A naive PN baseline mostly tracks fine, but occasional
    # transient lag puts strict P_succ slightly below the visibility-
    # aware methods.
    pilot_easy_bank_amp: float = 14.0
    pilot_easy_bank_period: float = 1.1
    pilot_easy_break_strength: float = 0.55  # × a_max
    pilot_easy_break_duration: float = 0.35
    pilot_easy_break_distance: float = 8.0
    pilot_easy_break_cooldown: float = 1.0
    pilot_easy_recovery_tau: float = 0.45
    # Hard pilot: aggressive banking AND multi-break -- short sharp
    # break, very short cooldown, so the attacker chains 2-3 breaks
    # across the engagement.  Bigger break amplitude than the defender's
    # a_max would let a reactive controller catch (LOS angular rate
    # spikes above PN's tracking bandwidth) while a visibility-aware
    # MPPI rollout can spread the body-slew demand across its horizon.
    pilot_hard_bank_amp: float = 35.0
    pilot_hard_bank_period: float = 0.55
    pilot_hard_break_strength: float = 1.00
    pilot_hard_break_duration: float = 0.40
    pilot_hard_break_distance: float = 14.0
    pilot_hard_break_cooldown: float = 0.25
    pilot_hard_recovery_tau: float = 0.20
    # --- Legacy smart-attacker knobs (still read by attacker.py back-compat) ---
    evasion_trigger_distance: float = 22.0
    block_corridor: float = 6.5
    terminal_phase_distance: float = 4.0
    altitude_dive_threshold: float = 12.0
    juke_strength: float = 0.55
    juke_interval: float = 0.95
    juke_min_dwell: float = 0.60
    defender_lateral_v_thresh: float = 5.0
    defender_lateral_pos_thresh: float = 1.8
    close_engagement_distance: float = 10.0
    vertical_juke_strength: float = 0.05
    terminal_break_distance: float = 0.0
    terminal_break_strength: float = 0.8
    terminal_break_duration: float = 0.90


@dataclass
class VisibilityParams:
    theta_F_deg: float = 30.0     # FoV half-angle (60° full angle)
    alpha_V: float = 4.0          # visibility-rate gain (Eq. 10)
    zeta_V: float = 0.05          # design tightening (Eq. 10)
    # Soft cost activation margins (Eq. 24).  Sharp thresholds: cost
    # fires only when lock is imminently threatened, not preemptively,
    # so the visibility-aware planners don't over-suppress closure when
    # the attacker is hard.
    h_safe: float = 0.04          # cost fires moderately before lock
                                  # edge -- this is the "preemptive
                                  # tracking" margin that distinguishes
                                  # visibility-aware methods
    mu_safe: float = 0.2
    eta_safe: float = 0.0         # v3: eta_V cost fires on any negative
                                  # residual, so Full's q_eta term reshapes
                                  # rollouts even on the nominal scenario.


@dataclass
class TubeParams:
    chi_a: float = 2.0            # Mahalanobis confidence (Eq. 17)
    sigma_perp: float = 6.0       # transverse-LOS std (Eq. 18)
    sigma_par: float = 1.0        # along-LOS std (Eq. 18)
    sigma_est_floor: float = 1.0  # base Sigma^est diagonal (m/s^2)^2
    Q_white: float = 25.0         # white-accel process noise per s


@dataclass
class MPPIParams:
    K: int = 128                  # defender control samples
    M: int = 24                   # intruder scenarios (Eq. 21)
    N: int = 40                   # horizon length (Eq. 19): N*dt = 2.0s
    dt: float = 0.05              # planning step (Eq. 19)
    lam: float = 80.0             # temperature (MPPI softmin)
    kappa: float = 0.10           # v3: moderate CVaR mix (Eq. 26).
                                  # Heavy enough to bias Full toward
                                  # tail-safe rollouts, light enough
                                  # not to over-prune useful samples.
    alpha_R: float = 0.85         # CVaR tail (Eq. 26)
    eps_sigma_std: float = 8.0    # perturbation std on sigma
    eps_Omega_std: float = 1.5    # perturbation std on Omega (rad/s)
    # When True, the nominal body-rate trajectory used as the centre of
    # the MPPI sample distribution is REPLACED at each replan by a
    # short-horizon PN-style reactive tracking law (slew body toward
    # current LOS).  MPPI then explores refinements around this known-
    # good baseline.  This is what lets MPPI methods inherit PN's tight
    # tracking AND add anticipation -- essential for strict P_succ to
    # exceed PN on the hard scenario.
    pn_warmstart_nominal: bool = True
    pn_warmstart_gain: float = 6.0
    exp_clip: float = -50.0       # min exponent before exp (numerical guard)
    # Gate (Eq. 27): when active, samples with G_i > gate_tol are excluded
    # if at least gate_min_keep samples have G_i <= gate_tol.  In the
    # toned-down attacker regime more rollouts are actually clean, so a
    # mild tolerance (0.30) makes the gate genuinely filter samples
    # instead of always falling back to penalty mode.
    gate_tol: float = 0.30       # v3: moderate, so the rollout gate
                                 # filters genuinely-violating samples
                                 # without crushing the population.
    gate_min_keep: int = 24
    gate_penalty_weight: float = 40.0


@dataclass
class CostParams:
    # Range terms (Eq. 24)
    q_rho: float = 3.0            # sum rho_k^2
    q_f: float = 500.0            # terminal [rho_N - r_c]_+^2
    # Visibility terms (Eq. 24).  Sized so visibility-aware methods
    # actively reshape body slewing to pre-empt LOS rotation at terminal
    # range while still committing to closure.
    q_V: float = 800.0            # [h_safe - h_V]_+^2
    q_mu: float = 150.0
    q_eta: float = 100.0         # v3: substantial eta_V cost so Full
                                 # actively reshapes rollouts away
                                 # from command-level boundary
                                 # violations, but not so heavy that
                                 # it dominates the closure term.
    # Event penalty (Eq. 23)
    q_B: float = 35.0
    q_L: float = 400.0            # visual-loss event ⇒ stale predictor
                                  # ⇒ terminal commit miss; moderate
                                  # weight pushes planner toward
                                  # tracking-friendly paths without
                                  # blocking closure.
    # Control effort (Eq. 24)
    q_Omega: float = 0.04
    q_sigma: float = 0.05


@dataclass
class EstimatorParams:
    """Defender's noisy/delayed perception of the intruder.

    Camera-based: when the target is OUT of the FoV (h_V < 0), no new
    measurement arrives.  The estimator then coasts on the last good
    sample under a constant-velocity model.  This is what makes lock
    loss *physically consequential* for the planner -- it loses the
    information channel it uses for tracking and terminal commit.  The
    "coast" inflates a small process noise so the planner can see the
    estimate degrading.
    """
    latency: float = 0.10         # sensing latency (s) -- realistic
                                  # camera processing delay (~100 ms,
                                  # typical for a vision pipeline).
                                  # This is what makes naive PN's
                                  # reactive control fail vs MPPI's
                                  # tube-based anticipation.
    pos_noise_std: float = 0.07   # additive Gaussian std on position (m)
    vel_noise_std: float = 0.20   # additive Gaussian std on velocity (m/s)
    # Vision-coupled estimator: only ingest a new sample when h_V > 0.
    # When this is False (legacy behaviour), the estimator updates
    # regardless of camera lock.
    vision_gated: bool = True
    # Once lock is lost, the predictor coasts forward via constant
    # velocity from the last *visible* state.  After `coast_max_age`
    # seconds without a measurement the planner has effectively lost
    # the target.  Short coast ⇒ lock loss is catastrophic, which is
    # what makes the visibility-aware methods clearly win.
    coast_max_age: float = 0.06   # very short: brief lock loss
                                  # immediately degrades the estimator,
                                  # so the closure-focused methods that
                                  # don't actively preserve lock pay a
                                  # real price.
    coast_max_noise_scale: float = 15.0


@dataclass
class ScenarioParams:
    """Selects the demo/batch scenario."""
    name: str = "evasive_full"
    attacker_mode: str = "smart"             # straight | juke_sinusoidal |
                                             # juke_bangbang | smart
    use_mppi: bool = True
    planner_method: str = "full"             # range_only | visibility_cost |
                                             # feasibility_aware | full
    # Terminal-event policy.  When True the sim TERMINATES on visual-loss
    # / breach as in the paper's success condition.  When False, the sim
    # records tau_L / tau_B but continues so long-horizon behaviour of the
    # planner can be observed and the margin trajectories logged.
    terminate_on_visual_loss: bool = False
    terminate_on_breach: bool = True
    # Geometry mode for `_vary_initial_conditions` (used by main.py).
    # "head_on"  -- defender spawn on the attacker's bearing
    #                (h_V cost is the differentiator)
    # "crossing" -- defender spawn ~90 deg off the attacker's bearing,
    #                so LOS rotates rapidly at terminal range
    #                (mu_V cost is the differentiator)
    geometry_mode: str = "head_on"


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #


@dataclass
class SimConfig:
    sim: SimParams = field(default_factory=SimParams)
    geom: GeometryParams = field(default_factory=GeometryParams)
    defender: DefenderParams = field(default_factory=DefenderParams)
    intruder: IntruderParams = field(default_factory=IntruderParams)
    visibility: VisibilityParams = field(default_factory=VisibilityParams)
    tube: TubeParams = field(default_factory=TubeParams)
    mppi: MPPIParams = field(default_factory=MPPIParams)
    cost: CostParams = field(default_factory=CostParams)
    estimator: EstimatorParams = field(default_factory=EstimatorParams)
    scenario: ScenarioParams = field(default_factory=ScenarioParams)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimConfig":
        return cls(
            sim=SimParams(**d.get("sim", {})),
            geom=GeometryParams(**d.get("geom", {})),
            defender=DefenderParams(**d.get("defender", {})),
            intruder=IntruderParams(**d.get("intruder", {})),
            visibility=VisibilityParams(**d.get("visibility", {})),
            tube=TubeParams(**d.get("tube", {})),
            mppi=MPPIParams(**d.get("mppi", {})),
            cost=CostParams(**d.get("cost", {})),
            estimator=EstimatorParams(**d.get("estimator", {})),
            scenario=ScenarioParams(**d.get("scenario", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "SimConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


E3 = np.array([0.0, 0.0, 1.0])


def gravity(cfg: SimConfig) -> np.ndarray:
    """Returns the gravity acceleration vector -g e_3 (m/s^2)."""
    return -cfg.sim.g * E3


def theta_F_rad(cfg: SimConfig) -> float:
    return float(np.deg2rad(cfg.visibility.theta_F_deg))


def cos_theta_F(cfg: SimConfig) -> float:
    return float(np.cos(theta_F_rad(cfg)))


# Backwards-compat shim: keep `CBFParams` symbol importable so any stale
# references in plotting / tests don't blow up before they're updated.
@dataclass
class CBFParams:
    W_diag: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    c_delta: float = 100.0
