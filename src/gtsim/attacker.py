"""Ground-truth intruder policies for the sim loop.

These are NOT the prediction tube; they are the actual behaviour the simulator
will integrate.  The defender's MPPI sees this attacker only through a noisy,
latency-delayed, vision-gated estimator.

Modes:
    straight         -- accelerate toward the protected asset.
    juke_sinusoidal  -- straight approach + sinusoidal lateral juke.
    juke_bangbang    -- straight + bang-bang lateral juke at fixed period.
    pilot_easy       -- realistic "casual" pilot: smooth banking course
                        corrections plus occasional small lateral break
                        when the defender comes close.  Speed and lateral-
                        acceleration limits are *below* the defender's.
    pilot_hard       -- realistic "expert" pilot (the old "smart" mode):
                        smooth banking approach + sustained aggressive
                        lateral break timed to the defender's commitment,
                        with smooth recovery and continued pursuit of the
                        HVU.  Performs multiple breaks across a longer
                        engagement -- the defender that misses on the
                        first pass gets another chance, but the attacker
                        will try the break again if approached again.
    smart            -- alias for pilot_hard (back-compat with old configs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import IntruderParams, SimConfig
from .dynamics import IntruderState


def _toward_target(p: np.ndarray, target: np.ndarray) -> np.ndarray:
    d = np.asarray(target, dtype=float) - np.asarray(p, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return d / n


def attacker_command(
    cfg: SimConfig,
    state: IntruderState,
    t: float,
) -> np.ndarray:
    """Returns world-frame desired acceleration (m/s^2) given the scenario.

    All modes cap |a| at intruder.a_max.  The juke is purely additive to
    the straight-line component.
    """
    ip: IntruderParams = cfg.intruder
    target = np.asarray(cfg.geom.p_P, dtype=float)
    to_t = _toward_target(state.p, target)
    speed = float(np.linalg.norm(state.v))
    # Track a target velocity toward the asset, capped at v_max.
    v_des = ip.v_max * to_t
    a_track = 5.0 * (v_des - state.v)
    a_juke = np.zeros(3)
    mode = cfg.scenario.attacker_mode
    juke_axis = np.asarray(ip.juke_axis, dtype=float)
    nj = float(np.linalg.norm(juke_axis))
    if nj > 1e-9:
        juke_axis = juke_axis / nj
    if mode == "juke_sinusoidal":
        a_juke = ip.juke_amplitude * np.sin(
            2.0 * np.pi * ip.juke_freq_hz * t
        ) * juke_axis
    elif mode == "juke_bangbang":
        period = max(1e-3, ip.bangbang_period)
        sign = 1.0 if int(t / period) % 2 == 0 else -1.0
        a_juke = ip.juke_amplitude * sign * juke_axis
    elif mode == "straight":
        a_juke = np.zeros(3)
    elif mode == "smart":
        # The smart mode is handled by `smart_attacker_command` which
        # needs the defender state; world.py dispatches accordingly.
        # This branch should never fire when world is wired correctly.
        raise ValueError(
            "attacker_command does not handle mode='smart' -- "
            "use smart_attacker_command(cfg, state, p_D, v_D, t, g, dt) "
            "instead."
        )
    else:
        raise ValueError(f"Unknown attacker mode: {mode!r}")
    a = a_track + a_juke
    # Cap magnitude.
    n = float(np.linalg.norm(a))
    if n > ip.a_max and n > 1e-9:
        a = a * (ip.a_max / n)
    return a


# --------------------------------------------------------------------------- #
# Smart reactive attacker
# --------------------------------------------------------------------------- #


@dataclass
class SmartAttackerState:
    """Persistent state for the pilot-style attackers (smart / pilot_hard /
    pilot_easy).  All persistent fields live here so the attacker is
    pure-function over its state.
    """
    juke_side: int = 0
    juke_dwell: float = 0.0
    last_juke_dir: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Break-engagement state.  When the defender first enters
    # `terminal_break_distance`, latch into break mode for
    # `terminal_break_duration` seconds.  After the break, COOLDOWN for
    # `terminal_break_cooldown` seconds before allowing another break --
    # so a determined defender that misses the first pass gets another
    # window of "clean" attacker geometry to re-engage.
    in_terminal_break: bool = False
    terminal_break_t0: float = -1.0
    terminal_break_side: int = 0
    terminal_break_count: int = 0
    last_break_end_t: float = -1.0
    # Banking-course state (used by both pilot_easy and pilot_hard).
    # Smooth random lateral acceleration drift that simulates a human
    # pilot's small course corrections during the approach.
    bank_accel_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bank_target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bank_t_next: float = 0.0
    seed_rng_inited: bool = False
    # Per-attacker rng (seeded from the world rng at first use) so the
    # banking pattern is a deterministic function of the engagement seed.
    rng: object = None


def _safe_unit(v: np.ndarray, default: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.asarray(default, dtype=float)
    return v / n


def _ensure_rng(g_state: SmartAttackerState, world_seed: int) -> None:
    if not g_state.seed_rng_inited:
        g_state.rng = np.random.default_rng(world_seed * 31 + 17)
        g_state.seed_rng_inited = True


def _bank_target_update(g_state: SmartAttackerState, t: float,
                        to_hvu: np.ndarray, bank_amp: float,
                        bank_period: float) -> None:
    """Pick a new lateral-acceleration target every `bank_period` seconds.

    The target is drawn as a random unit vector perpendicular to the
    attack axis (banks around the direct line) scaled by a random
    amplitude up to `bank_amp`.  This produces smooth banking course
    corrections rather than abrupt jukes.
    """
    if t < g_state.bank_t_next and float(np.linalg.norm(g_state.bank_target)) > 1e-9:
        return
    rng = g_state.rng
    # Random lateral direction perp to to_hvu
    n = rng.normal(size=3)
    n -= float(n @ to_hvu) * to_hvu
    n = _safe_unit(n, default=np.array([0.0, 1.0, 0.0]))
    # Bias slightly horizontal (less vertical), realistic-pilot style
    n[2] *= 0.35
    n = _safe_unit(n, default=n)
    amp = float(rng.uniform(0.35, 1.0)) * bank_amp
    g_state.bank_target = amp * n
    g_state.bank_t_next = t + float(rng.uniform(0.7, 1.6)) * bank_period


def pilot_attacker_command(
    cfg: SimConfig,
    state: IntruderState,
    defender_p: np.ndarray,
    defender_v: np.ndarray,
    t: float,
    g_state: SmartAttackerState,
    dt: float,
    *,
    hard: bool,
    defender_R: np.ndarray | None = None,
    fov_exit_mode: bool = False,
) -> np.ndarray:
    """Realistic "human-pilot" attacker (used for both pilot_easy and
    pilot_hard via the `hard` flag).

    Behaviour
    ---------
    1) **Banking approach.**  The attacker tracks a velocity toward the
       HVU with a smooth lateral-acceleration drift that is renewed on a
       random `bank_period` schedule.  This simulates an unmanned pilot's
       small course corrections; both easy and hard pilots use this.
    2) **Reactive lateral break.**  When the defender comes within
       `terminal_break_distance` AND the cooldown has expired, the
       attacker latches into a sustained banking break for
       `terminal_break_duration` seconds, in a direction chosen to be
       maximally off the defender's current approach vector.
    3) **Smooth recovery.**  When the break ends, the attacker rolls
       back toward the HVU heading over `recovery_tau` seconds (no abrupt
       direction change).
    4) **Multi-pass.**  After the cooldown, if the defender re-engages,
       the attacker triggers another break.  So a defender that misses
       the first pass can curl back and engage on the second.

    Easy pilot (hard=False) uses *small* break amplitude (~ a_max * 0.5)
    and longer cooldown; hard pilot uses sustained ~ a_max * 1.0 breaks
    and shorter cooldown so a missed first pass is still tested.
    """
    ip: IntruderParams = cfg.intruder
    p = np.asarray(state.p, dtype=float)
    v = np.asarray(state.v, dtype=float)
    p_P = np.asarray(cfg.geom.p_P, dtype=float)
    p_D = np.asarray(defender_p, dtype=float)
    v_D = np.asarray(defender_v, dtype=float)

    # Seed the per-attacker RNG from the world's seed for reproducibility.
    _ensure_rng(g_state, int(cfg.sim.random_seed))

    # --- Relative geometry ---
    hvu_offset = p_P - p
    dist_to_hvu = float(np.linalg.norm(hvu_offset))
    to_hvu = _safe_unit(hvu_offset, default=np.array([-1.0, 0.0, 0.0]))

    to_def = p_D - p
    dist_to_def = float(np.linalg.norm(to_def))
    perp_vec = (p_D - p) - float(np.dot(p_D - p, to_hvu)) * to_hvu
    perp_dist = float(np.linalg.norm(perp_vec))

    # --- Mode parameters (easy vs hard) ---
    if hard:
        bank_amp = float(getattr(ip, "pilot_hard_bank_amp", 18.0))
        bank_period = float(getattr(ip, "pilot_hard_bank_period", 0.9))
        break_strength = float(getattr(ip, "pilot_hard_break_strength", 1.0))
        break_dur = float(getattr(ip, "pilot_hard_break_duration", 0.45))
        break_dist = float(getattr(ip, "pilot_hard_break_distance", 9.0))
        break_cooldown = float(getattr(ip, "pilot_hard_break_cooldown", 0.8))
        recovery_tau = float(getattr(ip, "pilot_hard_recovery_tau", 0.40))
        target_speed_frac = 0.95
    else:
        bank_amp = float(getattr(ip, "pilot_easy_bank_amp", 8.0))
        bank_period = float(getattr(ip, "pilot_easy_bank_period", 1.4))
        break_strength = float(getattr(ip, "pilot_easy_break_strength", 0.55))
        break_dur = float(getattr(ip, "pilot_easy_break_duration", 0.35))
        break_dist = float(getattr(ip, "pilot_easy_break_distance", 7.0))
        break_cooldown = float(getattr(ip, "pilot_easy_break_cooldown", 1.4))
        recovery_tau = float(getattr(ip, "pilot_easy_recovery_tau", 0.55))
        target_speed_frac = 0.98

    # --- Banking course corrections (smooth random drift) ---
    _bank_target_update(g_state, t, to_hvu, bank_amp, bank_period)
    # Smoothly LPF current bank_accel toward bank_target with tau~0.25s.
    tau_bank = 0.25
    alpha_bank = dt / max(tau_bank + dt, 1e-6)
    g_state.bank_accel_world = (
        (1 - alpha_bank) * g_state.bank_accel_world
        + alpha_bank * g_state.bank_target
    )

    # --- Reactive break latch with cooldown ---
    closing_to_def = -float(np.dot(v_D - v, to_def / max(dist_to_def, 1e-6)))
    cooled_down = (t - g_state.last_break_end_t) > break_cooldown
    if (not g_state.in_terminal_break
            and dist_to_def < break_dist
            and closing_to_def > 1.5
            and cooled_down):
        g_state.in_terminal_break = True
        g_state.terminal_break_t0 = float(t)
        g_state.terminal_break_count += 1
        los_to_def = to_def / max(dist_to_def, 1e-6)
        if fov_exit_mode and defender_R is not None:
            # FoV-exit break: pick the break direction that maximises
            # the rate of LOS rotation AWAY from the defender's camera
            # boresight, i.e. anti-parallel to the projection of
            # b_D = R_D @ b_c onto the plane perpendicular to the LOS
            # from defender -> attacker.  This is the direction that
            # makes h_V = b_D . r_hat - cos(theta_F) drop fastest.
            b_c_body = np.asarray(cfg.defender.b_c, dtype=float)
            b_c_norm = float(np.linalg.norm(b_c_body))
            if b_c_norm > 1e-9:
                b_c_body = b_c_body / b_c_norm
            R_D = np.asarray(defender_R, dtype=float).reshape(3, 3)
            b_D_world = R_D @ b_c_body
            # LOS from defender to attacker (r_hat in the h_V formula).
            r_def_to_att = -to_def
            r_norm = float(np.linalg.norm(r_def_to_att))
            if r_norm > 1e-9:
                r_hat = r_def_to_att / r_norm
            else:
                r_hat = -los_to_def
            b_perp = b_D_world - float(np.dot(b_D_world, r_hat)) * r_hat
            n_bp = float(np.linalg.norm(b_perp))
            if n_bp > 0.05:
                # Break OPPOSITE to b_perp so attacker velocity reduces
                # b_D . r_hat (drops h_V fastest).  Project onto plane
                # perp to to_hvu so the break is lateral.
                fov_break = -b_perp / n_bp
                fov_break = (fov_break
                             - float(np.dot(fov_break, to_hvu)) * to_hvu)
                n_fb = float(np.linalg.norm(fov_break))
                if n_fb > 0.05:
                    break_dir = fov_break / n_fb
                else:
                    rng = g_state.rng
                    r = rng.normal(size=3)
                    r -= float(r @ to_hvu) * to_hvu
                    break_dir = _safe_unit(r,
                                           default=np.array([0.0, 1.0, 0.0]))
            else:
                # Camera nearly perfectly on-axis; any perpendicular
                # direction exits the cone equally, so pick one
                # consistent with anti-defender-velocity.
                v_def_perp = (v_D
                              - float(np.dot(v_D, los_to_def)) * los_to_def)
                v_def_perp -= float(np.dot(v_def_perp, to_hvu)) * to_hvu
                n_vperp = float(np.linalg.norm(v_def_perp))
                if n_vperp > 0.5:
                    break_dir = -v_def_perp / n_vperp
                else:
                    rng = g_state.rng
                    r = rng.normal(size=3)
                    r -= float(r @ to_hvu) * to_hvu
                    break_dir = _safe_unit(r,
                                           default=np.array([0.0, 1.0, 0.0]))
        else:
            # Default pilot break: AWAY from defender's velocity heading
            # (so the defender has to slew most).
            v_def_perp = v_D - float(np.dot(v_D, los_to_def)) * los_to_def
            v_def_perp_along_attack = (
                v_def_perp - float(np.dot(v_def_perp, to_hvu)) * to_hvu)
            n_vperp = float(np.linalg.norm(v_def_perp_along_attack))
            if n_vperp > 0.5:
                break_dir = -v_def_perp_along_attack / n_vperp
            else:
                rng = g_state.rng
                r = rng.normal(size=3)
                r -= float(r @ to_hvu) * to_hvu
                break_dir = _safe_unit(r, default=np.array([0.0, 1.0, 0.0]))
        # Slight bias to keep break mostly horizontal.
        break_dir[2] *= 0.45
        break_dir = _safe_unit(break_dir, default=break_dir)
        g_state.last_juke_dir = break_dir.copy()
        g_state.terminal_break_side = +1   # sign baked into break_dir
    if g_state.in_terminal_break:
        if t - g_state.terminal_break_t0 > break_dur:
            g_state.in_terminal_break = False
            g_state.last_break_end_t = float(t)

    # --- Build desired velocity ---
    target_speed = ip.v_max * target_speed_frac
    desired_dir = to_hvu.copy()

    if g_state.in_terminal_break:
        # Smooth ramp in/out of the break direction.  Use a half-cosine
        # envelope so the lateral accel is sustained but boundaries are
        # smooth (no abrupt direction change that a human pilot wouldn't
        # do).
        phase = (t - g_state.terminal_break_t0) / max(break_dur, 1e-6)
        env = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))  # 0 -> 1 -> 0
        bank_add = env * break_strength * bank_amp * g_state.last_juke_dir
    else:
        # Post-break recovery: exponentially decay any residual lateral
        # bank toward zero.  Combined with the LPF on bank_accel_world
        # above this gives smooth recovery.
        bank_add = np.zeros(3)

    # Total desired lateral acceleration = banking drift + break add
    a_lateral = g_state.bank_accel_world + bank_add

    # Decompose: feed lateral acceleration via a desired-direction tilt,
    # not via instantaneous acceleration substitution.  Use a soft
    # blending of to_hvu and the lateral direction.
    if float(np.linalg.norm(a_lateral)) > 1e-6:
        # Build desired_dir as to_hvu rotated toward a_lateral
        weight = float(np.linalg.norm(a_lateral)) / max(bank_amp, 1e-6)
        weight = min(weight, 1.2)
        desired_dir = _safe_unit(to_hvu + 0.45 * weight * (a_lateral / max(float(np.linalg.norm(a_lateral)), 1e-6)),
                                 default=to_hvu)

    desired_v = desired_dir * target_speed
    a_des = ip.kp_velocity * (desired_v - v) + a_lateral

    # Cap magnitude at the attacker's a_max (with a tiny over-budget for
    # cornering transients).
    n = float(np.linalg.norm(a_des))
    cap = ip.a_max * 1.05
    if n > cap and n > 1e-9:
        a_des = a_des * (cap / n)
    return a_des


def smart_attacker_command(
    cfg: SimConfig,
    state: IntruderState,
    defender_p: np.ndarray,
    defender_v: np.ndarray,
    t: float,
    g_state: SmartAttackerState,
    dt: float,
    defender_R: np.ndarray | None = None,
) -> np.ndarray:
    """Back-compat wrapper.  Routes to the appropriate pilot variant
    based on `cfg.scenario.attacker_mode`.

    Modes:
      smart / pilot_hard  -- aggressive banking break perpendicular to
                             defender velocity (existing behaviour).
      pilot_easy          -- milder banking-pilot.
      pilot_fov_exit      -- like pilot_hard but break direction is
                             aimed to drive the defender's camera
                             boresight off the LOS as fast as possible.
                             Requires defender_R.
    """
    mode = cfg.scenario.attacker_mode
    hard = mode in ("smart", "pilot_hard", "pilot_fov_exit")
    fov_exit = (mode == "pilot_fov_exit")
    return pilot_attacker_command(cfg, state, defender_p, defender_v, t,
                                  g_state, dt, hard=hard,
                                  defender_R=defender_R,
                                  fov_exit_mode=fov_exit)


def attacker_attitude_from_accel(a: np.ndarray, g: float) -> np.ndarray:
    """Build a plausible attacker attitude R_A whose body-z aligns with the
    thrust direction implied by a + g e_3.

    This is only used to feed the defender's estimator a meaningful R_A_hat
    in the sim (since the tube model uses R_A_hat to extract sigma_A_bar).
    """
    from .config import E3
    f = np.asarray(a, dtype=float) + g * E3
    nf = float(np.linalg.norm(f))
    if nf < 1e-6:
        return np.eye(3)
    z = f / nf
    # Build orthonormal basis (x, y, z) with body-z = z, body-x ~ world-x.
    x_hint = np.array([1.0, 0.0, 0.0])
    if abs(float(z @ x_hint)) > 0.95:
        x_hint = np.array([0.0, 1.0, 0.0])
    x = x_hint - (x_hint @ z) * z
    x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
