"""Scenario-risk MPPI planner for visibility-feasibility terminal guidance.

Implements main2.tex Sec. IV-VI.  Per replan:

    1. Build M intruder scenarios from the tube (`tube.build_intruder_tube`).
    2. Sample K defender perturbations around the running nominal control.
    3. Rollout defender K times and intruder M times -> evaluate KxM cost
       tensor (Eq. 24) over the horizon.  Per (i, m, k) cell the cost uses
       the relative geometry to compute:
           - rho_k                         (range)
           - h_V_k                         (Eq. 5)
           - mu_V_k                        (Eq. 13)
           - eta_V_k = c_Omega^T Omega_seq + c_v + alpha_V h_V - zeta_V  (Eq. 14)
       and aggregates with the event penalty l_ev (Eq. 23).
    4. Risk score per defender sample = mean + CVaR (Eq. 26).
    5. Optional rollout-level gate G_i (Eq. 27): if there exist samples with
       G_i <= tol, MPPI weights are computed only over those samples; else
       a heavy penalty is added on G_i and the engagement is reported as
       visibility-stressed.
    6. Soft-min update -> nominal sequence is updated; return first command.

The visibility CBF shield from v1 is REMOVED; the planner alone is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import (
    CostParams,
    E3,
    MPPIParams,
    SimConfig,
    cos_theta_F,
    method_weights,
)
from .dynamics import (
    DefenderState,
    rollout_defender,
    rollout_intruder_batch,
)
from .risk import mean_plus_cvar_batch
from .so3 import integrate_R
from .visibility import EPS_RHO


# --------------------------------------------------------------------------- #
# Sampling / clipping
# --------------------------------------------------------------------------- #


def sample_defender_perturbations(
    rng: np.random.Generator,
    K: int, N: int,
    eps_sigma_std: float, eps_Omega_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (eps_sigma, eps_Omega) with shapes (K, N) and (K, N, 3).

    Sample 0 is reserved for the noiseless nominal (zero perturbation).
    """
    eps_sigma = rng.normal(0.0, eps_sigma_std, size=(K, N))
    eps_Omega = rng.normal(0.0, eps_Omega_std, size=(K, N, 3))
    eps_sigma[0, :] = 0.0
    eps_Omega[0, :, :] = 0.0
    return eps_sigma, eps_Omega


def clip_inputs(sigma_seq: np.ndarray, Omega_seq: np.ndarray,
                sigma_max: float, Omega_max: float
                ) -> tuple[np.ndarray, np.ndarray]:
    sigma_clip = np.clip(sigma_seq, 0.0, sigma_max)
    Omega_clip = np.clip(Omega_seq, -Omega_max, Omega_max)
    return sigma_clip, Omega_clip


# --------------------------------------------------------------------------- #
# Vectorised defender rollouts
# --------------------------------------------------------------------------- #


def rollout_defender_batch(
    p0: np.ndarray, v0: np.ndarray, R0: np.ndarray,
    sigma_batch: np.ndarray,   # (K, N)
    Omega_batch: np.ndarray,   # (K, N, 3)
    dt: float, g: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns p_seq (K, N+1, 3), v_seq (K, N+1, 3), R_seq (K, N+1, 3, 3),
    b_seq (K, N+1, 3) -- the world-frame body-z = R e_3."""
    K, N = sigma_batch.shape
    p = np.zeros((K, N + 1, 3))
    v = np.zeros((K, N + 1, 3))
    R = np.zeros((K, N + 1, 3, 3))
    b = np.zeros((K, N + 1, 3))
    p[:, 0, :] = p0
    v[:, 0, :] = v0
    R[:, 0, :, :] = R0
    b[:, 0, :] = R0 @ E3
    for k in range(N):
        sig = np.clip(sigma_batch[:, k], 0.0, None)[:, None]
        a_world = -g * E3 + sig * b[:, k, :]
        v[:, k + 1, :] = v[:, k, :] + dt * a_world
        p[:, k + 1, :] = p[:, k, :] + dt * v[:, k + 1, :]
        Om = Omega_batch[:, k, :]
        theta = np.linalg.norm(Om * dt, axis=1)
        small = theta < 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            a_coef = np.where(small,
                              1.0 - theta * theta / 6.0,
                              np.sin(theta) / np.maximum(theta, 1e-12))
            b_coef = np.where(small,
                              0.5 - theta * theta / 24.0,
                              (1.0 - np.cos(theta)) / np.maximum(theta * theta, 1e-12))
        w_dt = Om * dt
        K_skew = np.zeros((K, 3, 3))
        K_skew[:, 0, 1] = -w_dt[:, 2]
        K_skew[:, 0, 2] =  w_dt[:, 1]
        K_skew[:, 1, 0] =  w_dt[:, 2]
        K_skew[:, 1, 2] = -w_dt[:, 0]
        K_skew[:, 2, 0] = -w_dt[:, 1]
        K_skew[:, 2, 1] =  w_dt[:, 0]
        K_skew_sq = np.einsum("kij,kjl->kil", K_skew, K_skew)
        exp_So3 = (np.eye(3)[None, :, :]
                   + a_coef[:, None, None] * K_skew
                   + b_coef[:, None, None] * K_skew_sq)
        R[:, k + 1, :, :] = np.einsum("kij,kjl->kil", R[:, k, :, :], exp_So3)
        b[:, k + 1, :] = R[:, k + 1, :, :] @ E3
    return p, v, R, b


# --------------------------------------------------------------------------- #
# Visibility tensor (h_V, mu_V, eta_V) over the (K, M, ...) tensor
# --------------------------------------------------------------------------- #


def visibility_tensor(
    p_D: np.ndarray,            # (K, N+1, 3)
    v_D: np.ndarray,            # (K, N+1, 3)
    R_D: np.ndarray,            # (K, N+1, 3, 3)
    Omega_seq: np.ndarray,      # (K, N, 3)
    p_A: np.ndarray,            # (M, N+1, 3)
    v_A: np.ndarray,            # (M, N+1, 3)
    b_c_world_seq: np.ndarray,  # (K, N+1, 3) = R_D b_c
    cfg: SimConfig,
) -> dict:
    """Compute h_V, mu_V, eta_V tensors over (K, M, N+1) / (K, M, N).

    Returns dict with rho (K,M,N+1), h_V (K,M,N+1), mu_V (K,M,N+1),
    eta_V (K,M,N) -- eta_V uses Omega_seq[k] over k=0..N-1.
    """
    K, Np1, _ = p_D.shape
    M = p_A.shape[0]
    N = Np1 - 1

    p_D_b = p_D[:, None, :, :]
    v_D_b = v_D[:, None, :, :]
    p_A_b = p_A[None, :, :, :]
    v_A_b = v_A[None, :, :, :]
    b_c_b = b_c_world_seq[:, None, :, :]

    r = p_A_b - p_D_b
    rho = np.linalg.norm(r, axis=-1)
    rho_safe = np.maximum(rho, EPS_RHO)
    r_hat = r / rho_safe[..., None]
    v_rel = v_A_b - v_D_b

    ctf = cos_theta_F(cfg)
    h_V_all = np.sum(b_c_b * r_hat, axis=-1) - ctf

    bv = np.sum(b_c_b * v_rel, axis=-1)
    br = np.sum(b_c_b * r_hat, axis=-1)
    rv = np.sum(r_hat * v_rel, axis=-1)
    c_v_all = (bv - br * rv) / rho_safe

    # c_Omega in BODY frame: c_Omega = R_D^T (b_D x r_hat) (Eq. 9)
    cross_world = np.cross(
        np.broadcast_to(b_c_b, (K, M, Np1, 3)),
        r_hat
    )
    R_T = np.swapaxes(R_D, -1, -2)
    R_T_b = R_T[:, None, :, :, :]
    c_Omega_body = np.einsum(
        "...ij,...j->...i",
        np.broadcast_to(R_T_b, (K, M, Np1, 3, 3)),
        cross_world,
    )                                                     # (K, M, N+1, 3)
    c_Omega_l1 = np.sum(np.abs(c_Omega_body), axis=-1)    # (K, M, N+1)

    Omega_max = float(cfg.defender.Omega_max)
    alpha_V = float(cfg.visibility.alpha_V)
    zeta_V = float(cfg.visibility.zeta_V)

    # mu_V (Eq. 13): support-function over the body-rate box.
    mu_V_all = Omega_max * c_Omega_l1 + c_v_all + alpha_V * h_V_all - zeta_V

    # eta_V (Eq. 14) at planning step k uses Omega_seq[k]; broadcast across M.
    # c_Omega_body has (K, M, N+1, 3); Omega_seq has (K, N, 3) -- pair them
    # at k = 0..N-1.
    co_kt = c_Omega_body[:, :, :N, :]                     # (K, M, N, 3)
    Om_kt = Omega_seq[:, None, :, :]                      # (K, 1, N, 3)
    eta_dot_term = np.sum(co_kt * Om_kt, axis=-1)         # (K, M, N)
    eta_V_all = (eta_dot_term + c_v_all[:, :, :N]
                 + alpha_V * h_V_all[:, :, :N] - zeta_V)
    return {
        "rho": rho,
        "h_V": h_V_all,
        "mu_V": mu_V_all,
        "eta_V": eta_V_all,
    }


# --------------------------------------------------------------------------- #
# Pairwise cost J_{im} (Eq. 24) and event penalty (Eq. 23)
# --------------------------------------------------------------------------- #


def rollout_costs_and_gate(
    p_D: np.ndarray,            # (K, N+1, 3)
    v_D: np.ndarray,            # (K, N+1, 3)
    R_D: np.ndarray,            # (K, N+1, 3, 3)
    sigma_seq: np.ndarray,      # (K, N)
    Omega_seq: np.ndarray,      # (K, N, 3)
    p_A: np.ndarray,            # (M, N+1, 3)
    v_A: np.ndarray,            # (M, N+1, 3)
    b_c_world_seq: np.ndarray,  # (K, N+1, 3) = R_D b_c
    cfg: SimConfig,
    weights: dict,              # output of method_weights(method)
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (J (K,M), G (K,), diagnostics).

    diagnostics contains the (K, M, N+1) visibility tensors used both by
    the cost and by the gate.
    """
    K, Np1, _ = p_D.shape
    M = p_A.shape[0]
    N = Np1 - 1
    cost: CostParams = cfg.cost
    mp: MPPIParams = cfg.mppi
    dt = mp.dt

    vis = visibility_tensor(
        p_D, v_D, R_D, Omega_seq, p_A, v_A, b_c_world_seq, cfg,
    )
    rho = vis["rho"]                # (K, M, N+1)
    h_V = vis["h_V"]
    mu_V = vis["mu_V"]
    eta_V = vis["eta_V"]            # (K, M, N)

    # ---- Event times tau_I, tau_B, tau_L over the rollout (Eq. 22) ----
    # tau_I: first k with rho_k <= r_c  (defender intercept)
    # tau_B: first k with attacker inside protected region
    # tau_L: first k with h_V_k < 0    (visual loss)
    # Each set to horizon * dt if event never occurs.
    horizon_t = float(N * dt)
    p_P = np.asarray(cfg.geom.p_P, dtype=float)
    r_P = float(cfg.geom.r_P)
    r_c = float(cfg.geom.r_c)

    # intercept event (K, M)
    intercept_mask = rho <= r_c
    intercept_any = intercept_mask.any(axis=-1)
    intercept_idx = np.where(intercept_any,
                             np.argmax(intercept_mask, axis=-1),
                             Np1).astype(float)
    tau_I = intercept_idx * dt
    tau_I[~intercept_any] = horizon_t

    # breach event (M,)
    dist_aP = np.linalg.norm(p_A - p_P, axis=-1)   # (M, N+1)
    breach_mask = dist_aP <= r_P
    breach_any = breach_mask.any(axis=-1)
    breach_idx = np.where(breach_any,
                          np.argmax(breach_mask, axis=-1),
                          Np1).astype(float)
    tau_B_m = breach_idx * dt
    tau_B_m[~breach_any] = horizon_t
    tau_B = np.broadcast_to(tau_B_m[None, :], (K, M))

    # visual-loss event (K, M)
    loss_mask = h_V < 0.0
    loss_any = loss_mask.any(axis=-1)
    loss_idx = np.where(loss_any,
                        np.argmax(loss_mask, axis=-1),
                        Np1).astype(float)
    tau_L = loss_idx * dt
    tau_L[~loss_any] = horizon_t

    # Event penalty (Eq. 23): only counts when intercept happens AFTER an
    # adverse event.
    ev_breach = np.maximum(tau_I - tau_B, 0.0) ** 2
    ev_loss = np.maximum(tau_I - tau_L, 0.0) ** 2
    l_ev = cost.q_B * ev_breach + cost.q_L * weights["q_L_mul"] * ev_loss

    # ---- Stage costs (Eq. 24) ----
    rho_stage = rho[:, :, :-1]                                      # (K,M,N)
    hv_stage = h_V[:, :, :-1]
    mv_stage = mu_V[:, :, :-1]
    eta_stage = eta_V                                                # (K,M,N)

    h_safe = cfg.visibility.h_safe
    mu_safe = cfg.visibility.mu_safe
    eta_safe = cfg.visibility.eta_safe

    stage = (
        cost.q_rho * rho_stage ** 2
        + cost.q_V * weights["q_V_mul"] * np.maximum(h_safe - hv_stage, 0.0) ** 2
        + cost.q_mu * weights["q_mu_mul"] * np.maximum(mu_safe - mv_stage, 0.0) ** 2
        + cost.q_eta * weights["q_eta_mul"] * np.maximum(eta_safe - eta_stage, 0.0) ** 2
    )

    omega_norm_sq = np.sum(Omega_seq ** 2, axis=-1)
    sigma_hov = cfg.sim.g
    sigma_dev = (sigma_seq - sigma_hov) ** 2
    control_cost = cost.q_Omega * omega_norm_sq + cost.q_sigma * sigma_dev
    stage = stage + control_cost[:, None, :]

    J = stage.sum(axis=-1)
    rho_N = rho[:, :, -1]
    J = J + cost.q_f * np.maximum(rho_N - r_c, 0.0) ** 2 + l_ev

    # ---- Rollout-level gate G_i (Eq. 27) ----
    # Per (i, m, k) violation amounts (only count active terms for the method)
    if weights["use_gate"]:
        violations = []
        if weights["q_V_mul"] > 0:
            violations.append(np.maximum(h_safe - h_V, 0.0))
        if weights["q_mu_mul"] > 0:
            violations.append(np.maximum(mu_safe - mu_V, 0.0))
        if weights["q_eta_mul"] > 0:
            # extend eta_V from (K,M,N) to (K,M,N+1) via repeat
            eta_pad = np.concatenate([eta_V, eta_V[:, :, -1:]], axis=-1)
            violations.append(np.maximum(eta_safe - eta_pad, 0.0))
        if violations:
            G_kmn = np.maximum.reduce(violations)            # (K, M, N+1)
            G_i = G_kmn.reshape(K, -1).max(axis=-1)          # (K,)
        else:
            G_i = np.zeros(K)
    else:
        G_i = np.zeros(K)

    return J, G_i, dict(
        rho=rho, h_V=h_V, mu_V=mu_V, eta_V=eta_V,
        tau_I=tau_I, tau_B=tau_B, tau_L=tau_L,
    )


# --------------------------------------------------------------------------- #
# Soft-min update with gate
# --------------------------------------------------------------------------- #


def softmin_update(
    risk_scores: np.ndarray,   # (K,)
    G_i: np.ndarray,           # (K,)
    eps_sigma: np.ndarray,     # (K, N)
    eps_Omega: np.ndarray,     # (K, N, 3)
    nominal_sigma: np.ndarray,
    nominal_Omega: np.ndarray,
    sigma_max: float, Omega_max: float,
    lam: float, exp_clip: float,
    gate_tol: float,
    gate_min_keep: int,
    gate_penalty_weight: float,
    use_gate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Returns (sigma_new, Omega_new, weights, stressed).

    `stressed` is True if the gate could not find at least gate_min_keep
    zero-violation samples; the planner then keeps all samples and adds
    a penalty.
    """
    K = risk_scores.shape[0]
    stressed = False
    if use_gate:
        clean = G_i <= gate_tol
        n_clean = int(clean.sum())
        if n_clean >= gate_min_keep:
            # Restrict to clean samples by setting other scores to +inf
            risk_used = np.where(clean, risk_scores, np.inf)
        else:
            # No clean samples available; engagement is visibility-stressed.
            stressed = True
            risk_used = risk_scores + gate_penalty_weight * (G_i ** 2)
    else:
        risk_used = risk_scores

    finite = np.isfinite(risk_used)
    if not finite.any():
        # Degenerate (shouldn't happen): fall back to uniform.
        weights_arr = np.full(K, 1.0 / K)
    else:
        R_min = float(np.min(risk_used[finite]))
        exponent = -(risk_used - R_min) / max(lam, 1e-6)
        exponent = np.where(finite, exponent, -np.inf)
        exponent = np.maximum(exponent, exp_clip)
        weights_arr = np.exp(exponent)
        w_sum = float(weights_arr.sum())
        if w_sum < 1e-12:
            weights_arr = np.full(K, 1.0 / K)
        else:
            weights_arr = weights_arr / w_sum

    update_sigma = np.einsum("k,kn->n", weights_arr, eps_sigma)
    update_Omega = np.einsum("k,knj->nj", weights_arr, eps_Omega)
    sigma_new = np.clip(nominal_sigma + update_sigma, 0.0, sigma_max)
    Omega_new = np.clip(nominal_Omega + update_Omega, -Omega_max, Omega_max)
    return sigma_new, Omega_new, weights_arr, stressed


# --------------------------------------------------------------------------- #
# Top-level planner
# --------------------------------------------------------------------------- #


@dataclass
class PlannerState:
    """Persistent state of the MPPI planner across replans."""
    nominal_sigma: np.ndarray
    nominal_Omega: np.ndarray

    @classmethod
    def from_config(cls, cfg: SimConfig) -> "PlannerState":
        N = cfg.mppi.N
        # Bias the initial nominal thrust toward "aggressive closure" so the
        # MPPI sampling cone covers useful magnitudes from the first replan.
        # Hovering is a poor seed for a terminal-interception problem -- the
        # planner would need many softmin updates to ramp up.
        sig0 = 0.5 * (cfg.sim.g + cfg.defender.sigma_max)
        return cls(
            nominal_sigma=np.full(N, sig0),
            nominal_Omega=np.zeros((N, 3)),
        )


@dataclass
class PlanDebug:
    """Snapshot of internal planning quantities at one replan instant."""
    eps_sigma: np.ndarray
    eps_Omega: np.ndarray
    sigma_samples: np.ndarray
    Omega_samples: np.ndarray
    p_D: np.ndarray              # (K, N+1, 3)
    p_A: np.ndarray              # (M, N+1, 3)
    risk_scores: np.ndarray      # (K,)
    weights: np.ndarray          # (K,)
    a_seq_batch: np.ndarray      # (M, N, 3)
    G_i: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stressed: bool = False


def plan(
    cfg: SimConfig,
    state: PlannerState,
    defender: DefenderState,
    p_A_hat: np.ndarray, v_A_hat: np.ndarray,
    R_A_hat: np.ndarray, a_A_hat: np.ndarray,
    rng: np.random.Generator,
    collect_debug: bool = False,
) -> tuple[float, np.ndarray, PlannerState, PlanDebug | None, bool]:
    """Run one MPPI replan.  Returns
    (sigma_0, Omega_0, updated_state, debug, stressed).

    `stressed` is True if the rollout gate was active but no zero-violation
    samples existed -- the engagement is in a visibility-infeasible
    geometry.
    """
    from .tube import build_intruder_tube
    mp: MPPIParams = cfg.mppi
    K, M, N, dt = mp.K, mp.M, mp.N, mp.dt
    g = cfg.sim.g

    weights = method_weights(cfg.scenario.planner_method)

    a_seq_batch, _, _ = build_intruder_tube(
        cfg, p_A_hat, v_A_hat, R_A_hat, a_A_hat, defender.p, rng
    )
    p_A_seq, v_A_seq = rollout_intruder_batch(p_A_hat, v_A_hat, a_seq_batch, dt)

    eps_sigma, eps_Omega = sample_defender_perturbations(
        rng, K, N, mp.eps_sigma_std, mp.eps_Omega_std
    )

    # Optional PN-style warmstart for the nominal trajectory.  When
    # active, the centre of the MPPI sample distribution is biased
    # toward a reactive PN tracking law evaluated at the CURRENT state.
    # MPPI then explores refinements around this known-good baseline.
    # This is critical for the visibility-aware MPPI methods to inherit
    # PN's tight body-on-LOS tracking AND add anticipation.
    nominal_sigma_warm = state.nominal_sigma.copy()
    nominal_Omega_warm = state.nominal_Omega.copy()
    if getattr(mp, "pn_warmstart_nominal", False):
        b_c = np.asarray(cfg.defender.b_c, dtype=float)
        b_c_unit = b_c / max(float(np.linalg.norm(b_c)), 1e-9)
        b_D = defender.R @ b_c_unit
        to_int = p_A_hat - defender.p
        n_los = float(np.linalg.norm(to_int))
        if n_los > 1e-6:
            los = to_int / n_los
            axis_world = np.cross(b_D, los)
            gain = float(getattr(mp, "pn_warmstart_gain", 5.0))
            Omega_pn_body = defender.R.T @ (gain * axis_world)
            Omega_pn_body = np.clip(Omega_pn_body,
                                    -cfg.defender.Omega_max,
                                    cfg.defender.Omega_max)
            # Decay the PN feed-forward across the horizon.  Longer
            # decay (~0.5 s) means MPPI's sample distribution stays
            # near the PN tracking law for most of the rollout, which
            # is essential for high strict P_succ.
            decay = np.exp(-np.arange(N) * dt / 0.50)
            nominal_Omega_warm = (decay[:, None] * Omega_pn_body[None, :]
                                  + (1.0 - decay[:, None])
                                  * state.nominal_Omega)
            sig_warm = (cfg.defender.sigma_max * 0.85
                        * decay + (1.0 - decay) * state.nominal_sigma)
            nominal_sigma_warm = sig_warm

    sigma_samples = nominal_sigma_warm[None, :] + eps_sigma
    Omega_samples = nominal_Omega_warm[None, :, :] + eps_Omega
    sigma_samples, Omega_samples = clip_inputs(
        sigma_samples, Omega_samples,
        cfg.defender.sigma_max, cfg.defender.Omega_max,
    )

    p_D_seq, v_D_seq, R_D_seq, _ = rollout_defender_batch(
        defender.p, defender.v, defender.R,
        sigma_samples, Omega_samples, dt, g,
    )
    b_c = np.asarray(cfg.defender.b_c, dtype=float)
    b_c_world_seq = np.einsum("knij,j->kni", R_D_seq, b_c)

    J, G_i, _diag = rollout_costs_and_gate(
        p_D_seq, v_D_seq, R_D_seq, sigma_samples, Omega_samples,
        p_A_seq, v_A_seq, b_c_world_seq, cfg, weights,
    )
    # Risk aggregation: with full method, use mean+CVaR; else plain mean.
    if weights["use_cvar"]:
        risk = mean_plus_cvar_batch(J, mp.kappa, mp.alpha_R)
    else:
        risk = J.mean(axis=1)

    nominal_sigma_new, nominal_Omega_new, weights_arr, stressed = softmin_update(
        risk, G_i, eps_sigma, eps_Omega,
        nominal_sigma_warm, nominal_Omega_warm,
        cfg.defender.sigma_max, cfg.defender.Omega_max,
        mp.lam, mp.exp_clip,
        mp.gate_tol, mp.gate_min_keep, mp.gate_penalty_weight,
        weights["use_gate"],
    )

    sigma_0 = float(nominal_sigma_new[0])
    Omega_0 = nominal_Omega_new[0].copy()
    new_state = PlannerState(
        nominal_sigma=np.concatenate([nominal_sigma_new[1:], [cfg.sim.g]]),
        nominal_Omega=np.vstack([nominal_Omega_new[1:], np.zeros((1, 3))]),
    )

    debug = None
    if collect_debug:
        debug = PlanDebug(
            eps_sigma=eps_sigma,
            eps_Omega=eps_Omega,
            sigma_samples=sigma_samples,
            Omega_samples=Omega_samples,
            p_D=p_D_seq,
            p_A=p_A_seq,
            risk_scores=risk,
            weights=weights_arr,
            a_seq_batch=a_seq_batch,
            G_i=G_i,
            stressed=stressed,
        )
    return sigma_0, Omega_0, new_state, debug, stressed
