"""Visibility geometry: h_V, c_Omega, c_v, mu_V, eta_V (Eqs. 5-14 of v2).

    h_V(x)     = b_D^T r_hat - cos(theta_F)                                (5)
    h_V_dot    = c_Omega(x)^T Omega_D + c_v(x)                             (8)
    c_Omega    = R_D^T (b_D x r_hat)                                       (9)
    c_v        = b_D^T (I - r_hat r_hat^T) v / rho                         (9)
    mu_V(x)    = Omega_max ||c_Omega||_1 + c_v + alpha_V h_V - zeta_V     (13)
    eta_V(x,u) = c_Omega^T Omega_D + c_v + alpha_V h_V - zeta_V           (14)

mu_V >= 0 means the visibility-rate inequality is feasible under the body-rate
box.  eta_V >= 0 means the *sampled* command satisfies that inequality.
"""

from __future__ import annotations

import numpy as np

from .config import (
    SimConfig,
    VisibilityParams,
    cos_theta_F,
)

EPS_RHO = 1e-3


def boresight_world(R_D: np.ndarray, b_c: np.ndarray) -> np.ndarray:
    return np.asarray(R_D, dtype=float) @ np.asarray(b_c, dtype=float)


def los(p_D: np.ndarray, p_A: np.ndarray
        ) -> tuple[np.ndarray, float, np.ndarray]:
    r = np.asarray(p_A, dtype=float) - np.asarray(p_D, dtype=float)
    rho = float(np.linalg.norm(r))
    if rho < EPS_RHO:
        return r, rho, np.array([1.0, 0.0, 0.0])
    return r, rho, r / rho


def h_V(R_D: np.ndarray, b_c: np.ndarray, p_D: np.ndarray, p_A: np.ndarray,
        cos_thetaF: float) -> float:
    b_D = boresight_world(R_D, b_c)
    _, _, r_hat = los(p_D, p_A)
    return float(np.dot(b_D, r_hat) - cos_thetaF)


def c_Omega(R_D: np.ndarray, b_c: np.ndarray, p_D: np.ndarray,
            p_A: np.ndarray) -> np.ndarray:
    b_D = boresight_world(R_D, b_c)
    _, _, r_hat = los(p_D, p_A)
    return np.asarray(R_D, dtype=float).T @ np.cross(b_D, r_hat)


def c_v(R_D: np.ndarray, b_c: np.ndarray, p_D: np.ndarray, p_A: np.ndarray,
        v_D: np.ndarray, v_A: np.ndarray) -> float:
    b_D = boresight_world(R_D, b_c)
    _, rho, r_hat = los(p_D, p_A)
    v = np.asarray(v_A, dtype=float) - np.asarray(v_D, dtype=float)
    P = np.eye(3) - np.outer(r_hat, r_hat)
    return float(np.dot(b_D, P @ v) / max(rho, EPS_RHO))


def mu_V(R_D: np.ndarray, b_c: np.ndarray, p_D: np.ndarray, p_A: np.ndarray,
         v_D: np.ndarray, v_A: np.ndarray, vp: VisibilityParams,
         Omega_max: float, cos_thetaF: float) -> float:
    co = c_Omega(R_D, b_c, p_D, p_A)
    cv = c_v(R_D, b_c, p_D, p_A, v_D, v_A)
    hv = h_V(R_D, b_c, p_D, p_A, cos_thetaF)
    return float(Omega_max * np.sum(np.abs(co)) + cv
                 + vp.alpha_V * hv - vp.zeta_V)


def eta_V(R_D: np.ndarray, b_c: np.ndarray, p_D: np.ndarray, p_A: np.ndarray,
          v_D: np.ndarray, v_A: np.ndarray, Omega: np.ndarray,
          vp: VisibilityParams, cos_thetaF: float) -> float:
    """Eq. 14: eta_V = c_Omega^T Omega_D + c_v + alpha_V h_V - zeta_V."""
    co = c_Omega(R_D, b_c, p_D, p_A)
    cv = c_v(R_D, b_c, p_D, p_A, v_D, v_A)
    hv = h_V(R_D, b_c, p_D, p_A, cos_thetaF)
    Omega = np.asarray(Omega, dtype=float).reshape(3)
    return float(np.dot(co, Omega) + cv + vp.alpha_V * hv - vp.zeta_V)


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #


def evaluate_visibility(cfg: SimConfig, R_D: np.ndarray, p_D: np.ndarray,
                        v_D: np.ndarray, p_A: np.ndarray, v_A: np.ndarray
                        ) -> dict:
    """Compute h_V, c_Omega, c_v, mu_V at the current state."""
    b_c = np.asarray(cfg.defender.b_c, dtype=float)
    Omega_max = float(cfg.defender.Omega_max)
    ctf = cos_theta_F(cfg)
    co = c_Omega(R_D, b_c, p_D, p_A)
    cv = c_v(R_D, b_c, p_D, p_A, v_D, v_A)
    hv = h_V(R_D, b_c, p_D, p_A, ctf)
    mv = float(Omega_max * np.sum(np.abs(co)) + cv
               + cfg.visibility.alpha_V * hv - cfg.visibility.zeta_V)
    return {"h_V": hv, "c_Omega": co, "c_v": cv, "mu_V": mv}


def evaluate_eta_V(cfg: SimConfig, R_D: np.ndarray, p_D: np.ndarray,
                   v_D: np.ndarray, p_A: np.ndarray, v_A: np.ndarray,
                   Omega: np.ndarray) -> float:
    """eta_V for the actually-applied body rate Omega (Eq. 14)."""
    return eta_V(
        R_D, np.asarray(cfg.defender.b_c, dtype=float),
        p_D, p_A, v_D, v_A, Omega,
        cfg.visibility, cos_theta_F(cfg),
    )
