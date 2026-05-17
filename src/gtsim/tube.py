"""Estimator-driven intruder prediction tube (Eqs. 11-16).

Nominal thrust axis      d_A_bar = R_A_hat e_3                       (Eq. 11)
Attitude-consistent scalar
    sigma_A_bar = clip_{[0, sigma_A^max]}( (a_hat_A + g e_3)^T d_A_bar )   (12)
    a_A_bar    = -g e_3 + sigma_A_bar d_A_bar                              (12)
Residual model
    a_{A,k} = a_A_bar + w_{A,k}                                            (13)
Tube constraint
    |a_{A,k}|       <= a_A^max                                             (14)
    (a-a_bar)^T Sigma^-1 (a-a_bar) <= chi_a^2                              (14)
    Sigma = Sigma^est + sigma_perp^2 (I - r_bar r_bar^T)
                       + sigma_par^2 r_bar r_bar^T                         (15)

Implementation refinement (audit item #1, plan file): the projection in
Eq. 12 silently discards the component of a_hat_A orthogonal to d_A_bar.
We compute that orthogonal residual explicitly and inflate Sigma^est at
step k=0 by ||w_perp||^2 I so the sampled tube actually reflects the
estimator disagreement at t.
"""

from __future__ import annotations

import numpy as np

from .config import E3, IntruderParams, SimConfig, TubeParams


def nominal_thrust_axis(R_A_hat: np.ndarray) -> np.ndarray:
    """Eq. 11: d_A_bar = R_A_hat e_3."""
    return np.asarray(R_A_hat, dtype=float) @ E3


def attitude_consistent_accel(
    a_hat_A: np.ndarray,
    R_A_hat: np.ndarray,
    g: float,
    sigma_A_max: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Eq. 12: returns (sigma_A_bar, a_A_bar, w_perp).

    w_perp is the part of (a_hat_A + g e_3) orthogonal to d_A_bar -- the
    component the projection throws away.  Caller can fold this into
    Sigma^est to recover the discarded information (plan audit item #1).
    """
    a_hat_A = np.asarray(a_hat_A, dtype=float).reshape(3)
    d = nominal_thrust_axis(R_A_hat)
    f = a_hat_A + g * E3
    scalar = float(np.dot(f, d))
    sigma_bar = float(np.clip(scalar, 0.0, sigma_A_max))
    a_bar = -g * E3 + sigma_bar * d
    # Orthogonal residual: f - scalar * d (regardless of clipping).
    w_perp = f - scalar * d
    return sigma_bar, a_bar, w_perp


def tube_covariance(
    Sigma_est: np.ndarray,
    r_bar: np.ndarray,
    tp: TubeParams,
) -> np.ndarray:
    """Eq. 15 covariance shaping with sigma_perp >= sigma_par >= 0."""
    r = np.asarray(r_bar, dtype=float).reshape(3)
    nr = float(np.linalg.norm(r))
    if nr < 1e-9:
        rh = np.array([1.0, 0.0, 0.0])
    else:
        rh = r / nr
    P_perp = np.eye(3) - np.outer(rh, rh)
    P_par = np.outer(rh, rh)
    return Sigma_est + (tp.sigma_perp ** 2) * P_perp + (tp.sigma_par ** 2) * P_par


def propagate_sigma_est(
    Sigma0: np.ndarray,
    k: int,
    dt: float,
    Q_white: float,
) -> np.ndarray:
    """Sigma^est_{k|t} = Sigma0 + k * dt * Q I.

    Simple constant-rate growth used as a placeholder until a real
    Kalman filter is wired in.  Matches the "estimator-propagated"
    wording in the paper (Eq. 15) without committing to a specific
    estimator.
    """
    return Sigma0 + (k * dt * Q_white) * np.eye(3)


def initial_sigma_est(
    tp: TubeParams,
    w_perp: np.ndarray,
) -> np.ndarray:
    """Sigma^est_{0|t} = sigma_est_floor I + ||w_perp||^2 I.

    Folds the projection residual back into the prior (plan audit item #1).
    """
    base = tp.sigma_est_floor * np.eye(3)
    w_perp = np.asarray(w_perp, dtype=float).reshape(3)
    inflation = float(np.dot(w_perp, w_perp)) * np.eye(3)
    return base + inflation


def sample_accelerations(
    a_bar: np.ndarray,
    Sigma_seq: list[np.ndarray],     # length N, each (3, 3)
    a_A_max: float,
    chi_a: float,
    M: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw M acceleration sequences from the tube (Eqs. 13-14).

    Returns shape (M, N, 3).  Implementation: for each step k, draw a
    Gaussian deviation with covariance Sigma_k, scale to fit the
    Mahalanobis bound and the |a|<=a_A_max ball.  The first scenario is
    the nominal (zero residual), and the next 2*N scenarios are sigma-
    point style deviations along the principal axes of Sigma_k (these
    explicitly cover the lateral-juke direction selected by Eq. 15).

    Remaining M - (1 + 2N) samples are random Gaussian draws -- this is
    the "residual-acceleration rollouts biased toward transverse LOS
    motion" mentioned just after Eq. 18.
    """
    a_bar = np.asarray(a_bar, dtype=float).reshape(3)
    N = len(Sigma_seq)
    if M < 1:
        raise ValueError("M must be >= 1")
    out = np.zeros((M, N, 3))
    # Scenario 0: nominal.
    out[0, :, :] = a_bar
    idx = 1
    # Sigma-point scenarios: +-chi_a * sqrt(lambda_j) v_j along each
    # eigenvector v_j of Sigma_k.  We build a *single* scenario per
    # sign-and-axis and replicate the same deviation across all k.
    # This is a coarse approximation -- a more faithful one would
    # generate one sigma point per (k, axis, sign), but with N=20 and
    # 3 axes that explodes M.  Keep it light: per axis-and-sign, use
    # eigendecomposition at the *first* step.
    Sigma0 = Sigma_seq[0]
    eigvals, eigvecs = np.linalg.eigh(Sigma0)
    eigvals = np.clip(eigvals, 0.0, None)
    for j in range(3):
        for sign in (+1.0, -1.0):
            if idx >= M:
                break
            dev = sign * chi_a * float(np.sqrt(eigvals[j])) * eigvecs[:, j]
            # Hold constant across the horizon (worst-case persistent
            # deviation, the most damaging to the predictor).
            a_seq = np.tile(a_bar + dev, (N, 1))
            # Project each step into |a| <= a_A_max
            mags = np.linalg.norm(a_seq, axis=1, keepdims=True)
            scale = np.minimum(1.0, a_A_max / np.maximum(mags, 1e-12))
            out[idx, :, :] = a_seq * scale
            idx += 1
        if idx >= M:
            break
    # Remaining: random samples per step, projected onto |a|<=a_A_max
    # and onto the Mahalanobis chi_a^2 ball using rejection-by-scaling.
    while idx < M:
        a_seq = np.zeros((N, 3))
        for k in range(N):
            Lk = np.linalg.cholesky(Sigma_seq[k] + 1e-9 * np.eye(3))
            z = rng.standard_normal(3)
            dev = Lk @ z
            # Scale to fit chi_a Mahalanobis ball.
            md = float(np.sqrt(max(0.0, z @ z)))
            if md > chi_a and md > 1e-9:
                dev *= (chi_a / md)
            a = a_bar + dev
            n = float(np.linalg.norm(a))
            if n > a_A_max and n > 1e-9:
                a *= (a_A_max / n)
            a_seq[k] = a
        out[idx, :, :] = a_seq
        idx += 1
    return out


def build_intruder_tube(
    cfg: SimConfig,
    p_A_hat: np.ndarray,
    v_A_hat: np.ndarray,
    R_A_hat: np.ndarray,
    a_A_hat: np.ndarray,
    p_D: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-level helper. Returns (a_seq_batch, sigma_A_bar, a_A_bar).

    a_seq_batch has shape (M, N, 3) ready to be fed to
    `dynamics.rollout_intruder_batch`.
    """
    g = cfg.sim.g
    ip: IntruderParams = cfg.intruder
    tp: TubeParams = cfg.tube
    sigma_bar, a_bar, w_perp = attitude_consistent_accel(
        a_A_hat, R_A_hat, g, ip.sigma_max
    )
    # Build per-step covariances using the propagated estimator covariance
    # and the LOS shaping (Eq. 15).  Nominal LOS direction is taken from
    # the nominal rollout starting at (p_A_hat, v_A_hat) under a_bar; if
    # at some step the LOS becomes degenerate, we reuse the last good one.
    Sigma0_est = initial_sigma_est(tp, w_perp)
    Sigma_seq: list[np.ndarray] = []
    dt = cfg.mppi.dt
    p_k = p_A_hat.copy()
    v_k = v_A_hat.copy()
    last_good_r = p_k - p_D
    if np.linalg.norm(last_good_r) < 1e-6:
        last_good_r = np.array([1.0, 0.0, 0.0])
    for k in range(cfg.mppi.N):
        Sigma_est_k = propagate_sigma_est(Sigma0_est, k, dt, tp.Q_white)
        r_k = p_k - p_D
        nr = float(np.linalg.norm(r_k))
        if nr < 1e-6:
            r_bar_k = last_good_r.copy()
        else:
            r_bar_k = r_k.copy()
            last_good_r = r_k.copy()
        Sigma_seq.append(tube_covariance(Sigma_est_k, r_bar_k, tp))
        # advance nominal rollout
        v_k = v_k + dt * a_bar
        p_k = p_k + dt * v_k
    a_seq_batch = sample_accelerations(
        a_bar, Sigma_seq, ip.a_max, tp.chi_a, cfg.mppi.M, rng
    )
    return a_seq_batch, sigma_bar, a_bar
