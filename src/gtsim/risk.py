"""Sampled-CVaR via the Rockafellar-Uryasev minimization (Eq. 22).

The paper writes

    CVaR_alpha(J) = min_nu [ nu + 1/((1-alpha) M) sum_m max(J_m - nu, 0) ]

with the closed-form minimiser nu* = VaR_alpha = J_(k*) for
k* = ceil(alpha * M).  For non-integer alpha * M the exact value is the
mixed quantile

    CVaR_alpha = 1/((1-alpha) M) * ( (k* - alpha M) J_(k*)
                                     + sum_{j>k*} J_(j) ),

NOT mean(sorted[-ceil((1-alpha)M):]) -- the naive form is biased when
(1-alpha) M is not an integer.  We implement the exact form here.

Also exposed: the convex combination R = (1 - kappa) mean(J) + kappa CVaR(J)
(Eq. 21).
"""

from __future__ import annotations

import numpy as np


def cvar_alpha(J: np.ndarray, alpha: float) -> float:
    """Exact sampled CVaR_alpha via the mixed-quantile form.

    Parameters
    ----------
    J     : (M,) array of scenario costs.
    alpha : tail parameter in (0, 1).  alpha=0 -> mean, alpha->1 -> max.
    """
    J = np.sort(np.asarray(J, dtype=float))
    M = int(J.shape[0])
    if M == 0:
        return 0.0
    if alpha <= 0.0:
        return float(np.mean(J))
    if alpha >= 1.0 - 1e-12:
        return float(J[-1])
    aM = alpha * M
    k_star = int(np.ceil(aM))                # 1-based index for sorted J
    if k_star >= M + 1:
        return float(J[-1])
    weighted_head = (k_star - aM) * J[k_star - 1]   # convert to 0-based
    tail_sum = float(np.sum(J[k_star:]))            # J_{k*+1} ... J_{M}
    return float((weighted_head + tail_sum) / ((1.0 - alpha) * M))


def mean_plus_cvar(J: np.ndarray, kappa: float, alpha: float) -> float:
    """Eq. 21 risk score R = (1 - kappa) mean(J) + kappa CVaR_alpha(J)."""
    J = np.asarray(J, dtype=float)
    if J.size == 0:
        return 0.0
    return float((1.0 - kappa) * np.mean(J) + kappa * cvar_alpha(J, alpha))


def mean_plus_cvar_batch(J: np.ndarray, kappa: float, alpha: float) -> np.ndarray:
    """Vectorised version over a (K, M) matrix of paired costs.

    Returns shape (K,) risk score per defender sample.
    """
    J = np.asarray(J, dtype=float)
    if J.ndim == 1:
        return np.array([mean_plus_cvar(J, kappa, alpha)])
    K, M = J.shape
    mean_term = J.mean(axis=1)
    if kappa <= 0.0:
        return mean_term
    sorted_J = np.sort(J, axis=1)
    if alpha >= 1.0 - 1e-12:
        cvar_term = sorted_J[:, -1]
    elif alpha <= 0.0:
        cvar_term = mean_term
    else:
        aM = alpha * M
        k_star = int(np.ceil(aM))
        # k_star is in 1..M (since 0 < alpha < 1 implies 0 < aM < M)
        if k_star >= M + 1:
            cvar_term = sorted_J[:, -1]
        else:
            weighted_head = (k_star - aM) * sorted_J[:, k_star - 1]
            tail_sum = sorted_J[:, k_star:].sum(axis=1) if k_star < M else 0.0
            cvar_term = (weighted_head + tail_sum) / ((1.0 - alpha) * M)
    return (1.0 - kappa) * mean_term + kappa * cvar_term
