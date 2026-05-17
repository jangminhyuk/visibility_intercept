"""SO(3) utilities: skew, Rodrigues exp/log, projection back to SO(3).

The defender attitude integrates as R_{k+1} = R_k * exp(S(Omega_k) dt) where
Omega is body-frame body rate (Eq. 1 of the paper).  The Rodrigues form
gives a closed-form matrix exponential of a skew-symmetric matrix:

    exp(S(w)) = I + (sin t / t) S(w) + ((1 - cos t) / t^2) S(w)^2
    with t = |w|.

We use a Taylor expansion when t is small to avoid 0/0.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def skew(w: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix S(w) with S(a) b = a x b."""
    w = np.asarray(w, dtype=float).reshape(3)
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def vee(W: np.ndarray) -> np.ndarray:
    """Inverse of `skew`: extract the 3-vector from a skew-symmetric matrix."""
    W = np.asarray(W, dtype=float)
    return 0.5 * np.array([W[2, 1] - W[1, 2],
                           W[0, 2] - W[2, 0],
                           W[1, 0] - W[0, 1]])


def exp_so3(w: np.ndarray) -> np.ndarray:
    """Rodrigues matrix exponential exp(S(w)) -> 3x3 rotation matrix.

    Uses a 4th-order Taylor expansion for small |w| to stay accurate near 0.
    """
    w = np.asarray(w, dtype=float).reshape(3)
    theta = float(np.linalg.norm(w))
    K = skew(w)
    if theta < 1e-6:
        # Taylor: sin t / t ~ 1 - t^2/6 + t^4/120
        #         (1 - cos t)/t^2 ~ 1/2 - t^2/24 + t^4/720
        a = 1.0 - theta * theta / 6.0
        b = 0.5 - theta * theta / 24.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Inverse of exp_so3: returns w in R^3 with exp(S(w)) = R (|w| <= pi)."""
    R = np.asarray(R, dtype=float)
    tr = float(np.clip(np.trace(R), -1.0, 3.0))
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-6:
        # Near identity: log ~ 0.5 (R - R^T) / (1 - theta^2/6)
        return vee(R - R.T)
    if theta > np.pi - 1e-6:
        # Antipodal: extract axis from (R + I)/2 = u u^T
        M = 0.5 * (R + np.eye(3))
        u = np.sqrt(np.maximum(np.diag(M), 0.0))
        # Pick signs by largest diagonal
        i = int(np.argmax(u))
        u_signed = np.zeros(3)
        u_signed[i] = u[i]
        for j in range(3):
            if j == i:
                continue
            u_signed[j] = M[i, j] / max(u[i], 1e-9)
        return float(theta) * u_signed / max(np.linalg.norm(u_signed), 1e-9)
    return (theta / (2.0 * np.sin(theta))) * vee(R - R.T)


def project_so3(M: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix back onto SO(3) via SVD (closest rotation)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def is_rotation(R: np.ndarray, atol: float = 1e-8) -> bool:
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        return False
    return float(np.linalg.det(R)) > 0.0


def integrate_R(R: np.ndarray, Omega: np.ndarray, dt: float) -> np.ndarray:
    """One body-frame SO(3) Euler step: R_{k+1} = R_k * exp(S(Omega) dt)."""
    return R @ exp_so3(Omega * dt)
