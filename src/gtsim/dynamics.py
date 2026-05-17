"""Vehicle integrators.

Defender (Eq. 1 of the paper):
    p_dot = v
    v_dot = -g e_3 + sigma_D R_D e_3
    R_dot = R_D S(Omega_D)

Intruder rollout (Eq. 16):
    p_{k+1} = p_k + dt v_k
    v_{k+1} = v_k + dt a_k

Both use forward Euler at the planning step.  Re-orthonormalisation of R is
applied every `RE_ORTHO_EVERY` steps to bleed off floating-point drift; in
practice 50-step interval keeps |R^T R - I|_F < 1e-9 on the engagements
this simulator is sized for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import E3
from .so3 import integrate_R, project_so3

RE_ORTHO_EVERY = 50


@dataclass
class DefenderState:
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    # Last applied command -- handy for logging the realised body rate /
    # thrust at the moment of the latest update.
    sigma: float = 0.0
    Omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    step_count: int = 0

    def copy(self) -> "DefenderState":
        return DefenderState(
            p=self.p.copy(),
            v=self.v.copy(),
            R=self.R.copy(),
            sigma=float(self.sigma),
            Omega=self.Omega.copy(),
            step_count=int(self.step_count),
        )


@dataclass
class IntruderState:
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "IntruderState":
        return IntruderState(
            p=self.p.copy(),
            v=self.v.copy(),
            R=self.R.copy(),
            a=self.a.copy(),
        )


def step_defender(
    state: DefenderState,
    sigma: float,
    Omega: np.ndarray,
    dt: float,
    g: float,
) -> DefenderState:
    """Integrate the defender for one step using Eq. (1).

    sigma is mass-normalised collective thrust (>=0). Omega is body-rate in
    the body frame (rad/s). The thrust direction in world frame is R e_3.
    """
    sigma = float(max(0.0, sigma))
    Omega = np.asarray(Omega, dtype=float).reshape(3)
    # v_dot = -g e_3 + sigma R e_3
    a_world = -g * E3 + sigma * (state.R @ E3)
    v_new = state.v + dt * a_world
    p_new = state.p + dt * v_new   # semi-implicit Euler is more stable
    R_new = integrate_R(state.R, Omega, dt)
    step_count = state.step_count + 1
    if step_count % RE_ORTHO_EVERY == 0:
        R_new = project_so3(R_new)
    return DefenderState(p=p_new, v=v_new, R=R_new,
                         sigma=sigma, Omega=Omega.copy(),
                         step_count=step_count)


def step_intruder_kinematic(
    state: IntruderState,
    a: np.ndarray,
    dt: float,
) -> IntruderState:
    """Pure-kinematic intruder rollout used inside MPPI prediction (Eq. 16).

    The intruder attitude is not propagated by the prediction tube; that is
    deliberate -- the tube models acceleration directly via Eq. 13 and is
    agnostic to how the attacker realises it.
    """
    a = np.asarray(a, dtype=float).reshape(3)
    v_new = state.v + dt * a
    p_new = state.p + dt * v_new
    return IntruderState(p=p_new, v=v_new, R=state.R.copy(), a=a.copy())


def rollout_defender(
    state0: DefenderState,
    sigma_seq: np.ndarray,  # shape (N,)
    Omega_seq: np.ndarray,  # shape (N, 3)
    dt: float,
    g: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised forward-Euler rollout returning (p_seq, v_seq, R_seq).

    Shapes: p_seq (N+1, 3), v_seq (N+1, 3), R_seq (N+1, 3, 3).  R is
    integrated sequentially because the matrix exponentials don't compose
    in a vectorisable way for arbitrary Omega sequences.
    """
    N = int(sigma_seq.shape[0])
    p = np.zeros((N + 1, 3))
    v = np.zeros((N + 1, 3))
    R = np.zeros((N + 1, 3, 3))
    p[0] = state0.p
    v[0] = state0.v
    R[0] = state0.R
    for k in range(N):
        a_world = -g * E3 + max(0.0, float(sigma_seq[k])) * (R[k] @ E3)
        v[k + 1] = v[k] + dt * a_world
        p[k + 1] = p[k] + dt * v[k + 1]
        R[k + 1] = integrate_R(R[k], Omega_seq[k], dt)
    return p, v, R


def rollout_intruder_batch(
    p0: np.ndarray,           # (3,)
    v0: np.ndarray,           # (3,)
    a_seq_batch: np.ndarray,  # (M, N, 3)
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised intruder rollouts.

    Returns p_seq_batch (M, N+1, 3) and v_seq_batch (M, N+1, 3).  All scenarios
    share the same initial state but distinct acceleration histories.
    """
    M, N, _ = a_seq_batch.shape
    p = np.zeros((M, N + 1, 3))
    v = np.zeros((M, N + 1, 3))
    p[:, 0, :] = p0
    v[:, 0, :] = v0
    for k in range(N):
        v[:, k + 1, :] = v[:, k, :] + dt * a_seq_batch[:, k, :]
        p[:, k + 1, :] = p[:, k, :] + dt * v[:, k + 1, :]
    return p, v
