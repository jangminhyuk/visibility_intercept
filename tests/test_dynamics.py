"""Tests for the defender + intruder integrators."""

import numpy as np

from gtsim.config import E3
from gtsim.dynamics import (
    DefenderState,
    IntruderState,
    step_defender,
    step_intruder_kinematic,
)
from gtsim.mppi import rollout_defender_batch


def test_zero_thrust_falls_at_gravity():
    s = DefenderState(p=np.zeros(3), v=np.zeros(3), R=np.eye(3))
    dt = 0.01
    g = 9.81
    for _ in range(100):  # 1 s
        s = step_defender(s, sigma=0.0, Omega=np.zeros(3), dt=dt, g=g)
    # v_z = -g * t = -9.81 m/s
    assert abs(s.v[2] + 9.81) < 1e-6
    # Horizontal: unchanged
    assert abs(s.v[0]) < 1e-9 and abs(s.v[1]) < 1e-9


def test_hover_thrust_keeps_velocity_constant():
    s = DefenderState(p=np.zeros(3), v=np.array([1.0, 0.0, 0.0]),
                      R=np.eye(3))
    dt = 0.01
    g = 9.81
    for _ in range(200):
        s = step_defender(s, sigma=g, Omega=np.zeros(3), dt=dt, g=g)
    # Vertical velocity should stay 0; horizontal velocity should stay 1.
    assert abs(s.v[2]) < 1e-9
    assert abs(s.v[0] - 1.0) < 1e-9


def test_body_rate_rotates_attitude():
    s = DefenderState(p=np.zeros(3), v=np.zeros(3), R=np.eye(3))
    # +z body rate for 0.5 s at 2 rad/s -> 1 rad rotation about z.
    dt = 0.01
    g = 9.81
    Om = np.array([0.0, 0.0, 2.0])
    for _ in range(50):
        s = step_defender(s, sigma=g, Omega=Om, dt=dt, g=g)
    # body x = R e1 should now be approximately [cos(1), sin(1), 0]
    bx = s.R @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(bx, [np.cos(1.0), np.sin(1.0), 0.0], atol=1e-3)


def test_intruder_kinematic_step():
    s = IntruderState(p=np.zeros(3), v=np.array([1.0, 0.0, 0.0]))
    s2 = step_intruder_kinematic(s, np.array([0.0, 2.0, 0.0]), dt=0.5)
    # v += a dt; v_new = (1, 1, 0); p += v_new dt -> (0.5, 0.5, 0).
    np.testing.assert_allclose(s2.v, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(s2.p, [0.5, 0.5, 0.0])


def test_rollout_defender_batch_matches_serial():
    """Vectorised K-sample rollout must agree with the serial step_defender."""
    g = 9.81
    K, N = 5, 8
    dt = 0.05
    p0 = np.array([1.0, 2.0, 3.0])
    v0 = np.array([0.5, 0.0, -0.2])
    R0 = np.eye(3)
    rng = np.random.default_rng(7)
    sigma_batch = rng.uniform(8.0, 12.0, size=(K, N))
    Omega_batch = rng.normal(0.0, 0.5, size=(K, N, 3))
    p_b, v_b, R_b, b_b = rollout_defender_batch(
        p0, v0, R0, sigma_batch, Omega_batch, dt, g
    )
    # Now serial check for sample i=2.
    i = 2
    s = DefenderState(p=p0.copy(), v=v0.copy(), R=R0.copy())
    p_serial = [s.p.copy()]
    v_serial = [s.v.copy()]
    for k in range(N):
        s = step_defender(s, sigma_batch[i, k], Omega_batch[i, k], dt, g)
        p_serial.append(s.p.copy())
        v_serial.append(s.v.copy())
    p_serial = np.array(p_serial)
    v_serial = np.array(v_serial)
    np.testing.assert_allclose(p_b[i], p_serial, atol=1e-9)
    np.testing.assert_allclose(v_b[i], v_serial, atol=1e-9)
